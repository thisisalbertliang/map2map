import torch
import torch.nn as nn
import torch.nn.functional as F

from map2map.models.uncertainty_quantification.styled_conv import ConvStyledBlock, ResStyledBlock
from map2map.models.narrow import narrow_by


class StyledVNet(nn.Module):
    def __init__(self, style_size, in_chan, out_chan, bypass=None, **kwargs):
        """V-Net like network with styles

        See `vnet.VNet`.
        """
        super().__init__()

        # activate non-identity skip connection in residual block
        # by explicitly setting out_chan
        self.conv_l00 = ResStyledBlock(style_size, in_chan, 64, seq='CACA')
        self.conv_l01 = ResStyledBlock(style_size, 64, 64, seq='CACA')
        self.down_l0 = ConvStyledBlock(style_size, 64, seq='DA')
        self.conv_l1 = ResStyledBlock(style_size, 64, 64, seq='CACA')
        self.down_l1 = ConvStyledBlock(style_size, 64, seq='DA')
        self.conv_l2 = ResStyledBlock(style_size, 64, 64, seq='CACA')
        self.down_l2 = ConvStyledBlock(style_size, 64, seq='DA')

        # Initialize the upsampling path for decoding the mean
        self.conv_c_mean = ResStyledBlock(style_size, 64, 64, seq='CACA')

        self.up_r2_mean = ConvStyledBlock(style_size, 64, seq='UA')
        self.conv_r2_mean = ResStyledBlock(style_size, 128, 64, seq='CACA')
        self.up_r1_mean = ConvStyledBlock(style_size, 64, seq='UA')
        self.conv_r1_mean = ResStyledBlock(style_size, 128, 64, seq='CACA')
        self.up_r0_mean = ConvStyledBlock(style_size, 64, seq='UA')
        self.conv_r00_mean = ResStyledBlock(style_size, 128, 64, seq='CACA')
        self.conv_r01_mean = ResStyledBlock(style_size, 64, out_chan, seq='CAC')

        # Initialize the upsampling path for decoding the variance
        self.conv_c_var = ResStyledBlock(style_size, 64, 64, seq='CACA')

        self.up_r2_var = ConvStyledBlock(style_size, 64, seq='UA')
        self.conv_r2_var = ResStyledBlock(style_size, 128, 64, seq='CACA')
        self.up_r1_var = ConvStyledBlock(style_size, 64, seq='UA')
        self.conv_r1_var = ResStyledBlock(style_size, 128, 64, seq='CACA')
        self.up_r0_var = ConvStyledBlock(style_size, 64, seq='UA')
        self.conv_r00_var = ResStyledBlock(style_size, 128, 64, seq='CACA')
        self.conv_r01_var = ResStyledBlock(style_size, 64, out_chan, seq='CAC')

        # Dict for switching between mean and variance decoder
        self.conv_c = {'mean': self.conv_c_mean, 'var': self.conv_c_var}

        self.up_r2 = {'mean': self.up_r2_mean, 'var': self.up_r2_var}
        self.conv_r2 = {'mean': self.conv_r2_mean, 'var': self.conv_r2_var}
        self.up_r1 = {'mean': self.up_r1_mean, 'var': self.up_r1_var}
        self.conv_r1 = {'mean': self.conv_r1_mean, 'var': self.conv_r1_var}
        self.up_r0 = {'mean': self.up_r0_mean, 'var': self.up_r0_var}
        self.conv_r00 = {'mean': self.conv_r00_mean, 'var': self.conv_r00_var}
        self.conv_r01 = {'mean': self.conv_r01_mean, 'var': self.conv_r01_var}

        if bypass is None:
            self.bypass = in_chan == out_chan
        else:
            self.bypass = bypass

    def encoder(self, x, s):
        # the contractive path of the V-Net
        x = self.conv_l00(x, s)
        y0 = self.conv_l01(x, s)
        x = self.down_l0(y0, s)

        y1 = self.conv_l1(x, s)
        x = self.down_l1(y1, s)

        y2 = self.conv_l2(x, s)
        x = self.down_l2(y2, s)

        return x, y0, y1, y2

    def decoder(self, x, s, y0, y1, y2, x0, which='mean'):
        # the expansive path of the V-Net
        x = self.conv_c[which](x, s)

        x = self.up_r2[which](x, s)
        y2 = narrow_by(y2, 4)
        x = torch.cat([y2, x], dim=1)
        # del y2
        x = self.conv_r2[which](x, s)

        x = self.up_r1[which](x, s)
        y1 = narrow_by(y1, 16)
        x = torch.cat([y1, x], dim=1)
        # del y1
        x = self.conv_r1[which](x, s)

        x = self.up_r0[which](x, s)
        y0 = narrow_by(y0, 40)
        x = torch.cat([y0, x], dim=1)
        # del y0
        x = self.conv_r00[which](x, s)
        x = self.conv_r01[which](x, s)

        if self.bypass:
            x0 = narrow_by(x0, 48)
            x += x0

        if which == 'var':
            # enforce positive variance
            x = F.relu(x)

        return x


    def forward(self, x, s):
        if self.bypass:
            x0 = x
        # the contractive path of the V-Net
        x, y0, y1, y2 = self.encoder(x, s)
        mean = self.decoder(x, s, y0, y1, y2, x0=x0 if self.bypass else None, which='mean')
        var = self.decoder(x, s, y0, y1, y2, x0=x0 if self.bypass else None, which='var')

        del y0, y1, y2
        return mean, var


        # if self.bypass:
        #     x0 = x

        # x = self.conv_l00(x, s)
        # y0 = self.conv_l01(x, s)
        # x = self.down_l0(y0, s)

        # y1 = self.conv_l1(x, s)
        # x = self.down_l1(y1, s)

        # y2 = self.conv_l2(x, s)
        # x = self.down_l2(y2, s)

        # x = self.conv_c(x, s)

        # x = self.up_r2(x, s)
        # y2 = narrow_by(y2, 4)
        # x = torch.cat([y2, x], dim=1)
        # del y2
        # x = self.conv_r2(x, s)

        # x = self.up_r1(x, s)
        # y1 = narrow_by(y1, 16)
        # x = torch.cat([y1, x], dim=1)
        # del y1
        # x = self.conv_r1(x, s)

        # x = self.up_r0(x, s)
        # y0 = narrow_by(y0, 40)
        # x = torch.cat([y0, x], dim=1)
        # del y0
        # x = self.conv_r00(x, s)
        # x = self.conv_r01(x, s)

        # if self.bypass:
        #     x0 = narrow_by(x0, 48)
        #     x += x0

        # return x