import torch
import torch.nn as nn

from .styled_conv import ConvStyledBlock, ResStyledBlock
from .narrow import narrow_by


class StyledVNet(nn.Module):
    def __init__(self, style_size, in_chan, out_chan, dropout_prob, bypass=None, **kwargs):
        """V-Net like network with styles

        See `vnet.VNet`.
        """
        super().__init__()

        # activate non-identity skip connection in residual block
        # by explicitly setting out_chan
        self.conv_l00 = ResStyledBlock(style_size=style_size, in_chan=in_chan, out_chan=64,seq='CACA', dropout_prob=dropout_prob)
        self.conv_l01 = ResStyledBlock(style_size=style_size, in_chan=64, out_chan=64, seq='CACA', dropout_prob=dropout_prob)
        self.down_l0 = ConvStyledBlock(style_size=style_size, in_chan=64, seq='DA', dropout_prob=dropout_prob)
        self.conv_l1 = ResStyledBlock(style_size=style_size, in_chan=64, out_chan=64, seq='CACA', dropout_prob=dropout_prob)
        self.down_l1 = ConvStyledBlock(style_size=style_size, in_chan=64, seq='DA', dropout_prob=dropout_prob)
        self.conv_l2 = ResStyledBlock(style_size=style_size, in_chan=64, out_chan=64, seq='CACA', dropout_prob=dropout_prob)
        self.down_l2 = ConvStyledBlock(style_size=style_size, in_chan=64, seq='DA', dropout_prob=dropout_prob)

        self.conv_c = ResStyledBlock(style_size=style_size, in_chan=64, out_chan=64, seq='CACA', dropout_prob=dropout_prob)

        self.up_r2 = ConvStyledBlock(style_size=style_size, in_chan=64, seq='UA', dropout_prob=dropout_prob)
        self.conv_r2 = ResStyledBlock(style_size=style_size, in_chan=128, out_chan=64, seq='CACA', dropout_prob=dropout_prob)
        self.up_r1 = ConvStyledBlock(style_size=style_size, in_chan=64, seq='UA', dropout_prob=dropout_prob)
        self.conv_r1 = ResStyledBlock(style_size=style_size, in_chan=128, out_chan=64, seq='CACA', dropout_prob=dropout_prob)
        self.up_r0 = ConvStyledBlock(style_size=style_size, in_chan=64, seq='UA', dropout_prob=dropout_prob)
        self.conv_r00 = ResStyledBlock(style_size=style_size, in_chan=128, out_chan=64, seq='CACA', dropout_prob=dropout_prob)
        # As mentioned in the "Dropout as Bayesian Approximation" paper, we do not use dropout in the last layer
        self.conv_r01 = ResStyledBlock(style_size=style_size, in_chan=64, out_chan=out_chan, seq='CAC', dropout_prob=0.0)

        if bypass is None:
            self.bypass = in_chan == out_chan
        else:
            self.bypass = bypass

    def forward(self, x, s):
        if self.bypass:
            x0 = x

        x = self.conv_l00(x, s)
        y0 = self.conv_l01(x, s)
        x = self.down_l0(y0, s)

        y1 = self.conv_l1(x, s)
        x = self.down_l1(y1, s)

        y2 = self.conv_l2(x, s)
        x = self.down_l2(y2, s)

        x = self.conv_c(x, s)

        x = self.up_r2(x, s)
        y2 = narrow_by(y2, 4)
        x = torch.cat([y2, x], dim=1)
        del y2
        x = self.conv_r2(x, s)

        x = self.up_r1(x, s)
        y1 = narrow_by(y1, 16)
        x = torch.cat([y1, x], dim=1)
        del y1
        x = self.conv_r1(x, s)

        x = self.up_r0(x, s)
        y0 = narrow_by(y0, 40)
        x = torch.cat([y0, x], dim=1)
        del y0
        x = self.conv_r00(x, s)
        x = self.conv_r01(x, s)

        if self.bypass:
            x0 = narrow_by(x0, 48)
            x += x0

        return x
