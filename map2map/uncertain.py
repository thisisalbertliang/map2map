import argparse
from math import sqrt
from pprint import pprint
import sys
import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from map2map import models
from map2map.data.fields import FieldDataset
from map2map.models import power
from map2map.models.model import StyledVNet
from map2map.models.narrow import narrow_cast
from map2map.utils.imp import import_attr
from map2map.utils.state import load_model_state_dict


def estimate_uncertainty(args: argparse.ArgumentParser):
    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')
    
    print('pytorch {}'.format(torch.__version__))
    pprint(vars(args))
    sys.stdout.flush()

    test_dataset = FieldDataset(
        style_pattern=args.test_style_pattern,
        in_patterns=args.test_in_patterns,
        tgt_patterns=args.test_tgt_patterns,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        callback_at=args.callback_at,
        augment=False,
        aug_shift=None,
        aug_add=None,
        aug_mul=None,
        crop=args.crop,
        crop_start=args.crop_start,
        crop_stop=args.crop_stop,
        crop_step=args.crop_step,
        in_pad=args.in_pad,
        tgt_pad=args.tgt_pad,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.loader_workers,
        pin_memory=True,
    )

    style_size = test_dataset.style_size
    in_chan = test_dataset.in_chan
    out_chan = test_dataset.tgt_chan
    
    model = import_attr(args.model, models, callback_at=args.callback_at)
    model = model(
        style_size,
        sum(in_chan),
        sum(out_chan),
        scale_factor=args.scale_factor
    )
    model.to(device)
    state = torch.load(args.load_state, map_location=device)
    load_model_state_dict(model, state['model'], strict=args.load_state_strict)
    print('model state at epoch {} loaded from {}'.format(state['epoch'], args.load_state))
    del state

    if args.square_cb:
        square_cb(args, model, test_loader, device)
    else:
        monte_carlo_dropout(args, model, test_loader, device)


def monte_carlo_dropout(
    args: argparse.Namespace,
    model: StyledVNet,
    loader: DataLoader,
    device: torch.device
):
    model.eval()
    model.enable_dropout(args.dropout_prob)

    logger = SummaryWriter()

    mse_loss = nn.MSELoss()
    with torch.no_grad():
        field_ordinals = []
        losses = []
        for i, data in enumerate(loader):
            style, input, target = data['style'], data['input'], data['target']

            style = style.to(device, non_blocking=True)
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            for s in range(args.sample_size):
                output = model(input, style)
                _, lag_out, lag_tgt = narrow_cast(input, output, target)
                _, P_lag_out, _ = power(lag_out)
                _, P_lag_tgt, _ = power(lag_tgt)

                field_ordinals.append(i)
                loss = mse_loss(P_lag_out, P_lag_tgt).cpu().item()
                losses.append(loss)

                logger.add_text(
                    tag=f"Monte Carlo Dropout (prob = {args.dropout_prob})",
                    text_string=f"Field {i}: Sample {s}: Power Spectrum MSE Loss: {loss}",
                    global_step=i * args.sample_size + s
                )
            
    plt.scatter(
        x=field_ordinals,
        y=losses,
        s=5
    )
    plt.xlabel("Field (Ordinal)")
    plt.ylabel("Power Spectrum MSE Losses")
    plt.title(f"Dropout Probability = {args.dropout_prob}: Power Spectrum MSE Losses v.s. Field (Ordinal)")
    fig = plt.gcf()

    logger.add_figure(f'fig/dropout-prob-{args.dropout_prob}-power-spec-mse-vs-field-ordinal', fig)
    logger.flush()

    logger.close()


def square_cb(
    args: argparse.Namespace,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
):
    model.eval()

    logger = SummaryWriter()

    mse_loss = nn.MSELoss()
    with torch.no_grad():
        losses = []
        min_i, min_loss = None, float('inf')

        for i, data in enumerate(loader):
            style, input, target = data['style'], data['input'], data['target']
            style = style.to(device, non_blocking=True)
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(input, style)
            _, lag_out, lag_tgt = narrow_cast(input, output, target)
            _, P_lag_out, _ = power(lag_out)
            _, P_lag_tgt, _ = power(lag_tgt)

            loss = mse_loss(P_lag_out, P_lag_tgt).cpu().item()
            if loss < min_loss:
                min_i, min_loss = i, loss

            losses.append(loss)

            logger.add_text(
                tag="SquareCB",
                text_string=f"Field {i}: Power Spectrum MSE Loss: {loss}",
                global_step=i
            )
    
    mu = len(loader.dataset)
    gamma = sqrt(mu)
    # diffs = [loss - min_loss for i, loss in enumerate(losses) if i != min_i]
    # total_diff = sum(diffs)
    # normal_diff = [diff / total_diff for diff in diffs]
    probs = [
        1 / (mu + gamma * (loss - min_loss))
        for i, loss in enumerate(losses)
        if i != min_i
    ]
    # probs.insert(min_i, 1 - sum(probs))
    # total = sum(probs)
    # probs = [p / total for p in probs]

    plt.scatter(
        x=[i for i in range(len(loader.dataset) - 1)],
        y=probs
    )
    plt.xlabel("Field (Ordinal)")
    plt.ylabel("SquareCB Weights")
    plt.title(f"SquareCB Weights v.s. Field (Ordinal)")
    fig = plt.gcf()

    logger.add_figure(f"fig/square-cb-weights-vs-field-ordinal", fig)
    logger.flush()

    logger.close()
