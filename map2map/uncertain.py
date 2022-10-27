import argparse
from math import sqrt
from pprint import pprint
import sys
from statistics import variance, mean
import torch
from torch.utils.data import DataLoader
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from map2map import models
from map2map.data.fields import FieldDataset
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

    criterion = import_attr(
        args.criterion,
        torch.nn,
        models,
        callback_at=args.callback_at
    )
    criterion = criterion()
    criterion.to(device)

    if args.square_cb:
        square_cb(args, model, criterion, test_loader, device)
    else:
        monte_carlo_dropout(args, model, criterion, test_loader, device)


def monte_carlo_dropout(
    args: argparse.Namespace,
    model: StyledVNet,
    criterion,
    loader: DataLoader,
    device: torch.device
):
    model.eval()
    model.enable_dropout(args.dropout_prob)

    logger = SummaryWriter()
    log_step = 0

    from itertools import islice

    with torch.no_grad():
        losses = []
        for i, data in enumerate(islice(loader, 10)):
            style, input, target = data['style'], data['input'], data['target']

            style = style.to(device, non_blocking=True)
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            samples = []
            for s in range(args.sample_size):
                output = model(input, style)
                _, output, target = narrow_cast(input, output, target)

                loss = criterion(output, target).cpu().item()
                samples.append(loss)

                logger.add_text(
                    tag=f"Monte Carlo Dropout (prob = {args.dropout_prob})",
                    text_string=f"Field {i}: Sample {s}: Output Displacement MSE Loss: {loss}",
                    global_step=log_step
                )
                log_step += 1
            losses.append(samples)
    
    _plot_violin(args, logger, log_step, losses=losses, num_fields_per_plot=10)
    _plot_errorbarr(args, logger, log_step, losses=losses, num_fields_per_plot=10)

    logger.flush()
    logger.close()


def _plot_errorbarr(args, logger, log_step, losses, num_fields_per_plot):
    means = [mean(sample) for sample in losses]
    variances = [variance(sample) for sample in losses]

    for i in range(0, len(losses), num_fields_per_plot):
        plt.errorbar(
            x=[j for j in range(i, min(len(losses), i + num_fields_per_plot))],
            y=means[i:i+num_fields_per_plot],
            yerr=variances[i:i+num_fields_per_plot],
            ls='none',
            fmt='o'
        )
        plt.xlabel("Field (Ordinal)")
        plt.ylabel("Output Displacement MSE Losses")
        plt.title(f"Dropout Probability = {args.dropout_prob}:\nOutput Displacement MSE Losses v.s. Field ({i} ~ {i + num_fields_per_plot - 1})")
        fig = plt.gcf()
        logger.add_figure(f'fig/errorbar/dropout-prob-{args.dropout_prob}-displacement-mse-vs-field-{i}-{i + num_fields_per_plot - 1}', fig)

    plt.errorbar(
        x=[j for j in range(len(losses))],
        y=means,
        yerr=variances,
        ls='none',
        fmt='o'
    )
    plt.xlabel("Field (Ordinal)")
    plt.ylabel("Output Displacement MSE Losses")
    plt.title(f"Dropout Probability = {args.dropout_prob}:\nOutput Displacement MSE Losses v.s. Field (All)")
    fig = plt.gcf()
    logger.add_figure(f'fig/errorbar/dropout-prob-{args.dropout_prob}-displacement-mse-vs-field-all', fig)

    logger.add_text(
        tag=f"Monte Carlo Dropout (prob = {args.dropout_prob})",
        text_string=f"Field with min MSE: {min((i for i in range(len(means))), key=lambda i: means[i])}",
        global_step=log_step
    )
    log_step += 1
    for i in range(len(losses)):
        logger.add_text(
            tag=f"Monte Carlo Dropout (prob = {args.dropout_prob})",
            text_string=f"Field {i}: MSE Mean = {means[i]}; MSE Variance = {variances[i]}",
            global_step=log_step
        )
        log_step += 1
        logger.flush()


def _plot_violin(args, logger, log_step, losses, num_fields_per_plot):
    for i in range(0, len(losses), num_fields_per_plot):
        plt.violinplot(losses[i : i + num_fields_per_plot])
        plt.xlabel("Field (Ordinal)")
        plt.ylabel("Output Displacement MSE Losses")
        plt.title(f"Dropout Probability = {args.dropout_prob}:\nOutput Displacement MSE Losses v.s. Field ({i} ~ {i + num_fields_per_plot - 1})")
        fig = plt.gcf()
        logger.add_figure(f'fig/violin/dropout-prob-{args.dropout_prob}-displacement-mse-vs-field-{i}-{i + num_fields_per_plot - 1}', fig)

    plt.violinplot(losses)
    plt.xlabel("Field (Ordinal)")
    plt.ylabel("Output Displacement MSE Losses")
    plt.title(f"Dropout Probability = {args.dropout_prob}:\nOutput Displacement MSE Losses v.s. Field (All)")
    fig = plt.gcf()
    logger.add_figure(f'fig/violin/dropout-prob-{args.dropout_prob}-displacement-mse-vs-field-all', fig)


def square_cb(
    args: argparse.Namespace,
    model: nn.Module,
    criterion,
    loader: DataLoader,
    device: torch.device
):
    model.eval()

    logger = SummaryWriter()
    log_step = 0

    with torch.no_grad():
        losses = []
        min_i, min_loss = None, float('inf')

        for i, data in enumerate(loader):
            style, input, target = data['style'], data['input'], data['target']
            style = style.to(device, non_blocking=True)
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(input, style)
            _, output, target = narrow_cast(input, output, target)

            loss = criterion(output, target).cpu().item()
            if loss < min_loss:
                min_i, min_loss = i, loss

            losses.append(loss)

            logger.add_text(
                tag="SquareCB",
                text_string=f"Field {i}: Output Displacement MSE Loss: {loss}",
                global_step=log_step
            )
            log_step += 1
    
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
    probs.insert(min_i, 1 - sum(probs))
    # total = sum(probs)
    # probs = [p / total for p in probs]

    plt.scatter(
        x=[i for i in range(len(loader.dataset))],
        y=probs
    )
    plt.xlabel("Field (Ordinal)")
    plt.ylabel("SquareCB Weights")
    plt.title(f"SquareCB Weights v.s. Field (Ordinal)")
    fig = plt.gcf()

    logger.add_figure(f"fig/square-cb-weights-vs-field-ordinal", fig)
    logger.flush()

    logger.add_text(
        tag="SquareCB",
        text_string=f"Field with min MSE: {min_i}",
        global_step=log_step
    )
    log_step += 1
    for i, p in enumerate(probs):
        logger.add_text(
            tag="SquareCB",
            text_string=f"Field {i} SquareCB weight: {p}",
            global_step=log_step
        )
        log_step += 1
        logger.flush()

    logger.close()
