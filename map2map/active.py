import argparse
import torch
from pprint import pprint
import sys
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from heapq import heappop, heappush
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List
from tqdm import tqdm
from math import sqrt
import numpy as np
from itertools import islice
import matplotlib.pyplot as plt

from map2map import models
from map2map.models import StyledVNet, lag2eul
from map2map.data.fields import FieldDataset
from map2map.models.narrow import narrow_cast
from map2map.utils import import_attr, load_model_state_dict


class MinHeap:

    def __init__(self, capacity: int) -> None:
        self.minHeap = []
        self.capacity = capacity
    
    def push(self, priority, elem):
        heappush(self.minHeap, (priority, elem))
        if len(self.minHeap) > self.capacity:
            heappop(self.minHeap)
    
    def pop(self):
        _, elem = heappop(self.minHeap)
        return elem
    
    def __len__(self):
        return len(self.minHeap)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda', 3)
    else:
        return torch.device('cpu')


def get_test_data(args: argparse.Namespace):
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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_workers,
        pin_memory=True,
    )
    
    return test_dataset, test_loader


def get_observed_target(args: argparse.Namespace):
    observed_dataset = FieldDataset(
        style_pattern='/user_data/ajliang/Linear/LH0001/4/params.npy',
        in_patterns=['/user_data/ajliang/Linear/LH0001/4/dis.npy'],
        tgt_patterns=['/user_data/ajliang/Nonlinear/LH0001/4/dis.npy'],
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
    observed_loader = DataLoader(
        observed_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.loader_workers,
        pin_memory=True,
    )
    data = next(iter(observed_loader))
    return data['target']
    

def compute_uncertainty(
    args: argparse.Namespace, model: StyledVNet,
    input: torch.Tensor, style: torch.Tensor, target: torch.Tensor,
    device: torch.device,
):
    model.enable_dropout(args.dropout_prob)
    losses = torch.zeros(size=(args.uncertain_sample_size,), device=device)
    for s in range(args.uncertain_sample_size):
        output = model(input, style)
        output, target = narrow_cast(output, target)

        loss = F.mse_loss(output, target)
        losses[s] = loss
    variance = torch.var(losses, unbiased=False)
    return variance


def compute_error(
    model: StyledVNet, input: torch.Tensor, style: torch.Tensor, observed_target: torch.Tensor
):
    model.disable_dropout()
    output = model(input, style)
    
    output, observed_target = narrow_cast(output, observed_target)
    return F.mse_loss(output, observed_target)


def find_top_squarecb_input(
    args: argparse.Namespace,
    model: StyledVNet,
    test_loader: DataLoader,
    observed_target: torch.Tensor,
    device: torch.device,
    logger: SummaryWriter,
    epoch: int
):
    device = observed_target.get_device()
    
    inputs = []
    min_ordinal, min_error = None, float('inf')
    with torch.no_grad():
        for i, data in tqdm(enumerate(islice(test_loader, args.load_limit))):
            style, input, target = data['style'], data['input'], data['target']
            style = style.to(device, non_blocking=True)
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            error = compute_error(
                model=model, input=input, style=style, observed_target=observed_target
            )
            error = error.item()

            min_error, min_ordinal = min((min_error, min_ordinal), (error, i))
            
            inputs.append(
                {
                    'input': input, 'style': style, 
                    'target': target, 'ordinal': i,
                    'error': error
                }
            )
    
    mu = len(test_loader.dataset)
    gamma = sqrt(mu)

    top_k_inputs = MinHeap(capacity=args.report_top_k)
    total_probs_except_best = 0
    for i, input in enumerate(inputs):
        if i != min_ordinal:
            prob = 1 / (mu + gamma * (input['error'] - min_error))
            input['prob'] = prob

            top_k_inputs.push(
                priority=prob,
                elem=input,
            )

            total_probs_except_best += prob

    best_prob = 1 - total_probs_except_best
    inputs[min_ordinal]['prob'] = best_prob
    top_k_inputs.push(
        priority=best_prob,
        elem=inputs[min_ordinal],
    )
    
    top_k_inputs = [top_k_inputs.pop() for _ in range(len(top_k_inputs))]
    log_top_k_squarecb_inputs(logger=logger, top_k_inputs=top_k_inputs, global_step=epoch)
    
    top_input = np.random.choice(
        inputs,
        p=[input['prob'] for input in inputs]
    )

    return top_input


def log_top_k_squarecb_inputs(logger: SummaryWriter, top_k_inputs: List, global_step: int):
    msg = []
    for rank, top_input in enumerate(top_k_inputs):
        msg.append(
            f'The top {rank + 1} input: '
            f'error = {top_input["error"]}, '
            f'SquareCB weight = {top_input["prob"]}, '
            f'ordinal = {top_input["ordinal"]}'
        )
    msg = '\n'.join(msg)
    logger.add_text(
        tag='Top SquareCB Inputs',
        text_string=msg,
        global_step=global_step,
    )


def find_top_ucb_input(
    args: argparse.Namespace,
    model: StyledVNet,
    test_loader: DataLoader,
    observed_target: torch.Tensor,
    device: torch.device,
    logger: SummaryWriter,
    epoch: int
):
    top_k_inputs = MinHeap(capacity=args.report_top_k)
    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(islice(test_loader, args.load_limit))):
            style, input, target = data['style'], data['input'], data['target']
            style = style.to(device, non_blocking=True)
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            uncertainty = compute_uncertainty(
                args=args, model=model, input=input,
                style=style, target=target,
                device=device,
            )
            error = compute_error(
                model=model, input=input, style=style, observed_target=observed_target
            )
            
            uncertainty = uncertainty.item()
            error = error.item()
            
            top_k_inputs.push(
                priority=error - uncertainty,
                elem={
                    "error": error,
                    "uncertainty": uncertainty,
                    "input": input,
                    "style": style,
                    "target": target,
                    "ordinal": i,
                }
            )
    
    top_k_inputs = [top_k_inputs.pop() for _ in range(len(top_k_inputs))]
    log_top_k_ucb_inputs(
        args=args, logger=logger,
        top_k_inputs=top_k_inputs, global_step=epoch
    )

    return top_k_inputs[0]


def log_top_k_ucb_inputs(args: argparse.Namespace, logger: SummaryWriter, top_k_inputs: List, global_step: int):
    msg = []
    for rank, top_input in enumerate(top_k_inputs):
        msg.append(
            f'The top {rank + 1} input: '
            f'error = {top_input["error"]}, '
            f'uncertainty = {top_input["uncertainty"]}, '
            f'ordinal = {top_input["ordinal"]}'
        )
    msg = '\n'.join(msg)
    logger.add_text(
        tag=f'Top UCB Inputs (Out of {args.load_limit})',
        text_string=msg,
        global_step=global_step,
    )


def get_model_and_optimizer(
    args: argparse.Namespace, style_size: int, in_chan:int, out_chan: int,
    device: torch.device
):
    model = StyledVNet(
        style_size,
        sum(in_chan),
        sum(out_chan),
        scale_factor=args.scale_factor
    )
    model.to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr,
    )

    state = torch.load(args.load_state, map_location=device)
    load_model_state_dict(model, state['model'], strict=args.load_state_strict)
    print('model state at epoch {} loaded from {}'.format(state['epoch'], args.load_state))

    del state

    return model, optimizer


def train_step(
    args: argparse.Namespace, epoch: int,
    top_input: Dict, model: StyledVNet, criterion,
    optimizer: torch.optim.Optimizer, logger: SummaryWriter
):
    model.train()

    output = model(top_input['input'], top_input['style'])
    output, target = narrow_cast(output, top_input['target'])
    lag_out, lag_tgt = output, target
    eul_out, eul_tgt = lag2eul([lag_out, lag_tgt], **args.misc_kwargs)

    lag_loss = criterion(lag_out, lag_tgt)
    eul_loss = criterion(eul_out, eul_tgt)
    train_loss = lag_loss ** 3 * eul_loss
    
    if train_loss > 0:

        optimizer.zero_grad()
        torch.log(train_loss).backward()
        optimizer.step()
        
        logger.add_scalar(
            tag='Train Loss',
            scalar_value=train_loss,
            global_step=epoch
        )
        logger.add_scalar(
            tag='Ordinal of Best Input',
            scalar_value=top_input['ordinal'],
            global_step=epoch
        )


def plot_best_error_and_uncertainty(
    epochs: List[int],
    best_errors: List[float],
    best_uncertainties: List[float],
    logger: SummaryWriter,
):
    plt.errorbar(
        x=epochs,
        y=best_errors,
        yerr=best_uncertainties,
        ls='none',
        fmt='o'
    )
    plt.xlabel('Epoch')
    plt.ylabel('MSE between Output Displacement and Observed Displacement + Uncertainty on Output Displacement')
    fig = plt.gcf()
    logger.add_figure(
        f'fig/best_error_and_uncertainty/epoch_{epochs[-1]}',
        figure=fig,
    )
    logger.flush()
    plt.clf()


def search(args: argparse.Namespace):
    pprint(vars(args))
    sys.stdout.flush()

    device = get_device()
    test_dataset, test_loader = get_test_data(args)
    
    model, optimizer = get_model_and_optimizer(
        args=args,
        style_size=test_dataset.style_size, in_chan=test_dataset.in_chan, out_chan=test_dataset.tgt_chan,
        device=device,
    )
    
    criterion = import_attr(args.criterion, nn, models, callback_at=args.callback_at)
    criterion = criterion()
    criterion.to(device)

    observed_target: torch.Tensor = get_observed_target(args)
    observed_target = observed_target.to(device)
    
    logger = SummaryWriter()
    
    epochs = []
    best_errors = []
    best_uncertainties = []
    for epoch in range(1, args.epochs + 1):        
        if args.square_cb:
            top_input = find_top_squarecb_input(
                args=args, model=model, test_loader=test_loader,
                observed_target=observed_target, device=device,
                logger=logger, epoch=epoch
            )
        else:
            top_input = find_top_ucb_input(
                args=args, model=model, test_loader=test_loader,
                observed_target=observed_target, device=device,
                logger=logger, epoch=epoch
            )
            
        epochs.append(epoch)
        best_errors.append(top_input['error'])
        best_uncertainties.append(top_input['uncertainty'])
        plot_best_error_and_uncertainty(epochs, best_errors, best_uncertainties, logger)

        train_step(
            args, epoch=epoch, top_input=top_input,
            model=model, criterion=criterion, 
            optimizer=optimizer, logger=logger,
        )
