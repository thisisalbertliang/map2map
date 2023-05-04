import pickle
import os
import socket
import time
import sys
from pprint import pformat
from tqdm import tqdm
from time import gmtime, strftime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn, get_kl_loss

from .data import FieldDataset, DistFieldSampler
from . import models
from .models import narrow_cast, resample, lag2eul, StyledVNet
from .utils import import_attr, load_model_state_dict, plt_slices, plt_power, get_logger, RunningStats, plt_power_with_error_bar


ckpt_link = 'd2d_forward_bnn.pt'


def node_worker(args):
    if args.distributed:
        if 'SLURM_STEP_NUM_NODES' in os.environ:
            args.nodes = int(os.environ['SLURM_STEP_NUM_NODES'])
        elif 'SLURM_JOB_NUM_NODES' in os.environ:
            args.nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        else:
            raise KeyError('missing node counts in slurm env')
        args.gpus_per_node = torch.cuda.device_count()
        args.world_size = args.nodes * args.gpus_per_node

        logging.info(f"GPUs per Node = {args.gpus_per_node}")
        node = int(os.environ['SLURM_NODEID'])

        if args.gpus_per_node < 1:
            raise RuntimeError('GPU not found on node {}'.format(node))

        spawn(gpu_worker, args=(node, args), nprocs=args.gpus_per_node)
    else:
        args.gpus_per_node = 1
        args.world_size = 1
        gpu_worker(0, 0, args)


def gpu_worker(local_rank, node, args):
    #device = torch.device('cuda', local_rank)
    #torch.cuda.device(device)  # env var recommended over this

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    device = torch.device('cuda', 0)

    rank = args.gpus_per_node * node + local_rank

    # Need randomness across processes, for sampler, augmentation, noise etc.
    # Note DDP broadcasts initial model states from rank 0
    torch.manual_seed(args.seed + rank)
    # good practice to disable cudnn.benchmark if enabling cudnn.deterministic
    #torch.backends.cudnn.deterministic = True

    if args.distributed:
        dist_init(rank, args)

    train_dataset = FieldDataset(
        style_pattern=args.train_style_pattern,
        in_patterns=args.train_in_patterns,
        tgt_patterns=args.train_tgt_patterns,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        callback_at=args.callback_at,
        augment=args.augment,
        aug_shift=args.aug_shift,
        aug_add=args.aug_add,
        aug_mul=args.aug_mul,
        crop=args.crop,
        crop_start=args.crop_start,
        crop_stop=args.crop_stop,
        crop_step=args.crop_step,
        in_pad=args.in_pad,
        tgt_pad=args.tgt_pad,
        scale_factor=args.scale_factor,
        **args.misc_kwargs,
    )
    if args.distributed:
        train_sampler = DistFieldSampler(train_dataset, shuffle=True,
                                     div_data=args.div_data,
                                     div_shuffle_dist=args.div_shuffle_dist)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.loader_workers,
        pin_memory=True,
    )

    if args.val:
        val_dataset = FieldDataset(
            style_pattern=args.val_style_pattern,
            in_patterns=args.val_in_patterns,
            tgt_patterns=args.val_tgt_patterns,
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
        if args.distributed:
            val_sampler = DistFieldSampler(val_dataset, shuffle=False,
                                       div_data=args.div_data,
                                       div_shuffle_dist=args.div_shuffle_dist)
        else:
            val_sampler = None
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.loader_workers,
            pin_memory=True,
        )

    args.style_size = train_dataset.style_size
    args.in_chan = train_dataset.in_chan
    args.out_chan = train_dataset.tgt_chan

    model = StyledVNet(args.style_size, sum(args.in_chan), sum(args.out_chan),
                       dropout_prob=args.dropout_prob,
                  scale_factor=args.scale_factor, **args.misc_kwargs)
    dnn_to_bnn(model, bnn_prior_parameters={
        'prior_mu': args.prior_mu, 'prior_sigma': args.prior_sigma,
        'posterior_mu_init': args.posterior_mu_init, 'posterior_rho_init': args.posterior_rho_init,
        'type': args.bnn_type,
        'moped_enable': args.moped_enable, 'moped_delta': args.moped_delta,

    })
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[device],
            process_group=dist.new_group())

    criterion = import_attr(args.criterion, nn, models,
                            callback_at=args.callback_at)
    criterion = criterion()
    criterion.to(device)

    optimizer = import_attr(args.optimizer, optim, callback_at=args.callback_at)
    optimizer = optimizer(
        [
            {
                'params': (param for name, param in model.named_parameters()
                           if 'mlp' in name or 'style' in name),
                'betas': (0.9, 0.99), 'weight_decay': 1e-4,
            },
            {
                'params': (param for name, param in model.named_parameters()
                           if 'mlp' not in name and 'style' not in name),
            },
        ],
        lr=args.lr,
        **args.optimizer_args,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, **args.scheduler_args)

    if args.load_state is None:
        logging.info('initializing model weights')

        if args.init_weight_std is not None:
            model.apply(init_weights)

        start_epoch = 0

        if rank == 0:
            min_loss = None
    else:
        state = torch.load(args.load_state, map_location=device)

        start_epoch = state['epoch']

        load_model_state_dict(model.module, state['model'],
                              strict=args.load_state_strict)

        if 'optimizer' in state:
            optimizer.load_state_dict(state['optimizer'])
        if 'scheduler' in state:
            scheduler.load_state_dict(state['scheduler'])

        torch.set_rng_state(state['rng'].cpu())  # move rng state back

        if rank == 0:
            min_loss = state['min_loss']

            logging.info('state at epoch {} loaded from {}'.format(
                state['epoch'], args.load_state))

        del state

    torch.backends.cudnn.benchmark = True

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    logger = None
    if rank == 0:
        logger = get_logger(args)

    if rank == 0:
        logging.info('pytorch {}'.format(torch.__version__))
        logging.info(pformat(vars(args)))
        sys.stdout.flush()
        ckpt_dir = f"checkpoints/{args.experiment_title}_{strftime('%Y-%m-%d-%H-%M-%S', gmtime())}"
        os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_loss = train(epoch, train_loader, model, criterion,
                           optimizer, scheduler, logger, device, args)
        epoch_loss = train_loss

        if args.val:
            val_loss = validate(epoch, val_loader, model, criterion,
                                logger, device, args)
            #epoch_loss = val_loss

        if args.reduce_lr_on_plateau:
            scheduler.step(epoch_loss[0] * epoch_loss[1])

        if rank == 0:
            logger.flush()

            if min_loss is None or epoch_loss[2] < min_loss:
                min_loss = epoch_loss[2]

            state = {
                'epoch': epoch + 1,
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'rng': torch.get_rng_state(),
                'min_loss': min_loss,
            }

            state_file = f'{ckpt_dir}/state_{epoch + 1}.pt'
            torch.save(state, state_file)
            del state

            tmp_link = '{}.pt'.format(time.time())
            os.symlink(state_file, tmp_link)  # workaround to overwrite
            os.rename(tmp_link, ckpt_link)

    dist.destroy_process_group()


def train(epoch, loader, model, criterion,
          optimizer, scheduler, logger, device, args):
    model.train()
    st = time.time()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    epoch_loss = torch.zeros(4, dtype=torch.float64, device=device)
    for i, data in tqdm(enumerate(loader)):
        batch = epoch * len(loader) + i + 1
        style, input, target = data['style'], data['input'], data['target']

        style = style.to(device, non_blocking=True)
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(input, style)
        kl = get_kl_loss(model); kl /= args.batch_size
        if batch <= 5 and rank == 0:
            logging.info(f'##### batch : {batch}')
            logging.info(f'style shape : {style.shape}')
            logging.info(f'input shape : {input.shape}')
            logging.info(f'output shape : {output.shape}')
            logging.info(f'target shape : {target.shape}')

        if (hasattr(model.module, 'scale_factor')
                and model.module.scale_factor != 1):
            input = resample(input, model.module.scale_factor, narrow=False)
        input, output, target = narrow_cast(input, output, target)
        if batch <= 5 and rank == 0:
            logging.info(f'narrowed shape : {output.shape}')

        lag_out, lag_tgt = output, target
        eul_out, eul_tgt = lag2eul([lag_out, lag_tgt], **args.misc_kwargs)
        if batch <= 5 and rank == 0:
            logging.info(f'Eulerian shape : {eul_out.shape}')

        lag_loss = criterion(lag_out, lag_tgt)
        eul_loss = criterion(eul_out, eul_tgt)
        loss = lag_loss ** 3 * eul_loss
        epoch_loss[0] += lag_loss.detach()
        epoch_loss[1] += eul_loss.detach()
        epoch_loss[2] += loss.detach()
        epoch_loss[3] += kl.detach()

        optimizer.zero_grad()
        (torch.log(loss) + kl).backward()  # NOTE actual loss is log(loss) + KL
        optimizer.step()
        grads = get_grads(model)

        if batch % args.log_interval == 0:
            log_every_interval(
                lag_loss=lag_loss, eul_loss=eul_loss, loss=loss, kl=kl,
                grads=grads,
                batch=batch, logger=logger, rank=rank, world_size=world_size,
                tag='train',
            )

    if args.distributed:
        dist.all_reduce(epoch_loss)
    epoch_loss /= len(loader) * world_size
    if rank == 0:
        log_every_epoch(
            args=args,
            lag_loss=epoch_loss[0], eul_loss=epoch_loss[1], kl=epoch_loss[3],
            input=input, lag_tgt=lag_tgt, eul_tgt=eul_tgt,
            model=model,
            logger=logger, epoch=epoch, tag='train',
        )

    print(st, time.time() - st)
    return epoch_loss


def validate(epoch, loader, model, criterion, logger, device, args):
    model.eval()

    rank = dist.get_rank() if args.distributed else 0
    world_size = dist.get_world_size() if args.distributed else 1

    epoch_loss = torch.zeros(4, dtype=torch.float64, device=device)

    with torch.no_grad():
        for data in tqdm(loader):
            style, input, target = data['style'], data['input'], data['target']

            style = style.to(device, non_blocking=True)
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(input, style)

            if (hasattr(model.module, 'scale_factor')
                    and model.module.scale_factor != 1):
                input = resample(input, model.module.scale_factor, narrow=False)
            input, output, target = narrow_cast(input, output, target)

            lag_out, lag_tgt = output, target
            eul_out, eul_tgt = lag2eul([lag_out, lag_tgt], **args.misc_kwargs)

            lag_loss = criterion(lag_out, lag_tgt)
            eul_loss = criterion(eul_out, eul_tgt)
            loss = lag_loss ** 3 * eul_loss
            epoch_loss[0] += lag_loss.detach()
            epoch_loss[1] += eul_loss.detach()
            epoch_loss[2] += loss.detach()
            epoch_loss[3] += get_kl_loss(model).detach()

    if args.distributed:
        dist.all_reduce(epoch_loss)
    epoch_loss /= len(loader) * world_size
    if rank == 0:
        log_every_epoch(
            args=args,
            lag_loss=epoch_loss[0], eul_loss=epoch_loss[1], kl=epoch_loss[3],
            input=input, lag_tgt=lag_tgt, eul_tgt=eul_tgt,
            model=model,
            logger=logger, epoch=epoch, tag='val',
        )

    return epoch_loss


def dist_init(rank, args):
    dist_file = 'dist_addr'

    if rank == 0:
        addr = socket.gethostname()

        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((addr, 0))
            _, port = s.getsockname()

        args.dist_addr = 'tcp://{}:{}'.format(addr, port)

        with open(dist_file, mode='w') as f:
            f.write(args.dist_addr)
    else:
        while not os.path.exists(dist_file):
            time.sleep(1)

        with open(dist_file, mode='r') as f:
            args.dist_addr = f.read()

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_addr,
        world_size=args.world_size,
        rank=rank,
    )
    dist.barrier()

    if rank == 0:
        os.remove(dist_file)


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        m.weight.data.normal_(0.0, args.init_weight_std)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.SyncBatchNorm, nn.LayerNorm, nn.GroupNorm,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
        if m.affine:
            # NOTE: dispersion from DCGAN, why?
            m.weight.data.normal_(1.0, args.init_weight_std)
            m.bias.data.fill_(0)


def set_requires_grad(module, requires_grad=False):
    for param in module.parameters():
        param.requires_grad = requires_grad


def get_grads(model):
    """gradients of the weights of the first and the last layer
    """
    grads = list(p.grad for n, p in model.named_parameters()
                 if '.weight' in n)
    grads = [grads[0], grads[-1]]
    grads = [g.detach().norm() for g in grads]
    return grads


def log_every_epoch(
    args,
    lag_loss:float, eul_loss:float, kl: float,
    input: torch.Tensor, lag_tgt: torch.Tensor, eul_tgt: torch.Tensor,
    model: torch.nn.parallel.DistributedDataParallel,
    logger, epoch: int, tag: str,
):
    lag_mean, lag_var = compute_mean_var(model, input, sample_size=args.sample_size)
    eul_mean = lag2eul([lag_mean])[0]

    logger.add_scalar(f'loss/epoch/{tag}/lag', lag_loss,
                      global_step=epoch+1)
    logger.add_scalar(f'loss/epoch/{tag}/eul', eul_loss,
                      global_step=epoch+1)
    logger.add_scalar(f'loss/epoch/{tag}/lxe', lag_loss * eul_loss,
                      global_step=epoch+1)
    logger.add_scalar(f'loss/epoch/{tag}/kl', kl,
                      global_step=epoch+1)

    try:
        fig = plt_slices(
            input[-1], lag_mean[-1], lag_tgt[-1], lag_mean[-1] - lag_tgt[-1],
                    eul_mean[-1], eul_tgt[-1], eul_mean[-1] - eul_tgt[-1],
            title=['in', 'lag_mean', 'lag_tgt', 'lag_mean - lag_tgt',
                        'eul_mean', 'eul_tgt', 'eul_mean - eul_tgt'],
            **args.misc_kwargs,
        )
        logger.add_figure(f'fig/{tag}/slices', fig, global_step=epoch+1)
        fig.clf()
    except Exception as e:
        print(e)
        error_dump_dir = 'error_dump'
        os.makedirs(error_dump_dir, exist_ok=True)
        vars = {
            'input': input,
            'lag_mean': lag_mean, 'lag_tgt': lag_tgt,
            'eul_mean': eul_mean, 'eul_tgt': eul_tgt,
            'epoch': epoch + 1,
            'model': model.module.state_dict(),
        }
        error_dump_filename = f'epoch-{epoch+1}-train.pkl'
        with open(os.path.join(error_dump_dir, error_dump_filename), 'wb') as f:
            pickle.dump(vars, f)

    fig = plt_power_with_error_bar(
        {'mean': input, 'var': None},
        {'mean': lag_mean, 'var': lag_var},
        {'mean': lag_tgt, 'var': None},
        label=['in', 'lag_mean', 'lag_tgt'],
        use_pylians=False,
    )
    logger.add_figure(f'fig/{tag}/power/lag-err-bar', fig, global_step=epoch+1)
    fig.clf()

    fig = plt_power(1.0,
        dis=[input, lag_mean, lag_tgt],
        label=['in', 'eul_mean', 'eul_tgt'],
        **args.misc_kwargs,
    )
    logger.add_figure(f'fig/{tag}/power/eul', fig, global_step=epoch+1)
    fig.clf()


def log_every_interval(
    args,
    lag_loss: torch.Tensor, eul_loss: torch.Tensor, loss: torch.Tensor, kl: torch.Tensor,
    grads: torch.Tensor,
    batch: int, logger, rank: int, world_size: int,
    tag: str,
):
    if args.distributed:
        dist.all_reduce(lag_loss)
        dist.all_reduce(eul_loss)
        dist.all_reduce(loss)
    lag_loss /= world_size
    eul_loss /= world_size
    loss /= world_size
    if rank == 0:
        logger.add_scalar(f'loss/batch/{tag}/lag', lag_loss.item(),
                            global_step=batch)
        logger.add_scalar(f'loss/batch/{tag}/eul', eul_loss.item(),
                            global_step=batch)
        logger.add_scalar(f'loss/batch/{tag}/lxe', lag_loss.item() * eul_loss.item(),
                            global_step=batch)
        logger.add_scalar(f'loss/batch/{tag}/kl', kl.item(),
                            global_step=batch)

        logger.add_scalar('grad/first', grads[0], global_step=batch)
        logger.add_scalar('grad/last', grads[-1], global_step=batch)


def compute_mean_var(
    model: torch.nn.parallel.DistributedDataParallel,
    input: torch.Tensor,
    sample_size: int,
):
    with torch.no_grad():
        stats = RunningStats()
        for _ in range(sample_size):
            output = model(input)
            stats.push(output)
        mean, var = stats.mean(), stats.variance()

    return mean, var
