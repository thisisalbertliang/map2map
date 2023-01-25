"""
Running this script (i.e. `python -m map2map.repro_fwd_model_bug`) will reproduce the bug that I am getting for the forward model

The generated plots are saved in the folder `runs/temp/` and can be viewed with `tensorboard --logdir runs`

Before running, please replace the paths (marked with `# TODO`) with your own paths
"""

from time import gmtime, strftime
from tqdm import tqdm
from itertools import islice
import argparse
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from .data.fields import FieldDataset
from .data.norms.cosmology import dis
from .models.d2d import StyledVNet
from .models.lag2eul import lag2eul
from .utils import plt_power, plt_slices, load_model_state_dict


def load_model(
    args: argparse.Namespace,
    device: torch.device,
    path_to_model_state: str,
):
    model: StyledVNet = StyledVNet(
        style_size=args.style_size,
        in_chan=sum(args.in_chan),
        out_chan=sum(args.out_chan),
    )
    model.to(device)

    state = torch.load(path_to_model_state, map_location=device)
    load_model_state_dict(model, state["model"], strict=True)
    print(
        f"Loaded model from {path_to_model_state}, which was trained for {state['epoch']} epochs.",
        flush=True
    )

    return model


def get_logger(args: argparse.Namespace):
    experiment_name = [args.experiment_name]
    experiment_name.append(strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    experiment_name = '_'.join(experiment_name)

    logger = SummaryWriter(
        os.path.join("runs", "temp", experiment_name)
    )
    logger.add_text(
        tag="Arguments",
        text_string=str(vars(args)),
        global_step=0,
    )

    args.experiment_name = experiment_name
    return logger


def plot(
    logger: SummaryWriter,
    input: torch.Tensor,
    lag_out: torch.Tensor, lag_tgt: torch.Tensor,
    eul_out: torch.Tensor, eul_tgt: torch.Tensor,
    crop_idx,
):
    # plot Lagrangian power spectrum
    fig = plt_power(
        input, lag_out, lag_tgt,
        label=["lag_in", "lag_out", "lag_tgt"],
    )
    logger.add_figure(
        f"fig/power/lag",
        fig,
        global_step=crop_idx,
    )
    fig.clf()

    # plot Eulerian power spectrum
    fig = plt_power(
        1.0,
        dis=[input, lag_out, lag_tgt],
        label=["eul_in", "eul_out", "eul_tgt"],
    )
    logger.add_figure(
        f"fig/power/eul",
        fig,
        global_step=crop_idx,
    )
    fig.clf()

    # plot slices of Lagrangian and Eulerian
    fig = plt_slices(
        input[-1],
        lag_out[-1], lag_tgt[-1],
        eul_out[-1], eul_tgt[-1],
        title=[
            f'in (crop-idx={crop_idx})',
            f'lag_out (crop-idx={crop_idx})', f'lag_tgt (crop-idx={crop_idx})',
            f'eul_out (crop-idx={crop_idx})', f'eul_tgt (crop-idx={crop_idx})',
        ],
    )
    logger.add_figure(
        f'fig/slices',
        fig,
        global_step=crop_idx,
    )
    fig.clf()


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cosmology = "LH0045"

    args = argparse.Namespace(
        input_path=f"/user_data/ajliang/Linear/val/{cosmology}/4/dis.npy", # TODO: replace with your own path for https://app.globus.org/file-manager?origin_id=9da966a0-58ec-11ed-89dc-ede5bae4f491&origin_path=%2FLinear%2FLH0045%2F4%2F
        style_path=f"/user_data/ajliang/Linear/val/{cosmology}/4/params.npy", # TODO: replace with your own path for https://app.globus.org/file-manager?origin_id=9da966a0-58ec-11ed-89dc-ede5bae4f491&origin_path=%2FLinear%2FLH0045%2F4%2F
        target_path=f"/user_data/ajliang/Nonlinear/val/{cosmology}/4/dis.npy", # TODO: replace with your own path for https://app.globus.org/file-manager?origin_id=9da966a0-58ec-11ed-89dc-ede5bae4f491&origin_path=%2FNonlinear%2FLH0045%2F4%2F
        crop=32,
        in_pad=48,
        experiment_name=f"fwd-model-bug-{cosmology}", # an arbitrary name for tensorboard purposes
    )

    dataset = FieldDataset(
        style_pattern=args.style_path,
        in_patterns=[args.input_path],
        tgt_patterns=[args.target_path],
        crop=args.crop,
        in_pad=args.in_pad,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    args.style_size = dataset.style_size
    args.in_chan = dataset.in_chan
    args.out_chan = dataset.tgt_chan
    model = load_model(
        args,
        device=device,
        path_to_model_state="/home/ajliang/repro/map2map/map2map/weights/d2d_weights.pt" # TODO: replace with your own path for https://github.com/dsjamieson/map2map_emu/blob/emud2d/map2map/weights/d2d_weights.pt
    )
    model.eval()

    logger = get_logger(args) # get a logger for tensorboard

    loader = islice(loader, 10) # just plot the first 10 crops
    for i, data in tqdm(enumerate(loader)):
        style, input, target = data['style'].to(device), data['input'].to(device), data['target'].to(device)

        with torch.no_grad():
            output = model(input, style)
            """
            Running this line `norms.cosmology.dis(output, undo=True)` below causes `ValueError: math domain error` for me. Since I am not sure if this normalization is needed, I have commented it out.
            @dsjamieson, could you please help me understand what this normalization is for if it's actually needed?
            """
            # dis(output, undo=True)

            lag_out, lag_tgt = output, target
            eul_out, eul_tgt = lag2eul([lag_out, lag_tgt])

            plot(
                logger=logger, input=input,
                lag_out=lag_out, lag_tgt=lag_tgt,
                eul_out=eul_out, eul_tgt=eul_tgt,
                crop_idx=i,
            )
