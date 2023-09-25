import argparse
from time import gmtime, strftime
from glob import glob
from torch.utils.tensorboard import SummaryWriter


def get_logger(args: argparse.Namespace):
    experiment_title = []
    if args.experiment_title:
        experiment_title.append(args.experiment_title)

    experiment_title.append(f'epochs={args.epochs}')
    experiment_title.append(strftime("%Y-%m-%d-%H-%M-%S", gmtime()))

    logger = SummaryWriter(
        log_dir=f"runs/{'_'.join(experiment_title)}"
    )
    logger.add_text(
        tag='Arguments',
        text_string=str(vars(args)),
        global_step=0
    )
    
    if hasattr(args, "train_style_pattern"):
        logger.add_text(
            tag='Train Files',
            text_string=str({
                'Style Files': str(sorted(glob(args.train_style_pattern))),
                'Input Files': str(list(zip(*[sorted(glob(p)) for p in args.train_in_patterns]))),
                'Target Files': str(list(zip(*[sorted(glob(p)) for p in args.train_tgt_patterns])))
            }),
            global_step=0
        )
    if hasattr(args, "val_style_pattern"):
        logger.add_text(
            tag='Val Files',
            text_string=str({
                'Style Files': str(sorted(glob(args.val_style_pattern))),
                'Input Files': str(list(zip(*[sorted(glob(p)) for p in args.val_in_patterns]))),
                'Target Files': str(list(zip(*[sorted(glob(p)) for p in args.val_tgt_patterns])))
            }),
            global_step=0
        )
    if hasattr(args, "test_style_pattern"):
        logger.add_text(
            tag='Test Files',
            text_string=str({
                'Style Files': str(sorted(glob(args.test_style_pattern))),
                'Input Files': str(list(zip(*[sorted(glob(p)) for p in args.test_in_patterns]))),
                'Target Files': str(list(zip(*[sorted(glob(p)) for p in args.test_tgt_patterns])))
            }),
            global_step=0
        )
    
    return logger
