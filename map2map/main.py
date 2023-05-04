from .args import get_args
from . import train_bnn
from . import train_gnll
from . import train
from . import test
from . import generate

import logging
import sys
import os
import argparse
from time import gmtime, strftime


def setup_logging(args: argparse.Namespace):
    os.makedirs('logs', exist_ok=True)

    experiment_title = []
    if args.experiment_title:
        experiment_title.append(args.experiment_title)
    experiment_title.append(strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
    experiment_title = '_'.join(experiment_title)

    targets = logging.StreamHandler(sys.stdout), logging.FileHandler(
        os.path.join('logs', f'{experiment_title}.txt')
    )
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=targets
    )


def main():

    args = get_args()
    setup_logging(args)

    if args.mode == 'train-bnn':
        train_bnn.node_worker(args)
    if args.mode == 'train-gnll':
        train_gnll.node_worker(args)
    elif args.mode == 'train':
        train.node_worker(args)
    elif args.mode == 'test':
        test.test(args)
    elif args.mode == 'generate':
        generate.generate(args)

if __name__ == '__main__':
    main()
