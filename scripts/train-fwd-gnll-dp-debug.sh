#!/bin/bash

#SBATCH --job-name=train-gnll-data-parallel
#SBATCH --output=%x-%j.out
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-32:1
#SBATCH --exclusive
#SBATCH --time=2-00:00:00

export SLURM_JOB_NUM_NODES=1

hostname; pwd; date

DATA_DIR="/ocean/projects/cis230021p/lianga/quijote"
TRAIN_QUIJOTE_NUMBER="LH1045"
VAL_QUIJOTE_NUMBER="LH0482"

python m2m.py train-gnll \
    --train-in-patterns ${DATA_DIR}/${TRAIN_QUIJOTE_NUMBER}/lin.npy \
    --train-style-pattern ${DATA_DIR}/${TRAIN_QUIJOTE_NUMBER}/params.npy \
    --train-tgt-patterns ${DATA_DIR}/${TRAIN_QUIJOTE_NUMBER}/nonlin.npy \
    --val-in-patterns ${DATA_DIR}/${VAL_QUIJOTE_NUMBER}/lin.npy \
    --val-style-pattern ${DATA_DIR}/${VAL_QUIJOTE_NUMBER}/params.npy \
    --val-tgt-patterns ${DATA_DIR}/${VAL_QUIJOTE_NUMBER}/nonlin.npy \
    --model model.StyledVNet \
    --in-norms cosmology.dis \
    --tgt-norms cosmology.dis \
    --callback-at . \
    --lr 1e-4 --batch-size 1 \
    --epochs 1024 \
    --seed 10620 \
    --loader-workers 2 \
    --batch-size 1 \
    --in-pad 48 \
    --tgt-pad 48 \
    --crop 64 \
    --experiment-title "train-fwd-pseudo-GNLL-dp-DEBUG"

date
