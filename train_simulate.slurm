#!/bin/bash

remote=false
while getopts 'r' flag; do
    case "${flag}" in
        r) remote=true ;;
    esac
done

if $remote
then
    srun --unbuffered --partition=gpu --gres=gpu:1 python3 ./m2m.py train \
        --train-style-pattern "/user_data/ajliang/Linear/train/*/4/params.npy" \
        --train-in-patterns "/user_data/ajliang/Linear/train/*/4/dis.npy" \
        --train-tgt-patterns "/user_data/ajliang/Nonlinear/train/*/4/dis.npy" \
        --val-style-pattern "/user_data/ajliang/Linear/val/*/4/params.npy" \
        --val-in-patterns "/user_data/ajliang/Linear/val/*/4/dis.npy" \
        --val-tgt-patterns "/user_data/ajliang/Nonlinear/val/*/4/dis.npy" \
        --in-norms cosmology.dis \
        --augment --aug-shift 16 \
        --crop 40 --crop-step 40 --pad 48 \
        --model model.StyledVNet --callback-at . \
        --lr 4e-4 --optimizer AdamW --optimizer-args '{"betas": [0.9, 0.999]}' \
        --reduce-lr-on-plateau --scheduler-args '{"factor": 0.1, "patience": 2, "threshold": 1e-3, "verbose": true}' \
        --batches 1 --loader-workers 2 --div-data --div-shuffle-dist 9 \
        --epochs 200 \
        --log-interval 50 --experiment-title "TRAIN-forward-model"
else
    python3 ./m2m.py train \
        --train-style-pattern "/user_data/ajliang/Linear/train/LH0030/4/params.npy" \
        --train-in-patterns "/user_data/ajliang/Linear/train/LH0030/4/dis.npy" \
        --train-tgt-patterns "/user_data/ajliang/Nonlinear/train/LH0030/4/dis.npy" \
        --val-style-pattern "/user_data/ajliang/Linear/val/LH0045/4/params.npy" \
        --val-in-patterns "/user_data/ajliang/Linear/val/LH0045/4/dis.npy" \
        --val-tgt-patterns "/user_data/ajliang/Nonlinear/val/LH0045/4/dis.npy" \
        --in-norms cosmology.dis \
        --augment --aug-shift 16 \
        --crop 40 --crop-step 40 --pad 48 \
        --model model.StyledVNet --callback-at . \
        --lr 4e-4 --optimizer AdamW --optimizer-args '{"betas": [0.9, 0.999]}' \
        --reduce-lr-on-plateau --scheduler-args '{"factor": 0.1, "patience": 2, "threshold": 1e-3, "verbose": true}' \
        --batches 1 --loader-workers 2 --div-data --div-shuffle-dist 9 \
        --epochs 200 \
        --log-interval 50 --experiment-title "ALBERT-FWD-MODEL"
fi
