#!/bin/bash

#SBATCH --job-name=train-gnll-data-parallel
#SBATCH --output=%x-%j.out
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-32:1
#SBATCH --exclusive
#SBATCH --time=2-00:00:00


# echo "This is a minimal example. See --help or args.py for more," \
#      "e.g. on augmentation, cropping, padding, and data division."
# echo "Training on 2 nodes with 8 GPUs."
# echo "input data: {train,val,test}/R{0,1}-*.npy"
# echo "target data: {train,val,test}/D{0,1}-*.npy"
# echo "normalization functions: {R,D}{0,1} in ./RnD.py," \
#      "see map2map/data/norms/*.py for examples"
# echo "model: Net in ./model.py, see map2map/models/*.py for examples"
# echo "Training with placeholder learning rate 1e-4 and batch size 1."


# hostname; pwd; date


# set computing environment, e.g. with module or anaconda

#module load python
#module list

#source $HOME/anaconda3/bin/activate pytorch_env
#conda info

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
    --experiment-title "train-forward-GNLL-data-paralell"

        # --load-state "/home/ajliang/search/model_weights/paper_fwd_d2d_weights.pt" \
        # --local \

date



# --train-in-patterns "/user_data/ajliang/Linear/train/*/4/dis.npy" \
# --train-style-pattern "/user_data/ajliang/Linear/train/*/4/params.npy" \

# --val-in-patterns "/user_data/ajliang/Linear/val/*/4/dis.npy" \
# --val-style-pattern "/user_data/ajliang/Linear/val/*/4/params.npy" \
# --val-tgt-patterns "/user_data/ajliang/Nonlinear/val/*/4/dis.npy" \

