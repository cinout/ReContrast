#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg4|dlg5

#SBATCH --job-name="jb"
#SBATCH --account=punim1623
#SBATCH --time=0-05:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

### "ntasks-per-node" should have same value as "res=gpu:"

#SBATCH --mem=40G

module purge

eval "$(conda shell.bash hook)"
conda activate eff

set -e
set -x

python -u recontrast_mvtecloco.py \
  --subdataset juice_bottle \
  --seeds 10 20 30 \
  --stg1_ckpt outputs/stg1_only/output_20240123_141224_0_55_[jb]_sd10/trainings/mvtec_loco/juice_bottle/model_stg1.pth \
  --iters_stg2 8000 \
  --loss_mode extreme \
  --logicano_select percent \
  --num_logicano 10 \
  --percent_logicano 1.0 \
  --lr_stg2 0.00001 \
  --attn_count 10 \
  --attn_in_deconv \
  --similarity_priority pointwise \
  --note "pointwise, 100% logicano, fixed_ref 10%, compress_bn" \
  --fixed_ref \
  --fixed_ref_percent 0.1 \
  --compress_bn \
  
  # --debug_mode \
  # --debug_logicano_count 20 \

my-job-stats -a -n -s
##