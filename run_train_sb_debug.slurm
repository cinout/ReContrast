#!/bin/bash

###SBATCH --partition=gpu-a100

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

###SBATCH --partition=deeplearn
###SBATCH --qos=gpgpudeeplearn
###SBATCH --constraint=dlg4|dlg5

#SBATCH --job-name="sb"
#SBATCH --account=punim1623
#SBATCH --time=0-00:40:00

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
  --subdataset screw_bag \
  --seeds 10 \
  --stg1_ckpt outputs/stg1_only/outputs_20231212_234343_9_7_[sb]_sd10/trainings/mvtec_loco/screw_bag/model_stg1.pth \
  --stg2_ckpt outputs/attn10_deconv/output_20240114_230009_74_96_[sb]_sd10/trainings/mvtec_loco/screw_bag/model_stg2.pth \
  --iters_stg2 8000 \
  --loss_mode extreme \
  --logicano_select absolute \
  --num_logicano 10 \
  --lr_stg2 0.00005 \
  --attn_count 10 \
  --attn_in_deconv \
  --debug_mode_2 \
  # --debug_logicano_count 20 \

my-job-stats -a -n -s
##