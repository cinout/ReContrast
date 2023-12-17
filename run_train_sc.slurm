#!/bin/bash

###SBATCH --partition=gpu-a100

###SBATCH --partition=feit-gpu-a100
###SBATCH --qos=feit

#SBATCH --partition=deeplearn
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg4|dlg5

#SBATCH --job-name="recontra"
#SBATCH --account=punim1623
#SBATCH --time=0-15:00:00

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
  --subdataset splicing_connectors \
  --seeds 
  --stg1_ckpt outputs/stg1_only/outputs_20231212_234343_47_9_[sc]_sd10/trainings/mvtec_loco/splicing_connectors/model_stg1.pth \
  --iters_stg2 6000 \

my-job-stats -a -n -s
##