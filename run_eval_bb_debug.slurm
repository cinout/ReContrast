#!/bin/bash

###SBATCH --partition=cascade

#SBATCH --partition=feit-gpu-a100
#SBATCH --qos=feit

#SBATCH --job-name="eval_bb"
#SBATCH --account=punim1623
#SBATCH --time=0-01:10:00


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1


#SBATCH --cpus-per-task=1
#SBATCH --mem=40G

### SBATCH --mail-user=haitianh@student.unimelb.edu.au
### SBATCH --mail-type=BEGIN

module purge

eval "$(conda shell.bash hook)"
conda activate eff

timestamp="20231219_172145_8_67_[bb]_sd10" 
folder_name="stg2_debug"

python -u mvtec_loco_ad_evaluation/evaluate_experiment.py \
  --object_name "breakfast_box" \
  --dataset_base_dir "./datasets/loco/" \
  --anomaly_maps_dir "./outputs/${folder_name}/output_${timestamp}/anomaly_maps/mvtec_loco/" \
  --output_dir "./outputs/${folder_name}/output_${timestamp}/metrics/mvtec_loco/" \
  --timestamp "${timestamp}" \
  --folder_name "${folder_name}" \
  --generate_visual \

## FIXME: update timestamp
##Log this job's resource usage stats### 
my-job-stats -a -n -s
##