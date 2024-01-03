timestamp="20231221_100437_39_4_[bb]_sd10"
folder_name="debug"

python -u mvtec_loco_ad_evaluation/evaluate_experiment.py \
  --object_name "breakfast_box" \
  --dataset_base_dir "./datasets/loco/" \
  --anomaly_maps_dir "./outputs/${folder_name}/output_${timestamp}/anomaly_maps/mvtec_loco/" \
  --output_dir "./outputs/${folder_name}/output_${timestamp}/metrics/mvtec_loco/" \
  --timestamp "${timestamp}" \
  --folder_name "${folder_name}" \
  # --generate_visual \



# breakfast_box
# juice_bottle
# pushpins
# screw_bag
# splicing_connectors