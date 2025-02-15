python -u recontrast_mvtecloco.py \
  --subdataset splicing_connectors \
  --seeds 30 \
  --loss_mode extreme \
  --stg1_ckpt outputs/stg1_only/outputs_20231212_234313_77_49_[bb]_sd10/trainings/mvtec_loco/breakfast_box/model_stg1.pth \
  --stg2_ckpt outputs/attn10_deconv/output_20240114_233125_53_23_[sc]_sd10/trainings/mvtec_loco/splicing_connectors/model_stg2.pth \
  --iters_stg2 10 \
  --logicano_select percent \
  --num_logicano 10 \
  --lr_stg2 0.00001 \
  --attn_in_deconv \
  --attn_count 10 \
  --debug_mode_2 \
  # --logicano_only \
  # --stg2_ckpt outputs/stg2_debug/output_20231219_161047_50_43_[bb]_sd10/trainings/mvtec_loco/breakfast_box/model_stg2.pth



# breakfast_box
# juice_bottle
# pushpins
# screw_bag
# splicing_connectors