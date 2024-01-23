python -u recontrast_mvtecloco.py \
  --subdataset breakfast_box \
  --seeds 20 \
  --loss_mode extreme \
  --stg1_ckpt outputs/stg1_only/outputs_20231212_234313_77_49_[bb]_sd10/trainings/mvtec_loco/breakfast_box/model_stg1.pth \
  --iters_stg2 10 \
  --logicano_select absolute \
  --num_logicano 10 \
  --lr_stg2 0.00001 \
  --attn_in_deconv \
  --attn_count 10 \
  --similarity_priority pointwise \
  --note "" \
  --fixed_ref \
  --fixed_ref_percent 0.1 \
  
  # --debug_mode_3 \
  # --logicano_only \
  # --stg2_ckpt outputs/attn10_deconv/output_20240114_225418_59_17_[jb]_sd10/trainings/mvtec_loco/juice_bottle/model_stg2.pth \



# breakfast_box
# juice_bottle
# pushpins
# screw_bag
# splicing_connectors