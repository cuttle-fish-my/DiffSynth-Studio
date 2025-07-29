EXP="token_loss_0.1-lora_alpha_64_no_mse"

CUDA_VISIBLE_DEVICES="0,1,2,3" python examples/train/flux/train_flux_lora.py \
  --pretrained_text_encoder_path models/FLUX/FLUX.1-dev/text_encoder/model.safetensors \
  --pretrained_text_encoder_2_path models/FLUX/FLUX.1-dev/text_encoder_2 \
  --pretrained_dit_path models/FLUX/FLUX.1-dev/flux1-dev.safetensors \
  --pretrained_vae_path models/FLUX/FLUX.1-dev/ae.safetensors \
  --preset_lora_path models/lora/entity_control/model_bf16.safetensors \
  --dataset_path cywang143/OverLayBench_Dataset \
  --output_path ./exp/$EXP \
  --max_epochs 50 \
  --steps_per_epoch 5000 \
  --height 1024 \
  --width 1024 \
  --center_crop \
  --precision "bf16" \
  --learning_rate 1e-4 \
  --lora_rank 64 \
  --lora_alpha 64 \
  --accumulate_grad_batches 2 \
  --use_gradient_checkpointing \
  --align_to_opensource_format \
  --training_strategy "deepspeed_stage_1" \
  --use_swanlab \
  --swanlab_mode cloud \
  --swanlab_project_name "overlap_eligen" \
  --swanlab_experiment_name $EXP \
  --token_loss_weight 0.1