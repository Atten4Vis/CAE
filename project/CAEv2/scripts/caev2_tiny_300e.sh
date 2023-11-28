tmp_my_name=caev2_tiny_300e
my_name=${tmp_my_name%.*}

OUTPUT_DIR='./output/'$my_name
echo $OUTPUT_DIR

DATA_PATH=/path/to/imagenet1k/train
TOKENIZER_PATH=./clip_model
TOKENIZER_TYPE=clip
NUM_OUT_DIM=512
CROP_MIN_SIZE=0.4
CROP_MAX_SIZE=1.0
MAIN_LOSS_WEIGHT=1
ALIGN_LOSS_WEIGHT=0
SECOND_INPUT_SIZE=224
NUM_MASK_PATCHES=30

PORT=8942
ADDR=ADDR_FOR_THIS_MACHINE    
NNODES=1     
RANK=RANK_FOR_THIS_MACHINE 


# ============================ pretraining ============================
OMP_NUM_THREADS=1 python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --nnodes=$NNODES \
  --node_rank=$RANK \
  --master_addr=$ADDR \
  --master_port=$PORT \
  tools/run_pretraining.py \
  --model_type caev2 \
  --data_path ${DATA_PATH} \
  --output_dir ${OUTPUT_DIR} \
  --model cae_tiny_patch16_224_8k_vocab \
  --discrete_vae_weight_path ${TOKENIZER_PATH} \
  --discrete_vae_type $TOKENIZER_TYPE \
  --second_input_size $SECOND_INPUT_SIZE \
  --second_interpolation bicubic \
  --num_out_dim $NUM_OUT_DIM \
  --crop_min_size $CROP_MIN_SIZE \
  --crop_max_size $CROP_MAX_SIZE \
  --batch_size 256 \
  --lr 1.5e-3 --warmup_epochs 10 --epochs 300 \
  --clip_grad 3.0 --layer_scale_init_value 0.1 \
  --imagenet_default_mean_and_std \
  --color_jitter 0.4 \
  --drop_path 0.1 \
  --mask_generator block \
  --num_mask_patches $NUM_MASK_PATCHES \
  --decoder_layer_scale_init_value 0.1 \
  --no_auto_resume \
  --save_ckpt_freq 100 \
  --exp_name $my_name \
  --regressor_depth 1 \
  --decoder_depth 0 \
  --decoder_embed_dim 96 \
  --align_loss_weight 0 \
  --latent_alignment_loss_weight 1


# ============================ linear probing ============================
sleep 20s
DATA_PATH=/path/to/imagenet1k/
MODEL_PATH=$OUTPUT_DIR'/'$tmp_my_name'_checkpoint-299.pth' #/path/to/pretrained/model

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=$NNODES \
    --node_rank=$RANK \
    --master_addr=$ADDR \
    --master_port=$PORT \
    tools/run_linear.py \
    --model cae_tiny_patch16_224 --data_path $DATA_PATH \
    --finetune $MODEL_PATH \
    --nb_classes 1000 \
    --batch_size 2048 \
    --epochs 90 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval --data_path ${DATA_PATH} \
    --output_dir $OUTPUT_DIR'/LIN/' \
    --log_dir $OUTPUT_DIR'/LIN/' \
    --enable_linear_eval \
    --use_cls \
    --dist_eval \
    --save_freq 50 \
    --disable_rel_pos_bias \
    --linear_type standard \
    --exp_name $my_name

# ============================finetune ============================
sleep 20s
DATA_PATH=/path/to/imagenet1k/
MODEL_PATH=$OUTPUT_DIR'/'$tmp_my_name'_checkpoint-299.pth' #/path/to/pretrained/model

OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=$NNODES \
    --node_rank=$RANK \
    --master_addr=$ADDR \
    --master_port=$PORT \
    tools/run_class_finetuning.py \
    --model cae_tiny_patch16_224  \
    --data_path $DATA_PATH \
    --finetune $MODEL_PATH \
    --nb_classes 1000 \
    --data_set IMNET \
    --imagenet_default_mean_and_std \
    --output_dir $OUTPUT_DIR'/Finetune/' \
    --log_dir $OUTPUT_DIR'/Finetune/' \
    --save_ckpt_freq 20 \
    --batch_size 128 \
    --lr 4e-3 \
    --update_freq 1 \
    --warmup_epochs 5 \
    --epochs 100 \
    --drop_path 0.0 \
    --weight_decay 0.05 \
    --layer_decay 0.85 \
    --aa rand-m10-mstd0.5-inc1 \
    --smoothing 0.0 \
    --mixup 0.2 \
    --cutmix 0.0 \
    --color_jitter 0.3 \
    --sin_pos_emb \
    --dist_eval \
    --no_auto_resume \
    --exp_name $my_name \
