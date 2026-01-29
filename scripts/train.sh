#!/bin/bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ----------------------------------
# Generate prior datasets on the fly
# ----------------------------------
EXPERIMENTS_BASE="/path/to/checkpoints/"


NAME="Softari"
SCRIPT=run_Softari

mkdir -p "$EXPERIMENTS_BASE/logs"

max_features=64
max_classes=10

lr=1e-4

LOG_FILE="$EXPERIMENTS_BASE/logs/output_mix_mask_em_small_${max_features}_${max_classes}_${NAME}_${lr}_stage1.log"

export NUMEXPR_NUM_THREADS=10
echo > "$LOG_FILE"
# pretrained
nohup env PYTHONUNBUFFERED=1 torchrun --nproc_per_node=4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 exp/$SCRIPT.py \
            --log_to_tensorboard True \
            --device cuda \
            --dtype float32 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 10000 \
            --batch_size 512 \
            --micro_batch_size 4 \
            --optimizer adamw \
            --weight_decay 0.01 \
            --beta1 0.9 \
            --beta2 0.95 \
            --eps 1e-6 \
            --lr $lr \
            --warmup_proportion -1 \
            --warmup_steps 2000 \
            --scheduler cosine_warmup \
            --gradient_clipping 1.0 \
            --grad_stats True \
            --prior_config configs.prior_config \
            --prior_anneal_proportion 0.0 \
            --prior_type mix_prior \
            --prior_device cpu \
            --batch_size_per_gp 4 \
            --min_features 2 \
            --max_features $max_features \
            --max_classes $max_classes \
            --replay_small True \
            --min_seq_len 500 \
            --max_seq_len 1000 \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 4 \
            --col_num_inds 128 \
            --freeze_col False \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --freeze_row False \
            --cluster_num_blocks 6 \
            --cluster_nhead 4 \
            --cluster_use_rope_cross_attn False \
            --cluster_rope_base 100000 \
            --cluster_use_representation_self_att False \
            --freeze_cluster False \
            --ff_factor 2 \
            --norm_first True \
            --checkpoint_path "path/to/tabicl/tabicl.ckpt" \
            --only_load_model True \
            --checkpoint_dir "$EXPERIMENTS_BASE/$NAME" \
            --save_temp_every 100 \
            --save_perm_every 5000 \
            --max_checkpoints 3 \
            --prob_method softmax \
            --k_decision_weight 1.0 \
            >> "$LOG_FILE" 2>&1 &
echo $! >> "$LOG_FILE"