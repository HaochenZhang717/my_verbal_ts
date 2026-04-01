#!/bin/bash

# =========================
# GPU
# =========================
export CUDA_VISIBLE_DEVICES=0

# =========================
# Paths（自己改）
# =========================
TRAIN_PATH="/playpen-shared/haochenz/LitsDatasets/128_len_ts/ETTh1/train_ts.npy"
VAL_PATH="/playpen-shared/haochenz/LitsDatasets/128_len_ts/ETTh1/valid_ts.npy"
SAVE_DIR="../ckpts_multiscale/ETTh1"

# =========================
# WandB
# =========================
export WANDB_PROJECT=multiscale_vae
export WANDB_NAME=ETTh1

# =========================
# Run
# =========================
python train_multiscale_vae.py \
    --train_path ${TRAIN_PATH} \
    --val_path ${VAL_PATH} \
    \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-4 \
    \
    --z_channels 32 \
    --latent_channels 32 \
    --ch 64 \
    --dropout 0.0 \
    \
    --ch_mult 1 2 2 4 \
    \
    --decomposition_width 64 \
    --frequency_groups 4 \
    --moving_avg_kernel_sizes 5 7 \
    \
    --lambda_recon_low 1.0 \
    --lambda_recon_mid 1.0 \
    --lambda_recon_high 1.0 \
    --lambda_recon_total 1.0 \
    \
    --lambda_kl_low 1e-3 \
    --lambda_kl_mid 5e-4 \
    --lambda_kl_high 1e-4 \
    \
    --save_dir ${SAVE_DIR} \
    --device cuda \
    \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_name ${WANDB_NAME}