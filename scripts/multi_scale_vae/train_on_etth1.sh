#!/bin/bash

# =========================
# GPU
# =========================
export CUDA_VISIBLE_DEVICES=0

# =========================
# Paths
# =========================
TRAIN_PATH="/playpen-shared/haochenz/LitsDatasets/128_len_ts/ETTh1/train_ts.npy"
VAL_PATH="/playpen-shared/haochenz/LitsDatasets/128_len_ts/ETTh1/valid_ts.npy"
SAVE_DIR="../ckpts_multiscale/ETTh1"

# =========================
# WandB
# =========================
WANDB_PROJECT="multiscale_vae"
WANDB_NAME="ETTh1"

# =========================
# Training params
# =========================
BATCH_SIZE=64
EPOCHS=100
LR=1e-4

# =========================
# Model params
# =========================
Z_CHANNELS=32
LATENT_CHANNELS=32
CH=64
DROPOUT=0.0

CH_MULT="1 2 2 4"

# =========================
# Decomposition params
# =========================
DECOMP_WIDTH=128
FREQ_GROUPS=4
MA_KERNELS="5 7 9 11"

# =========================
# Loss weights
# =========================
LAMBDA_RECON_LOW=1.0
LAMBDA_RECON_MID=1.0
LAMBDA_RECON_HIGH=1.0
LAMBDA_RECON_TOTAL=1.0

LAMBDA_KL_LOW=1e-3
LAMBDA_KL_MID=5e-4
LAMBDA_KL_HIGH=1e-4

# =========================
# Common args (🔥关键)
# =========================
COMMON_ARGS="\
--train_path ${TRAIN_PATH} \
--val_path ${VAL_PATH} \
--batch_size ${BATCH_SIZE} \
--epochs ${EPOCHS} \
--lr ${LR} \
--z_channels ${Z_CHANNELS} \
--latent_channels ${LATENT_CHANNELS} \
--ch ${CH} \
--dropout ${DROPOUT} \
--ch_mult ${CH_MULT} \
--decomposition_width ${DECOMP_WIDTH} \
--frequency_groups ${FREQ_GROUPS} \
--moving_avg_kernel_sizes ${MA_KERNELS} \
--lambda_recon_low ${LAMBDA_RECON_LOW} \
--lambda_recon_mid ${LAMBDA_RECON_MID} \
--lambda_recon_high ${LAMBDA_RECON_HIGH} \
--lambda_recon_total ${LAMBDA_RECON_TOTAL} \
--lambda_kl_low ${LAMBDA_KL_LOW} \
--lambda_kl_mid ${LAMBDA_KL_MID} \
--lambda_kl_high ${LAMBDA_KL_HIGH} \
--save_dir ${SAVE_DIR} \
--device cuda \
--wandb_project ${WANDB_PROJECT} \
--wandb_name ${WANDB_NAME} \
"

# =========================
# Train
# =========================
echo "🚀 Starting training..."
#python train_multiscale_vae.py ${COMMON_ARGS}

# =========================
# Eval
# =========================
echo "📊 Running evaluation..."
python train_multiscale_vae.py \
    ${COMMON_ARGS} \
    --eval_only \
    --num_plot_samples 10