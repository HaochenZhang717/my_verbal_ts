#!/bin/bash

# =========================
# CONFIG
# =========================
export CUDA_VISIBLE_DEVICES=5

CAPS_PATH="/playpen-shared/haochenz/LitsDatasets/128_len_ts/synthetic_u"
SAVE_ROOT="/playpen-shared/haochenz/LitsDatasets/128_len_ts/synthetic_u"

SCRIPT_PATH="precompute_long_clip_embeds.py"

BATCH_SIZE=512
DEVICE="cuda"

mkdir -p $SAVE_ROOT

echo "Start precomputing LongClip embeddings..."

# =========================
# TRAIN
# =========================
python $SCRIPT_PATH \
    --caps_path $CAPS_PATH \
    --save_path $SAVE_ROOT/train_embeds_long_clip_seq_0324.pt \
    --split train \
    --batch_size $BATCH_SIZE \
    --device $DEVICE

# =========================
# VAL
# =========================
python $SCRIPT_PATH \
    --caps_path $CAPS_PATH \
    --save_path $SAVE_ROOT/valid_embeds_long_clip_seq_0324.pt \
    --split valid \
    --batch_size $BATCH_SIZE \
    --device $DEVICE

# =========================
# TEST
# =========================
python $SCRIPT_PATH \
    --caps_path $CAPS_PATH \
    --save_path $SAVE_ROOT/test_embeds_long_clip_seq_0324.pt \
    --split test \
    --batch_size $BATCH_SIZE \
    --device $DEVICE

echo "Done!"