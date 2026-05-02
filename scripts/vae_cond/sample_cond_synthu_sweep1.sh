export USE_CAUSAL=false
export SCHEDULER=cosine


LR_LIST=(1e-3)
BS_LIST=(128)

LAYERS=10
CHANNELS=128
NHEADS=8
DIFFUSION_EMBEDDING_DIM=128

CKPT_DIR="/playpen-shared/haochenz/my_verbal_ts/sweep/synth_u/vae_cond/0/ckpts"


for ckpt_path in ${CKPT_DIR}/model_*.pth
do
  ckpt_name=$(basename ${ckpt_path})
  ckpt_name_noext=${ckpt_name%.pth}
  echo "======================================"
  echo "Running checkpoint: ${ckpt_name}"
  echo "======================================"

  for LR in "${LR_LIST[@]}"
  do
    for BS in "${BS_LIST[@]}"
    do
      echo "Running lr=$LR bs=$BS"

      export WANDB_NAME="vae_cond"

      CUDA_VISIBLE_DEVICES=4 python run_vae_cond.py \
          --cond_modal vae_embed \
          --training_stage finetune \
          --save_folder ./sweep/synth_u/vae_cond \
          --model_diff_config_path configs/synth_u_vae_cond/diff/model_text2ts_dep.yaml \
          --model_cond_config_path configs/synth_u_vae_cond/cond/text_msmdiffmv.yaml \
          --train_config_path configs/synth_u_vae_cond/train.yaml \
          --evaluate_config_path configs/synth_u_vae_cond/evaluate.yaml \
          --data_folder /playpen-shared/haochenz/LitsDatasets/128_len_ts/synthetic_u \
          --clip_folder "" \
          --multipatch_num 3 \
          --L_patch_len 2 \
          --base_patch 4 \
          --epochs 2500 \
          --layers ${LAYERS} \
          --channels ${CHANNELS} \
          --nheads ${NHEADS} \
          --diffusion_embedding_dim ${DIFFUSION_EMBEDDING_DIM} \
          --batch_size ${BS} \
          --lr ${LR} \
          --clip_cache_path "" \
          --samples_name "real_text_samples_${ckpt_name_noext}.pt" \
          --model_ckpt_name ${ckpt_name} \
          --only_evaluate True
    done
  done
done