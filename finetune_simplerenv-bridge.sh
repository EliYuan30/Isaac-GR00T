#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_NVLS_ENABLE=0

NUM_GPUS=4 MAX_STEPS=20000 GLOBAL_BATCH_SIZE=1024 SAVE_STEPS=1000 uv run bash examples/finetune.sh \
    --base-model-path /root/models--nvidia--GR00T-N1.7-3B \
    --dataset-path examples/SimplerEnv/bridge_orig_lerobot/ \
    --embodiment-tag SIMPLER_ENV_WIDOWX \
    --output-dir /root/VLA/Isaac-GR00T/outputs/bridge_finetune_qwen3vl2b \
    --state-dropout-prob 0.8 \
    -- \
    --vlm-path /root/models--Qwen--Qwen3-VL-2B-Instruct