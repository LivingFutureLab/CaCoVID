#!/bin/bash
WORLD_SIZE=${1:-"1"}
MAX_SAMPLE_RATE=${2:-"0.02"}
COMPRESSION_POLICY=${3:-"CompressionPolicy"}
NUM_GENERATION=${4:-"32"}
LR_FOR_CP=${5:-"0.000001"}
KL=${6:-"0.0"}
MAX_FRAMES=${7:-'32'}
FRAME_RATIO=${8:-"0.125"}
GROUP_SIZE=${9:-'2.0'}
MAX_GROUP_PROB=${10:-'2.0'}
DATASET=${11:-"./data/Video-R1-llava_onevision_filtered.json"}
EPOCH=${12:-"1"}
VIDEO_ROOT=${13:-"/data/oss_bucket_0/mllm_dataset/public_datasets/Video-R1/"}
PRETRAIN_MODEL=${14:-"llava-hf/llava-onevision-qwen2-7b-ov-hf"}

FREEZE_POLICY_LAYERS="false"
ENTRY_FILE="grpo_video"

export DEBUG_MODE="false"
export LOG_PATH="src/r1-v/vllm_run_cacovid_llava_onevision.txt"

HF_DATASET=$DATASET
HF_DATASET_NAME=$(basename "$HF_DATASET")
HF_DATASET_NAME="${HF_DATASET_NAME%.*}"

OUTPUT_DIR="src/r1-v/log/CaCoVID_LLaVA-OneVision"
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi

DS_CONFIG="src/r1-v/local_scripts/zero3.json"

MAX_PIXELS=256
MIN_PIXELS=128

LR=$(awk -v ws="$WORLD_SIZE" 'BEGIN {printf "%.6f", 1e-6 * ws / 8}')
options="--use_vllm false \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${PRETRAIN_MODEL} \
    --dataset_name ${HF_DATASET} \
    --max_prompt_length 65536 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate $LR \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --logging_steps 1 \
    --bf16 true \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --min_pixels $(($MIN_PIXELS * 28 * 28)) \
    --max_pixels $(($MAX_PIXELS * 28 * 28)) \
    --max_num_images $MAX_FRAMES \
    --num_train_epochs $EPOCH \
    --video_root $VIDEO_ROOT \
    --reward_funcs accuracy \
    --apply_compression true \
    --video_selection_kl_weight $KL \
    --compression_policy $COMPRESSION_POLICY \
    --learning_rate_for_compression_policy $LR_FOR_CP \
    --learning_rate $LR_FOR_CP \
    --max_sample_rate $MAX_SAMPLE_RATE \
    --save_steps 100 \
    --save_only_model false \
    --temporal true \
    --len_control true \
    --report_to none \
    --beta 0.04 \
    --max_grad_norm 5 \
    --temperature 1.0 \
    --num_generations $NUM_GENERATION \
    --group_size $GROUP_SIZE \
    --max_group_prob $MAX_GROUP_PROB \
    --freeze_vision_modules true \
    --freeze_language_modules true \
    --freeze_policy_layers_modules $FREEZE_POLICY_LAYERS \
    --frame_ratio $FRAME_RATIO \
    --deepspeed ${DS_CONFIG} \
    --seed 42"

ENTRY_FILE=src/r1-v/src/open_r1/$ENTRY_FILE.py

torchrun --nproc_per_node=$WORLD_SIZE \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port=$((10000 + RANDOM % 90000)) \
    $ENTRY_FILE \
    $options