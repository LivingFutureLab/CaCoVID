export HF_ENDPOINT=https://hf-mirror.com

MAX_FRAMES=${1:-"32"}
MIN_PIXELS=${2:-"128"}
MAX_PIXELS=${3:-"256"}
WORLD_SIZE=${4:-"1"}
RANDOM_COMPRESSION=${5:-"0"}
TOKEN_SELECT_STRATEGY=${6:-"token"}
TOPK=${7:-"0.10"} # 0.10, 0.15, 0.20, 0.25
IMP_RATIO=${8:-"0.5"}
MAX_SAMPLE_RATIO_FOR_EACH_FRAME=${9:-"0.15"} # NOTE: 1.5*TOPK
TASK_NAME=${10:-"mlvu_test"} # longvideobench_val_v,mlvu_test,mlvu_dev,videomme
PRETRAINED=${11:-"/home/mayinchao.myc/projects/CaCoVID_release/checkpoints/cacovid_llava_onevision_7b_prev"}

model_args="pretrained=${PRETRAINED},\
attn_implementation=flash_attention_2,\
max_frames_num=${MAX_FRAMES},\
random_compression=${RANDOM_COMPRESSION},\
max_pixels=$(($MAX_PIXELS * 28 * 28)),\
min_pixels=$(($MIN_PIXELS * 28 * 28)),\
topk=$TOPK,\
imp_ratio=$IMP_RATIO,\
max_sample_ratio_for_each_frame=$MAX_SAMPLE_RATIO_FOR_EACH_FRAME,\
token_select_strategy=$TOKEN_SELECT_STRATEGY"

output_path=$PRETRAINED/logs/logs_${RANDOM_COMPRESSION}_${TOPK}_${MAX_SAMPLE_RATIO_FOR_EACH_FRAME}_${TOKEN_SELECT_STRATEGY}_${IMP_RATIO}${CONFIG_NAME:41}_${CKPT}_$(date +"%Y-%m-%d-%H-%M-%S")
options="--model=llava_hf \
--model_args=${model_args} \
--tasks=${TASK_NAME} \
--batch_size=1 \
--log_samples \
--log_samples_suffix=qwen2_5_vl_exp \
--output_path=${output_path}"

export MASTER_ADDR=localhost
export MASTER_PORT=$(shuf -i 1000-65500 -n 1)
python -m accelerate.commands.launch \
    --num_processes $WORLD_SIZE \
    --main_process_port ${MASTER_PORT} \
    -m lmms_eval \
    $options \
    --verbosity=DEBUG