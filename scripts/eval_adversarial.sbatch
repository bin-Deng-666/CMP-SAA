#!/bin/bash
#SBATCH --job-name=evaluate_job
#SBATCH --partition=cluster-1
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --gres=gpu:a800:1
#SBATCH --output=evaluation_results/maximize_Laion_random_yolo_center_edge.log

# 定义默认参数
MODEL_NAME="blip2"
METHOD="maximize_Laion_random_yolo_center_edge"
DEVICE=0
TARGET_TEXT="Unknown"
EVAL_BATCH_SIZE=4
MAX_GEN_LENGTH=50
NUM_BEAMS=5
LENGTH_PENALTY=1.0

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name|-m)
            MODEL_NAME="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --device|-d)
            DEVICE="$2"
            shift 2
            ;;
        --target_text|-t)
            TARGET_TEXT="$2"
            shift 2
            ;;
        --eval_batch_size|-b)
            EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --max_gen_length|-l)
            MAX_GEN_LENGTH="$2"
            shift 2
            ;;
        --num_beams|-n)
            NUM_BEAMS="$2"
            shift 2
            ;;
        --length_penalty|-lp)
            LENGTH_PENALTY="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 运行评估脚本
python test.py \
    --model_name "$MODEL_NAME" \
    --method "$METHOD" \
    --device "$DEVICE" \
    --target_text "$TARGET_TEXT" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --max_generation_length "$MAX_GEN_LENGTH" \
    --num_beams "$NUM_BEAMS" \
    --length_penalty "$LENGTH_PENALTY"

echo "评估任务已完成"
