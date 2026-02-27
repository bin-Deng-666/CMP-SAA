#!/bin/bash

# SLURM批处理作业配置
#SBATCH --job-name=maximize_attack
#SBATCH --partition=cluster-1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --gres=gpu:a800:1
#SBATCH --output=logs/maximize_attack_%j.log

# 设置错误时退出
set -e

# 定义默认参数
initialize_default_parameters() {
    # 模型参数
    BACKBONES=("B16")
    
    # 攻击参数
    FRACTION=1.0
    ITERS=50
    ALPHA=0.01
    EPSILON=8.0
    OUTPUT_DIR="./output"
    
    # 硬件配置
    DEVICE="cuda"
}

# 解析命令行参数
parse_command_line_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --backbones|-b)
                BACKBONES=($2)
                shift 2
                ;;
            --fraction|-f)
                FRACTION="$2"
                shift 2
                ;;
            --iters|-i)
                ITERS="$2"
                shift 2
                ;;
            --alpha|-a)
                ALPHA="$2"
                shift 2
                ;;
            --epsilon|-e)
                EPSILON="$2"
                shift 2
                ;;
            --output-dir|-o)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --device|-d)
                DEVICE="$2"
                shift 2
                ;;
            *)
                echo "未知参数: $1"
                exit 1
                ;;
        esac
    done
}

# 显示配置信息
display_configuration() {
    echo "=== 最大化攻击配置信息 ==="
    echo "任务ID: $SLURM_JOB_ID"
    echo "骨干网络: ${BACKBONES[*]}"
    echo "数据集比例: $FRACTION"
    echo "迭代次数: $ITERS"
    echo "学习率: $ALPHA"
    echo "最大扰动: $EPSILON"
    echo "输出目录: $OUTPUT_DIR"
    echo "设备: $DEVICE"
    echo "=========================="
}

# 执行最大化攻击任务
execute_attack() {
    echo "开始执行最大化攻击任务..."
    python maximize.py \
        --backbones "${BACKBONES[@]}" \
        --fraction "$FRACTION" \
        --iters "$ITERS" \
        --alpha "$ALPHA" \
        --epsilon "$EPSILON" \
        --output-dir "$OUTPUT_DIR" \
        --device "$DEVICE"
    
    echo "最大化攻击任务已完成"
}

# 主函数
main() {
    # 确保日志目录存在
    mkdir -p logs
    
    # 初始化默认参数
    initialize_default_parameters
    
    # 解析命令行参数
    parse_command_line_arguments "$@"
    
    # 显示配置信息
    display_configuration
    
    # 执行攻击任务
    execute_attack
}

# 调用主函数
main "$@"