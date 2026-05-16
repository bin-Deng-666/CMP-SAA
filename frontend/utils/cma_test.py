"""
cma_image.py 测试脚本
用于测试跨模态辅助攻击方法的图像级别实现

srun --job-name=cma_test --partition=cluster-1 --cpus-per-task=1 --mem=40G --gres=gpu:a800:1 python frontend/utils/cma_test.py
"""

import os
import sys
import time
import torch
from PIL import Image

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from frontend.utils.cma_image import generate_adversarial_image_cma


def main():
    print("=" * 60)
    print("跨模态辅助攻击 (CMA) - 图像级别测试")
    print("=" * 60)
    
    # 加载测试图像
    test_image_path = os.path.join(project_root, "data", "val2014", "COCO_val2014_000000000294.jpg")
    print(f"测试图像: {test_image_path}")
    
    if not os.path.exists(test_image_path):
        print(f"错误: 测试图像不存在: {test_image_path}")
        return False
    
    image = Image.open(test_image_path).convert("RGB")
    print(f"图像尺寸: {image.size}")
    
    # 设置输出目录
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"输出目录: {output_dir}")
    
    # 测试参数配置
    print("\n" + "-" * 60)
    print("攻击参数配置:")
    print("-" * 60)
    config = {
        "image_id": "294",  # 图像ID（用于获取提示词）
        "model_name": "blip2",  # 模型名称
        "method": "embed_adv",  # 攻击方法
        "target_text": "Unknown",  # 目标攻击文本
        "adversarial_length": 16,  # 对抗文本后缀长度
        "prompt_num": 50,  # 用于训练的文本提示数量
        "iterations": 100,  # 迭代次数（减少以便快速测试）
        "epsilon": 32/255,  # 扰动大小限制
        "alpha": 1/255,  # 学习率
        "device": 0 if torch.cuda.is_available() else -1  # GPU设备号
    }
    
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 进度回调
    def progress_callback(current, total, loss):
        print(f"  迭代 {current}/{total}, Loss: {loss:.4f}")
    
    # 运行攻击
    print("\n" + "=" * 60)
    print("开始生成对抗图像...")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        original, adversarial, perturbation = generate_adversarial_image_cma(
            image=image,
            progress_callback=progress_callback,
            **config
        )
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print(f"攻击完成! 总耗时: {elapsed_time:.2f} 秒")
        print("=" * 60)
        
        # 保存结果
        original_path = os.path.join(output_dir, "cma_test_original.png")
        adversarial_path = os.path.join(output_dir, "cma_test_adversarial.png")
        perturbation_path = os.path.join(output_dir, "cma_test_perturbation.png")
        
        original.save(original_path)
        adversarial.save(adversarial_path)
        perturbation.save(perturbation_path)
        
        print(f"\n结果已保存:")
        print(f"  - 原始图像: {original_path}")
        print(f"  - 对抗图像: {adversarial_path}")
        print(f"  - 扰动可视化: {perturbation_path}")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)