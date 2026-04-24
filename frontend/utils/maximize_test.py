"""
maximize_image.py 单实验测试脚本
使用完整参数配置进行一次攻击测试
"""

"""
srun --job-name=maximize_test --partition=cluster-1 --cpus-per-task=1 --mem=40G --gres=gpu:a800:1 python frontend/utils/maximize_test.py
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

from frontend.utils.maximize_image import generate_adversarial_image


def main():
    print("=" * 60)
    print("多维度语义攻击 - 完整参数测试")
    print("=" * 60)
    
    # 加载测试图像
    test_image_path = os.path.join(project_root, "data", "val2014", "COCO_val2014_000000000042.jpg")
    print(f"测试图像: {test_image_path}")
    
    image = Image.open(test_image_path).convert("RGB")
    print(f"图像尺寸: {image.size}")
    
    # 设置输出目录
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"输出目录: {output_dir}")
    
    # 完整参数配置
    print("\n" + "-" * 60)
    print("攻击参数配置:")
    print("-" * 60)
    config = {
        "backbones": ["B16"],  # 全部骨干网络
        "crop_types": ["random", "yolo", "center", "grid", "edge"],  # 全部裁剪类型
        "num_crops": 10,  # 裁剪数量
        "iterations": 100,  # 迭代次数
        "alpha": 1.0,  # 学习率
        "epsilon": 16.0,  # 扰动上限
        "min_crop_ratio": 0.5,  # 最小裁剪比例
        "yolo_confidence": 0.5,  # YOLO置信度
        "yolo_min_area_ratio": 0.05,  # YOLO最小面积比例
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 进度回调
    def progress_callback(current, total, loss):
        print(f"  迭代 {current}/{total}, 距离: {loss:.4f}")
    
    # 运行攻击
    print("\n" + "=" * 60)
    print("开始生成对抗图像...")
    print("=" * 60)
    
    start_time = time.time()
    
    original, adversarial, perturbation = generate_adversarial_image(
        image=image,
        progress_callback=progress_callback,
        **config
    )
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print(f"攻击完成! 总耗时: {elapsed_time:.2f} 秒")
    print("=" * 60)
    
    # 保存结果
    original_path = os.path.join(output_dir, "test_original.png")
    adversarial_path = os.path.join(output_dir, "test_adversarial.png")
    perturbation_path = os.path.join(output_dir, "test_perturbation.png")
    
    original.save(original_path)
    adversarial.save(adversarial_path)
    perturbation.save(perturbation_path)
    
    print(f"\n结果已保存:")
    print(f"  - 原始图像: {original_path}")
    print(f"  - 对抗图像: {adversarial_path}")
    print(f"  - 扰动可视化: {perturbation_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()