import os
import argparse
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
from typing import List, Dict
from torch import nn
from tqdm import tqdm

# 导入特征提取器
from feature_extractors import (
    ClipB16FeatureExtractor,
    ClipL336FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
)

# 骨干网络名称到模型类的映射
BACKBONE_MAP = {
    "L336": ClipL336FeatureExtractor,
    "B16": ClipB16FeatureExtractor,
    "B32": ClipB32FeatureExtractor,
    "Laion": ClipLaionFeatureExtractor,
}

from utils.attack_tool import (
    load_dataset,
    get_subset
)
from utils.crop_images import random_crop, yolo_crop, center_crop, grid_crop, edge_crop

class AttackConfig:
    """对抗攻击配置类，集中管理攻击参数"""
    def __init__(
        self,
        extractors,
        datasets,
        fraction: float = 0.05,
        iters: int = 50,
        alpha: float = 1,
        epsilon: float = 32,
        device: str = "cuda",
        crop_types: List[str] = ["random"],
        num_crops: int = 10
    ):
        """初始化攻击配置

        Args:
            extractors: 特征提取器
            datasets: 数据集元组
            fraction: 数据比例
            iters: 优化迭代次数
            alpha: 学习率
            epsilon: 最大扰动范围
            device: 运行设备
            crop_types: 裁剪类型列表，可选值: random, yolo, center, grid, edge, saliency
            num_crops: 裁剪数量
        """
        self.extractors = extractors
        self.datasets = datasets
        self.fraction = fraction
        self.iters = iters
        self.alpha = alpha
        self.epsilon = epsilon
        self.device = device
        self.crop_types = crop_types
        self.num_crops = num_crops

        # 输出目录将在 generate_adversarial_perturbation 函数中创建

def load_feature_extractors(backbones: List[str], device: str = "cuda") -> List[nn.Module]:
    """加载多个特征提取器

    Args:
        backbones: 骨干网络名称列表
        device: 运行设备

    Returns:
        特征提取器列表
    """
    extractors = []
    for backbone_name in backbones:
        if backbone_name not in BACKBONE_MAP:
            raise ValueError(f"未知骨干网络: {backbone_name}")
        model_class = BACKBONE_MAP[backbone_name]
        model = model_class().eval().to(device).requires_grad_(False)
        extractors.append(model)
    return extractors

def get_embeddings(models: List[nn.Module], images: torch.Tensor) -> Dict[int, torch.Tensor]:
    """获取图像在所有模型中的嵌入向量

    Args:
        models: 模型列表
        images: 输入图像张量

    Returns:
        各模型的嵌入向量字典
    """
    features = {}
    for i, model in enumerate(models):
        features[i] = model(images).squeeze()
    return features

def compute_distance(
    features1: Dict[int, torch.Tensor],
    features2: Dict[int, torch.Tensor]
) -> torch.Tensor:
    """计算两组特征之间的平均点积相似度"""
    similarity = 0
    for i in features1.keys():
        feat1 = features1[i]
        feat2 = features2[i]
        
        # 计算点积并取平均值
        if feat1.dim() == 1:
            # 一维张量，直接计算点积
            similarity += torch.sum(feat1 * feat2)
        else:
            # 多维张量，在特征维度求和
            similarity += torch.mean(torch.sum(feat1 * feat2, dim=1))

    return similarity / len(features1)

def load_image_collection(image_tensor, crop_types: List[str] = ["random"], num_crops: int = 10) -> List[Dict[str, torch.Tensor]]:
    """加载图像集合，返回裁剪后的图像和区域信息

    Args:
        image_tensor: 原始图像张量
        crop_types: 裁剪类型列表，可选值: random, yolo, center, grid, edge, saliency
        num_crops: 裁剪数量（控制张数）

    Returns:
        包含裁剪图像和区域信息的字典列表
        每个字典包含:
        - "cropped_image": 裁剪后的图像张量
        - "region": 区域信息 (x1, y1, x2, y2)
        - "original_size": 原始尺寸 (height, width)
        - （当crop_type=yolo时）"class_id": 物体类别ID
        - （当crop_type=yolo时）"confidence": 检测置信度
    """
    # 获取图像尺寸
    _, _, height, width = image_tensor.shape

    crops = []
    original_image_info = {
        "cropped_image": image_tensor.clone(),
        "region": (0, 0, width, height),
        "original_size": (height, width),
        "crop_type": "original"
    }
    crops.append(original_image_info)

    for crop_type in crop_types:
        if crop_type == "random":
            # 随机裁剪
            random_crops = random_crop(
                image_tensor,
                num_crops=num_crops,
                min_size=min(height, width) // 4,
                max_size=min(height, width)
            )
            crops.extend(random_crops)
        elif crop_type == "yolo":
            # YOLO物体裁剪
            yolo_crops = yolo_crop(
                image_tensor,
                top_k=num_crops
            )
            crops.extend(yolo_crops)
        elif crop_type == "center":
            # 中心裁剪
            center_crop_result = center_crop(
                image_tensor,
                size=128
            )
            crops.append(center_crop_result)
        elif crop_type == "grid":
            # 网格裁剪
            grid_crops = grid_crop(
                image_tensor,
                grid_size=(3, 3)
            )
            crops.extend(grid_crops)
        elif crop_type == "edge":
            # 边缘裁剪
            edge_crops = edge_crop(
                image_tensor,
                num_crops=num_crops,
                size=128
            )
            crops.extend(edge_crops)
        else:
            raise ValueError(f"不支持的裁剪类型: {crop_type}")
    
    return crops

def generate_adversarial_perturbation(
    image_tensor,
    image_collection: List[Dict[str, torch.Tensor]],
    extractors=None,
    iters: int = 50,
    alpha: float = 0.01,
    epsilon: float = 8.0,
    device: str = "cuda"
):
    """生成对抗扰动，最大化图像集合与原始图像的嵌入向量距离

    Args:
        image_tensor: 原始图像张量
        image_collection: 目标图像集合，包含裁剪图像和区域信息
        extractors: 特征提取器
        iters: 优化迭代次数
        alpha: 学习率
        epsilon: 最大扰动范围
        device: 运行设备

    Returns:
        tuple: (原始图像, 对抗图像, 扰动)
    """
    # 将图像移至设备
    original_image = image_tensor.to(device)

    # 初始化整个图像的扰动
    delta = torch.zeros_like(original_image, requires_grad=True, device=device)

    # 优化过程
    pbar = tqdm(range(iters), desc="Attack progress")
    for i in pbar:
        total_distance = 0

        # 创建当前对抗图像
        adv_image = original_image + delta

        # 处理每个image_collection中的图像
        for item in image_collection:
            region = item["region"]
            # 根据区域坐标裁剪对抗图像
            x1, y1, x2, y2 = region
            cropped_adv = adv_image[:, :, y1:y2, x1:x2]
            
            # 调整大小以适应模型输入
            resize_transform = transforms.Resize((224, 224))
            resized_cropped = resize_transform(cropped_adv)
            
            # 获取特征并计算距离
            with torch.no_grad():
                original_features = get_embeddings(extractors, original_image)
            perturbed_features = get_embeddings(extractors, resized_cropped)
            distance = compute_distance(perturbed_features, original_features)
            total_distance += distance

        # 计算平均距离
        avg_distance = total_distance / len(image_collection)

        # 梯度上升优化（最大化距离）
        if delta.grad is not None:
            delta.grad.zero_()  # 清零梯度

        loss = -avg_distance  # 负距离作为损失函数，梯度上升
        loss.backward()  # 反向传播计算梯度

        # 更新扰动并裁剪到合法范围
        with torch.no_grad():
            delta.data = torch.clamp(
                delta + alpha * torch.sign(delta.grad),
                min=-epsilon,
                max=epsilon
            )

        # 每10次迭代打印结果
        if (i + 1) % 10 == 0:
            print(f"\n迭代 {i+1}/{iters} 平均距离: {avg_distance.item():.4f}")

    # 创建最终对抗图像
    adv_image = original_image + delta
    # 裁剪到合法范围
    adv_image = torch.clamp(adv_image, min=0, max=255)

    return original_image, adv_image, delta


def print_config(config: AttackConfig) -> None:
    """输出攻击配置参数

    Args:
        config: 攻击配置对象
    """
    print(f"\n=== 攻击配置信息 ===")
    print(f"- 数据集比例: {config.fraction}")
    print(f"- 迭代次数: {config.iters}")
    print(f"- 学习率: {config.alpha}")
    print(f"- 最大扰动: {config.epsilon}")
    print(f"- 设备: {config.device}")
    print(f"- 裁剪类型: {', '.join(config.crop_types)}")
    print(f"- 裁剪数量: {config.num_crops}")
    print(f"- 骨干网络: {'B16, B32, L336, Laion' if config.extractors else '未知'}")
    print(f"- 训练集大小: {len(config.datasets[0])}")
    print(f"- 测试集大小: {len(config.datasets[1])}")


# 转换 PIL Image 为 PyTorch Tensor
def to_tensor(pic):
    """将 PIL Image 转换为 PyTorch 张量

    Args:
        pic: PIL Image 对象

    Returns:
        torch.Tensor: 转换后的张量
    """
    import numpy as np
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


def attack(config: AttackConfig) -> None:
    """主攻击函数，使用配置对象管理参数

    Args:
        config: 攻击配置对象
    """
    print(f"\n=== 开始执行对抗攻击 ===")
    # 输出配置参数
    print_config(config)

    # 加载数据集子集
    _, test_dataset = config.datasets
    test_dataset = get_subset(test_dataset, config.fraction)

    # 生成对抗扰动
    for i, item in enumerate(test_dataset):
        print(f"\n正在处理第 {i+1}/{len(test_dataset)} 个样本...")

        # 获取图像并转换为张量
        image = item["image"]
        image_tensor = to_tensor(image).unsqueeze(0).to(config.device)

        # 获取图像对应的集合
        image_collection = load_image_collection(image_tensor, config.crop_types, config.num_crops)

        # 生成对抗扰动
        original_image, adv_image, delta = generate_adversarial_perturbation(
            image_tensor=image_tensor,
            image_collection=image_collection,
            extractors=config.extractors,
            iters=config.iters,
            alpha=config.alpha,
            epsilon=config.epsilon,
            device=config.device
        )

        # 创建保存目录结构
        output_dir = os.path.join("adversarial_images", "maximize"+ "_"+ )
        os.makedirs(output_dir, exist_ok=True)

        # 为每个样本创建独立文件夹
        img_id = str(item["image_id"])
        img_dir = os.path.join(output_dir, img_id)
        os.makedirs(img_dir, exist_ok=True)

        # 将图像转换为PIL格式以便保存
        transform = transforms.Compose([
            transforms.Lambda(lambda x: (x.cpu().squeeze(0) / 255.0).clamp(0, 1)),
            transforms.ToPILImage()
        ])

        original_pil = transform(original_image)
        adversarial_pil = transform(adv_image)

        # 保存扰动可视化
        perturbation_np = delta.squeeze(0).detach().cpu().numpy()
        perturbation_np = np.transpose(perturbation_np, (1, 2, 0))  # CHW->HWC
        # 归一化到0-255范围以便可视化
        perturbation_np = ((perturbation_np - perturbation_np.min()) * 
                         (255/(perturbation_np.max()-perturbation_np.min()))).astype(np.uint8)
        pert_img = Image.fromarray(perturbation_np)

        # 保存所有文件
        original_pil.save(os.path.join(img_dir, "original.png"))
        adversarial_pil.save(os.path.join(img_dir, "adversarial.png"))
        pert_img.save(os.path.join(img_dir, "perturbation_vis.png"))  # 可视化版本
        torch.save(delta.detach().cpu(), os.path.join(img_dir, "perturbation.pt"))  # 原始张量

    print("\n=== 攻击完成 ===")

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='对抗攻击参数配置')

    parser.add_argument(
        '--backbones', '-b', type=str, nargs='+', default=["B16"],
        help='使用的骨干网络列表，可选值: B16, B32, L336, Laion (默认: B16 B32)'
    )
    parser.add_argument(
        '--fraction', '-f', type=float, default=1.0,
        help='数据集使用比例 (默认: 1.0)'
    )
    parser.add_argument(
        '--iters', '-i', type=int, default=50,
        help='优化迭代次数 (默认: 50)'
    )
    parser.add_argument(
        '--alpha', '-a', type=float, default=0.01,
        help='学习率 (默认: 0.01)'
    )
    parser.add_argument(
        '--epsilon', '-e', type=float, default=8.0,
        help='最大扰动范围 (默认: 8.0)'
    )
    parser.add_argument(
        '--output-dir', '-o', type=str, default="./output",
        help='对抗图像输出目录 (默认: ./output)'
    )
    parser.add_argument(
        '--device', '-d', type=str, default="cuda",
        help='运行设备 (默认: cuda)'
    )

    parser.add_argument(
        '--crop-types', '-c', type=str, nargs='+', default=["random"],
        help='裁剪类型列表，可选值: random, yolo, center, grid, edge, saliency (默认: random)'
    )

    parser.add_argument(
        '--num-crops', '-n', type=int, default=10,
        help='裁剪数量 (默认: 10)'
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 加载图像编码器和数据集
    extractors = load_feature_extractors(backbones=args.backbones, device=args.device)
    train_dataset, test_dataset = load_dataset()

    config = AttackConfig(
        extractors=extractors,
        datasets=(train_dataset, test_dataset),
        fraction=args.fraction,
        iters=args.iters,
        alpha=args.alpha,
        epsilon=args.epsilon,
        device=args.device,
        crop_types=args.crop_types,
        num_crops=args.num_crops
    )

    # 执行攻击
    attack(config)