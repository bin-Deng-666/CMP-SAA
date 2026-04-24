"""
多维度语义攻击方法 - 图像级别实现
从 maximize.py 提取并适配为单图像调用接口
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple, Union
from torch import nn
import torchvision.transforms as transforms
from dataclasses import dataclass

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入特征提取器
from feature_extractors import (
    ClipB16FeatureExtractor,
    ClipL336FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
)

# 导入裁剪工具
from utils.crop_images import random_crop, yolo_crop, center_crop, grid_crop, edge_crop

# 骨干网络名称到模型类的映射
BACKBONE_MAP = {
    "L336": ClipL336FeatureExtractor,
    "B16": ClipB16FeatureExtractor,
    "B32": ClipB32FeatureExtractor,
    "Laion": ClipLaionFeatureExtractor,
}


@dataclass
class MaximizeAttackConfig:
    """多维度语义攻击配置类
    
    参考 maximize.py 中的 AttackConfig 设计
    """
    backbones: List[str] = None
    crop_types: List[str] = None
    num_crops: int = 6
    iterations: int = 80
    alpha: float = 1.0
    epsilon: float = 16.0
    min_crop_ratio: float = 0.5
    yolo_confidence: float = 0.5
    yolo_min_area_ratio: float = 0.05
    device: str = "cuda"
    
    def __post_init__(self):
        if self.backbones is None:
            self.backbones = ["B16"]
        if self.crop_types is None:
            self.crop_types = ["random", "yolo"]


def pil_to_tensor(pic: Image.Image) -> torch.Tensor:
    """将 PIL Image 转换为 PyTorch 张量
    
    Args:
        pic: PIL Image 对象
    
    Returns:
        torch.Tensor: 转换后的张量，shape为 [1, 3, H, W]，值范围 [0, 255]
    """
    img = torch.from_numpy(np.array(pic, np.uint8, copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    # 转换为 float 并添加 batch 维度
    return img.unsqueeze(0).float()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """将 PyTorch 张量转换为 PIL Image
    
    Args:
        tensor: 输入张量，shape为 [1, 3, H, W] 或 [3, H, W]，值范围 [0, 255]
    
    Returns:
        PIL.Image: 转换后的图像
    """
    # 移除 batch 维度
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # 裁剪到合法范围并转换为 numpy（需要先 detach 去除梯度）
    img_np = tensor.detach().cpu().clamp(0, 255).numpy().astype(np.uint8)
    # 转换维度从 CHW 到 HWC
    img_np = np.transpose(img_np, (1, 2, 0))
    
    # 处理不同的通道数
    if img_np.shape[2] == 1:
        img_np = img_np.squeeze(2)
    
    return Image.fromarray(img_np)


def load_feature_extractors(backbones: List[str], device: str = "cuda") -> List[nn.Module]:
    """加载多个特征提取器
    
    Args:
        backbones: 骨干网络名称列表，可选值: B16, B32, L336, Laion
        device: 运行设备
    
    Returns:
        特征提取器列表
    """
    extractors = []
    for backbone_name in backbones:
        if backbone_name not in BACKBONE_MAP:
            raise ValueError(f"未知骨干网络: {backbone_name}，可选值: {list(BACKBONE_MAP.keys())}")
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
    """计算两组特征之间的平均点积相似度
    
    Args:
        features1: 第一组特征
        features2: 第二组特征
    
    Returns:
        相似度张量
    """
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


def load_image_collection(
    image_tensor: torch.Tensor,
    crop_types: List[str] = ["random"],
    num_crops: int = 10,
    min_crop_ratio: float = 0.5,
    yolo_confidence: float = 0.5,
    yolo_min_area_ratio: float = 0.05
) -> List[Dict[str, torch.Tensor]]:
    """加载图像集合，返回裁剪后的图像和区域信息
    
    Args:
        image_tensor: 原始图像张量，shape为 [1, 3, H, W]
        crop_types: 裁剪类型列表，可选值: random, yolo, center, grid, edge
        num_crops: 裁剪数量（控制张数）
        min_crop_ratio: random裁剪的最小尺寸比例（相对于图像短边）
        yolo_confidence: YOLO检测的最小置信度
        yolo_min_area_ratio: YOLO裁剪的最小面积比例（相对于图像总面积）
    
    Returns:
        包含裁剪图像和区域信息的字典列表
    """
    # 获取图像尺寸
    _, _, height, width = image_tensor.shape
    
    crops = []
    
    # 添加原始图像
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
            min_size = int(min(height, width) * min_crop_ratio)
            max_size = min(height, width)
            random_crops = random_crop(
                image_tensor,
                num_crops=num_crops,
                min_size=min_size,
                max_size=max_size
            )
            for crop in random_crops:
                crop["crop_type"] = "random"
            crops.extend(random_crops)
            
        elif crop_type == "yolo":
            # YOLO物体裁剪
            yolo_crops = yolo_crop(
                image_tensor,
                min_confidence=yolo_confidence,
                top_k=num_crops
            )
            # 过滤掉太小的检测
            total_area = height * width
            min_area = total_area * yolo_min_area_ratio
            filtered_crops = []
            for crop in yolo_crops:
                h, w = crop["original_size"]
                if h * w >= min_area:
                    crop["crop_type"] = "yolo"
                    filtered_crops.append(crop)
            crops.extend(filtered_crops)
            
        elif crop_type == "center":
            # 中心裁剪
            size = min(height, width) // 2
            center_crop_result = center_crop(image_tensor, size=size)
            center_crop_result["crop_type"] = "center"
            crops.append(center_crop_result)
            
        elif crop_type == "grid":
            # 网格裁剪
            grid_crops = grid_crop(image_tensor, grid_size=(3, 3))
            for crop in grid_crops:
                crop["crop_type"] = "grid"
            crops.extend(grid_crops)
            
        elif crop_type == "edge":
            # 边缘裁剪
            size = min(height, width) // 4
            edge_crops = edge_crop(image_tensor, num_crops=min(4, num_crops), size=size)
            for crop in edge_crops:
                crop["crop_type"] = "edge"
            crops.extend(edge_crops)
            
        else:
            raise ValueError(f"不支持的裁剪类型: {crop_type}")
    
    return crops


def generate_adversarial_image(
    image: Image.Image,
    backbones: List[str] = ["B16"],
    crop_types: List[str] = ["random"],
    num_crops: int = 6,
    iterations: int = 80,
    alpha: float = 1.0,
    epsilon: float = 16.0,
    min_crop_ratio: float = 0.5,
    yolo_confidence: float = 0.5,
    yolo_min_area_ratio: float = 0.05,
    device: str = "cuda",
    progress_callback: Optional[callable] = None
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """生成对抗图像 - 图像级别接口（使用独立参数）
    
    使用多维度语义最大化方法生成对抗样本，通过最大化原始图像与其裁剪版本
    之间的特征距离来生成对抗扰动。
    
    Args:
        image: 原始 PIL 图像
        backbones: 特征提取器列表，可选值: B16, B32, L336, Laion
        crop_types: 裁剪类型列表，可选值: random, yolo, center, grid, edge
        num_crops: 每种裁剪类型的裁剪数量
        iterations: 优化迭代次数
        alpha: 学习率（扰动更新步长）
        epsilon: 最大扰动范围（像素级，0-255范围）
        min_crop_ratio: random裁剪的最小尺寸比例
        yolo_confidence: YOLO检测的最小置信度
        yolo_min_area_ratio: YOLO裁剪的最小面积比例
        device: 运行设备，"cuda" 或 "cpu"
        progress_callback: 进度回调函数，接收 (current_iter, total_iters, current_loss) 参数
    
    Returns:
        Tuple[Image.Image, Image.Image, Image.Image]: 
            (原始图像, 对抗图像, 扰动可视化图像)
    """
    config = MaximizeAttackConfig(
        backbones=backbones,
        crop_types=crop_types,
        num_crops=num_crops,
        iterations=iterations,
        alpha=alpha,
        epsilon=epsilon,
        min_crop_ratio=min_crop_ratio,
        yolo_confidence=yolo_confidence,
        yolo_min_area_ratio=yolo_min_area_ratio,
        device=device
    )
    return generate_adversarial_image_with_config(image, config, progress_callback)


def generate_adversarial_image_with_config(
    image: Image.Image,
    config: MaximizeAttackConfig,
    progress_callback: Optional[callable] = None
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """生成对抗图像 - 使用配置对象
    
    类似 maximize.py 中的 attack(config) 函数风格
    
    Args:
        image: 原始 PIL 图像
        config: 攻击配置对象
        progress_callback: 进度回调函数
    
    Returns:
        Tuple[Image.Image, Image.Image, Image.Image]: 
            (原始图像, 对抗图像, 扰动可视化图像)
    """
    # 检查设备可用性
    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，切换到 CPU")
        device = "cpu"
    
    # 加载特征提取器
    extractors = load_feature_extractors(config.backbones, device)
    
    # 将 PIL 图像转换为张量
    image_tensor = pil_to_tensor(image).to(device)
    
    # 获取图像集合（原始图像 + 各种裁剪）
    image_collection = load_image_collection(
        image_tensor,
        crop_types=config.crop_types,
        num_crops=config.num_crops,
        min_crop_ratio=config.min_crop_ratio,
        yolo_confidence=config.yolo_confidence,
        yolo_min_area_ratio=config.yolo_min_area_ratio
    )
    
    # 生成对抗扰动
    original_image, adv_image, delta = _generate_adversarial_perturbation(
        image_tensor=image_tensor,
        image_collection=image_collection,
        extractors=extractors,
        iters=config.iterations,
        alpha=config.alpha,
        epsilon=config.epsilon,
        device=device,
        progress_callback=progress_callback
    )
    
    # 转换为 PIL Image
    original_pil = tensor_to_pil(original_image)
    adv_pil = tensor_to_pil(adv_image)
    
    # 生成扰动可视化图像
    perturbation_pil = _create_perturbation_visualization(delta)
    
    # 清理 GPU 内存
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return original_pil, adv_pil, perturbation_pil


def _create_perturbation_visualization(delta: torch.Tensor) -> Image.Image:
    """创建扰动可视化图像
    
    Args:
        delta: 扰动张量
    
    Returns:
        PIL.Image: 可视化后的扰动图像
    """
    perturbation_np = delta.squeeze(0).detach().cpu().numpy()
    perturbation_np = np.transpose(perturbation_np, (1, 2, 0))  # CHW -> HWC
    
    # 归一化到 0-255 范围以便可视化
    p_min, p_max = perturbation_np.min(), perturbation_np.max()
    if p_max > p_min:
        perturbation_np = ((perturbation_np - p_min) * (255 / (p_max - p_min))).astype(np.uint8)
    else:
        perturbation_np = np.zeros_like(perturbation_np, dtype=np.uint8)
    
    return Image.fromarray(perturbation_np)


def _generate_adversarial_perturbation(
    image_tensor: torch.Tensor,
    image_collection: List[Dict[str, torch.Tensor]],
    extractors: List[nn.Module],
    iters: int = 50,
    alpha: float = 1.0,
    epsilon: float = 16.0,
    device: str = "cuda",
    progress_callback: Optional[callable] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """内部函数：生成对抗扰动
    
    最大化图像集合与原始图像的嵌入向量距离
    
    Args:
        image_tensor: 原始图像张量
        image_collection: 目标图像集合
        extractors: 特征提取器列表
        iters: 优化迭代次数
        alpha: 学习率
        epsilon: 最大扰动范围
        device: 运行设备
        progress_callback: 进度回调函数
    
    Returns:
        tuple: (原始图像, 对抗图像, 扰动)
    """
    # 将图像移至设备
    original_image = image_tensor.to(device)
    
    # 初始化整个图像的扰动
    delta = torch.zeros_like(original_image, requires_grad=True, device=device)
    
    # 调整大小变换
    resize_transform = transforms.Resize((224, 224))
    
    # 预计算原始图像特征（只需要计算一次，不需要梯度）
    with torch.no_grad():
        original_features = get_embeddings(extractors, original_image)
    
    # 优化过程
    for i in range(iters):
        total_distance = 0
        
        # 创建当前对抗图像
        adv_image = original_image + delta
        
        # 处理每个 image_collection 中的图像
        for item in image_collection:
            region = item["region"]
            # 根据区域坐标裁剪对抗图像
            x1, y1, x2, y2 = region
            cropped_adv = adv_image[:, :, y1:y2, x1:x2]
            
            # 调整大小以适应模型输入
            resized_cropped = resize_transform(cropped_adv)
            
            # 获取对抗图像的特征并计算距离
            perturbed_features = get_embeddings(extractors, resized_cropped)
            distance = compute_distance(perturbed_features, original_features)
            total_distance += distance
        
        # 计算平均距离
        avg_distance = total_distance / len(image_collection)
        
        # 梯度上升优化（最大化距离）
        if delta.grad is not None:
            delta.grad.zero_()
        
        loss = -avg_distance  # 负距离作为损失函数，梯度上升
        loss.backward()  # 反向传播计算梯度
        
        # 更新扰动并裁剪到合法范围
        with torch.no_grad():
            delta.data = torch.clamp(
                delta + alpha * torch.sign(delta.grad),
                min=-epsilon,
                max=epsilon
            )
        
        # 调用进度回调
        if progress_callback is not None and (i + 1) % 5 == 0:
            progress_callback(i + 1, iters, avg_distance.item())
    
    # 创建最终对抗图像
    adv_image = original_image + delta
    # 裁剪到合法范围 [0, 255]
    adv_image = torch.clamp(adv_image, min=0, max=255)
    
    return original_image, adv_image, delta


# 便捷函数：使用默认参数快速生成对抗图像
def quick_attack(
    image: Image.Image,
    device: str = "cuda"
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """快速生成对抗图像（使用默认参数）
    
    Args:
        image: 原始 PIL 图像
        device: 运行设备
    
    Returns:
        Tuple[Image.Image, Image.Image, Image.Image]: 
            (原始图像, 对抗图像, 扰动可视化图像)
    """
    config = MaximizeAttackConfig(device=device)
    return generate_adversarial_image_with_config(image, config)


# 向后兼容的别名
attack = generate_adversarial_image_with_config
AttackConfig = MaximizeAttackConfig