import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from ultralytics import YOLO
import cv2


def random_crop(
    image_tensor: torch.Tensor,
    num_crops: int = 10,
    min_size: int = 64,
    max_size: int = 128
) -> List[Dict[str, torch.Tensor]]:
    """随机裁剪图像，通过随机生成坐标来进行裁剪

    Args:
        image_tensor: 原始图像张量 (shape: [1, 3, H, W])
        num_crops: 裁剪数量（控制张数）
        min_size: 最小裁剪尺寸
        max_size: 最大裁剪尺寸

    Returns:
        包含裁剪图像和区域信息的字典列表
        每个字典包含:
        - "cropped_image": 裁剪后的图像张量
        - "region": 区域信息 (x1, y1, x2, y2)
        - "original_size": 原始尺寸 (height, width)
    """
    # 获取图像尺寸
    _, _, height, width = image_tensor.shape
    
    # 生成裁剪区域
    crops = []
    
    for i in range(num_crops):
        # 随机生成裁剪尺寸
        crop_width = np.random.randint(min_size, min(max_size, width) + 1)
        crop_height = np.random.randint(min_size, min(max_size, height) + 1)
        
        # 随机生成裁剪位置坐标
        x1 = np.random.randint(0, width - crop_width + 1)
        y1 = np.random.randint(0, height - crop_height + 1)
        
        # 计算裁剪区域坐标
        x2 = x1 + crop_width
        y2 = y1 + crop_height
        
        # 执行裁剪
        cropped_image = image_tensor[:, :, y1:y2, x1:x2].clone()
        
        # 构建返回字典
        crop_info = {
            "cropped_image": cropped_image,
            "region": (x1, y1, x2, y2),
            "original_size": (crop_height, crop_width)
        }
        
        # 添加到结果列表
        crops.append(crop_info)
    
    return crops


def center_crop(
    image_tensor: torch.Tensor,
    size: int = 128
) -> Dict[str, torch.Tensor]:
    """中心裁剪图像

    Args:
        image_tensor: 原始图像张量 (shape: [1, 3, H, W])
        size: 裁剪尺寸

    Returns:
        包含裁剪图像和区域信息的字典
    """
    _, _, height, width = image_tensor.shape
    
    # 计算裁剪位置
    x1 = (width - size) // 2
    y1 = (height - size) // 2
    x2 = x1 + size
    y2 = y1 + size
    
    # 执行裁剪
    cropped_image = image_tensor[:, :, y1:y2, x1:x2].clone()
    
    return {
        "cropped_image": cropped_image,
        "region": (x1, y1, x2, y2),
        "original_size": (size, size)
    }


def grid_crop(
    image_tensor: torch.Tensor,
    grid_size: Tuple[int, int] = (3, 3)
) -> List[Dict[str, torch.Tensor]]:
    """网格裁剪图像

    Args:
        image_tensor: 原始图像张量 (shape: [1, 3, H, W])
        grid_size: 网格大小 (行数, 列数)

    Returns:
        包含裁剪图像和区域信息的字典列表
    """
    _, _, height, width = image_tensor.shape
    rows, cols = grid_size
    
    # 计算每个网格的大小
    grid_height = height // rows
    grid_width = width // cols
    
    crops = []
    
    for i in range(rows):
        for j in range(cols):
            # 计算裁剪位置
            y1 = i * grid_height
            y2 = (i + 1) * grid_height
            x1 = j * grid_width
            x2 = (j + 1) * grid_width
            
            # 处理边界情况
            if i == rows - 1:
                y2 = height
            if j == cols - 1:
                x2 = width
            
            # 执行裁剪
            cropped_image = image_tensor[:, :, y1:y2, x1:x2].clone()
            
            # 构建返回字典
            crop_info = {
                "cropped_image": cropped_image,
                "region": (x1, y1, x2, y2),
                "original_size": (y2 - y1, x2 - x1)
            }
            
            # 添加到结果列表
            crops.append(crop_info)
    
    return crops


def edge_crop(
    image_tensor: torch.Tensor,
    num_crops: int = 4,
    size: int = 128
) -> List[Dict[str, torch.Tensor]]:
    """边缘裁剪图像

    Args:
        image_tensor: 原始图像张量 (shape: [1, 3, H, W])
        num_crops: 裁剪数量（每个边缘1个）
        size: 裁剪尺寸

    Returns:
        包含裁剪图像和区域信息的字典列表
    """
    _, _, height, width = image_tensor.shape
    crops = []
    
    # 边缘位置定义
    edge_positions = [
        ("top", 0, 0, width, size),
        ("bottom", 0, height - size, width, height),
        ("left", 0, 0, size, height),
        ("right", width - size, 0, width, height)
    ]
    
    for edge_name, x1, y1, x2, y2 in edge_positions[:num_crops]:
        # 执行裁剪
        cropped_image = image_tensor[:, :, y1:y2, x1:x2].clone()
        
        # 构建返回字典
        crop_info = {
            "cropped_image": cropped_image,
            "region": (x1, y1, x2, y2),
            "original_size": (y2 - y1, x2 - x1),
            "edge": edge_name
        }
        
        # 添加到结果列表
        crops.append(crop_info)
    
    return crops


def saliency_crop(
    image_tensor: torch.Tensor,
    num_crops: int = 3,
    size: int = 128
) -> List[Dict[str, torch.Tensor]]:
    """基于显著性的裁剪图像

    Args:
        image_tensor: 原始图像张量 (shape: [1, 3, H, W])
        num_crops: 裁剪数量
        size: 裁剪尺寸

    Returns:
        包含裁剪图像和区域信息的字典列表
    """
    _, _, height, width = image_tensor.shape
    crops = []
    
    # 简化实现：使用中心区域作为显著性区域
    # 实际应用中可以使用预训练的显著性检测模型
    center_x = width // 2
    center_y = height // 2
    
    for i in range(num_crops):
        # 围绕中心生成不同偏移的裁剪
        offset_x = np.random.randint(-size//4, size//4)
        offset_y = np.random.randint(-size//4, size//4)
        
        x1 = max(0, center_x - size//2 + offset_x)
        y1 = max(0, center_y - size//2 + offset_y)
        x2 = min(width, x1 + size)
        y2 = min(height, y1 + size)
        
        # 调整位置以确保尺寸一致
        if x2 - x1 < size:
            x1 = max(0, x2 - size)
        if y2 - y1 < size:
            y1 = max(0, y2 - size)
        
        # 执行裁剪
        cropped_image = image_tensor[:, :, y1:y2, x1:x2].clone()
        
        # 构建返回字典
        crop_info = {
            "cropped_image": cropped_image,
            "region": (x1, y1, x2, y2),
            "original_size": (y2 - y1, x2 - x1)
        }
        
        # 添加到结果列表
        crops.append(crop_info)
    
    return crops


def yolo_crop(
    image_tensor: torch.Tensor,
    model_path: str = "models/YOLO/yolov8m.pt",
    min_confidence: float = 0.5,
    min_width: int = 50,
    min_height: int = 50,
    top_k: int = 8
) -> List[Dict[str, torch.Tensor]]:
    """使用YOLO模型检测并裁剪图像中的物体

    Args:
        image_tensor: 原始图像张量 (shape: [1, 3, H, W])
        model_path: YOLO模型路径
        min_confidence: 最小置信度阈值
        min_width: 最小物体宽度
        min_height: 最小物体高度
        top_k: 返回前k个主要物体

    Returns:
        包含裁剪图像和区域信息的字典列表
        每个字典包含:
        - "cropped_image": 裁剪后的图像张量
        - "region": 区域信息 (x1, y1, x2, y2)
        - "original_size": 原始尺寸 (height, width)
        - "class_id": 物体类别ID
        - "confidence": 检测置信度
    """
    # 加载YOLO模型
    model = YOLO(model_path)
    
    # 将张量转换为OpenCV格式
    # 转换步骤：1. 移至CPU 2. 调整维度 3. 转换为numpy数组 4. 转换颜色空间 5. 调整范围到0-255
    img_np = image_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # 运行YOLO推理
    results = model(img_cv, conf=min_confidence)
    
    # 收集检测结果
    all_detections = []
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        for box, class_id, conf in zip(boxes, class_ids, confidences):
            # 计算面积
            x1, y1, x2, y2 = map(int, box)
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            # 过滤掉太小的物体
            if width >= min_width and height >= min_height:
                all_detections.append((box, class_id, conf, area))
    
    # 按置信度和面积排序，选择前k个
    all_detections.sort(key=lambda x: (x[2], x[3]), reverse=True)
    main_objects = all_detections[:top_k]
    
    # 生成裁剪结果
    crops = []
    for box, class_id, conf, _ in main_objects:
        x1, y1, x2, y2 = map(int, box)
        width = x2 - x1
        height = y2 - y1
        
        # 执行裁剪
        cropped_image = image_tensor[:, :, y1:y2, x1:x2].clone()
        
        # 构建返回字典
        crop_info = {
            "cropped_image": cropped_image,
            "region": (x1, y1, x2, y2),
            "original_size": (height, width),
            "class_id": int(class_id),
            "confidence": float(conf)
        }
        
        # 添加到结果列表
        crops.append(crop_info)
    
    return crops