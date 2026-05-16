"""
跨模态辅助攻击方法 (CMA Attack) - 图像级别实现
从 cma.py 提取 embed_adv 方法并适配为单图像调用接口
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import importlib

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.attack_tool import (
    load_model,
    get_img_id_train_prompt_map,
    get_img_id_environment_map,
    get_intended_token_ids,
)


@dataclass
class CMAAttackConfig:
    """跨模态辅助攻击配置类
    
    参考 cma.py 中的 AttackConfig 设计
    """
    model_name: str = "blip2"  # 模型名称: blip2, instructblip
    method: str = "embed_adv"  # 攻击方法: embed_adv, token_adv 等
    target_text: str = "Unknown"  # 目标攻击文本
    adversarial_length: int = 16  # 对抗文本后缀长度
    prompt_num: int = 50  # 用于训练的文本提示数量
    iters: int = 500  # 迭代次数
    epsilon: float = 32/255  # 扰动大小限制 (像素级)
    alpha: float = 1/255  # 学习率
    device: int = 0  # GPU设备号
    debug: bool = False  # 调试模式


def pil_to_tensor(pic: Image.Image, device: str = "cuda") -> torch.Tensor:
    """将 PIL Image 转换为 PyTorch 张量 (归一化到 [0, 1])
    
    Args:
        pic: PIL Image 对象
        device: 运行设备
    
    Returns:
        torch.Tensor: 转换后的张量，shape为 [1, 3, 224, 224]
    """
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # 转换到 [0, 1]
    ])
    tensor = transform(pic).unsqueeze(0)
    if device != "cpu":
        tensor = tensor.to(f"cuda:{device}" if isinstance(device, int) else device)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """将 PyTorch 张量转换为 PIL Image
    
    Args:
        tensor: 输入张量，shape为 [1, 3, H, W] 或 [3, H, W]，值范围 [0, 1]
    
    Returns:
        PIL.Image: 转换后的图像
    """
    # 移除 batch 维度
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # 转换为 numpy 并调整维度
    img_np = tensor.detach().cpu().clamp(0, 1).numpy()
    img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC
    
    # 转换到 0-255 范围
    img_np = (img_np * 255).astype(np.uint8)
    
    return Image.fromarray(img_np)


def _build_inputs_for_embed_adv(
    question_inputs: Dict[str, torch.Tensor],
    answer_inputs: torch.Tensor,
    question_embeddings: torch.Tensor,
    answer_embeddings: torch.Tensor,
    current_target: torch.Tensor,
    adversarial_embeddings: torch.Tensor,
    adversarial_length: int,
    processor,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """与 cma.py 中 build_inputs_for_embed_adv 保持一致。"""
    combined_embeddings = torch.cat(
        [question_embeddings, adversarial_embeddings, answer_embeddings], dim=1
    )

    pad_token_id = processor.tokenizer.pad_token_id
    padded_input_ids = torch.cat(
        [
            question_inputs.input_ids,
            torch.full(
                (1, adversarial_length),
                pad_token_id,
                device=question_inputs.input_ids.device,
            ),
            answer_inputs,
        ],
        dim=1,
    )
    combined_attention_mask = torch.cat(
        [
            question_inputs.attention_mask,
            torch.ones((1, adversarial_length), device=question_inputs.input_ids.device),
            torch.ones_like(answer_inputs),
        ],
        dim=1,
    )
    inputs = {
        "input_ids": padded_input_ids,
        "attention_mask": combined_attention_mask,
    }

    labels = get_intended_token_ids(inputs["input_ids"], current_target)
    return inputs, labels, combined_embeddings


def generate_adversarial_image_cma(
    image: Image.Image,
    image_id: str,
    model_name: str = "blip2",
    method: str = "embed_adv",
    target_text: str = "Unknown",
    adversarial_length: int = 16,
    prompt_num: int = 50,
    iterations: int = 500,
    epsilon: float = 32/255,
    alpha: float = 1/255,
    device: int = 0,
    progress_callback: Optional[callable] = None
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """生成对抗图像 - CMA 跨模态辅助攻击 (图像级别接口)
    
    使用 embed_adv 方法生成对抗样本，通过优化对抗文本嵌入和图像扰动
    来误导 VQA 模型输出目标文本。
    
    Args:
        image: 原始 PIL 图像
        image_id: 图像ID（用于获取提示词）
        model_name: 模型名称，如 "blip2"
        method: 攻击方法，默认 "embed_adv"
        target_text: 目标攻击文本
        adversarial_length: 对抗文本后缀长度
        prompt_num: 用于训练的文本提示数量
        iterations: 迭代次数
        epsilon: 扰动大小限制 (像素级，如 32/255)
        alpha: 学习率 (如 1/255)
        device: GPU设备号
        progress_callback: 进度回调函数，接收 (current_iter, total_iters, current_loss) 参数
    
    Returns:
        Tuple[Image.Image, Image.Image, Image.Image]: 
            (原始图像, 对抗图像, 扰动可视化图像)
    """
    config = CMAAttackConfig(
        model_name=model_name,
        method=method,
        target_text=target_text,
        adversarial_length=adversarial_length,
        prompt_num=prompt_num,
        iters=iterations,
        epsilon=epsilon,
        alpha=alpha,
        device=device
    )
    return generate_adversarial_image_cma_with_config(image, image_id, config, progress_callback)


def generate_adversarial_image_cma_with_config(
    image: Image.Image,
    image_id: str,
    config: CMAAttackConfig,
    progress_callback: Optional[callable] = None
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """生成对抗图像 - 使用配置对象
    
    Args:
        image: 原始 PIL 图像
        image_id: 图像ID
        config: CMA攻击配置对象
        progress_callback: 进度回调函数
    
    Returns:
        Tuple[Image.Image, Image.Image, Image.Image]: 
            (原始图像, 对抗图像, 扰动可视化图像)
    """
    # 设置设备（与 cma.py 保持一致：CUDA 用设备号，CPU 用字符串）
    tensor_device = config.device if config.device >= 0 else "cpu"

    # 加载模型（沿用 cma.py 的 load_model 流程）
    print(f"加载模型: {config.model_name}")
    module = importlib.import_module(f"models.{config.model_name}")
    eval_model = load_model(config.device, module, config.model_name)
    processor = eval_model.processor
    
    # 准备原始图像（与 cma.py 对齐）
    item_images = [[image]]
    original_image = eval_model._prepare_images(
        item_images, normalize=False
    ).to(tensor_device).requires_grad_(False)
    
    # 初始化图像扰动
    image_perturbation = torch.randn(
        [1, 3, 224, 224], requires_grad=True, device=tensor_device
    )
    
    # 获取提示词
    img_id_to_train_prompt = get_img_id_train_prompt_map(config.prompt_num)
    img_id_to_environment = get_img_id_environment_map()
    
    total_prompt_list = img_id_to_train_prompt.get(str(image_id), [])
    environment = img_id_to_environment.get(str(image_id), "unknown")
    
    if not total_prompt_list:
        print(f"警告: 未找到图像 {image_id} 的提示词，使用默认提示词")
        total_prompt_list = ["What is this?", "What can you see?", "Describe this image."]
    
    print(f"图像 {image_id} 对应的提示词数量: {len(total_prompt_list)}")
    
    # 初始化对抗嵌入 (embed_adv 方法)
    if config.method == "embed_adv":
        target_token_ids = processor.tokenizer.encode(config.target_text, add_special_tokens=False)
        adversarial_embeddings = eval_model.model.get_input_embeddings()(
            torch.tensor(target_token_ids, device=tensor_device)
        ).repeat(1, config.adversarial_length // len(target_token_ids) + 1, 1)[:, :config.adversarial_length, :]
        adversarial_embeddings = adversarial_embeddings.clone().detach().requires_grad_(True)
        adversarial_embeddings_init = adversarial_embeddings.clone().detach()
    else:
        raise NotImplementedError(f"方法 {config.method} 尚未实现")
    
    # 训练循环
    best_loss = float('inf')
    best_attack = None
    
    import random
    from collections import deque
    
    access_order = list(range(len(total_prompt_list)))
    random.shuffle(access_order)
    access_order = deque(access_order)
    index_count = 0
    
    print(f"\n开始训练，迭代次数: {config.iters}")
    
    for ep in range(config.iters):
        # 提示词轮换
        if index_count != 0 and index_count % len(total_prompt_list) == 0:
            rotation_offset = random.randint(0, len(total_prompt_list)-1)
            access_order.rotate(rotation_offset)
            index_count = 0
        text_idx = access_order[index_count]
        index_count += 1
        
        # 获取当前提示词
        current_question = total_prompt_list[text_idx]
        current_question = f"Against the background of {environment}, {current_question}"
        
        # 构建 VQA 模板（与 cma.py 对齐）
        current_question, current_answer = eval_model.get_vqa_prompt(
            question=current_question, answer=config.target_text
        )
        current_text = current_question + current_answer

        # 处理整体文本输入
        current_inputs = processor(
            text=[current_text],
            padding=True,
            truncation=True,
            max_length=1000,
            return_tensors="pt"
        ).to(tensor_device)

        # 问题部分输入
        question_inputs = processor(
            text=[current_question],
            padding=True,
            truncation=True,
            max_length=1000,
            return_tensors="pt"
        ).to(tensor_device)
        
        # 答案部分输入
        answer_inputs = processor.tokenizer.encode(
            current_answer,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(tensor_device).detach()
        
        # 目标 token ids
        current_target = processor.tokenizer.encode(
            config.target_text,
            add_special_tokens=True,
            return_tensors="pt"
        ).to(tensor_device).detach()
        
        # 构建组合嵌入与标签（与 cma.py 的 embed_adv 一致）
        question_embeddings = eval_model.model.get_input_embeddings()(question_inputs['input_ids'])
        answer_embeddings = eval_model.model.get_input_embeddings()(answer_inputs)
        model_inputs, labels, combined_embeddings = _build_inputs_for_embed_adv(
            question_inputs=question_inputs,
            answer_inputs=answer_inputs,
            question_embeddings=question_embeddings,
            answer_embeddings=answer_embeddings,
            current_target=current_target,
            adversarial_embeddings=adversarial_embeddings,
            adversarial_length=config.adversarial_length,
            processor=processor,
        )
        
        # 对抗图像
        adv_image = original_image + image_perturbation
        
        # 设置自定义嵌入
        eval_model.set_custom_embeddings(combined_embeddings)
        
        # 前向传播
        outputs = eval_model.model(
            input_ids=model_inputs["input_ids"],
            pixel_values=adv_image,
            attention_mask=model_inputs["attention_mask"],
            labels=labels
        )
        
        # 清除自定义嵌入
        eval_model.clear_custom_embeddings()
        
        loss = outputs.loss
        
        # 反向传播
        loss.backward()
        
        # 更新最佳攻击
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_attack = image_perturbation.clone().detach()
        
        # 更新图像扰动
        grad_img = image_perturbation.grad.detach()
        image_perturbation.data = torch.clamp(
            image_perturbation.data - config.alpha * torch.sign(grad_img),
            min=-config.epsilon,
            max=config.epsilon
        )
        image_perturbation.grad.zero_()
        
        # 更新对抗嵌入
        grad_embeddings = adversarial_embeddings.grad.detach()
        update = config.alpha * torch.sign(grad_embeddings) * (1 - ep/config.iters)
        adversarial_embeddings.data = torch.clamp(
            adversarial_embeddings.data + update,
            min=adversarial_embeddings_init - 1,
            max=adversarial_embeddings_init + 1
        )
        adversarial_embeddings.grad.zero_()
        
        # 进度回调
        if progress_callback is not None and (ep + 1) % 10 == 0:
            progress_callback(ep + 1, config.iters, loss.item())
        
        if (ep + 1) % 50 == 0:
            print(f"  迭代 {ep+1}/{config.iters}, Loss: {loss.item():.4f}")
    
    # 与 cma.py 保持一致：最终使用当前迭代扰动
    final_perturbation = image_perturbation.detach()
    
    # 生成对抗图像
    adv_image = original_image + final_perturbation
    adv_image = torch.clamp(adv_image, 0, 1)
    
    # 转换为 PIL
    original_pil = tensor_to_pil(original_image)
    adv_pil = tensor_to_pil(adv_image)
    
    # 生成扰动可视化
    pert_np = final_perturbation.squeeze(0).detach().cpu().numpy()
    pert_np = np.transpose(pert_np, (1, 2, 0))
    p_min, p_max = pert_np.min(), pert_np.max()
    if p_max > p_min:
        pert_np = ((pert_np - p_min) / (p_max - p_min) * 255).astype(np.uint8)
    else:
        pert_np = np.zeros_like(pert_np, dtype=np.uint8)
    perturbation_pil = Image.fromarray(pert_np)
    
    print(f"\n攻击完成! 最佳 Loss: {best_loss:.4f}")
    
    # 清理 GPU 内存
    if config.device >= 0:
        torch.cuda.empty_cache()
    
    return original_pil, adv_pil, perturbation_pil


def quick_attack_cma(
    image: Image.Image,
    image_id: str,
    device: int = 0
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """快速 CMA 攻击（使用默认参数）
    
    Args:
        image: 原始 PIL 图像
        image_id: 图像ID
        device: GPU设备号
    
    Returns:
        Tuple[Image.Image, Image.Image, Image.Image]: 
            (原始图像, 对抗图像, 扰动可视化图像)
    """
    config = CMAAttackConfig(device=device)
    return generate_adversarial_image_cma_with_config(image, image_id, config)


# 向后兼容的别名
attack = generate_adversarial_image_cma_with_config
AttackConfig = CMAAttackConfig