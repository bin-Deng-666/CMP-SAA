"""将生成结果保存到 frontend/data/{image_id}/，与评估页加载格式一致。"""

import os
from typing import Optional

from PIL import Image

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def get_image_data_dir(image_id: str) -> str:
    return os.path.join(DATA_ROOT, str(image_id))


def save_adversarial_results(
    image_id: str,
    original: Image.Image,
    adversarial: Image.Image,
    perturbation: Optional[Image.Image] = None,
) -> str:
    """
    保存到 data/{image_id}/：
      - original.png
      - adversarial.png
      - perturbation_vis.png（可选）
    返回保存目录绝对路径。
    """
    img_dir = get_image_data_dir(image_id)
    os.makedirs(img_dir, exist_ok=True)

    original.save(os.path.join(img_dir, "original.png"))
    adversarial.save(os.path.join(img_dir, "adversarial.png"))
    if perturbation is not None:
        perturbation.save(os.path.join(img_dir, "perturbation_vis.png"))

    return img_dir
