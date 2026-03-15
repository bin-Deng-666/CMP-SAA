import torch
from PIL import Image
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model_name = "blip2"

if model_name == "blip2":
    # 初始化模型参数
    model_args = {
        "processor_path": "models/Salesforce/blip2-opt-2.7b",
        "lm_path": "models/Salesforce/blip2-opt-2.7b", 
        "device": 0 if torch.cuda.is_available() else -1
    }

# 导入模型模块
try:
    from models.blip2 import EvalModel
    model = EvalModel(model_args)
    print(f"Successfully loaded {model_name} model")
except ImportError as e:
    print(f"Error importing model: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error initializing model: {e}")
    sys.exit(1)

# 检查测试图片路径
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
img1_url = os.path.join(data_dir, 'demo1.png')
img2_url = os.path.join(data_dir, 'demo2.jpg')

# 确认图片文件存在
for img_url in [img1_url, img2_url]:
    if not os.path.exists(img_url):
        print(f"Error: Image file not found at {img_url}")
        sys.exit(1)

# 加载图片
try:
    raw_image1 = Image.open(img1_url).convert('RGB')
    raw_image2 = Image.open(img2_url).convert('RGB')
    print(f"Successfully loaded 2 images: {img1_url}, {img2_url}")
except Exception as e:
    print(f"Error loading image: {e}")
    sys.exit(1)

# 打印设备信息
print(f"Using device: {'GPU' if model_args['device'] >=0 else 'CPU'}")

# 定义测试问题（2个相同的文本提示）
test_questions = [
    "what is the main object in the image?",
    "what is the main object in the image?"
]

# 准备prompt和图片批次
batch_text = []
for question in test_questions:
    # 获取每个问题的prompt格式
    question_part, answer_part = model.get_vqa_prompt(question)
    batch_text.append(question_part+answer_part)

# 准备图片批次：2张图片，每张图片对应1个相同的文本提示
batch_images = [[raw_image1], [raw_image2]]

print(f"\nBatch size: {len(batch_text)} questions, {len(batch_images)} images")

# 生成回答
try:
    outputs = model.get_outputs(
        batch_text=batch_text,
        batch_images=batch_images,
        max_generation_length=50,  # 增加生成长度以获得更完整的回答
        num_beams=3,               # 减少beam数量以提高速度
        length_penalty=1.0
    )
    
    # 打印问题和对应的回答
    print("\nQuestion-Answer Results:")
    for i, (question, output) in enumerate(zip(test_questions, outputs)):
        print(output)
except Exception as e:
    print(f"Error during model inference: {e}")
    sys.exit(1)