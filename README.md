# GraduationProject

对抗攻击研究项目，包含对抗样本生成、评估和可视化工具。

## 文件树

```
GraduationProject/
├── feature_extractors/          # 特征提取器模块
│   ├── Base.py                  # 基础特征提取器类
│   ├── ClipB16.py              # CLIP B16 特征提取器
│   ├── ClipB32.py              # CLIP B32 特征提取器
│   ├── ClipL336.py             # CLIP L336 特征提取器
│   ├── ClipLaion.py            # CLIP Laion 特征提取器
│   └── __init__.py             # 模块初始化
├── frontend/                    # 前端可视化界面
│   ├── pages/                   # 页面目录
│   │   ├── 1-对抗图像生成.py   # 对抗图像生成页面
│   │   └── 2-对抗图像测试.py   # 对抗图像测试页面
│   └── 主页.py                  # 主页面入口
├── models/                      # 模型定义
│   ├── flamingo_src/            # Flamingo 模型源码
│   │   ├── __init__.py
│   │   ├── detail of lang encoder.md
│   │   ├── factory.py
│   │   ├── flamingo.py
│   │   ├── flamingo_lm.py
│   │   ├── helpers.py
│   │   └── utils.py
│   ├── BaseEvalModel.py         # 基础评估模型类
│   ├── blip2.py                # BLIP2 模型实现
│   └── instructblip.py         # InstructBLIP 模型实现
├── test/                        # 测试脚本
│   ├── test_dataset_loading.py  # 数据集加载测试
│   ├── test_embeddings.py       # 嵌入测试
│   ├── test_model_loading.py    # 模型加载测试
│   └── test_pytorch.py          # PyTorch 测试
├── utils/                       # 工具函数
│   ├── __init__.py
│   ├── attack_tool.py           # 攻击工具函数
│   ├── crop_images.py           # 图像裁剪工具
│   ├── crop_objects.py          # 目标裁剪工具
│   ├── env_desc.py              # 环境描述
│   ├── eval_datasets.py         # 评估数据集
│   └── eval_tool.py             # 评估工具函数
├── .gitignore                   # Git 忽略配置
├── cma.py                       # CMA-ES 攻击算法
├── maximize.py                  # 最大化攻击算法
└── test.py                      # 主测试脚本
```

## 主要功能模块

- **对抗攻击算法**: `cma.py`, `maximize.py`
- **模型实现**: `models/` 目录下的 BLIP2、InstructBLIP 等
- **评估工具**: `utils/eval_tool.py`, `test.py`
- **可视化界面**: `frontend/` 目录下的 Streamlit 应用
- **特征提取**: `feature_extractors/` 目录下的 CLIP 系列提取器
