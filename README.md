# CMP-SAA: Cross-Modal Plug-in Suffix Adversarial Attack towards VLMs

Official implementation of **CMP-SAA**, a cross-modal adversarial attack method for Vision-Language Models (VLMs) such as **BLIP-2** and **InstructBLIP**.

> 📄 Paper under review. Code will be fully released upon acceptance.

## 🔍 Overview

CMP-SAA injects a **lightweight plug-in suffix** into the input of VLMs through joint text-image adversarial optimization, with three key components:

1. **Cross-Modal Adversarial Mechanism**
   - **Image side**: optimizes perturbations in a continuous embedding space to *raise* the probability of generating the target text
   - **Text side**: reverse-optimizes text embeddings to *lower* the probability of the original output, sharpening the attack

2. **Lightweight Adversarial Suffix**
   - Continuous embeddings are optimized and mapped to discrete tokens via a **Straight-Through Estimator (STE)**
   - Implemented as a *plug-in suffix* appended after the prompt — no retraining, no modification of model weights
   - One suffix is **shared across multiple prompts** (`--prompt_num`), enabling cross-prompt transferability

3. **Environment-Aware Prefix**
   - A VLM-generated **environment description prefix** (see `utils/env_desc.py`) provides richer textual guidance while constraining the semantic space

## 📊 Evaluation

- **Models**: BLIP-2 (OPT-2.7B), InstructBLIP (Vicuna-7B)
- **Datasets**: COCO / VQAv2
- **Metric**: Attack Success Rate (generating the target text)

## 📁 Project Structure

```
CMP-SAA/
├── cma.py                     # Main attack pipeline & entry point
├── models/                    # VLM implementations (BLIP-2 / InstructBLIP / Flamingo)
├── feature_extractors/        # CLIP-based feature extractors (B16 / B32 / L336 / Laion)
├── utils/
│   ├── attack_tool.py         # Data/model loading, STE, prompt mapping
│   ├── eval_tool.py           # Attack success evaluation
│   ├── eval_datasets.py       # COCO dataset loading
│   └── env_desc.py            # VLM-generated environment descriptions
├── frontend/                  # Streamlit demo: adversarial generation & evaluation
└── test/                      # Environment sanity checks
```

## 🚀 Usage

### 1. Environment

```bash
pip install -r requirements.txt   # PyTorch, transformers, etc.
python test/test_pytorch.py       # sanity check
```

### 2. Data & Models

- Download BLIP-2 / InstructBLIP weights to `models/Salesforce/...`
- Download VQAv2 (COCO images + questions/annotations) and place under `data/` (paths in `utils/attack_tool.py`)

### 3. Run the Attack

```bash
python cma.py \
    --model_name blip2 \
    --method token_adv \
    --prompt_num 50 \
    --adversarial_length 10 \
    --iters 800 \
    --epsilon 0.125 --alpha 0.004
```

**Key arguments**:

| Argument | Description |
|----------|-------------|
| `--model_name` | `blip2` / `instructblip` |
| `--method` | attack variant: `token_adv` / `embed_adv` / `grad_embed_noise` / baselines |
| `--prompt_num` | number of prompts sharing one suffix (transferability) |
| `--adversarial_length` | suffix length |
| `--iters` / `--epsilon` / `--alpha` | optimization iterations & PGD budget/step |

### 4. Demo

A Streamlit front-end for interactive adversarial image generation and evaluation:

```bash
streamlit run frontend/home.py
```

## 📚 Citation

```bibtex
@inproceedings{deng2026cmpsaa,
  title     = {CMP-SAA: Cross-Modal Plug-in Suffix Adversarial Attack towards Vision-Language Models},
  author    = {Brin},
  booktitle = {Under review},
  year      = {2026}
}
```
