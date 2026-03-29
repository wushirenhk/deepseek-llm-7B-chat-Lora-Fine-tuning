# 🏥 DeepSeek-LLM-7B-Chat 医疗领域 LoRA 微调项目

> 基于 DeepSeek-LLM-7B-Chat 的医疗对话领域 LoRA 高效微调，支持多轮问诊对话与心理疏导

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 项目介绍

本项目专注于使用 **LoRA (Low-Rank Adaptation)** 技术对 [DeepSeek-LLM-7B-Chat](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat) 大语言模型进行**医疗领域**的高效微调。

### 🎯 核心能力

| 能力 | 说明 |
|------|------|
| 🏥 **医疗问诊** | 理解症状、提供专业医疗建议 |
| 🧠 **心理疏导** | 心理焦虑、学业压力等问题的专业疏导 |
| 💬 **多轮对话** | 保持上下文连贯，支持无限轮次对话 |
| 🔐 **隐私安全** | 本地部署，数据不外传 |

---

## 🗂️ 项目结构

```
deepseek-llm-7B-chat-Lora-Fine-tuning/
├── 📄 finetune-multi-conv.py          # 🧠 标准微调脚本 (Transformers Trainer)
├── 📄 finetune-multi-openmind.py      # 🚀 分布式微调脚本 (OpenMind)
├── 📄 merge_model.py                  # 🔗 LoRA 权重合并脚本
├── 📄 model_load_modelscope.py        # 📥 ModelScope 模型下载
├── 📄 reasoning.py                     # 💭 模型推理示例
├── 📄 requirements.txt                 # 📦 依赖列表
├── 📂 data/
│   ├── medical_multi_data.json        # 🏥 医疗多轮对话数据集
│   └── medical_multi_data_pro.json     # 📈 增强版数据集
├── 📖 README.md                        # 📘 项目说明文档
└── 📖 TUTORIAL.md                      # 📘 完整教程（函数解析、难点分析）
```

---

## 🚀 快速开始

### 1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

**核心依赖：**

| 包 | 版本 | 用途 |
|----|------|------|
| transformers | ≥4.36 | 模型加载与训练 |
| peft | ≥0.4 | LoRA 高效微调 |
| torch | ≥2.0 | 深度学习框架 |
| datasets | - | 数据集处理 |
| accelerate | - | 加速训练 |
| swanlab | - | 实验可视化 |

### 2️⃣ 下载模型

```python
# 方式一：ModelScope 下载
python model_load_modelscope.py

# 方式二：HuggingFace
git lfs install
git clone https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat \
    ./model/deepseek-ai/deepseek-llm-7b-chat
```

### 3️⃣ 开始微调

```bash
# 标准微调（单卡，推荐入门使用）
python finetune-multi-conv.py

# 分布式训练（多卡）
torchrun --nproc_per_node=8 finetune-multi-openmind.py
```

### 4️⃣ 合并模型

```python
from merge_model import merge_lora_to_base_model

merge_lora_to_base_model(
    model_name_or_path='./model/deepseek-ai/deepseek-llm-7b-chat',
    adapter_name_or_path='./output/deepseek-mutil-test/',
    save_path='./output/merged-model'
)
```

### 5️⃣ 推理测试

```python
from reasoning import deepseek_multi_conversation_inference

model_path = "./output/merged-model"
chat_history = []

# 多轮对话
result, chat_history = deepseek_multi_conversation_inference(
    model_path,
    "医生，我最近总是感到很焦虑",
    chat_history
)
print(result)  # 模型回复
```

---

## 📘 完整教程

**📖 查看 [TUTORIAL.md](TUTORIAL.md) 获取：**

- 🔬 LoRA 技术原理解析
- 📝 数据处理流程详解
- 🧩 每个函数功能的深入分析
- ⚠️ 项目难点及解决方案
- ⚙️ 超参数配置指南
- 🚀 模型部署与推理

---

## 💡 核心设计亮点

### 1. Label Masking 机制

只让模型学习"回答"，忽略"问题"部分：

```python
# 🚫 用户输入不计算 loss
# ✅ 助手回复计算 loss
labels = [-100] * len(input_ids) + output_ids + [eos_token_id]
```

### 2. 多轮对话序列化

自动将多轮对话转换为连续的 token 序列：

```
<|bos|>system<|user|>Q1<|assistant|>A1<|user|>Q2<|assistant|>A2<|eos|>
```

### 3. 显存优化策略

| 技术 | 效果 |
|------|------|
| `fp16` | 显存减半 |
| `gradient_checkpointing` | 显存再降 30% |
| `gradient_accumulation` | 等效大 batch |
| `device_map="auto"` | 多卡自动分配 |

---

## ⚙️ 硬件配置参考

| 训练模式 | 最低显存 | 推荐配置 | 可训练参数 |
|---------|---------|---------|-----------|
| **LoRA (fp16)** | 16GB | 24GB+ (RTX 3090/A10G) | ~7M |
| **LoRA (bf16)** | 20GB | 32GB+ (A100 40G) | ~7M |
| **QLoRA (4bit)** | 8GB | 16GB+ (T4/RTX 4060) | ~7M |

---

## 📊 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                     用户输入                            │
│            "医生，我最近很焦虑..."                        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  DeepSeek Chat Template                  │
│   <|bos|><|system|><|user|>...<|assistant|>            │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              DeepSeek-LLM-7B + LoRA                      │
│                                                         │
│   ┌─────────────────────────────────────────────────┐   │
│   │  Frozen: q_proj, k_proj, v_proj, o_proj        │   │
│   │  Trainable: LoRA_A, LoRA_B (rank=64)           │   │
│   └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   模型输出                               │
│            "我理解你的感受..."                           │
└─────────────────────────────────────────────────────────┘
```

---

## 🔬 技术栈

| 类别 | 技术 | 说明 |
|------|------|------|
| 🧠 **基座模型** | DeepSeek-LLM-7B-Chat | 7B 参数大语言模型 |
| 🔧 **微调框架** | PEFT | LoRA/QLORA/QLoRA 支持 |
| 🏋️ **训练框架** | Transformers Trainer | 成熟稳定的训练流程 |
| 🚀 **分布式** | OpenMind / DDP | 多卡并行训练 |
| 📊 **监控** | SwanLab | 实验追踪与可视化 |
| 📥 **模型源** | ModelScope / HF | 模型与数据集下载 |

---

## 📝 数据集格式

```json
{
    "conversation": [
        {
            "system": "现在你是一个心理专家...",
            "input": "医生，我最近总是感到很焦虑...",
            "output": "你好，首先感谢你对我敞开心扉..."
        },
        {
            "input": "是的，我知道应该理性看待...",
            "output": "了解你的情况后，我建议..."
        }
    ]
}
```

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

---

## 🙏 致谢

- [DeepSeek](https://www.deepseek.com/) - 提供预训练大语言模型
- [Hugging Face](https://huggingface.co/) - Transformers 生态
- [PEFT](https://github.com/huggingface/peft) - 高效微调工具
- [ModelScope](https://modelscope.cn/) - 模型托管平台
- [SwanLab](https://swanlab.cn/) - 实验可视化平台

---

## 📄 License

MIT License

---

<p align="center">
  Built with ❤️ for Medical AI · Powered by DeepSeek LLM
</p>
