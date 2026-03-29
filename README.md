# 🏥 DeepSeek-LLM-7B-Chat 医疗领域 LoRA 微调项目

> 基于 DeepSeek-LLM-7B-Chat 的医疗对话领域 LoRA 高效微调，支持多轮问诊对话与心理疏导

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 项目概述

本项目专注于使用 **LoRA (Low-Rank Adaptation)** 技术对 [DeepSeek-LLM-7B-Chat](https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat) 大语言模型进行医疗领域的高效微调。

### 🎯 核心特性

| 特性 | 描述 |
|------|------|
| 🔧 **高效微调** | 采用 LoRA 技术，仅训练 0.1% 参数，大幅降低显存占用 |
| 🏥 **医疗对话** | 支持多轮医疗问诊对话，包含心理疏导场景 |
| 💬 **多轮对话** | 完整的上下文理解能力，支持多轮连续对话 |
| 🔄 **模型合并** | 提供 LoRA 权重合并脚本，生成独立部署模型 |
| 📊 **实验追踪** | 集成 SwanLab 可视化监控训练过程 |
| ⚡ **分布式训练** | 支持多卡分布式训练 (OpenMind) |

---

## 📁 项目结构

```
deepseek-llm-7B-chat-Lora-Fine-tuning/
├── 📄 finetune-multi-conv.py          # 🧠 Transformers Trainer 微调脚本
├── 📄 finetune-multi-openmind.py      # 🚀 OpenMind 分布式微调脚本
├── 📄 merge_model.py                  # 🔗 LoRA 权重合并脚本
├── 📄 model_load_modelscope.py        # 📥 ModelScope 模型下载工具
├── 📄 reasoning.py                    # 💭 多模型推理示例
├── 📄 requirements.txt                 # 📦 依赖包列表
├── 📂 data/
│   ├── medical_multi_data.json        # 🏥 医疗多轮对话数据集
│   └── medical_multi_data_pro.json     # 📈 增强版医疗数据集
└── 📂 output/                         # 📁 微调模型输出目录
```

---

## 🚀 快速开始

### 1️⃣ 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 核心依赖
transformers >= 4.36.0
datasets
accelerate
peft
torch >= 2.0.0
pandas
```

### 2️⃣ 下载模型

```bash
# 使用 ModelScope 下载 DeepSeek-LLM-7B-Chat
python model_load_modelscope.py
```

或手动下载:
```bash
git lfs install
git clone https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat ./model/deepseek-ai/deepseek-llm-7b-chat
```

### 3️⃣ 开始微调

**方式一：标准微调 (单卡，推荐入门)**

```bash
python finetune-multi-conv.py
```

**方式二：分布式训练 (多卡)**

```bash
torchrun --nproc_per_node=8 finetune-multi-openmind.py
```

---

## 📖 核心脚本说明

### 🧠 `finetune-multi-conv.py` - 标准微调

基于 Hugging Face Transformers Trainer 的 LoRA 微调脚本。

**关键配置：**

```python
# LoRA 参数
lora_config = LoraConfig(
    r=64,                           # 🔢 LoRA rank
    lora_alpha=32,                  # 📊 LoRA alpha
    lora_dropout=0.05,               # 🎲 Dropout
    target_modules=['up_proj', 'gate_proj', 'q_proj', 'o_proj', 'down_proj', 'v_proj', 'k_proj'],
    task_type=TaskType.CAUSAL_LM,
)

# 训练参数
train_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
)
```

---

### 🚀 `finetune-multi-openmind.py` - 分布式微调

支持多卡并行训练的高级微调脚本，具备：

- ✅ 分布式训练 (DDP)
- ✅ QLoRA / LoRA 模式切换
- ✅ 自动识别全连接层
- ✅ 自定义 DataCollator

**命令行参数：**

```bash
python finetune-multi-openmind.py \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-3 \
    --lora_rank 64 \
    --train_mode lora \
    --distributed True
```

---

### 🔗 `merge_model.py` - 模型合并

将训练好的 LoRA Adapter 合并回基础模型，生成可直接部署的独立模型：

```python
# 使用示例
merge_lora_to_base_model(
    model_name_or_path='./model/deepseek-ai/deepseek-llm-7b-chat',
    adapter_name_or_path='./output/deepseek-mutil-test/',
    save_path='./output/merged-model'
)
```

---

### 💭 `reasoning.py` - 推理示例

提供多种模型的推理接口，包含：

```python
# 单轮对话
deepseek_model_inference(model_path, prompt)

# 多轮对话
deepseek_multi_conversation_inference(model_path, prompt, chat_history)
```

---

## 🏥 医疗数据集格式

数据集采用多轮对话格式：

```json
{
    "conversation": [
        {
            "system": "现在你是一个心理专家，我有一些心理问题，请你用专业的知识帮我解决。",
            "input": "医生，我最近总是感到很焦虑...",
            "output": "你好，首先感谢你对我敞开心扉..."
        },
        {
            "input": "是的，我知道应该理性看待，但就是忍不住会去比较...",
            "output": "了解你的情况后，我建议你在睡前尝试进行放松训练..."
        }
    ]
}
```

---

## ⚙️ 硬件配置参考

| 训练模式 | 最低显存 | 推荐配置 |
|---------|---------|---------|
| LoRA (fp16) | 16GB | 24GB+ (如 A10G, RTX 3090) |
| LoRA (bf16) | 20GB | 32GB+ (如 A100 40G) |
| QLoRA (4bit) | 8GB | 16GB+ (如 T4, RTX 4060) |

---

## 📊 训练监控

使用 [SwanLab](https://swanlab.cn) 进行实验追踪：

```python
swanlab_callback = SwanLabCallback(
    project="deepseek-finetune-test",
    experiment_name="medical-chat",
    config={"dataset": "medical_multi_data.json", "peft": "lora"}
)
```

访问 https://swanlab.115.zone 查看训练曲线。

---

## 🔬 技术栈

<p>

| 库 | 用途 |
|----|-----|
| 🧠 **Transformers** | 模型加载与训练 |
| 💉 **PEFT** | LoRA 高效微调 |
| 🔥 **PyTorch** | 深度学习框架 |
| 📊 **SwanLab** | 实验可视化 |
| 🌍 **OpenMind** | 分布式训练 (可选) |
| 📦 **ModelScope** | 模型下载 |

</p>

---

## 📝 License

本项目基于 MIT License 开源。

---

## 🙏 致谢

- [DeepSeek](https://www.deepseek.com/) - 提供预训练大语言模型
- [Hugging Face](https://huggingface.co/) - Transformers 生态
- [PEFT](https://github.com/huggingface/peft) - 高效微调工具
- [ModelScope](https://modelscope.cn/) - 模型托管平台

---

<p align="center">
  <sub>Built with ❤️ for Medical AI · Powered by DeepSeek LLM</sub>
</p>
