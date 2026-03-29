# 🏥 DeepSeek-LLM-7B-Chat 医疗领域 LoRA 微调 - 完整教程

> 本教程详细介绍了如何使用 LoRA 技术对 DeepSeek-LLM-7B-Chat 进行医疗领域微调，包括核心原理、代码解析、难点分析及解决方案

---

## 📑 目录

1. [项目概述](#1-项目概述)
2. [技术原理](#2-技术原理)
3. [数据处理详解](#3-数据处理详解)
4. [核心函数解析](#4-核心函数解析)
5. [难点与解决方案](#5-难点与解决方案)
6. [训练配置指南](#6-训练配置指南)
7. [模型部署与推理](#7-模型部署与推理)

---

## 1. 项目概述

### 1.1 项目背景

本项目针对 **医疗问诊对话** 场景，对 DeepSeek-LLM-7B-Chat 大语言模型进行领域适配微调，使其能够：

- ✅ 理解患者描述的症状和问题
- ✅ 提供专业的医疗建议和心理疏导
- ✅ 保持多轮对话的上下文连贯性
- ✅ 以专业的医患沟通方式回应

### 1.2 技术选型

| 技术                           | 选择理由                                   |
| ------------------------------ | ------------------------------------------ |
| **LoRA**                 | 仅训练 0.1% 参数，显存需求低，适合单卡微调 |
| **DeepSeek-7B-Chat**     | 中文能力强，训练对话理解能力优秀           |
| **Transformers Trainer** | 成熟的训练框架，支持混合精度、梯度累积     |
| **SwanLab**              | 轻量级实验追踪，支持中文界面               |

### 1.3 项目工作流

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   数据准备   │ ──▶ │   模型加载   │ ──▶ │   LoRA微调  │ ──▶ │   模型合并  │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │
   医疗对话JSON      加载预训练模型      训练Adapter       合并为独立模型
   格式转换          配置分词器          监控训练曲线       部署推理
```

---

## 2. 技术原理

### 2.1 LoRA 核心原理

**LoRA (Low-Rank Adaptation)** 是一种高效微调方法，其核心思想是：

```
原始权重: W₀ ∈ ℝ^(d×k)
LoRA更新: ΔW = BA, 其中 B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)
最终权重: W = W₀ + ΔW = W₀ + BA
```

**在训练过程中：**

- 🔒 `W₀` 冻结不变，不计算梯度
- 📝 只训练 `A` 和 `B` 矩阵（rank = r 的低秩分解）
- 🚀 参数量从 `d×k` 减少到 `r×(d+k)`

### 2.2 为什么选择 LoRA？

| 对比项   | Full Fine-tune | LoRA        | QLoRA       |
| -------- | -------------- | ----------- | ----------- |
| 参数量   | 7B (100%)      | ~7M (~0.1%) | ~7M (~0.1%) |
| 显存占用 | ~28GB          | ~16GB       | ~8GB        |
| 训练速度 | 慢             | 快          | 中等        |
| 效果     | 最好           | 接近全量    | 略低        |

### 2.3 目标模块选择

本项目针对 Transformer 的 attention 组件进行 LoRA 注入：

```python
target_modules = [
    'up_proj',   # FFN 升维投影
    'gate_proj', # Gate 门控投影
    'q_proj',    # Query 投影
    'o_proj',    # Output 投影
    'down_proj', # FFN 降维投影
    'v_proj',    # Value 投影
    'k_proj'     # Key 投影
]
```

---

## 3. 数据处理详解

### 3.1 数据格式

原始数据采用多轮对话格式 (`medical_multi_data.json`)：

```json
{
    "conversation": [
        {
            "system": "现在你是一个心理专家，我有一些心理问题，请你用专业的知识帮我解决。",
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

**格式说明：**

- `system`: 系统提示词（可选），定义 AI 角色
- `input`: 用户输入（Human）
- `output`: AI 助手回复（Assistant）

### 3.2 数据转换为对话模板

DeepSeek 模型使用特殊的对话模板格式：

```
<|bos|><|system|>{system_prompt}<|user|>{user_input}<|assistant|>{assistant_response}<|eos|>
```

**关键函数：`process_data()`**

```python
def process_data(data, tokenizer, max_seq_length):
    """
    将原始对话数据转换为模型可训练的格式
    """
    input_ids, attention_mask, labels = [], [], []

    for conv in conversations:
        # 1. 获取对话内容
        system_text = conv.get('instruction', '')  # system prompt
        human_text = conv['input']   # 用户输入
        assistant_text = conv['output']  # 助手回复

        # 2. 构建输入文本
        input_text = (
            f"{tokenizer.bos_token}"           # 起始符 <|bos|>
            f"{system_text}\n\n"                # 系统提示
            f"User:{human_text}\n\n"           # 用户输入
            f"Assistant:"                        # 助手标识
        )

        # 3. Tokenize
        input_tokenizer = tokenizer(input_text, add_special_tokens=False)
        output_tokenizer = tokenizer(assistant_text, add_special_tokens=False)

        # 4. 拼接并添加结束符
        input_ids += (
            input_tokenizer["input_ids"] +
            output_tokenizer["input_ids"] +
            [tokenizer.eos_token_id]
        )

        # 5. 构建 Mask（关键！）
        attention_mask += input_tokenizer["attention_mask"] +
                         output_tokenizer["attention_mask"] + [1]

        # 6. 构建 Labels（关键！）
        # 用户输入部分用 -100 屏蔽，不计算 loss
        labels += (
            [-100] * len(input_tokenizer["input_ids"]) +  # 🚫 屏蔽用户输入
            output_tokenizer["input_ids"] +                # ✅ 计算助手回复 loss
            [tokenizer.eos_token_id]
        )

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
```

### 3.3 Labels 掩码机制（重点理解）

```
┌──────────────────┬─────────────────────────────────┬────────┐
│   User Input     │       Assistant Response        │  EOS   │
├──────────────────┼─────────────────────────────────┼────────┤
│   Labels: -100   │    Labels: [actual token ids]    │  EOS   │
│   (不计算 loss)  │         (计算 loss)             │ (loss) │
└──────────────────┴─────────────────────────────────┴────────┘
     ❌ 忽略                    ✅ 优化                       ✅ 优化
```

**为什么这样做？**

1. **只学习助手回复**：模型只需要学习如何生成正确的回复
2. **避免学习用户输入**：用户输入只是"问题"，不是"答案"
3. **提高训练效率**：减少不必要的梯度计算

---

## 4. 核心函数解析

### 4.1 `finetune-multi-conv.py` - 标准微调脚本

#### 4.1.1 加载模型

```python
# 加载模型
model_path = "./model/deepseek-ai/deepseek-llm-7b-chat"
model_kwargs = {
    "torch_dtype": torch.float16,    # 半精度，节省显存
    "use_cache": True,               # 推理时使用
    "device_map": "auto"              # 自动设备分配
}
model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
```

#### 4.1.2 配置 LoRA

```python
lora_config = LoraConfig(
    r=64,                          # 🔢 LoRA rank，越大越强但越慢
    lora_alpha=32,                 # 📊 缩放因子，通常为 rank 的一半
    lora_dropout=0.05,              # 🎲 Dropout 防止过拟合
    bias="none",                   # 不训练 bias
    target_modules=[...],           # 目标模块列表
    task_type=TaskType.CAUSAL_LM,  # 因果语言模型任务
    inference_mode=False           # 训练模式
)
```

#### 4.1.3 训练参数

```python
train_args = TrainingArguments(
    output_dir="./output/deepseek-mutil-test",
    per_device_train_batch_size=2,       # 每卡 batch size
    gradient_accumulation_steps=8,        # 梯度累积，等效 batch = 2×8 = 16
    num_train_epochs=3,                  # 训练轮数
    save_steps=5000,                      # 每 5000 步保存一次
    learning_rate=2e-5,                   # 学习率
    gradient_checkpointing=True,         # 显存优化
    fp16=True,                            # 混合精度
)
```

---

### 4.2 `finetune-multi-openmind.py` - 分布式微调脚本

#### 4.2.1 分布式训练设置

```python
def setup_distributed(args):
    """初始化分布式环境"""
    if args.distributed:
        dist.init_process_group(backend="nccl")  # NCCL 用于 GPU 通信
        torch.cuda.set_device(args.local_rank)
        model.to(args.local_rank)
        model = DDP(model, device_ids=[args.local_rank])
```

#### 4.2.2 自动识别 LoRA 目标模块

```python
def find_all_linear_names(model, train_mode):
    """自动找出所有全连接层"""
    cls = bnb.nn.Linear4bit if train_mode == 'qlora' else nn.Linear
    lora_module_names = set()

    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')

    return list(lora_module_names)
```

#### 4.2.3 自定义 DataCollator

```python
class DataCollatorForSeq2SeqCustom:
    """自定义序列到序列的数据整理器"""
    def __call__(self, batch):
        # 提取 batch 中的数据
        input_ids = [example['input_ids'] for example in batch]
        attention_mask = [example['attention_mask'] for example in batch]
        labels = [example['labels'] for example in batch]

        # 填充到同一长度
        input_ids = self.pad_sequence(input_ids)
        attention_mask = self.pad_sequence(attention_mask)
        labels = self.pad_sequence(labels)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels)
        }
```

---

### 4.3 `merge_model.py` - 模型合并脚本

#### 4.3.1 为什么需要合并？

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│  Base Model │    +    │ LoRA Adapter │   ──▶   │ Merged Model │
│   (7B params)│         │ (~7M params) │         │  (7B params) │
└─────────────┘         └─────────────┘         └─────────────┘
     冻结                    训练                  可直接部署
```

#### 4.3.2 合并函数

```python
def merge_lora_to_base_model(model_name_or_path, adapter_name_or_path, save_path):
    """将 LoRA Adapter 合并到基础模型"""

    # 1. 加载基础模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 2. 加载 LoRA Adapter
    model = PeftModel.from_pretrained(
        model, adapter_name_or_path,
        device_map="auto",
        trust_remote_code=True
    )

    # 3. 合并权重（核心操作）
    merged_model = model.merge_and_unload()

    # 4. 保存合并后的模型
    tokenizer.save_pretrained(save_path)
    merged_model.save_pretrained(save_path, safe_serialization=False)

    # 5. 复制非权重文件（如 config.json）
    copy_files_not_in_B(model_name_or_path, save_path)
```

---

### 4.4 `reasoning.py` - 推理函数

#### 4.4.1 单轮对话推理

```python
def deepseek_model_inference(model_path: str, prompt: str):
    """单轮对话推理"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    messages = [{"role": "user", "content": prompt}]

    # 使用对话模板
    input_tensor = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  # 添加 generation prompt
        return_tensors="pt"
    )

    # 生成回复
    outputs = model.generate(
        input_tensor.to(model.device),
        max_new_tokens=2048
    )

    # 解码（去除输入部分）
    result = tokenizer.decode(
        outputs[0][input_tensor.shape[1]:],
        skip_special_tokens=True
    )
    return result
```

#### 4.4.2 多轮对话推理

```python
def deepseek_multi_conversation_inference(model_path, prompt, chat_history):
    """多轮对话推理"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(...)

    messages = chat_history.copy()  # 保留历史
    messages.append({"role": "user", "content": prompt})

    input_tensor = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    outputs = model.generate(input_tensor.to(model.device), max_new_tokens=2048)

    result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)

    # 更新对话历史
    chat_history.append({"role": "user", "content": prompt})
    chat_history.append({"role": "assistant", "content": result})

    return result, chat_history
```

---

## 5. 难点与解决方案

### 5.1 难点一：多轮对话序列化

**问题**：原始数据是多轮对话列表，如何转换为连续的 token 序列？

**解决**：遍历每轮对话，按顺序拼接

```
原始: [{role:user, content:A}, {role:assistant, content:B}, {role:user, content:C}]
转换: <|bos|>system<|user|>A<|assistant|>B<|user|>C<|assistant|>
```

### 5.2 难点二：Label Masking

**问题**：如何只让模型学习"回答"而忽略"问题"？

**解决**：使用 `-100` 屏蔽用户输入的 loss

```python
labels = (
    [-100] * len(input_ids) +      # 用户输入不计算 loss
    output_ids +                   # 助手回复计算 loss
    [eos_token_id]
)
```

### 5.3 难点三：显存不足

**问题**：7B 模型全量加载需要 ~28GB 显存

**解决**：采用多种显存优化技术

| 技术                       | 显存节省 | 效果                 |
| -------------------------- | -------- | -------------------- |
| `torch.float16`          | ~50%     | 精度略有下降，可接受 |
| `gradient_checkpointing` | ~30%     | 增加约 20% 计算时间  |
| `gradient_accumulation`  | -        | 等效大 batch         |
| `device_map="auto"`      | -        | 智能分配到多卡       |

### 5.4 难点四：LoRA 目标模块选择

**问题**：哪些层应该添加 LoRA？

**解决**：自动识别所有全连接层

```python
# 方法：遍历所有模块，找出 Linear 层
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        # 添加到 target_modules
```

### 5.5 难点五：模型合并后文件缺失

**问题**：合并后可能缺少 config.json 等配置文件

**解决**：显式复制非权重文件

```python
def copy_files_not_in_B(A_path, B_path):
    """将 A 路径中 B 路径没有的文件复制过去"""
    files_in_A = set(os.listdir(A_path))
    files_in_B = set(os.listdir(B_path))
    files_to_copy = files_in_A - files_in_B

    for file in files_to_copy:
        shutil.copy2(os.path.join(A_path, file), os.path.join(B_path, file))
```

---

## 6. 训练配置指南

### 6.1 超参数推荐配置

#### 场景一：快速实验（单卡 16GB）

```python
lora_config = LoraConfig(
    r=16,               # 较小的 rank
    lora_alpha=16,
    lora_dropout=0.1,
)

train_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # 补偿小 batch
    num_train_epochs=3,
    learning_rate=3e-4,              # 稍大的学习率
    fp16=True,
)
```

#### 场景二：生产训练（多卡 8×A6000）

```python
lora_config = LoraConfig(
    r=64,               # 更大的 rank
    lora_alpha=32,
    lora_dropout=0.05,
)

train_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=2e-5,
    fp16=True,
    ddp_find_unused_parameters=False,
)
```

### 6.2 学习率调度

```python
# 余弦退火
lr_scheduler_type="cosine"

# 带 warmup 的多项式衰减
lr_scheduler_type="polynomial"
warmup_ratio=0.1  # 前 10% 步预热
```

### 6.3 早停策略

```python
train_args = TrainingArguments(
    evaluation_strategy="steps",
    eval_steps=5000,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)
```

---

## 7. 模型部署与推理

### 7.1 部署步骤

```bash
# 1. 合并模型
python merge_model.py

# 2. 推理测试
python reasoning.py
```

### 7.2 推理示例

```python
from reasoning import deepseek_multi_conversation_inference

model_path = "./output/merged-model"
chat_history = []

# 第一轮
result, chat_history = deepseek_multi_conversation_inference(
    model_path,
    "医生，我最近总是感到很焦虑",
    chat_history
)
print(result)

# 第二轮（自动包含历史）
result, chat_history = deepseek_multi_conversation_inference(
    model_path,
    "我应该怎么办？",
    chat_history
)
print(result)
```

### 7.3 使用 Gradio 搭建 Demo

```python
import gradio as gr
from reasoning import deepseek_model_inference

def chat(prompt, history):
    result = deepseek_model_inference(model_path, prompt)
    return result

demo = gr.ChatInterface(fn=chat, title="🏥 医疗问诊助手")
demo.launch()
```

---

## 📚 扩展阅读

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [PEFT: Parameter-Efficient Fine-Tuning](https://github.com/huggingface/peft)
- [DeepSeek-LLM Technical Report](https://github.com/deepseek-ai/DeepSeek-LLM)

---

<p align="center">
  💡 如有问题，请提交 Issue 或联系维护者
</p>
