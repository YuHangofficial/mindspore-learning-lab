# MindSpore Learning Lab

一个面向 AI Infrastructure 入门学习的 MindSpore 实践项目。  
A beginner-friendly MindSpore practice project for learning AI infrastructure fundamentals.

---

## 项目简介 | Project Overview

### 中文

本项目是我围绕 **AI Infra / AI 算子技术** 方向搭建的学习型仓库，主要用于理解 AI 框架中的基础计算过程。

项目基于 **Linux / WSL + Python + MindSpore + NumPy**，从最基础的 Tensor 表示开始，逐步实践常见 AI Operator、最小推理流程以及基础训练机制。

当前重点不是追求复杂模型效果，而是理解：

```text
Tensor 如何表示数据
Operator 如何组织计算
Inference Pipeline 如何完成分类
Training 如何通过 Loss / Gradient / Optimizer 更新参数
```

---

### English

This repository is a hands-on learning project for understanding AI infrastructure fundamentals with **MindSpore**.

It starts from basic Tensor representation and gradually explores common AI operators, a minimal inference pipeline, and a simple training mechanism with NumPy.

The goal is not to build a complex model, but to understand how AI computation is organized:

```text
Tensor → Operator → Inference Pipeline → Training Fundamentals
```

---

## 技术栈 | Tech Stack

- Python
- NumPy
- MindSpore
- Linux / WSL
- Git / GitHub

---

## 项目结构 | Project Structure

```text
mindspore-learning-lab/
├── tensor_basic.py
├── operators_demo.py
├── mini_classifier.py
├── train_one_weight.py
└── README.md
```

| 文件 | 作用 |
|---|---|
| `tensor_basic.py` | Tensor 创建与 shape 理解 |
| `operators_demo.py` | 常见 AI Operator 演示 |
| `mini_classifier.py` | 最小分类推理流程 demo |
| `train_one_weight.py` | NumPy 手搓单参数训练 demo |
| `README.md` | 项目说明文档 |

---

## 学习内容 | What I Learned

### 1. Tensor 基础 | Tensor Basics

通过 `tensor_basic.py` 理解：

- Tensor 是 AI 框架中的核心数据结构
- Tensor 可以看作多维数组
- shape 表示 Tensor 的结构和维度

示例：

```python
x = Tensor(np.array([[1, 2], [3, 4]]), ms.float32)
print(x.shape)
```

输出：

```text
(2, 2)
```

---

### 2. AI Operator 基础 | AI Operators

通过 `operators_demo.py` 演示常见算子：

- Add：逐元素加法
- Multiply：逐元素乘法
- MatMul：矩阵乘法
- ReLU：激活函数
- Softmax：分类概率输出
- ArgMax：选择最大概率类别

这些 Operator 是神经网络计算的基本组成单元。

---

### 3. 最小推理流程 | Mini Inference Pipeline

通过 `mini_classifier.py` 构建一个最小分类器，模拟神经网络推理流程：

```text
Input Tensor
↓
MatMul
↓
ReLU
↓
MatMul
↓
Softmax
↓
ArgMax
↓
Classification Result
```

该 demo 帮助理解：

- 推理不是魔法，而是一连串 Tensor 和 Operator 的计算
- Softmax 将原始分数转换为概率
- ArgMax 根据最大概率得到最终类别
- AI Infra 关注的是如何让这条计算链更快、更稳定、更高效地执行

---

### 4. 最小训练机制 | Minimal Training Demo

通过 `train_one_weight.py` 使用 NumPy 手动实现一个最小训练过程：

目标任务：

```text
学习 y = 2x
```

模型形式：

```text
y_pred = w * x
```

训练过程：

```text
Forward
↓
Loss
↓
Gradient
↓
Optimizer Update
↓
Repeat
```

该 demo 展示：

- Loss 如何衡量预测误差
- Gradient 如何指示参数调整方向
- Learning Rate 如何控制更新步长
- Optimizer 如何更新权重
- Training 与 Inference 的区别

训练结果中可以观察到：

```text
Weight → 2.0
Loss → 0
```

说明模型通过梯度下降学到了正确参数。

---

## 如何运行 | How to Run

### 1. 创建并激活虚拟环境 | Create Virtual Environment

```bash
python3 -m venv ms-env
source ms-env/bin/activate
```

---

### 2. 安装依赖 | Install Dependencies

```bash
pip install mindspore numpy
```

如果默认 PyPI 下载较慢，可以使用清华镜像：

```bash
pip install mindspore numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### 3. 运行示例 | Run Demos

```bash
python tensor_basic.py
python operators_demo.py
python mini_classifier.py
python train_one_weight.py
```

---

## 当前理解 | Current Understanding

通过这个项目，我初步理解了 AI 计算的基本链路：

```text
数据以 Tensor 形式进入模型
↓
模型由多个 Operator 组成
↓
推理阶段只进行 forward 计算
↓
训练阶段还需要 Loss、Gradient 和参数更新
↓
AI Infra 负责让这些计算在底层系统和硬件上高效执行
```

---

## 后续计划 | Future Plans

- [ ] 使用 MindSpore 构建更完整的简单神经网络
- [ ] 学习 ONNX 模型格式与推理部署流程
- [ ] 理解 Runtime、Compiler、Backend 等 AI Infra 概念
- [ ] 继续学习 Huawei Ascend / CANN 生态
- [ ] 尝试参与 MindSpore / Ascend 相关开源文档或示例贡献

---

## 项目定位 | Project Positioning

这是一个持续迭代的 AI Infra 学习项目。  
它记录了我从 Tensor、Operator、Inference 到 Training 基础机制的学习过程，也作为后续深入学习 Huawei Ascend / CANN 和 AI 算子技术的基础。

This is an ongoing AI infrastructure learning project and a foundation for my future exploration of Huawei Ascend / CANN and AI operator technology.
