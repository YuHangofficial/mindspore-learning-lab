# MindSpore Learning Lab / MindSpore 学习实验室

A beginner-friendly AI infrastructure learning project based on MindSpore.

一个基于 MindSpore 的 AI 基础设施入门学习项目。

---

## Project Introduction / 项目简介

This repository is a hands-on learning project for understanding fundamental AI infrastructure concepts through MindSpore.

这个仓库是一个实践型学习项目，用于通过 MindSpore 理解 AI 基础设施中的核心概念。

The project focuses on tensor computation and common neural network operators, which are foundational concepts for AI frameworks, inference systems, and operator optimization.

项目聚焦于 Tensor（张量计算）和常见神经网络算子，这些内容是 AI 框架、推理系统和算子优化的基础。

---

## Features / 功能内容

Implemented operator demonstrations including:

实现了以下经典算子演示：

- Tensor creation / Tensor 创建
- Tensor shape inspection / Tensor 结构查看
- Element-wise addition / 逐元素加法
- Element-wise multiplication / 逐元素乘法
- Matrix multiplication / 矩阵乘法
- ReLU activation operator / ReLU 激活算子
- Softmax probability output / Softmax 概率输出

---

## Tech Stack / 技术栈

- Python
- MindSpore
- NumPy
- Linux (WSL)

---

## Project Structure / 项目结构

```text
mindspore-learning-lab/
├── tensor_basic.py
├── operators_demo.py
└── README.md
```

- `tensor_basic.py`  
  Basic Tensor creation and shape exploration  
  Tensor 创建与 shape 基础探索

- `operators_demo.py`  
  Common AI operator demonstrations  
  常见 AI 算子演示

---

## Why This Project / 为什么做这个项目

This project is part of my learning path toward AI infrastructure engineering.

这个项目是我学习 AI 基础设施工程过程中的实践项目。

It helps me understand:

它帮助我理解：

- how tensor computation works / Tensor 计算如何工作
- how neural network operators are executed / 神经网络算子如何执行
- how AI frameworks organize computation / AI 框架如何组织计算
- foundations for future Huawei Ascend / CANN ecosystem learning / 为后续学习华为 Ascend / CANN 生态打基础

---

## How to Run / 如何运行

Create virtual environment and install dependencies:

创建虚拟环境并安装依赖：

```bash
python3 -m venv ms-env
source ms-env/bin/activate
pip install mindspore numpy
```

Run:

运行：

```bash
python tensor_basic.py
python operators_demo.py
```

---

## Learning Status / 当前状态

This is an ongoing learning project.

这是一个持续迭代中的学习项目。

Future plans:

后续计划：

- more MindSpore operator exploration
- simple neural network implementation
- inference pipeline experiments
- Huawei AI infrastructure ecosystem exploration

即：

- 更多 MindSpore 算子实践
- 简单神经网络实现
- 推理流程实验
- 华为 AI Infra 生态探索
