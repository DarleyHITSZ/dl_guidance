# Transformer模型及变体对比实验（德英翻译任务）

## 项目简介

本项目是一个基于PyTorch的Transformer模型实验框架，聚焦德英翻译任务，旨在对比不同位置编码、分词器及模型结构对翻译性能的影响，并支持模型量化优化。

### 核心功能

- 实现原始Transformer及三类变体模型（Transformer+RoPE、Transformer+SentencePiece、Mamba）
- 支持多种位置编码：正弦位置编码、RoPE（旋转位置编码）、可学习位置编码
- 实现两类分词器：BPE（字节对编码）、SentencePiece（子词分词）
- 支持INT8量化优化，对比量化前后的性能与效率
- 自动下载与预处理WMT14德英翻译数据集
- 提供单模型训练测试和多模型对比实验功能

## 安装说明

### 环境要求

- Python 3.8~3.10
- PyTorch 1.10+
- NumPy 1.21+
- SentencePiece

### 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 基础训练演示

运行单个Transformer模型的训练与测试：

```bash
python training_and_testing.py --device cpu --model transformer --pos_encoding sine --tokenizer bpe
```

### 2. 对比实验演示

批量运行多种模型的对比实验：

```bash
python comparison_experiments.py
```

### 3. 量化实验演示

运行INT8量化训练：

```bash
python training_and_testing.py --quantize int8 --device cpu --model transformer
```

## 项目结构

```
lab4-transformer/
├── download_dataset.py        # 数据集下载、预处理与增强脚本
├── training_and_testing.py    # 单模型训练与测试脚本
├── transformer_model.py       # Transformer核心实现（含多位置编码）
├── mamba_model.py             # 简化版Mamba模型实现
├── tokenizers.py              # BPE、SentencePiece分词器实现
├── comparison_experiments.py  # 多模型对比实验脚本
├── experiment_results.json    # 实验结果存储文件（自动生成）
├── requirements.txt           # 依赖列表
├── README.md                  # 项目说明文档
└── EXPERIMENT_GUIDE.md        # 详细实验指南
```

## 核心模块说明

### 1. download_dataset.py
- 实现WMT14德英数据集的自动下载与预处理
- 提供数据增强功能，生成标准化训练和测试样本
- 支持断点续传和数据过滤

### 2. tokenizers.py
- 实现BPE和SentencePiece两种分词器
- 统一接口设计，便于切换使用
- 支持词汇表动态构建和保存/加载

### 3. transformer_model.py
- 实现Transformer核心组件：编码器、解码器、多头注意力、前馈网络
- 支持三种位置编码：正弦位置编码、RoPE、可学习位置编码
- 提供INT8量化版本

### 4. mamba_model.py
- 实现简化版Mamba模型
- 包含状态空间层和选择性扫描操作
- 与Transformer保持一致的接口

### 5. training_and_testing.py
- 单模型训练与测试的主脚本
- 支持命令行参数配置
- 输出训练日志和性能指标

### 6. comparison_experiments.py
- 多模型对比实验脚本
- 自动运行多种配置的实验
- 生成实验结果比较报告

## 常见问题

### 1. 内存不足

- 减小批量大小（--batch_size）
- 降低模型维度（--d_model）
- 减少层数（--num_layers）

### 2. 训练速度慢

- 使用GPU进行训练（--device cuda）
- 减小模型尺寸
- 减少训练轮次（--epochs）

### 3. 模型不收敛

- 调整学习率（--lr）
- 检查数据集质量
- 增加训练轮次
- 调整模型超参数

### 4. 量化效果不佳

- 确保使用支持量化的模型
- 增加校准数据集大小
- 检查量化配置

## 实验结果分析

实验结果保存在`experiment_results.json`文件中，包含以下关键指标：

- 训练损失和测试损失
- 训练时间
- 显存占用
- 推理延迟

可以使用这些指标对比不同模型配置的性能和效率。

## 扩展实验

### 自定义位置编码

在`transformer_model.py`中继承`PositionalEncoding`基类，实现自定义位置编码。

### 新增模型结构

参考`mamba_model.py`的实现，添加新的模型结构，保持与现有接口一致。

### 超参数调优

修改`training_and_testing.py`或`comparison_experiments.py`中的参数配置，进行超参数搜索。

## 许可证

本项目采用MIT许可证。
