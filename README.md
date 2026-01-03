# 深度学习实验指南（dl_guidance）

## 项目概述
本项目包含一系列深度学习实验实现，涵盖多层感知机（MLP）、卷积神经网络（CNN）、Transformer及变体（如Mamba）等经典模型，旨在帮助理解深度学习核心算法的原理与实现细节。

## 目录结构
```
dl_guidance/
├── LICENSE               # 许可证信息
├── lab1/                 # MLP实验
│   └── lab1-mlp/
│       ├── mlp/          # MLP层实现
│       ├── requirements.txt  # 依赖列表
│       └── README.md     # MLP实验说明
├── lab2/                 # CNN实验
│   └── cnn_mnist.py      # 基于MNIST的CNN实现
├── lab4/                 # Transformer及变体实验
│   └── lab4-transformer/
│       ├── mamba_model.py  # Mamba块实现
│       ├── transformer_model.py  # Transformer编码器实现
│       ├── download_dataset.py   # 数据集下载脚本
│       ├── tokenizers.py         # 分词器工具
│       └── EXPERIMENT_GUIDE.md   # 实验指南
```

## 环境要求
### 基础依赖
- Python 3.7+
- 核心库：
  ```
  numpy>=1.21
  pandas
  scikit-learn
  matplotlib
  ```
### 特定实验依赖
- lab2（CNN）：`tensorflow`
- lab4（Transformer/Mamba）：`torch`、`json`（内置）

可通过以下命令安装基础依赖：
```bash
pip install -r lab1/lab1-mlp/requirements.txt
```

## 实验内容

### 1. MLP实验（lab1）
- 基于NumPy实现多层感知机（MLP）
- 数据集：波士顿房价数据集（来自OpenML）
- 包含自定义层实现（`mlp/layers.py`），支持前向传播、反向传播及参数更新

### 2. CNN实验（lab2）
- 基于NumPy实现卷积神经网络（CNN），用于MNIST手写数字识别
- 包含核心组件：
  - 激活函数（ReLU、Softmax）
  - 损失函数（交叉熵）
  - 卷积层、池化层、全连接层
  - 自定义`im2col`/`col2im`实现高效卷积计算

### 3. Transformer及变体实验（lab4）
- 实现Transformer编码器及Mamba模型
- 支持量化实验，非量化模型训练命令：
  ```bash
  python training_and_testing.py --device cpu --model transformer --pos_encoding sine --tokenizer bpe --name non_quantized
  ```
- 数据集处理：支持数据集下载与预处理（`download_dataset.py`）
- 分词器：支持BPE分词器保存与加载（`tokenizers.py`）

## 致谢
- 波士顿房价数据集来自OpenML
- 实验实现参考了多个在线深度学习教程与资源
- 部分模型实现基于PyTorch框架

## 许可证
本项目遵循开源许可证条款，详情参见[LICENSE](LICENSE)。贡献者需遵守许可证中关于贡献、版权许可及专利许可的相关规定。
