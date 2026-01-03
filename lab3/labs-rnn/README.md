# 基于NumPy的Mamba与MinGRU序列模型实现

## 1. 项目简介

本项目是一个基于纯NumPy实现的序列模型库，聚焦于MinGRU（简化版GRU）与Mamba（基于状态空间模型SSM）两种算法，用于解决股票价格时间序列预测任务。通过纯NumPy实现模型的前向传播、反向传播及参数更新，深入理解序列模型的底层工作机制，对比传统RNN类模型与新型SSM模型的性能差异。

### 核心功能
- **纯NumPy实现**：不依赖任何深度学习框架，完全通过NumPy实现模型核心功能
- **两种序列模型**：
  - MinGRU：简化版GRU模型，保持核心门控机制
  - Mamba：基于状态空间模型的新型序列模型，具有线性时间复杂度
- **完整训练流程**：数据下载、预处理、模型训练、验证和评估
- **性能对比**：同时训练两种模型并提供详细的性能对比报告
- **可视化支持**：训练曲线、预测结果、性能对比可视化

## 2. 环境搭建

### 系统要求
- Python 3.8+
- Windows/macOS/Linux

### 依赖安装

1. 创建并激活虚拟环境（推荐）：

```bash
# Windows
python -m venv venv
env\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

2. 安装依赖库：

```bash
pip install -r requirements.txt
```

### 依赖版本说明
- numpy>=1.24.0
- pandas>=2.0.0
- yfinance>=0.2.0
- scikit-learn>=1.3.0
- matplotlib>=3.4.0

## 3. 快速开始

### 单模型训练

使用`test_model.py`脚本训练单个模型（MinGRU或Mamba）：

```bash
python test_model.py
```

**配置模型类型**：
在`test_model.py`文件中修改`model_type`参数：

```python
HYPERPARAMS = {
    'model_type': 'min_gru',  # 'min_gru' 或 'mamba'
    # 其他超参数...
}
```

### 模型性能对比

使用`benchmark.py`脚本同时训练两种模型并生成对比报告：

```bash
python benchmark.py
```

执行完成后，将生成：
- `mamba_vs_mingru_comparison.png`：性能对比可视化图
- `benchmark_results.csv`：详细的训练结果数据

## 4. 项目结构

```
labs-rnn/
├── models/                # 模型核心实现目录
│   ├── min_gru.py         # MinGRU模型类（前向/反向传播、参数更新）
│   └── mamba.py           # Mamba模型类（状态空间运算、选择性扫描机制）
├── utils/                 # 工具函数目录
│   └── data_loader.py     # 数据加载、预处理、批次创建工具类
├── test_model.py          # 单模型训练与测试脚本
├── benchmark.py           # 模型性能对比脚本
├── requirements.txt       # 依赖库清单
├── README.md              # 项目简介、快速启动指南
└── EXPERIMENT_GUIDE.md    # 详细实验指南
```

## 5. 超参数配置

### 通用超参数（test_model.py和benchmark.py）

```python
HYPERPARAMS = {
    'seq_len': 30,          # 序列长度
    'batch_size': 32,        # 批次大小
    'hidden_size': 128,      # 隐藏层大小
    'learning_rate': 0.001,  # 学习率
    'epochs': 10,            # 训练轮数
    'stock_symbol': 'AAPL',  # 股票代码
    'feature_cols': ['Close'] # 使用的特征列
}
```

### Mamba专用超参数

```python
HYPERPARAMS = {
    'state_size': 64,        # 状态空间大小
    'kernel_size': 4,        # 卷积核大小
    # 其他通用超参数...
}
```

## 6. 评估指标

模型训练和测试过程中使用以下评估指标：

- **均方误差（MSE）**：预测值与真实值之差的平方的平均值
- **均方根误差（RMSE）**：MSE的平方根
- **平均绝对误差（MAE）**：预测值与真实值之差的绝对值的平均值
- **决定系数（R²）**：模型对数据的解释能力

## 7. 常见问题

### 数据下载失败

**问题**：运行脚本时出现yfinance下载失败的错误

**解决方案**：
1. 检查网络连接
2. 尝试修改`start_date`和`end_date`参数
3. 手动创建`data`目录：`mkdir data`

### 维度不匹配错误

**问题**：运行时出现"shape mismatch"错误

**解决方案**：
1. 检查输入数据的维度是否正确
2. 确保`seq_len`、`batch_size`等超参数配置合理
3. 检查数据预处理过程中的维度转换

### 梯度爆炸

**问题**：训练过程中损失值变得极大或NaN

**解决方案**：
1. 减小学习率
2. 增加批次大小
3. 尝试使用梯度裁剪（可在模型的`update`方法中添加）

### 训练速度慢

**问题**：纯NumPy实现导致训练速度较慢

**解决方案**：
1. 减小`hidden_size`或`state_size`
2. 增加`batch_size`
3. 减少训练轮数
4. 在支持CUDA的环境中可考虑使用CuPy替代NumPy

## 8. 扩展与改进

### 功能扩展
- 添加更多序列模型（如LSTM、Transformer）
- 支持多变量时间序列预测
- 实现更复杂的优化算法（如Adam、RMSprop）
- 添加早停机制防止过拟合

### 性能优化
- 实现向量化操作替代循环
- 增加并行计算支持
- 添加CuPy支持以利用GPU加速

### 应用扩展
- 支持其他时间序列预测任务（如天气预测、交通流量预测）
- 添加模型保存和加载功能
- 实现在线学习功能

## 9. 许可证

本项目采用MIT许可证。

## 10. 联系方式

如有问题或建议，欢迎提交Issue或Pull Request。