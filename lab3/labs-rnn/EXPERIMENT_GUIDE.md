# Mamba与MinGRU序列模型实验指南

## 1. 实验背景

时间序列预测是机器学习中的重要任务，在金融、气象、交通等领域有广泛应用。传统的循环神经网络（RNN）及其变体（如LSTM、GRU）在处理长序列时存在梯度消失和计算效率低的问题。近年来，基于状态空间模型（State Space Models, SSM）的新型架构如Mamba展现出了优异的性能，在保持线性时间复杂度的同时，能够有效捕捉长序列中的依赖关系。

### MinGRU与Mamba的核心差异

- **MinGRU**：简化版GRU模型，保留了门控循环单元的核心机制（更新门和重置门），通过控制信息流动来捕捉序列依赖关系。
- **Mamba**：基于状态空间模型的序列模型，通过选择性扫描机制（selective scanning）处理输入，具有线性时间复杂度和线性内存占用，能够高效处理极长序列。

## 2. 实验目标

本实验的主要目标是：

1. **实现理解**：通过纯NumPy实现两种序列模型，深入理解它们的底层工作机制
2. **性能对比**：对比MinGRU和Mamba在股票价格预测任务上的性能差异
3. **数据分析**：学习如何处理时间序列数据，包括下载、预处理、序列构建等
4. **模型调试**：掌握序列模型训练过程中的常见问题及解决方案
5. **参数调优**：理解不同超参数对模型性能的影响

## 3. 详细实验步骤

### 3.1 环境验证

首先验证Python环境和依赖库是否正确安装：

```bash
# 检查Python版本
python --version

# 检查依赖库版本
pip list | grep -E "numpy|pandas|yfinance|scikit-learn|matplotlib"
```

预期输出：
```
Python 3.8.x 或更高版本
numpy>=1.24.0
pandas>=2.0.0
yfinance>=0.2.0
scikit-learn>=1.3.0
matplotlib>=3.4.0
```

### 3.2 数据下载与预处理

执行以下步骤下载和预处理股票数据：

1. **运行数据加载测试**：

```python
from utils.data_loader import DataLoader

# 初始化数据加载器
data_loader = DataLoader(
    stock_symbol='AAPL',
    start_date='2010-01-01',
    end_date='2023-12-31',
    seq_len=30,
    batch_size=32
)

# 下载和预处理数据
data = data_loader.load_and_preprocess(feature_cols=['Close'])

# 检查数据形状
print(f"训练数据形状: X_train={data['X_train'].shape}, y_train={data['y_train'].shape}")
print(f"测试数据形状: X_test={data['X_test'].shape}, y_test={data['y_test'].shape}")
print(f"训练批次数量: {len(data['X_train_batches'])}")
```

2. **数据可视化**：

```python
import matplotlib.pyplot as plt
import numpy as np

# 获取原始数据
raw_data = data_loader.download_data()
close_prices = raw_data['Close'].values

# 绘制收盘价曲线
plt.figure(figsize=(12, 6))
plt.plot(close_prices)
plt.title('AAPL Stock Price (2010-2023)')
plt.xlabel('Time Step')
plt.ylabel('Close Price')
plt.grid(True)
plt.savefig('aapl_price.png')
plt.close()
```

### 3.3 单模型训练

#### 3.3.1 训练MinGRU模型

1. 修改`test_model.py`中的模型类型：

```python
HYPERPARAMS = {
    'model_type': 'min_gru',
    # 其他超参数保持默认
}
```

2. 执行训练脚本：

```bash
python test_model.py
```

3. 观察训练过程，记录以下信息：
   - 每轮训练损失
   - 验证损失
   - 训练时间
   - 最终评估指标

#### 3.3.2 训练Mamba模型

1. 修改`test_model.py`中的模型类型：

```python
HYPERPARAMS = {
    'model_type': 'mamba',
    # 其他超参数保持默认
}
```

2. 执行训练脚本：

```bash
python test_model.py
```

3. 同样记录训练过程信息。

### 3.4 模型性能对比

执行`benchmark.py`脚本同时训练两种模型并生成对比报告：

```bash
python benchmark.py
```

该脚本将：
- 使用相同的数据集和超参数训练两种模型
- 记录训练时间、损失值等信息
- 生成性能对比可视化图
- 保存详细的训练结果到CSV文件

### 3.5 参数调优实验

尝试修改以下超参数，观察对模型性能的影响：

1. **学习率**：尝试0.0001, 0.001, 0.01
2. **隐藏层大小**：尝试32, 64, 128, 256
3. **序列长度**：尝试10, 30, 60, 120
4. **批次大小**：尝试16, 32, 64
5. **训练轮数**：尝试5, 10, 20

**示例**：修改学习率和隐藏层大小

```python
HYPERPARAMS = {
    'learning_rate': 0.01,
    'hidden_size': 64,
    # 其他超参数...
}
```

## 4. 结果分析方法

### 4.1 损失曲线分析

- **训练损失**：观察损失是否稳定下降，是否存在梯度爆炸或消失
- **验证损失**：与训练损失对比，判断是否过拟合
- **收敛速度**：比较两种模型达到相同损失所需的训练轮数

### 4.2 性能指标分析

主要关注以下指标：

- **MSE（均方误差）**：反映预测值与真实值的平均平方误差
- **RMSE（均方根误差）**：MSE的平方根，单位与原始数据一致
- **MAE（平均绝对误差）**：反映预测值与真实值的平均绝对误差
- **R²（决定系数）**：模型对数据的解释能力，越接近1越好

### 4.3 训练效率分析

- **训练时间**：比较两种模型的总训练时间和每轮训练时间
- **内存占用**：观察训练过程中的内存使用情况
- **计算复杂度**：理论上Mamba应具有线性时间复杂度优势

### 4.4 预测结果可视化分析

观察预测曲线与真实曲线的拟合程度：
- 短期预测准确性
- 长期趋势捕捉能力
- 对突变点的响应

## 5. 扩展实验建议

### 5.1 模型改进

1. **MinGRU改进**：
   - 添加梯度裁剪防止梯度爆炸
   - 实现多层MinGRU结构
   - 添加dropout正则化

2. **Mamba改进**：
   - 实现更高效的状态空间更新算法
   - 添加门控机制的可学习参数
   - 尝试不同的卷积核大小和状态空间维度

### 5.2 数据扩展

1. **多特征预测**：
   ```python
   feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
   ```

2. **多股票预测**：
   ```python
   stock_symbol = 'GOOGL'  # 或其他股票代码
   ```

3. **其他时间序列任务**：
   - 气象数据预测
   - 交通流量预测
   - 电力负荷预测

### 5.3 架构扩展

1. **模型融合**：结合MinGRU和Mamba的优势
2. **注意力机制**：在模型中添加注意力层
3. **编码器-解码器结构**：实现序列到序列预测

## 6. 调试技巧

### 6.1 形状检查

在模型训练过程中，经常检查张量形状是否正确：

```python
# 在forward方法中添加
print(f"x shape: {x.shape}")
print(f"h shape: {h.shape}")
print(f"y shape: {y.shape}")
```

### 6.2 梯度检查

检查梯度是否存在异常（如NaN、无穷大）：

```python
# 在backward方法中添加
print(f"Gradient norm: {np.linalg.norm(gradients['dW_z'])}")
print(f"Gradient max: {np.max(gradients['dW_z'])}")
print(f"Gradient min: {np.min(gradients['dW_z'])}")
```

### 6.3 小数据集测试

在调试模型时，使用小数据集进行测试：

```python
# 在DataLoader中添加
self.test_split = 0.9  # 只使用10%的数据进行测试
```

### 6.4 单步调试

使用Python调试器或在关键位置添加断点：

```python
# 在训练循环中添加
import pdb; pdb.set_trace()
```

### 6.5 损失监控

在训练过程中实时监控损失值：

```python
# 在训练循环中添加
if (epoch + 1) % 1 == 0:
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.6f}")
    plt.plot(epoch+1, loss, 'ro')
    plt.pause(0.001)
```

### 6.6 数值稳定性

处理指数运算时注意数值稳定性：

```python
# 使用稳定的指数计算
np.exp(x, out=np.zeros_like(x), where=x < 100)  # 避免溢出
```

## 7. 常见问题及解决方案

### 7.1 数据下载失败

**问题**：
```
Exception in thread Thread-1: HTTPSConnectionPool(host='query1.finance.yahoo.com', port=443)
```

**解决方案**：
1. 检查网络连接
2. 尝试修改yfinance的下载参数
3. 手动下载数据并保存到data目录

### 7.2 梯度爆炸

**问题**：
```
Loss: inf
Gradient norm: inf
```

**解决方案**：
1. 减小学习率
2. 添加梯度裁剪
3. 使用更稳定的初始化方法
4. 增加批次大小

### 7.3 模型不收敛

**问题**：
```
Loss: 0.1234
Loss: 0.1235
Loss: 0.1233
# 损失几乎不变
```

**解决方案**：
1. 增大学习率
2. 检查数据预处理是否正确
3. 调整隐藏层大小
4. 检查反向传播实现是否正确

### 7.4 过拟合

**问题**：
```
Train Loss: 0.0012
Val Loss: 0.0567
```

**解决方案**：
1. 增加正则化（如dropout）
2. 减小模型复杂度
3. 增加训练数据
4. 提前停止训练

## 8. 实验总结

完成实验后，总结以下内容：

1. **模型实现**：两种模型的核心机制及实现难点
2. **性能对比**：两种模型在各指标上的表现差异
3. **参数影响**：不同超参数对模型性能的影响
4. **问题解决**：实验过程中遇到的主要问题及解决方案
5. **改进方向**：模型可以进一步改进的地方

通过本实验，您将深入理解序列模型的工作原理，掌握时间序列预测任务的完整流程，并学会如何分析和改进序列模型。