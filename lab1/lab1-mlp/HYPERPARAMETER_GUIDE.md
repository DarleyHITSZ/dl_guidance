# MLP 超参数调整指南

本指南详细介绍了如何在 MLP 项目中调整各种超参数，包括学习率、隐藏层结构、激活函数、优化器等。

## 1. 超参数概览

以下是可以调整的主要超参数：

| 超参数 | 描述 | 默认值 | 调整范围 |
|--------|------|--------|----------|
| 学习率 | 控制模型权重更新的步长 | 0.001 | 1e-5 到 1e-1 |
| 隐藏层 | 隐藏层的数量和每个层的神经元数 | [64, 32] | 1-5 层，每层 16-256 个神经元 |
| 激活函数 | 引入非线性的函数 | ReLU | ReLU, Sigmoid, Tanh, Linear |
| 优化器 | 用于更新权重的算法 | Adam | Adam, SGD, RMSprop |
| 批次大小 | 每次更新权重时使用的样本数 | 32 | 8-128 |
| 训练轮数 | 完整训练数据集的次数 | 100 | 50-1000 |
| 权重初始化 | 权重的初始值生成方法 | he | he, xavier, uniform |
| 测试集比例 | 用于测试的数据集比例 | 0.2 | 0.1-0.3 |
| 随机种子 | 确保结果可复现 | 42 | 任意整数 |

## 2. 超参数调整位置

### 2.1 数据加载超参数 (demo.py 第 17 行)

```python
# 加载和预处理数据集
loader = BostonHousingLoader(test_size=0.2, random_state=42)
```

- `test_size`: 测试集占总数据集的比例，范围 0.1-0.3
- `random_state`: 随机种子，用于确保结果可复现

### 2.2 优化器和学习率 (demo.py 第 21 行)

```python
model = MLP(loss=MSE(), optimizer=Adam(learning_rate=0.001))
```

- `learning_rate`: 学习率，控制权重更新的步长
- 优化器选择：可以替换为 `SGD`, `RMSprop` 等

### 2.3 隐藏层结构和权重初始化 (demo.py 第 22-27 行)

```python
model.add(Dense(13, 64, weight_init="he"))
model.add(ReLU())
model.add(Dense(64, 32, weight_init="he"))
model.add(ReLU())
model.add(Dense(32, 1, weight_init="he"))
model.add(Linear())
```

- 隐藏层数量：通过添加或删除 `Dense` 层来调整
- 每层神经元数：修改 `Dense` 层的第二个参数
- 权重初始化：`weight_init` 参数，可选值为 "he", "xavier", "uniform"
- 激活函数：可以替换为 `Sigmoid`, `Tanh` 等

### 2.4 训练超参数 (demo.py 第 39-45 行)

```python
history = model.train(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    X_val=X_test,
    y_val=y_test,
    verbose=True
)
```

- `epochs`: 训练轮数，完整训练数据集的次数
- `batch_size`: 每次更新权重时使用的样本数
- `verbose`: 是否显示训练过程，`True` 或 `False`

## 3. 超参数调整示例

以下是一个完整的示例，展示如何调整多个超参数：

```python
# 调整后的数据加载
loader = BostonHousingLoader(test_size=0.15, random_state=123)

# 调整后的模型构建
model = MLP(loss=MSE(), optimizer=SGD(learning_rate=0.01, momentum=0.9))
model.add(Dense(13, 128, weight_init="xavier"))  # 增加第一层神经元数
model.add(ReLU())
model.add(Dense(128, 64, weight_init="xavier"))  # 增加第二层神经元数
model.add(ReLU())
model.add(Dense(64, 32, weight_init="xavier"))   # 增加第三层
model.add(ReLU())
model.add(Dense(32, 1, weight_init="xavier"))
model.add(Linear())

# 调整后训练参数
history = model.train(
    X_train, y_train,
    epochs=200,          # 增加训练轮数
    batch_size=16,       # 减小批次大小
    X_val=X_test,
    y_val=y_test,
    verbose=True
)
```

## 4. 超参数对模型性能的影响

### 4.1 学习率

- **太小**: 模型训练缓慢，需要更多轮次才能收敛
- **太大**: 可能导致模型发散，损失函数值震荡或增加
- **建议**: 从 0.001 开始，根据训练曲线调整

### 4.2 隐藏层结构

- **太少/太小**: 模型可能欠拟合，无法捕捉数据中的复杂模式
- **太多/太大**: 模型可能过拟合，在测试集上表现不佳
- **建议**: 从 2-3 层开始，每层 32-128 个神经元

### 4.3 激活函数

- **ReLU**: 解决了梯度消失问题，训练速度快
- **Sigmoid/Tanh**: 可能导致梯度消失，适合二分类输出层
- **建议**: 隐藏层使用 ReLU，输出层根据任务选择

### 4.4 优化器

- **Adam**: 自适应学习率，训练稳定，适合大多数任务
- **SGD**: 可能需要更长时间训练，但有时能找到更好的最小值
- **RMSprop**: 适合处理非平稳目标
- **建议**: 先使用 Adam，再尝试其他优化器

### 4.5 批次大小

- **太小**: 训练不稳定，梯度估计噪声大
- **太大**: 内存消耗大，训练速度慢，可能陷入局部最小值
- **建议**: 32 或 64，根据可用内存调整

### 4.6 训练轮数

- **太少**: 模型训练不充分，欠拟合
- **太多**: 模型过拟合，在测试集上表现下降
- **建议**: 使用早停法或监控验证集性能

## 5. 超参数调优策略

1. **网格搜索**: 尝试超参数的不同组合
2. **随机搜索**: 随机选择超参数组合，通常更高效
3. **贝叶斯优化**: 使用概率模型指导搜索
4. **早停法**: 当验证集性能不再提升时停止训练

## 6. 调整建议

1. 每次只调整少数几个超参数，以便观察其影响
2. 使用验证集评估超参数性能，而不是测试集
3. 保持随机种子不变，确保结果可复现
4. 记录所有实验结果，以便比较不同超参数组合
5. 从简单模型开始，逐步增加复杂度

## 7. 高级调整

### 7.1 权重初始化

- **he**: 适合 ReLU 激活函数
- **xavier**: 适合 Sigmoid 和 Tanh 激活函数
- **uniform**: 简单的均匀分布初始化

### 7.2 正则化

如果模型过拟合，可以考虑添加正则化：

```python
# 在 Dense 层中添加 L2 正则化
model.add(Dense(13, 64, weight_init="he", l2_reg=0.01))
```

### 7.3 学习率衰减

随着训练进行逐渐减小学习率：

```python
# 使用学习率衰减的 SGD 优化器
optimizer = SGD(learning_rate=0.01, momentum=0.9, decay=0.001)
```

---

通过合理调整超参数，可以显著提高模型性能。建议结合实际数据集和任务需求，系统地进行超参数调优。
