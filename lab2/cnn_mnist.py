import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os

# 设置TensorFlow镜像源以加速下载
os.environ['TFDS_DATA_DIR'] = 'D:/tensorflow_datasets'

# ------------------------------
# 激活函数
# ------------------------------
class ReLU:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Softmax:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        # 数值稳定的softmax实现
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        self.out = exp_x / sum_exp_x
        return self.out
    
    def backward(self, dout):
        # softmax的反向传播通常与交叉熵损失结合使用，这里单独实现
        dx = self.out * dout
        sum_dx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sum_dx
        return dx

# ------------------------------
# 损失函数
# ------------------------------
class CrossEntropyLoss:
    def __init__(self):
        self.y = None
        self.t = None
        self.batch_size = None
    
    def forward(self, y, t):
        self.y = y
        self.t = t
        self.batch_size = y.shape[0]
        # 处理one-hot编码
        if t.ndim == 1:
            t = np.eye(10)[t]
        self.t = t
        
        # 数值稳定的交叉熵计算
        y_clipped = np.clip(y, 1e-15, 1 - 1e-15)
        loss = -np.sum(t * np.log(y_clipped)) / self.batch_size
        return loss
    
    def backward(self):
        # 交叉熵与softmax结合的反向传播简化
        dx = (self.y - self.t) / self.batch_size
        return dx

# ------------------------------
# 卷积层
# ------------------------------
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 权重和偏置初始化（He初始化）
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.b = np.zeros(out_channels)
        
        # 用于反向传播的梯度
        self.dW = None
        self.db = None
        self.x = None
        self.col = None
        self.col_W = None
    
    def im2col(self, x, kernel_size, stride, padding):
        """将输入图像转换为列格式以高效计算卷积"""
        N, C, H, W = x.shape
        out_h = (H + 2 * padding - kernel_size) // stride + 1
        out_w = (W + 2 * padding - kernel_size) // stride + 1
        
        # 填充
        img = np.pad(x, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')
        col = np.zeros((N, C, kernel_size, kernel_size, out_h, out_w))
        
        for y in range(kernel_size):
            y_max = y + stride * out_h
            for x_ in range(kernel_size):
                x_max = x_ + stride * out_w
                col[:, :, y, x_, :, :] = img[:, :, y:y_max:stride, x_:x_max:stride]
        
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        return col
    
    def col2im(self, col, x_shape, kernel_size, stride, padding):
        """将列格式转换回图像格式"""
        N, C, H, W = x_shape
        out_h = (H + 2 * padding - kernel_size) // stride + 1
        out_w = (W + 2 * padding - kernel_size) // stride + 1
        col = col.reshape(N, out_h, out_w, C, kernel_size, kernel_size).transpose(0, 3, 4, 5, 1, 2)
        
        img = np.zeros((N, C, H + 2 * padding + stride - 1, W + 2 * padding + stride - 1))
        for y in range(kernel_size):
            y_max = y + stride * out_h
            for x_ in range(kernel_size):
                x_max = x_ + stride * out_w
                img[:, :, y:y_max:stride, x_:x_max:stride] += col[:, :, y, x_, :, :]
        
        return img[:, :, padding:H + padding, padding:W + padding]
    
    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        
        # 将权重转换为列格式
        col_W = self.W.reshape(self.out_channels, -1).T
        
        # 将输入转换为列格式
        col = self.im2col(x, self.kernel_size, self.stride, self.padding)
        
        # 计算卷积结果
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, (H + 2 * self.padding - self.kernel_size) // self.stride + 1, 
                         (W + 2 * self.padding - self.kernel_size) // self.stride + 1, self.out_channels)
        out = out.transpose(0, 3, 1, 2)  # (N, C_out, H_out, W_out)
        
        self.col = col
        self.col_W = col_W
        return out
    
    def backward(self, dout):
        N, C_out, H_out, W_out = dout.shape
        
        # 调整dout形状以匹配前向传播
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, C_out)
        
        # 计算偏置梯度
        self.db = np.sum(dout, axis=0)
        
        # 计算权重梯度
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        
        # 计算输入梯度
        dcol = np.dot(dout, self.col_W.T)
        dx = self.col2im(dcol, self.x.shape, self.kernel_size, self.stride, self.padding)
        
        return dx

# ------------------------------
# 池化层
# ------------------------------
class MaxPool2D:
    def __init__(self, pool_size, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.x = None
        self.arg_max = None
    
    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1
        
        # 将输入转换为列格式
        col = self.im2col(x, self.pool_size, self.stride, 0)
        col = col.reshape(-1, self.pool_size * self.pool_size)
        
        # 计算最大池化
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        
        self.arg_max = arg_max
        return out
    
    def backward(self, dout):
        N, C, H_out, W_out = dout.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, 1)
        
        # 初始化梯度为0
        dcol = np.zeros((dout.size, self.pool_size * self.pool_size))
        dcol[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        
        # 将列格式转换回图像格式
        dcol = dcol.reshape(N, H_out, W_out, C, self.pool_size, self.pool_size)
        dcol = dcol.transpose(0, 3, 4, 5, 1, 2)
        
        dx = self.col2im(dcol, self.x.shape, self.pool_size, self.stride, 0)
        return dx
    
    def im2col(self, x, kernel_size, stride, padding):
        """池化层使用的im2col实现"""
        N, C, H, W = x.shape
        out_h = (H + 2 * padding - kernel_size) // stride + 1
        out_w = (W + 2 * padding - kernel_size) // stride + 1
        
        img = np.pad(x, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')
        col = np.zeros((N, C, kernel_size, kernel_size, out_h, out_w))
        
        for y in range(kernel_size):
            y_max = y + stride * out_h
            for x_ in range(kernel_size):
                x_max = x_ + stride * out_w
                col[:, :, y, x_, :, :] = img[:, :, y:y_max:stride, x_:x_max:stride]
        
        col = col.transpose(0, 2, 3, 4, 5, 1).reshape(N * kernel_size * kernel_size * out_h * out_w, C)
        return col
    
    def col2im(self, col, x_shape, kernel_size, stride, padding):
        """池化层使用的col2im实现"""
        N, C, H, W = x_shape
        out_h = (H + 2 * padding - kernel_size) // stride + 1
        out_w = (W + 2 * padding - kernel_size) // stride + 1
        
        col = col.reshape(N, kernel_size, kernel_size, out_h, out_w, C).transpose(0, 5, 1, 2, 3, 4)
        img = np.zeros((N, C, H + 2 * padding + stride - 1, W + 2 * padding + stride - 1))
        
        for y in range(kernel_size):
            y_max = y + stride * out_h
            for x_ in range(kernel_size):
                x_max = x_ + stride * out_w
                img[:, :, y:y_max:stride, x_:x_max:stride] += col[:, :, y, x_, :, :]
        
        return img[:, :, padding:H + padding, padding:W + padding]

# ------------------------------
# 全连接层
# ------------------------------
class Dense:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        
        # 权重和偏置初始化（Xavier初始化）
        self.W = np.random.randn(in_features, out_features) * np.sqrt(1.0 / in_features)
        self.b = np.zeros(out_features)
        
        # 用于反向传播的梯度
        self.dW = None
        self.db = None
        self.x = None
        self.out = None
    
    def forward(self, x):
        self.x = x
        self.out = np.dot(x, self.W) + self.b
        return self.out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

# ------------------------------

# ------------------------------
# 展平层
# ------------------------------
class Flatten:
    def __init__(self):
        self.input_shape = None
    
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, dout):
        return dout.reshape(self.input_shape)

# ------------------------------
# CNN模型
# ------------------------------
class CNN:
    def __init__(self, learning_rate=0.001, weight_decay=1e-4, patience=5, max_grad_norm=1.0):
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate  # 保存初始学习率用于调度
        self.weight_decay = weight_decay  # L2正则化强度
        self.patience = patience  # 早停策略的容忍轮次
        self.max_grad_norm = max_grad_norm  # 梯度裁剪的最大范数
        self.layers = []
        self.loss_layer = None
        self.accuracy = []
        self.val_accuracy = []
        self.loss = []
        self.val_loss = []
        self.best_val_loss = float('inf')
        self.early_stop_count = 0
        self.best_params = None
    
    def add(self, layer):
        """添加层到模型"""
        self.layers.append(layer)
    
    def set_loss(self, loss_layer):
        """设置损失函数"""
        self.loss_layer = loss_layer
    
    def forward(self, x):
        """前向传播"""
        for layer in self.layers:
            if hasattr(layer, 'forward'):
                x = layer.forward(x)
        return x
    
    def backward(self):
        """反向传播，添加梯度裁剪"""
        # 反向传播损失梯度
        dout = self.loss_layer.backward()
        
        # 反向传播通过所有层，并应用梯度裁剪
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
            # 对权重梯度进行裁剪
            if hasattr(layer, 'dW') and layer.dW is not None:
                layer.dW = self.clip_gradient(layer.dW, self.max_grad_norm)
    
    def clip_gradient(self, grad, max_norm=1.0):
        """梯度裁剪，防止梯度爆炸"""
        norm = np.linalg.norm(grad)
        if norm > max_norm:
            return grad * (max_norm / norm)
        return grad
    
    def update(self):
        """更新权重，添加L2正则化"""
        for layer in self.layers:
            if hasattr(layer, 'W'):
                # 添加L2正则化到权重更新
                layer.W -= self.learning_rate * (layer.dW + self.weight_decay * layer.W)
                layer.b -= self.learning_rate * layer.db  # 偏置一般不做正则化
    
    def train(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=32, verbose=True):
        """训练模型，添加早停策略和学习率调度"""
        num_samples = x_train.shape[0]
        num_batches = num_samples // batch_size
        
        # 重置早停计数器和最佳损失
        self.best_val_loss = float('inf')
        self.early_stop_count = 0
        self.best_params = None
        
        for epoch in range(epochs):
            # 学习率调度：阶梯式衰减 (每10轮衰减一半)
            self.learning_rate = self.initial_lr * (0.5 ** (epoch // 10))
            
            start_time = time.time()
            epoch_loss = 0.0
            correct = 0
            
            # 打乱训练数据
            indices = np.random.permutation(num_samples)
            x_train_shuffled = x_train[indices]
            y_train_shuffled = y_train[indices]
            
            for batch_idx in range(num_batches):
                # 获取批量数据
                start = batch_idx * batch_size
                end = start + batch_size
                x_batch = x_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                
                # 前向传播
                y_pred = self.forward(x_batch)
                
                # 计算损失
                loss = self.loss_layer.forward(y_pred, y_batch)
                epoch_loss += loss
                
                # 计算准确率
                predictions = np.argmax(y_pred, axis=1)
                batch_correct = np.sum(predictions == y_batch)
                correct += batch_correct
                
                # 反向传播和权重更新
                self.backward()
                self.update()
                
                if verbose and (batch_idx + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{num_batches}, Loss: {loss:.4f}, LR: {self.learning_rate:.6f}")
            
            # 计算平均损失和准确率
            avg_loss = epoch_loss / num_batches
            train_acc = correct / num_samples
            
            # 评估验证集
            val_loss, val_acc = self.evaluate(x_val, y_val, batch_size)
            
            # 保存指标
            self.loss.append(avg_loss)
            self.accuracy.append(train_acc)
            self.val_loss.append(val_loss)
            self.val_accuracy.append(val_acc)
            
            # 早停策略检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stop_count = 0
                # 保存最佳模型参数
                self.best_params = {}
                for i, layer in enumerate(self.layers):
                    if hasattr(layer, 'W'):
                        self.best_params[f'layer{i}_W'] = layer.W.copy()
                        self.best_params[f'layer{i}_b'] = layer.b.copy()
            else:
                self.early_stop_count += 1
                if self.early_stop_count >= self.patience:
                    print(f"\n早停策略触发！在第 {epoch+1} 轮停止训练")
                    # 加载最佳模型参数
                    for i, layer in enumerate(self.layers):
                        if hasattr(layer, 'W'):
                            layer.W = self.best_params[f'layer{i}_W']
                            layer.b = self.best_params[f'layer{i}_b']
                    break
            
            epoch_time = time.time() - start_time
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - LR: {self.learning_rate:.6f} - Time: {epoch_time:.2f}s")
    
    def evaluate(self, x_test, y_test, batch_size=32):
        """评估模型"""
        num_samples = x_test.shape[0]
        num_batches = num_samples // batch_size
        total_loss = 0.0
        correct = 0
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            x_batch = x_test[start:end]
            y_batch = y_test[start:end]
            
            # 前向传播
            y_pred = self.forward(x_batch)
            
            # 计算损失
            loss = self.loss_layer.forward(y_pred, y_batch)
            total_loss += loss
            
            # 计算准确率
            predictions = np.argmax(y_pred, axis=1)
            correct += np.sum(predictions == y_batch)
        
        avg_loss = total_loss / num_batches
        accuracy = correct / num_samples
        return avg_loss, accuracy
    
    def predict(self, x):
        """预测"""
        y_pred = self.forward(x)
        return np.argmax(y_pred, axis=1)

# ------------------------------
# 数据加载与预处理
# ------------------------------
def load_mnist(reduce_data=False, reduce_factor=10):
    """加载MNIST数据集并进行预处理"""
    print("正在加载MNIST数据集...")
    start_time = time.time()
    
    # 加载数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # 数据预处理
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    # 添加通道维度 (N, H, W) -> (N, C, H, W)
    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)
    
    # 减少数据量以便快速测试
    if reduce_data:
        x_train = x_train[:x_train.shape[0] // reduce_factor]
        y_train = y_train[:y_train.shape[0] // reduce_factor]
        x_test = x_test[:x_test.shape[0] // reduce_factor]
        y_test = y_test[:y_test.shape[0] // reduce_factor]
    
    load_time = time.time() - start_time
    print(f"数据集加载完成，耗时 {load_time:.2f}s")
    print(f"训练集: {x_train.shape[0]} samples, 测试集: {x_test.shape[0]} samples")
    
    return x_train, y_train, x_test, y_test

# ------------------------------
# 可视化功能
# ------------------------------
def plot_training_curves(history):
    """绘制训练曲线"""
    epochs = range(1, len(history['loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_predictions(model, x_test, y_test, num_samples=10):
    """可视化预测结果"""
    indices = np.random.choice(x_test.shape[0], num_samples, replace=False)
    x_sample = x_test[indices]
    y_sample = y_test[indices]
    y_pred = model.predict(x_sample)
    
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(x_sample[i, 0], cmap='gray')
        plt.title(f"True: {y_sample[i]}\nPred: {y_pred[i]}")
        plt.axis('off')
        # 根据预测结果设置不同颜色
        if y_sample[i] == y_pred[i]:
            plt.gca().set_title(f"True: {y_sample[i]}\nPred: {y_pred[i]}", color='green')
        else:
            plt.gca().set_title(f"True: {y_sample[i]}\nPred: {y_pred[i]}", color='red')
    
    plt.tight_layout()
    plt.show()

# ------------------------------
# 主函数
# ------------------------------
def main():
    # 设置参数
    reduce_data = True  # 是否减少数据量以便快速测试
    reduce_factor = 10  # 减少因子
    epochs = 25
    batch_size = 32
    learning_rate = 0.001
    
    # 加载数据集
    x_train, y_train, x_test, y_test = load_mnist(reduce_data=reduce_data, reduce_factor=reduce_factor)
    
    # 创建模型
    model = CNN(learning_rate=learning_rate)
    
    # 添加层（移除BatchNorm层）
    model.add(Conv2D(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=1))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=2, stride=2))
    model.add(Conv2D(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=1))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=2, stride=2))
    model.add(Flatten())
    model.add(Dense(in_features=32 * 5 * 5, out_features=128))  # 卷积后输出尺寸为5×5：32通道 × 5×5 = 800
    model.add(ReLU())
    model.add(Dense(in_features=128, out_features=10))
    model.add(Softmax())
    
    # 设置损失函数
    model.set_loss(CrossEntropyLoss())
    
    # 训练模型
    print("开始训练模型...")
    train_start_time = time.time()
    model.train(x_train, y_train, x_test, y_test, epochs=epochs, batch_size=batch_size)
    train_end_time = time.time()
    
    print(f"\n训练完成，总耗时: {train_end_time - train_start_time:.2f}s")
    
    # 评估模型
    print("\n评估模型性能...")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    
    # 输出训练结果
    print("\n训练结果:")
    print("-" * 50)
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}: Loss={model.loss[epoch]:.4f}, Acc={model.accuracy[epoch]:.4f}, Val Loss={model.val_loss[epoch]:.4f}, Val Acc={model.val_accuracy[epoch]:.4f}")
    print("-" * 50)
    
    # 可视化结果
    history = {
        'loss': model.loss,
        'accuracy': model.accuracy,
        'val_loss': model.val_loss,
        'val_accuracy': model.val_accuracy
    }
    
    print("\n绘制训练曲线...")
    plot_training_curves(history)
    
    print("\n可视化预测结果...")
    visualize_predictions(model, x_test, y_test)

if __name__ == "__main__":
    main()