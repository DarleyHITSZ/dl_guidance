import numpy as np

class MinGRU:
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化MinGRU模型
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重 (使用小随机值保证训练稳定性)
        self.W_z = np.random.randn(hidden_size, hidden_size + input_size) * 0.01  # 更新门权重
        self.W_r = np.random.randn(hidden_size, hidden_size + input_size) * 0.01  # 重置门权重
        self.W_h = np.random.randn(hidden_size, hidden_size + input_size) * 0.01  # 候选隐藏状态权重
        self.W_y = np.random.randn(output_size, hidden_size) * 0.01                # 输出层权重
        
        # 初始化偏置
        self.b_z = np.zeros((hidden_size, 1))  # 更新门偏置
        self.b_r = np.zeros((hidden_size, 1))  # 重置门偏置
        self.b_h = np.zeros((hidden_size, 1))  # 候选隐藏状态偏置
        self.b_y = np.zeros((output_size, 1))  # 输出层偏置
        
        # 保存中间变量（用于反向传播）
        self.h_prev = None       # 前一时间步隐藏状态
        self.combined = None     # 隐藏状态与输入的拼接结果
        self.z = None            # 更新门输出
        self.r = None            # 重置门输出
        self.combined_r = None   # 重置门作用后的隐藏状态与输入的拼接结果
        self.h_tilde = None      # 候选隐藏状态
        self.h = None            # 当前时间步隐藏状态
        
        # 重置梯度
        self.reset_grads()
    
    def sigmoid(self, x):
        """Sigmoid激活函数 (数值稳定版)"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        """Tanh激活函数 (数值稳定版)"""
        return np.tanh(np.clip(x, -500, 500))
    
    def reset_grads(self):
        """重置所有参数的梯度"""
        self.dW_z = np.zeros_like(self.W_z)
        self.dW_r = np.zeros_like(self.W_r)
        self.dW_h = np.zeros_like(self.W_h)
        self.dW_y = np.zeros_like(self.W_y)
        
        self.db_z = np.zeros_like(self.b_z)
        self.db_r = np.zeros_like(self.b_r)
        self.db_h = np.zeros_like(self.b_h)
        self.db_y = np.zeros_like(self.b_y)
    
    def forward_step(self, x, h_prev):
        """
        单时间步前向传播
        参数:
            x: 当前时间步输入 (input_size, batch_size)
            h_prev: 前一时间步隐藏状态 (hidden_size, batch_size)
        返回:
            y: 当前时间步输出 (output_size, batch_size)
            h: 当前时间步隐藏状态 (hidden_size, batch_size)
        """
        # 保存前一时间步隐藏状态（用于反向传播）
        self.h_prev = h_prev
        
        # 拼接隐藏状态和输入: (hidden_size + input_size, batch_size)
        self.combined = np.concatenate([h_prev, x], axis=0)
        
        # 1. 更新门计算: z = σ(W_z · [h_prev; x] + b_z)
        self.z = self.sigmoid(np.dot(self.W_z, self.combined) + self.b_z)
        
        # 2. 重置门计算: r = σ(W_r · [h_prev; x] + b_r)
        self.r = self.sigmoid(np.dot(self.W_r, self.combined) + self.b_r)
        
        # 3. 候选隐藏状态计算: h_tilde = tanh(W_h · [r⊙h_prev; x] + b_h)
        self.combined_r = np.concatenate([self.r * h_prev, x], axis=0)
        self.h_tilde = self.tanh(np.dot(self.W_h, self.combined_r) + self.b_h)
        
        # 4. 新隐藏状态计算: h = (1 - z)⊙h_prev + z⊙h_tilde
        self.h = (1 - self.z) * h_prev + self.z * self.h_tilde
        
        # 5. 输出计算: y = W_y · h + b_y
        y = np.dot(self.W_y, self.h) + self.b_y
        
        return y, self.h
    
    def forward(self, x_seq):
        """
        序列级前向传播（处理整个序列）
        参数:
            x_seq: 输入序列 (seq_len, input_size, batch_size)
        返回:
            y_seq: 输出序列 (seq_len, output_size, batch_size)
            h_seq: 隐藏状态序列 (seq_len, hidden_size, batch_size)
        """
        seq_len, input_size, batch_size = x_seq.shape
        
        # 初始化隐藏状态
        h_prev = np.zeros((self.hidden_size, batch_size))
        
        # 保存输出和隐藏状态序列
        y_seq = np.zeros((seq_len, self.output_size, batch_size))
        h_seq = np.zeros((seq_len, self.hidden_size, batch_size))
        
        # 逐个时间步处理
        for t in range(seq_len):
            y_t, h_t = self.forward_step(x_seq[t], h_prev)
            y_seq[t] = y_t
            h_seq[t] = h_t
            h_prev = h_t
        
        return y_seq, h_seq
    
    def backward_step(self, dy, dh_next):
        """
        单时间步反向传播
        参数:
            dy: 当前时间步输出梯度 (output_size, batch_size)
            dh_next: 下一时间步隐藏状态梯度 (hidden_size, batch_size)
        返回:
            dx: 当前时间步输入梯度 (input_size, batch_size)
            dh_prev: 前一时间步隐藏状态梯度 (hidden_size, batch_size)
        """
        # 1. 输出层梯度计算
        self.dW_y += np.dot(dy, self.h.T)
        self.db_y += np.sum(dy, axis=1, keepdims=True)
        
        # 2. 隐藏状态梯度（结合输出梯度和下一时间步隐藏状态梯度）
        dh = np.dot(self.W_y.T, dy) + dh_next
        
        # 3. 分解隐藏状态梯度
        dh_tilde = dh * self.z  # 候选隐藏状态梯度
        dz = dh * (self.h_prev - self.h_tilde)  # 更新门梯度
        
        # 4. 更新门相关梯度计算
        dz_sigmoid = self.z * (1 - self.z) * dz  # sigmoid导数
        self.dW_z += np.dot(dz_sigmoid, self.combined.T)
        self.db_z += np.sum(dz_sigmoid, axis=1, keepdims=True)
        
        # 5. 候选隐藏状态相关梯度计算
        dh_tilde_tanh = (1 - self.h_tilde ** 2) * dh_tilde  # tanh导数
        self.dW_h += np.dot(dh_tilde_tanh, self.combined_r.T)
        self.db_h += np.sum(dh_tilde_tanh, axis=1, keepdims=True)
        
        # 6. 重置门相关梯度计算
        dr_combined = np.dot(self.W_h.T, dh_tilde_tanh)
        dr = dr_combined[:self.hidden_size] * self.h_prev
        dr_sigmoid = self.r * (1 - self.r) * dr  # sigmoid导数
        self.dW_r += np.dot(dr_sigmoid, self.combined.T)
        self.db_r += np.sum(dr_sigmoid, axis=1, keepdims=True)
        
        # 7. 输入梯度和前一时间步隐藏状态梯度计算
        dx_combined = np.dot(self.W_z.T, dz_sigmoid) + np.dot(self.W_r.T, dr_sigmoid)
        dx = dx_combined[self.hidden_size:]  # 输入梯度（截取后input_size维）
        dh_prev = dx_combined[:self.hidden_size] + dh * (1 - self.z)  # 前一时间步隐藏状态梯度
        
        return dx, dh_prev
    
    def backward(self, x_seq, y_seq, target_seq, h_seq):
        """
        序列级反向传播（计算整个序列的梯度）
        参数:
            x_seq: 输入序列 (seq_len, input_size, batch_size)
            y_seq: 模型输出序列 (seq_len, output_size, batch_size)
            target_seq: 目标序列 (seq_len, output_size, batch_size)
            h_seq: 隐藏状态序列 (seq_len, hidden_size, batch_size)
        返回:
            dx_seq: 输入序列梯度 (seq_len, input_size, batch_size)
            total_loss: 序列总损失（MSE）
        """
        seq_len, output_size, batch_size = target_seq.shape
        
        # 初始化梯度
        self.reset_grads()
        dx_seq = np.zeros_like(x_seq)
        dh_next = np.zeros((self.hidden_size, batch_size))  # 初始下一时间步隐藏状态梯度为0
        total_loss = 0.0
        
        # 从后往前反向传播（时间步逆序）
        for t in reversed(range(seq_len)):
            # 计算MSE损失和输出梯度
            dy = y_seq[t] - target_seq[t]  # 输出梯度（MSE损失导数）
            loss_t = 0.5 * np.mean(np.square(dy))  # 单时间步MSE损失
            total_loss += loss_t
            
            # 单时间步反向传播
            dx_t, dh_prev = self.backward_step(dy, dh_next)
            dx_seq[t] = dx_t
            dh_next = dh_prev  # 传递梯度到前一时间步
        
        # 计算平均损失
        total_loss /= seq_len
        
        return dx_seq, total_loss
    
    def update(self, lr, weight_decay=0.0):
        """
        参数更新（随机梯度下降）
        参数:
            lr: 学习率
            weight_decay: L2正则化系数（可选，默认0）
        """
        # 应用L2正则化（权重衰减）
        self.dW_z += weight_decay * self.W_z
        self.dW_r += weight_decay * self.W_r
        self.dW_h += weight_decay * self.W_h
        self.dW_y += weight_decay * self.W_y
        
        # 更新权重和偏置
        self.W_z -= lr * self.dW_z
        self.W_r -= lr * self.dW_r
        self.W_h -= lr * self.dW_h
        self.W_y -= lr * self.dW_y
        
        self.b_z -= lr * self.db_z
        self.b_r -= lr * self.db_r
        self.b_h -= lr * self.db_h
        self.b_y -= lr * self.db_y
        
        # 重置梯度（为下一轮训练做准备）
        self.reset_grads()

# 测试代码（验证模型可运行）
if __name__ == "__main__":
    # 超参数设置
    input_size = 1
    hidden_size = 128
    output_size = 1
    seq_len = 30
    batch_size = 32
    lr = 0.01
    
    # 初始化模型
    model = MinGRU(input_size, hidden_size, output_size)
    
    # 生成随机测试数据
    x_test = np.random.randn(seq_len, input_size, batch_size)  # 输入序列
    target_test = np.random.randn(seq_len, output_size, batch_size)  # 目标序列
    
    # 前向传播
    y_pred, h_seq = model.forward(x_test)
    print(f"前向传播测试 - 输出形状: {y_pred.shape}, 隐藏状态形状: {h_seq.shape}")
    
    # 反向传播
    dx_seq, loss = model.backward(x_test, y_pred, target_test, h_seq)
    print(f"反向传播测试 - 输入梯度形状: {dx_seq.shape}, 损失值: {loss:.6f}")
    
    # 参数更新
    model.update(lr)
    print("参数更新完成")