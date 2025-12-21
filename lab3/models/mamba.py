import numpy as np

class Mamba:
    def __init__(self, input_size, hidden_size, output_size, state_size=64, kernel_size=4, dropout_rate=0.0):
        """
        初始化Mamba模型（基于状态空间模型SSM）
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            output_size: 输出维度
            state_size: 状态空间维度（默认64）
            kernel_size: 卷积核大小（预留扩展）
            dropout_rate: Dropout概率（默认0.0，禁用Dropout）
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.state_size = state_size
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        
        # 权重初始化（遵循文档的Xavier初始化）
        self.W_in = np.random.randn(hidden_size, input_size) * np.sqrt(2.0 / input_size)
        self.A = np.ones((state_size, 1)) * -1.0  # 状态衰减系数（固定为负）
        self.B = np.random.randn(hidden_size, state_size) * np.sqrt(2.0 / hidden_size)
        self.C = np.random.randn(hidden_size, state_size) * np.sqrt(2.0 / state_size)
        self.D = np.random.randn(hidden_size, 1) * 0.1
        self.W_out = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.b_out = np.zeros((output_size, 1))
        
        # 保存中间结果（用于反向传播）
        self.x = None
        self.x_proj = None
        self.gate_history = None
        self.s_history = None
        self.y = None
        self.output = None
        self.dropout_mask = None
        
        # 保存历史梯度（调试用）
        self.grad_history = []
        
        # 初始化梯度（调用修复后的reset_grads方法）
        self.reset_grads()
    
    def silu(self, x):
        """SiLU激活函数（数值稳定版）"""
        return x * (1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))))
    
    def silu_deriv(self, x):
        """SiLU激活函数的导数"""
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return sigmoid + x * sigmoid * (1 - sigmoid)
    
    def reset_grads(self):
        """关键修复：实现梯度重置方法，初始化所有参数梯度为0"""
        self.dW_in = np.zeros_like(self.W_in)
        self.dB = np.zeros_like(self.B)
        self.dC = np.zeros_like(self.C)
        self.dD = np.zeros_like(self.D)
        self.dW_out = np.zeros_like(self.W_out)
        self.db_out = np.zeros_like(self.b_out)
        
        # 清空历史梯度（调试用）
        if hasattr(self, 'grad_history'):
            self.grad_history.clear()
    
    def dropout(self, x, training=True):
        """Dropout层（训练时随机失活，验证时禁用）"""
        if not training or self.dropout_rate == 0.0:
            return x
        self.dropout_mask = (np.random.rand(*x.shape) > self.dropout_rate).astype(np.float32)
        return x * self.dropout_mask / (1 - self.dropout_rate)
    
    def forward(self, x, training=True):
        """前向传播（处理整个输入序列）"""
        seq_len, input_size, batch_size = x.shape
        assert input_size == self.input_size, f"输入维度不匹配：期望{self.input_size}，实际{input_size}"
        
        self.x = x
        self.x_proj = np.zeros((seq_len, self.hidden_size, batch_size))
        for t in range(seq_len):
            self.x_proj[t] = np.clip(np.dot(self.W_in, x[t]), -5, 5)
        
        s = np.zeros((self.state_size, batch_size))
        self.s_history = []
        self.gate_history = []
        self.y = np.zeros((seq_len, self.hidden_size, batch_size))
        
        for t in range(seq_len):
            # 选通门计算 + Dropout
            gate = self.silu(self.x_proj[t])
            gate = self.dropout(gate, training=training)
            gate = np.clip(gate, -3, 3)
            self.gate_history.append(gate.copy())
            
            # 状态更新
            s = s * np.exp(self.A) + np.clip(np.dot(self.B.T, gate), -1, 1)
            self.s_history.append(s.copy())
            
            # 隐藏层输出 + Dropout
            c_s = np.clip(np.dot(self.C, s), -3, 3)
            d_x = self.D * self.x_proj[t]
            hidden_out = gate * (c_s + d_x)
            hidden_out = self.dropout(hidden_out, training=training)
            self.y[t] = np.clip(hidden_out, -5, 5)
        
        self.s_history = np.array(self.s_history)
        self.gate_history = np.array(self.gate_history)
        
        # 最终输出投影
        self.output = np.zeros((seq_len, self.output_size, batch_size))
        for t in range(seq_len):
            self.output[t] = np.clip(np.dot(self.W_out, self.y[t]) + self.b_out, -5, 5)
        
        return self.output
    
    def backward(self, dout):
        """反向传播（计算所有参数的梯度）"""
        seq_len, output_size, batch_size = dout.shape
        assert output_size == self.output_size, f"输出梯度维度不匹配：期望{self.output_size}，实际{output_size}"
        assert hasattr(self, 'output'), "请先运行forward()进行前向传播"
        
        # 保存梯度历史（调试用）
        grad_snapshot = {
            'dW_out': self.dW_out.copy(),
            'db_out': self.db_out.copy(),
            'dC': self.dC.copy(),
            'dB': self.dB.copy(),
            'dW_in': self.dW_in.copy(),
            'dD': self.dD.copy()
        }
        self.grad_history.append(grad_snapshot)
        
        dy = np.zeros((seq_len, self.hidden_size, batch_size))
        ds_next = np.zeros((self.state_size, batch_size))
        dx_proj = np.zeros((seq_len, self.hidden_size, batch_size))
        dx = np.zeros_like(self.x)
        
        # 从后往前传播梯度
        for t in reversed(range(seq_len)):
            # 输出层梯度
            self.dW_out += np.dot(np.clip(dout[t], -1, 1), self.y[t].T)
            self.db_out += np.clip(np.sum(dout[t], axis=1, keepdims=True), -1, 1)
            dy[t] = np.clip(np.dot(self.W_out.T, dout[t]), -1, 1)
            
            # 隐藏层梯度分解
            gate_t = self.gate_history[t]
            c_s_t = np.dot(self.C, self.s_history[t])
            d_x_proj_t = self.D * self.x_proj[t]
            csd_t = c_s_t + d_x_proj_t
            
            d_gate_t = np.clip(dy[t] * csd_t, -1, 1)
            d_csd_t = np.clip(dy[t] * gate_t, -1, 1)
            
            # 状态空间梯度
            self.dC += np.dot(d_csd_t, self.s_history[t].T)
            ds_t = np.clip(np.dot(self.C.T, d_csd_t) + ds_next * np.exp(self.A), -1, 1)
            self.dB += np.dot(gate_t, ds_t.T)
            ds_next = ds_t
            
            # 输入投影梯度
            d_gate_xproj = np.clip(self.silu_deriv(self.x_proj[t]) * d_gate_t, -1, 1)
            d_d_xproj = np.clip(self.D * d_csd_t, -1, 1)
            dx_proj[t] = d_gate_xproj + d_d_xproj
            
            # D参数梯度
            self.dD += np.clip(np.sum(d_csd_t * self.x_proj[t], axis=1, keepdims=True), -0.01, 0.01)
        
        # W_in梯度
        for t in range(seq_len):
            self.dW_in += np.dot(dx_proj[t], self.x[t].T)
            dx[t] = np.dot(self.W_in.T, dx_proj[t])
        
        # 梯度裁剪
        self.dW_in = np.clip(self.dW_in, -0.1, 0.1)
        self.dB = np.clip(self.dB, -0.1, 0.1)
        self.dC = np.clip(self.dC, -0.1, 0.1)
        self.dD = np.clip(self.dD, -0.1, 0.1)
        self.dW_out = np.clip(self.dW_out, -0.1, 0.1)
        
        return dx
    
    def update(self, lr, weight_decay=0.0):
        """参数更新（带L2正则化）"""
        # 应用L2正则化
        self.dW_in += weight_decay * self.W_in
        self.dB += weight_decay * self.B
        self.dC += weight_decay * self.C
        self.dD += weight_decay * self.D
        self.dW_out += weight_decay * self.W_out
        
        # 参数更新与裁剪
        self.W_in = np.clip(self.W_in - lr * self.dW_in, -1, 1)
        self.B = np.clip(self.B - lr * self.dB, -1, 1)
        self.C = np.clip(self.C - lr * self.dC, -1, 1)
        self.D = np.clip(self.D - lr * self.dD, -0.1, 0.1)
        self.W_out = np.clip(self.W_out - lr * self.dW_out, -1, 1)
        self.b_out = np.clip(self.b_out - lr * self.db_out, -0.1, 0.1)
        
        # 重置梯度
        self.reset_grads()

# 测试代码（验证模型可初始化）
if __name__ == "__main__":
    model = Mamba(input_size=1, hidden_size=64, output_size=1)
    print("Mamba模型初始化成功！")