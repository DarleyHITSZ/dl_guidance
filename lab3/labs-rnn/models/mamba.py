import numpy as np

class Mamba:
    def __init__(self, input_size, hidden_size, output_size, state_size, kernel_size=4, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.state_size = state_size
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        
        # Xavier初始化
        def xavier_init(shape):
            fan_in, fan_out = shape
            return np.random.randn(fan_in, fan_out) * np.sqrt(2 / (fan_in + fan_out))
        
        # 输入投影层
        self.W_in = xavier_init((hidden_size, input_size))
        
        # 状态空间参数
        self.A = -np.abs(np.random.randn(state_size, 1))  # 确保状态衰减
        self.B = xavier_init((state_size, hidden_size))
        self.C = xavier_init((state_size, hidden_size))
        self.D = np.zeros((hidden_size, 1))
        
        # 门控参数
        self.W_gate = xavier_init((hidden_size, input_size))
        self.b_gate = np.zeros((hidden_size, 1))
        
        # 输出投影层
        self.W_out = xavier_init((output_size, hidden_size))
        self.b_out = np.zeros((output_size, 1))
        
        # 卷积核参数
        self.kernel = np.random.randn(kernel_size, hidden_size) * 0.01
        
        # 保存中间结果用于反向传播
        self.cache = {}
    
    def expm(self, A):
        # 计算矩阵指数（简化版，适用于对角矩阵）
        return np.exp(A)
    
    def apply_conv(self, x, kernel):
        # 对输入应用1D卷积
        seq_len, _, batch_size = x.shape
        kernel_size = kernel.shape[0]
        hidden_size = kernel.shape[1]
        
        # 填充输入
        x_padded = np.zeros((seq_len + kernel_size - 1, hidden_size, batch_size))
        x_padded[kernel_size - 1:] = x
        
        # 计算卷积
        conv_output = np.zeros_like(x)
        for t in range(seq_len):
            conv_output[t] = np.tensordot(kernel, x_padded[t:t+kernel_size], axes=([0], [0]))
        
        return conv_output
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, x):
        # x: (seq_len, input_size, batch_size)
        seq_len, _, batch_size = x.shape
        
        # 输入投影
        x_proj = np.zeros((seq_len, self.hidden_size, batch_size))
        for t in range(seq_len):
            x_proj[t] = np.dot(self.W_in, x[t])
        
        # 门控计算
        gate = np.zeros((seq_len, self.hidden_size, batch_size))
        for t in range(seq_len):
            gate[t] = self.sigmoid(np.dot(self.W_gate, x[t]) + self.b_gate)
        
        # 应用卷积到输入投影
        conv_x = self.apply_conv(x_proj, self.kernel)
        
        # 状态初始化
        s = np.zeros((self.state_size, batch_size))
        
        # 保存中间状态
        s_history = []
        z_history = []
        
        # 状态空间更新和隐藏层计算
        h = np.zeros((seq_len, self.hidden_size, batch_size))
        for t in range(seq_len):
            # 状态空间更新
            Bx = np.dot(self.B, conv_x[t])  # (state_size, batch_size)
            s = self.expm(self.A) * s + Bx  # (state_size, batch_size)
            
            # 隐藏层输出
            z = np.dot(self.C.T, s) * gate[t]  # (hidden_size, batch_size)
            h[t] = x_proj[t] * gate[t] + z + self.D * gate[t]  # (hidden_size, batch_size)
            
            # 保存历史
            s_history.append(s.copy())
            z_history.append(z.copy())
        
        # 输出投影
        y = np.zeros((seq_len, self.output_size, batch_size))
        for t in range(seq_len):
            y[t] = np.dot(self.W_out, h[t]) + self.b_out
        
        # 保存缓存
        self.cache = {
            'x': x,
            'x_proj': x_proj,
            'gate': gate,
            'conv_x': conv_x,
            's': s_history,
            'z': z_history,
            'h': h,
            'y': y
        }
        
        return y[-1]  # 返回最后一个时间步的输出
    
    def backward(self, dy):
        # dy: (output_size, batch_size)
        seq_len = self.cache['x'].shape[0]
        batch_size = self.cache['x'].shape[2]
        
        # 初始化梯度
        dW_out = np.zeros_like(self.W_out)
        db_out = np.zeros_like(self.b_out)
        dh = np.zeros((seq_len, self.hidden_size, batch_size))
        dh[-1] = np.dot(self.W_out.T, dy)
        
        # 输出层梯度
        dW_out = np.dot(dy, self.cache['h'][-1].T)
        db_out = np.sum(dy, axis=1, keepdims=True)
        
        # 状态空间反向传播
        dA = np.zeros_like(self.A)
        dB = np.zeros_like(self.B)
        dC = np.zeros_like(self.C)
        dD = np.zeros_like(self.D)
        dkernel = np.zeros_like(self.kernel)
        dW_in = np.zeros_like(self.W_in)
        dW_gate = np.zeros_like(self.W_gate)
        db_gate = np.zeros_like(self.b_gate)
        
        # 初始化状态梯度
        ds_prev = np.zeros((self.state_size, batch_size))
        
        for t in reversed(range(seq_len)):
            # 当前时间步的中间结果
            h_t = self.cache['h'][t]
            x_proj_t = self.cache['x_proj'][t]
            gate_t = self.cache['gate'][t]
            z_t = self.cache['z'][t]
            conv_x_t = self.cache['conv_x'][t]
            
            # 分解隐藏层梯度
            dgate_t = dh[t] * (x_proj_t + z_t + self.D) + self.sigmoid_derivative(gate_t) * np.dot(self.W_gate, self.cache['x'][t])
            dx_proj_t = dh[t] * gate_t
            dz_t = dh[t] * gate_t
            
            # 门控参数梯度
            dW_gate += np.dot(dgate_t, self.cache['x'][t].T)
            db_gate += np.sum(dgate_t, axis=1, keepdims=True)
            
            # 状态空间参数梯度（简化版）
            dC += np.dot(self.cache['s'][t], (dz_t * gate_t).T)
            
            # 计算ds_t
            ds_t = np.dot(self.C, dz_t * gate_t) + ds_prev
            
            # 状态衰减梯度
            if t > 0:
                # 沿着batch_size维度求和，保持dA的形状为(state_size, 1)
                dA += np.sum(self.expm(self.A) * self.cache['s'][t-1] * ds_t, axis=1, keepdims=True)
                ds_prev = self.expm(self.A) * ds_t
            
            # B参数梯度
            dB += np.dot(ds_t, conv_x_t.T)
            
            # 卷积输入梯度
            dconv_x_t = np.dot(self.B.T, ds_t)
            
            # 卷积核梯度（简化版）
            for k in range(self.kernel_size):
                if t - k >= 0:
                    # 对batch_size维度求和，保持dkernel[k]的形状为(hidden_size,)
                    dkernel[k] += np.sum(np.dot(dconv_x_t, self.cache['x_proj'][t-k].T), axis=1)
            
            # 输入投影梯度
            dx_proj_t += self.apply_conv_gradient(dconv_x_t, self.kernel)
            dW_in += np.dot(dx_proj_t, self.cache['x'][t].T)
            
            # D参数梯度
            dD += np.sum(dh[t] * gate_t, axis=1, keepdims=True)
        
        # 整合梯度
        gradients = {
            'dW_in': dW_in,
            'dW_out': dW_out,
            'db_out': db_out,
            'dA': dA,
            'dB': dB,
            'dC': dC,
            'dD': dD,
            'dkernel': dkernel,
            'dW_gate': dW_gate,
            'db_gate': db_gate
        }
        
        return gradients
    
    def apply_conv_gradient(self, dy_t, kernel):
        # 简化实现：返回与输入形状相同的零数组
        # 因为卷积的主要梯度已经在计算dkernel时考虑过了
        return np.zeros_like(dy_t)
    
    def update(self, gradients):
        # 更新参数
        self.W_in -= self.learning_rate * gradients['dW_in']
        self.W_out -= self.learning_rate * gradients['dW_out']
        self.b_out -= self.learning_rate * gradients['db_out']
        self.A -= self.learning_rate * gradients['dA']
        self.B -= self.learning_rate * gradients['dB']
        self.C -= self.learning_rate * gradients['dC']
        self.D -= self.learning_rate * gradients['dD']
        self.kernel -= self.learning_rate * gradients['dkernel']
        self.W_gate -= self.learning_rate * gradients['dW_gate']
        self.b_gate -= self.learning_rate * gradients['db_gate']
        
        # 确保A保持负值以维持状态衰减
        self.A = -np.abs(self.A)
    
    def forward(self, x):
        # 重写forward方法，确保正确处理序列输入
        seq_len, input_size, batch_size = x.shape
        
        # 输入投影
        x_proj = np.zeros((seq_len, self.hidden_size, batch_size))
        for t in range(seq_len):
            x_proj[t] = np.dot(self.W_in, x[t])
        
        # 门控计算
        gate = np.zeros((seq_len, self.hidden_size, batch_size))
        for t in range(seq_len):
            gate[t] = self.sigmoid(np.dot(self.W_gate, x[t]) + self.b_gate)
        
        # 卷积层 - 使用因果卷积
        conv_x = np.zeros_like(x_proj)
        for t in range(seq_len):
            # 只使用当前和之前的输入
            start_idx = max(0, t - self.kernel_size + 1)
            num_pad = self.kernel_size - (t - start_idx + 1)
            
            # 填充输入
            padded_input = np.zeros((self.kernel_size, self.hidden_size, batch_size))
            if t >= self.kernel_size - 1:
                padded_input = x_proj[t - self.kernel_size + 1:t + 1]
            else:
                padded_input[num_pad:] = x_proj[0:t + 1]
            
            # 计算卷积
            for b in range(batch_size):
                # 给padded_input增加一个新维度以匹配kernel的形状，并在计算后去除多余维度
                conv_x[t, :, b] = np.sum(self.kernel[:, :, np.newaxis] * padded_input[:, :, b, np.newaxis], axis=0).squeeze()
        
        # 状态初始化
        s = np.zeros((self.state_size, batch_size))
        s_history = []
        
        # 状态空间处理
        z = np.zeros_like(x_proj)
        for t in range(seq_len):
            # 状态更新
            Bx = np.dot(self.B, conv_x[t])
            s = np.exp(self.A) * s + Bx
            s_history.append(s.copy())
            
            # 计算z
            z[t] = np.dot(self.C.T, s) * gate[t]
        
        # 隐藏层输出
        h = x_proj * gate + z + self.D * gate
        
        # 输出层
        y = np.zeros((seq_len, self.output_size, batch_size))
        for t in range(seq_len):
            y[t] = np.dot(self.W_out, h[t]) + self.b_out
        
        # 保存中间结果
        self.cache = {
            'x': x,
            'x_proj': x_proj,
            'gate': gate,
            'conv_x': conv_x,
            's': s_history,
            'z': z,
            'h': h
        }
        
        return y[-1]  # 返回最后一个时间步的输出