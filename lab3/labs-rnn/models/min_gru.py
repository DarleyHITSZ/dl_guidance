import numpy as np

class MinGRU:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 权重初始化（方差0.01）
        self.W_z = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.W_r = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.W_h = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.W_y = np.random.randn(output_size, hidden_size) * 0.01
        
        # 偏置初始化（零偏置）
        self.b_z = np.zeros((hidden_size, 1))
        self.b_r = np.zeros((hidden_size, 1))
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        
        # 保存中间结果用于反向传播
        self.cache = {}
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - x**2
    
    def forward_step(self, x_t, h_prev):
        # x_t: (input_size, batch_size)
        # h_prev: (hidden_size, batch_size)
        
        # 拼接输入和前一隐藏状态
        combined = np.concatenate((x_t, h_prev), axis=0)  # (input_size + hidden_size, batch_size)
        
        # 更新门
        z_t = self.sigmoid(np.dot(self.W_z, combined) + self.b_z)  # (hidden_size, batch_size)
        
        # 重置门
        r_t = self.sigmoid(np.dot(self.W_r, combined) + self.b_r)  # (hidden_size, batch_size)
        
        # 候选隐藏状态
        r_h_prev = r_t * h_prev  # (hidden_size, batch_size)
        combined_h = np.concatenate((x_t, r_h_prev), axis=0)  # (input_size + hidden_size, batch_size)
        h_tilde = self.tanh(np.dot(self.W_h, combined_h) + self.b_h)  # (hidden_size, batch_size)
        
        # 新隐藏状态
        h_t = z_t * h_prev + (1 - z_t) * h_tilde  # (hidden_size, batch_size)
        
        # 输出
        y_t = np.dot(self.W_y, h_t) + self.b_y  # (output_size, batch_size)
        
        return h_t, y_t, z_t, r_t, h_tilde, combined, combined_h
    
    def forward(self, x):
        # x: (seq_len, input_size, batch_size)
        seq_len, _, batch_size = x.shape
        
        # 初始化隐藏状态
        h_prev = np.zeros((self.hidden_size, batch_size))
        
        # 保存每一步的结果
        h_list = []
        y_list = []
        z_list = []
        r_list = []
        h_tilde_list = []
        combined_list = []
        combined_h_list = []
        
        for t in range(seq_len):
            x_t = x[t]
            h_prev, y_t, z_t, r_t, h_tilde, combined, combined_h = self.forward_step(x_t, h_prev)
            
            h_list.append(h_prev)
            y_list.append(y_t)
            z_list.append(z_t)
            r_list.append(r_t)
            h_tilde_list.append(h_tilde)
            combined_list.append(combined)
            combined_h_list.append(combined_h)
        
        # 保存到缓存
        self.cache['x'] = x
        self.cache['h'] = h_list
        self.cache['y'] = y_list
        self.cache['z'] = z_list
        self.cache['r'] = r_list
        self.cache['h_tilde'] = h_tilde_list
        self.cache['combined'] = combined_list
        self.cache['combined_h'] = combined_h_list
        
        return y_list[-1], h_prev  # 返回最后一个时间步的输出和隐藏状态
    
    def backward_step(self, dy_t, dh_next, h_prev, h_t, z_t, r_t, h_tilde, combined, combined_h):
        # dy_t: (output_size, batch_size)
        # dh_next: (hidden_size, batch_size)
        
        # 输出层梯度
        dW_y = np.dot(dy_t, h_t.T)
        db_y = np.sum(dy_t, axis=1, keepdims=True)
        dh_t = np.dot(self.W_y.T, dy_t) + dh_next
        
        # 隐藏层梯度
        dh_prev = dh_t * z_t
        dh_tilde = dh_t * (1 - z_t)
        
        # 候选隐藏状态梯度
        dcombined_h = np.dot(self.W_h.T, self.tanh_derivative(h_tilde) * dh_tilde)
        dW_h = np.dot(self.tanh_derivative(h_tilde) * dh_tilde, combined_h.T)
        db_h = np.sum(self.tanh_derivative(h_tilde) * dh_tilde, axis=1, keepdims=True)
        
        # 重置门梯度
        dr_h_prev = dcombined_h[self.input_size:]
        dr_t = dr_h_prev * h_prev
        
        # 更新门梯度
        dz_t = dh_t * h_prev - dh_t * h_tilde
        dz_t = self.sigmoid_derivative(z_t) * dz_t
        
        # 拼接输入和前一隐藏状态的梯度
        dcombined_r = np.dot(self.W_r.T, self.sigmoid_derivative(r_t) * dr_t)
        dcombined_z = np.dot(self.W_z.T, dz_t)
        
        # 合并梯度
        dcombined = dcombined_r + dcombined_z
        dx_t = dcombined[:self.input_size]
        dh_prev += dcombined[self.input_size:]
        
        # 权重和偏置的梯度
        dW_r = np.dot(self.sigmoid_derivative(r_t) * dr_t, combined.T)
        db_r = np.sum(self.sigmoid_derivative(r_t) * dr_t, axis=1, keepdims=True)
        
        dW_z = np.dot(dz_t, combined.T)
        db_z = np.sum(dz_t, axis=1, keepdims=True)
        
        return dx_t, dh_prev, dW_z, db_z, dW_r, db_r, dW_h, db_h, dW_y, db_y
    
    def backward(self, dy, h_prev):
        # dy: (output_size, batch_size)
        # h_prev: (hidden_size, batch_size)
        
        seq_len = len(self.cache['h'])
        batch_size = self.cache['x'].shape[2]
        
        # 初始化梯度
        dh_next = np.zeros((self.hidden_size, batch_size))
        
        # 累积梯度
        dW_z_total = np.zeros_like(self.W_z)
        db_z_total = np.zeros_like(self.b_z)
        dW_r_total = np.zeros_like(self.W_r)
        db_r_total = np.zeros_like(self.b_r)
        dW_h_total = np.zeros_like(self.W_h)
        db_h_total = np.zeros_like(self.b_h)
        dW_y_total = np.zeros_like(self.W_y)
        db_y_total = np.zeros_like(self.b_y)
        
        # 从最后一个时间步向前计算梯度
        for t in reversed(range(seq_len)):
            x_t = self.cache['x'][t]
            h_prev_t = self.cache['h'][t-1] if t > 0 else h_prev
            h_t = self.cache['h'][t]
            z_t = self.cache['z'][t]
            r_t = self.cache['r'][t]
            h_tilde = self.cache['h_tilde'][t]
            combined = self.cache['combined'][t]
            combined_h = self.cache['combined_h'][t]
            
            dy_t = dy if t == seq_len - 1 else np.zeros_like(dy)
            
            dx_t, dh_next, dW_z, db_z, dW_r, db_r, dW_h, db_h, dW_y, db_y = self.backward_step(
                dy_t, dh_next, h_prev_t, h_t, z_t, r_t, h_tilde, combined, combined_h
            )
            
            dW_z_total += dW_z
            db_z_total += db_z
            dW_r_total += dW_r
            db_r_total += db_r
            dW_h_total += dW_h
            db_h_total += db_h
            dW_y_total += dW_y
            db_y_total += db_y
        
        # 返回权重和偏置的梯度
        gradients = {
            'dW_z': dW_z_total,
            'db_z': db_z_total,
            'dW_r': dW_r_total,
            'db_r': db_r_total,
            'dW_h': dW_h_total,
            'db_h': db_h_total,
            'dW_y': dW_y_total,
            'db_y': db_y_total
        }
        
        return gradients
    
    def update(self, gradients):
        # 梯度下降更新
        self.W_z -= self.learning_rate * gradients['dW_z']
        self.b_z -= self.learning_rate * gradients['db_z']
        self.W_r -= self.learning_rate * gradients['dW_r']
        self.b_r -= self.learning_rate * gradients['db_r']
        self.W_h -= self.learning_rate * gradients['dW_h']
        self.b_h -= self.learning_rate * gradients['db_h']
        self.W_y -= self.learning_rate * gradients['dW_y']
        self.b_y -= self.learning_rate * gradients['db_y']
    
    def forward(self, x):
        # 重写forward方法，处理整个序列
        seq_len, _, batch_size = x.shape
        h_prev = np.zeros((self.hidden_size, batch_size))
        y_list = []
        
        # 保存每一步的中间结果
        self.cache = {
            'h': [],
            'z': [],
            'r': [],
            'h_tilde': [],
            'combined': [],
            'combined_h': []
        }
        
        for t in range(seq_len):
            x_t = x[t]
            combined = np.concatenate((x_t, h_prev), axis=0)
            
            z_t = self.sigmoid(np.dot(self.W_z, combined) + self.b_z)
            r_t = self.sigmoid(np.dot(self.W_r, combined) + self.b_r)
            
            r_h_prev = r_t * h_prev
            combined_h = np.concatenate((x_t, r_h_prev), axis=0)
            h_tilde = self.tanh(np.dot(self.W_h, combined_h) + self.b_h)
            
            h_prev = z_t * h_prev + (1 - z_t) * h_tilde
            y_t = np.dot(self.W_y, h_prev) + self.b_y
            
            # 保存中间结果
            self.cache['h'].append(h_prev)
            self.cache['z'].append(z_t)
            self.cache['r'].append(r_t)
            self.cache['h_tilde'].append(h_tilde)
            self.cache['combined'].append(combined)
            self.cache['combined_h'].append(combined_h)
            y_list.append(y_t)
        
        self.cache['x'] = x
        self.cache['y'] = y_list
        
        return y_list[-1]  # 返回最后一个时间步的输出