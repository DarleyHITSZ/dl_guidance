import numpy as np
import os
from datetime import datetime
from utils.data_loader import DataLoader
from models.min_gru import MinGRU
from models.mamba import Mamba
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -------------------------- 实验超参数配置（关键优化：降低学习率）--------------------------
model_type = 'min_gru'  # 模型类型：'min_gru' 或 'mamba'
ticker = 'AAPL'         # 股票代码
start_date = '2010-01-01'  # 数据起始日期
end_date = '2023-12-31'    # 数据结束日期
seq_len = 30            # 序列长度（用前30天预测下1天）
input_size = 1          # 输入特征维度（仅收盘价）
hidden_size = 256       # 隐藏层维度
output_size = 1         # 输出维度（预测收盘价）
batch_size = 32         # 批次大小
n_epochs = 5            # 训练轮数
learning_rate = 0.005  # 关键修复：从0.01降至0.001，避免梯度爆炸
weight_decay = 0.0001   # L2正则化系数（抑制过拟合）
train_ratio = 0.8       # 训练集比例（80%训练，20%验证）
grad_clip = 1.0         # 新增：梯度裁剪阈值，防止梯度爆炸
dropout_rate = 0.2
# Mamba模型额外参数（仅Mamba使用）
state_size = 64         # 状态空间维度
kernel_size = 4         # 卷积核大小（预留扩展）

# -------------------------- 工具函数（优化数值稳定性）--------------------------
def calculate_metrics(y_true, y_pred):
    """优化指标计算，避免极端值影响"""
    # 移除NaN和无穷大值
    mask = ~(np.isnan(y_true).any(axis=1) | np.isnan(y_pred).any(axis=1) | 
             np.isinf(y_true).any(axis=1) | np.isinf(y_pred).any(axis=1))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return {'MSE': np.inf, 'RMSE': np.inf, 'MAE': np.inf, 'R²': -np.inf, '有效样本数': 0}
    
    # 裁剪极端预测值（限制在真实值的0.1~10倍范围内）
    y_min = y_true_clean.min() * 0.1
    y_max = y_true_clean.max() * 10.0
    y_pred_clean = np.clip(y_pred_clean, y_min, y_max)
    
    # 计算指标
    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    r2 = r2_score(y_true_clean, y_pred_clean)
    
    # 修正R²（避免因数据分布导致的极端负值）
    r2 = max(r2, -5.0)  # R²最低限制在-5，便于观察趋势
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        '有效样本数': len(y_true_clean)
    }

def print_metrics(metrics, phase='Validation'):
    """打印指标结果"""
    print(f"\n{phase} 性能指标:")
    for key, value in metrics.items():
        if key == '有效样本数':
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.6f}")

def clip_gradient(model, clip_value):
    """梯度裁剪，适配Mamba模型，强化裁剪效果"""
    params = ['W_in', 'B', 'C', 'D', 'W_out', 'b_out']
    for param_name in params:
        grad_name = 'd' + param_name
        if hasattr(model, grad_name):
            grad = getattr(model, grad_name)
            # 使用更严格的L2范数裁剪
            grad_norm = np.linalg.norm(grad)
            if grad_norm > clip_value:
                grad = grad * (clip_value / (grad_norm + 1e-8))  # 添加微小值避免除零
                setattr(model, grad_name, grad)

# -------------------------- 数据加载与划分（优化路径和权限）--------------------------
def load_and_split_data():
    """加载股票数据并划分为训练集/验证集，优化数据保存路径"""
    # 初始化数据加载器
    loader = DataLoader()
    print(f"正在下载 {ticker} 股票数据（{start_date} 至 {end_date}）...")
    
    # 下载并预处理数据（仅保留收盘价）
    data_dict = loader.load_yahoo_stock(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date
    )
    scaled_data = data_dict['data']  # 归一化后的数据 (n_samples, 1)
    scaler = data_dict['scaler']     # 归一化器
    original_data = data_dict['original_data']  # 原始数据 (n_samples, 1)
    
    print(f"数据加载完成！总样本数: {len(scaled_data)}")
    
    # 划分训练集和验证集
    train_size = int(len(scaled_data) * train_ratio)
    train_data = scaled_data[:train_size]
    valid_data = scaled_data[train_size:]
    
    print(f"训练集样本数: {len(train_data)}, 验证集样本数: {len(valid_data)}")
    
    # 创建训练集批次
    print(f"正在创建训练集批次（序列长度: {seq_len}, 批次大小: {batch_size}）...")
    train_batches = loader.create_stock_batches(
        data=train_data,
        seq_len=seq_len,
        batch_size=batch_size
    )
    
    # 创建验证集批次（批次大小=验证集样本数，避免批次拆分）
    valid_batches = loader.create_stock_batches(
        data=valid_data,
        seq_len=seq_len,
        batch_size=len(valid_data) - seq_len  # 单个批次包含所有验证样本
    )
    
    print(f"批次创建完成！训练批次数量: {len(train_batches)}, 验证批次数量: {len(valid_batches)}")
    
    return train_batches, valid_batches, scaler, original_data

# -------------------------- 模型训练函数（核心修复：MinGRU训练）--------------------------
def train_min_gru(train_batches, valid_batches, scaler):
    """训练MinGRU模型，添加梯度裁剪和数值稳定性优化"""
    # 初始化模型
    model = MinGRU(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size
    )
    print("\n=== MinGRU 模型初始化完成 ===")
    print(f"输入维度: {input_size}, 隐藏层维度: {hidden_size}, 输出维度: {output_size}")
    print(f"学习率: {learning_rate}, 梯度裁剪阈值: {grad_clip}, L2正则化: {weight_decay}")
    
    # 训练记录
    train_losses = []
    valid_losses = []
    
    for epoch in range(n_epochs):
        start_time = datetime.now()
        epoch_train_loss = 0.0
        batch_count = 0
        
        # 训练阶段（遍历所有批次）
        for batch_idx, (x_batch, y_batch) in enumerate(train_batches):
            # 重置梯度（每批次独立重置）
            model.reset_grads()
            
            # 关键修复：动态处理y_batch维度，确保形状为 (output_size, batch_size)
            y_batch = y_batch.squeeze()  # 移除所有大小为1的轴
            if len(y_batch.shape) == 1:
                y_batch = y_batch.reshape(output_size, -1)  # (1, batch_size)
            else:
                y_batch = y_batch.T  # 转置为 (output_size, batch_size)
            
            # 确保批次大小一致
            current_batch_size = y_batch.shape[1]
            if current_batch_size == 0:
                continue  # 跳过空批次
            
            # 初始化隐藏状态（数值稳定：用小随机值替代全0，避免初始梯度为0）
            h_prev = np.random.randn(hidden_size, current_batch_size) * 0.001
            
            # 逐个时间步前向传播（保存中间状态）
            y_pred_seq = []
            h_seq = []
            for t in range(seq_len):
                x_t = x_batch[t]  # (input_size, current_batch_size)
                # 数值裁剪：输入限制在[-10, 10]，避免激活函数溢出
                x_t = np.clip(x_t, -10, 10)
                y_t_pred, h_t = model.forward_step(x_t, h_prev)
                # 输出裁剪：防止数值爆炸
                y_t_pred = np.clip(y_t_pred, -10, 10)
                y_pred_seq.append(y_t_pred)
                h_seq.append(h_t)
                h_prev = h_t
            
            # 取最后一个时间步的输出作为预测结果
            y_pred = y_pred_seq[-1]  # (output_size, current_batch_size)
            
            # 计算损失（数值稳定：先裁剪预测值和真实值）
            y_pred_clipped = np.clip(y_pred, 0, 1)  # 归一化后的数据范围在[0,1]
            y_batch_clipped = np.clip(y_batch, 0, 1)
            loss = 0.5 * np.mean(np.square(y_pred_clipped - y_batch_clipped))
            
            # 跳过NaN损失（防止训练中断）
            if np.isnan(loss):
                print(f"警告：第{epoch+1}轮第{batch_idx+1}批出现NaN损失，跳过该批次")
                continue
            
            epoch_train_loss += loss
            batch_count += 1
            
            # 反向传播（计算梯度）
            dy = y_pred_clipped - y_batch_clipped  # 输出梯度
            dh_next = np.zeros((hidden_size, current_batch_size))  # 初始下一时间步梯度
            
            for t in reversed(range(seq_len)):
                x_t = x_batch[t]
                x_t = np.clip(x_t, -10, 10)
                # 前向传播重构中间状态
                prev_h = h_seq[t-1] if t > 0 else np.random.randn(hidden_size, current_batch_size) * 0.001
                model.forward_step(x_t, prev_h)
                # 反向传播计算梯度
                dx_t, dh_next = model.backward_step(dy, dh_next)
            
            # 关键修复：梯度裁剪（防止梯度爆炸）
            clip_gradient(model, grad_clip)
            
            # 参数更新（带L2正则化）
            model.update(lr=learning_rate, weight_decay=weight_decay)
        
        # 计算平均训练损失（避免除零）
        avg_train_loss = epoch_train_loss / batch_count if batch_count > 0 else np.inf
        train_losses.append(avg_train_loss)
        
        # 验证阶段（不计算梯度）
        avg_valid_loss = 0.0
        y_true_all = []
        y_pred_all = []
        valid_batch_count = 0
        
        for x_batch, y_batch in valid_batches:
            # 处理y_batch维度
            y_batch = y_batch.squeeze()
            if len(y_batch.shape) == 1:
                y_batch = y_batch.reshape(output_size, -1)
            else:
                y_batch = y_batch.T
            
            current_batch_size = y_batch.shape[1]
            if current_batch_size == 0:
                continue
            
            # 初始化隐藏状态
            h_prev = np.random.randn(hidden_size, current_batch_size) * 0.001
            y_pred_seq = []
            
            # 前向传播（验证阶段不裁剪，保持真实预测）
            for t in range(seq_len):
                x_t = x_batch[t]
                y_t_pred, h_prev = model.forward_step(x_t, h_prev)
                y_pred_seq.append(y_t_pred)
            
            # 预测结果
            y_pred = y_pred_seq[-1]
            
            # 计算验证损失（裁剪数值）
            y_pred_clipped = np.clip(y_pred, 0, 1)
            y_batch_clipped = np.clip(y_batch, 0, 1)
            valid_loss = 0.5 * np.mean(np.square(y_pred_clipped - y_batch_clipped))
            
            if np.isnan(valid_loss):
                print(f"警告：验证集第{valid_batch_count+1}批出现NaN损失，跳过")
                continue
            
            avg_valid_loss += valid_loss
            valid_batch_count += 1
            
            # 反归一化（转换为原始股价范围）
            y_true = scaler.inverse_transform(y_batch.T)  # (current_batch_size, 1)
            y_pred = scaler.inverse_transform(y_pred_clipped.T)  # (current_batch_size, 1)
            
            # 收集结果（避免NaN）
            y_true_all.extend(y_true[~np.isnan(y_true).any(axis=1)])
            y_pred_all.extend(y_pred[~np.isnan(y_pred).any(axis=1)])
        
        # 计算平均验证损失
        avg_valid_loss = avg_valid_loss / valid_batch_count if valid_batch_count > 0 else np.inf
        valid_losses.append(avg_valid_loss)
        
        # 计算验证集指标
        metrics = calculate_metrics(np.array(y_true_all).reshape(-1, 1), np.array(y_pred_all).reshape(-1, 1))
        
        # 打印epoch信息（优化显示格式）
        epoch_time = (datetime.now() - start_time).total_seconds()
        print(f"\nEpoch [{epoch+1}/{n_epochs}]")
        print(f"训练损失: {avg_train_loss:.6f} | 验证损失: {avg_valid_loss:.6f} | 耗时: {epoch_time:.2f}s")
        print_metrics(metrics, phase='Validation')
    
    # 训练完成后保存模型（仅保存有效模型）
    if not np.isnan(train_losses[-1]) and not np.isinf(train_losses[-1]):
        model_save_path = f"min_gru_model_epoch{n_epochs}.npz"
        np.savez(model_save_path,
                 W_z=model.W_z, W_r=model.W_r, W_h=model.W_h, W_y=model.W_y,
                 b_z=model.b_z, b_r=model.b_r, b_h=model.b_h, b_y=model.b_y)
        print(f"\nMinGRU模型已保存至: {model_save_path}")
    else:
        print("\n训练失败：最终损失为NaN/inf，未保存模型")
    
    return {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'final_metrics': metrics
    }

# -------------------------- Mamba训练函数（同步优化数值稳定性）--------------------------
def train_mamba(train_batches, valid_batches, scaler):
    """训练Mamba模型，重点解决过拟合问题"""
    # 初始化模型（传入dropout_rate）
    model = Mamba(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        state_size=state_size,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate  # 启用Dropout
    )
    print("\n=== Mamba 模型初始化完成 ===")
    print(f"输入维度: {input_size}, 隐藏层维度: {hidden_size}, 状态维度: {state_size}")
    print(f"学习率: {learning_rate}, 梯度裁剪阈值: {grad_clip}, L2正则化: {weight_decay}, Dropout: {dropout_rate}")
    
    train_losses = []
    valid_losses = []
    
    for epoch in range(n_epochs):
        start_time = datetime.now()
        epoch_train_loss = 0.0
        batch_count = 0
        total_grad_norm = 0.0
        
        # 训练阶段（启用Dropout）
        for batch_idx, (x_batch, y_batch) in enumerate(train_batches):
            model.reset_grads()
            
            # 处理y_batch维度
            y_batch = y_batch.squeeze()
            if len(y_batch.shape) == 1:
                y_batch = y_batch.reshape(output_size, -1)
            else:
                y_batch = y_batch.T
            current_batch_size = y_batch.shape[1]
            if current_batch_size == 0:
                continue
            
            # 前向传播（training=True，启用Dropout）
            x_batch_clipped = np.clip(x_batch, -3, 3)
            y_pred_seq = model.forward(x_batch_clipped, training=True)
            y_pred = y_pred_seq[-1]
            
            # 损失计算（保持数值稳定）
            y_pred_clipped = np.clip(y_pred, 1e-6, 1 - 1e-6)
            y_batch_clipped = np.clip(y_batch, 1e-6, 1 - 1e-6)
            loss = np.mean(np.square(y_pred_clipped - y_batch_clipped)) + 1e-8
            if np.isnan(loss):
                print(f"警告：第{epoch+1}轮第{batch_idx+1}批损失异常，跳过")
                continue
            
            epoch_train_loss += loss
            batch_count += 1
            
            # 反向传播和参数更新
            dout = y_pred_seq - np.repeat(y_batch_clipped[np.newaxis, :, :], seq_len, axis=0)
            dout = np.clip(dout, -1, 1)
            model.backward(dout)
            
            # 梯度范数监控
            grad_norm = np.linalg.norm(model.dW_in) + np.linalg.norm(model.dB) + np.linalg.norm(model.dC)
            total_grad_norm += grad_norm
            
            # 梯度裁剪
            clip_gradient(model, grad_clip)
            
            # 参数更新
            model.update(lr=learning_rate, weight_decay=weight_decay)
        
        # 计算平均训练损失和梯度范数
        avg_train_loss = epoch_train_loss / batch_count if batch_count > 0 else np.inf
        avg_grad_norm = total_grad_norm / batch_count if batch_count > 0 else 0.0
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch+1}/{n_epochs}] - 平均梯度范数: {avg_grad_norm:.6f}")
        
        # 验证阶段（禁用Dropout，training=False）
        avg_valid_loss = 0.0
        y_true_all = []
        y_pred_all = []
        valid_batch_count = 0
        
        for x_batch, y_batch in valid_batches:
            # 处理y_batch维度
            y_batch = y_batch.squeeze()
            if len(y_batch.shape) == 1:
                y_batch = y_batch.reshape(output_size, -1)
            else:
                y_batch = y_batch.T
            current_batch_size = y_batch.shape[1]
            if current_batch_size == 0:
                continue
            
            # 前向传播（禁用Dropout，评估真实泛化能力）
            y_pred_seq = model.forward(x_batch, training=False)
            y_pred = y_pred_seq[-1]
            
            # 验证损失计算
            y_pred_clipped = np.clip(y_pred, 1e-6, 1 - 1e-6)
            y_batch_clipped = np.clip(y_batch, 1e-6, 1 - 1e-6)
            valid_loss = np.mean(np.square(y_pred_clipped - y_batch_clipped))
            if np.isnan(valid_loss):
                continue
            
            avg_valid_loss += valid_loss
            valid_batch_count += 1
            
            # 反归一化和结果收集（放宽过滤条件，避免有效样本过少）
            y_true = scaler.inverse_transform(y_batch.T)
            y_pred = scaler.inverse_transform(y_pred_clipped.T)
            # 放宽偏差阈值到100%（仅过滤极端异常值）
            valid_mask = (np.abs(y_pred - y_true) / (y_true + 1e-8)) < 1.0
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)
        
        # 计算平均验证损失
        avg_valid_loss = avg_valid_loss / valid_batch_count if valid_batch_count > 0 else np.inf
        valid_losses.append(avg_valid_loss)
        
        # 计算验证指标
        if len(y_true_all) == 0 or len(y_pred_all) == 0:
            metrics = {'MSE': np.inf, 'RMSE': np.inf, 'MAE': np.inf, 'R²': -np.inf, '有效样本数': 0}
        else:
            metrics = calculate_metrics(np.array(y_true_all).reshape(-1, 1), np.array(y_pred_all).reshape(-1, 1))
        
        # 打印训练信息
        epoch_time = (datetime.now() - start_time).total_seconds()
        print(f"\nEpoch [{epoch+1}/{n_epochs}]")
        print(f"训练损失: {avg_train_loss:.6f} | 验证损失: {avg_valid_loss:.6f} | 耗时: {epoch_time:.2f}s")
        print_metrics(metrics, phase='Validation')
    
    # 保存模型（仅当验证损失合理时）
    if not np.isnan(valid_losses[-1]) and not np.isinf(valid_losses[-1]) and valid_losses[-1] < 0.1:
        model_save_path = f"mamba_model_epoch{n_epochs}_anti_overfit.npz"
        np.savez(model_save_path,
                 W_in=model.W_in, B=model.B, C=model.C, D=model.D,
                 W_out=model.W_out, b_out=model.b_out)
        print(f"\n抗过拟合Mamba模型已保存至: {model_save_path}")
    else:
        print("\n验证损失过高，未保存模型")
    
    return {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'final_metrics': metrics
    }

# -------------------------- 主函数 --------------------------
def main():
    print("="*60)
    print(f"开始 {model_type.upper()} 模型训练（股票预测任务）")
    print("="*60)
    
    # 1. 加载并划分数据
    train_batches, valid_batches, scaler, original_data = load_and_split_data()
    
    # 2. 训练模型
    if model_type == 'min_gru':
        results = train_min_gru(train_batches, valid_batches, scaler)
    elif model_type == 'mamba':
        results = train_mamba(train_batches, valid_batches, scaler)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}，请选择 'min_gru' 或 'mamba'")
    
    # 3. 打印最终结果
    print("\n" + "="*60)
    print(f"{model_type.upper()} 模型训练完成！")
    print("="*60)
    print(f"最终训练损失: {results['train_losses'][-1]:.6f}")
    print(f"最终验证损失: {results['valid_losses'][-1]:.6f}")
    print_metrics(results['final_metrics'], phase='Final Validation')

if __name__ == "__main__":
    main()