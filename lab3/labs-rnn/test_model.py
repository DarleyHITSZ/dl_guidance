import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import DataLoader
from models.min_gru import MinGRU
from models.mamba import Mamba
import time

# 超参数配置
HYPERPARAMS = {
    'model_type': 'mamba',  # 'min_gru' 或 'mamba'
    'seq_len': 30,
    'batch_size': 32,
    'hidden_size': 128,
    'state_size': 64,  # 仅Mamba使用
    'kernel_size': 4,  # 仅Mamba使用
    'learning_rate': 0.001,
    'epochs': 10,
    'stock_symbol': 'AAPL',
    'feature_cols': ['Close']
}

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def rmse_loss(y_pred, y_true):
    return np.sqrt(mse_loss(y_pred, y_true))

def mae_loss(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

def r2_score(y_pred, y_true):
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean)**2)
    ss_res = np.sum((y_true - y_pred)**2)
    return 1 - (ss_res / ss_tot)

def train_model():
    print("=== 模型训练开始 ===")
    print(f"模型类型: {HYPERPARAMS['model_type']}")
    print(f"序列长度: {HYPERPARAMS['seq_len']}")
    print(f"批次大小: {HYPERPARAMS['batch_size']}")
    print(f"隐藏层大小: {HYPERPARAMS['hidden_size']}")
    print(f"学习率: {HYPERPARAMS['learning_rate']}")
    print(f"训练轮数: {HYPERPARAMS['epochs']}")
    print(f"股票代码: {HYPERPARAMS['stock_symbol']}")
    print(f"特征列: {HYPERPARAMS['feature_cols']}")
    print("="*30)
    
    # 加载数据
    print("加载数据...")
    data_loader = DataLoader(
        stock_symbol=HYPERPARAMS['stock_symbol'],
        seq_len=HYPERPARAMS['seq_len'],
        batch_size=HYPERPARAMS['batch_size']
    )
    
    data = data_loader.load_and_preprocess(feature_cols=HYPERPARAMS['feature_cols'])
    X_train_batches = data['X_train_batches']
    y_train_batches = data['y_train_batches']
    X_test_batches = data['X_test_batches']
    y_test_batches = data['y_test_batches']
    
    input_size = data['input_size']
    output_size = 1  # 预测收盘价
    
    print(f"数据加载完成")
    print(f"训练批次数量: {len(X_train_batches)}")
    print(f"测试批次数量: {len(X_test_batches)}")
    print(f"输入维度: {input_size}")
    print("="*30)
    
    # 初始化模型
    print("初始化模型...")
    if HYPERPARAMS['model_type'] == 'min_gru':
        model = MinGRU(
            input_size=input_size,
            hidden_size=HYPERPARAMS['hidden_size'],
            output_size=output_size,
            learning_rate=HYPERPARAMS['learning_rate']
        )
    elif HYPERPARAMS['model_type'] == 'mamba':
        model = Mamba(
            input_size=input_size,
            hidden_size=HYPERPARAMS['hidden_size'],
            output_size=output_size,
            state_size=HYPERPARAMS['state_size'],
            kernel_size=HYPERPARAMS['kernel_size'],
            learning_rate=HYPERPARAMS['learning_rate']
        )
    else:
        raise ValueError(f"不支持的模型类型: {HYPERPARAMS['model_type']}")
    
    print("模型初始化完成")
    print("="*30)
    
    # 训练历史
    train_loss_history = []
    val_loss_history = []
    total_train_time = 0
    
    # 开始训练
    for epoch in range(HYPERPARAMS['epochs']):
        print(f"\nEpoch {epoch+1}/{HYPERPARAMS['epochs']}")
        
        epoch_start_time = time.time()
        epoch_train_loss = 0
        
        # 训练批次
        for batch_idx in range(len(X_train_batches)):
            X_batch = X_train_batches[batch_idx]
            y_batch = y_train_batches[batch_idx]
            
            # 前向传播
            y_pred = model.forward(X_batch)
            
            # 计算损失
            loss = mse_loss(y_pred, y_batch)
            epoch_train_loss += loss
            
            # 反向传播
            dy = y_pred - y_batch
            if HYPERPARAMS['model_type'] == 'min_gru':
                h_prev = np.zeros((HYPERPARAMS['hidden_size'], HYPERPARAMS['batch_size']))
                gradients = model.backward(dy, h_prev)
            else:
                gradients = model.backward(dy)
            
            # 参数更新
            model.update(gradients)
            
            # 打印批次信息
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(X_train_batches):
                print(f"  Batch {batch_idx+1}/{len(X_train_batches)} - Loss: {loss:.6f}")
        
        # 计算平均训练损失
        avg_train_loss = epoch_train_loss / len(X_train_batches)
        train_loss_history.append(avg_train_loss)
        
        # 验证
        val_loss = 0
        for batch_idx in range(len(X_test_batches)):
            X_batch = X_test_batches[batch_idx]
            y_batch = y_test_batches[batch_idx]
            
            y_pred = model.forward(X_batch)
            loss = mse_loss(y_pred, y_batch)
            val_loss += loss
        
        avg_val_loss = val_loss / len(X_test_batches)
        val_loss_history.append(avg_val_loss)
        
        epoch_time = time.time() - epoch_start_time
        total_train_time += epoch_time
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f} - Time: {epoch_time:.2f}s")
    
    print("\n" + "="*30)
    print("=== 模型训练完成 ===")
    print(f"总训练时间: {total_train_time:.2f}s")
    print(f"最终训练损失: {train_loss_history[-1]:.6f}")
    print(f"最终验证损失: {val_loss_history[-1]:.6f}")
    
    # 评估模型性能
    print("\n=== 模型性能评估 ===")
    all_preds = []
    all_trues = []
    
    for batch_idx in range(len(X_test_batches)):
        X_batch = X_test_batches[batch_idx]
        y_batch = y_test_batches[batch_idx]
        
        y_pred = model.forward(X_batch)
        all_preds.append(y_pred.flatten())
        all_trues.append(y_batch.flatten())
    
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    
    # 转换回原始尺度
    all_preds_original = data_loader.inverse_transform(all_preds)
    all_trues_original = data_loader.inverse_transform(all_trues)
    
    # 计算评估指标
    mse = mse_loss(all_preds_original, all_trues_original)
    rmse = rmse_loss(all_preds_original, all_trues_original)
    mae = mae_loss(all_preds_original, all_trues_original)
    r2 = r2_score(all_preds_original, all_trues_original)
    
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.4f}")
    
    # 可视化结果
    plt.figure(figsize=(12, 6))
    
    # 训练曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title(f'{HYPERPARAMS["model_type"]} Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    # 预测结果
    plt.subplot(1, 2, 2)
    plt.plot(all_trues_original, label='True Values')
    plt.plot(all_preds_original, label='Predicted Values')
    plt.title(f'{HYPERPARAMS["model_type"]} Prediction Results')
    plt.xlabel('Time Step')
    plt.ylabel(f'{HYPERPARAMS["stock_symbol"]} Close Price')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{HYPERPARAMS["model_type"]}_results.png')
    print(f"\n结果可视化已保存到 {HYPERPARAMS['model_type']}_results.png")
    plt.close()
    
    return {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'total_time': total_train_time
    }

if __name__ == "__main__":
    train_model()