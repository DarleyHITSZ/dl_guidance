import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import DataLoader
from models.min_gru import MinGRU
from models.mamba import Mamba
import time

# 超参数配置（两种模型共享）
HYPERPARAMS = {
    'seq_len': 30,
    'batch_size': 32,
    'hidden_size': 128,
    'state_size': 64,  # Mamba专用
    'kernel_size': 4,  # Mamba专用
    'learning_rate': 0.001,
    'epochs': 10,
    'stock_symbol': 'AAPL',
    'feature_cols': ['Close']
}

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true)**2)

def train_single_model(model_type, X_train_batches, y_train_batches, X_test_batches, y_test_batches, input_size, output_size):
    print(f"\n=== 训练 {model_type} 模型 ===")
    
    # 初始化模型
    if model_type == 'min_gru':
        model = MinGRU(
            input_size=input_size,
            hidden_size=HYPERPARAMS['hidden_size'],
            output_size=output_size,
            learning_rate=HYPERPARAMS['learning_rate']
        )
    else:
        model = Mamba(
            input_size=input_size,
            hidden_size=HYPERPARAMS['hidden_size'],
            output_size=output_size,
            state_size=HYPERPARAMS['state_size'],
            kernel_size=HYPERPARAMS['kernel_size'],
            learning_rate=HYPERPARAMS['learning_rate']
        )
    
    # 训练历史
    train_loss_history = []
    val_loss_history = []
    epoch_times = []
    
    # 开始训练
    for epoch in range(HYPERPARAMS['epochs']):
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
            if model_type == 'min_gru':
                h_prev = np.zeros((HYPERPARAMS['hidden_size'], HYPERPARAMS['batch_size']))
                gradients = model.backward(dy, h_prev)
            else:
                gradients = model.backward(dy)
            
            # 参数更新
            model.update(gradients)
        
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
        
        # 计算训练时间
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        print(f"Epoch {epoch+1}/{HYPERPARAMS['epochs']} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f} - Time: {epoch_time:.2f}s")
    
    # 评估模型
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
    
    final_mse = mse_loss(all_preds, all_trues)
    
    return {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'epoch_times': epoch_times,
        'total_time': sum(epoch_times),
        'final_mse': final_mse,
        'predictions': all_preds,
        'ground_truth': all_trues
    }

def run_benchmark():
    print("=== 模型性能对比实验 ===")
    print("="*40)
    print("超参数配置:")
    for key, value in HYPERPARAMS.items():
        print(f"  {key}: {value}")
    print("="*40)
    
    # 加载数据
    print("\n加载数据...")
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
    output_size = 1
    
    print(f"数据加载完成")
    print(f"训练样本数: {len(X_train_batches) * HYPERPARAMS['batch_size']}")
    print(f"测试样本数: {len(X_test_batches) * HYPERPARAMS['batch_size']}")
    print("="*40)
    
    # 训练MinGRU模型
    mingru_results = train_single_model(
        'min_gru',
        X_train_batches, y_train_batches,
        X_test_batches, y_test_batches,
        input_size, output_size
    )
    
    # 训练Mamba模型
    mamba_results = train_single_model(
        'mamba',
        X_train_batches, y_train_batches,
        X_test_batches, y_test_batches,
        input_size, output_size
    )
    
    # 生成对比报告
    generate_comparison_report(mingru_results, mamba_results, data_loader)
    
    return mingru_results, mamba_results

def generate_comparison_report(mingru_results, mamba_results, data_loader):
    print("\n" + "="*40)
    print("=== 模型性能对比报告 ===")
    print("="*40)
    
    # 训练时间对比
    print("\n1. 训练时间对比:")
    print(f"   MinGRU总训练时间: {mingru_results['total_time']:.2f}秒")
    print(f"   Mamba总训练时间: {mamba_results['total_time']:.2f}秒")
    print(f"   Mamba比MinGRU快: {(mingru_results['total_time'] - mamba_results['total_time']):.2f}秒 ({(1 - mamba_results['total_time']/mingru_results['total_time'])*100:.1f}%)")
    
    # 最终损失对比
    print("\n2. 最终损失对比 (MSE):")
    print(f"   MinGRU训练损失: {mingru_results['train_loss'][-1]:.6f}")
    print(f"   Mamba训练损失: {mamba_results['train_loss'][-1]:.6f}")
    print(f"   MinGRU验证损失: {mingru_results['val_loss'][-1]:.6f}")
    print(f"   Mamba验证损失: {mamba_results['val_loss'][-1]:.6f}")
    
    # 预测结果评估（转换回原始尺度）
    mingru_preds_original = data_loader.inverse_transform(mingru_results['predictions'])
    mamba_preds_original = data_loader.inverse_transform(mamba_results['predictions'])
    true_values_original = data_loader.inverse_transform(mamba_results['ground_truth'])
    
    mingru_mse = np.mean((mingru_preds_original - true_values_original)**2)
    mamba_mse = np.mean((mamba_preds_original - true_values_original)**2)
    mingru_rmse = np.sqrt(mingru_mse)
    mamba_rmse = np.sqrt(mamba_mse)
    mingru_mae = np.mean(np.abs(mingru_preds_original - true_values_original))
    mamba_mae = np.mean(np.abs(mamba_preds_original - true_values_original))
    
    print("\n3. 预测性能对比 (原始尺度):")
    print(f"   MinGRU MSE: {mingru_mse:.2f}")
    print(f"   Mamba MSE: {mamba_mse:.2f}")
    print(f"   MinGRU RMSE: {mingru_rmse:.2f}")
    print(f"   Mamba RMSE: {mamba_rmse:.2f}")
    print(f"   MinGRU MAE: {mingru_mae:.2f}")
    print(f"   Mamba MAE: {mamba_mae:.2f}")
    
    # 可视化结果
    visualize_results(mingru_results, mamba_results, mingru_preds_original, mamba_preds_original, true_values_original)

def visualize_results(mingru_results, mamba_results, mingru_preds_original, mamba_preds_original, true_values_original):
    plt.figure(figsize=(15, 10))
    
    # 1. 训练损失对比
    plt.subplot(2, 2, 1)
    plt.plot(mingru_results['train_loss'], label='MinGRU', linewidth=2, color='blue')
    plt.plot(mamba_results['train_loss'], label='Mamba', linewidth=2, color='red')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. 验证损失对比
    plt.subplot(2, 2, 2)
    plt.plot(mingru_results['val_loss'], label='MinGRU', linewidth=2, color='blue')
    plt.plot(mamba_results['val_loss'], label='Mamba', linewidth=2, color='red')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    
    # 3. 训练时间对比
    plt.subplot(2, 2, 3)
    epochs = np.arange(1, HYPERPARAMS['epochs'] + 1)
    plt.bar(epochs - 0.2, mingru_results['epoch_times'], width=0.4, label='MinGRU', color='blue')
    plt.bar(epochs + 0.2, mamba_results['epoch_times'], width=0.4, label='Mamba', color='red')
    plt.title('Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True, axis='y')
    
    # 4. 预测结果对比（原始尺度）
    plt.subplot(2, 2, 4)
    sample_indices = np.arange(0, len(true_values_original), max(1, len(true_values_original) // 200))
    plt.plot(sample_indices, true_values_original[sample_indices], label='True Values', linewidth=2, color='black')
    plt.plot(sample_indices, mingru_preds_original[sample_indices], label='MinGRU Predictions', linewidth=1.5, color='blue', alpha=0.8)
    plt.plot(sample_indices, mamba_preds_original[sample_indices], label='Mamba Predictions', linewidth=1.5, color='red', alpha=0.8)
    plt.title('Stock Price Prediction Comparison (Original Scale)')
    plt.xlabel('Time Step')
    plt.ylabel('AAPL Close Price')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('mamba_vs_mingru_comparison.png', dpi=300)
    print(f"\n可视化结果已保存到 'mamba_vs_mingru_comparison.png'")
    plt.close()
    
    # 保存结果数据到文件
    import pandas as pd
    results_df = pd.DataFrame({
        'Epoch': range(1, HYPERPARAMS['epochs'] + 1),
        'MinGRU_Train_Loss': mingru_results['train_loss'],
        'Mamba_Train_Loss': mamba_results['train_loss'],
        'MinGRU_Val_Loss': mingru_results['val_loss'],
        'Mamba_Val_Loss': mamba_results['val_loss'],
        'MinGRU_Epoch_Time': mingru_results['epoch_times'],
        'Mamba_Epoch_Time': mamba_results['epoch_times']
    })
    results_df.to_csv('benchmark_results.csv', index=False)
    print(f"对比结果数据已保存到 'benchmark_results.csv'")

if __name__ == "__main__":
    run_benchmark()