#!/usr/bin/env python3
"""
超参数调整示例脚本

本脚本展示了如何调整多个超参数并比较不同组合的效果。
"""

import numpy as np
import matplotlib.pyplot as plt
from mlp.datasets import BostonHousingLoader
from mlp.layers import Dense
from mlp.activations import ReLU, Linear, Sigmoid, Tanh
from mlp.losses import MSE
from mlp.optimizers import Adam, SGD
from mlp.model import MLP

def load_dataset():
    """加载并预处理数据集"""
    print("加载数据集...")
    loader = BostonHousingLoader(test_size=0.2, random_state=42)
    X_train, y_train, X_test, y_test = loader.load_data()
    return X_train, y_train, X_test, y_test

def build_model(hidden_layers, activation, optimizer, weight_init):
    """构建模型"""
    model = MLP(loss=MSE(), optimizer=optimizer)
    
    # 输入层到第一个隐藏层
    model.add(Dense(13, hidden_layers[0], weight_init=weight_init))
    model.add(activation())
    
    # 隐藏层之间的连接
    for i in range(len(hidden_layers) - 1):
        model.add(Dense(hidden_layers[i], hidden_layers[i+1], weight_init=weight_init))
        model.add(activation())
    
    # 输出层
    model.add(Dense(hidden_layers[-1], 1, weight_init=weight_init))
    model.add(Linear())
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    """训练模型"""
    print(f"训练模型，轮数: {epochs}, 批次大小: {batch_size}")
    history = model.train(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        X_val=X_test,
        y_val=y_test,
        verbose=False
    )
    return history

def evaluate_model(model, X_test, y_test):
    """评估模型"""
    metrics = model.evaluate(X_test, y_test)
    return metrics

def main():
    """主函数"""
    print("=== MLP 超参数调整示例 ===")
    print("本示例将比较不同超参数组合的性能")
    
    # 加载数据集
    X_train, y_train, X_test, y_test = load_dataset()
    
    # 定义要尝试的超参数组合
    hyperparameter_combinations = [
        {
            "name": "Baseline",
            "hidden_layers": [64, 32],
            "activation": ReLU,
            "optimizer": Adam(learning_rate=0.001),
            "weight_init": "he",
            "epochs": 100,
            "batch_size": 32
        },
        {
            "name": "Larger Layers",
            "hidden_layers": [128, 64],
            "activation": ReLU,
            "optimizer": Adam(learning_rate=0.001),
            "weight_init": "he",
            "epochs": 100,
            "batch_size": 32
        },
        {
            "name": "Higher LR",
            "hidden_layers": [64, 32],
            "activation": ReLU,
            "optimizer": Adam(learning_rate=0.01),
            "weight_init": "he",
            "epochs": 100,
            "batch_size": 32
        },
        {
            "name": "Lower LR",
            "hidden_layers": [64, 32],
            "activation": ReLU,
            "optimizer": Adam(learning_rate=0.0001),
            "weight_init": "he",
            "epochs": 100,
            "batch_size": 32
        },
        {
            "name": "Larger Batch",
            "hidden_layers": [64, 32],
            "activation": ReLU,
            "optimizer": Adam(learning_rate=0.001),
            "weight_init": "he",
            "epochs": 100,
            "batch_size": 64
        },
        {
            "name": "Tanh Activation",
            "hidden_layers": [64, 32],
            "activation": Tanh,
            "optimizer": Adam(learning_rate=0.001),
            "weight_init": "xavier",
            "epochs": 100,
            "batch_size": 32
        },
        {
            "name": "SGD Optimizer",
            "hidden_layers": [64, 32],
            "activation": ReLU,
            "optimizer": SGD(learning_rate=0.01, momentum=0.9),
            "weight_init": "he",
            "epochs": 100,
            "batch_size": 32
        }
    ]
    
    # 训练和评估所有超参数组合
    results = []
    for i, params in enumerate(hyperparameter_combinations):
        print(f"\n--- 测试超参数组合 {i+1}/{len(hyperparameter_combinations)}: {params['name']} ---")
        
        # 构建模型
        model = build_model(
            params["hidden_layers"],
            params["activation"],
            params["optimizer"],
            params["weight_init"]
        )
        
        # 训练模型
        history = train_model(
            model,
            X_train, y_train,
            X_test, y_test,
            params["epochs"],
            params["batch_size"]
        )
        
        # 评估模型
        metrics = evaluate_model(model, X_test, y_test)
        
        # 记录结果
        result = {
            "name": params["name"],
            "params": params,
            "metrics": metrics,
            "history": history
        }
        results.append(result)
        
        # 打印结果
        print(f"超参数组合: {params['name']}")
        print(f"隐藏层: {params['hidden_layers']}")
        print(f"激活函数: {params['activation'].__name__}")
        print(f"优化器: {type(params['optimizer']).__name__}, 学习率: {params['optimizer'].learning_rate}")
        print(f"权重初始化: {params['weight_init']}")
        print(f"轮数: {params['epochs']}, 批次大小: {params['batch_size']}")
        print(f"测试 MSE: {metrics['mse']:.4f}")
        print(f"测试 MAE: {metrics['mae']:.4f}")
        print(f"测试 R2: {metrics['r2']:.4f}")
    
    # 比较所有结果
    print("\n" + "="*60)
    print("所有超参数组合的性能比较")
    print("="*60)
    
    # 按 R2 分数排序
    results.sort(key=lambda x: x["metrics"]["r2"], reverse=True)
    
    # 打印排序后的结果
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['name']}")
        print(f"   MSE: {result['metrics']['mse']:.4f}, MAE: {result['metrics']['mae']:.4f}, R2: {result['metrics']['r2']:.4f}")
    
    # 找出最佳超参数组合
    best_result = results[0]
    print("\n" + "="*60)
    print(f"最佳超参数组合: {best_result['name']}")
    print("="*60)
    print(f"隐藏层: {best_result['params']['hidden_layers']}")
    print(f"激活函数: {best_result['params']['activation'].__name__}")
    print(f"优化器: {type(best_result['params']['optimizer']).__name__}, 学习率: {best_result['params']['optimizer'].learning_rate}")
    print(f"权重初始化: {best_result['params']['weight_init']}")
    print(f"轮数: {best_result['params']['epochs']}, 批次大小: {best_result['params']['batch_size']}")
    print(f"测试 MSE: {best_result['metrics']['mse']:.4f}")
    print(f"测试 MAE: {best_result['metrics']['mae']:.4f}")
    print(f"测试 R2: {best_result['metrics']['r2']:.4f}")
    
    # 可视化不同超参数组合的性能
    visualize_results(results)
    
    print("\n=== 超参数调整示例完成! ===")

def visualize_results(results):
    """可视化不同超参数组合的性能"""
    print("\nGenerating visualization results...")
    
    plt.figure(figsize=(15, 8))
    
    # 1. Compare R2 scores of different hyperparameter combinations
    plt.subplot(2, 2, 1)
    names = [result["name"] for result in results]
    r2_scores = [result["metrics"]["r2"] for result in results]
    plt.barh(names, r2_scores, color='skyblue')
    plt.xlabel('R2 Score')
    plt.title('R2 Scores of Different Hyperparameter Combinations')
    plt.grid(axis='x', alpha=0.3)
    
    # 2. Compare MSE of different hyperparameter combinations
    plt.subplot(2, 2, 2)
    mse_scores = [result["metrics"]["mse"] for result in results]
    plt.barh(names, mse_scores, color='lightcoral')
    plt.xlabel('MSE')
    plt.title('MSE of Different Hyperparameter Combinations')
    plt.grid(axis='x', alpha=0.3)
    
    # 3. Plot training history of the best model
    best_result = results[0]
    plt.subplot(2, 2, 3)
    plt.plot(best_result["history"]["loss"], label='Training Loss')
    plt.plot(best_result["history"]["val_loss"], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training History of Best Model ({best_result["name"]})')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 4. Plot validation loss comparison of top 3 models
    plt.subplot(2, 2, 4)
    for result in results[:3]:  # Show only top 3 models
        plt.plot(result["history"]["val_loss"], label=result["name"])
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison of Top 3 Models')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
    print("可视化结果已保存到 hyperparameter_tuning_results.png")
    plt.show()

if __name__ == "__main__":
    main()
