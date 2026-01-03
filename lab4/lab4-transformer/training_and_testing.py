import argparse
import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Tuple

from download_dataset import WMT14DatasetDownloader
from tokenizers import get_tokenizer
from transformer_model import get_transformer
from mamba_model import get_mamba

class TranslationDataset(Dataset):
    """翻译数据集"""
    def __init__(self, data: List[Tuple[str, str]], src_tokenizer, tgt_tokenizer, max_len: int = 50):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回源语言和目标语言的编码序列"""
        src_text, tgt_text = self.data[idx]
        
        # 编码
        src_indices = self.src_tokenizer.encode(src_text)
        tgt_indices = self.tgt_tokenizer.encode(tgt_text)
        
        # 截断过长序列
        src_indices = src_indices[:self.max_len]
        tgt_indices = tgt_indices[:self.max_len]
        
        # 创建掩码
        src_mask = torch.ones(len(src_indices), dtype=torch.long)
        tgt_mask = torch.ones(len(tgt_indices), dtype=torch.long)
        
        # 转换为张量
        src = torch.tensor(src_indices, dtype=torch.long)
        tgt = torch.tensor(tgt_indices, dtype=torch.long)
        
        return src, tgt, src_mask, tgt_mask

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """批处理函数"""
    src_batch, tgt_batch, src_mask_batch, tgt_mask_batch = zip(*batch)
    
    # 计算最大长度
    max_src_len = max(len(src) for src in src_batch)
    max_tgt_len = max(len(tgt) for tgt in tgt_batch)
    
    # 填充到最大长度
    src_padded = []
    src_mask_padded = []
    for src, mask in zip(src_batch, src_mask_batch):
        pad_len = max_src_len - len(src)
        src_padded.append(torch.cat([src, torch.zeros(pad_len, dtype=torch.long)]))
        src_mask_padded.append(torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)]))
    
    tgt_padded = []
    tgt_mask_padded = []
    for tgt, mask in zip(tgt_batch, tgt_mask_batch):
        pad_len = max_tgt_len - len(tgt)
        tgt_padded.append(torch.cat([tgt, torch.zeros(pad_len, dtype=torch.long)]))
        tgt_mask_padded.append(torch.cat([mask, torch.zeros(pad_len, dtype=torch.long)]))
    
    return (
        torch.stack(src_padded),
        torch.stack(tgt_padded),
        torch.stack(src_mask_padded),
        torch.stack(tgt_mask_padded)
    )

def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device, epoch: int) -> float:
    """训练一个epoch"""
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (src, tgt, src_mask, tgt_mask) in enumerate(dataloader):
        # 移动到设备
        src = src.to(device)
        tgt = tgt.to(device)
        
        # 准备输入和目标
        tgt_input = tgt[:, :-1]  # 不包括最后一个token
        tgt_expected = tgt[:, 1:]  # 不包括第一个token
        
        # 前向传播
        optimizer.zero_grad()
        output = model(src, tgt_input)
        
        # 计算损失
        output = output.reshape(-1, output.shape[-1])
        tgt_expected = tgt_expected.reshape(-1)
        loss = criterion(output, tgt_expected)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 打印进度
        if batch_idx % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, Time: {elapsed:.2f}s")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def test(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """测试模型"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt, src_mask, tgt_mask in dataloader:
            # 移动到设备
            src = src.to(device)
            tgt = tgt.to(device)
            
            # 准备输入和目标
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]
            
            # 前向传播
            output = model(src, tgt_input)
            
            # 计算损失
            output = output.reshape(-1, output.shape[-1])
            tgt_expected = tgt_expected.reshape(-1)
            loss = criterion(output, tgt_expected)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def translate(model: nn.Module, src_text: str, src_tokenizer, tgt_tokenizer, device: torch.device,
              max_len: int = 50) -> str:
    """使用模型进行翻译"""
    model.eval()
    
    # 编码源文本
    src_indices = src_tokenizer.encode(src_text)
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)
    
    # 初始化目标序列
    tgt_indices = [tgt_tokenizer.special_tokens['<s>']]  # 开始标记
    
    # 生成翻译
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_indices, dtype=torch.long).unsqueeze(0).to(device)
        
        # 前向传播
        output = model(src_tensor, tgt_tensor)
        
        # 取最后一个token的预测
        next_token = output[0, -1, :].argmax().item()
        
        # 添加到目标序列
        tgt_indices.append(next_token)
        
        # 检查是否到达结束标记
        if next_token == tgt_tokenizer.special_tokens['</s>']:
            break
    
    # 解码
    translation = tgt_tokenizer.decode(tgt_indices)
    return translation

def get_memory_usage(device: torch.device) -> float:
    """获取显存使用量（MB）"""
    if device.type == 'cuda':
        return torch.cuda.memory_allocated(device) / 1024 / 1024
    return 0

def get_inference_time(model: nn.Module, src: torch.Tensor, tgt_input: torch.Tensor, device: torch.device) -> float:
    """获取推理时间（毫秒）"""
    model.eval()
    
    with torch.no_grad():
        start_time = time.time()
        model(src, tgt_input)
        end_time = time.time()
    
    return (end_time - start_time) * 1000

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Transformer训练与测试脚本')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='使用的设备')
    parser.add_argument('--model', type=str, default='transformer', choices=['transformer', 'mamba'], help='模型类型')
    parser.add_argument('--pos_encoding', type=str, default='sine', choices=['sine', 'rope', 'learnable'], help='位置编码类型')
    parser.add_argument('--tokenizer', type=str, default='bpe', choices=['bpe', 'sentencepiece'], help='分词器类型')
    parser.add_argument('--quantize', type=str, default=None, choices=['int8'], help='量化类型')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮次')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--d_model', type=int, default=256, help='模型维度')
    parser.add_argument('--num_layers', type=int, default=2, help='层数')
    parser.add_argument('--num_heads', type=int, default=4, help='头数')
    parser.add_argument('--max_len', type=int, default=50, help='最大序列长度')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 数据集准备
    print("\n1. 准备数据集...")
    downloader = WMT14DatasetDownloader()
    downloader.run()
    train_data, test_data = downloader.load_processed_data()
    
    # 2. 分词器初始化
    print("\n2. 初始化分词器...")
    
    # 收集训练数据用于构建词汇表
    all_train_text = []
    for de, en in train_data:
        all_train_text.append(de)
        all_train_text.append(en)
    
    # 根据分词器类型设置合适的词汇表大小
    if args.tokenizer == 'sentencepiece':
        # 对于SentencePiece，根据数据量动态调整词汇表大小
        vocab_size = min(200, len(set(' '.join(all_train_text))))
    else:
        vocab_size = 8000
    
    print(f"使用词汇表大小: {vocab_size}")
    
    # 初始化分词器
    src_tokenizer = get_tokenizer(args.tokenizer, vocab_size)
    tgt_tokenizer = get_tokenizer(args.tokenizer, vocab_size)
    
    # 构建词汇表
    src_tokenizer.build_vocab([de for de, en in train_data])
    tgt_tokenizer.build_vocab([en for de, en in train_data])
    
    # 3. 数据集加载
    print("\n3. 加载数据集...")
    train_dataset = TranslationDataset(train_data, src_tokenizer, tgt_tokenizer, args.max_len)
    test_dataset = TranslationDataset(test_data, src_tokenizer, tgt_tokenizer, args.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    
    # 4. 模型初始化
    print("\n4. 初始化模型...")
    src_vocab_size = len(src_tokenizer.vocab)
    tgt_vocab_size = len(tgt_tokenizer.vocab)
    
    if args.model == 'transformer':
        model = get_transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            pos_encoding=args.pos_encoding,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_model * 4,
            quantize=args.quantize is not None
        )
    else:  # mamba
        model = get_mamba(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=args.d_model,
            num_layers=args.num_layers,
            d_state=16,
            expand_factor=2
        )
    
    model.to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 5. 训练配置
    print("\n5. 配置训练...")
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略pad标记
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 6. 量化配置
    if args.quantize == 'int8':
        print("\n6. 配置INT8量化...")
        if hasattr(model, 'prepare_for_quantization'):
            model.prepare_for_quantization()
            print("模型已准备好进行量化")
        else:
            print("警告: 该模型不支持量化")
            args.quantize = None
    
    # 7. 训练过程
    print("\n7. 开始训练...")
    training_start_time = time.time()
    
    train_losses = []
    test_losses = []
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch+1)
        test_loss = test(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    training_time = time.time() - training_start_time
    
    # 8. 完成量化转换
    if args.quantize == 'int8' and hasattr(model, 'convert_to_quantized'):
        print("\n8. 完成量化转换...")
        model.convert_to_quantized()
        print("模型已转换为INT8量化版本")
    
    # 9. 性能评估
    print("\n9. 性能评估...")
    
    # 测试集损失
    final_test_loss = test(model, test_loader, criterion, device)
    print(f"最终测试损失: {final_test_loss:.4f}")
    
    # 显存使用
    memory_usage = get_memory_usage(device)
    print(f"显存使用: {memory_usage:.2f} MB")
    
    # 推理延迟
    if len(test_loader) > 0:
        src, tgt, _, _ = next(iter(test_loader))
        src = src.to(device)[:1]
        tgt_input = tgt.to(device)[:1, :-1]
        inference_time = get_inference_time(model, src, tgt_input, device)
        print(f"推理延迟: {inference_time:.2f} ms")
    else:
        inference_time = 0
    
    # 10. 翻译示例
    print("\n10. 翻译示例...")
    if len(test_data) > 0:
        src_text, tgt_text = test_data[0]
        translation = translate(model, src_text, src_tokenizer, tgt_tokenizer, device)
        print(f"源文本: {src_text}")
        print(f"参考译文: {tgt_text}")
        print(f"模型翻译: {translation}")
    
    # 11. 保存结果
    print("\n11. 保存结果...")
    results = {
        'model': args.model,
        'pos_encoding': args.pos_encoding if args.model == 'transformer' else None,
        'tokenizer': args.tokenizer,
        'quantize': args.quantize,
        'device': str(device),
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'final_test_loss': final_test_loss,
        'training_time_seconds': training_time,
        'memory_usage_mb': memory_usage,
        'inference_time_ms': inference_time,
        'train_losses': train_losses,
        'test_losses': test_losses
    }
    
    # 保存到文件
    results_file = 'experiment_results.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    all_results.append(results)
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
    
    print(f"结果已保存到 {results_file}")
    
    print("\n训练和测试完成！")

if __name__ == '__main__':
    main()
