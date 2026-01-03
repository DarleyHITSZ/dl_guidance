import subprocess
import time
import json
import os
from typing import List, Dict

class ComparisonExperiment:
    """多模型对比实验"""
    def __init__(self):
        self.experiments = []
        self.results = []
        self.experiment_results_file = 'experiment_results.json'
    
    def add_experiment(self, model: str, pos_encoding: str = 'sine', tokenizer: str = 'bpe', 
                       quantize: str = None, description: str = '') -> None:
        """添加一个实验配置"""
        experiment = {
            'model': model,
            'pos_encoding': pos_encoding,
            'tokenizer': tokenizer,
            'quantize': quantize,
            'description': description
        }
        self.experiments.append(experiment)
    
    def build_command(self, experiment: Dict[str, str]) -> List[str]:
        """构建实验命令"""
        command = [
            'python', 'training_and_testing.py',
            '--device', 'cpu',  # 默认使用CPU
            '--model', experiment['model'],
            '--pos_encoding', experiment['pos_encoding'],
            '--tokenizer', experiment['tokenizer'],
            '--batch_size', '8',
            '--epochs', '5',  # 减少轮次以加快对比实验速度
            '--d_model', '128',  # 减小模型维度以加快速度
            '--num_layers', '2',
            '--num_heads', '2'
        ]
        
        if experiment['quantize']:
            command.extend(['--quantize', experiment['quantize']])
        
        return command
    
    def run_experiment(self, experiment: Dict[str, str]) -> Dict[str, any]:
        """运行单个实验"""
        print(f"\n=== 运行实验: {experiment['description']} ===")
        print(f"配置: {experiment}")
        
        command = self.build_command(experiment)
        print(f"命令: {' '.join(command)}")
        
        start_time = time.time()
        
        # 运行命令，捕获字节输出以避免UnicodeDecodeError
        result = subprocess.run(command, capture_output=True, text=False)
        
        # 显式解码输出
        stdout = result.stdout.decode('utf-8', errors='replace')
        stderr = result.stderr.decode('utf-8', errors='replace')
        
        experiment_time = time.time() - start_time
        
        print(f"实验耗时: {experiment_time:.2f} 秒")
        
        if result.returncode != 0:
            print(f"实验失败: {stderr}")
            return {
                'experiment': experiment,
                'success': False,
                'error': stderr
            }
        
        print(f"实验成功: {stdout}")
        
        # 从结果文件中获取最新的实验结果
        if os.path.exists(self.experiment_results_file):
            with open(self.experiment_results_file, 'r') as f:
                all_results = json.load(f)
            if all_results:
                return {
                    'experiment': experiment,
                    'success': True,
                    'result': all_results[-1]
                }
        
        return {
            'experiment': experiment,
            'success': True,
            'result': None
        }
    
    def run_all_experiments(self) -> None:
        """运行所有实验"""
        print("开始多模型对比实验...")
        total_start_time = time.time()
        
        for i, experiment in enumerate(self.experiments):
            print(f"\n\n===== 实验 {i+1}/{len(self.experiments)} =====")
            result = self.run_experiment(experiment)
            self.results.append(result)
        
        total_time = time.time() - total_start_time
        
        print(f"\n\n=== 所有实验完成 ===")
        print(f"总耗时: {total_time:.2f} 秒")
        
        # 打印总结
        self.print_summary()
        
        # 保存实验结果
        self.save_results()
    
    def print_summary(self) -> None:
        """打印实验总结"""
        print("\n\n=== 实验总结 ===")
        
        for i, result in enumerate(self.results):
            experiment = result['experiment']
            status = "成功" if result['success'] else "失败"
            
            print(f"\n实验 {i+1}: {experiment['description']}")
            print(f"状态: {status}")
            
            if result['success'] and result['result']:
                print(f"测试损失: {result['result'].get('final_test_loss', 'N/A'):.4f}")
                print(f"训练时间: {result['result'].get('training_time_seconds', 'N/A'):.2f} 秒")
                print(f"显存使用: {result['result'].get('memory_usage_mb', 'N/A'):.2f} MB")
                print(f"推理延迟: {result['result'].get('inference_time_ms', 'N/A'):.2f} ms")
            elif not result['success']:
                print(f"错误信息: {result['error'][:100]}...")
    
    def save_results(self) -> None:
        """保存实验结果"""
        summary_file = 'comparison_summary.json'
        
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'experiments': self.experiments,
            'results': self.results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n实验总结已保存到 {summary_file}")
    
    def load_previous_results(self) -> List[Dict[str, any]]:
        """加载之前的实验结果"""
        if os.path.exists(self.experiment_results_file):
            with open(self.experiment_results_file, 'r') as f:
                return json.load(f)
        return []

def main():
    """主函数"""
    # 创建实验实例
    experiment = ComparisonExperiment()
    
    # 添加实验配置
    # 1. 原始Transformer（正弦位置编码，BPE分词器）
    experiment.add_experiment(
        model='transformer',
        pos_encoding='sine',
        tokenizer='bpe',
        quantize=None,
        description='原始Transformer (正弦位置编码, BPE分词器)'
    )
    
    # 2. Transformer+RoPE（旋转位置编码）
    experiment.add_experiment(
        model='transformer',
        pos_encoding='rope',
        tokenizer='bpe',
        quantize=None,
        description='Transformer+RoPE (旋转位置编码)'
    )
    
    # 3. Transformer+SentencePiece（SentencePiece分词器）
    experiment.add_experiment(
        model='transformer',
        pos_encoding='sine',
        tokenizer='sentencepiece',
        quantize=None,
        description='Transformer+SentencePiece (SentencePiece分词器)'
    )
    
    # 4. Mamba模型
    experiment.add_experiment(
        model='mamba',
        pos_encoding='sine',  # Mamba模型忽略位置编码参数
        tokenizer='bpe',
        quantize=None,
        description='Mamba模型'
    )
    
    # 5. 可选：量化实验
    # experiment.add_experiment(
    #     model='transformer',
    #     pos_encoding='sine',
    #     tokenizer='bpe',
    #     quantize='int8',
    #     description='Transformer (INT8量化)'
    # )
    
    # 运行所有实验
    experiment.run_all_experiments()

if __name__ == '__main__':
    main()
