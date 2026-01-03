import os
import json
import shutil
import sentencepiece as spm
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

class Tokenizer:
    """分词器抽象基类，定义统一接口"""
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inv_vocab = {}
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<s>': 2,
            '</s>': 3
        }
    
    def build_vocab(self, data: List[str]) -> None:
        """构建词汇表"""
        raise NotImplementedError
    
    def encode(self, text: str) -> List[int]:
        """文本转索引"""
        raise NotImplementedError
    
    def decode(self, indices: List[int]) -> str:
        """索引转文本"""
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        """保存分词器"""
        raise NotImplementedError
    
    def load(self, path: str) -> None:
        """加载分词器"""
        raise NotImplementedError

class BPETokenizer(Tokenizer):
    """BPE分词器实现"""
    def __init__(self, vocab_size: int = 8000):
        super().__init__(vocab_size)
        self.merges = {}
        self.vocab_size = vocab_size
    
    def get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """获取相邻字符对的统计"""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
    
    def merge_vocab(self, pair: Tuple[str, str], vocab_in: Dict[str, int]) -> Dict[str, int]:
        """合并最频繁的字符对"""
        vocab_out = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word_in, freq in vocab_in.items():
            word_out = word_in.replace(bigram, replacement)
            vocab_out[word_out] = freq
        return vocab_out
    
    def build_vocab(self, data: List[str]) -> None:
        """构建BPE词汇表"""
        print(f"开始构建BPE词汇表，目标大小: {self.vocab_size}")
        
        # 统计词频
        word_counts = Counter()
        for text in data:
            for word in text.split():  # 简单分词
                word_counts[word] += 1
        
        # 初始化词汇表，每个字符作为单独的标记
        vocab = {}
        for word, freq in word_counts.items():
            # 添加词尾标记</w>
            vocab[' '.join(word) + ' </w>'] = freq
        
        # 添加特殊标记
        current_vocab = {
            '<pad>': 0,
            '<unk>': 1,
            '<s>': 2,
            '</s>': 3
        }
        
        # 将单个字符添加到词汇表
        chars = set()
        for word in vocab.keys():
            chars.update(word.split())
        
        for char in chars:
            if char not in current_vocab:
                current_vocab[char] = len(current_vocab)
        
        # 执行BPE合并
        num_merges = max(0, self.vocab_size - len(current_vocab))
        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            self.merges[best] = i
            # 添加新的合并标记到词汇表
            merged_token = ''.join(best)
            if merged_token not in current_vocab:
                current_vocab[merged_token] = len(current_vocab)
        
        self.vocab = current_vocab
        self.inv_vocab = {v: k for k, v in current_vocab.items()}
        
        print(f"BPE词汇表构建完成，实际大小: {len(self.vocab)}")
    
    def encode(self, text: str) -> List[int]:
        """文本转索引"""
        tokens = []
        for word in text.split():
            # 将单词转换为字符列表
            word = list(word) + ['</w>']
            while len(word) > 1:
                # 找到最应该合并的相邻对
                pairs = [(tuple(word[i:i+2]), i) for i in range(len(word)-1)]
                best_pair = None
                best_idx = -1
                for pair, idx in pairs:
                    if pair in self.merges:
                        if best_pair is None or self.merges[pair] < self.merges[best_pair]:
                            best_pair = pair
                            best_idx = idx
                
                if best_pair is None:
                    break
                
                # 合并最佳对
                merged = ''.join(best_pair)
                word = word[:best_idx] + [merged] + word[best_idx+2:]
            
            # 将每个token转换为索引
            for token in word:
                tokens.append(self.vocab.get(token, self.vocab['<unk>']))
        
        # 添加开始和结束标记
        tokens = [self.vocab['<s>']] + tokens + [self.vocab['</s>']]
        return tokens
    
    def decode(self, indices: List[int]) -> str:
        """索引转文本"""
        tokens = []
        for idx in indices:
            if idx in self.inv_vocab:
                token = self.inv_vocab[idx]
                if token == '<pad>':
                    continue
                elif token == '<s>' or token == '</s>':
                    continue
                elif token.endswith('</w>'):
                    tokens.append(token[:-4] + ' ')
                else:
                    tokens.append(token)
        
        return ''.join(tokens).strip()
    
    def save(self, path: str) -> None:
        """保存分词器"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'vocab': self.vocab,
            'merges': {str(k): v for k, v in self.merges.items()},
            'vocab_size': self.vocab_size
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    
    def load(self, path: str) -> None:
        """加载分词器"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.merges = {tuple(eval(k)): v for k, v in data['merges'].items()}
        self.vocab_size = data['vocab_size']
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

class SentencePieceTokenizer(Tokenizer):
    """SentencePiece分词器实现"""
    def __init__(self, vocab_size: int = 8000):
        super().__init__(vocab_size)
        self.model_prefix = 'spm_model'
        self.sp = None
    
    def build_vocab(self, data: List[str]) -> None:
        """构建SentencePiece词汇表"""
        print(f"开始构建SentencePiece词汇表，目标大小: {self.vocab_size}")
        
        # 创建临时训练文件
        temp_file = 'temp_train.txt'
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in data:
                f.write(text + '\n')
        
        # 训练SentencePiece模型
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            model_type='unigram',  # 使用unigram模型
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece='<pad>',
            unk_piece='<unk>',
            bos_piece='<s>',
            eos_piece='</s>',
            character_coverage=1.0
        )
        
        # 加载训练好的模型
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(f'{self.model_prefix}.model')
        
        # 构建词汇表
        self.vocab = {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}
        self.inv_vocab = {i: self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())}
        
        # 删除临时文件
        os.remove(temp_file)
        
        print(f"SentencePiece词汇表构建完成，实际大小: {self.sp.get_piece_size()}")
    
    def encode(self, text: str) -> List[int]:
        """文本转索引"""
        if self.sp is None:
            raise ValueError("分词器未初始化，请先构建词汇表或加载模型")
        return self.sp.encode(text, add_bos=True, add_eos=True)
    
    def decode(self, indices: List[int]) -> str:
        """索引转文本"""
        if self.sp is None:
            raise ValueError("分词器未初始化，请先构建词汇表或加载模型")
        return self.sp.decode(indices)
    
    def save(self, path: str) -> None:
        """保存分词器"""
        if self.sp is None:
            raise ValueError("分词器未初始化，请先构建词汇表")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_path = path + '.model'
        vocab_path = path + '.vocab'
        
        # 复制模型文件
        shutil.copy(f'{self.model_prefix}.model', model_path)
        shutil.copy(f'{self.model_prefix}.vocab', vocab_path)
        
        print(f"SentencePiece分词器已保存到 {path}.model 和 {path}.vocab")
    
    def load(self, path: str) -> None:
        """加载分词器"""
        model_path = path + '.model'
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        
        # 重新构建词汇表
        self.vocab = {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}
        self.inv_vocab = {i: self.sp.id_to_piece(i) for i in range(self.sp.get_piece_size())}
        
        print(f"SentencePiece分词器已从 {model_path} 加载")

# 创建分词器工厂函数
def get_tokenizer(tokenizer_type: str, vocab_size: int = 8000) -> Tokenizer:
    """获取分词器实例"""
    if tokenizer_type == 'bpe':
        return BPETokenizer(vocab_size)
    elif tokenizer_type == 'sentencepiece':
        return SentencePieceTokenizer(vocab_size)
    else:
        raise ValueError(f"不支持的分词器类型: {tokenizer_type}")
