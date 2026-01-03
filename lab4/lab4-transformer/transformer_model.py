import torch
import torch.nn as nn
import math
from typing import Optional, Any

class PositionalEncoding(nn.Module):
    """位置编码基类"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

class SinusoidalPositionalEncoding(PositionalEncoding):
    """正弦位置编码（原始Transformer默认）"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__(d_model, max_len)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch_size, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]

class RoPE(PositionalEncoding):
    """旋转位置编码（Rotary Position Embedding）"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__(d_model, max_len)
        
        # 生成旋转矩阵
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.max_len = max_len
    
    def get_rotary_matrix(self, seq_len: int, device: torch.device) -> tuple:
        """生成旋转矩阵"""
        position = torch.arange(seq_len, device=device)
        inv_freq = self.inv_freq.to(device)
        
        freqs = position[:, None] * inv_freq[None, :]
        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        
        return sin, cos
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """将向量的后半部分旋转180度"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch_size, seq_len, d_model)"""
        seq_len = x.size(1)
        sin, cos = self.get_rotary_matrix(seq_len, x.device)
        
        # 扩展sin和cos的维度以匹配x的形状
        # x: (batch_size, seq_len, d_model)
        # sin, cos: (seq_len, d_model//2)
        batch_size, _, d_model = x.size()
        
        # 重复sin和cos到d_model维度
        # (seq_len, d_model//2) -> (seq_len, d_model)
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
        
        # 扩展到batch_size维度
        # (seq_len, d_model) -> (1, seq_len, d_model) -> (batch_size, seq_len, d_model)
        cos = cos.unsqueeze(0).expand(batch_size, -1, -1)
        sin = sin.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 对x应用旋转编码
        x_rot = x * cos + self.rotate_half(x) * sin
        return x_rot

class LearnablePositionalEncoding(PositionalEncoding):
    """可学习位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__(d_model, max_len)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.xavier_uniform_(self.pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch_size, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_weights = None
    
    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """缩放点积注意力"""
        # q, k, v: (batch_size, num_heads, seq_len, d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # 确保掩码能够正确广播到scores的形状
            if len(mask.shape) == 2:  # (seq_len, seq_len) - 用于下三角掩码
                # 扩展为(batch_size, num_heads, seq_len, seq_len)
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif len(mask.shape) == 3:  # (batch_size, 1, seq_len) - 用于源掩码
                # 扩展为(batch_size, 1, seq_len, seq_len)
                mask = mask.unsqueeze(2)
            elif len(mask.shape) == 4:  # 已经是正确形状
                pass
            # 使用bool掩码进行更高效的操作
            if mask.dtype != torch.bool:
                mask = mask.bool()
            # 对掩码取反，因为我们要屏蔽掉不需要的位置
            scores = scores.masked_fill(~mask, -1e9)
        
        attn_probs = torch.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        self.attn_weights = attn_probs
        return torch.matmul(attn_probs, v)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """q, k, v: (batch_size, seq_len, d_model)"""
        batch_size = q.size(0)
        
        # 线性变换并分块
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 应用注意力
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)
        
        # 合并头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        attn_output = self.w_o(attn_output)
        
        return attn_output

class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    """编码器层"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力（带掩码）
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # 交叉注意力
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x

class Encoder(nn.Module):
    """编码器"""
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    """解码器"""
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)

class Transformer(nn.Module):
    """Transformer模型"""
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512, num_layers: int = 6,
                 num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1, pos_encoding: str = 'sine'):
        super().__init__()
        
        # 位置编码选择
        if pos_encoding == 'sine':
            self.src_pos_encoding = SinusoidalPositionalEncoding(d_model)
            self.tgt_pos_encoding = SinusoidalPositionalEncoding(d_model)
        elif pos_encoding == 'rope':
            self.src_pos_encoding = RoPE(d_model)
            self.tgt_pos_encoding = RoPE(d_model)
        elif pos_encoding == 'learnable':
            self.src_pos_encoding = LearnablePositionalEncoding(d_model)
            self.tgt_pos_encoding = LearnablePositionalEncoding(d_model)
        else:
            raise ValueError(f"不支持的位置编码类型: {pos_encoding}")
        
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, dropout)
        
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self) -> None:
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """生成下三角掩码"""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        # src: (batch_size, src_seq_len)
        # tgt: (batch_size, tgt_seq_len)
        
        # 处理掩码形状
        if src_mask is not None:
            # 如果src_mask形状是(batch_size, seq_len)，转换为(batch_size, 1, seq_len)
            if len(src_mask.shape) == 2:
                src_mask = src_mask.unsqueeze(1)
        else:
            src_mask = torch.ones((src.size(0), 1, src.size(1)), device=src.device)
            
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        
        # 嵌入和位置编码
        src_embedded = self.dropout(self.src_pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model)))
        tgt_embedded = self.dropout(self.tgt_pos_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model)))
        
        # 编码和解码
        enc_output = self.encoder(src_embedded, src_mask)
        dec_output = self.decoder(tgt_embedded, enc_output, src_mask, tgt_mask)
        
        # 输出层
        output = self.fc_out(dec_output)
        
        return output

class QuantizedTransformer(nn.Module):
    """量化Transformer模型"""
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512, num_layers: int = 6,
                 num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1, pos_encoding: str = 'sine'):
        super().__init__()
        
        # 创建原始Transformer模型（不直接继承，避免Embedding层被量化）
        self.transformer = Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            pos_encoding=pos_encoding
        )
        
        # 准备量化组件
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # 配置量化 - 使用动态量化配置
        self.qconfig = torch.quantization.default_dynamic_qconfig
        self.quantized = False
    
    def prepare_for_quantization(self) -> None:
        """准备模型进行量化"""
        # 动态量化不需要prepare阶段
        pass
    
    def convert_to_quantized(self) -> None:
        """将模型转换为量化版本 - 使用动态量化"""
        self.eval()
        # 只对encoder和decoder进行动态量化，保留embedding层为FP32
        self.transformer.encoder = torch.quantization.quantize_dynamic(
            self.transformer.encoder,
            {nn.Linear, nn.LayerNorm},  # 只量化线性层和LayerNorm
            dtype=torch.qint8
        )
        self.transformer.decoder = torch.quantization.quantize_dynamic(
            self.transformer.decoder,
            {nn.Linear, nn.LayerNorm},  # 只量化线性层和LayerNorm
            dtype=torch.qint8
        )
        self.quantized = True
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """量化模型前向传播"""
        # 使用原始transformer的forward方法
        return self.transformer(src, tgt, src_mask, tgt_mask, memory_mask)

# 创建Transformer模型工厂函数
def get_transformer(src_vocab_size: int, tgt_vocab_size: int, pos_encoding: str = 'sine', quantize: bool = False, **kwargs) -> nn.Module:
    """获取Transformer模型实例"""
    if quantize:
        return QuantizedTransformer(src_vocab_size, tgt_vocab_size, pos_encoding=pos_encoding, **kwargs)
    else:
        return Transformer(src_vocab_size, tgt_vocab_size, pos_encoding=pos_encoding, **kwargs)
