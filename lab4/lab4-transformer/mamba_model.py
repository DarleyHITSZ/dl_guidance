import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class MambaStateSpaceLayer(nn.Module):
    """Mamba状态空间层"""
    def __init__(self, d_model: int, d_state: int = 16, expand_factor: int = 2):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.expand_factor = expand_factor
        self.d_inner = d_model * expand_factor
        
        # 输入投影层
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # 状态空间参数
        self.A_log = nn.Parameter(torch.ones(self.d_inner, self.d_state))
        self.B = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.C = nn.Parameter(torch.randn(self.d_inner, self.d_state))
        self.D = nn.Parameter(torch.randn(self.d_inner))
        
        # 门控层
        self.gate_proj = nn.Linear(d_model, self.d_inner)
        
        # 输出投影层
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self) -> None:
        """初始化模型参数"""
        nn.init.kaiming_uniform_(self.in_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.out_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.gate_proj.weight, a=math.sqrt(5))
        
        nn.init.uniform_(self.A_log, -3, -1)
        nn.init.normal_(self.B, std=0.02)
        nn.init.normal_(self.C, std=0.02)
        nn.init.normal_(self.D, std=0.02)
    
    def selective_scan(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """选择性扫描操作"""
        # x: (batch_size, seq_len, d_inner)
        # delta: (batch_size, seq_len, d_inner)
        
        batch_size, seq_len, d_inner = x.shape
        
        # 计算A矩阵
        A = -torch.exp(self.A_log).unsqueeze(0).unsqueeze(0)  # (1, 1, d_inner, d_state)
        
        # 计算delta和B的乘积
        delta_B = delta.unsqueeze(-1) * self.B.unsqueeze(0).unsqueeze(0)  # (batch_size, seq_len, d_inner, d_state)
        
        # 计算输入x与delta_B的乘积
        x_B = x.unsqueeze(-1) * delta_B  # (batch_size, seq_len, d_inner, d_state)
        
        # 初始化状态
        state = torch.zeros(batch_size, d_inner, self.d_state, device=x.device)
        
        # 序列处理（递归计算状态）
        output = []
        for t in range(seq_len):
            # 更新状态
            state = state * torch.exp(A[:, :, :, :] * delta[:, t:t+1, :, None]) + x_B[:, t:t+1, :, :]
            # 计算输出
            y_t = (state @ self.C.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2))[:, 0, :, 0]
            output.append(y_t)
        
        output = torch.stack(output, dim=1)  # (batch_size, seq_len, d_inner)
        
        # 加上D项
        output = output + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # x: (batch_size, seq_len, d_model)
        
        # 计算门控
        gate = torch.sigmoid(self.gate_proj(x))  # (batch_size, seq_len, d_inner)
        
        # 输入投影
        projected = self.in_proj(x)  # (batch_size, seq_len, d_inner * 2)
        x_proj, delta_proj = projected.chunk(2, dim=-1)  # 各为 (batch_size, seq_len, d_inner)
        
        # 计算delta
        delta = torch.relu(delta_proj)  # (batch_size, seq_len, d_inner)
        
        # 选择性扫描
        ss_output = self.selective_scan(x_proj, delta)  # (batch_size, seq_len, d_inner)
        
        # 应用门控
        gated_output = gate * ss_output  # (batch_size, seq_len, d_inner)
        
        # 输出投影
        output = self.out_proj(gated_output)  # (batch_size, seq_len, d_model)
        
        return output

class MambaBlock(nn.Module):
    """Mamba块"""
    def __init__(self, d_model: int, d_state: int = 16, expand_factor: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.mamba_layer = MambaStateSpaceLayer(d_model, d_state, expand_factor)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        residual = x
        x = self.norm(x)
        x = self.mamba_layer(x)
        x = self.dropout(x)
        return x + residual

class MambaEncoder(nn.Module):
    """Mamba编码器"""
    def __init__(self, num_layers: int, d_model: int, d_state: int = 16, expand_factor: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, expand_factor, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class MambaDecoder(nn.Module):
    """Mamba解码器"""
    def __init__(self, num_layers: int, d_model: int, d_state: int = 16, expand_factor: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, expand_factor, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
        # 跨注意力层（简化版，只使用前馈网络模拟）
        self.cross_attn_approx = nn.Linear(d_model * 2, d_model)
    
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # x: (batch_size, tgt_seq_len, d_model)
        # enc_output: (batch_size, src_seq_len, d_model)
        
        # 简化的跨注意力：将编码器输出的平均值与解码器输入拼接
        enc_mean = enc_output.mean(dim=1, keepdim=True).expand(-1, x.size(1), -1)
        x_with_enc = torch.cat([x, enc_mean], dim=-1)
        x = x + self.cross_attn_approx(x_with_enc)
        
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class Mamba(nn.Module):
    """简化版Mamba模型"""
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512, num_layers: int = 6,
                 d_state: int = 16, expand_factor: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 编码器和解码器
        self.encoder = MambaEncoder(num_layers, d_model, d_state, expand_factor, dropout)
        self.decoder = MambaDecoder(num_layers, d_model, d_state, expand_factor, dropout)
        
        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # 位置编码（可选）
        self.pos_encoding = nn.Embedding(5000, d_model)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self) -> None:
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        nn.init.uniform_(self.fc_out.weight, -0.02, 0.02)
        nn.init.zeros_(self.fc_out.bias)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播"""
        # src: (batch_size, src_seq_len)
        # tgt: (batch_size, tgt_seq_len)
        
        batch_size, src_seq_len = src.shape
        batch_size, tgt_seq_len = tgt.shape
        
        # 添加位置编码
        src_pos = torch.arange(0, src_seq_len, device=src.device).unsqueeze(0).expand(batch_size, -1)
        tgt_pos = torch.arange(0, tgt_seq_len, device=tgt.device).unsqueeze(0).expand(batch_size, -1)
        
        # 嵌入
        src_embedded = self.src_embedding(src) + self.pos_encoding(src_pos)
        tgt_embedded = self.tgt_embedding(tgt) + self.pos_encoding(tgt_pos)
        
        # 编码和解码
        enc_output = self.encoder(src_embedded)
        dec_output = self.decoder(tgt_embedded, enc_output)
        
        # 输出层
        output = self.fc_out(dec_output)
        
        return output

# 创建Mamba模型工厂函数
def get_mamba(src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512, num_layers: int = 6,
              d_state: int = 16, expand_factor: int = 2, dropout: float = 0.1) -> nn.Module:
    """获取Mamba模型实例"""
    return Mamba(src_vocab_size, tgt_vocab_size, d_model, num_layers, d_state, expand_factor, dropout)
