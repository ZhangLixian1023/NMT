import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .embedding import LayerNorm, RMSNorm, PositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, norm_type='layernorm', relative_position=False):
        """
        Transformer编码器层
        :param d_model: 模型维度
        :param n_heads: 头数
        :param d_ff: 前馈网络隐藏层维度
        :param dropout: dropout概率
        :param norm_type: 归一化类型，可选：layernorm, rmsnorm
        :param relative_position: 是否使用相对位置嵌入
        """
        super(EncoderLayer, self).__init__()
        
        # 归一化层
        if norm_type == 'layernorm':
            self.norm1 = LayerNorm(d_model)
            self.norm2 = LayerNorm(d_model)
        else:  # rmsnorm
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        # 保存归一化类型以便在forward中选择处理顺序
        self.norm_type = norm_type
        
        # 多头自注意力机制
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, relative_position)
        
        # 前馈神经网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        编码器层前向传播
        :param x: 输入序列，形状：(batch_size, seq_len, d_model)
        :param mask: 掩码矩阵，形状：(batch_size, seq_len, seq_len)
        :return: 编码器层输出
        """
        # 根据归一化类型选择不同的子层顺序：
        # - RMSNorm (pre-norm): Norm -> Attention -> Add ; Norm -> FFN -> Add
        # - LayerNorm (post-norm): Attention -> Add -> LayerNorm ; FFN -> Add -> LayerNorm
        if self.norm_type == 'rmsnorm':
            # pre-norm: apply RMSNorm before each sublayer
            x_norm = self.norm1(x)
            attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, mask)
            x = x + self.dropout1(attn_output)

            ff_input = self.norm2(x)
            ff_output = self.feed_forward(ff_input)
            x = x + self.dropout2(ff_output)

            return x
        else:
            # post-norm: apply attention/ffn then norm
            attn_output, _ = self.self_attn(x, x, x, mask)
            x = self.norm1(x + self.dropout1(attn_output))
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout2(ff_output))

            return x

class Encoder(nn.Module):
    def __init__(self, input_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, 
                 norm_type='layernorm', relative_position=False):
        """
        Transformer编码器
        :param input_size: 输入词汇表大小
        :param d_model: 模型维度
        :param n_layers: 层数
        :param n_heads: 头数
        :param d_ff: 前馈网络隐藏层维度
        :param dropout: dropout概率
        :param norm_type: 归一化类型，可选：layernorm, rmsnorm
        :param relative_position: 位置嵌入类型，可选：False,True
        """
        super(Encoder, self).__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(input_size, d_model)
        
        self.relative_position = relative_position
        # 位置嵌入
        if not relative_position:
            self.positional_encoding = PositionalEncoding(d_model)
        
        # 编码器层列表
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, norm_type, relative_position)
            for _ in range(n_layers)
        ])
        
        # 归一化层
        if norm_type == 'rmsnorm':
            self.norm = RMSNorm(d_model)
        # 保存归一化类型
        self.norm_type = norm_type
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, mask=None):
        """
        编码器前向传播
        :param src: 源语言序列，形状：(batch_size, seq_len)
        :param mask: 掩码矩阵，形状：(batch_size, 1, 1, seq_len)
        :return: 编码器输出
        """
        # 嵌入层：(batch_size,seq_len) → (batch_size,seq_len, d_model)
        x = self.dropout(self.embedding(src))
        
        # 添加位置嵌入
        if not self.relative_position:
            x = self.positional_encoding(x)
        
        # 经过所有编码器层
        for layer in self.layers:
            x = layer(x, mask)

        # 最终归一化：根据归一化类型显式分支（保持行为一致）
        if self.norm_type == 'rmsnorm':
            x = self.norm(x)

        return x
