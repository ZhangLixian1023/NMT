import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .embedding import LayerNorm, RMSNorm, PositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, norm_type='layernorm', use_relative_embedding=False):
        """
        Transformer编码器层
        :param d_model: 模型维度
        :param n_heads: 头数
        :param d_ff: 前馈网络隐藏层维度
        :param dropout: dropout概率
        :param norm_type: 归一化类型，可选：layernorm, rmsnorm
        :param use_relative_embedding: 是否使用相对位置嵌入
        """
        super(EncoderLayer, self).__init__()
        
        # 归一化层
        if norm_type == 'layernorm':
            self.norm1 = LayerNorm(d_model)
            self.norm2 = LayerNorm(d_model)
        else:  # rmsnorm
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        
        # 多头自注意力机制
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, use_relative_embedding)
        
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
    
    def forward(self, x, mask=None, relative_positions=None, relative_embeddings=None):
        """
        编码器层前向传播
        :param x: 输入序列，形状：(seq_len, batch_size, d_model)
        :param mask: 掩码矩阵，形状：(batch_size, seq_len, seq_len)
        :param relative_positions: 相对位置索引，形状：(seq_len, seq_len)
        :param relative_embeddings: 相对位置嵌入，形状：(seq_len, seq_len, d_k)
        :return: 编码器层输出
        """
        # 多头自注意力子层
        attn_output, _ = self.self_attn(x, x, x, mask, relative_positions, relative_embeddings)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 前馈神经网络子层
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x

class Encoder(nn.Module):
    def __init__(self, input_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, 
                 norm_type='layernorm', embedding_type='absolute'):
        """
        Transformer编码器
        :param input_size: 输入词汇表大小
        :param d_model: 模型维度
        :param n_layers: 层数
        :param n_heads: 头数
        :param d_ff: 前馈网络隐藏层维度
        :param dropout: dropout概率
        :param norm_type: 归一化类型，可选：layernorm, rmsnorm
        :param embedding_type: 位置嵌入类型，可选：absolute, relative
        """
        super(Encoder, self).__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(input_size, d_model)
        
        # 位置嵌入
        self.positional_encoding = PositionalEncoding(d_model, embedding_type=embedding_type)
        
        # 编码器层列表
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, norm_type, 
                       use_relative_embedding=(embedding_type == 'relative'))
            for _ in range(n_layers)
        ])
        
        # 归一化层
        if norm_type == 'layernorm':
            self.norm = LayerNorm(d_model)
        else:  # rmsnorm
            self.norm = RMSNorm(d_model)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 是否使用相对位置嵌入
        self.use_relative_embedding = (embedding_type == 'relative')
    
    def forward(self, src, mask=None):
        """
        编码器前向传播
        :param src: 源语言序列，形状：(seq_len, batch_size)
        :param mask: 掩码矩阵，形状：(batch_size, seq_len, seq_len)
        :return: 编码器输出
        """
        # 嵌入层：(seq_len, batch_size) → (seq_len, batch_size, d_model)
        x = self.dropout(self.embedding(src))
        
        # 添加位置嵌入
        x = self.positional_encoding(x)
        
        # 准备相对位置信息（如果使用）
        relative_positions = None
        relative_embeddings = None
        if self.use_relative_embedding:
            seq_len = src.size(0)
            relative_positions = self.positional_encoding.get_relative_positions(seq_len)
            relative_embeddings = self.positional_encoding.relative_embeddings(relative_positions)
        
        # 经过所有编码器层
        for layer in self.layers:
            x = layer(x, mask, relative_positions, relative_embeddings)
        
        # 最终归一化
        x = self.norm(x)
        
        return x
