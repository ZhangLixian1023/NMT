import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .embedding import LayerNorm, RMSNorm, PositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, norm_type='layernorm', use_relative_embedding=False):
        """
        Transformer解码器层
        :param d_model: 模型维度
        :param n_heads: 头数
        :param d_ff: 前馈网络隐藏层维度
        :param dropout: dropout概率
        :param norm_type: 归一化类型，可选：layernorm, rmsnorm
        :param use_relative_embedding: 是否使用相对位置嵌入
        """
        super(DecoderLayer, self).__init__()
        
        # 归一化层
        if norm_type == 'layernorm':
            self.norm1 = LayerNorm(d_model)
            self.norm2 = LayerNorm(d_model)
            self.norm3 = LayerNorm(d_model)
        else:  # rmsnorm
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        
        # 多头自注意力机制（带掩码）
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, use_relative_embedding)
        
        # 编码器-解码器注意力机制
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
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
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None, 
                relative_positions=None, relative_embeddings=None):
        """
        解码器层前向传播
        :param x: 输入序列，形状：(batch_size, seq_len, d_model)
        :param enc_output: 编码器输出，形状：(batch_size,seq_len,  d_model)
        :param src_mask: 源语言掩码矩阵，形状：(batch_size, seq_len, seq_len)
        :param tgt_mask: 目标语言掩码矩阵，形状：(batch_size, seq_len, seq_len)
        :param relative_positions: 相对位置索引，形状：(seq_len, seq_len)
        :param relative_embeddings: 相对位置嵌入，形状：(seq_len, seq_len, d_k)
        :return: 解码器层输出
        """
        # 多头自注意力子层（带掩码）
        attn_output, _ = self.self_attn(x, x, x, tgt_mask, relative_positions, relative_embeddings)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 编码器-解码器注意力子层
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = x + self.dropout2(attn_output)
        x = self.norm2(x)
        
        # 前馈神经网络子层
        ff_output = self.feed_forward(x)
        x = x + self.dropout3(ff_output)
        x = self.norm3(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self, output_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, 
                 norm_type='layernorm', embedding_type='absolute'):
        """
        Transformer解码器
        :param output_size: 输出词汇表大小
        :param d_model: 模型维度
        :param n_layers: 层数
        :param n_heads: 头数
        :param d_ff: 前馈网络隐藏层维度
        :param dropout: dropout概率
        :param norm_type: 归一化类型，可选：layernorm, rmsnorm
        :param embedding_type: 位置嵌入类型，可选：absolute, relative
        """
        super(Decoder, self).__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(output_size, d_model)
        
        # 位置嵌入
        self.positional_encoding = PositionalEncoding(d_model, embedding_type=embedding_type)
        
        # 解码器层列表
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, norm_type, 
                       use_relative_embedding=(embedding_type == 'relative'))
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, output_size)
        
        # 归一化层
        if norm_type == 'layernorm':
            self.norm = LayerNorm(d_model)
        else:  # rmsnorm
            self.norm = RMSNorm(d_model)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 是否使用相对位置嵌入
        self.use_relative_embedding = (embedding_type == 'relative')
    
    def forward(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        """
        解码器前向传播
        :param tgt: 目标语言序列，形状：(batch_size, seq_len )
        :param enc_output: 编码器输出，形状：(batch_size, seq_len, d_model)
        :param src_mask: 源语言掩码矩阵，形状：(batch_size, seq_len, seq_len)
        :param tgt_mask: 目标语言掩码矩阵，形状：(batch_size, seq_len, seq_len)
        :return: 解码器输出
        """
        # 嵌入层：(batch_size, seq_len) → (batch_size, seq_len, d_model)
        x = self.dropout(self.embedding(tgt))
        
        # # 维度检查
        # print("\n\nDecoder 维度检查:")
        # print(f"tgt: {tgt.shape}")
        # print(f"enc_out: {enc_output.shape}")
        
        # print(f"src_mask: {src_mask.shape}")
        # print(f"tgt_mask: {tgt_mask.shape}")

        # 添加位置嵌入
        x = self.positional_encoding(x)
        
        # 准备相对位置信息（如果使用）
        relative_positions = None
        relative_embeddings = None
        if self.use_relative_embedding:
            seq_len = tgt.size(1)
            relative_positions = self.positional_encoding.get_relative_positions(seq_len)
            relative_embeddings = self.positional_encoding.relative_embeddings(relative_positions)
        
        # 经过所有解码器层
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask, relative_positions, relative_embeddings)
        
        # 最终归一化
        x = self.norm(x)
        
        # 输出层：(batch_size, seq_len, d_model) → (batch_size, seq_len, output_size)
        output = self.fc_out(x)
        
        return output
