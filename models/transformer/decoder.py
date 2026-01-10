import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .embedding import LayerNorm, RMSNorm, PositionalEncoding

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, norm_type='layernorm', relative_position=False):
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
        # 保存归一化类型以便在forward中选择处理顺序
        self.norm_type = norm_type
        
        # 多头自注意力机制（带掩码）
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, relative_position)
        
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
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        解码器层前向传播
        :param x: 输入序列，形状：(batch_size, seq_len, d_model)
        :param enc_output: 编码器输出，形状：(batch_size,seq_len,  d_model)
        :param src_mask: 源语言掩码矩阵，形状：(batch_size, seq_len, seq_len)
        :param tgt_mask: 目标语言掩码矩阵，形状：(batch_size, seq_len, seq_len)
        :return: 解码器层输出
        """
        # 根据归一化类型选择不同的子层顺序：
        # - RMSNorm (pre-norm): Norm -> Attention -> Add ; Norm -> Cross-Attn -> Add ; Norm -> FFN -> Add
        # - LayerNorm (post-norm): Attention -> Add -> LayerNorm ; Cross-Attn -> Add -> LayerNorm ; FFN -> Add -> LayerNorm
        if self.norm_type == 'rmsnorm':
            # pre-norm for self-attention
            x_norm = self.norm1(x)
            attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, tgt_mask)
            x = x + self.dropout1(attn_output)

            # pre-norm for cross-attention
            x_norm = self.norm2(x)
            attn_output, _ = self.cross_attn(x_norm, enc_output, enc_output, src_mask)
            x = x + self.dropout2(attn_output)

            # pre-norm for feed-forward
            ff_input = self.norm3(x)
            ff_output = self.feed_forward(ff_input)
            x = x + self.dropout3(ff_output)

            return x
        else:
            # post-norm ordering
            attn_output, _ = self.self_attn(x, x, x, tgt_mask)
            x = x + self.dropout1(attn_output)
            x = self.norm1(x)

            attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
            x = x + self.dropout2(attn_output)
            x = self.norm2(x)

            ff_output = self.feed_forward(x)
            x = x + self.dropout3(ff_output)
            x = self.norm3(x)

            return x

class Decoder(nn.Module):
    def __init__(self, output_size, d_model, n_layers, n_heads, d_ff, dropout=0.1, 
                 norm_type='layernorm', relative_position=False):
        """
        Transformer解码器
        :param output_size: 输出词汇表大小
        :param d_model: 模型维度
        :param n_layers: 层数
        :param n_heads: 头数
        :param d_ff: 前馈网络隐藏层维度
        :param dropout: dropout概率
        :param norm_type: 归一化类型，可选：layernorm, rmsnorm
        :param relative_position: 位置嵌入类型，可选：True,False
        """
        super(Decoder, self).__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(output_size, d_model)
        
        self.relative_position = relative_position
        # 位置嵌入
        if not relative_position:
            self.positional_encoding = PositionalEncoding(d_model)
        
        # 解码器层列表
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, norm_type,relative_position)
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.fc_out = nn.Linear(d_model, output_size)
        
        # 归一化层
        if norm_type == 'rmsnorm':
            self.norm = RMSNorm(d_model)
        # 保存归一化类型
        self.norm_type = norm_type
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
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

        # 添加位置嵌入
        if not self.relative_position:
            x = self.positional_encoding(x)
        
        # 经过所有解码器层
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        # 最终归一化：根据归一化类型显式分支（保持行为一致）
        if self.norm_type == 'rmsnorm':
            x = self.norm(x)

        # 输出层：(batch_size, seq_len, d_model) → (batch_size, seq_len, output_size)
        output = self.fc_out(x)

        return output
