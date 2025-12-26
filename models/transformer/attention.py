import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, use_relative_embedding=False):
        """
        多头注意力机制
        :param d_model: 模型维度
        :param n_heads: 头数
        :param dropout: dropout概率
        :param use_relative_embedding: 是否使用相对位置嵌入
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 是否使用相对位置嵌入
        self.use_relative_embedding = use_relative_embedding
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None, relative_positions=None, relative_embeddings=None):
        """
        缩放点积注意力
        :param Q: 查询矩阵，形状：(batch_size, n_heads, seq_len_q, d_k)
        :param K: 键矩阵，形状：(batch_size, n_heads, seq_len_k, d_k)
        :param V: 值矩阵，形状：(batch_size, n_heads, seq_len_v, d_k)
        :param mask: 掩码矩阵，形状：(batch_size, 1, seq_len_q, seq_len_k)
        :param relative_positions: 相对位置索引，形状：(seq_len_q, seq_len_k)
        :param relative_embeddings: 相对位置嵌入，形状：(seq_len_q, seq_len_k, d_k)
        :return: 注意力输出和注意力权重
        """
        # 计算注意力分数：(batch_size, n_heads, seq_len_q, seq_len_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 如果使用相对位置嵌入，添加相对位置分数
        if self.use_relative_embedding and relative_positions is not None and relative_embeddings is not None:
            # 计算相对位置分数：(seq_len_q, seq_len_k, d_k) → (1, 1, seq_len_q, seq_len_k, d_k)
            relative_embeddings = relative_embeddings.unsqueeze(0).unsqueeze(0)
            # Q: (batch_size, n_heads, seq_len_q, d_k) → (batch_size, n_heads, seq_len_q, 1, d_k)
            Q_reshaped = Q.unsqueeze(-2)
            # 相对位置分数：(batch_size, n_heads, seq_len_q, seq_len_k)
            relative_attn_scores = torch.sum(Q_reshaped * relative_embeddings, dim=-1) / math.sqrt(self.d_k)
            attn_scores = attn_scores + relative_attn_scores
        
        # 应用掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重：(batch_size, n_heads, seq_len_q, seq_len_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算注意力输出：(batch_size, n_heads, seq_len_q, d_k)
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
    
    def forward(self, Q, K, V, mask=None, relative_positions=None, relative_embeddings=None):
        """
        多头注意力前向传播
        :param Q: 查询矩阵，形状：(seq_len_q, batch_size, d_model)
        :param K: 键矩阵，形状：(seq_len_k, batch_size, d_model)
        :param V: 值矩阵，形状：(seq_len_v, batch_size, d_model)
        :param mask: 掩码矩阵，形状：(batch_size, seq_len_q, seq_len_k)
        :param relative_positions: 相对位置索引，形状：(seq_len_q, seq_len_k)
        :param relative_embeddings: 相对位置嵌入，形状：(seq_len_q, seq_len_k, d_k)
        :return: 注意力输出和注意力权重
        """
        batch_size = Q.size(1)
        
        # 线性变换并分拆成多头
        # Q: (seq_len_q, batch_size, d_model) → (batch_size, seq_len_q, d_model) → (batch_size, seq_len_q, n_heads, d_k) → (batch_size, n_heads, seq_len_q, d_k)
        Q = self.W_q(Q.transpose(0, 1)).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K.transpose(0, 1)).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V.transpose(0, 1)).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 调整掩码形状：(batch_size, seq_len_q, seq_len_k) → (batch_size, 1, seq_len_q, seq_len_k)
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        # 计算缩放点积注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask, relative_positions, relative_embeddings)
        
        # 合并多头注意力输出：(batch_size, n_heads, seq_len_q, d_k) → (batch_size, seq_len_q, n_heads, d_k) → (batch_size, seq_len_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 线性变换：(batch_size, seq_len_q, d_model) → (seq_len_q, batch_size, d_model)
        output = self.W_o(attn_output).transpose(0, 1)
        
        return output, attn_weights
