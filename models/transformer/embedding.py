import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, embedding_type='absolute'):
        """
        位置嵌入模块
        :param d_model: 模型维度
        :param max_len: 最大序列长度
        :param embedding_type: 位置嵌入类型，可选：absolute, relative
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.embedding_type = embedding_type
        
        if embedding_type == 'absolute':
            # 绝对位置嵌入：使用正弦和余弦函数
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)
        elif embedding_type == 'relative':
            # 相对位置嵌入：使用可学习的位置嵌入
            self.relative_embeddings = nn.Embedding(max_len * 2, d_model)
            self.max_len = max_len
    
    def forward(self, x):
        """
        位置嵌入前向传播
        :param x: 输入序列，形状：(batch_size,seq_len,  d_model)
        :return: 带有位置嵌入的序列
        """
        if self.embedding_type == 'absolute':
            # 绝对位置嵌入：直接相加
            x = x + self.pe[:x.size(1), :] # pe 自动广播为 (batch_size , max_len,  d_model)
            return x
        else:  # relative
            # 相对位置嵌入：需要在注意力机制中使用，这里暂时返回原输入
            return x
    
    def get_relative_positions(self, seq_len):
        """
        获取相对位置索引
        :param seq_len: 序列长度
        :return: 相对位置索引矩阵
        """
        positions = torch.arange(seq_len, dtype=torch.long)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        # 将相对位置映射到[0, 2*max_len-1]范围内
        relative_positions = relative_positions + self.max_len - 1
        return relative_positions

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        """
        层归一化
        :param d_model: 模型维度
        :param eps: 防止除零的小值
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        """
        RMS归一化
        :param d_model: 模型维度
        :param eps: 防止除零的小值
        """
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        # 计算RMS：sqrt(mean(x^2) + eps)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 归一化并应用权重
        return self.weight * (x / rms)
