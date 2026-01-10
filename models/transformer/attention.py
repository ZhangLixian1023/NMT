import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, relative_position=False):
        """
        多头注意力机制
        :param d_model: 模型维度
        :param n_heads: 头数
        :param dropout: dropout概率
        :param relative_position: 是否使用相对位置
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
        
        # 是否使用相对位置
        self.relative_position = relative_position
        if relative_position:
        # RoPE requires even d_k (pairing dims)
            if self.d_k % 2 != 0:
                raise ValueError("d_k (d_model // n_heads) must be even for RoPE")
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        缩放点积注意力
        :param Q: 查询矩阵，形状：(batch_size, n_heads, seq_len_q, d_k)
        :param K: 键矩阵，形状：(batch_size, n_heads, seq_len_k, d_k)
        :param V: 值矩阵，形状：(batch_size, n_heads, seq_len_v, d_k)
        :param mask: 掩码矩阵，形状：(batch_size, 1, seq_len_q, seq_len_k)
        :return: 注意力输出和注意力权重
        """
        # 计算注意力分数：(batch_size, n_heads, seq_len_q, seq_len_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # print("点积注意力:")
        # print("维度:")
        # print(f"mask:{mask.shape}")
        # print(f"Q:{Q.shape}")
        # print(f"K:{K.shape}")
        # print(f"atten_scores:{attn_scores.shape}")
        # 应用掩码
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # 计算注意力权重：(batch_size, n_heads, seq_len_q, seq_len_k)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        assert K.size(2) == V.size(2) # seq_len_k == seq_len_v
        # 计算注意力输出：(batch_size, n_heads, seq_len_q, d_k)
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
    
    def _rotate_half(self, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack((-x2, x1), dim=-1)
        return x_rot.flatten(-2)

    def _apply_rope(self, x, cos, sin):
        # x: (batch, n_heads, seq_len, d_k)
        # cos, sin: (seq_len, d_k) -> expand to (1,1,seq_len,d_k)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin

    def _build_rope(self, seq_len, device, dtype=torch.float32):
        # build rotary cos/sin for dimension d_k
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_k, 2, device=device, dtype=dtype) / self.d_k))
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        angles = torch.einsum('i,j->ij', positions, inv_freq)  # (seq_len, d_k/2)
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        # interleave to (seq_len, d_k)
        sin = torch.repeat_interleave(sin, repeats=2, dim=-1)
        cos = torch.repeat_interleave(cos, repeats=2, dim=-1)
        return cos, sin

    def forward(self, Q, K, V, mask=None):
        """
        多头注意力前向传播
        :param Q: 查询矩阵，形状：(batch_size, seq_len_q,  d_model)
        :param K: 键矩阵，形状：(batch_size, seq_len_k,  d_model)
        :param V: 值矩阵，形状：(batch_size, seq_len_v,  d_model)
        :param mask: 掩码矩阵，形状：(batch_size, 1, seq_len_q, seq_len_k)
        :return: 注意力输出和注意力权重
        """
        batch_size = Q.size(0)


        # 线性变换并分拆成多头
        # Q: (batch_size, seq_len_q, d_model) → (batch_size, seq_len_q, n_heads, d_k) → (batch_size, n_heads, seq_len_q, d_k)
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
                
        # print("维度检查:")
        # print(f"Q:{Q.shape}")
        # print(f"K:{K.shape}")
        # input()
        # mask: (batch_size, 1, seq_len_q, seq_len_k)
        
        # 如果启用相对位置，使用RoPE对Q和K进行旋转位置编码
        if self.relative_position:
            seq_len_q = Q.size(2)
            seq_len_k = K.size(2)
            max_len = max(seq_len_q, seq_len_k)
            # build rope with same dtype as Q to avoid unwanted casts (supports fp16)
            cos, sin = self._build_rope(max_len, Q.device, Q.dtype)
            # trim to actual lengths
            cos_q, sin_q = cos[:seq_len_q], sin[:seq_len_q]
            cos_k, sin_k = cos[:seq_len_k], sin[:seq_len_k]
            Q = self._apply_rope(Q, cos_q, sin_q)
            K = self._apply_rope(K, cos_k, sin_k)

        # 计算缩放点积注意力
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 合并多头注意力输出：(batch_size, n_heads, seq_len_q, d_k) → (batch_size, seq_len_q, n_heads, d_k) → (batch_size, seq_len_q, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 线性变换：(batch_size, seq_len_q, d_model) → (batch_size,seq_len_q,  d_model)
        output = self.W_o(attn_output)
        
        return output, attn_weights
