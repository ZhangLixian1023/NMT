import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size, attention_type='dot'):
        """
        优化后的注意力机制模块
        :param hidden_size: 隐藏层大小
        :param attention_type: 注意力类型，可选：'dot', 'multiplicative', 'additive'
        """
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_type = attention_type
        
        if attention_type == 'multiplicative':
            # Luong General Attention: score = q^T W k
            self.attn = nn.Linear(hidden_size, hidden_size, bias=False)
        elif attention_type == 'additive':
            # Bahdanau Attention: score = v^T tanh(Wq + Wk)
            self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
            self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)
        elif attention_type not in ['dot', 'multiplicative', 'additive']:
            raise ValueError("Invalid attention_type. Choose from 'dot', 'multiplicative', 'additive'.")

    def forward(self, hidden, encoder_outputs):
        """
        :param hidden: 解码器当前隐藏状态 (batch_size, hidden_size)
        :param encoder_outputs: 编码器输出 (seq_len, batch_size, hidden_size)
        :return: attn_weights (batch, seq_len), context (batch, hidden_size)
        """
        # 统一维度为 (batch, seq, feature)
        # encoder_outputs: (B, S, H)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        batch_size, seq_len, _ = encoder_outputs.size()
        
        # 将 hidden 增加序列维度: (B, 1, H)
        query = hidden.unsqueeze(1) 

        if self.attention_type == 'dot':
            # 点积：(B, 1, H) * (B, H, S) -> (B, 1, S)
            attn_energies = torch.bmm(query, encoder_outputs.transpose(1, 2))
            attn_energies = attn_energies.squeeze(1) # (B, S)

        elif self.attention_type == 'multiplicative':
            # 乘法：query * (W * encoder_outputs)
            # (B, S, H) -> (B, S, H)
            key = self.attn(encoder_outputs)
            # (B, 1, H) * (B, H, S) -> (B, 1, S)
            attn_energies = torch.bmm(query, key.transpose(1, 2)).squeeze(1)

        elif self.attention_type == 'additive':
            # 加性：v * tanh(Wq * q + Wk * k)
            # query: (B, 1, H), key: (B, S, H)
            # 利用广播机制相加: (B, 1, H) + (B, S, H) -> (B, S, H)
            q_part = self.w_q(query)      # (B, 1, H)
            k_part = self.w_k(encoder_outputs) # (B, S, H)
            
            features = torch.tanh(q_part + k_part) # 广播相加
            attn_energies = self.v(features).squeeze(2) # (B, S)

        # 归一化注意力权重
        attn_weights = F.softmax(attn_energies, dim=1) # (B, S)

        # 计算上下文向量: (B, 1, S) * (B, S, H) -> (B, 1, H)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return attn_weights, context