import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size, attention_type='dot'):
        """
        注意力机制基类
        :param hidden_size: 隐藏层大小
        :param attention_type: 注意力类型，可选：dot, multiplicative, additive
        """
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_type = attention_type
        
        if attention_type == 'multiplicative':
            # 乘法注意力：需要一个权重矩阵
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif attention_type == 'additive':
            # 加性注意力：需要一个前馈网络
            self.attn = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
    
    def forward(self, hidden, encoder_outputs):
        """
        注意力机制前向传播
        :param hidden: 解码器当前隐藏状态，形状：(batch_size, hidden_size)
        :param encoder_outputs: 编码器所有时间步的输出，形状：(seq_len, batch_size, hidden_size)
        :return: 注意力权重和上下文向量
        """
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        
        if self.attention_type == 'dot':
            # 点积注意力
            # 将hidden扩展为(seq_len, batch_size, hidden_size)以便与encoder_outputs进行点积
            hidden_expanded = hidden.unsqueeze(0).repeat(seq_len, 1, 1)
            # 计算点积：(seq_len, batch_size, hidden_size) * (seq_len, batch_size, hidden_size) → (seq_len, batch_size)
            attn_energies = torch.sum(hidden_expanded * encoder_outputs, dim=2)
        
        elif self.attention_type == 'multiplicative':
            # 乘法注意力：h^T * W * h'
            # 对encoder_outputs应用线性变换：(seq_len, batch_size, hidden_size) → (seq_len, batch_size, hidden_size)
            encoder_transformed = self.attn(encoder_outputs)
            # 将hidden扩展为(seq_len, batch_size, hidden_size)
            hidden_expanded = hidden.unsqueeze(0).repeat(seq_len, 1, 1)
            # 计算点积：(seq_len, batch_size, hidden_size) * (seq_len, batch_size, hidden_size) → (seq_len, batch_size)
            attn_energies = torch.sum(encoder_transformed * hidden_expanded, dim=2)
        
        elif self.attention_type == 'additive':
            # 加性注意力：v^T * tanh(W1*h + W2*h')
            # 将hidden扩展为(seq_len, batch_size, hidden_size)
            hidden_expanded = hidden.unsqueeze(0).repeat(seq_len, 1, 1)
            # 拼接hidden和encoder_outputs：(seq_len, batch_size, hidden_size*2)
            combined = torch.cat((hidden_expanded, encoder_outputs), dim=2)
            # 应用前馈网络：(seq_len, batch_size, hidden_size*2) → (seq_len, batch_size, 1)
            attn_energies = self.attn(combined).squeeze(2)
        
        # 将注意力能量转换为权重：(seq_len, batch_size) → (batch_size, seq_len)
        attn_energies = attn_energies.transpose(0, 1)  # (batch_size, seq_len)
        attn_weights = F.softmax(attn_energies, dim=1)  # (batch_size, seq_len)
        
        # 计算上下文向量：(batch_size, seq_len) * (batch_size, seq_len, hidden_size) → (batch_size, hidden_size)
        # 先将encoder_outputs转置为(batch_size, seq_len, hidden_size)
        encoder_outputs_transposed = encoder_outputs.transpose(0, 1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs_transposed).squeeze(1)
        
        return attn_weights, context
