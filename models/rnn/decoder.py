import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import Attention

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=2, dropout=0.1, attention_type='dot',
                 pretrained_embedding=None, freeze_embedding=False):
        """
        RNN解码器
        :param output_size: 输出词汇表大小
        :param hidden_size: 隐藏层大小
        :param num_layers: 循环层数量
        :param dropout: dropout概率
        :param attention_type: 注意力类型
        :param pretrained_embedding: 预训练词向量矩阵 (可选)
        :param freeze_embedding: 是否冻结嵌入层参数，False表示可训练
        """
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 嵌入层
        if pretrained_embedding is not None:
            # 使用预训练词向量初始化嵌入层
            self.embedding = nn.Embedding.from_pretrained(pretrained_embedding, 
                                                         freeze=freeze_embedding, 
                                                         padding_idx=0)
        else:
            # 随机初始化嵌入层
            self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=0)
        
        # 检查嵌入层输出维度是否与隐藏层大小匹配
        embedding_dim = self.embedding.embedding_dim
        if embedding_dim != hidden_size:
            # 如果不匹配，添加线性映射层
            self.embedding_proj = nn.Linear(embedding_dim, hidden_size)
            self.need_proj = True
        else:
            self.need_proj = False
        
        # 注意力机制
        self.attention = Attention(hidden_size, attention_type)
        
        # GRU层：输入是嵌入向量和上下文向量的拼接
        self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers=num_layers, 
                          dropout=dropout if num_layers > 1 else 0, 
                          batch_first=False)
        
        # 输出层：将隐藏状态转换为词汇表概率分布
        self.out = nn.Linear(hidden_size * 2, output_size)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_step, last_hidden, encoder_outputs):
        """
        解码器前向传播（单步）
        :param input_step: 当前时间步的输入，形状：(batch_size)
        :param last_hidden: 上一个时间步的隐藏状态，形状：(num_layers, batch_size, hidden_size)
        :param encoder_outputs: 编码器所有时间步的输出，形状：(seq_len, batch_size, hidden_size)
        :return: 当前时间步的输出、隐藏状态和注意力权重
        """
        # 嵌入层：(batch_size) → (1, batch_size, embedding_dim)（添加时间步维度）
        embedded = self.dropout(self.embedding(input_step).unsqueeze(0))
        
        # 如果嵌入维度与隐藏层大小不匹配，进行线性映射
        if self.need_proj:
            embedded = self.embedding_proj(embedded)
        
        # 获取顶层隐藏状态用于注意力计算：(num_layers, batch_size, hidden_size) → (batch_size, hidden_size)
        top_hidden = last_hidden[-1, :, :]
        
        # 计算注意力权重和上下文向量
        attn_weights, context = self.attention(top_hidden, encoder_outputs)
        
        # 将上下文向量扩展为(1, batch_size, hidden_size)以便与嵌入向量拼接
        context = context.unsqueeze(0)
        
        # 拼接嵌入向量和上下文向量：(1, batch_size, hidden_size * 2)
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # GRU前向传播
        # output: (1, batch_size, hidden_size) - 当前时间步的输出
        # hidden: (num_layers, batch_size, hidden_size) - 当前时间步的隐藏状态
        output, hidden = self.gru(rnn_input, last_hidden)
        
        # 拼接GRU输出和上下文向量：(1, batch_size, hidden_size * 2)
        output = torch.cat((output.squeeze(0), context.squeeze(0)), dim=1)
        
        # 输出层：(batch_size, hidden_size * 2) → (batch_size, output_size)
        output = self.out(output)
        
        # 应用softmax得到概率分布
        output = F.log_softmax(output, dim=1)
        
        return output, hidden, attn_weights
