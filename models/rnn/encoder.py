import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1, 
                 pretrained_embedding=None, freeze_embedding=False):
        """
        RNN编码器
        :param input_size: 输入词汇表大小
        :param hidden_size: 隐藏层大小
        :param num_layers: 循环层数量
        :param dropout: dropout概率
        :param pretrained_embedding: 预训练词向量矩阵 (可选)
        :param freeze_embedding: 是否冻结嵌入层参数，False表示可训练
        """
        super(Encoder, self).__init__()
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
            self.embedding = nn.Embedding(input_size, 300, padding_idx=0)
        
        # 检查嵌入层输出维度是否与隐藏层大小匹配
        embedding_dim = self.embedding.embedding_dim
        self.embedding_proj = nn.Linear(embedding_dim, hidden_size)

        
        # GRU层
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, 
                          dropout=dropout if num_layers > 1 else 0, 
                          batch_first=False)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_lengths):
        """
        编码器前向传播
        :param src: 源语言序列，形状：(seq_len, batch_size)
        :param src_lengths: 源语言序列长度，形状：(batch_size)
        :return: 编码器输出和最终隐藏状态
        """
        # 嵌入层：(seq_len, batch_size) → (seq_len, batch_size, embedding_dim)
        embedded = self.dropout(self.embedding(src))
        embedded = self.embedding_proj(embedded)
        
        # 使用pack_padded_sequence处理变长序列
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), enforce_sorted=True)
        
        # GRU前向传播
        # outputs: (seq_len, batch_size, hidden_size) - 每个时间步的输出
        # hidden: (num_layers, batch_size, hidden_size) - 最后一个时间步的隐藏状态
        packed_outputs, hidden = self.gru(packed)
        
        # 解包得到完整序列输出
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_outputs)
        
        return outputs, hidden
