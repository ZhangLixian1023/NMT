import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .embedding import LayerNorm, RMSNorm, PositionalEncoding
from .encoder import Encoder
from .decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, encoder, decoder,  device):
        """
        Transformer模型
        :param encoder: 编码器对象
        :param decoder: 解码器对象
        :param src_pad_idx: 源语言填充索引
        :param tgt_pad_idx: 目标语言填充索引
        :param device: 运行设备
        """
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = 0
        self.tgt_pad_idx = 0
        self.device = device
    
    def make_src_mask(self, src):
        """
        src: (batch_size, seq_len)
        return: (batch_size, 1, 1, seq_len)
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def make_tgt_mask(self, tgt):
        """
        tgt: (batch_size, seq_len)
        return: (batch_size, 1, seq_len, seq_len)
        """
        batch_size, seq_len = tgt.shape

        # 填充掩码: (batch_size, 1, 1, seq_len)
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)

        # 未来掩码: (1, 1, seq_len, seq_len)
        tgt_sub_mask = torch.tril(torch.ones((seq_len, seq_len), device=self.device)).bool()
        tgt_sub_mask = tgt_sub_mask.unsqueeze(0).unsqueeze(0)

        # 组合: (batch_size, 1, seq_len, seq_len)
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask.to(self.device)
    
    def forward(self, src, tgt):
        """
        模型前向传播
        :param src: 源语言序列，形状：(batch_size,seq_len)
        :param tgt: 目标语言序列，形状：(batch_size, tgt_seq_len)
        :return: 模型输出 (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # 创建掩码
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        # 编码器前向传播
        # 维度检查
        # print("\nTransformer forward:")
        # print(f"src shape = {src.shape}")
        # print(f"tgt shape = {tgt.shape}")
        enc_output = self.encoder(src, src_mask)
        
        # 解码器前向传播
        output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        
        return output
    
    def predict(self, src, max_length=50):
        """
        模型预测函数（用于推理）
        :param src: 源语言序列，形状：(batch_size = 1,seq_len, 1) 
        :param max_length: 最大输出长度
        :return: 预测的目标语言序列索引
        """
        
        # 创建源语言掩码
        src_mask = self.make_src_mask(src)
        
        # print("\n\nTransformer Predict 维度检查:")
        # print(f"src_mask: {src_mask.shape}")
        # print(f"src: {src.shape}")
        # 编码器前向传播
        enc_output = self.encoder(src, src_mask) # ( 1, seq_len, d_model )
        
        # 初始化目标序列：只包含<sos>标记
        tgt = torch.tensor([2], device=self.device).unsqueeze(0)  # <sos>的索引是2，形状：(1,1)
        
        # # 维度检查
        # print(f"tgt: {tgt.shape}")
        
        # print(f"enc_out: {enc_output.shape}")

        for _ in range(max_length):
            # tgt : (1, seq_len)
            # 创建目标语言掩码
            tgt_mask = self.make_tgt_mask(tgt) 
            
            # 解码器前向传播
            output = self.decoder(tgt, enc_output, src_mask, tgt_mask) # (1, seq_len, vocab_size )
            
            # 获取最后一个时间步的输出
            output = output[:, -1, :] # (1, seq_len, vocab_size ) --> (1, vocab_size )
            
            # 获取预测的词汇索引
            top1 = output.argmax(-1).unsqueeze(0)  # (1 token)--> (1 batch, 1 token) 最大概率的词
            
            # 将预测词添加到目标序列
            tgt = torch.cat((tgt, top1), dim=1) # (1, seq_len + 1 )
            
            # 如果预测到<eos>标记，结束预测
            if top1.item() == 3:  # <eos>的索引是3
                break
        
        return tgt