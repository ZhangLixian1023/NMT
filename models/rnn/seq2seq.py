import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        """
        序列到序列模型
        :param encoder: 编码器对象
        :param decoder: 解码器对象
        :param device: 运行设备
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        """
        模型前向传播
        :param src: 源语言序列，形状：(seq_len, batch_size)
        :param src_lengths: 源语言序列长度，形状：(batch_size)
        :param tgt: 目标语言序列，形状：(seq_len, batch_size)
        :param teacher_forcing_ratio: 教师强制比例
        :return: 解码器所有时间步的输出
        """
        batch_size = src.size(1)
        tgt_len = tgt.size(0)
        tgt_vocab_size = self.decoder.output_size
        
        # 存储解码器所有时间步的输出
        outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(self.device)
        
        # 编码器前向传播
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # 解码器初始输入为<sos>标记
        input = tgt[0, :]  # tgt的第一个时间步是<sos>标记
        
        for t in range(1, tgt_len):
            # 解码器前向传播
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            
            # 存储当前时间步的输出
            outputs[t] = output
            
            # 决定是否使用教师强制
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # 获取预测的词汇索引
            top1 = output.argmax(1)
            
            # 下一个时间步的输入：如果使用教师强制则使用真实目标词，否则使用预测词
            input = tgt[t] if teacher_force else top1
        
        return outputs
    
    def predict(self, src, src_lengths, max_length=50):
        """
        模型预测函数（用于推理）
        :param src: 源语言序列，形状：(seq_len, 1)
        :param src_lengths: 源语言序列长度，形状：(1)
        :param max_length: 最大输出长度
        :return: 预测的目标语言序列索引
        """
        batch_size = src.size(1)
        
        # 编码器前向传播
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # 存储预测结果
        outputs = []
        
        # 解码器初始输入为<sos>标记
        input = torch.tensor([2], device=self.device)  # <sos>的索引是2
        
        for t in range(max_length):
            # 解码器前向传播
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            
            # 获取预测的词汇索引
            top1 = output.argmax(1).item()
            
            # 存储预测结果
            outputs.append(top1)
            
            # 如果预测到<eos>标记，结束预测
            if top1 == 3:  # <eos>的索引是3
                break
            
            # 下一个时间步的输入为当前预测词
            input = torch.tensor([top1], device=self.device)
        
        return outputs
