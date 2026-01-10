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
    
    def predict(self, src,  max_length=200, strategy='greedy', beam_width=5): 
        """
        模型预测函数（用于推理）
        :param src: 源语言序列，形状：(1,seq_len)
        :param max_length: 最大输出长度
        :param strategy: 'greedy' 或 'beam'
        :param beam_width: beam search 宽度（仅在 strategy=='beam' 时生效）
        :param sos_idx: <sos> 的索引
        :param eos_idx: <eos> 的索引
        :return: 预测的目标语言序列索引（不包含<sos>，包含<eos>若预测到）
        """
        sos_idx=2
        eos_idx=3
        # 编码器需要 (seq_len,1)
        src = src.transpose(0,1).to(self.device)  # (seq_len, 1)
        # 编码器前向传播
        encoder_outputs, hidden = self.encoder(src, src_lengths=torch.tensor([src.size(0)]).to(self.device))
        
        if strategy == 'greedy':
            outputs = []
            input = torch.tensor([sos_idx], device=self.device)  # <sos>
            with torch.no_grad():
                for _ in range(max_length):
                    output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
                    top1 = output.argmax(1).item()
                    outputs.append(top1)
                    if top1 == eos_idx:
                        break
                    input = torch.tensor([top1], device=self.device)
            return outputs
        
        elif strategy == 'beam':
            with torch.no_grad():
                # 每个 beam: (token_list, hidden, score, finished)
                beams = [([sos_idx], hidden, 0.0, False)]
                for _ in range(max_length):
                    all_candidates = []
                    for seq, h, score, finished in beams:
                        if finished:
                            all_candidates.append((seq, h, score, finished))
                            continue
                        last_token = torch.tensor([seq[-1]], device=self.device)
                        output, h_new, _ = self.decoder(last_token, h, encoder_outputs)
                        log_probs = torch.log_softmax(output, dim=1).squeeze(0)  # vocab_size
                        topk_logp, topk_idx = torch.topk(log_probs, min(beam_width, log_probs.size(0)))
                        for k in range(topk_idx.size(0)):
                            idx = topk_idx[k].item()
                            logp = topk_logp[k].item()
                            new_seq = seq + [idx]
                            new_score = score + logp
                            new_finished = (idx == eos_idx)
                            # detach hidden to avoid unexpected graph growth (safety)
                            h_detached = h_new
                            all_candidates.append((new_seq, h_detached, new_score, new_finished))
                    # 选 top beam_width
                    all_candidates.sort(key=lambda x: x[2], reverse=True)
                    beams = all_candidates[:beam_width]
                    # 如果所有 beam 都已结束，停止扩展
                    if all(b[3] for b in beams):
                        break
                # 选择最优已结束 beam，否则选择得分最高的 beam
                finished_beams = [b for b in beams if b[3]]
                best = finished_beams[0] if finished_beams else beams[0]
                tokens = best[0][1:] if best[0] and best[0][0] == sos_idx else best[0]
                # 若末尾含多个 token 后续处理可由调用方完成
                return tokens
        
        else:
            raise ValueError("Unknown strategy: choose 'greedy' or 'beam'")
