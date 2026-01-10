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
    
    def predict(self, src, src_lengths=None, max_length=99, strategy='greedy', beam_width=5):
        """
        模型预测函数（用于推理）
        :param src: 源语言序列，形状：(batch_size = 1,seq_len)
        :param max_length: 最大输出长度
        :param strategy: 'greedy' 或 'beam'
        :param beam_width: beam search 宽度（仅在 strategy=='beam' 时生效）
        :param sos_idx: <sos> 索引
        :param eos_idx: <eos> 索引
        :return: 预测的目标语言序列索引（不包含<sos>，包含<eos>若预测到）
        """
        sos_idx=2
        eos_idx=3
        src = src.to(self.device)
        src_mask = self.make_src_mask(src)
        enc_output = self.encoder(src, src_mask)  # (1, seq_len, d_model)

        if strategy == 'greedy':
            tgt = torch.tensor([[sos_idx]], device=self.device)  # (1,1)
            outputs = []
            with torch.no_grad():
                for _ in range(max_length):
                    tgt_mask = self.make_tgt_mask(tgt)
                    out = self.decoder(tgt, enc_output, src_mask, tgt_mask)  # (1, seq_len, vocab)
                    logits = out[:, -1, :]  # (1, vocab)
                    top1 = logits.argmax(-1).item()
                    outputs.append(top1)
                    tgt = torch.cat((tgt, torch.tensor([[top1]], device=self.device)), dim=1)
                    if top1 == eos_idx:
                        break
            return outputs

        elif strategy == 'beam':
            with torch.no_grad():
                # beams: list of (tokens_tensor, score, finished)
                beams = [(torch.tensor([[sos_idx]], device=self.device), 0.0, False)]
                for _ in range(max_length):
                    all_candidates = []
                    for tokens, score, finished in beams:
                        if finished:
                            all_candidates.append((tokens, score, True))
                            continue
                        tgt_mask = self.make_tgt_mask(tokens)
                        out = self.decoder(tokens, enc_output, src_mask, tgt_mask)  # (1, seq_len, vocab)
                        log_probs = torch.log_softmax(out[:, -1, :], dim=-1).squeeze(0)  # (vocab,)
                        topk_logp, topk_idx = torch.topk(log_probs, min(beam_width, log_probs.size(0)))
                        for k in range(topk_idx.size(0)):
                            idx = topk_idx[k].item()
                            lp = topk_logp[k].item()
                            new_tokens = torch.cat((tokens, torch.tensor([[idx]], device=self.device)), dim=1)
                            new_score = score + lp
                            new_finished = (idx == eos_idx)
                            all_candidates.append((new_tokens, new_score, new_finished))
                    # keep top beams
                    all_candidates.sort(key=lambda x: x[1], reverse=True)
                    beams = all_candidates[:beam_width]
                    if all(b[2] for b in beams):
                        break
                finished_beams = [b for b in beams if b[2]]
                best = finished_beams[0] if finished_beams else beams[0]
                token_list = best[0].squeeze(0).tolist()
                if token_list and token_list[0] == sos_idx:
                    token_list = token_list[1:]
                return token_list

        else:
            raise ValueError("Unknown strategy: choose 'greedy' or 'beam'")
