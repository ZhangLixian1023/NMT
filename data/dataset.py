import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence

class TranslationDataset(Dataset):
    def __init__(self, data: List[Tuple[str, str]], src_vocab, tgt_vocab, max_length=20):
        """
        翻译数据集类
        :param data: 平行语料列表，每个元素是(src_text, tgt_text)元组
        :param src_vocab: 源语言词汇表
        :param tgt_vocab: 目标语言词汇表
        :param max_length: 最大序列长度
        """
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        
        # 源语言：只截断 + padding，不加 sos/eos
        src_tokens = src_text.split()[:self.max_length]
        src_indices = [self.src_vocab.word_to_idx(word) for word in src_tokens]

        
        # 目标语言序列处理：添加<sos>和<eos>标记，并转换为索引    
        # # 目标语言：原始 tokens
        tgt_tokens = tgt_text.split()[:self.max_length - 1] 
        tgt_indices = [self.tgt_vocab.word_to_idx(word) for word in tgt_tokens]

        # 构造 decoder input 和 target
        tgt_input = [2] + tgt_indices # 头加<sos>
        tgt_output = tgt_indices + [3] # 尾加<eos>
        
        return {
        'src': torch.tensor(src_indices, dtype=torch.long),
        'tgt_input': torch.tensor(tgt_input, dtype=torch.long),
        'tgt_output': torch.tensor(tgt_output, dtype=torch.long),
        'src_length': len(src_indices),
        'tgt_length': len(tgt_output) 
    }

def collate_fn(batch):
    """
    自定义的collate函数，用于处理变长序列
    :param batch: 批次数据
    :return: 整理后的批次数据
    """
    # 按序列长度排序（用于pack_padded_sequence）
    batch.sort(key=lambda x: x['src_length'], reverse=True)

    src_lengths = torch.tensor([item['src_length'] for item in batch])
    src_padded = pad_sequence([item['src'] for item in batch], batch_first=True, padding_value=0)
    tgt_in_padded = pad_sequence([item['tgt_input'] for item in batch], batch_first=True, padding_value=0)
    tgt_out_padded = pad_sequence([item['tgt_output'] for item in batch], batch_first=True, padding_value=0)
    return {
        'src': src_padded,
        'tgt_input': tgt_in_padded,
        'tgt_output': tgt_out_padded,
        'src_lengths': src_lengths
    }
