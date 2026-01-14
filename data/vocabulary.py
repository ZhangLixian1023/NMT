import json
from collections import Counter

class Vocabulary:
    def __init__(self, word2idx_dict):
        """
        初始化词汇表
        :param max_size: 词汇表最大大小
        """        
        self.word2idx = {
            '<pad>': 0,
            '<unk>': 1,
            '<sos>': 2,
            '<eos>': 3
        }
        self.word2idx = word2idx_dict
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.n_words = len(self.idx2word)  # 初始词汇表大小
    
    def word_to_idx(self, word):
        """
        将词转换为索引
        :param word: 要转换的词
        :return: 对应的索引
        """
        return self.word2idx.get(word, self.word2idx['<unk>'])
    
    def idx_to_word(self, idx):
        return self.idx2word.get(idx, '<unk>')
