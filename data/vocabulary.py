import json
from collections import Counter

class Vocabulary:
    def __init__(self, language, max_size=None):
        """
        初始化词汇表
        :param language: 语言类型
        :param max_size: 词汇表最大大小
        """
        self.language = language
        self.max_size = max_size
        self.word2idx = {
            '<pad>': 0,
            '<unk>': 1,
            '<sos>': 2,
            '<eos>': 3
        }
        self.idx2word = {
            0: '<pad>',
            1: '<unk>',
            2: '<sos>',
            3: '<eos>'
        }
        self.word_count = Counter()
        self.n_words = 4  # 初始词汇表大小
    
    def add_word(self, word):
        self.word_count[word] += 1
    
    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)
    
    def build_vocab(self):
        """
        构建词汇表
        如果提供了预训练词向量词汇表，会优先保留与预训练词向量匹配的词
        """
        # 获取所有候选词
        all_words = list(self.word_count.items())
        
        sorted_words = sorted(all_words, key=lambda x: x[1], reverse=True)
        
        # 限制词汇表大小
        if self.max_size:
            sorted_words = sorted_words[:self.max_size - 4]  # 减去特殊标记词
        
        # 构建词汇表映射
        for word, _ in sorted_words:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1
    
    def word_to_idx(self, word):
        """
        将词转换为索引
        :param word: 要转换的词
        :return: 对应的索引
        """
        return self.word2idx.get(word, self.word2idx['<unk>'])
    
    def idx_to_word(self, idx):
        return self.idx2word.get(idx, '<unk>')
    
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'language': self.language,
                'max_size': self.max_size,
                'word2idx': self.word2idx,
                'idx2word': {int(k): v for k, v in self.idx2word.items()},
                'n_words': self.n_words
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls(
            data['language'], 
            data['max_size'],
            oov_policy=data.get('oov_policy', 'unk')
        )
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        vocab.n_words = data['n_words']
        return vocab
