import os
import numpy as np
import torch
from gensim.models import KeyedVectors

class PretrainedEmbedding:
    def __init__(self, embedding_path, embedding_type='glove'):
        """
        预训练词向量加载类
        :param embedding_path: 预训练词向量文件路径
        :param embedding_type: 词向量类型，支持 'glove' (英文) 或 'tencent' (中文)
        """
        self.embedding_path = embedding_path
        self.embedding_type = embedding_type
        self.word2vec = None
        self.vector_dim = 200
        self.vocab = set()
        
        # 加载词向量
        self.load_embedding()
    
    def load_embedding(self):
        """
        加载预训练词向量
        """
        if not os.path.exists(self.embedding_path):
            raise FileNotFoundError(f"预训练词向量文件不存在: {self.embedding_path}")
        
        if self.embedding_type == 'glove':
            self._load_glove()
        elif self.embedding_type == 'tencent':
            self._load_tencent()
        elif self.embedding_type == 'char300':
            self._load_char300()
        else:
            raise ValueError(f"不支持的词向量类型: {self.embedding_type}")
    def _load_char300(self):
        """
        加载Character300词向量 (.txt格式)
        """
        print(f"加载Character300词向量: {self.embedding_path}")
        
        # 初始化词向量字典
        self.word2vec = {}
        
        with open(self.embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                if len(values) != 301:
                    continue
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)
                
                self.word2vec[word] = vector
                self.vocab.add(word)
                
        
        print(f"Character300词向量加载完成，词汇量: {len(self.vocab)}，向量维度: {self.vector_dim}")

    def _load_glove(self):
        """
        加载GloVe词向量 (.txt格式)
        """
        print(f"加载GloVe词向量: {self.embedding_path}")
        
        # 初始化词向量字典
        self.word2vec = {}
        
        with open(self.embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                if len(values) > 201:
                    continue
                word = values[0]
                vector = np.array(values[1:], dtype=np.float32)
                
                self.word2vec[word] = vector
                self.vocab.add(word)
                
        
        print(f"GloVe词向量加载完成，词汇量: {len(self.vocab)}，向量维度: {self.vector_dim}")



    def _load_tencent(self):
        """
        加载腾讯AILAB词向量 (.bin格式)
        """
        print(f"加载腾讯AILAB词向量: {self.embedding_path}")
        
        # 使用gensim加载word2vec模型
        self.word2vec = KeyedVectors.load_word2vec_format(self.embedding_path, binary=True)
        self.vocab = set(self.word2vec.index_to_key)
        self.vector_dim = self.word2vec.vector_size
        
        print(f"腾讯AILAB词向量加载完成，词汇量: {len(self.vocab)}，向量维度: {self.vector_dim}")
    
    def get_vector(self, word, default=None):
        """
        获取指定词的向量
        :param word: 要查询的词
        :param default: 如果词不存在，返回的默认值
        :return: 词向量 (numpy.ndarray) 或 default
        """
        if self.embedding_type == 'glove':
            return self.word2vec.get(word, default)
        else:  # tencent
            if word in self.vocab:
                return self.word2vec[word]
            return default
    
    def get_embedding_matrix(self, word2idx, embedding_dim):
        """
        生成嵌入矩阵，用于初始化模型的嵌入层
        目前的实现中 unkown pad eos sos 均为零向量
        由于词向量文件中通常不包含这些特殊标记词
        这些词的向量将在模型训练时随机初始化吗？
        另外，未匹配到的词也保持为零向量
        :param word2idx: 词汇表的word到index映射
        :param embedding_dim: 期望的嵌入维度
        :return: 嵌入矩阵 (torch.Tensor)
        """
        # 初始化嵌入矩阵
        vocab_size = len(word2idx)
        embedding_matrix = torch.zeros((vocab_size, embedding_dim))
        
        # 统计匹配到的词向量数量
        matched_count = 0
        
        for word, idx in word2idx.items():
            vector = self.get_vector(word)
            if vector is not None:
                embedding_matrix[idx] = torch.from_numpy(np.array(vector, copy=True))
                matched_count += 1
            # 否则保持零向量（随机初始化会在模型中完成）
        
        print(f"嵌入矩阵生成完成，词汇表大小: {vocab_size}，匹配到词向量: {matched_count}")
        return embedding_matrix
    
    def get_vocab(self):
        """
        获取词向量的词汇表
        :return: 词汇表集合
        """
        return self.vocab
    
    def get_dim(self):
        """
        获取词向量维度
        :return: 词向量维度
        """
        return self.vector_dim
