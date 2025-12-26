import os
import numpy as np
import torch
from gensim.models import KeyedVectors

class PretrainedEmbedding:
    def __init__(self, embedding_path, embedding_type='glove'):
        """
        预训练词向量加载类
        :param embedding_path: 预训练词向量文件路径
        :param embedding_type: 词向量类型，支持 'glove' (英文) 或 'char300' 或 'tencent' (中文)
        """
        self.embedding_path = embedding_path
        self.embedding_type = embedding_type
        self.word2vec = None
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
                
        
        print(f"Character300词向量加载完成，词汇量: {len(self.vocab)}")

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
                
        
        print(f"GloVe词向量加载完成，词汇量: {len(self.vocab)}")



    def _load_tencent(self):
        """
        加载腾讯AILAB词向量 (.bin格式)
        """
        print(f"加载腾讯AILAB词向量: {self.embedding_path}")
        
        # 使用gensim加载word2vec模型
        self.word2vec = KeyedVectors.load_word2vec_format(self.embedding_path, binary=True)
        self.vocab = set(self.word2vec.index_to_key)
        
        print(f"腾讯AILAB词向量加载完成，词汇量: {len(self.vocab)}")
    
    def get_vector(self, word, default=None):
        """
        获取指定词的向量
        :param word: 要查询的词
        :param default: 如果词不存在，返回的默认值
        :return: 词向量 (numpy.ndarray) 或 default
        """
        if self.embedding_type == 'tencent':
            if word in self.vocab:
                return self.word2vec[word]
            return default
        else:  # tencent
            return self.word2vec.get(word, default)
    
    def get_embedding_matrix(self, word2idx, embedding_dim):
        """
        生成嵌入矩阵，用于初始化模型的嵌入层
        pad为零向量
        其他特殊标记词和预训练中没有的词将在模型训练时随机初始化
        :param word2idx: 词汇表的word到index映射
        :param embedding_dim: 期望的嵌入维度
        :return: 嵌入矩阵 (torch.Tensor)
        """
        # 初始化嵌入矩阵
        vocab_size = len(word2idx)
        embedding_matrix = torch.zeros((vocab_size, embedding_dim))
        

        matched_vectors=[]

        for word, idx in word2idx.items():
            vector = self.get_vector(word)
            if vector is not None:
                embedding_matrix[idx] = torch.from_numpy(np.array(vector, copy=True))
                matched_vectors.append(vector)
           
        mean = np.mean(matched_vectors)
        std = np.std(matched_vectors)

        for word, idx in word2idx.items():
            vector = self.get_vector(word)
            if word not in ['<pad>', '<unk>', '<sos>', '<eos>']:
                # 未登录词（但不在特殊符号中）
                embedding_matrix[idx] = torch.normal(mean=mean, std=std, size=(embedding_dim,))  # 随机初始化

        # <unk>, <sos>, <eos> 可用随机小值初始化（或用预训练词向量的均值+标准差）
        for idx in [1, 2, 3]:  # <unk>, <sos>, <eos>
            embedding_matrix[idx] = torch.normal(mean=mean, std=std, size=(embedding_dim,))
            
        # <pad> 通常初始化为 0（因为会被忽略）
        embedding_matrix[0] = torch.zeros(embedding_dim)  # <pad>

        print(f"嵌入矩阵生成完成，词汇表大小: {vocab_size}，匹配到词向量: {len(matched_vectors)}")
        return embedding_matrix
    
    def get_vocab(self):
        """
        获取词向量的词汇表
        :return: 词汇表集合
        """
        return self.vocab
    