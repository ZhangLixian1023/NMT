import argparse
import os
import pickle
import nltk
from data import Preprocessor, Vocabulary
from utils.embedding import PretrainedEmbedding

# 命令行参数配置
parser = argparse.ArgumentParser(description='数据准备：构建词表和加载预训练词向量')

# 通用参数
parser.add_argument('--dataset', type=str, default='train_10k.jsonl', help='数据集')
parser.add_argument('--save_dir', type=str, default='models/saved', help='数据保存目录')
parser.add_argument('--max_vocab_size', type=int, default=20000, help='词汇表最大大小')
parser.add_argument('--max_seq_len', type=int, default=50, help='最大序列长度')

# 预训练词向量参数
parser.add_argument('--src_embedding_path', type=str, default='wiki_giga_2024_200_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt', help='源语言预训练词向量路径')
parser.add_argument('--tgt_embedding_path', type=str, default='tencent.bin', help='目标语言预训练词向量路径')
parser.add_argument('--src_embedding_type', type=str, default='glove', choices=['glove', 'tencent'], help='源语言词向量类型')
parser.add_argument('--tgt_embedding_type', type=str, default='tencent', choices=['glove', 'tencent'], help='目标语言词向量类型')
parser.add_argument('--embedding_dim', type=int, default=200, help='嵌入层维度，不指定则使用词向量自身维度')

def main():
    """
    主函数：实现词表构建和预训练词向量加载
    """
    # 解析命令行参数
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    print('=== 开始数据准备 ===')
    
    # 1. 加载数据
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    print('正在加载和预处理数据...')
    preprocessor = Preprocessor()
    
    # 加载训练数据
    train_files = [os.path.join('dataset',args.dataset)]
    train_data = preprocessor.load_data(train_files)
    train_pairs = preprocessor.prepare_data(train_data)
    
    print(f'加载完成，训练数据对数量: {len(train_pairs)}')
    
    # 2. 构建词表
    print('正在构建词汇表...')
    
    # 加载预训练词向量（如果提供了路径）
    src_embedding_dim = args.embedding_dim
    tgt_embedding_dim = args.embedding_dim
    
    # 加载源语言预训练词向量
    src_pretrained_emb = PretrainedEmbedding(args.src_embedding_path, args.src_embedding_type)
    print(f'源语言词向量维度: {src_embedding_dim}')
    
    # 加载目标语言预训练词向量
    tgt_pretrained_emb = PretrainedEmbedding(args.tgt_embedding_path, args.tgt_embedding_type)
    print(f'目标语言词向量维度: {tgt_embedding_dim}')
    
    # 获取预训练词向量词汇表
    src_pretrained_vocab = src_pretrained_emb.get_vocab()
    tgt_pretrained_vocab = tgt_pretrained_emb.get_vocab()
    
    # 初始化词汇表
    src_vocab = Vocabulary('en', max_size=args.max_vocab_size, pretrained_vocab=src_pretrained_vocab)
    tgt_vocab = Vocabulary('zh', max_size=args.max_vocab_size, pretrained_vocab=tgt_pretrained_vocab)
    
    # 为训练数据构建词汇表
    for src, tgt in train_pairs:
        src_vocab.add_sentence(src)
        tgt_vocab.add_sentence(tgt)
    
    # 构建词汇表
    src_vocab.build_vocab()
    tgt_vocab.build_vocab()
    
    print(f'源语言词汇表大小: {src_vocab.n_words}')
    print(f'目标语言词汇表大小: {tgt_vocab.n_words}')
    
    # 3. 生成词向量矩阵
    print('正在生成词向量矩阵...')
    
    
    # 生成源语言词向量矩阵
    src_embedding_matrix = None
    src_embedding_matrix = src_pretrained_emb.get_embedding_matrix(
        src_vocab.word2idx, 
        src_embedding_dim
    )
    
    # 生成目标语言词向量矩阵
    tgt_embedding_matrix = None
    tgt_embedding_matrix = tgt_pretrained_emb.get_embedding_matrix(
        tgt_vocab.word2idx, 
        tgt_embedding_dim
    )
    
    # 4. 保存词表和词向量矩阵
    print(f'正在保存数据到目录: {args.save_dir}')
    
    # 保存词表为pkl文件
    src_vocab_path = os.path.join(args.save_dir, 'src_vocab.pkl')
    tgt_vocab_path = os.path.join(args.save_dir, 'tgt_vocab.pkl')
    
    with open(src_vocab_path, 'wb') as f:
        pickle.dump(src_vocab, f)
    
    with open(tgt_vocab_path, 'wb') as f:
        pickle.dump(tgt_vocab, f)
    
    print(f'源语言词表已保存到: {src_vocab_path}')
    print(f'目标语言词表已保存到: {tgt_vocab_path}')
    
    # 保存词向量矩阵为pkl文件
    src_embedding_path = os.path.join(args.save_dir, 'src_embedding.pkl')
    tgt_embedding_path = os.path.join(args.save_dir, 'tgt_embedding.pkl')
    
    with open(src_embedding_path, 'wb') as f:
        pickle.dump(src_embedding_matrix, f)
    
    with open(tgt_embedding_path, 'wb') as f:
        pickle.dump(tgt_embedding_matrix, f)
    
    print(f'源语言词向量矩阵已保存到: {src_embedding_path}')
    print(f'目标语言词向量矩阵已保存到: {tgt_embedding_path}')
    
    print('=== 数据准备完成 ===')

if __name__ == '__main__':
    main()
