import torch
import argparse
import os
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from data import Preprocessor, Vocabulary, TranslationDataset, collate_fn
from models.rnn.encoder import Encoder as RNNEncoder
from models.rnn.decoder import Decoder as RNNDecoder
from models.rnn.seq2seq import Seq2Seq as RNNSeq2Seq
from models.transformer.encoder import Encoder as TransformerEncoder
from models.transformer.decoder import Decoder as TransformerDecoder, Transformer

# 配置参数
parser = argparse.ArgumentParser(description='神经机器翻译模型评估')

# 通用参数
parser.add_argument('--model_dir', type=str, default='models/saved', help='模型文件目录')
parser.add_argument('--data_dir', type=str, default='dataset', help='数据集目录')
parser.add_argument('--batch_size', type=int, default=200, help='批次大小')
parser.add_argument('--max_seq_len', type=int, default=100, help='最大序列长度')

# 解析命令行参数
args = parser.parse_args()

# 从模型目录读取训练参数
import json
import os
import pickle
# 查找模型类型和参数文件
model_files = os.listdir(args.model_dir)
model_type = None
params_file = None
model_file = None

for file in model_files:
    if file.endswith('_params.json'):
        model_type = 'rnn'
        params_file = os.path.join(args.model_dir, file)
    elif file.endswith('_model.pt'):
        model_file = os.path.join(args.model_dir, file)

if not params_file or not model_file:
    raise ValueError(f'在模型目录 {args.model_dir} 中找不到模型参数文件或模型文件')

# 加载训练参数
with open(params_file, 'r', encoding='utf-8') as f:
    train_args_dict = json.load(f)

# 将训练参数转换为命名空间对象
class ArgsNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

train_args = ArgsNamespace(**train_args_dict)

# 设置模型类型
model_type = 'rnn'

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

# 加载词汇表
print('正在加载词汇表...')
with open('models/saved/src_vocab.pkl', 'rb') as f:
    src_vocab= pickle.load(f)
with open('models/saved/tgt_vocab.pkl', 'rb') as f:
    tgt_vocab= pickle.load(f)

# 加载预训练词向量嵌入矩阵
with open('models/saved/src_embedding.pkl', 'rb') as f:
    src_embedding_matrix = pickle.load(f)
with open('models/saved/tgt_embedding.pkl', 'rb') as f:
    tgt_embedding_matrix = pickle.load(f)

# 初始化模型
print('正在初始化模型...')
if model_type == 'rnn':
    # 初始化RNN模型
    encoder = RNNEncoder(
        input_size=src_vocab.n_words,
        hidden_size=train_args.rnn_hidden_size,
        num_layers=train_args.rnn_num_layers,
        dropout=train_args.rnn_dropout,
        pretrained_embedding=src_embedding_matrix,
        freeze_embedding=False
    ).to(device)
    
    decoder = RNNDecoder(
        output_size=tgt_vocab.n_words,
        hidden_size=train_args.rnn_hidden_size,
        num_layers=train_args.rnn_num_layers,
        dropout=train_args.rnn_dropout,
        attention_type=train_args.attention_type,
        pretrained_embedding=tgt_embedding_matrix,
        freeze_embedding=False
    ).to(device)
    
    model = RNNSeq2Seq(encoder, decoder, device).to(device)
    
    print(f'RNN模型初始化完成，注意力机制类型: {train_args.attention_type}')
    
elif model_type == 'transformer':
    # 初始化Transformer模型
    encoder = TransformerEncoder(
        input_size=src_vocab.n_words,
        d_model=train_args.transformer_d_model,
        n_layers=train_args.transformer_n_layers,
        n_heads=train_args.transformer_n_heads,
        d_ff=train_args.transformer_d_ff,
        dropout=train_args.transformer_dropout,
        norm_type=train_args.norm_type,
        embedding_type=train_args.positional_encoding
    ).to(device)
    
    decoder = TransformerDecoder(
        output_size=tgt_vocab.n_words,
        d_model=train_args.transformer_d_model,
        n_layers=train_args.transformer_n_layers,
        n_heads=train_args.transformer_n_heads,
        d_ff=train_args.transformer_d_ff,
        dropout=train_args.transformer_dropout,
        norm_type=train_args.norm_type,
        embedding_type=train_args.positional_encoding
    ).to(device)
    
    model = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=src_vocab.word_to_idx('<pad>'),
        tgt_pad_idx=tgt_vocab.word_to_idx('<pad>'),
        device=device
    ).to(device)
    
    print(f'Transformer模型初始化完成，位置嵌入: {train_args.positional_encoding}, 归一化: {train_args.norm_type}')

# 加载模型权重
print(f'正在加载模型: {model_file}')
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()

# 加载测试数据
print('正在加载测试数据...')
def load_pairs(file_path):
    pairs = []
    count=0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            count+=1
            if count>100:
                break
            if line:
                record = json.loads(line)
                pairs.append((record["src"], record["tgt"]))
    print("end loading pairs")
    return pairs

test_pairs = load_pairs('dataset/train_100k_pairs.jsonl')

# 创建测试数据集
test_dataset = TranslationDataset(test_pairs, src_vocab, tgt_vocab, max_length=args.max_seq_len)

# 翻译函数
def translate_sentence(model, sentence, src_vocab, tgt_vocab, device, max_length=50):
    """
    翻译单个句子
    :param model: 模型对象
    :param sentence: 源语言句子
    :param src_vocab: 源语言词汇表
    :param tgt_vocab: 目标语言词汇表
    :param device: 运行设备
    :param max_length: 最大输出长度
    :return: 翻译结果
    """
    model.eval()
    
    # 预处理源语言句子
    src_tokens = sentence.split()
    src_tokens = ['<sos>'] + src_tokens[:max_length - 2] + ['<eos>']
    src_indices = [src_vocab.word_to_idx(word) for word in src_tokens]
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(1).to(device)  # (seq_len, 1)
    src_lengths = torch.tensor([len(src_indices)], dtype=torch.long).to(device)
    
    if isinstance(model, RNNSeq2Seq):
        # RNN模型翻译
        output_indices = model.predict(src_tensor, src_lengths, max_length)
    elif isinstance(model, Transformer):
        # Transformer模型翻译
        output_tensor = model.predict(src_tensor, max_length)
        output_indices = output_tensor.squeeze(1).tolist()
    
    # 将索引转换为单词
    output_words = []
    for idx in output_indices:
        word = tgt_vocab.idx_to_word(idx)
        if word == '<eos>':
            break
        output_words.append(word)
    
    return output_words
# 在您提供的代码块之后，添加以下内容

from collections import Counter
import math

def calculate_bleu4(references, hypotheses):
    """
    根据给定的参考译文和假设译文列表，计算 BLEU-4 分数及其组成部分 p1, p2, p3, p4。
    不使用 brevity penalty。
    
    Args:
        references: List of lists. 每个元素是一个包含一个或多个参考译文（词列表）的列表。
        hypotheses: List of lists. 每个元素是模型生成的译文（词列表）。
    
    Returns:
        tuple: (bleu4_score, p1, p2, p3, p4)
    """
    # 初始化 n-gram 精确度的分子和分母
    numerator = [0, 0, 0, 0]  # p1, p2, p3, p4 的分子
    denominator = [0, 0, 0, 0]  # p1, p2, p3, p4 的分母

    # 遍历每一对参考译文和假设译文
    for i, (ref_list, hyp) in enumerate(zip(references, hypotheses)):
        # 通常我们只取第一个参考译文进行 n-gram 匹配（简化版）
        ref = ref_list[0]

        # 对于每个 n (1到4)
        for n in range(1, 5):
            # 计算假设译文的 n-gram
            hyp_ngrams = []
            for j in range(len(hyp) - n + 1):
                ngram = tuple(hyp[j:j+n])
                hyp_ngrams.append(ngram)
            
            # 计算参考译文的 n-gram 及其最大计数
            ref_ngrams_count = {}
            for j in range(len(ref) - n + 1):
                ngram = tuple(ref[j:j+n])
                ref_ngrams_count[ngram] = ref_ngrams_count.get(ngram, 0) + 1
            
            # 计算匹配的 n-gram 数量
            matched = 0
            hyp_ngrams_count = Counter(hyp_ngrams)
            for ngram, count in hyp_ngrams_count.items():
                if ngram in ref_ngrams_count:
                    matched += min(count, ref_ngrams_count[ngram])
            
            # 累加分子和分母
            numerator[n-1] += matched
            denominator[n-1] += len(hyp_ngrams)

    # 计算 p1, p2, p3, p4
    p1 = numerator[0] / denominator[0] if denominator[0] > 0 else 0
    p2 = numerator[1] / denominator[1] if denominator[1] > 0 else 0
    p3 = numerator[2] / denominator[2] if denominator[2] > 0 else 0
    p4 = numerator[3] / denominator[3] if denominator[3] > 0 else 0

    bleu4 = p1 * p2 * p3 * p4

    return bleu4, p1, p2, p3, p4
# 评估模型
def evaluate_model(model, test_pairs, src_vocab, tgt_vocab, device, max_length=50):
    """
    评估模型性能
    :param model: 模型对象
    :param test_pairs: 测试数据对
    :param src_vocab: 源语言词汇表
    :param tgt_vocab: 目标语言词汇表
    :param device: 运行设备
    :param max_length: 最大输出长度
    :return: BLEU分数
    """
    references = []
    hypotheses = []
    
    for src, tgt in tqdm(test_pairs, desc='评估'):
        # 翻译源语言句子
        output_words = translate_sentence(model, src, src_vocab, tgt_vocab, device, max_length)

        # 准备参考译文
        tgt_words = tgt.split()  
        
        # 添加到参考和假设列表
        references.append([tgt_words])
        hypotheses.append(output_words)
    


    # 计算BLEU分数
    smooth = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(references, hypotheses, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    # 调用函数计算 BLEU-4 分数
    bleu4_score, p1, p2, p3, p4 = calculate_bleu4(references, hypotheses)
    return {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'BLEU-3': bleu3,
        'BLEU-4': bleu4
    }




# 运行评估
print('开始评估模型...')
bleu_scores = evaluate_model(model, test_pairs, src_vocab, tgt_vocab, device, args.max_seq_len)

# 打印评估结果
print('\n评估结果:')
for metric, score in bleu_scores.items():
    print(f'{metric}: {score * 100:.2f}')

# 生成示例翻译
print('\n生成示例翻译:')
for i in range(5):
    src, tgt = test_pairs[i]
    print(f'\n源语言: {src}')
    print(f'真实译文: {tgt}')
    output_words = translate_sentence(model, src, src_vocab, tgt_vocab, device, args.max_seq_len)
    print(f'模型译文: {" ".join(output_words)}')
