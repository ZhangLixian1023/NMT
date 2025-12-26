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
src_vocab = Vocabulary.load(os.path.join(args.model_dir, 'src_vocab.json'))
tgt_vocab = Vocabulary.load(os.path.join(args.model_dir, 'tgt_vocab.json'))

print(f'源语言词汇表大小: {src_vocab.n_words}')
print(f'目标语言词汇表大小: {tgt_vocab.n_words}')

# 初始化模型
print('正在初始化模型...')
if model_type == 'rnn':
    # 初始化RNN模型
    encoder = RNNEncoder(
        input_size=src_vocab.n_words,
        hidden_size=train_args.rnn_hidden_size,
        num_layers=train_args.rnn_num_layers,
        dropout=train_args.rnn_dropout,
        pretrained_embedding=None,
        freeze_embedding=False
    ).to(device)
    
    decoder = RNNDecoder(
        output_size=tgt_vocab.n_words,
        hidden_size=train_args.rnn_hidden_size,
        num_layers=train_args.rnn_num_layers,
        dropout=train_args.rnn_dropout,
        attention_type=train_args.attention_type,
        pretrained_embedding=None,
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
preprocessor = Preprocessor()
test_data = preprocessor.load_data([os.path.join(args.data_dir, 'test.jsonl')])
test_pairs = preprocessor.prepare_data(test_data)

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
        
        # 准备参考译文（去掉<sos>和<eos>标记）
        tgt_words = tgt.split()[1:-1]  # 去掉<sos>和<eos>标记
        
        # 添加到参考和假设列表
        references.append([tgt_words])
        hypotheses.append(output_words)
    
    # 计算BLEU分数
    smooth = SmoothingFunction().method1
    bleu1 = corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu3 = corpus_bleu(references, hypotheses, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
    
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
