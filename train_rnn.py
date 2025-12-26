import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
import pickle
from tqdm import tqdm
from data import TranslationDataset, collate_fn
from data import Preprocessor
from models.rnn.encoder import Encoder as RNNEncoder
from models.rnn.decoder import Decoder as RNNDecoder
from models.rnn.seq2seq import Seq2Seq as RNNSeq2Seq

# 配置参数
parser = argparse.ArgumentParser(description='神经机器翻译模型训练')

# 通用参数
parser.add_argument('--dataset', type=str, default='train_10k.jsonl', help='数据集')
parser.add_argument('--save_dir', type=str, default='models/saved', help='模型保存目录')
parser.add_argument('--batch_size', type=int, default=512, help='批次大小')
parser.add_argument('--num_epochs', type=int, default=8 ,help='训练轮数')
parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
parser.add_argument('--max_seq_len', type=int, default=50, help='最大序列长度')
parser.add_argument('--early_stopping_patience', type=int, default=2, help='早停耐心值')

# 预训练词向量参数
parser.add_argument('--freeze_embedding', action='store_true', help='是否冻结嵌入层参数')

# RNN模型参数
parser.add_argument('--rnn_hidden_size', type=int, default=200, help='RNN隐藏层大小')
parser.add_argument('--rnn_num_layers', type=int, default=2, help='RNN层数')
parser.add_argument('--rnn_dropout', type=float, default=0.3, help='RNN dropout概率')
parser.add_argument('--attention_type', type=str, default='dot', choices=['dot', 'multiplicative', 'additive'], help='注意力机制类型')


args = parser.parse_args()

# 创建保存目录
os.makedirs(args.save_dir, exist_ok=True)

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')
# 初始化预处理器
preprocessor = Preprocessor()
# 加载训练数据
train_files = [os.path.join('dataset',args.dataset)]
train_data = preprocessor.load_data(train_files)
train_pairs = preprocessor.prepare_data(train_data)

# 加载验证数据
valid_data = preprocessor.load_data([os.path.join('dataset', 'valid.jsonl')])
valid_pairs = preprocessor.prepare_data(valid_data)

src_vocab_path = os.path.join(args.save_dir, 'src_vocab.pkl')
tgt_vocab_path = os.path.join(args.save_dir, 'tgt_vocab.pkl')
with open(src_vocab_path, 'rb') as f:
    src_vocab= pickle.load(f)
with open(tgt_vocab_path, 'rb') as f:
    tgt_vocab= pickle.load(f)

# 创建数据集和数据加载器
train_dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab, max_length=args.max_seq_len)
valid_dataset = TranslationDataset(valid_pairs, src_vocab, tgt_vocab, max_length=args.max_seq_len)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

src_embedding_path = os.path.join(args.save_dir, 'src_embedding.pkl')
tgt_embedding_path = os.path.join(args.save_dir, 'tgt_embedding.pkl')
with open(tgt_embedding_path, 'rb') as f:
    tgt_embedding_matrix = pickle.load(f)
with open(src_embedding_path, 'rb') as f:
    src_embedding_matrix = pickle.load(f)
# 初始化RNN模型
encoder = RNNEncoder(
    input_size=src_vocab.n_words,
    hidden_size=args.rnn_hidden_size,
    num_layers=args.rnn_num_layers,
    dropout=args.rnn_dropout,
    pretrained_embedding=src_embedding_matrix,
    freeze_embedding=args.freeze_embedding
).to(device)

decoder = RNNDecoder(
    output_size=tgt_vocab.n_words,
    hidden_size=args.rnn_hidden_size,
    num_layers=args.rnn_num_layers,
    dropout=args.rnn_dropout,
    attention_type=args.attention_type,
    pretrained_embedding=tgt_embedding_matrix,
    freeze_embedding=args.freeze_embedding
).to(device)

model = RNNSeq2Seq(encoder, decoder, device).to(device)

# 损失函数和优化器
criterion = nn.NLLLoss(ignore_index=0)  # 忽略填充标记
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

print(f'RNN模型初始化完成，注意力机制类型: {args.attention_type}')
print(f'嵌入层是否可训练: {not args.freeze_embedding}')

# 加载模型权重
# print(f'正在加载模型: {"rnn_model6.108.pt"}')
# model.load_state_dict(torch.load(os.path.join(args.save_dir, "rnn_model6.108.pt"), map_location=device))
# model.eval()

# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc='训练'):
        # 准备数据
        src = batch['src'].transpose(0, 1).to(device)  # (seq_len, batch_size)
        tgt = batch['tgt'].transpose(0, 1).to(device)  # (seq_len, batch_size)
        src_lengths = batch['src_lengths'].to(device)
        
        optimizer.zero_grad()
        
        # RNN模型前向传播
        output = model(src, src_lengths, tgt)
        output_dim = output.shape[-1]
        
        # 重塑输出和目标序列以便计算损失
        output = output[1:].reshape(-1, output_dim)  # 跳过<sos>标记
        tgt = tgt[1:].reshape(-1)
        
        loss = criterion(output, tgt)
        
        # 反向传播和优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 验证函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='验证'):
            # 准备数据
            src = batch['src'].transpose(0, 1).to(device)  # (seq_len, batch_size)
            tgt = batch['tgt'].transpose(0, 1).to(device)  # (seq_len, batch_size)
            src_lengths = batch['src_lengths'].to(device)
            
            # RNN模型前向传播
            output = model(src, src_lengths, tgt, teacher_forcing_ratio=0.0)
            output_dim = output.shape[-1]
            
            # 重塑输出和目标序列以便计算损失
            output = output[1:].reshape(-1, output_dim)  # 跳过<sos>标记
            tgt = tgt[1:].reshape(-1)
            
            loss = criterion(output, tgt)
                
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 翻译函数
def translate_sentence(model, sentence, src_vocab, tgt_vocab, device, max_length):
    """
    翻译单个句子
    :param model: 模型对象
    :param sentence: 源语言句子
    :param src_vocab: 源语言词汇表
    :param tgt_vocab: 目标语言词汇表
    :param device: 运行设备
    :param max_length: 最大输出长度
    :return: 翻译结果句子
    """
    model.eval()
    
    # 预处理源语言句子
    src_tokens = sentence.split()
    src_tokens = ['<sos>'] + src_tokens[:max_length - 2] + ['<eos>']
    src_indices = [src_vocab.word_to_idx(word) for word in src_tokens]
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(1).to(device)  # (seq_len, 1)
    src_lengths = torch.tensor([len(src_indices)], dtype=torch.long).to(device)
    
    # RNN模型翻译
    output_indices = model.predict(src_tensor, src_lengths, max_length)
    
    # 将索引转换为单词
    output_words = []
    for idx in output_indices:
        word = tgt_vocab.idx_to_word(idx)
        if word == '<eos>':
            break
        output_words.append(word)
    
    # 移除<sos>标记
    if output_words and output_words[0] == '<sos>':
        output_words = output_words[1:]
    
    return ' '.join(output_words)

# 生成翻译示例
def generate_translation_examples(model, train_pairs, src_vocab, tgt_vocab, device, epoch,  num_examples, max_length):
    """
    生成翻译示例
    :param model: 模型对象
    :param train_pairs: 训练数据对
    :param src_vocab: 源语言词汇表
    :param tgt_vocab: 目标语言词汇表
    :param device: 运行设备
    :param epoch: 当前epoch
    :param num_examples: 生成的示例数量
    :param max_length: 最大输出长度
    """
    print(f'\n=== Epoch {epoch+1} 翻译示例 ===')
    
    # 选取前num_examples个句子对
    examples = train_pairs[:num_examples]
    
    for i, (src, tgt) in enumerate(examples, 1):
        # 翻译源语言句子
        translated = translate_sentence(model, src, src_vocab, tgt_vocab, device, max_length)
        
        # 输出结果
        print(f'\n示例 {i}:')
        print(f'源语言: {src}')
        print(f'目标语言: {tgt}')
        print(f'模型翻译: {translated}')
    
    print('====================')


# 训练循环
print('开始训练...')
best_valid_loss = float('inf')
early_stopping_counter = 0

for epoch in range(args.num_epochs):
    start_time = time.time()
    
    # 训练一个轮次
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # 验证模型
    valid_loss = evaluate(model, valid_loader, criterion, device,)
    
    # 生成翻译示例
    generate_translation_examples(
        model=model,
        train_pairs=train_pairs,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        epoch=epoch,
        num_examples=15,
        max_length=args.max_seq_len
    )
    
    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
    
    print(f'轮次: {epoch+1:02} | 时间: {epoch_mins}m {epoch_secs:.0f}s')
    print(f'训练损失: {train_loss:.3f} | 验证损失: {valid_loss:.3f}')
    
    # 保存最佳模型
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        early_stopping_counter = 0
        
        # 保存模型参数和状态
        model_save_path = os.path.join(args.save_dir, 'rnn_model.pt')
        torch.save(model.state_dict(), model_save_path)
        
        # 保存训练参数
        import json
        params_save_path = os.path.join(args.save_dir, 'rnn_params.json')
        with open(params_save_path, 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
        
        print(f'模型已保存到 {model_save_path}')
        print(f'训练参数已保存到 {params_save_path}')
    else:
        early_stopping_counter += 1
        print(f'早停计数: {early_stopping_counter}/{args.early_stopping_patience}')
    
    # 早停检查
    if early_stopping_counter >= args.early_stopping_patience:
        print('早停机制触发，训练结束')
        break

print('训练完成!')
