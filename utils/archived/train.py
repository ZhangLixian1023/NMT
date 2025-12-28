import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import time
from tqdm import tqdm
import nltk
from data import Preprocessor, Vocabulary, TranslationDataset, collate_fn
from models.rnn.encoder import Encoder as RNNEncoder
from models.rnn.decoder import Decoder as RNNDecoder
from models.rnn.seq2seq import Seq2Seq as RNNSeq2Seq
from models.transformer.encoder import Encoder as TransformerEncoder
from models.transformer.decoder import Decoder as TransformerDecoder, Transformer
from utils.embedding import PretrainedEmbedding

# 配置参数
parser = argparse.ArgumentParser(description='神经机器翻译模型训练')

# 通用参数
parser.add_argument('--model_type', type=str, default='rnn', choices=['rnn', 'transformer'], help='模型类型')
parser.add_argument('--data_dir', type=str, default='dataset', help='数据集目录')
parser.add_argument('--save_dir', type=str, default='models/saved', help='模型保存目录')
parser.add_argument('--batch_size', type=int, default=1024, help='批次大小')
parser.add_argument('--num_epochs', type=int, default=15 ,help='训练轮数')
parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
parser.add_argument('--max_vocab_size', type=int, default=15000, help='词汇表最大大小')
parser.add_argument('--max_seq_len', type=int, default=50, help='最大序列长度')
parser.add_argument('--early_stopping_patience', type=int, default=5, help='早停耐心值')

# 预训练词向量参数
parser.add_argument('--src_embedding_path', type=str, default='wiki_giga_2024_200_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt', help='源语言预训练词向量路径')
parser.add_argument('--tgt_embedding_path', type=str, default='tencent.bin', help='目标语言预训练词向量路径')
parser.add_argument('--src_embedding_type', type=str, default='glove', choices=['glove', 'tencent'], help='源语言词向量类型')
parser.add_argument('--tgt_embedding_type', type=str, default='tencent', choices=['glove', 'tencent'], help='目标语言词向量类型')
parser.add_argument('--freeze_embedding', action='store_true', help='是否冻结嵌入层参数')
parser.add_argument('--embedding_dim', type=int, default=200, help='嵌入层维度，不指定则使用词向量自身维度')

# RNN模型参数
parser.add_argument('--rnn_hidden_size', type=int, default=200, help='RNN隐藏层大小')
parser.add_argument('--rnn_num_layers', type=int, default=2, help='RNN层数')
parser.add_argument('--rnn_dropout', type=float, default=0.3, help='RNN dropout概率')
parser.add_argument('--attention_type', type=str, default='dot', choices=['dot', 'multiplicative', 'additive'], help='注意力机制类型')

# Transformer模型参数
parser.add_argument('--transformer_d_model', type=int, default=512, help='Transformer模型维度')
parser.add_argument('--transformer_n_layers', type=int, default=6, help='Transformer层数')
parser.add_argument('--transformer_n_heads', type=int, default=8, help='Transformer头数')
parser.add_argument('--transformer_d_ff', type=int, default=2048, help='Transformer前馈网络隐藏层维度')
parser.add_argument('--transformer_dropout', type=float, default=0.1, help='Transformer dropout概率')
parser.add_argument('--positional_encoding', type=str, default='absolute', choices=['absolute', 'relative'], help='位置嵌入类型')
parser.add_argument('--norm_type', type=str, default='layernorm', choices=['layernorm', 'rmsnorm'], help='归一化类型')

args = parser.parse_args()

# 创建保存目录
os.makedirs(args.save_dir, exist_ok=True)

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')
# 自动下载 nltk 依赖（首次运行时）
nltk.data.find('tokenizers/punkt')
nltk.data.find('tokenizers/punkt_tab')
# 数据预处理
print('正在加载和预处理数据...')
preprocessor = Preprocessor()

# 加载训练数据
train_files = [os.path.join(args.data_dir, 'train_100k.jsonl')]
train_data = preprocessor.load_data(train_files)
train_pairs = preprocessor.prepare_data(train_data)

# 加载验证数据
valid_data = preprocessor.load_data([os.path.join(args.data_dir, 'valid.jsonl')])
valid_pairs = preprocessor.prepare_data(valid_data)

# 加载预训练词向量
src_pretrained_emb = 'wiki_giga_2024_200_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt'
tgt_pretrained_emb = 'tencent.bin'
src_embedding_dim = args.embedding_dim
tgt_embedding_dim = args.embedding_dim

# 加载源语言预训练词向量
if args.src_embedding_path:
    src_pretrained_emb = PretrainedEmbedding(args.src_embedding_path, args.src_embedding_type)
    print(f'源语言词向量维度: {src_embedding_dim}')

# 加载目标语言预训练词向量
if args.tgt_embedding_path:
    tgt_pretrained_emb = PretrainedEmbedding(args.tgt_embedding_path, args.tgt_embedding_type)
    print(f'目标语言词向量维度: {tgt_embedding_dim}')

# 构建词汇表
print('正在构建词汇表...')
# 获取预训练词向量词汇表
src_pretrained_vocab = src_pretrained_emb.get_vocab() if src_pretrained_emb else None
tgt_pretrained_vocab = tgt_pretrained_emb.get_vocab() if tgt_pretrained_emb else None

src_vocab = Vocabulary('en', max_size=args.max_vocab_size, pretrained_vocab=src_pretrained_vocab)
tgt_vocab = Vocabulary('zh', max_size=args.max_vocab_size, pretrained_vocab=tgt_pretrained_vocab)

# 为训练数据构建词汇表
for src, tgt in train_pairs:
    src_vocab.add_sentence(src)
    tgt_vocab.add_sentence(tgt)

# 构建词汇表
src_vocab.build_vocab()
tgt_vocab.build_vocab()

# 保存词汇表
src_vocab.save(os.path.join(args.save_dir, 'src_vocab.json'))
tgt_vocab.save(os.path.join(args.save_dir, 'tgt_vocab.json'))

print(f'源语言词汇表大小: {src_vocab.n_words}')
print(f'目标语言词汇表大小: {tgt_vocab.n_words}')

# 创建数据集和数据加载器
train_dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab, max_length=args.max_seq_len)
valid_dataset = TranslationDataset(valid_pairs, src_vocab, tgt_vocab, max_length=args.max_seq_len)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

# 初始化模型
print('正在初始化模型...')
if args.model_type == 'rnn':
    # 生成预训练词向量矩阵
    src_embedding_matrix = None
    tgt_embedding_matrix = None
    
    # 生成源语言嵌入矩阵
    if src_pretrained_emb:
        src_embedding_matrix = src_pretrained_emb.get_embedding_matrix(
            src_vocab.word2idx, 
            src_embedding_dim if args.embedding_dim else args.rnn_hidden_size
        )
    
    # 生成目标语言嵌入矩阵
    if tgt_pretrained_emb:
        tgt_embedding_matrix = tgt_pretrained_emb.get_embedding_matrix(
            tgt_vocab.word2idx, 
            tgt_embedding_dim if args.embedding_dim else args.rnn_hidden_size
        )
    
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
    if src_pretrained_emb:
        print(f'源语言使用预训练词向量: {args.src_embedding_path}')
    if tgt_pretrained_emb:
        print(f'目标语言使用预训练词向量: {args.tgt_embedding_path}')
    print(f'嵌入层是否可训练: {not args.freeze_embedding}')
    
elif args.model_type == 'transformer':
    # 生成预训练词向量矩阵
    src_embedding_matrix = None
    tgt_embedding_matrix = None
    
    # 生成源语言嵌入矩阵
    if src_pretrained_emb:
        src_embedding_matrix = src_pretrained_emb.get_embedding_matrix(
            src_vocab.word2idx, 
            src_embedding_dim if args.embedding_dim else args.transformer_d_model
        )
    
    # 生成目标语言嵌入矩阵
    if tgt_pretrained_emb:
        tgt_embedding_matrix = tgt_pretrained_emb.get_embedding_matrix(
            tgt_vocab.word2idx, 
            tgt_embedding_dim if args.embedding_dim else args.transformer_d_model
        )
    
    # 初始化Transformer模型
    # 注意：Transformer模型的嵌入层初始化需要在模型内部处理，这里先简单处理
    encoder = TransformerEncoder(
        input_size=src_vocab.n_words,
        d_model=args.transformer_d_model,
        n_layers=args.transformer_n_layers,
        n_heads=args.transformer_n_heads,
        d_ff=args.transformer_d_ff,
        dropout=args.transformer_dropout,
        norm_type=args.norm_type,
        embedding_type=args.positional_encoding
    ).to(device)
    
    decoder = TransformerDecoder(
        output_size=tgt_vocab.n_words,
        d_model=args.transformer_d_model,
        n_layers=args.transformer_n_layers,
        n_heads=args.transformer_n_heads,
        d_ff=args.transformer_d_ff,
        dropout=args.transformer_dropout,
        norm_type=args.norm_type,
        embedding_type=args.positional_encoding
    ).to(device)
    
    model = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=src_vocab.word_to_idx('<pad>'),
        tgt_pad_idx=tgt_vocab.word_to_idx('<pad>'),
        device=device
    ).to(device)
    
    # 初始化Transformer参数
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充标记
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    print(f'Transformer模型初始化完成，位置嵌入: {args.positional_encoding}, 归一化: {args.norm_type}')
    if src_pretrained_emb or tgt_pretrained_emb:
        print('警告：Transformer模型暂不支持预训练词向量')

# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device, model_type):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc='训练'):
        # 准备数据
        src = batch['src'].transpose(0, 1).to(device)  # (seq_len, batch_size)
        tgt = batch['tgt'].transpose(0, 1).to(device)  # (seq_len, batch_size)
        src_lengths = batch['src_lengths'].to(device)
        
        optimizer.zero_grad()
        
        if model_type == 'rnn':
            # RNN模型前向传播
            output = model(src, src_lengths, tgt)
            output_dim = output.shape[-1]
            
            # 重塑输出和目标序列以便计算损失
            output = output[1:].reshape(-1, output_dim)  # 跳过<sos>标记
            tgt = tgt[1:].reshape(-1)
            
            loss = criterion(output, tgt)
            
        elif model_type == 'transformer':
            # Transformer模型前向传播
            output = model(src, tgt[:-1])
            output_dim = output.shape[-1]
            
            # 重塑输出和目标序列以便计算损失
            output = output.reshape(-1, output_dim)
            tgt = tgt[1:].reshape(-1)
            
            loss = criterion(output, tgt)
        
        # 反向传播和优化
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 验证函数
def evaluate(model, dataloader, criterion, device, model_type):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='验证'):
            # 准备数据
            src = batch['src'].transpose(0, 1).to(device)  # (seq_len, batch_size)
            tgt = batch['tgt'].transpose(0, 1).to(device)  # (seq_len, batch_size)
            src_lengths = batch['src_lengths'].to(device)
            
            if model_type == 'rnn':
                # RNN模型前向传播
                output = model(src, src_lengths, tgt, teacher_forcing_ratio=0.0)
                output_dim = output.shape[-1]
                
                # 重塑输出和目标序列以便计算损失
                output = output[1:].reshape(-1, output_dim)  # 跳过<sos>标记
                tgt = tgt[1:].reshape(-1)
                
                loss = criterion(output, tgt)
                
            elif model_type == 'transformer':
                # Transformer模型前向传播
                output = model(src, tgt[:-1])
                output_dim = output.shape[-1]
                
                # 重塑输出和目标序列以便计算损失
                output = output.reshape(-1, output_dim)
                tgt = tgt[1:].reshape(-1)
                
                loss = criterion(output, tgt)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# 翻译函数
def translate_sentence(model, sentence, src_vocab, tgt_vocab, device, max_length=50, model_type='rnn'):
    """
    翻译单个句子
    :param model: 模型对象
    :param sentence: 源语言句子
    :param src_vocab: 源语言词汇表
    :param tgt_vocab: 目标语言词汇表
    :param device: 运行设备
    :param max_length: 最大输出长度
    :param model_type: 模型类型，rnn或transformer
    :return: 翻译结果句子
    """
    model.eval()
    
    # 预处理源语言句子
    src_tokens = sentence.split()
    src_tokens = ['<sos>'] + src_tokens[:max_length - 2] + ['<eos>']
    src_indices = [src_vocab.word_to_idx(word) for word in src_tokens]
    src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(1).to(device)  # (seq_len, 1)
    src_lengths = torch.tensor([len(src_indices)], dtype=torch.long).to(device)
    
    if model_type == 'rnn':
        # RNN模型翻译
        output_indices = model.predict(src_tensor, src_lengths, max_length)
    elif model_type == 'transformer':
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
    
    # 移除<sos>标记
    if output_words and output_words[0] == '<sos>':
        output_words = output_words[1:]
    
    return ' '.join(output_words)

# 生成翻译示例
def generate_translation_examples(model, train_pairs, src_vocab, tgt_vocab, device, epoch, model_type='rnn', num_examples=5, max_length=50):
    """
    生成翻译示例
    :param model: 模型对象
    :param train_pairs: 训练数据对
    :param src_vocab: 源语言词汇表
    :param tgt_vocab: 目标语言词汇表
    :param device: 运行设备
    :param epoch: 当前epoch
    :param model_type: 模型类型
    :param num_examples: 生成的示例数量
    :param max_length: 最大输出长度
    """
    print(f'\n=== Epoch {epoch+1} 翻译示例 ===')
    
    # 选取前num_examples个句子对
    examples = train_pairs[:num_examples]
    
    for i, (src, tgt) in enumerate(examples, 1):
        # 翻译源语言句子
        translated = translate_sentence(model, src, src_vocab, tgt_vocab, device, max_length, model_type)
        
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
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.model_type)
    
    # 验证模型
    valid_loss = evaluate(model, valid_loader, criterion, device, args.model_type)
    
    # 生成翻译示例
    generate_translation_examples(
        model=model,
        train_pairs=train_pairs,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        epoch=epoch,
        model_type=args.model_type,
        num_examples=5,
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
        model_save_path = os.path.join(args.save_dir, f'{args.model_type}_model.pt')
        torch.save(model.state_dict(), model_save_path)
        
        # 保存训练参数
        import json
        params_save_path = os.path.join(args.save_dir, f'{args.model_type}_params.json')
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
