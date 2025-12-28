import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from trainer import RNN_Trainer
from tqdm import tqdm
from data import TranslationDataset, collate_fn
from models.rnn.encoder import Encoder as RNNEncoder
from models.rnn.decoder import Decoder as RNNDecoder
from models.rnn.seq2seq import Seq2Seq as RNNSeq2Seq
from utils import Demo
params = {
    "train_dataset": "train_100k_pairs.jsonl", # 训练数据集文件名
    "max_seq_len": 50,
    "hidden_size": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "batch_size": 64,
    "n_epochs": 20,
    "learning_rate": 1e-4,
    "attention_type": "dot",  # 注意力机制类型：'bahdanau' 或 'luong'
    "freeze_embedding": False,     # 是否冻结预训练词向量
    "patience": 5,
    "src_vocab_size":None,
    "tgt_vocab_size":None
}
# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

def load_pairs(file_path):
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                pairs.append((record["src"], record["tgt"]))
    print("Done: loading pairs: " + file_path)
    return pairs


# 加载训练数据和验证数据
print('正在加载训练数据和验证数据...')
train_pairs = load_pairs(os.path.join('./dataset/', params['train_dataset']))
valid_pairs = load_pairs('./dataset/valid_pairs.jsonl')

# 加载词库 (词-->index)
with open('./saved_vocab_embedding/src_vocab_small.pkl', 'rb') as f:
    src_vocab= pickle.load(f)
with open('./saved_vocab_embedding/tgt_vocab_small.pkl', 'rb') as f:
    tgt_vocab= pickle.load(f)
params['src_vocab_size']=src_vocab.n_words
params['tgt_vocab_size']=tgt_vocab.n_words

# 创建数据集和数据加载器
train_dataset = TranslationDataset(train_pairs, src_vocab, tgt_vocab, max_length=params['max_seq_len'])
valid_dataset = TranslationDataset(valid_pairs, src_vocab, tgt_vocab, max_length=params['max_seq_len'])

train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], collate_fn=collate_fn, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], collate_fn=collate_fn)
# 加载预训练词向量嵌入矩阵
with open('./saved_vocab_embedding/src_embedding.pkl', 'rb') as f:
    src_embedding_matrix = pickle.load(f)
with open('./saved_vocab_embedding/tgt_embedding.pkl', 'rb') as f:
    tgt_embedding_matrix = pickle.load(f)

# 初始化RNN模型
encoder = RNNEncoder(
    input_size=src_vocab.n_words,
    hidden_size=params['hidden_size'],
    num_layers=params['num_layers'],
    dropout=params['dropout'],
    pretrained_embedding=src_embedding_matrix,
    freeze_embedding=params['freeze_embedding']
).to(device)

decoder = RNNDecoder(
    output_size=tgt_vocab.n_words,
    hidden_size=params['hidden_size'],
    num_layers=params['num_layers'],
    dropout=params['dropout'],
    attention_type=params['attention_type'],
    pretrained_embedding=tgt_embedding_matrix,
    freeze_embedding=params['freeze_embedding']
).to(device)

model = RNNSeq2Seq(encoder, decoder, device).to(device)
# 损失函数和优化器
criterion = nn.NLLLoss(ignore_index=0)  # 忽略填充标记
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

demo = Demo(
    model,
    src_vocab,
    tgt_vocab,
    examples=train_pairs[100:120],
    device=device
)
trainer= RNN_Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=valid_loader,
    optimizer = optimizer,
    criterion=criterion,
    device=device,
    demo=demo
)

print(f'RNN模型初始化完成，注意力机制类型: {params["attention_type"]}')
print(f'嵌入层是否可训练: {not params["freeze_embedding"]}')

trainer.train(
    n_epochs=params['n_epochs'],
    settings=params,
    patience=params['patience']
    )