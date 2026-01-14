import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from data import TranslationDataset, collate_fn
from models.rnn.encoder import Encoder as RNNEncoder
from models.rnn.decoder import Decoder as RNNDecoder
from models.rnn.seq2seq import Seq2Seq as RNNSeq2Seq
from utils import Demo, calculate_bleu4
from exp_frame import Exp_frame
import numpy as np
class RNN_frame(Exp_frame):
    """
    交互训练、评估、测试框架
    通过 python -i 交互式启动
    流程：
    一、加载词库
    词库 与 模型 密切相关  词库一旦加载就不要再重新加载
    二、设定模型参数
    三、从文件加载模型 load_model 或 随机初始化模型 init_model （默认已经随机初始化）
    四、实验设定 通过self.exp_setting 修改 数据集 batch_size lr max_seq_len 等等 
    五、评估 evaluate 输出 验证集上的 loss  不会输出样例
    六、训练 train 可以训了一段时间后中途改设定 每epoch的训练设定和结果会自动保存下来 有最丰富的输出
    七、测试 test 直接输入pairs 或者 单独的已经分好词的 ['欧洲 国家 普遍 相信'] 会输出样例
    """
    def __init__(self):
        super().__init__()
        self.model_params = {
            "architecture":"rnn",
            "hidden_size": 256,
            "num_layers": 2,
            "dropout": 0.25,
            "attention_type": 'multiplicative',  # 注意力机制类型：'dot','multiplicative','additive'
            "src_vocab_size":self.src_vocab.n_words,
            "tgt_vocab_size":self.tgt_vocab.n_words,
            "freeze_embedding": False     # 是否冻结预训练词向量
            }
        self.exp_setting={
            "max_seq_len": 90,
            "batch_size": 128,
            "learning_rate": 1e-4,
            "patience": 2,
            "teacher_forcing_ratio":1.0,
            "start_from": "scratch",
            "others": ""
            }

    def init_model(self,
    src_embedding_file='./processed_data/zh_matrix.npy' ,
    tgt_embedding_file='./processed_data/en_matrix.npy',save=True
    ):
        # 加载预训练词向量嵌入矩阵
        src_embedding_matrix = torch.from_numpy(np.load(src_embedding_file))
        tgt_embedding_matrix = torch.from_numpy(np.load(tgt_embedding_file))

        # 初始化RNN模型
        encoder = RNNEncoder(
            input_size=self.src_vocab.n_words,
            hidden_size=self.model_params['hidden_size'],
            num_layers=self.model_params['num_layers'],
            dropout=self.model_params['dropout'],
            pretrained_embedding=src_embedding_matrix.float(),
            freeze_embedding=self.model_params['freeze_embedding']
        ).to(self.device)

        decoder = RNNDecoder(
            output_size=self.tgt_vocab.n_words,
            hidden_size=self.model_params['hidden_size'],
            num_layers=self.model_params['num_layers'],
            dropout=self.model_params['dropout'],
            attention_type=self.model_params['attention_type'],
            pretrained_embedding=tgt_embedding_matrix.float(),
            freeze_embedding=self.model_params['freeze_embedding']
        ).to(self.device)

        self.model = RNNSeq2Seq(encoder, decoder, self.device).to(self.device)
        self.exp_setting['start_from'] = 'scratch'
        self.demo = Demo(
            self.model,
            self.src_vocab,
            self.tgt_vocab,
            examples=self.test_pairs,
            device=self.device
        )
        print(f"Done: Init model with {src_embedding_file} and {tgt_embedding_file}.")
        if save==True:
            self._init_saver()
        self.save=save

    def load_model(self,model_file="saved_models/rnn0109181343_from_scratch/rnn0109181343_epoch_2.pt",save=True):
        self.exp_setting['from_model']=model_file
        # 初始化RNN模型
        encoder = RNNEncoder(
            input_size=self.src_vocab.n_words,
            hidden_size=self.model_params['hidden_size'],
            num_layers=self.model_params['num_layers'],
            dropout=self.model_params['dropout']
        ).to(self.device)

        decoder = RNNDecoder(
            output_size=self.tgt_vocab.n_words,
            hidden_size=self.model_params['hidden_size'],
            num_layers=self.model_params['num_layers'],
            dropout=self.model_params['dropout'],
            attention_type=self.model_params['attention_type']
        ).to(self.device)

        self.model = RNNSeq2Seq(encoder, decoder, self.device).to(self.device)
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.exp_setting['start_from'] = 'other'
        print(f"Done: Load model {model_file}.")
        if save==True:
            self._init_saver()
        self.save=save

    def train_epoch(self,train_loader,criterion,optimizer) -> float:
        """Train one epoch."""
        self.model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc='训练'):
            # 从 batch 中取出数据（ (batch_size, seq_len) ）
            src = batch['src'].to(self.device)                    # (batch_size, src_len)
            tgt_input = batch['tgt_input'].to(self.device)        # (batch_size, tgt_len)
            tgt_output = batch['tgt_output'].to(self.device)      # (batch_size, tgt_len)
            src_lengths = batch['src_lengths'].to(self.device)

            # GRU 模型要求输入是 (seq_len, batch_size)，则转置
            src = src.transpose(0, 1)          # (src_len, batch_size)
            tgt_input = tgt_input.transpose(0, 1)  # (tgt_len, batch_size)
            tgt_output = tgt_output.transpose(0, 1)  # (tgt_len, batch_size)
            
            optimizer.zero_grad()

            # 前向传播：用 tgt_input 作为 decoder 的输入（teacher forcing）
            output = self.model(
                src, 
                src_lengths, 
                tgt_input,
                self.exp_setting['teacher_forcing_ratio']
                )   # output shape: (tgt_len, batch_size, vocab_size)

            # 计算损失：用 output 和 tgt_output
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)       # (tgt_len * batch_size, vocab_size)
            tgt_output = tgt_output.reshape(-1)           # (tgt_len * batch_size,)

            loss = criterion(output, tgt_output)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        self.exp_setting['start_from']='continue'
        return epoch_loss / len(train_loader)
  
    def evaluate(self,valid_loader=None,criterion=None) -> float:
        """Evaluate the model."""
        if valid_loader == None:
            # 创建数据集和数据加载器
            valid_dataset = TranslationDataset(self.test_pairs, self.src_vocab, self.tgt_vocab, max_length=self.exp_setting['max_seq_len'])
            valid_loader = DataLoader(valid_dataset, batch_size=self.exp_setting['batch_size'], collate_fn=collate_fn)
            # 损失函数和优化器
            criterion = nn.NLLLoss(ignore_index=0)  # 忽略填充标记
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc='验证'):
                # 从 batch 中取出数据（ (batch_size, seq_len) ）
                src = batch['src'].to(self.device)                    # (batch_size, src_len)
                tgt_input = batch['tgt_input'].to(self.device)        # (batch_size, tgt_len)
                tgt_output = batch['tgt_output'].to(self.device)      # (batch_size, tgt_len)
                src_lengths = batch['src_lengths'].to(self.device)
                # GRU 模型要求输入是 (seq_len, batch_size)，则转置
                src = src.transpose(0, 1)          # (src_len, batch_size)
                tgt_input = tgt_input.transpose(0, 1)  # (tgt_len, batch_size)
                tgt_output = tgt_output.transpose(0, 1)  # (tgt_len, batch_size)
                # RNN模型前向传播
                output = self.model(src, src_lengths, tgt_input, teacher_forcing_ratio=0.0) # output shape: (tgt_len, batch_size, vocab_size)
                # 计算 loss
                output_dim = output.shape[-1]
                output_flat = output.reshape(-1, output_dim)# (tgt_len * batch_size, vocab_size)
                tgt_flat = tgt_output.reshape(-1)# (tgt_len * batch_size,)
                loss = criterion(output_flat, tgt_flat)
                epoch_loss += loss.item()
        return epoch_loss / len(valid_loader)

