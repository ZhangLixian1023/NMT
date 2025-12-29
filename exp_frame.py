from datetime import datetime
import shutil
import time
import json
import matplotlib
matplotlib.use('Agg')  # 使用非交互式 backend
import pandas as pd
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sacrebleu
from tqdm import tqdm
from data import TranslationDataset, collate_fn
from models.rnn.encoder import Encoder as RNNEncoder
from models.rnn.decoder import Decoder as RNNDecoder
from models.rnn.seq2seq import Seq2Seq as RNNSeq2Seq
from utils import Demo, load_pairs
class Exp_frame:
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
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {self.device}')
        self.load_vocab(
            src_vocab_file = './saved_vocab_embedding/src_vocab.pkl' ,
            tgt_vocab_file = './saved_vocab_embedding/tgt_vocab.pkl'
            )
        self.model_params = {
            "architecture":"rnn-gru",
            "hidden_size": 512,
            "num_layers": 2,
            "dropout": 0.3,
            "attention_type": "dot",  # 注意力机制类型：'bahdanau' 或 'luong'
            "src_vocab_size":self.src_vocab.n_words,
            "tgt_vocab_size":self.tgt_vocab.n_words,
            "freeze_embedding": False     # 是否冻结预训练词向量
            }
        self.exp_setting={
            "train_dataset": "./dataset/train_100k_pairs.jsonl", # 训练集
            "valid_dataset": "./dataset/valid_pairs.jsonl", # 验证集
            "test_dataset":"./dataset/test_pairs.jsonl", # 测试集
            "max_seq_len": 40,
            "batch_size": 64,
            "learning_rate": 1e-4,
            "patience": 2,
            "teacher_forcing_ratio":1.0,
            "start_from": "scratch",
            "others": ""
            }
        self.load_data()
        self.trained_epochs=0
        self.train_losses=[]
        self.valid_losses=[]
        self.bleu_scores=[]
        self.best_valid_loss=float('inf')
        self.save=True

    def load_vocab(self,src_vocab_file,tgt_vocab_file):
        # 加载词库 (词-->index)
        with open(src_vocab_file, 'rb') as f:
            self.src_vocab= pickle.load(f)
        with open(tgt_vocab_file, 'rb') as f:
            self.tgt_vocab= pickle.load(f)
        print("Done: Load vocabulary.")

    def init_model(self,
    src_embedding_file='./saved_vocab_embedding/src_embedding.pkl' ,
    tgt_embedding_file='./saved_vocab_embedding/tgt_embedding.pkl',save=True
    ):
        # 加载预训练词向量嵌入矩阵
        with open(src_embedding_file, 'rb') as f:
            src_embedding_matrix = pickle.load(f)
        with open(tgt_embedding_file, 'rb') as f:
            tgt_embedding_matrix = pickle.load(f)

        # 初始化RNN模型
        encoder = RNNEncoder(
            input_size=self.src_vocab.n_words,
            hidden_size=self.model_params['hidden_size'],
            num_layers=self.model_params['num_layers'],
            dropout=self.model_params['dropout'],
            pretrained_embedding=src_embedding_matrix,
            freeze_embedding=self.model_params['freeze_embedding']
        ).to(self.device)

        decoder = RNNDecoder(
            output_size=self.tgt_vocab.n_words,
            hidden_size=self.model_params['hidden_size'],
            num_layers=self.model_params['num_layers'],
            dropout=self.model_params['dropout'],
            attention_type=self.model_params['attention_type'],
            pretrained_embedding=tgt_embedding_matrix,
            freeze_embedding=self.model_params['freeze_embedding']
        ).to(self.device)

        self.model = RNNSeq2Seq(encoder, decoder, self.device).to(self.device)
        self.exp_setting['start_from'] = 'scratch'
        self.demo = Demo(
            self.model,
            self.src_vocab,
            self.tgt_vocab,
            examples=self.test_pairs[0:8],
            device=self.device
        )
        print(f"Done: Init model with {src_embedding_file} and {tgt_embedding_file}.")
        if save==True:
            self._init_saver()
        self.save=save
    
    def load_data(self):
        # 加载数据集
        self.train_pairs = load_pairs(self.exp_setting['train_dataset'])
        self.valid_pairs = load_pairs(self.exp_setting['valid_dataset'])
        self.test_pairs = load_pairs(self.exp_setting['test_dataset'])
    
    def load_model(self,model_file="./saved_models/rnn1227193814/model_epoch19.pt",save=True):
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

    def _init_saver(self):
        # 获取当前日期和时间（本地时间）
        formatted = datetime.now().strftime("%m%d%H%M%S")
        # 全部内容保存目录
        self.save_dir=f"./saved_models/rnn{formatted}_from_{self.exp_setting['start_from']}"
        self.save_prefix = f"rnn{formatted}"
        os.makedirs(self.save_dir, exist_ok=True)

        # 保存实验设定 和 模型结构
        params_save_path = os.path.join(self.save_dir, self.save_prefix + 'model_params.txt')
        with open(params_save_path, 'w', encoding='utf-8') as f:
            f.write("\n模型结构:\n")
            f.write(json.dumps(self.model_params, indent=4, ensure_ascii=False))
            f.write("\n")
            print(self.model, file=f)
        self.save_setting()
        # 备份代码
        rnn_path = "./models/rnn/"
        main_path = "./exp_frame.py"
        dst_path = os.path.join(self.save_dir, self.save_prefix+'_code_backup')
        os.makedirs(dst_path, exist_ok=True)
        shutil.copytree(rnn_path, os.path.join(dst_path, 'rnn'), dirs_exist_ok=True)
        shutil.copy(main_path, dst_path)
        print("Done: init saver.")

    def save_setting(self,comment=''):
        settings_save_path = os.path.join(self.save_dir, self.save_prefix + 'exp_setting.txt')
        with open(settings_save_path, 'a', encoding='utf-8') as f:
            if comment !='':
                f.write("\n\ncomment:\n"+comment+'\n')
            f.write("本次实验参数:\n")
            f.write(json.dumps(self.exp_setting, indent=4, ensure_ascii=False))
            
    def _save_epoch(self):
        if self.save==False:
            return
        # 保存翻译示例（每个 epoch 都记录）
        examples_save_path = os.path.join(self.save_dir, self.save_prefix+f'_examples_epoch{self.trained_epochs}.csv')
        examples_output = self.demo.generate_translation_examples()
        dt = pd.DataFrame(examples_output)
        dt.to_csv(examples_save_path,index=False)

        # 保存 loss 列表（pickle）
        loss_save_path = os.path.join(self.save_dir, self.save_prefix+'_losses.pkl')
        losses = {'train_losses': self.train_losses, 'valid_losses': self.valid_losses}
        with open(loss_save_path, 'wb') as file:
            pickle.dump(losses, file)

        # 保存 bleu_scores 列表 txt
        bleu_save_path = os.path.join(self.save_dir, self.save_prefix+'_bleu_scores.txt')
        with open(bleu_save_path, 'w', encoding='utf-8') as f:
            for score in self.bleu_scores:
                f.write(f"{score}\n")

        import matplotlib.pyplot as plt
        # 绘制并保存 loss 曲线图（每次 epoch 后更新图）
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', marker='o')
        plt.plot(self.valid_losses, label='Validation Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss per Epoch')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, self.save_prefix+'_loss_curve.png'))
        plt.close()  # 释放内存
        
        # 保存模型
        model_save_path = os.path.join(self.save_dir, self.save_prefix+f'_epoch_{self.trained_epochs}.pt')
        torch.save(self.model.state_dict(), model_save_path)
        print(f'模型已保存到 {model_save_path}')

    def train(self, n_epochs = 2):
        # 创建数据集和数据加载器
        train_dataset = TranslationDataset(self.train_pairs, self.src_vocab, self.tgt_vocab, max_length=self.exp_setting['max_seq_len'])
        valid_dataset = TranslationDataset(self.valid_pairs, self.src_vocab, self.tgt_vocab, max_length=self.exp_setting['max_seq_len'])
        train_loader = DataLoader(train_dataset, batch_size=self.exp_setting['batch_size'], collate_fn=collate_fn, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.exp_setting['batch_size'], collate_fn=collate_fn)
        # 损失函数和优化器
        criterion = nn.NLLLoss(ignore_index=0)  # 忽略填充标记
        optimizer = optim.Adam(self.model.parameters(), lr=self.exp_setting['learning_rate'])

        patience_counter = 0
        epoch = 0
        while epoch <= n_epochs:
            if epoch == n_epochs:
                more = input("Last epoch. Enter 0 to quit. Enter n for more epochs.\n")
                if int(more)==0:
                    break
                n_epochs += int(more)

            epoch+=1
            self.trained_epochs+=1
            start_time = time.time()
            train_loss = self.train_epoch(train_loader,criterion,optimizer)
            valid_loss, bleu_score = self.evaluate(valid_loader,criterion)
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            self.bleu_scores.append(bleu_score)
            self._save_epoch()
            # 打印 Loss 和 用时
            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
            print(f'Epoch: {self.trained_epochs:02} | 用时: {epoch_mins}m {epoch_secs:.0f}s')
            print(f'Train Loss: {train_loss:.3f} | Valid Loss: {valid_loss:.3f} | BLEU: {bleu_score*100:.2f}')

            # 早停
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                patience_counter = 0
            else:
                patience_counter += 1
                print(f'早停计数: {patience_counter}/{self.exp_setting["patience"]}')
            
            if patience_counter >= self.exp_setting["patience"]:
                print(f'Early stopping at epoch {self.trained_epochs}')
                break

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
            valid_dataset = TranslationDataset(self.valid_pairs, self.src_vocab, self.tgt_vocab, max_length=self.exp_setting['max_seq_len'])
            valid_loader = DataLoader(valid_dataset, batch_size=self.exp_setting['batch_size'], collate_fn=collate_fn)
            # 损失函数和优化器
            criterion = nn.NLLLoss(ignore_index=0)  # 忽略填充标记

        self.model.eval()
        epoch_loss = 0
        all_preds = []   # 存储所有预测句子（字符串）
        all_refs = []    # 存储所有参考句子（字符串）

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

                # 准备计算 BLEU 分数
                pred_tokens = output.argmax(dim=-1).transpose(0, 1)  # (B, tgt_len)
                tgt_output_batch = tgt_output.transpose(0, 1)        # (B, tgt_len)
                # 遍历 batch 中每个样本
                for i in range(pred_tokens.size(0)):
                    # 移除填充（假设 0 是 <pad>）
                    pred_seq = pred_tokens[i].cpu().tolist()
                    ref_seq = tgt_output_batch[i].cpu().tolist()
                    # 移除 <pad>（0）
                    pred_seq = [x for x in pred_seq if x != 0]
                    ref_seq = [x for x in ref_seq if x != 0]

                    # 转回 token 字符串（使用 tgt_vocab 的反向映射）
                    pred_text = ' '.join([self.tgt_vocab.idx_to_word(idx) for idx in pred_seq])
                    ref_text = ' '.join([self.tgt_vocab.idx_to_word(idx) for idx in ref_seq])

                    all_preds.append(pred_text)
                    all_refs.append(ref_text)

        avg_loss = epoch_loss / len(valid_loader)
        bleu = sacrebleu.corpus_bleu(all_preds, all_refs)
        return avg_loss, bleu.score

    def test(self, test_data=None, noref = False):
        # 加载测试数据
        if test_data == None:
            test_data = self.test_pairs[0:1]
        
        # 在测试集上展示demo
        demo = Demo(
            self.model,
            self.src_vocab,
            self.tgt_vocab,
            examples=test_data,
            device=self.device
        )
        result = demo.generate_translation_examples(noref)

        # 如果有参考，计算BLEU分数
        if not noref:        
            refs=[]
            hyps=[]
            for (src,tgt,translation) in result:
                refs.append(tgt)
                hyps.append(translation)
            bleu = sacrebleu.corpus_bleu(refs,hyps)
            print(f"BLEU: {bleu.score}\n")
