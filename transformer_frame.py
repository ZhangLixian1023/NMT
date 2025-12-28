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
from tqdm import tqdm
from data import TranslationDataset, collate_fn
from models.transformer.encoder import Encoder as TransformerEncoder
from models.transformer.decoder import Decoder as TransformerDecoder
from models.transformer.transformer import Transformer
from utils import Demo, calculate_bleu4, load_pairs
class transformer_frame:
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
            "architecture":"Transformer",
            "d_model": 512,
            "n_layers": 6,  # 层数
            "n_heads":8,    # attention头数
            "d_ff":1024,    # 前馈网络隐藏层维度
            "dropout": 0.3,
            "positional": "absolute",  # 位置嵌入类型 absolute 或 relative
            "norm" : "layernorm", # 归一化类型 layer 或 rms
            "src_vocab_size":self.src_vocab.n_words,
            "tgt_vocab_size":self.tgt_vocab.n_words,
            "freeze_embedding": False     # 是否冻结预训练词向量
            }
        self.exp_setting={
            # "train_dataset":"./dataset/short.jsonl",
            # "valid_dataset":"./dataset/short.jsonl",            
            # "test_dataset":"./dataset/short.jsonl",
            "train_dataset": "./dataset/train_100k_pairs.jsonl", # 训练集
            "valid_dataset": "./dataset/valid_pairs.jsonl", # 验证集
            "test_dataset":"./dataset/test_pairs.jsonl", # 测试集
            "max_seq_len": 40,
            "batch_size": 512,
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

        # 初始化Transformer模型
        encoder = TransformerEncoder(
            input_size=self.src_vocab.n_words,
            d_model=self.model_params['d_model'],
            n_layers=self.model_params['n_layers'],
            n_heads=self.model_params['n_heads'],
            d_ff=self.model_params['d_ff'],
            dropout=self.model_params['dropout'],
            norm_type=self.model_params['norm'],
            embedding_type=self.model_params['positional']
        ).to(self.device)

        decoder = TransformerDecoder(
            output_size=self.tgt_vocab.n_words,
            d_model=self.model_params['d_model'],
            n_layers=self.model_params['n_layers'],
            n_heads=self.model_params['n_heads'],
            d_ff=self.model_params['d_ff'],
            dropout=self.model_params['dropout'],
            norm_type=self.model_params['norm'],
            embedding_type=self.model_params['positional']
        ).to(self.device)

        self.model = Transformer(encoder, decoder, self.device).to(self.device)
        self.exp_setting['start_from'] = 'scratch'
        self.demo = Demo(
            self.model,
            self.src_vocab,
            self.tgt_vocab,
            examples=self.train_pairs[0:20],
            device=self.device
        )
        #print(f"Done: Init model with {src_embedding_file} and {tgt_embedding_file}.")
        if save==True:
            self._init_saver()
        self.save=save
    
    def load_data(self):
        # 加载数据集
        self.train_pairs = load_pairs(self.exp_setting['train_dataset'])
        self.valid_pairs = load_pairs(self.exp_setting['valid_dataset'])
        self.test_pairs = load_pairs(self.exp_setting['test_dataset'])
    
    def load_model(self,model_file="./saved_models/transformer1228174409_from_scratch/Transformer1228174409_epoch_40.pt",save=True):
        self.exp_setting['from_model']=model_file
        # 初始化Transformer模型
        encoder = TransformerEncoder(
            input_size=self.src_vocab.n_words,
            d_model=self.model_params['d_model'],
            n_layers=self.model_params['n_layers'],
            n_heads=self.model_params['n_heads'],
            d_ff=self.model_params['d_ff'],
            dropout=self.model_params['dropout'],
            norm_type=self.model_params['norm'],
            embedding_type=self.model_params['positional']
        ).to(self.device)

        decoder = TransformerDecoder(
            output_size=self.tgt_vocab.n_words,
            d_model=self.model_params['d_model'],
            n_layers=self.model_params['n_layers'],
            n_heads=self.model_params['n_heads'],
            d_ff=self.model_params['d_ff'],
            dropout=self.model_params['dropout'],
            norm_type=self.model_params['norm'],
            embedding_type=self.model_params['positional']
        ).to(self.device)

        self.model = Transformer(encoder, decoder, self.device).to(self.device)
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
        self.save_dir=f"./saved_models/transformer{formatted}_from_{self.exp_setting['start_from']}"
        self.save_prefix = f"Transformer{formatted}"
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
        Transformer_path = "./models/transformer/"
        main_path = "./transformer_frame.py"
        dst_path = os.path.join(self.save_dir, self.save_prefix+'_code_backup')
        os.makedirs(dst_path, exist_ok=True)
        shutil.copytree(Transformer_path, os.path.join(dst_path, 'Transformer'), dirs_exist_ok=True)
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
        train_loader = DataLoader(train_dataset, batch_size=self.exp_setting['batch_size'], collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=self.exp_setting['batch_size'], collate_fn=collate_fn)

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充标记
        optimizer = optim.Adam(self.model.parameters(), lr=self.exp_setting['learning_rate'], betas=(0.9, 0.98), eps=1e-9)

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
            valid_loss = self.evaluate(valid_loader,criterion)
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            self._save_epoch()
            # 打印 Loss 和 用时
            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
            print(f'Epoch: {self.trained_epochs:02} | 用时: {epoch_mins}m {epoch_secs:.0f}s')
            print(f'Train Loss: {train_loss:.3f} | Valid Loss: {valid_loss:.3f}')

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

            # # 维度检查
            # print("\n train epoch")
            # print(f"src shape = {src.shape}")
            # print(f"tgt_input shape = {tgt_input.shape}")
            optimizer.zero_grad()

            # 前向传播：用 tgt_input 作为 decoder 的输入（teacher forcing）
            output = self.model(src, tgt_input)   # output shape: (batch_size, tgt_len,  vocab_size)

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
            criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充标记

        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc='验证'):
                # 从 batch 中取出数据（ (batch_size, seq_len) ）
                src = batch['src'].to(self.device)                    # (batch_size, src_len)
                tgt_input = batch['tgt_input'].to(self.device)        # (batch_size, tgt_len)
                tgt_output = batch['tgt_output'].to(self.device)      # (batch_size, tgt_len)
                src_lengths = batch['src_lengths'].to(self.device)

                # Transformer模型前向传播
                output = self.model(src,  tgt_input)  # (batch_size, tgt_len, vocab_size )
                # 计算损失：用 output 和 tgt_output
                output_dim = output.shape[-1]
                output = output.reshape(-1, output_dim)       # (tgt_len * batch_size, vocab_size)
                tgt_output = tgt_output.reshape(-1)           # (tgt_len * batch_size,)

                loss = criterion(output, tgt_output)
                epoch_loss += loss.item()
        return epoch_loss / len(valid_loader)

    def test(self, test_data=None, noref = False):
        # 加载测试数据
        if test_data == None:
            test_data = self.test_pairs[50:60]
        
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
            bleu4, p1, p2, p3, p4 = calculate_bleu4(refs,hyps)
            print(f"p1: {p1}\n")
            print(f"p2: {p2}\n")
            print(f"p3: {p3}\n")
            print(f"p4: {p4}\n")
            print(f"bleu4: {bleu4}\n")
