from abc import ABC, abstractmethod
from datetime import datetime
import shutil
import time
import json
import matplotlib
matplotlib.use('Agg')  # 使用非交互式 backend
import pandas as pd
import os
from utils import Demo, load_pairs , calculate_bleu4
from data import TranslationDataset, collate_fn
from torch.utils.data import DataLoader
from pprint import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
class Exp_frame(ABC):
    """抽象基类 实验框架"""
    def __init__(self):
        self.device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'使用设备: {self.device}')
        self.load_vocab(
            src_vocab_file = './saved_vocab_embedding/src_vocab.pkl' ,
            tgt_vocab_file = './saved_vocab_embedding/tgt_vocab.pkl'
            )
        self.dataset_paths = {
            "train_dataset": "./dataset/train_100k_pairs.jsonl", # 训练集
            "valid_dataset": "./dataset/valid_pairs.jsonl", # 验证集
            "test_dataset":"./dataset/test_pairs.jsonl", # 测试集
            }
        self.load_data()        
        self.trained_epochs=0
        self.train_losses=[]
        self.valid_losses=[]
        self.best_valid_loss=float('inf')
        self.save=True

    @abstractmethod
    def load_model(self,model_file,save=True):
        """load model from file."""
        pass
    
    @abstractmethod
    def train_epoch(self, train_loader, criterion, optimizer) -> float: 
        """Train one epoch."""
        pass

    @abstractmethod
    def evaluate(self, valid_loader=None, criterion=None) -> float:
        """Evaluate the model."""
        pass
    
    def load_data(self):
        # 加载数据集
        self.train_pairs = load_pairs(self.dataset_paths['train_dataset'])
        self.valid_pairs = load_pairs(self.dataset_paths['valid_dataset'])
        self.test_pairs = load_pairs(self.dataset_paths['test_dataset'])
    
    def load_vocab(self,src_vocab_file,tgt_vocab_file):
        # 加载词库 (词-->index)
        with open(src_vocab_file, 'rb') as f:
            self.src_vocab= pickle.load(f)
        with open(tgt_vocab_file, 'rb') as f:
            self.tgt_vocab= pickle.load(f)
        print("Done: Load vocabulary.")

    def _init_saver(self):
        # 获取当前日期和时间（本地时间）
        formatted = datetime.now().strftime("%m%d%H%M%S")
        architecture = self.model_params['architecture']
        # 全部内容保存目录
        self.save_dir=f"./saved_models/{architecture}{formatted}_from_{self.exp_setting['start_from']}"
        self.save_prefix = f"{architecture}{formatted}"
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
        model_path = f"./models/{architecture}/"
        main_path = "./exp_frame.py"
        dst_path = os.path.join(self.save_dir, self.save_prefix+'_code_backup')
        os.makedirs(dst_path, exist_ok=True)
        shutil.copytree(model_path, os.path.join(dst_path, architecture), dirs_exist_ok=True)
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
        # examples_save_path = os.path.join(self.save_dir, self.save_prefix+f'_examples_epoch{self.trained_epochs}.csv')
        # examples_output = self.demo.generate_translation_examples()
        # dt = pd.DataFrame(examples_output)
        # dt.to_csv(examples_save_path,index=False)

        # 保存 loss 列表（pickle）
        loss_save_path = os.path.join(self.save_dir, self.save_prefix+'_losses.txt')
        losses = {'train_losses': self.train_losses, 'valid_losses': self.valid_losses}
        with open(loss_save_path, 'w') as file:
            file.write(str(losses))

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
        if self.model_params['architecture']=='transformer':
            print('Exp_frame: transformer, use CE loss')
            criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略填充标记
        else:
            print('Exp_frame: rnn, use NLL Loss')
            criterion = nn.NLLLoss(ignore_index=0)  # 忽略填充标记
        optimizer = optim.Adam(self.model.parameters(), lr=self.exp_setting['learning_rate'])

        patience_counter = 0
        epoch = 0
        while epoch < n_epochs:
            #if epoch == n_epochs:
                # more = input("Last epoch. Enter 0 to quit. Enter n for more epochs.\n")
                # if int(more)==0:
                #     break
                # n_epochs += int(more)

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
    
    def test(self, test_data, noref = False,strategy='greedy',show=True):
        """Test the model on test_data (list of (src_sentence, tgt_sentence) pairs)."""       
        '''
        可以选择数据集中的部分数据进行测试
        test_data = self.test_pairs[0:10]  # 仅测试前10条
        或者自己构造数据
        例如[('欧洲 国家 普遍 相信', 'European countries generally believe')]
        '''
        # 在测试集上展示demo
        demo = Demo(
            self.model,
            self.src_vocab,
            self.tgt_vocab,
            examples=test_data,
            device=self.device,
            strategy = strategy
        )
        result = demo.generate_translation_examples(noref,show=show)

        # 如果有参考，计算BLEU分数
        if not noref:
            refs=[]
            hyps=[]
            for item in result:
                tgt = item['tgt']
                translation = item['translation']
                refs.append(tgt.split())
                tlist = translation.split()
                if tlist[-1]=='<eos>':
                    tlist = tlist[:-1]
                hyps.append(tlist)
                #print(len(tlist))
            bleu = calculate_bleu4(hyps, refs)
            pprint(bleu)
