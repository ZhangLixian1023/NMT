from datetime import datetime
import pickle
import shutil
import os
import time
import torch
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 使用非交互式 backend
class RNN_Trainer:
    """Trainer class for the Seq2Seq model."""
    
    def __init__(self,
                 model: Seq2SeqGRU,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: optim.Optimizer,
                 criterion: nn.Module,
                 demo:Demo,
                 device: torch.device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.demo = demo
        self.device = device
        
    def train_epoch(self, teacher_forcing_ratio: float = 0.5) -> float:
        """Train one epoch."""
        self.model.train()
        epoch_loss = 0
        
        for batch in tqdm(self.train_loader, desc='训练'):
            # 从 batch 中取出数据（ (batch_size, seq_len) ）
            src = batch['src'].to(self.device)                    # (batch_size, src_len)
            tgt_input = batch['tgt_input'].to(self.device)        # (batch_size, tgt_len)
            tgt_output = batch['tgt_output'].to(self.device)      # (batch_size, tgt_len)
            src_lengths = batch['src_lengths'].to(self.device)

            # GRU 模型要求输入是 (seq_len, batch_size)，则转置
            src = src.transpose(0, 1)          # (src_len, batch_size)
            tgt_input = tgt_input.transpose(0, 1)  # (tgt_len, batch_size)
            tgt_output = tgt_output.transpose(0, 1)  # (tgt_len, batch_size)
            
            self.optimizer.zero_grad()

            # 前向传播：用 tgt_input 作为 decoder 的输入（teacher forcing）
            output = self.model(src, src_lengths, tgt_input,teacher_forcing_ratio)   # output shape: (tgt_len, batch_size, vocab_size)

            # 计算损失：用 output 和 tgt_output
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)       # (tgt_len * batch_size, vocab_size)
            tgt_output = tgt_output.reshape(-1)           # (tgt_len * batch_size,)

            loss = self.criterion(output, tgt_output)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=3.0)
            
            self.optimizer.step()
            epoch_loss += loss.item()
            
        return epoch_loss / len(self.train_loader)
    
    def evaluate(self) -> float:
        """Evaluate the model."""
        self.model.eval()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='验证'):
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
                output = self.model(src, src_lengths, tgt_input, teacher_forcing_ratio=0.0)
                # 计算损失：用 output 和 tgt_output
                output_dim = output.shape[-1]
                output = output.reshape(-1, output_dim)       # (tgt_len * batch_size, vocab_size)
                tgt_output = tgt_output.reshape(-1)           # (tgt_len * batch_size,)

                loss = self.criterion(output, tgt_output)
                epoch_loss += loss.item()
                
        return epoch_loss / len(self.val_loader)
    
    def train(self,
              n_epochs: int,
              settings: dict,
              patience: int = 5) -> Dict[str, Any]:
        """Complete training loop with early stopping."""
        import matplotlib.pyplot as plt
        import pandas as pd
        best_valid_loss = float('inf')
        patience_counter = 0
        train_losses = []
        valid_losses = []

        # 获取当前日期和时间（本地时间）
        now = datetime.now()
        # 格式化输出
        formatted = now.strftime("%m%d%H%M%S")
        save_dir=f"./saved_models/rnn{formatted}"
        os.makedirs(save_dir, exist_ok=True)

        # 保存实验设定和模型结构
        settings_save_path = os.path.join(save_dir, 'settings_params.txt')
        with open(settings_save_path, 'w', encoding='utf-8') as f:
            f.write("本次实验参数:\n")
            f.write(json.dumps(settings, indent=4, ensure_ascii=False))
            f.write("\n\n模型结构:\n")
            print(self.model, file=f)

        # 备份代码
        rnn_path = "./models/rnn/"
        trainer_path = "./trainer/rnn_trainer.py"
        main_path = "./main.py"
        dst_path = os.path.join(save_dir, 'code_backup')
        # 确保目标目录存在
        os.makedirs(dst_path, exist_ok=True)
        # 复制目录（注意：目标子目录不能已存在）
        shutil.copytree(rnn_path, os.path.join(dst_path, 'rnn'), dirs_exist_ok=True)
        # 复制文件
        shutil.copy(trainer_path, dst_path)
        shutil.copy(main_path, dst_path)


        for epoch in range(n_epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch(teacher_forcing_ratio=0.5)
            valid_loss = self.evaluate()
            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
            print(f'Epoch: {epoch+1:02} | 用时: {epoch_mins}m {epoch_secs:.0f}s')
            print(f'Train Loss: {train_loss:.3f} | Valid Loss: {valid_loss:.3f}')

            # 保存翻译示例（每个 epoch 都记录）
            # 文件路径用于保存翻译示例
            examples_save_path = os.path.join(save_dir, f'translation_examples_epoch{epoch+1}.csv')
            examples_output = self.demo.generate_translation_examples()
            dt=pd.DataFrame(examples_output)
            dt.to_csv(examples_save_path,index=False)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            # 保存 loss 列表（pickle）
            loss_save_path = os.path.join(save_dir, "losses.pkl")
            losses = {'train_losses': train_losses, 'valid_losses': valid_losses}
            with open(loss_save_path, 'wb') as file:
                pickle.dump(losses, file)

            # 绘制并保存 loss 曲线图（每次 epoch 后更新图）
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss', marker='o')
            plt.plot(valid_losses, label='Validation Loss', marker='s')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss per Epoch')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
            plt.close()  # 释放内存

            # 保存最佳模型
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
                model_save_path = os.path.join(save_dir, f'model_epoch{epoch+1}.pt')
                torch.save(self.model.state_dict(), model_save_path)
                print(f'模型已保存到 {model_save_path}')
            else:
                patience_counter += 1
                print(f'早停计数: {patience_counter}/{patience}')
            
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
                
        return {
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'best_valid_loss': best_valid_loss
        }