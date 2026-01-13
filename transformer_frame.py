import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import TranslationDataset, collate_fn
from models.transformer.encoder import Encoder as TransformerEncoder
from models.transformer.decoder import Decoder as TransformerDecoder
from models.transformer.transformer import Transformer
from utils import Demo, calculate_bleu4
from exp_frame import Exp_frame
class Transformer_frame(Exp_frame):
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
            "architecture":"transformer",
            "d_model": 512,  # 嵌入维度
            "n_layers": 6,  # 层数
            "n_heads":8,    # attention头数
            "d_ff":1024,    # 前馈网络隐藏层维度
            "dropout": 0.3,
            "relative_position": True,  # 位置嵌入类型 absolute 或 relative
            "norm" : "rmsnorm", # 归一化类型 layer 或 rms
            "src_vocab_size":self.src_vocab.n_words,
            "tgt_vocab_size":self.tgt_vocab.n_words,
            "freeze_embedding": False     # 是否冻结预训练词向量
            } 
        self.exp_setting={
            "max_seq_len": 90, # 不要超过100，因为位置编码最大长度是100
            "batch_size": 64,
            "learning_rate": 1e-4,
            "patience": 2,
            "teacher_forcing_ratio":1.0,
            "start_from": "scratch",
            "others": ""
            }

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
            relative_position=self.model_params['relative_position']
        ).to(self.device)

        decoder = TransformerDecoder(
            output_size=self.tgt_vocab.n_words,
            d_model=self.model_params['d_model'],
            n_layers=self.model_params['n_layers'],
            n_heads=self.model_params['n_heads'],
            d_ff=self.model_params['d_ff'],
            dropout=self.model_params['dropout'],
            norm_type=self.model_params['norm'],
            relative_position=self.model_params['relative_position']
        ).to(self.device)

        self.model = Transformer(encoder, decoder, self.device).to(self.device)
        self.demo = Demo(
            self.model,
            self.src_vocab,
            self.tgt_vocab,
            examples=self.valid_pairs[0:20],
            device=self.device
        )
        self.exp_setting['start_from'] = 'scratch'

        #print(f"Done: Init model with {src_embedding_file} and {tgt_embedding_file}.")
        if save==True:
            self._init_saver()
        self.save=save
    
    def load_model(self,model_file="./saved_models/transformer1228190737_from_other/Transformer1228190737_epoch_12.pt",save=True):
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
            relative_position=self.model_params['relative_position']
        ).to(self.device)

        decoder = TransformerDecoder(
            output_size=self.tgt_vocab.n_words,
            d_model=self.model_params['d_model'],
            n_layers=self.model_params['n_layers'],
            n_heads=self.model_params['n_heads'],
            d_ff=self.model_params['d_ff'],
            dropout=self.model_params['dropout'],
            norm_type=self.model_params['norm'],
            relative_position=self.model_params['relative_position']
        ).to(self.device)

        self.model = Transformer(encoder, decoder, self.device).to(self.device)
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.exp_setting['start_from'] = 'other'
        self.demo = Demo(
            self.model,
            self.src_vocab,
            self.tgt_vocab,
            examples=self.test_pairs,
            device=self.device
        )
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
                # Transformer模型前向传播
                output = self.model(src,  tgt_input)  # (batch_size, tgt_len, vocab_size )
                # 计算损失：用 output 和 tgt_output
                output_dim = output.shape[-1]
                output_flat = output.reshape(-1, output_dim)  # (tgt_len * batch_size, vocab_size)
                tgt_flat = tgt_output.reshape(-1)# (tgt_len * batch_size,)
                loss = criterion(output_flat, tgt_flat)
                epoch_loss += loss.item()
        return epoch_loss / len(valid_loader)
