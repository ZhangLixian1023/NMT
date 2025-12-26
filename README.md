# 神经机器翻译系统

本项目实现了基于RNN和Transformer两种架构的神经机器翻译系统，支持中英文翻译。

## 项目结构

```
├── data/              # 数据相关代码
│   ├── dataset.py       # 数据集类
│   ├── preprocessor.py  # 数据预处理
│   ├── vocabulary.py    # 词汇表类
│   └── __init__.py      # 包初始化文件
├── dataset/           # 数据集文件
│   ├── train_10k.jsonl  # 10k训练集
│   ├── train_100k.jsonl # 100k训练集
│   ├── valid.jsonl      # 验证集
│   └── test.jsonl       # 测试集
├── models/            # 模型定义
│   ├── rnn/            # RNN相关模型
│   │   ├── attention.py  # 注意力机制
│   │   ├── decoder.py    # 解码器
│   │   ├── encoder.py    # 编码器
│   │   ├── seq2seq.py    # Seq2Seq模型
│   │   └── __init__.py   # 包初始化文件
│   └── transformer/    # Transformer相关模型
│       ├── attention.py      # 注意力机制
│       ├── decoder.py        # 解码器
│       ├── embedding.py      # 嵌入层和位置编码
│       ├── encoder.py        # 编码器
│       └── __init__.py       # 包初始化文件
├── evaluate.py        # 评估脚本
├── requirements.txt   # 依赖库
├── train.py           # 训练脚本
└── README.md          # 项目说明
```

## 环境依赖

```bash
pip install -r requirements.txt
```

## 模型训练

### RNN模型训练

```bash
# 点积注意力
python train.py --model_type rnn --attention_type dot

# 乘法注意力
python train.py --model_type rnn --attention_type multiplicative

# 加性注意力
python train.py --model_type rnn --attention_type additive
```

### Transformer模型训练

```bash
# 绝对位置嵌入 + LayerNorm
python train.py --model_type transformer --positional_encoding absolute --norm_type layernorm

# 相对位置嵌入 + LayerNorm
python train.py --model_type transformer --positional_encoding relative --norm_type layernorm

# 绝对位置嵌入 + RMSNorm
python train.py --model_type transformer --positional_encoding absolute --norm_type rmsnorm

# 相对位置嵌入 + RMSNorm
python train.py --model_type transformer --positional_encoding relative --norm_type rmsnorm
```

## 模型评估

```bash
# 评估RNN模型
python evaluate.py --model_type rnn --model_path models/saved/rnn_model.pt

# 评估Transformer模型
python evaluate.py --model_type transformer --model_path models/saved/transformer_model.pt
```

## 实验结果

实验将比较不同模型架构和配置的性能，包括：

1. **翻译质量**：使用BLEU分数评估
2. **训练效率**：训练时间和收敛速度
3. **推理速度**：生成翻译的速度
4. **资源消耗**：参数量和计算复杂度

## 注意事项

1. 首次运行时，系统会自动构建词汇表
2. 模型会保存在`models/saved/`目录下
3. 可以通过命令行参数调整各种超参数
4. 建议在GPU环境下运行，以获得更快的训练速度
