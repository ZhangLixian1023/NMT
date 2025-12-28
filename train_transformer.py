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
