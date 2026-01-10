# Copilot instructions for this repository

Purpose: give AI coding agents the precise, discoverable knowledge to be productive immediately.

- **Big picture:** This repo implements a Chinese↔English NMT system with two model families:
  - RNN-based seq2seq under `models/rnn/` (encoder, decoder, attention, `seq2seq.py`).
  - Transformer under `models/transformer/` (encoder, decoder, attention, `transformer.py`).
  Training/evaluation workflows are provided as interactive experiment frames: `exp_frame.py` (RNN) and `transformer_frame.py` (Transformer).

- **Entrypoints / developer commands:**
  - Install deps: `pip install -r requirements.txt` (HanLP and sacrebleu required).
  - Interactive experiments: run `python -i exp_frame.py` or `python -i transformer_frame.py` and instantiate `Exp_frame()` / `transformer_frame()` as shown in those files.
  - Data preprocessing is in `data/preprocessor.py`; use `Preprocessor` to produce space-separated token strings consumed by `data/dataset.py`.

- **Data & tokenization conventions:**
  - Raw dataset files are newline-delimited JSON in `dataset/` (fields `en` and `zh`).
  - `Preprocessor.tokenize()` returns space-separated tokens. `TranslationDataset` expects tokenized strings and returns index tensors.
  - Special token indices (used across code): `pad=0`, `sos=2`, `eos=3`.
  - `TranslationDataset.__getitem__` produces `tgt_input` (prefixed with `sos`) and `tgt_output` (suffixed with `eos`). `collate_fn` pads with 0.

- **Model I/O and shapes to watch:**
  - RNN Seq2Seq uses shape (seq_len, batch_size) for encoder/decoder inputs; Transformer uses (batch_size, seq_len).
  - RNN training uses `nn.NLLLoss(ignore_index=0)`; Transformer uses `nn.CrossEntropyLoss(ignore_index=0)`.
  - Gradient clipping: `torch.nn.utils.clip_grad_norm_(..., max_norm=3.0)` is applied in training loops.

- **Saved artifacts & vocab/embeddings:**
  - Prebuilt vocab and embeddings are under `saved_vocab_embedding/` (e.g. `src_vocab.pkl`, `src_embedding.pkl`). Load these when initializing models.
  - Trained models are saved as state_dict files under `saved_models/` in folder-per-experiment; use `torch.load(..., map_location=device)` to restore.

- **Training conventions and hyperparameters to copy:**
  - Transformer default LR and optimizer details are in `transformer_frame.py` (Adam with betas=(0.9,0.98), eps=1e-9).
  - Teacher forcing default is `1.0` in the frames; patience-based early stopping is implemented using `exp_setting['patience']`.
  - Transformer often uses a very large `batch_size` in the frame (example: 1536) — be careful with GPU memory.

- **Evaluation & BLEU:**
  - BLEU is computed with `sacrebleu.corpus_bleu(...)` in `exp_frame.py` and `transformer_frame.py`. Utility wrappers in `utils/` also contain helper functions like `calculate_bleu4`.

- **Project-specific patterns to follow when editing:**
  - Use the experiment frames for integrating new training loops or diagnostics rather than adding disparate scripts — they centralize saving, demo generation, and backing up code.
  - Keep data tokenization and vocabulary generation in `data/` to avoid subtle index mismatches; vocab objects expose `word_to_idx`, `idx_to_word`, and `n_words`.
  - When adding features that change token indices or vocab, ensure backward compatibility with pickled `saved_vocab_embedding` files.

- **Files to open first when changing behavior:**
  - `exp_frame.py`, `transformer_frame.py` — experiment orchestration and saving conventions.
  - `data/preprocessor.py`, `data/dataset.py` — tokenization and batching rules.
  - `models/rnn/*` and `models/transformer/*` — implement model behavior and expected input shapes.

Quick examples (interactively):

1) Start transformer interactive session, init model, train a few epochs:

   python -i transformer_frame.py
   f = transformer_frame()       # loads vocab and settings
   f.init_model()                # initialize model (loads embeddings if available)
   f.train(n_epochs=2)           # trains and saves checkpoints

2) Load a saved RNN model and run a demo:

   python -i exp_frame.py
   e = Exp_frame()
   e.load_model('saved_models/your_rnn_dir/your_model.pt')
   e.test(e.test_pairs[0:10])

If anything here is unclear or you'd like more detail (examples for a specific file or a small runnable test), tell me which part to expand.
