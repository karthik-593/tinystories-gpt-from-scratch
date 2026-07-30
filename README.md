# TinyStories GPT

A small GPT-style language model built from scratch in PyTorch (no `transformers` library), pretrained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset and instruction fine-tuned to generate short children's stories from natural-language prompts.

Tuned for a single RTX 4060 Ti (16GB), but runs on any CUDA GPU or CPU.

## Architecture

Decoder-only Transformer ([model/gpt.py](model/gpt.py)), config in [model/config.py](model/config.py):

- `d_model=448`, `n_heads=7` (head_dim=64), `n_layers=7`, `d_ff=1792`
- `context_len=256`, `vocab_size=32,000`
- Pre-norm residual blocks, fused QKV projection, GELU MLP, weight-tied embedding/LM head
- ~31M parameters

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

`torch` in `requirements.txt` installs a CPU build by default. For GPU training, install the matching CUDA wheel from https://pytorch.org/get-started/locally/ instead (this project was trained against `torch==2.11.0+cu128`).

All commands below are run from the repo root.

## Pipeline

Run in order:

### 1. Data setup (one-time)

```bash
python scripts/setup_data.py
```

Downloads 75% of TinyStories, trains a 32K-vocab BPE tokenizer (`tokenizer/tokenizer.json`), and tokenizes everything into `uint16` shards under `data/bin/` (`train_shard_*.bin`, `val_shard_*.bin`).

### 2. Pretraining

```bash
python scripts/train.py
# or, for multi-GPU:
torchrun --nproc_per_node=2 scripts/train.py
```

Trains on the shards from step 1 (~330M tokens, 2 epochs, effective batch of 512 sequences). Saves checkpoints to `checkpoints/` (`ckpt_latest.pt`, `ckpt_best.pt`, `ckpt_final.pt`). The best checkpoint used for fine-tuning is kept at the repo root as `ckpt_best.pt`.

### 3. Clean instruction data

```bash
python scripts/clean_data.py --input data/final_data.jsonl --output data/final_data_cleaned.jsonl
```

Dedupes, caps outputs per instruction (5), drops outputs under 10 words, truncates outputs that would overflow the model's context window (measured in actual tokens against the exact fine-tuning prompt template, not a word-count guess), and fixes missing terminal punctuation.

### 4. Instruction fine-tuning

```bash
python scripts/finetune.py
```

Loads `ckpt_best.pt` and fine-tunes on `data/final_data_cleaned.jsonl` using an Alpaca-style `### Instruction / ### Input / ### Response` template, with loss masked to response tokens only. Saves to `checkpoints_ft/` (`ft_ckpt_best.pt`, `ft_ckpt_latest.pt`, `ft_ckpt_final.pt`).

### 5. Inference

```bash
python scripts/inference.py                                    # uses checkpoints_ft/ft_ckpt_best.pt
python scripts/inference.py --ckpt ckpt_best.pt                 # use the pretrained (non-instruction-tuned) model instead
```

Interactive REPL — enter an instruction (and optional input), then adjust generation params. In-session commands: `:temp <float>`, `:top_p <float>`, `:tokens <int>`, `:quit`.

## Project layout

```
model/                           Model package
  config.py                        Model/training hyperparameters (ModelConfig)
  gpt.py                           Transformer model definition
scripts/                         Entry-point scripts (run from the repo root)
  setup_data.py                    Downloads TinyStories, trains tokenizer, builds token shards
  train.py                         Pretraining loop
  clean_data.py                    Cleans raw instruction data for fine-tuning
  finetune.py                      Instruction fine-tuning loop
  inference.py                     Interactive generation REPL
data/
  final_data.jsonl                 Raw instruction dataset
  final_data_cleaned.jsonl         Cleaned instruction dataset (fine-tuning input)
tokenizer/tokenizer.json         Trained BPE tokenizer
ckpt_best.pt                     Best pretrained checkpoint
checkpoints_ft/                  Fine-tuned checkpoints
```
