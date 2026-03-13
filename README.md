# BERT Pre-training from Scratch

完整的 BERT 預訓練實驗框架，遵循原始論文 (Devlin et al., 2019)。

- **預訓練任務**: Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)
- **數據集**: English Wikipedia（可選 BookCorpus）
- **框架**: PyTorch + HuggingFace Transformers
- **分佈式**: 支持單卡 / 多卡（via Accelerate）

## 項目結構

```
bert_pretrain/
├── configs/
│   ├── bert_base.json          # BERT-base (110M params)
│   └── bert_small.json         # BERT-small (用於調試，~14M params)
├── scripts/
│   ├── download_data.sh        # 數據下載與預處理
│   ├── train.sh                # 單卡訓練
│   ├── train_distributed.sh    # 多卡分佈式訓練
│   └── evaluate.sh             # 模型評估
├── src/
│   ├── dataset.py              # MLM + NSP 數據集實現
│   ├── preprocess.py           # Wikipedia/BookCorpus 預處理
│   ├── pretrain.py             # 單卡訓練腳本
│   ├── pretrain_accelerate.py  # 分佈式訓練腳本 (Accelerate)
│   └── evaluate.py             # 評估腳本 (MLM PPL + NSP Acc)
├── requirements.txt
└── README.md
```

## 快速開始

### 0. 環境準備

```bash
cd bert_pretrain
pip install -r requirements.txt
```

### 1. 數據下載與預處理

```bash
# 下載完整 Wikipedia（約需 20-40 分鐘，取決於網速）
bash scripts/download_data.sh

# 或者：僅下載少量數據用於調試
bash scripts/download_data.sh --max_articles 10000

# 同時下載 Wikipedia + BookCorpus
bash scripts/download_data.sh --dataset all
```

數據會被處理成文檔級別的 pickle 文件，存放在 `data/processed/` 下。

### 2. 訓練

#### 單卡訓練

```bash
# BERT-small 快速調試（推薦先跑通）
bash scripts/train.sh --config bert_small --seq_len 128 --batch_size 64 --fp16

# BERT-base 標準訓練
bash scripts/train.sh --config bert_base --seq_len 128 --batch_size 32 --epochs 3 --fp16
```

#### 多卡分佈式訓練

```bash
# 首次使用需配置 Accelerate
accelerate config

# 4 卡訓練
bash scripts/train_distributed.sh --num_gpus 4 --config bert_base --batch_size 32 --fp16

# 使用梯度累積模擬更大 batch size（每卡 batch=32, 累積 8 步 → 有效 batch=32×4×8=1024）
bash scripts/train_distributed.sh --num_gpus 4 --batch_size 32 --grad_accum 8 --fp16
```

### 3. 評估

```bash
bash scripts/evaluate.sh --model_path output/bert_base_seq128/final_model
```

輸出指標：
- **MLM Perplexity**: 越低越好，BERT-base 在 Wiki 上通常 ~4-8
- **NSP Accuracy**: 通常 >95%

## 訓練策略說明

### 原始 BERT 論文的兩階段訓練

原論文使用兩階段：
1. **Phase 1**: `max_seq_length=128`，訓練 90% 的步數
2. **Phase 2**: `max_seq_length=512`，訓練剩餘 10% 的步數

執行方式：

```bash
# Phase 1: seq_len=128, 90% steps
bash scripts/train.sh --config bert_base --seq_len 128 --max_steps 900000 --batch_size 256 --fp16

# Phase 2: seq_len=512, 從 Phase 1 的 checkpoint 繼續，10% steps
python src/pretrain.py \
    --data_dirs data/processed/wikipedia \
    --config_file configs/bert_base.json \
    --resume_from output/bert_base_seq128/checkpoint-900000 \
    --max_seq_length 512 \
    --train_batch_size 64 \
    --max_steps 100000 \
    --learning_rate 1e-4 \
    --fp16 \
    --output_dir output/bert_base_seq512
```

### 超參數參考（BERT-base）

| 參數 | 原論文值 | 本項目默認值 |
|------|---------|-------------|
| Batch size | 256 | 32 (需自行調大或用梯度累積) |
| Learning rate | 1e-4 | 1e-4 |
| Warmup steps | 10,000 | 10% of total steps |
| Max seq length | 128 → 512 | 128 |
| MLM probability | 15% | 15% |
| Weight decay | 0.01 | 0.01 |
| Optimizer | Adam (β1=0.9, β2=0.999) | AdamW (same betas) |
| Total steps | 1,000,000 | 由 epochs 決定 |

### GPU 顯存估算

| 配置 | seq_len=128 | seq_len=512 |
|------|------------|------------|
| BERT-small, bs=64 | ~4 GB | ~8 GB |
| BERT-base, bs=32 | ~12 GB | ~24 GB |
| BERT-base, bs=32, fp16 | ~8 GB | ~16 GB |

> 以上為大致估算，啟用 FP16 可減少約 40% 顯存。

## 常見問題

### Q: Wikipedia 下載很慢怎麼辦？
HuggingFace Datasets 會自動緩存到 `~/.cache/huggingface/`。如果網絡不穩定，可以設置鏡像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/download_data.sh
```

### Q: 如何從斷點恢復訓練？
```bash
python src/pretrain.py \
    --data_dirs data/processed/wikipedia \
    --resume_from output/bert_base_seq128/checkpoint-50000 \
    --output_dir output/bert_base_seq128 \
    ...其他參數保持一致...
```

### Q: 如何用預訓練好的模型做下游任務？
```python
from transformers import BertForSequenceClassification, BertTokenizerFast

# 加載你預訓練好的模型
model = BertForSequenceClassification.from_pretrained(
    "output/bert_base_seq128/final_model",
    num_labels=2,  # 根據下游任務設置
)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
```

### Q: BookCorpus 下載失敗？
BookCorpus 有訪問限制，下載失敗是正常的。僅使用 Wikipedia 即可完成預訓練，效果差異不大。
