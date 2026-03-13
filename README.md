# BERT Pre-training from Scratch

完整的 BERT 預訓練實驗框架，遵循原始論文 (Devlin et al., 2019)。

- **預訓練任務**: Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)
- **數據集**: English Wikipedia（可選 BookCorpus）
- **框架**: PyTorch + HuggingFace Transformers
- **分佈式**: 支持單卡 / 多卡（via Accelerate）
- **應用演示**: Gradio Web UI（Fill-Mask / 語義相似度 / 關鍵詞提取）

## 項目結構

```
bert_pretrain/
├── configs/
│   ├── bert_base.json            # BERT-base (110M params)
│   ├── bert_medium.json          # BERT-medium (40M params, 適合 8GB 顯卡)
│   └── bert_small.json           # BERT-small (14M params, 用於調試)
├── scripts/
│   ├── setup.sh                  # 環境安裝
│   ├── download_data.sh          # 數據下載與預處理
│   ├── train.sh                  # 單卡訓練（通用）
│   ├── train_rtx3070.sh          # RTX 3070 一鍵訓練（含自動下載數據）
│   ├── train_distributed.sh      # 多卡分佈式訓練
│   ├── evaluate.sh               # 模型評估
│   └── run_app.sh                # 啟動 Gradio 演示應用
├── src/
│   ├── dataset.py                # MLM + NSP 數據集實現
│   ├── preprocess.py             # Wikipedia/BookCorpus 預處理（支持 streaming）
│   ├── pretrain.py               # 單卡訓練腳本
│   ├── pretrain_accelerate.py    # 分佈式訓練腳本 (Accelerate)
│   ├── evaluate.py               # 評估腳本 (MLM PPL + NSP Acc)
│   └── app.py                    # Gradio 演示應用
├── requirements.txt
└── README.md
```

## 快速開始

### 0. 環境準備

```bash
cd bert_pretrain
bash scripts/setup.sh
```

### 1. 數據下載與預處理

```bash
# 下載完整 Wikipedia（約需 20-40 分鐘，取決於網速）
bash scripts/download_data.sh

# 僅下載少量數據用於調試
bash scripts/download_data.sh --max_articles 10000

# 同時下載 Wikipedia + BookCorpus
bash scripts/download_data.sh --dataset all
```

數據會被處理成文檔級別的 pickle 文件，存放在 `data/processed/` 下。
設置 `--max_articles` 時自動使用 streaming 模式，節省磁碟空間。

### 2. 訓練

#### RTX 3070 (8GB) 一鍵訓練（推薦）

```bash
# 一鍵完成：自動下載數據 + 訓練（BERT-Medium, 10 epochs）
bash scripts/train_rtx3070.sh

# 自定義 epochs
bash scripts/train_rtx3070.sh --epochs 15

# 從 checkpoint 恢復
bash scripts/train_rtx3070.sh --resume_from output/bert_medium/checkpoint-step4000
```

#### 單卡通用訓練

```bash
# BERT-small 快速調試
bash scripts/train.sh --config bert_small --seq_len 128 --batch_size 64 --fp16

# BERT-base 標準訓練
bash scripts/train.sh --config bert_base --seq_len 128 --batch_size 32 --epochs 3 --fp16
```

#### 多卡分佈式訓練

```bash
accelerate config
bash scripts/train_distributed.sh --num_gpus 4 --config bert_base --batch_size 32 --fp16
```

### 3. 評估

```bash
bash scripts/evaluate.sh --model_path output/bert_medium/checkpoint-final
```

輸出指標：
- **MLM Perplexity**: 越低越好，BERT-base 在 Wiki 上通常 ~4-8
- **NSP Accuracy**: 通常 >95%

### 4. 演示應用

```bash
bash scripts/run_app.sh
# 或指定 checkpoint
bash scripts/run_app.sh --model_path output/bert_medium/checkpoint-epoch5
```

應用功能：
- **Fill-Mask**: 預測被遮蔽的詞
- **Semantic Similarity**: 計算兩個句子的語義相似度
- **Keyword Extraction**: 基於 MLM surprise score 的關鍵詞提取

## 模型配置對比

| 配置 | 參數量 | 層數 | Hidden | Heads | 推薦 GPU |
|------|--------|------|--------|-------|----------|
| bert_small | ~14M | 4 | 256 | 4 | 任意（調試用） |
| bert_medium | ~40M | 8 | 512 | 8 | 8GB（RTX 3070/4060） |
| bert_base | ~110M | 12 | 768 | 12 | 16GB+（V100/A100） |

## GPU 顯存估算

| 配置 | seq_len=128 | seq_len=512 |
|------|------------|------------|
| BERT-small, bs=64 | ~4 GB | ~8 GB |
| BERT-medium, bs=32, fp16 | ~4-5 GB | ~10 GB |
| BERT-base, bs=32, fp16 | ~8 GB | ~16 GB |

## 訓練策略說明

### 原始 BERT 論文的兩階段訓練

1. **Phase 1**: `max_seq_length=128`，訓練 90% 的步數
2. **Phase 2**: `max_seq_length=512`，訓練剩餘 10% 的步數

```bash
# Phase 1
bash scripts/train.sh --config bert_base --seq_len 128 --max_steps 900000 --batch_size 256 --fp16

# Phase 2
python src/pretrain.py \
    --data_dirs data/processed/wikipedia \
    --config_file configs/bert_base.json \
    --resume_from output/bert_base_seq128/checkpoint-900000 \
    --max_seq_length 512 \
    --train_batch_size 64 \
    --max_steps 100000 \
    --fp16 \
    --output_dir output/bert_base_seq512
```

## 常見問題

### Q: Wikipedia 下載很慢怎麼辦？
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

model = BertForSequenceClassification.from_pretrained(
    "output/bert_medium/checkpoint-final",
    num_labels=2,
)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
```

### Q: BookCorpus 下載失敗？
BookCorpus 有訪問限制，下載失敗是正常的。僅使用 Wikipedia 即可完成預訓練。
