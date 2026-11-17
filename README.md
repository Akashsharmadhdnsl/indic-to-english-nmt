# Indic-to-English NMT (Neural Machine Translation)

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)
![CTranslate2](https://img.shields.io/badge/Inference-CTranslate2-orange)
![Kaggle](https://img.shields.io/badge/Kaggle-Assessment-20BEFF)

Fine-tuned Meta's `facebook/m2m100_418M` with **LoRA (Low-Rank Adaptation)** to translate **Hindi, Bengali, Tamil, Marathi, Gujarati, Malayalam and Urdu into English**, with confidence-based review flagging and fast CTranslate2 INT8 inference.

## TL;DR

| Metric | Result |
|---|---|
| Macro chrF++ (character n-gram F-score) | **40.79 → 45.70** vs zero-shot |
| Macro BLEU (Bilingual Evaluation Understudy) | **16.26 → 20.44** vs zero-shot |
| Trainable parameters (LoRA) | 4.7M of 488.6M (**0.97%**), adapter size 18.9 MB |
| GPU latency (batch 32) | **74.0 → 26.7 ms/sentence** (CTranslate2 `int8_float16`) |
| Model size | **974 MB → 491 MB** |

## Task and Data

- **Task:** Translate 7 Indic languages into English.
- **Training data:** 6,800 parallel pairs (Hindi 1,500 · Bengali 1,300 · Tamil 1,100 · Marathi 900 · Gujarati 800 · Malayalam 700 · Urdu 500).
- **Test data:** 3,563 rows (509 per language, 7 languages).
- **Data audit:** no nulls, no duplicate pairs, no exact train/test source overlap.
- **Note:** The competition data is not included in this repo. Download it from the Kaggle competition page.

## Approach

1. **Normalization:** Unicode NFC, removal of zero-width/directional marks, whitespace collapse.
2. **Language mapping:** dataset codes mapped to M2M100 tokenizer IDs (`hi, bn, ta, mr, gu, ml, ur`); decoder forced to the `en` token.
3. **Validation:** 10% `GroupShuffleSplit` grouped by English reference (`target_en`), giving 6,103 train / 697 validation rows with zero reference leakage. Metrics are computed per language, then macro-averaged.
4. **Baseline:** zero-shot M2M100, beam search with 4 beams.
5. **Full fine-tuning:** 4 epochs, learning rate `5e-5`, effective batch size 32, label smoothing 0.1, FP16, gradient checkpointing.
6. **LoRA:** rank `r=16`, alpha 32, dropout 0.05 on `q_proj`, `k_proj`, `v_proj`, `out_proj`; learning rate `5e-4`, 6 epochs, about 15 minutes of training. Adapter merged into the base model for inference.

## Results

Macro averages over 7 languages on the validation split:

| System | Decoding | Macro BLEU | Macro chrF++ |
|---|---|---:|---:|
| Zero-shot M2M100 | Beam 4 | 16.26 | 40.79 |
| Full fine-tuning | Beam 4 | 19.13 | 44.81 |
| Full fine-tuning | Beam 5, no-repeat trigram | 19.89 | 45.51 |
| **LoRA (r=16)** | Beam 5, `no_repeat_ngram_size=3` | **20.44** | **45.70** |

### Per-language chrF++ (selected LoRA model)

| Language | chrF++ |
|---|---:|
| Hindi | 57.02 |
| Bengali | 51.69 |
| Marathi | 49.17 |
| Urdu | 45.22 |
| Malayalam | 44.04 |
| Gujarati | 37.27 |
| Tamil | 35.51 |

Tamil and Gujarati are the weakest languages. Error inspection showed semantic substitutions, lost named entities and misread long clauses.

### Overfitting observation

Full fine-tuning training loss fell from 2.91 to 1.90 across 4 epochs, while validation loss rose from 3.25 to 3.37 after epoch 1. This is why model selection was based on validation metrics instead of assuming more epochs help.

<!-- Optional: add assets/per_language_chrf.png and assets/loss_curve.png here -->

## Translation Confidence and Review Routing

The generated sequence log-probability is used as a confidence proxy.

- Spearman rank correlation with sentence-level chrF++: **0.683** (sample of up to 400 validation rows).
- Mean chrF++ rises from **33.25** (lowest-confidence quartile) to **58.17** (highest).
- Sequences scoring below **-0.655** (25th percentile) are flagged for human review.

This is a ranking heuristic, not a calibrated probability, and should be revalidated on production traffic.

## Inference Optimization

| Runtime | Batch-32 (ms/sent) | Batch-1 (ms/sent) | Val chrF++ | Size |
|---|---:|---:|---:|---:|
| PyTorch FP16, GPU | 74.0 | 581.1 | 44.89 | 974 MB |
| CTranslate2 `int8_float16`, GPU | 26.7 | 200.9 | 45.19 | 491 MB |
| CTranslate2 INT8, CPU | n/a | 2,595.4 | 45.69* | 491 MB |

\*CPU chrF++ was measured on a 35-sentence subset versus 210 sentences for GPU, so the numbers are not strictly comparable. Latencies come from the notebook environment and are not production SLAs (Service Level Agreements).

### Proposed serving design (not deployed)

Validate request → normalize → batch by language and length → CTranslate2 inference (GPU, CPU INT8 fallback) → return translation, language, model version and review flag. Intended stack: FastAPI + Docker.

## Limitations

- Small, language-imbalanced dataset (6,800 pairs).
- Small validation set, especially for Urdu and Malayalam, so small score differences may be noise.
- Hindi and Marathi share the Devanagari script, so script-only language detection cannot tell them apart; the source language must be given explicitly.
- Confidence score is not calibrated.

## Repository Structure

```
indic-to-english-nmt/
├── notebooks/
│   └── indic_to_english_nmt.ipynb   # full pipeline
├── docs/
│   └── technical_report.md          # detailed report
├── assets/                          # plots
├── requirements.txt
└── README.md
```

## How to Run

1. Open the notebook in Google Colab (GPU runtime) or Kaggle.
2. Add the competition `train.parquet` and `test.parquet` to the working directory.
3. Run all cells: preprocessing → grouped split → baseline → full fine-tune → LoRA → evaluation → CTranslate2 benchmark → submission.

```bash
pip install -r requirements.txt
```

## Tech Stack

Python · PyTorch · Hugging Face Transformers · PEFT (Parameter-Efficient Fine-Tuning) · CTranslate2 · sacreBLEU · scikit-learn

## Author

**Akash Sharma**, Data Scientist · [Portfolio](https://akash-myportfolio.netlify.app)
