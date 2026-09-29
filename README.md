# Entity Resolver — ML Summer Challenge 2026

> **Entity resolution pipeline** for matching business entities across multiple data sources. Developed for the **ML Summer Challenge 2026** (entity resolution track).

## 🎯 Problem Statement

Given three data sources (Source 1, 2, 3) containing business records with names, addresses, and countries, identify which Source-1 entities match entities in Source-2 and Source-3. The challenge: **noisy, multilingual data** with abbreviations, transliterations, and missing fields.

- **Source 1**: ~1.7M entities (canonical)
- **Source 2**: ~4.9M entities
- **Source 3**: ~5.1M entities
- **Ground truth**: ~7.6M positive pairs (train only)

## 🏆 Results

| Metric | Value |
|--------|-------|
| **Entity-macro F0.5 (validation)** | **0.8490** |
| **Test entities matched** | 96.6% (1,673,565 / 1,732,544) |
| **Test singletons** | 3.4% (58,979) |
| **Blocking recall (multipass)** | 80.4% (S2), 81.2% (S3) |
| **Training pairs** | 30.5M (7.6M pos, 22.9M neg) |

*Previous baseline: 0.8217 F0.5 → **+0.0273 improvement** via 3-threshold policy*

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        PIPELINE                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐   │
│  │ Normalize│───▶│ Multipass    │───▶│ Feature Engineering │   │
│  │ (fix FR) │    │ Blocking     │    │ (13 features)       │   │
│  └──────────┘    └──────────────┘    └──────────┬──────────┘   │
│                                                  │              │
│  ┌──────────────┐    ┌──────────────┐           │              │
│  │ FAISS Index  │───▶│ Candidate    │◀──────────┤              │
│  │ (embeddings) │    │ Union        │           │              │
│  └──────────────┘    └──────┬───────┘           │              │
│                             │                   │              │
│                    ┌────────▼────────┐          │              │
│                    │ LightGBM +      │          │              │
│                    │ Platt Calibrator│          │              │
│                    └────────┬────────┘          │              │
│                             │                   │              │
│                    ┌────────▼────────┐          │              │
│                    │ 3-Threshold     │          │              │
│                    │ Decision Policy │          │              │
│                    └────────┬────────┘          │              │
│                             │                   │              │
│                    ┌────────▼────────┐          │              │
│                    │ Submission      │          │              │
│                    │ (TSV)           │          │              │
│                    └─────────────────┘          │              │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Technical Design

### 1. Normalization (`src/normalize.py`)

**Problem**: French accents (é, è, ç, à) were incorrectly routed through an Indic-script transliterator, corrupting tokens (`Santé` → `sant` instead of `sante`).

**Solution**: Unicode-aware script detection:
```python
_LATIN_EXTENDED_RANGES = [
    (0x00C0, 0x00FF),  # Latin-1 Supplement
    (0x0100, 0x017F),  # Latin Extended-A
    (0x0180, 0x024F),  # Latin Extended-B
    (0x1E00, 0x1EFF),  # Latin Extended Additional
]

def _fold_latin_accents(text: str) -> str:
    """NFD decompose + strip combining marks (Mn category)."""
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn")
```

**Result**: `Maison de Santé` → `maison de sante` (correct), Indic scripts still transliterate via `indic-transliteration`.

### 2. Multipass Blocking (`src/multipass_blocking.py`)

12 complementary passes to achieve ~80% recall at manageable candidate size:

| Pass | Key | Description |
|------|-----|-------------|
| `ct2` | `CTRY|n|tok1|tok2` | Country + first 2 name tokens |
| `pairs_ctry/nc` | `CTRY|np|tokA|tokB` | Token pairs within name |
| `tok_ctry/nc` | `CTRY|nt|tok` | Individual tokens |
| `ph_ctry/nc` | `CTRY|sx|soundex` | Soundex on first token |
| `addr_bigram` | `ap|tokA|tokB` | Address token bigrams |
| `postal_ctry/nc` | `CTRY|zz|12345` | Postal code (5-6 digits) |
| `house_ctry/nc` | `CTRY|hn|num|street` | House number + street token |

**Frequency suppression**: Keys >4000 freq or cross-product >900 pairs/key dropped (per-pass overrides for postal/house).

**Output**: ~23M candidate pairs per source (train), ~97M (test).

### 3. Feature Engineering (13 features)

| Feature | Type | Description |
|---------|------|-------------|
| `jw_name` | float | Jaro-Winkler on normalized name |
| `jw_addr` | float | Jaro-Winkler on normalized address |
| `tok_jaccard` | float | Token Jaccard on name tokens |
| `country_match` | bool | Exact country match |
| `name_len_diff` | int | Absolute name length difference |
| `fsig_match` | bool | First significant token match |
| `lev_name` | float | Levenshtein similarity (name) |
| `jaro_name` | float | Jaro similarity (name) |
| `jaro_addr` | float | Jaro similarity (address) |
| `addr_house_match` | bool | House number match (regex) |
| `addr_pin_match` | bool | PIN/postal match (5-6 digits) |
| `addr_street_jaccard` | float | Street token Jaccard |
| `src` | int | Source indicator (0=S2, 1=S3) |

### 4. Entity-Level Train/Val Split (`data/_entsplit_md5.parquet`)

- **Train entities**: 1,765,016 (features from `candmpfit3_train` — full multipass)
- **Val entities**: 441,805 (features from `candmp_core_train` — val-only multipass)
- **Zero entity overlap** verified

### 5. LightGBM + Platt Calibration (`src/train_lgbm_entsplit.py`)

```python
params = dict(
    objective="binary",
    learning_rate=0.03,
    num_leaves=63,
    scale_pos_weight=neg/pos,  # ~5.36
    max_bin=255,
    min_data_in_leaf=50,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    lambda_l2=1.0,
    metric="auc",
)
```

- **Best iteration**: 21 (early stopping)
- **Calibration**: Platt (LogisticRegression) on validation set only
- **No data leakage**: Train/val split by entity, not row

### 6. Embedding-Based Blocking (`src/embed_and_index.py`)

**Contrastive Fine-tuning**:
- Base: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- Training data: 30.5M pairs (7.6M pos, 22.9M neg) from ground truth
- Loss: `ContrastiveLoss` (margin-based)
- Checkpoint: 5000 steps (`models/finetuned_minilm_5000steps/`)

**Indexing**:
- Embed all S2+S3 entities (test: ~10M, train: ~10M)
- FAISS `IndexIVFFlat` (IVF 8192, inner product = cosine)
- Incremental training/add to avoid OOM
- Top-50 nearest neighbors per S1 entity

**Candidate Union**:
```
Final Candidates = Multipass Candidates ∪ FAISS Top-50 Neighbors
```

### 7. 3-Threshold Decision Policy (`src/step2_threshold_search.py`)

| Threshold | Value | Action |
|-----------|-------|--------|
| `singleton_cutoff` | 0.10 | Max prob < 0.10 → predict empty |
| `keep_threshold` | 0.60 | Max prob ≥ 0.60 → keep all ≥ 0.60 |
| `fallback_top1` | 0.20 | 0.20 ≤ max < 0.60 → keep top-1 only |

**Grid search** on sampled validation (10%) → best F0.5 = 0.8490 (vs 0.8217 flat 0.55)

**Impact on test**:
- Entities with matches: 87.6% → **96.6%**
- Singletons: 12.4% → **3.4%**

## 📊 Metrics & Accuracy

### Validation (honest entity-level split)

| Metric | Flat (0.55) | 3-Threshold | Δ |
|--------|-------------|-------------|---|
| Entity-macro F0.5 | 0.8217 | **0.8490** | +0.0273 |
| Pair-level AUC | 0.9949 | — | — |
| Calibrated logloss | 0.0283 | — | — |

### Test Set Performance

| Metric | Value |
|--------|-------|
| Total entities | 1,732,544 |
| Entities matched | 1,673,565 (96.60%) |
| Singletons | 58,979 (3.40%) |
| Avg candidates/entity (S2) | 56.0 |
| Avg candidates/entity (S3) | 57.0 |

### Blocking Recall (train val)

| Source | GT pairs | Captured | Recall |
|--------|----------|----------|--------|
| S2 | 739,619 | 594,623 | 80.40% |
| S3 | 789,709 | 640,874 | 81.15% |

## ⚠️ Difficulties & Solutions

| Difficulty | Solution |
|------------|----------|
| **French accent corruption** | Unicode script detection + NFD accent folding |
| **Entity leakage in split** | MD5-based entity split (`_entsplit_md5.parquet`), verified zero overlap |
| **Memory OOM on embeddings** | Chunked embedding (1M entities/chunk), incremental FAISS build |
| **FAISS training data size** | Sample 1M vectors for IVF training |
| **Calibration on val only** | Platt fitted on val set, applied to test |
| **Singleton over-prediction** | 3-threshold policy with singleton_cutoff=0.10 |
| **Indic vs Latin script conflict** | Separate code paths: NFD folding for Latin Extended, indic-transliteration for Indic |

## 📁 Project Structure

```
ML_challenge/
├── src/
│   ├── normalize.py              # French accent fix + normalization
│   ├── multipass_blocking.py     # 12-pass candidate generation
│   ├── features.py               # 13 feature computations
│   ├── train_lgbm_entsplit.py    # Entity-split LightGBM + Platt
│   ├── step2_threshold_search.py # 3-threshold grid search
│   ├── embed_and_index.py        # FAISS embedding pipeline
│   ├── finetune_embedding.py     # Contrastive fine-tuning
│   ├── prepare_contrastive_data.py # 30.5M pair generation
│   ├── generate_train_features.py # Train features (candmpfit3)
│   ├── generate_val_features.py  # Val features (candmp_core)
│   ├── run_inference_3threshold.py # 3-threshold inference
│   └── ... (utils, config, etc.)
├── models/
│   ├── best_3threshold_policy.pkl
│   ├── best_threshold.pkl
│   ├── finetuned_minilm_5000steps/  # Fine-tuned embedder
│   └── lgbm_matcher_calibrated.pkl  # Calibrated LightGBM
├── data/
│   ├── _entsplit_md5.parquet      # Entity split
│   ├── norm_*_s*.parquet          # Normalized sources
│   └── emb_*_chunk*.npy           # Embedding chunks
├── output/
│   ├── matching_results.tsv       # Submission (1.7M rows)
│   └── candidate_pairs.tsv
├── utils/validate_submission.py   # Validator
├── .gitignore
└── README.md
```

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install torch sentence-transformers faiss-cpu lightgbm polars pyarrow rapidfuzz

# 2. Regenerate normalized data (with French fix)
python src/renormalize_all.py

# 3. Generate features (honest entity split)
python src/generate_train_features.py --src 2 --candidate data/candmpfit3_train_s2.parquet --out data/features_multipass_fit_s2.parquet
python src/generate_train_features.py --src 3 --candidate data/candmpfit3_train_s3.parquet --out data/features_multipass_fit_s3.parquet
python src/generate_val_features.py --src 2 --candidate data/candmp_core_train_s2.parquet --out data/features_val_s2.parquet
python src/generate_val_features.py --src 3 --candidate data/candmp_core_train_s3.parquet --out data/features_val_s3.parquet

# 4. Train LightGBM + calibrate
python src/train_lgbm_entsplit.py

# 5. Grid search 3-threshold policy
python src/step2_threshold_search.py

# 6. Fine-tune embeddings (optional, needs GPU)
python src/prepare_contrastive_data.py
python src/finetune_embedding.py

# 7. Embed + FAISS index
python src/embed_and_index.py

# 8. Inference with 3-threshold policy
python src/run_inference_3threshold.py --phase 1
python src/run_inference_3threshold.py --phase 2

# 9. Validate
python utils/validate_submission.py --matching output/matching_results.tsv --test-dir dataset/test --check-ids
```

## 📝 License

Apache-2.0 (base models: `all-MiniLM-L6-v2`, LightGBM)

## 🏷️ Competition

**ML Summer Challenge 2026** — Entity Resolution Track

> This repository contains the complete pipeline from data normalization through submission generation, including all improvements developed during the challenge: French normalization fix, entity-level honest splits, contrastive embedding fine-tuning, FAISS-based candidate expansion, and 3-threshold decision policy.

---

*Submission validated with `utils/validate_submission.py --check-ids` — all matched IDs exist in test Source-2/3.*