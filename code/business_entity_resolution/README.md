# Amazon ML Challenge 2026: Business Entity Resolution Pipeline
**Team:** SISTem Overload  
**Target Metric:** Macro-Averaged $F_{0.5}$ (Precision $2\times$ Recall)  
**Deliverables:** `output/candidate_pairs.tsv` and `output/matching_results.tsv`

---

## 1. System Overview

This repository contains the complete, self-contained, and reproducible Entity Resolution pipeline for the Amazon ML Challenge 2026. The solution resolves noisy business records across three heterogeneous data sources (`S1`, `S2`, `S3`) without any external APIs, lookups, or databases.

### Core Innovations & Design Highlights
1. **Memory-Bounded Sparse Blocking (< 3.5 GB RAM):**
   - Independent country-partition isolation (`India`, `US`, `France`).
   - Word n-gram (1, 2) TF-IDF sparse representation (`max_features=40,000`, `dtype=float32`, `sublinear_tf=True`).
   - Chunked sparse matrix dot-product candidate retrieval with strict similarity cutoff ($\ge 0.25$) and Top-$k$ cap ($k \le 10$).
   - Direct NumPy CSR buffer manipulation with explicit garbage collection.
2. **Domain-Invariant & French Generalization:**
   - Unseen test country `France` handled seamlessly through diacritics removal (Unicode NFKD normalization: `é` $\to$ `e`, `ç` $\to$ `c`), French address expansion (`boulevard`, `avenue`, `sarl`, `sa`), and language-agnostic pairwise string/token metrics.
3. **Asymmetric $F_{0.5}$ Model Optimization:**
   - Balanced negative subsampling (100% positive pairs retained, 4:1 negative-to-positive ratio per entity).
   - High-precision LightGBM binary classifier (`n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`).
   - 1D grid search threshold calibration directly maximizing Macro $F_{0.5}$ on an 80/20 entity-stratified holdout split ($\tau \approx 0.65\text{--}0.75$), heavily suppressing false merges on both non-singletons and singletons.
4. **Strict Output Invariants:**
   - Guarantees `matched_entity_ids` $\subseteq$ `candidate_entity_ids`.
   - Exactly one row per test S1 entity, zero duplicates, tab-delimited.

---

## 2. Environment Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.12.10 on Windows / Linux)
- Minimum 8 GB RAM (Peak pipeline memory footprint bounded to $< 4$ GB)

### Installation
From the root of this project or the unpacked archive:

```bash
pip install -r requirements.txt
```

Pinned dependencies include:
- `polars>=0.20.0`
- `pyarrow>=14.0.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `scipy>=1.11.0`
- `scikit-learn>=1.3.0`
- `lightgbm>=4.0.0`
- `rapidfuzz>=3.5.0`
- `tqdm>=4.65.0`

---

## 3. Directory Structure

```
code/business_entity_resolution/
├── requirements.txt         # Pinned production dependencies
├── README.md                # Reproduction documentation
└── src/
    ├── __init__.py
    ├── config.py            # Centralized paths and hyperparameters
    ├── preprocess.py        # NFKD Unicode normalization and text cleaning
    ├── blocking.py          # Memory-bounded TF-IDF candidate generation
    ├── features.py          # RapidFuzz and digit Jaccard feature extraction
    ├── model.py             # LightGBM training and threshold persistence
    ├── evaluate.py          # Exact Macro F0.5 metric and grid search optimizer
    └── pipeline.py          # End-to-end CLI driver (--mode train/predict/all)
```

---

## 4. How to Reproduce End-to-End

Ensure `dataset/` is available at the repository root with `train/` and `test/` TSV files.

### Step 1: Train Model & Calibrate Macro $F_{0.5}$ Threshold
```bash
python -m src.pipeline --mode train
```
*Generates `experiments/lgbm_model.txt` and `experiments/optimal_threshold.json`.*

### Step 2: Run Full Test Inference & Generate Deliverables
```bash
python -m src.pipeline --mode predict
```
*Generates `output/candidate_pairs.tsv` and `output/matching_results.tsv`, and automatically runs `utils/validate_submission.py` to confirm zero formatting errors.*

### Step 3: Run Both Steps in Sequence
```bash
python -m src.pipeline --mode all
```

---

## 5. Verification & Submission Packaging

Run the standalone validation utility:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

Generate the official submission archive:
```powershell
tar -a -c -f "sistem_overload_submission.zip" output code\business_entity_resolution Documentation_template.md
```
