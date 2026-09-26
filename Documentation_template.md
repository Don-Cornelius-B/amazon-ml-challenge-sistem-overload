# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** SISTem Overload  
**Team Members:** Team SISTem Overload  
**Submission Date:** September 2026  

---

## 1. Executive Summary

Team **SISTem Overload** developed a high-efficiency, memory-bounded Entity Resolution (ER) pipeline tailored specifically for the Amazon ML Challenge 2026. The solution resolves noisy business records across three independent sources (`S1`, `S2`, `S3`) through a two-stage architecture: country-isolated, word n-gram TF-IDF sparse candidate blocking (bounding candidate sets to an average of $\le 5\text{--}10$ records per entity under $< 3.5$ GB peak RAM), followed by an asymmetric LightGBM pairwise classifier calibrated directly against the competition's macro-averaged $F_{0.5}$ metric ($\tau = 0.7400$). The pipeline is 100% compliant with competition rules—using zero external APIs, zero external databases, lightweight open-source algorithms, and achieving an internal validation macro $F_{0.5}$ score of **0.9564**.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis across the multi-million record challenge corpus revealed four fundamental challenges:
1. **Severe Noise and Multi-Source Inconsistency:**
   - Business names exhibit heavy orthographic variance, typographical errors, abbreviations (`Corp` vs. `Corporation`, `Pvt Ltd` vs. `Private Limited`, `SARL` vs. `S.A.R.L.`), and word-order transpositions.
   - Address strings feature inconsistent component ordering, missing PIN/postal codes, landmark references (`Near Fortis Hospital`), and abbreviations (`Rd`, `St`, `Blvd`, `Apt`).
2. **Domain Generalization to Unseen Geography (`France`):**
   - The training set contains only `US` and `India`, whereas the test set introduces `France`. Models relying on country one-hot categorical splits or Anglo-centric stop words suffer extreme distribution shift.
   - French entities frequently feature Unicode diacritics (`é`, `è`, `ê`, `ç`, `à`) and unique French corporate suffixes (`SARL`, `SAS`, `SA`, `EURL`).
3. **Severe Class Imbalance & Candidate Set Ranking Penalty:**
   - With ~2.2M training S1 records and over 10M pooled target records, brute-force cross-product yields $> 10^{13}$ pairs.
   - Contest rules penalize pipelines generating large candidate pools; efficiency is measured by candidate set reduction without sacrificing recall.
4. **Precision-Weighted Evaluation Metric ($F_{0.5}$):**
   - $F_{0.5}$ penalizes false positive merges twice as heavily as false negative misses.
   - A single false positive match on a true singleton drops that entity's score from $1.0$ directly to $0.0$.

### 2.2 Solution Strategy
We implemented a **Two-Stage Modular Architecture**:
- **Stage 1: Memory-Bounded Country-Partitioned Sparse Blocking (`blocking.py`)**  
  Eliminates cross-country comparisons, encodes business text into compact word $n$-grams (1, 2) with TF-IDF, caps sparse dimensions at 40,000 features (`float32`), and executes chunked dot-product queries with strict similarity thresholding ($\ge 0.25$) and Top-$k$ caps ($k \le 10$).
- **Stage 2: Precision-Calibrated Pairwise Gradient Boosting (`model.py`, `features.py`)**  
  Extracts 13 country-invariant string distance, phonetic, token reordering, and numeric/postal Jaccard features using RapidFuzz. Trains a LightGBM classifier with 4:1 negative-to-positive subsampling, followed by a 1D grid-search threshold calibration ($\tau = 0.7400$) to maximize Macro $F_{0.5}$.

**Approach Type:** Country-Partitioned Sparse TF-IDF Blocking + Precision-Calibrated LightGBM Pairwise Classifier  
**Core Innovation:** High-throughput streaming dot-product blocking with on-the-fly digit Jaccard verification and asymmetric $F_{0.5}$ decision boundary calibration, keeping total host RAM bounded below 3.5 GB.

---

## 3. Candidate Generation (Blocking)

To reduce the search space from trillions of pairs while strictly satisfying the host's 6.15 GB RAM ceiling:

- **Partition Isolation:** Records are strictly partitioned by `country` (`US`, `India`, `France`). Queries never cross international borders.
- **Text Normalization (`preprocess.py`):**
  - Unicode NFKD decomposition strips accents across all European text (`é` $\to$ `e`, `ç` $\to$ `c`).
  - Single-pass compiled regex canonicalizes legal entity forms across commonwealth, American, and French business structures (`pvt ltd`, `inc`, `corp`, `llc`, `sarl`, `sas`, `sa`, `gmbh`).
  - Standardizes address abbreviations (`road`, `street`, `avenue`, `boulevard`, `apartment`, `suite`).
- **Compact Sparse Representation:**
  - Tokenizer: Word $n$-grams (1, 2) on `clean_name + " " + clean_address`.
  - Dimension cap: `max_features=40,000`, `sublinear_tf=True`, `dtype=np.float32`.
  - Stop token pruning: `max_df=0.25`, `min_df=2`.
- **Chunked Matrix Search:**
  - Target S2 + S3 records indexed into a single CSR sparse matrix per country.
  - S1 reference records streamed in batches of 25,000 rows.
  - Dot-product queries executed in sub-chunks of 2,500 rows (`s1_chunk.dot(target_csr.T)`).
  - Candidates extracted directly from NumPy CSR buffer arrays (`indptr`, `indices`, `data`) where cosine similarity $\ge 0.25$, keeping top $k \le 10$ per S1 record.
- **Candidate Set Size & Recall Preservation:**
  - Average candidates generated per S1 entity: $\approx 1.5\text{--}3.5$ (well below the $\le 10$ competition limit).
  - Explicit garbage collection (`del` + `gc.collect()`) releases native C++ buffers after each country partition.

---

## 4. Matching Model

### 4.1 Features Used
For each `(S1, Candidate)` pair produced by blocking, we compute 13 dense, country-invariant similarity features:
1. `name_ratio`: Levenshtein similarity ratio between cleaned names (`rapidfuzz.fuzz.ratio`).
2. `name_token_sort_ratio`: Word-order invariant token sort ratio (`rapidfuzz.fuzz.token_sort_ratio`).
3. `name_token_set_ratio`: Substring/subset token set ratio (`rapidfuzz.fuzz.token_set_ratio`).
4. `name_partial_ratio`: Partial string match ratio (`rapidfuzz.fuzz.partial_ratio`).
5. `name_prefix_match`: Boolean indicator (1.0 if first non-trivial token matches exactly).
6. `name_length_diff`: Normalized difference in character lengths ($|L_1 - L_2| / \max(L_1, L_2)$).
7. `address_token_set_ratio`: Token set similarity on normalized addresses.
8. `address_token_sort_ratio`: Token sort similarity on normalized addresses.
9. `address_numeric_jaccard`: Jaccard similarity of extracted discrete numeric/postal digit tokens ($|D_1 \cap D_2| / |D_1 \cup D_2|$).
10. `address_has_exact_digit_match`: Boolean flag (1.0 if any building/postal number matches).
11. `address_both_have_digits`: Indicator if both records have numeric address components.
12. `tfidf_candidate_score`: Cosine similarity score preserved from the blocking stage.
13. `is_source2`: Indicator whether the candidate originates from Source 2 (1.0) or Source 3 (0.0).

### 4.2 Model Type & Hyperparameters
- **Model:** `lightgbm.LGBMClassifier` (Gradient Boosted Decision Trees).
- **Hyperparameters:**
  - `objective`: `"binary"`
  - `metric`: `"binary_logloss"`
  - `n_estimators`: `300`
  - `learning_rate`: `0.05`
  - `num_leaves`: `31`
  - `random_state`: `42`
- **Training Pair Subsampling:**
  - Retains 100% of ground-truth positive pairs present in the candidate set.
  - Subsamples negative candidate pairs to a fixed 4:1 negative-to-positive ratio per S1 entity to avoid class imbalance distortion.
  - Singletons retained with a controlled negative candidate to preserve "no match" calibration.

### 4.3 Threshold Selection Method
- An entity-stratified 80/20 train/validation split was evaluated directly against the competition Macro $F_{0.5}$ metric:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
- A 1D grid search over $\tau \in [0.50, 0.85]$ in steps of $0.02$ revealed an optimal decision threshold at **$\tau = 0.7400$**.
- This conservative threshold reflects the $2\times$ precision weighting, heavily suppressing false positive merges on both multi-match entities and singletons.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score (Validation Split):** **0.9564** (95.64%)
- **Optimal Decision Threshold ($\tau$):** **0.7400**
- **Feature Importance Ranking:**
  1. `address_token_set_ratio` (Importance: 1482)
  2. `tfidf_candidate_score` (Importance: 1123)
  3. `address_token_sort_ratio` (Importance: 1022)
  4. `name_token_sort_ratio` (Importance: 865)
  5. `name_partial_ratio` (Importance: 793)
  6. `name_ratio` (Importance: 783)
  7. `name_length_diff` (Importance: 779)
  8. `name_token_set_ratio` (Importance: 756)
  9. `address_numeric_jaccard` (Importance: 572)
  10. `is_source2` (Importance: 256)
  11. `name_prefix_match` (Importance: 202)
  12. `address_has_exact_digit_match` (Importance: 191)
  13. `address_both_have_digits` (Importance: 176)

### Error Patterns & Mitigations
- **False Positives (Wrong Merges):**
  - *Pattern:* Franchises or retail chains sharing identical brand names in adjacent localities (e.g., branches in nearby municipal wards).
  - *Mitigation:* Heavy weight on `address_token_set_ratio` and `address_numeric_jaccard` coupled with the elevated threshold ($\tau = 0.70$) prevents cross-branch false merges.
- **False Negatives (Missed Matches):**
  - *Pattern:* Extreme address omissions (e.g., S1 has full street and postal code, but S2 has only state or generic landmark).
  - *Mitigation:* `name_token_set_ratio` and `name_partial_ratio` compensate for missing address components when the business name is highly unique.

---

## 6. Conclusion

The SISTem Overload solution delivers an enterprise-grade, memory-bounded Entity Resolution pipeline achieving **0.9564 Macro $F_{0.5}$** while maintaining peak RAM below 3.5 GB. By pairing country-isolated sparse TF-IDF blocking with asymmetric LightGBM classification and strict threshold calibration, the system generates ultra-compact candidate sets ($\le 5\text{--}10$ average) and guarantees zero formatting violations.

---

## Appendix

### A. Code Artefacts
The pipeline is packaged under `code/business_entity_resolution/` in the submission archive:
- `code/business_entity_resolution/src/config.py`: Centralized configuration, hyperparams, and file paths.
- `code/business_entity_resolution/src/preprocess.py`: Unicode NFKD normalization and legal suffix standardizer.
- `code/business_entity_resolution/src/blocking.py`: Country-partitioned word n-gram TF-IDF blocking.
- `code/business_entity_resolution/src/features.py`: RapidFuzz and digit Jaccard feature extraction.
- `code/business_entity_resolution/src/model.py`: LightGBM training, 4:1 negative subsampling, and checkpointing.
- `code/business_entity_resolution/src/evaluate.py`: Macro $F_{0.5}$ metric and 1D grid search threshold optimizer.
- `code/business_entity_resolution/src/pipeline.py`: End-to-end execution driver.
- `code/business_entity_resolution/requirements.txt`: Pinned dependencies.
- `code/business_entity_resolution/README.md`: Reproduction documentation.

**Reproduction Commands:**
```bash
# 1. Train Model & Optimize Threshold
python code/business_entity_resolution/src/pipeline.py --mode train

# 2. Run Test Inference & Generate Submission TSVs
python code/business_entity_resolution/src/pipeline.py --mode predict

# 3. Format Verification
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

### B. Computational Performance & Compliance Summary
- **External Lookups:** Zero (100% compliant with Fair Play rules).
- **Model Parameters:** LightGBM GBDT with 300 trees ($< 5\text{ MB}$ weights, far below 8B parameter cap).
- **Peak RAM Usage:** $< 3.5$ GB across all partitions.
- **Deliverables:** `output/matching_results.tsv`, `output/candidate_pairs.tsv`, fully passing `validate_submission.py`.
