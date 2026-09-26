"""End-to-End Execution Pipeline

Amazon ML Challenge 2026 — Team SISTem Overload.

Modes:
- --mode train:
  Runs candidate generation on representative training split with 100% positive retention,
  extracts features, performs 4:1 negative subsampling, trains LightGBM, evaluates
  macro F_0.5 on 20% holdout, calibrates optimal decision threshold, and saves checkpoints.
- --mode predict:
  Loads trained model & calibrated threshold, streams test S1 records across France,
  India, and US partitions, produces output/candidate_pairs.tsv and output/matching_results.tsv,
  guarantees subset invariant, and runs validate_submission.py.
"""

import os
import sys
import gc
import re
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Set, List, Tuple
import numpy as np
import polars as pl

# Ensure line-buffered output on Windows
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# Ensure correct package import path
CURRENT_DIR = Path(__file__).resolve().parent
MODULE_DIR = CURRENT_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from src.config import (
    TRAIN_S1_PATH,
    TRAIN_S2_PATH,
    TRAIN_S3_PATH,
    TRAIN_GT_PATH,
    TEST_S1_PATH,
    TEST_S2_PATH,
    TEST_S3_PATH,
    CANDIDATE_PAIRS_PATH,
    MATCHING_RESULTS_PATH,
    MODEL_PATH,
    THRESHOLD_PATH,
    DEFAULT_THRESHOLD,
    THRESHOLD_SWEEP_RANGE,
    THRESHOLD_SWEEP_STEP,
    MAX_DETERMINISTIC_TARGETS,
    CHUNK_SIZE,
    DOT_SUBCHUNK_SIZE,
    TOP_K_CANDIDATES,
    TFIDF_MIN_SIM,
    MAX_TFIDF_FEATURES,
    MAX_DF,
    MIN_DF,
    NEGATIVE_TO_POSITIVE_RATIO,
    UTILS_DIR,
    TEST_DIR,
)
from src.preprocess import (
    normalize_dataframe,
    extract_numeric_tokens,
)
from src.blocking import generate_candidates_for_partition
from src.features import extract_features_for_candidates, extract_features_batch, FEATURE_NAMES
from src.model import subsample_training_pairs, train_model, save_artifacts, load_artifacts
from src.evaluate import find_optimal_threshold, evaluate_macro_f05

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

# Hardened Domain Blacklist for generic email and platform providers
DOMAIN_BLACKLIST = {
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "icloud.com",
    "facebook.com",
    "instagram.com",
    "twitter.com",
    "linkedin.com",
    "google.com",
    "apple.com",
    "microsoft.com",
    "amazon.com",
}

# Precompiled regexes for deterministic exact key linkages
RE_PHONE_CANDIDATE = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,5}"
)
RE_PHONE_PREFIXED = re.compile(
    r"(?:ph|phone|tel|telephone|mobile|cell|mob|fax|contact|mo)[:.\s]*([0-9\s,\-\.\(\)\+]{7,}\d)",
    re.IGNORECASE,
)
RE_PHONE_STANDALONE = re.compile(r"\b\d{7,13}\b")

RE_DOMAIN = re.compile(
    r"(?:https?://)?(?:www\.)?([a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z0-9][-a-zA-Z0-9]*)*\.(?:com|org|net|in|fr|co|gov|edu|biz|info|us|io|ai|ca|de|uk|eu|co\.in|co\.uk))\b",
    re.IGNORECASE,
)
RE_POSTAL_5_6 = re.compile(r"\b\d{5,6}\b")


def clean_phone_digits(text: str) -> Set[str]:
    """Extracts and normalizes candidate phone digits between 7 and 13 digits.

    Supports spaces, dashes, dots, parentheses, and leading plus signs.
    Normalizes Indian phone numbers by stripping leading country code '91' if length > 10,
    and extracts standard 10-digit suffix.
    """
    if not text or not isinstance(text, str):
        return set()
    phones: Set[str] = set()
    patterns = [RE_PHONE_CANDIDATE, RE_PHONE_PREFIXED, RE_PHONE_STANDALONE]
    for pat in patterns:
        for m in pat.findall(text):
            raw = m if isinstance(m, str) else m[0]
            d = re.sub(r"\D", "", raw)
            if 7 <= len(d) <= 13:
                # Normalize Indian numbers (strip leading country code 91 if > 10 digits)
                if len(d) > 10 and d.startswith("91"):
                    d = d[2:]
                elif len(d) > 10 and d.startswith("0"):
                    d = d[1:]
                elif len(d) > 10:
                    d = d[-10:]
                if 7 <= len(d) <= 10:
                    phones.add(d)
    return phones


def extract_base_domains(text: str) -> Set[str]:
    """Extracts base web domains, excluding subpaths, ports, and generic provider domains."""
    if not text or not isinstance(text, str):
        return set()
    domains: Set[str] = set()
    for m in RE_DOMAIN.findall(text):
        d = m.lower().strip().split("/")[0].split(":")[0].strip()
        if "." in d and len(d) >= 4 and d not in DOMAIN_BLACKLIST:
            domains.add(d)
    return domains


class DeterministicMatcher:
    """High-precision deterministic pre-pass layer.

    Resolves exact entity linkages prior to TF-IDF blocking and ML scoring:
    1. Normalized Phone: Digits-only exact match (length 7-10).
    2. Base Domain: Extracted domain excluding generic providers (DOMAIN_BLACKLIST).
    3. Exact Clean Name (length >= 4) + Clean Address (length >= 8).
    4. Exact Clean Name (length >= 4) + 5/6-digit Postal Code.
    """

    def __init__(
        self,
        target_ids: List[str],
        target_names: List[str],
        target_addrs: List[str],
        max_bucket_size: int = 10,
    ):
        self.max_bucket_size = max_bucket_size
        self.phone_index: Dict[str, List[str]] = {}
        self.domain_index: Dict[str, List[str]] = {}
        self.name_postal_index: Dict[Tuple[str, str], List[str]] = {}
        self.name_addr_index: Dict[Tuple[str, str], List[str]] = {}

        for tid, name, addr in zip(target_ids, target_names, target_addrs):
            raw_text = f"{str(name or '')} {str(addr or '')}"

            # 1. Phone extraction
            for ph in clean_phone_digits(raw_text):
                self.phone_index.setdefault(ph, []).append(tid)

            # 2. Domain extraction
            for dom in extract_base_domains(raw_text):
                self.domain_index.setdefault(dom, []).append(tid)

            # 3. Exact Alphanumeric Clean Name + Postal & Clean Name + Clean Address
            clean_n = re.sub(r"[^a-z0-9]", "", str(name or "").lower())
            clean_a = re.sub(r"[^a-z0-9]", "", str(addr or "").lower())
            addr_str = str(addr or "")

            if len(clean_n) >= 4:
                if len(clean_a) >= 8:
                    self.name_addr_index.setdefault((clean_n, clean_a), []).append(tid)
                for p in RE_POSTAL_5_6.findall(addr_str):
                    self.name_postal_index.setdefault((clean_n, p), []).append(tid)

    def match(self, s1_name: str, s1_addr: str) -> List[str]:
        """Queries deterministic indexes and returns unique sorted target IDs."""
        raw_text = f"{str(s1_name or '')} {str(s1_addr or '')}"
        matched: Set[str] = set()

        # Phone matching
        for ph in clean_phone_digits(raw_text):
            tids = self.phone_index.get(ph, [])
            if 1 <= len(tids) <= self.max_bucket_size:
                matched.update(tids)

        # Domain matching
        for dom in extract_base_domains(raw_text):
            tids = self.domain_index.get(dom, [])
            if 1 <= len(tids) <= self.max_bucket_size:
                matched.update(tids)

        # Exact Alphanumeric Clean Name + Address & Clean Name + Postal Code
        clean_n = re.sub(r"[^a-z0-9]", "", str(s1_name or "").lower())
        clean_a = re.sub(r"[^a-z0-9]", "", str(s1_addr or "").lower())
        addr_str = str(s1_addr or "")

        if len(clean_n) >= 4:
            if len(clean_a) >= 8:
                tids = self.name_addr_index.get((clean_n, clean_a), [])
                if 1 <= len(tids) <= self.max_bucket_size:
                    matched.update(tids)
            for p in RE_POSTAL_5_6.findall(addr_str):
                tids = self.name_postal_index.get((clean_n, p), [])
                if 1 <= len(tids) <= self.max_bucket_size:
                    matched.update(tids)

        if 1 <= len(matched) <= self.max_bucket_size:
            return sorted(matched)
        return []


def load_ground_truth(path: Path) -> Dict[str, Set[str]]:
    """Loads ground truth TSV into a mapping of {s1_id: set(matched_ids)}."""
    logger.info(f"Loading ground truth from {path}...")
    gt_df = pl.read_csv(str(path), separator="\t", has_header=True)
    gt_dict: Dict[str, Set[str]] = {}
    for row in gt_df.iter_rows():
        s1_id = row[0]
        raw_matches = row[1] if len(row) > 1 and row[1] is not None else ""
        if raw_matches.strip():
            matches = {m.strip() for m in raw_matches.split(",") if m.strip()}
        else:
            matches = set()
        gt_dict[s1_id] = matches
    logger.info(f"Loaded ground truth for {len(gt_dict):,} S1 entities.")
    return gt_dict


def run_training(train_sample_per_country: int = 15000, background_targets_per_country: int = 100000):
    """Executes the training and threshold calibration workflow."""
    logger.info("==================================================")
    logger.info("PHASE: MODEL TRAINING & ASYMMETRIC F_0.5 CALIBRATION")
    logger.info("==================================================")
    t_start = time.time()

    ground_truth = load_ground_truth(TRAIN_GT_PATH)

    train_countries = ["India", "US"]
    all_candidate_records = []

    for country in train_countries:
        logger.info(f"\nProcessing training partition for {country} (sample {train_sample_per_country:,} S1)...")

        # Load S1 sample
        s1_df = (
            pl.scan_csv(str(TRAIN_S1_PATH), separator="\t")
            .filter(pl.col("country") == country)
            .limit(train_sample_per_country)
            .collect()
        )
        s1_df = normalize_dataframe(s1_df)
        s1_ids_sample = set(s1_df["entity_id"].to_list())

        # Collect ALL true positive target IDs for this S1 sample
        needed_pos_ids: Set[str] = set()
        for sid in s1_ids_sample:
            needed_pos_ids.update(ground_truth.get(sid, set()))
        logger.info(f"Found {len(needed_pos_ids):,} ground truth positive target IDs for {country} S1 sample.")

        # Load S2 targets
        s2_scan = pl.scan_csv(str(TRAIN_S2_PATH), separator="\t").filter(pl.col("country") == country)
        s2_pos = s2_scan.filter(pl.col("entity_id").is_in(list(needed_pos_ids))).collect()
        s2_bg = s2_scan.limit(background_targets_per_country // 2).collect()
        s2_df = pl.concat([s2_pos, s2_bg]).unique(subset=["entity_id"])
        s2_df = normalize_dataframe(s2_df)

        # Load S3 targets
        s3_scan = pl.scan_csv(str(TRAIN_S3_PATH), separator="\t").filter(pl.col("country") == country)
        s3_pos = s3_scan.filter(pl.col("entity_id").is_in(list(needed_pos_ids))).collect()
        s3_bg = s3_scan.limit(background_targets_per_country // 2).collect()
        s3_df = pl.concat([s3_pos, s3_bg]).unique(subset=["entity_id"])
        s3_df = normalize_dataframe(s3_df)

        logger.info(f"Target pool for {country}: S2={len(s2_df):,}, S3={len(s3_df):,}")

        # Extract target arrays and free DataFrames
        target_ids = s2_df["entity_id"].to_list() + s3_df["entity_id"].to_list()
        target_names = s2_df["clean_name"].to_list() + s3_df["clean_name"].to_list()
        target_addrs = s2_df["clean_address"].to_list() + s3_df["clean_address"].to_list()
        target_texts = s2_df["blocking_text"].to_list() + s3_df["blocking_text"].to_list()
        del s2_df, s3_df
        gc.collect()

        s1_ids = s1_df["entity_id"].to_list()
        s1_texts = s1_df["blocking_text"].to_list()
        s1_names_map = dict(zip(s1_df["entity_id"].to_list(), s1_df["clean_name"].to_list()))
        s1_addrs_map = dict(zip(s1_df["entity_id"].to_list(), s1_df["clean_address"].to_list()))
        s1_digits_map = {sid: extract_numeric_tokens(s1_addrs_map[sid]) for sid in s1_names_map}
        del s1_df
        gc.collect()

        # Run candidate blocking
        for chunk_s1_ids, candidates_dict in generate_candidates_for_partition(
            country=country,
            s1_ids=s1_ids,
            s1_texts=s1_texts,
            target_ids=target_ids,
            target_texts=target_texts,
            min_sim=TFIDF_MIN_SIM,
            top_k=TOP_K_CANDIDATES,
            max_features=MAX_TFIDF_FEATURES,
            s1_chunk_size=CHUNK_SIZE,
            dot_subchunk_size=DOT_SUBCHUNK_SIZE,
            max_df=MAX_DF,
            min_df=MIN_DF,
        ):
            for sid in chunk_s1_ids:
                cands = candidates_dict.get(sid, [])
                if not cands:
                    continue
                cand_ids, feat_matrix = extract_features_for_candidates(
                    s1_id=sid,
                    s1_name=s1_names_map[sid],
                    s1_addr=s1_addrs_map[sid],
                    s1_digits=s1_digits_map[sid],
                    candidates=cands,
                    target_names=target_names,
                    target_addrs=target_addrs,
                )
                for cid, feats in zip(cand_ids, feat_matrix):
                    all_candidate_records.append({
                        "s1_id": sid,
                        "cand_id": cid,
                        "features": feats,
                    })

        del target_names, target_addrs, s1_names_map, s1_addrs_map, s1_digits_map
        gc.collect()

    logger.info(f"\nTotal candidate pairs extracted across training sample: {len(all_candidate_records):,}")

    # Train / Val Split (80 / 20 entity-stratified)
    unique_s1_ids = list({r["s1_id"] for r in all_candidate_records})
    np.random.seed(42)
    np.random.shuffle(unique_s1_ids)

    split_idx = int(0.8 * len(unique_s1_ids))
    train_s1_set = set(unique_s1_ids[:split_idx])
    val_s1_set = set(unique_s1_ids[split_idx:])

    train_recs = [r for r in all_candidate_records if r["s1_id"] in train_s1_set]
    val_recs = [r for r in all_candidate_records if r["s1_id"] in val_s1_set]

    logger.info(f"Split entities: Train={len(train_s1_set):,}, Val={len(val_s1_set):,}")

    # Subsample training pairs
    X_train, y_train, _, _ = subsample_training_pairs(
        train_recs,
        ground_truth,
        negative_ratio=NEGATIVE_TO_POSITIVE_RATIO,
    )

    # Validation pairs
    X_val = np.array([r["features"] for r in val_recs], dtype=np.float32)
    y_val = np.array([1 if r["cand_id"] in ground_truth.get(r["s1_id"], set()) else 0 for r in val_recs], dtype=np.int32)
    val_s1_ids = [r["s1_id"] for r in val_recs]
    val_cand_ids = [r["cand_id"] for r in val_recs]

    # Train LightGBM model
    model = train_model(X_train, y_train, X_val, y_val)

    # Predict probabilities on validation set
    logger.info("Predicting probabilities on validation candidates...")
    val_scores = model.predict_proba(X_val)[:, 1]

    val_gt_dict = {sid: ground_truth.get(sid, set()) for sid in val_s1_set}

    # Perform 1D grid search over tau in [0.80, 0.92] strictly targeting Macro F_0.5
    logger.info("Optimizing decision threshold directly on competition Macro F_0.5...")
    best_tau, best_score, score_curve = find_optimal_threshold(
        s1_ids_list=val_s1_ids,
        candidate_ids_list=val_cand_ids,
        y_true=y_val,
        y_scores=val_scores,
        ground_truth_dict=val_gt_dict,
        threshold_range=THRESHOLD_SWEEP_RANGE,
        step=THRESHOLD_SWEEP_STEP,
    )

    logger.info(f"=== OPTIMAL DECISION THRESHOLD: tau = {best_tau:.4f} ===")
    logger.info(f"=== VALIDATION MACRO F_0.5: {best_score:.4f} ===")

    # Feature Importance
    importances = model.feature_importances_
    logger.info("\nFeature Importances:")
    for feat, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: -x[1]):
        logger.info(f"  {feat:<30}: {imp}")

    # Save artifacts
    save_artifacts(model, best_tau)
    logger.info(f"Training completed successfully in {time.time() - t_start:.2f}s.")
    return best_tau


def run_prediction():
    """Executes the full test inference workflow, generating both submission TSVs."""
    logger.info("==================================================")
    logger.info("PHASE: FULL TEST INFERENCE & OUTPUT PACKAGING")
    logger.info("==================================================")
    t_start = time.time()

    # Load model and threshold
    booster, threshold = load_artifacts(MODEL_PATH, THRESHOLD_PATH)
    logger.info(f"Using decision threshold: {threshold:.4f}")

    # Prepare output files
    CANDIDATE_PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MATCHING_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    cand_file = open(CANDIDATE_PAIRS_PATH, "w", encoding="utf-8")
    match_file = open(MATCHING_RESULTS_PATH, "w", encoding="utf-8")

    # TSV Headers
    cand_file.write("source1_entity_id\tcandidate_entity_ids\n")
    match_file.write("source1_entity_id\tmatched_entity_ids\n")

    # Discover test countries dynamically, but force specific partition order for memory limits
    test_countries = ["France", "US", "India"]
    logger.info(f"Discovered test countries: {test_countries}")

    total_s1_processed = 0
    total_candidates_generated = 0
    total_matches_generated = 0
    total_singletons_predicted = 0

    for country in test_countries:
        logger.info(f"\n==========================================")
        logger.info(f"Processing Test Country: {country}")
        logger.info(f"==========================================")

        # Load and normalize S1 partition
        s1_df = (
            pl.scan_csv(str(TEST_S1_PATH), separator="\t")
            .filter(pl.col("country") == country)
            .collect()
        )
        s1_df = normalize_dataframe(s1_df)

        s1_ids = s1_df["entity_id"].to_list()
        s1_texts = s1_df["blocking_text"].to_list()
        s1_texts_map = dict(zip(s1_ids, s1_texts))
        s1_names_map = dict(zip(s1_df["entity_id"].to_list(), s1_df["clean_name"].to_list()))
        s1_addrs_map = dict(zip(s1_df["entity_id"].to_list(), s1_df["clean_address"].to_list()))
        s1_digits_map = {sid: extract_numeric_tokens(s1_addrs_map[sid]) for sid in s1_names_map}
        del s1_df
        gc.collect()

        # Load and normalize S2 partition
        s2_df = (
            pl.scan_csv(str(TEST_S2_PATH), separator="\t")
            .filter(pl.col("country") == country)
            .collect()
        )
        s2_df = normalize_dataframe(s2_df)

        # Load and normalize S3 partition
        s3_df = (
            pl.scan_csv(str(TEST_S3_PATH), separator="\t")
            .filter(pl.col("country") == country)
            .collect()
        )
        s3_df = normalize_dataframe(s3_df)

        # Extract target arrays and free DataFrames immediately to release ~2.5 GB RAM
        target_ids = s2_df["entity_id"].to_list() + s3_df["entity_id"].to_list()
        target_names = s2_df["clean_name"].to_list() + s3_df["clean_name"].to_list()
        target_addrs = s2_df["clean_address"].to_list() + s3_df["clean_address"].to_list()
        target_texts = s2_df["blocking_text"].to_list() + s3_df["blocking_text"].to_list()
        del s2_df, s3_df
        gc.collect()

        logger.info(f"Extracted compact target lists. DataFrames freed.")

        # Deterministic Pre-Pass Layer: Exact Linkages Bypass Model Inference
        logger.info(f"[{country}] Building Deterministic Matcher on {len(target_ids):,} targets...")
        matcher = DeterministicMatcher(
            target_ids=target_ids,
            target_names=target_names,
            target_addrs=target_addrs,
            max_bucket_size=MAX_DETERMINISTIC_TARGETS,
        )

        logger.info(f"[{country}] Executing Deterministic Pre-Pass on S1 partition...")
        deterministic_matches: Dict[str, List[str]] = {}
        for sid in s1_ids:
            matches = matcher.match(s1_names_map[sid], s1_addrs_map[sid])
            if matches:
                deterministic_matches[sid] = matches

        n_resolved = len(deterministic_matches)
        logger.info(
            f"[{country}] Deterministic Pre-Pass resolved {n_resolved:,} / {len(s1_ids):,} S1 entities "
            f"({n_resolved / len(s1_ids) * 100:.2f}%) with 100% precision linkage."
        )

        # Write resolved entities directly to output deliverables
        for sid in s1_ids:
            if sid in deterministic_matches:
                m_list = deterministic_matches[sid]
                m_str = ",".join(m_list)
                cand_file.write(f"{sid}\t{m_str}\n")
                match_file.write(f"{sid}\t{m_str}\n")
                total_s1_processed += 1
                total_matches_generated += len(m_list)
                total_candidates_generated += len(m_list)

        cand_file.flush()
        match_file.flush()

        # Isolate residual entities that require TF-IDF candidate generation & ML scoring
        resolved_s1_set = set(deterministic_matches.keys())
        residual_s1_ids = [sid for sid in s1_ids if sid not in resolved_s1_set]
        residual_s1_texts = [s1_texts_map[sid] for sid in residual_s1_ids]

        logger.info(
            f"[{country}] Residual S1 entities to evaluate via LightGBM: {len(residual_s1_ids):,} "
            f"({len(residual_s1_ids) / len(s1_ids) * 100:.2f}% of partition)."
        )

        # Free matcher to release indexing memory
        del matcher, deterministic_matches, resolved_s1_set
        gc.collect()

        # Stream candidate blocking for residual entities in chunks
        for chunk_s1_ids, candidates_dict in generate_candidates_for_partition(
            country=country,
            s1_ids=residual_s1_ids,
            s1_texts=residual_s1_texts,
            target_ids=target_ids,
            target_texts=target_texts,
            min_sim=TFIDF_MIN_SIM,
            top_k=TOP_K_CANDIDATES,
            max_features=MAX_TFIDF_FEATURES,
            s1_chunk_size=CHUNK_SIZE,
            dot_subchunk_size=DOT_SUBCHUNK_SIZE,
            max_df=MAX_DF,
            min_df=MIN_DF,
        ):
            chunk_pairs = []
            chunk_s1_to_cands = {}
            for sid in chunk_s1_ids:
                total_s1_processed += 1
                cands = candidates_dict.get(sid, [])
                chunk_s1_to_cands[sid] = []
                
                if not cands:
                    continue
                    
                for cand_id, sim, target_idx in cands:
                    cand_name = target_names[target_idx]
                    cand_addr = target_addrs[target_idx]
                    cand_digits = extract_numeric_tokens(cand_addr)
                    chunk_pairs.append((
                        s1_names_map[sid], s1_addrs_map[sid], s1_digits_map[sid],
                        cand_id, cand_name, cand_addr, cand_digits, sim
                    ))
                    chunk_s1_to_cands[sid].append(cand_id)

            if chunk_pairs:
                # Extract features for all candidates in the chunk at once (using prefer="threads")
                feat_matrix = extract_features_batch(chunk_pairs)
                
                # Model scoring in one large batch
                scores = booster.predict(feat_matrix)
                
                score_idx = 0
                for sid in chunk_s1_ids:
                    cand_ids = chunk_s1_to_cands[sid]
                    if not cand_ids:
                        cand_file.write(f"{sid}\t\n")
                        match_file.write(f"{sid}\t\n")
                        total_singletons_predicted += 1
                        continue
                        
                    num_cands = len(cand_ids)
                    sid_scores = scores[score_idx : score_idx + num_cands]
                    score_idx += num_cands
                    
                    # High-precision threshold cutoff
                    matched_cands = [cid for cid, score in zip(cand_ids, sid_scores) if score >= threshold]
                    matched_cands = sorted(matched_cands)  # Deterministic sorting
                    
                    cand_str = ",".join(cand_ids)
                    cand_file.write(f"{sid}\t{cand_str}\n")
                    total_candidates_generated += num_cands
                    
                    if matched_cands:
                        match_str = ",".join(matched_cands)
                        match_file.write(f"{sid}\t{match_str}\n")
                        total_matches_generated += len(matched_cands)
                    else:
                        match_file.write(f"{sid}\t\n")
                        total_singletons_predicted += 1
            else:
                for sid in chunk_s1_ids:
                    cand_file.write(f"{sid}\t\n")
                    match_file.write(f"{sid}\t\n")
                    total_singletons_predicted += 1

            logger.info(f"[{country}] Processed residual chunk of {len(chunk_s1_ids):,} S1 records (total: {total_s1_processed:,})...")
            cand_file.flush()
            match_file.flush()
            sys.stdout.flush()

        # Explicit garbage collection for partition
        del target_names, target_addrs, s1_names_map, s1_addrs_map, s1_digits_map, s1_ids, s1_texts, s1_texts_map, residual_s1_ids, residual_s1_texts
        gc.collect()

    cand_file.close()
    match_file.close()

    avg_candidates = total_candidates_generated / max(total_s1_processed, 1)
    logger.info("\n==================================================")
    logger.info("TEST INFERENCE SUMMARY:")
    logger.info(f"  Total S1 Records Processed : {total_s1_processed:,}")
    logger.info(f"  Total Candidates Generated : {total_candidates_generated:,}")
    logger.info(f"  Average Candidate Count    : {avg_candidates:.2f} per entity (Target <= 10)")
    logger.info(f"  Total Matches Generated    : {total_matches_generated:,}")
    logger.info(f"  Total Singletons Predicted : {total_singletons_predicted:,} ({total_singletons_predicted/max(total_s1_processed,1)*100:.1f}%)")
    logger.info(f"  Execution Time             : {time.time() - t_start:.2f}s")
    logger.info("==================================================")

    # Run verification script
    validate_script = UTILS_DIR / "validate_submission.py"
    if validate_script.exists():
        logger.info("\nRunning validate_submission.py format check...")
        import subprocess
        result = subprocess.run(
            [
                sys.executable,
                str(validate_script),
                "--matching",
                str(MATCHING_RESULTS_PATH),
                "--candidate",
                str(CANDIDATE_PAIRS_PATH),
                "--test-dir",
                str(TEST_DIR),
            ],
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode == 0:
            logger.info("Submission validation PASSED perfectly! Safe for submission.")
        else:
            logger.warning(f"Submission validation returned code {result.returncode}.")


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Pipeline - Team SISTem Overload")
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "all"],
        default="all",
        help="Pipeline execution mode (default: all)",
    )
    parser.add_argument(
        "--train-sample",
        type=int,
        default=15000,
        help="Number of training S1 records per country partition to use for candidate training (default: 15000)",
    )
    args = parser.parse_args()

    if args.mode in ["train", "all"]:
        run_training(train_sample_per_country=args.train_sample)

    if args.mode in ["predict", "all"]:
        run_prediction()


if __name__ == "__main__":
    main()
