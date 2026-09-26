"""Model Training and Prediction Module

Amazon ML Challenge 2026 — Team SISTem Overload.

Features:
- Balanced negative subsampling (100% true positives retained, 4:1 negative-to-positive ratio)
- LightGBM binary classifier (n_estimators=300, lr=0.05, num_leaves=31)
- Model persistence to experiments/ directory
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import lightgbm as lgb
from .config import LGBM_PARAMS, MODEL_PATH, THRESHOLD_PATH, DEFAULT_THRESHOLD, NEGATIVE_TO_POSITIVE_RATIO
from .features import FEATURE_NAMES

logger = logging.getLogger(__name__)


def subsample_training_pairs(
    candidate_records: List[Dict],
    ground_truth: Dict[str, Set[str]],
    negative_ratio: int = NEGATIVE_TO_POSITIVE_RATIO,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Subsamples candidate pairs to achieve a fixed negative-to-positive ratio.

    1. Retains 100% of ground-truth positive pairs present in the candidate set.
    2. Subsamples negative candidates up to `negative_ratio * len(positives)` per entity.
    3. For entities with 0 true positives, samples up to 1 negative candidate to maintain singleton awareness.

    Returns:
        X: Feature matrix of shape (N_samples, N_features)
        y: Binary labels (0 or 1)
        s1_ids: Corresponding S1 IDs
        cand_ids: Corresponding Candidate IDs
    """
    rng = np.random.RandomState(random_state)
    logger.info(f"Subsampling candidate pairs with {negative_ratio}:1 negative ratio...")

    # Group candidate records by s1_id
    grouped: Dict[str, List[Dict]] = {}
    for rec in candidate_records:
        sid = rec["s1_id"]
        if sid not in grouped:
            grouped[sid] = []
        grouped[sid].append(rec)

    selected_records = []
    for sid, recs in grouped.items():
        true_matches = ground_truth.get(sid, set())
        pos_recs = [r for r in recs if r["cand_id"] in true_matches]
        neg_recs = [r for r in recs if r["cand_id"] not in true_matches]

        # Always keep all true positive candidates
        selected_records.extend(pos_recs)

        if pos_recs:
            max_negs = len(pos_recs) * negative_ratio
            if len(neg_recs) > max_negs:
                neg_indices = rng.choice(len(neg_recs), size=max_negs, replace=False)
                selected_records.extend([neg_recs[i] for i in neg_indices])
            else:
                selected_records.extend(neg_recs)
        else:
            # Singleton record with only negative candidates: keep at most 1 negative
            if neg_recs:
                selected_records.append(rng.choice(neg_recs))

    logger.info(f"Selected {len(selected_records):,} pairs from {len(candidate_records):,} total pairs.")
    X = np.array([r["features"] for r in selected_records], dtype=np.float32)
    y = np.array([1 if r["cand_id"] in ground_truth.get(r["s1_id"], set()) else 0 for r in selected_records], dtype=np.int32)
    s1_ids = [r["s1_id"] for r in selected_records]
    cand_ids = [r["cand_id"] for r in selected_records]

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    logger.info(f"Class balance: Positives={n_pos:,} ({n_pos/len(y)*100:.1f}%), Negatives={n_neg:,} ({n_neg/len(y)*100:.1f}%)")

    return X, y, s1_ids, cand_ids


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
) -> lgb.LGBMClassifier:
    """Trains a LightGBM binary classifier on feature matrix."""
    train_params = dict(LGBM_PARAMS)
    if params:
        train_params.update(params)

    model = lgb.LGBMClassifier(**train_params)
    logger.info(f"Training LightGBM on {len(X_train):,} pairs with {len(FEATURE_NAMES)} features...")

    eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
    model.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        feature_name=FEATURE_NAMES,
    )
    logger.info("LightGBM training completed.")
    return model


def save_artifacts(model: lgb.LGBMClassifier, optimal_threshold: float, model_path: Path = MODEL_PATH, threshold_path: Path = THRESHOLD_PATH):
    """Saves the trained model and calibrated decision threshold."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")

    threshold_data = {
        "optimal_threshold": float(optimal_threshold),
        "feature_names": FEATURE_NAMES,
    }
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump(threshold_data, f, indent=2)
    logger.info(f"Threshold metadata saved to {threshold_path}")


def load_artifacts(model_path: Path = MODEL_PATH, threshold_path: Path = THRESHOLD_PATH) -> Tuple[lgb.Booster, float]:
    """Loads the trained model booster and calibrated decision threshold."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Run training first.")
    
    booster = lgb.Booster(model_file=str(model_path))
    threshold = DEFAULT_THRESHOLD

    if threshold_path.exists():
        with open(threshold_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            threshold = float(data.get("optimal_threshold", DEFAULT_THRESHOLD))
            logger.info(f"Loaded calibrated threshold: {threshold:.4f}")
    else:
        logger.warning(f"Threshold file not found at {threshold_path}. Using default: {threshold}")

    return booster, threshold
