"""Evaluation & Threshold Calibration Module

Amazon ML Challenge 2026 — Team SISTem Overload.

Implements:
1. Exact competition Macro F_0.5 evaluation metric:
   F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
   - Macro-averaged across ALL Source 1 entities in the evaluation set.
   - Singletons (true empty matches):
     * Correct empty prediction -> 1.0
     * Any false positive prediction -> 0.0
2. 1D Grid Search Threshold Optimizer over tau in [0.50, 0.85] with step 0.02.
"""

from typing import Dict, Set, List, Tuple
import numpy as np


def compute_entity_f05(ground_truth: Set[str], predicted: Set[str]) -> float:
    """Computes F_0.5 for a single Source 1 entity.

    Weights precision 2x over recall. Handles singletons according to competition rules.
    """
    if not ground_truth and not predicted:
        # Correctly identified singleton
        return 1.0
    if not ground_truth and predicted:
        # False merge on a singleton
        return 0.0
    if ground_truth and not predicted:
        # Missed match on a non-singleton
        return 0.0

    # Both ground truth and predicted are non-empty
    true_positives = len(ground_truth & predicted)
    if true_positives == 0:
        return 0.0

    precision = true_positives / len(predicted)
    recall = true_positives / len(ground_truth)

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0

    return (1.25 * precision * recall) / denom


def evaluate_macro_f05(
    ground_truth_dict: Dict[str, Set[str]],
    predictions_dict: Dict[str, Set[str]],
) -> float:
    """Computes the official competition Macro-averaged F_0.5 score across all S1 entities."""
    total_entities = len(ground_truth_dict)
    if total_entities == 0:
        return 0.0

    total_f05 = 0.0
    for s1_id, gt_set in ground_truth_dict.items():
        pred_set = predictions_dict.get(s1_id, set())
        total_f05 += compute_entity_f05(gt_set, pred_set)

    return total_f05 / total_entities


def find_optimal_threshold(
    s1_ids_list: List[str],
    candidate_ids_list: List[str],
    y_true: np.ndarray,
    y_scores: np.ndarray,
    ground_truth_dict: Dict[str, Set[str]],
    threshold_range: Tuple[float, float] = (0.50, 0.85),
    step: float = 0.02,
) -> Tuple[float, float, Dict[float, float]]:
    """Performs 1D grid search over decision thresholds to directly maximize macro F_0.5.

    Returns:
        best_threshold: The threshold yielding highest macro F_0.5
        best_score: The highest macro F_0.5 score
        threshold_curve: Mapping of threshold -> macro F_0.5
    """
    # Pre-group candidates and scores by S1 ID
    s1_candidates_map: Dict[str, List[Tuple[str, float]]] = {}
    for s1_id, cand_id, score in zip(s1_ids_list, candidate_ids_list, y_scores):
        if s1_id not in s1_candidates_map:
            s1_candidates_map[s1_id] = []
        s1_candidates_map[s1_id].append((cand_id, float(score)))

    best_threshold = 0.65
    best_score = -1.0
    threshold_curve: Dict[float, float] = {}

    thresholds = np.arange(threshold_range[0], threshold_range[1] + step / 2, step)
    for tau in thresholds:
        tau_float = round(float(tau), 4)
        pred_dict: Dict[str, Set[str]] = {}

        for s1_id in ground_truth_dict:
            cands = s1_candidates_map.get(s1_id, [])
            matched = {cid for cid, score in cands if score >= tau_float}
            pred_dict[s1_id] = matched

        score = evaluate_macro_f05(ground_truth_dict, pred_dict)
        threshold_curve[tau_float] = score

        if score > best_score:
            best_score = score
            best_threshold = tau_float

    return best_threshold, best_score, threshold_curve
