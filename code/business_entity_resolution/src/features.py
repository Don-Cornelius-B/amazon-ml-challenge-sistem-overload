"""Pairwise Similarity Feature Extraction Module

Amazon ML Challenge 2026 — Team SISTem Overload.

Features extracted per (S1, Candidate) pair:
1. name_ratio: RapidFuzz Levenshtein ratio on cleaned names
2. name_token_sort_ratio: RapidFuzz token sort ratio (handles word reordering)
3. name_token_set_ratio: RapidFuzz token set ratio (handles substrings/subsets)
4. name_partial_ratio: RapidFuzz partial ratio
5. name_prefix_match: Boolean flag indicating first token exact match
6. name_length_diff: Relative difference in character lengths
7. address_token_set_ratio: RapidFuzz token set ratio on cleaned addresses
8. address_token_sort_ratio: RapidFuzz token sort ratio on cleaned addresses
9. address_numeric_jaccard: Jaccard similarity of discrete digit/postal tokens
10. address_has_exact_digit_match: Indicator if any numeric token matches
11. address_both_have_digits: Indicator if both addresses have numeric tokens
12. tfidf_candidate_score: Cosine similarity score from blocking stage
13. is_source2: 1.0 if candidate is from S2, 0.0 if from S3
"""

from typing import List, Tuple, Set, Dict, Any
import numpy as np
from rapidfuzz import fuzz
from joblib import Parallel, delayed
from .preprocess import extract_numeric_tokens, extract_primary_token

FEATURE_NAMES = [
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_prefix_match",
    "name_length_diff",
    "address_token_set_ratio",
    "address_token_sort_ratio",
    "address_numeric_jaccard",
    "address_has_exact_digit_match",
    "address_both_have_digits",
    "tfidf_candidate_score",
    "is_source2",
]


def compute_pair_features(
    s1_name: str,
    s1_addr: str,
    s1_digits: Set[str],
    cand_id: str,
    cand_name: str,
    cand_addr: str,
    cand_digits: Set[str],
    tfidf_sim: float,
) -> List[float]:
    """Computes country-invariant similarity features for a single (S1, Candidate) pair."""
    # Name similarities
    n_ratio = fuzz.ratio(s1_name, cand_name) / 100.0
    n_sort = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    n_set = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
    n_partial = fuzz.partial_ratio(s1_name, cand_name) / 100.0

    # Prefix match
    s1_prefix = extract_primary_token(s1_name)
    c_prefix = extract_primary_token(cand_name)
    prefix_match = 1.0 if (s1_prefix and s1_prefix == c_prefix) else 0.0

    # Length diff
    max_len = max(len(s1_name), len(cand_name), 1)
    len_diff = abs(len(s1_name) - len(cand_name)) / max_len

    # Address similarities
    a_set = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
    a_sort = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0

    # Numeric / PIN / street digit overlap
    both_digits = 1.0 if (s1_digits and cand_digits) else 0.0
    if s1_digits or cand_digits:
        intersection = len(s1_digits & cand_digits)
        union = len(s1_digits | cand_digits)
        num_jaccard = float(intersection / union) if union > 0 else 0.0
        has_digit_match = 1.0 if intersection > 0 else 0.0
    else:
        num_jaccard = 1.0
        has_digit_match = 0.0

    is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0

    return [
        n_ratio,
        n_sort,
        n_set,
        n_partial,
        prefix_match,
        len_diff,
        a_set,
        a_sort,
        num_jaccard,
        has_digit_match,
        both_digits,
        float(tfidf_sim),
        is_s2,
    ]


def extract_features_for_candidates(
    s1_id: str,
    s1_name: str,
    s1_addr: str,
    s1_digits: Set[str],
    candidates: List[Tuple[str, float, int]],
    target_names: List[str],
    target_addrs: List[str],
) -> Tuple[List[str], np.ndarray]:
    """Extracts features for all candidates belonging to a single S1 record.

    Candidate digits are extracted on-the-fly to eliminate upfront memory overhead.
    """
    if not candidates:
        return [], np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    cand_ids = []
    feat_matrix = []

    for cand_id, sim, target_idx in candidates:
        cand_name = target_names[target_idx]
        cand_addr = target_addrs[target_idx]
        cand_digits = extract_numeric_tokens(cand_addr)

        feats = compute_pair_features(
            s1_name=s1_name,
            s1_addr=s1_addr,
            s1_digits=s1_digits,
            cand_id=cand_id,
            cand_name=cand_name,
            cand_addr=cand_addr,
            cand_digits=cand_digits,
            tfidf_sim=sim,
        )
        cand_ids.append(cand_id)
        feat_matrix.append(feats)

    return cand_ids, np.array(feat_matrix, dtype=np.float32)


def _process_pair_tuple(p):
    return compute_pair_features(
        s1_name=p[0],
        s1_addr=p[1],
        s1_digits=p[2],
        cand_id=p[3],
        cand_name=p[4],
        cand_addr=p[5],
        cand_digits=p[6],
        tfidf_sim=p[7],
    )

def extract_features_batch(
    pairs: List[Tuple[str, str, Set[str], str, str, str, Set[str], float]],
    n_jobs: int = 4,
) -> np.ndarray:
    """Batch compute features for a list of (S1, Cand) pairs using joblib.Parallel with thread backend.

    Using prefer="threads" bypasses Windows process serialization and pickling locks,
    leveraging RapidFuzz's GIL-releasing C++ implementation for zero-overhead multi-core execution.
    """
    if not pairs:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    feats = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=2000)(
        delayed(_process_pair_tuple)(p) for p in pairs
    )
    return np.array(feats, dtype=np.float32)
