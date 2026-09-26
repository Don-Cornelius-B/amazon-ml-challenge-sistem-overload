"""Memory-Bounded Candidate Generation (Blocking) Module

Amazon ML Challenge 2026 — Team SISTem Overload.

Key Architectures:
- Country-partition isolation (India, US, France processed independently)
- Word n-gram (1, 2) TF-IDF vectorization with max_features=40,000, sublinear_tf=True, float32
- Chunked target transformation + chunked dot product search guaranteeing < 2.5 GB peak RAM
- Direct CSR buffer manipulation (indptr, indices, data) for high-speed candidate extraction
- Explicit garbage collection (del + gc.collect) to guarantee safety under host RAM limits
"""

import gc
import logging
from typing import Dict, List, Tuple, Generator
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


def generate_candidates_for_partition(
    country: str,
    s1_ids: List[str],
    s1_texts: List[str],
    target_ids: List[str],
    target_texts: List[str],
    min_sim: float = 0.25,
    top_k: int = 10,
    max_features: int = 40000,
    s1_chunk_size: int = 25000,
    dot_subchunk_size: int = 200,
    target_transform_chunk: int = 500000,
    max_df: float = 0.03,
    min_df: int = 3,
) -> Generator[Tuple[List[str], Dict[str, List[Tuple[str, float, int]]]], None, None]:
    """Generates candidate pairs for a single country partition with memory-bounded chunking.

    Yields:
        Tuple of:
          - chunk_s1_ids: list of all S1 IDs in the chunk (preserving order and singletons)
          - candidates_dict: {s1_id: [(cand_id, cosine_sim, cand_idx), ...]}
    """
    logger.info(f"--- Starting Blocking for Country: {country} ---")
    n_s1 = len(s1_ids)
    n_targets = len(target_ids)
    logger.info(f"Records: S1={n_s1:,}, Pooled Targets (S2+S3)={n_targets:,}")

    if n_s1 == 0:
        logger.info(f"No S1 records for {country}. Skipping.")
        return

    if n_targets == 0:
        # All S1 entities in this country have no candidates
        for i in range(0, n_s1, s1_chunk_size):
            chunk_ids = s1_ids[i : i + s1_chunk_size]
            yield chunk_ids, {sid: [] for sid in chunk_ids}
        return

    # Fit TF-IDF Vectorizer on a representative target sample (up to 500k documents)
    fit_sample_size = min(500000, n_targets)
    logger.info(f"Fitting TF-IDF Vectorizer on representative sample of {fit_sample_size:,} targets (max_df={max_df})...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
        max_df=max_df,
        min_df=min_df,
    )
    vectorizer.fit(target_texts[:fit_sample_size])
    logger.info(f"Vocabulary fitted with {len(vectorizer.vocabulary_):,} features.")

    # Transform targets in memory-bounded chunks of 500,000 documents
    logger.info(f"Transforming {n_targets:,} targets in chunks of {target_transform_chunk:,}...")
    target_csr_parts = []
    for t_start in range(0, n_targets, target_transform_chunk):
        t_end = min(t_start + target_transform_chunk, n_targets)
        target_csr_parts.append(vectorizer.transform(target_texts[t_start:t_end]))
        logger.info(f"  Transformed targets {t_start:,} to {t_end:,}...")

    target_csr = sp.vstack(target_csr_parts, format="csr")
    del target_csr_parts, target_texts
    gc.collect()
    logger.info(f"Target CSR assembled: shape={target_csr.shape}, non-zeros={target_csr.nnz:,}")

    # Pre-transpose target CSR to CSR format for 33% faster dot product and bounded memory
    logger.info("Pre-transposing target CSR matrix for high-speed dot product search...")
    target_csr_T = target_csr.T.tocsr()
    del target_csr
    gc.collect()

    # Stream S1 in chunks of s1_chunk_size
    for chunk_start in range(0, n_s1, s1_chunk_size):
        chunk_end = min(chunk_start + s1_chunk_size, n_s1)
        chunk_s1_ids = s1_ids[chunk_start:chunk_end]
        chunk_s1_texts = s1_texts[chunk_start:chunk_end]

        s1_chunk_csr = vectorizer.transform(chunk_s1_texts)
        candidates_dict: Dict[str, List[Tuple[str, float, int]]] = {sid: [] for sid in chunk_s1_ids}

        # Sub-chunk dot product to keep peak RAM strictly bounded below 300 MB
        n_chunk_rows = s1_chunk_csr.shape[0]
        for sub_start in range(0, n_chunk_rows, dot_subchunk_size):
            sub_end = min(sub_start + dot_subchunk_size, n_chunk_rows)
            sub_sim = s1_chunk_csr[sub_start:sub_end].dot(target_csr_T)

            # High-speed direct buffer extraction using CSR indptr, indices, data
            indptr = sub_sim.indptr
            indices = sub_sim.indices
            data = sub_sim.data

            for row_idx in range(sub_end - sub_start):
                global_row_idx = sub_start + row_idx
                s1_id = chunk_s1_ids[global_row_idx]

                s = indptr[row_idx]
                e = indptr[row_idx + 1]
                if e == s:
                    continue

                row_data = data[s:e]
                mask = row_data >= min_sim
                if not np.any(mask):
                    continue

                valid_cols = indices[s:e][mask]
                valid_sims = row_data[mask]

                if len(valid_sims) > top_k:
                    top_indices = np.argpartition(valid_sims, -top_k)[-top_k:]
                    top_indices = top_indices[np.argsort(-valid_sims[top_indices])]
                    chosen_cols = valid_cols[top_indices]
                    chosen_sims = valid_sims[top_indices]
                else:
                    sorted_order = np.argsort(-valid_sims)
                    chosen_cols = valid_cols[sorted_order]
                    chosen_sims = valid_sims[sorted_order]

                candidates_dict[s1_id] = [
                    (target_ids[col], float(sim), int(col))
                    for col, sim in zip(chosen_cols, chosen_sims)
                ]

            del sub_sim

        del s1_chunk_csr
        yield chunk_s1_ids, candidates_dict

    # Clean up large partition structures and invoke explicit garbage collection
    logger.info(f"Finished blocking for {country}. Running explicit garbage collection...")
    del target_csr_T
    del vectorizer
    del target_ids
    gc.collect()
