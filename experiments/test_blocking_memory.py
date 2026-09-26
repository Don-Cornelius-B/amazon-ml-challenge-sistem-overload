import gc
import time
import psutil
import os
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

def get_mem_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_blocking_chunk():
    print(f"Initial Memory: {get_mem_mb():.1f} MB")
    
    # Load 50,000 target records (simulating subset of country)
    print("Loading target sample...")
    df_s2 = pl.read_csv("dataset/train/train_source2.tsv", separator="\t", n_rows=50000)
    df_s3 = pl.read_csv("dataset/train/train_source3.tsv", separator="\t", n_rows=50000)
    targets = df_s2["business_name"].fill_null("").to_list() + df_s3["business_name"].fill_null("").to_list()
    target_ids = df_s2["entity_id"].to_list() + df_s3["entity_id"].to_list()
    print(f"Loaded {len(targets)} targets. Memory: {get_mem_mb():.1f} MB")
    
    # Load 5,000 query records
    df_s1 = pl.read_csv("dataset/train/train_source1.tsv", separator="\t", n_rows=5000)
    queries = df_s1["business_name"].fill_null("").to_list()
    query_ids = df_s1["entity_id"].to_list()
    
    # Fit vectorizer
    t0 = time.time()
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=40000,
        sublinear_tf=True,
        dtype=np.float32,
        max_df=0.25,
        min_df=2
    )
    target_csr = vec.fit_transform(targets)
    print(f"Vectorized targets in {time.time() - t0:.2f}s. Target CSR shape: {target_csr.shape}. Memory: {get_mem_mb():.1f} MB")
    
    # Transform queries
    s1_csr = vec.transform(queries)
    print(f"Transformed queries. S1 CSR shape: {s1_csr.shape}. Memory: {get_mem_mb():.1f} MB")
    
    # Matrix multiply in sub-chunks
    SUB_CHUNK = 1000
    top_k = 10
    min_sim = 0.25
    candidates_count = 0
    
    t0 = time.time()
    for start in range(0, s1_csr.shape[0], SUB_CHUNK):
        end = min(start + SUB_CHUNK, s1_csr.shape[0])
        chunk_sim = s1_csr[start:end].dot(target_csr.T)
        
        # Extract top-k per row
        for row_idx in range(chunk_sim.shape[0]):
            row = chunk_sim.getrow(row_idx)
            if row.nnz > 0:
                mask = row.data >= min_sim
                if np.any(mask):
                    valid_cols = row.indices[mask]
                    valid_sims = row.data[mask]
                    if len(valid_sims) > top_k:
                        top_order = np.argpartition(valid_sims, -top_k)[-top_k:]
                        top_order = top_order[np.argsort(-valid_sims[top_order])]
                        candidates_count += len(top_order)
                    else:
                        candidates_count += len(valid_sims)
                        
    print(f"Search completed in {time.time() - t0:.2f}s. Found {candidates_count} candidates. Memory: {get_mem_mb():.1f} MB")
    print("Test passed successfully!")

if __name__ == "__main__":
    test_blocking_chunk()
