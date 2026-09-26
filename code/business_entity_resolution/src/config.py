"""Central configuration file for the Amazon ML Challenge 2026

Business Entity Resolution Pipeline — Team SISTem Overload.
Centralizes all file paths, system constants, hyperparameters, and thresholds.
"""

from pathlib import Path

# Base directories
SRC_DIR = Path(__file__).resolve().parent
MODULE_DIR = SRC_DIR.parent
CODE_DIR = MODULE_DIR.parent
PROJECT_ROOT = CODE_DIR.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
UTILS_DIR = PROJECT_ROOT / "utils"

# Training Data Paths
TRAIN_S1_PATH = TRAIN_DIR / "train_source1.tsv"
TRAIN_S2_PATH = TRAIN_DIR / "train_source2.tsv"
TRAIN_S3_PATH = TRAIN_DIR / "train_source3.tsv"
TRAIN_GT_PATH = TRAIN_DIR / "train_ground_truth.tsv"

# Test Data Paths
TEST_S1_PATH = TEST_DIR / "test_source1.tsv"
TEST_S2_PATH = TEST_DIR / "test_source2.tsv"
TEST_S3_PATH = TEST_DIR / "test_source3.tsv"

# Output Deliverables
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"

# Saved Model & Artifact Paths
MODEL_PATH = EXPERIMENTS_DIR / "lgbm_model.txt"
THRESHOLD_PATH = EXPERIMENTS_DIR / "optimal_threshold.json"

# Competition & Metric Constants
BETA = 0.5  # F_0.5 weighting (precision 2x over recall)
RANDOM_SEED = 42

# Blocking Hyperparameters (Memory-Bounded for <= 6.15 GB RAM)
CHUNK_SIZE = 25000  # S1 streaming batch size
DOT_SUBCHUNK_SIZE = 2000  # Sub-chunk dot product size to strictly bound sparse matrix allocation < 300 MB
TOP_K_CANDIDATES = 10  # Max candidate matches per S1 entity
TFIDF_MIN_SIM = 0.30  # Strict cosine similarity cutoff
MAX_TFIDF_FEATURES = 40000  # Cap sparse dimension
TFIDF_NGRAM_RANGE = (1, 2)  # Word n-grams (1, 2)
MAX_DF = 0.03  # Prune terms appearing in > 3% of documents to prevent sparse allocation blowup
MIN_DF = 3  # Prune rare noise terms

# Training & Downsampling Parameters
NEGATIVE_TO_POSITIVE_RATIO = 4  # 4:1 negative to positive candidate subsampling
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "n_estimators": 300,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# Default conservative decision threshold for F_0.5 optimization
DEFAULT_THRESHOLD = 0.65
