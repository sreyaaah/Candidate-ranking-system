# Configuration Constants for the Candidate Ranking Pipeline

# Paths
DB_PATH = "data/candidates.db"
JD_PATH = "data/jobs/job_description.docx"
SUBMISSION_PATH = "team_submission.csv"

# Model Configurations
DENSE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Dense Retrieval Parameters
MAX_SEQ_LENGTH = 384
BATCH_SIZE = 16

# Retrieval & Fusion Parameters
RRF_K = 60
TOP_N_HYBRID = 2000

# MMR (Diversity) Parameters
MMR_LAMBDA = 0.6
MMR_TOP_K = 100

# LTR Model Paths
LTR_PAIRWISE_PATH = "models/xgb_ranker_pairwise.json"
LTR_NDCG_PATH = "models/xgb_ranker_ndcg.json"
LTR_LEGACY_PATH = "models/xgb_ranker.json"
