# 🎓 AI-Powered Candidate Ranking System: Model Training Guide

This guide provides a comprehensive, step-by-step walkthrough to train the **Learning-to-Rank (LTR) Ensemble** model for the Candidate Ranking System. The training pipeline utilizes **Knowledge Distillation** (using a Neural Cross-Encoder teacher to label data) and fits a dual-objective **XGBoost Ranker** student on 46 engineered features.

---

## 🛠️ 1. Environment & Prerequisites

### Hardware Requirements
- **macOS (Apple Silicon):** Fully supported with GPU acceleration via PyTorch MPS (Metal Performance Shaders).
- **Windows / Linux:** CPU fallback is automated, featuring INT8 model quantization for CPU-based BGE query embedding generation to prevent OOM errors (e.g., Windows Error 1455).

### Setup Dependencies
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

Verify that the following configurations are set correctly in `src/config.py`:
- `DB_PATH = "data/candidates.db"`
- `JD_PATH = "data/jobs/job_description.docx"`
- `DENSE_MODEL_NAME = "BAAI/bge-base-en-v1.5"`
- `CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"`

---

## 🚀 2. Step-by-Step Training Pipeline

The training process follows a strict sequence to build the SQLite database, embed candidates, generate features, distill training labels, and fit the LTR student models.

### Step 1: Rebuild the Database
Extract raw candidate profiles from `data/candidates.jsonl` into a clean SQLite schema. This step normalizes qualifications, majors, and career history, and generates natural-language skill sentences.
```bash
python src/build_db.py
```
- **Output:** SQLite DB initialized at `data/candidates.db`.

### Step 2: Generate Dense Section Embeddings
To enable **Late-Fusion Multi-Vector Retrieval**, candidates are split into 4 core sections: **Summary, Skills, Experience, and Education**. This script encodes these sections using the local BGE model.
```bash
python src/embed_candidates_jsonl.py
```
- **Output:** Multi-vector embeddings stored in the `embeddings/` directory.

### Step 3: Embed the Job Description (JD)
Apply the TF-IDF Section Classifier to paragraph-pool and encode the Job Description document to create the dense search vector:
```bash
python src/embed_jd.py
```

### Step 4: Build Search Indices
Build the indexes necessary for candidate retrieval:
1. **FAISS Vector Index** for dense search over section embeddings:
   ```bash
   python src/build_faiss.py
   ```
2. **BM25 Sparse Lexical Index** for exact keyword and token matches:
   ```bash
   python src/build_bm25.py
   ```

### Step 5: Feature Extraction & Engineering
Compute and store **46 advanced tabular features** for every candidate profile. These cover promotion velocity, tenure stability, AI exposure, open-source involvement, academic tiering, and honeypot flags.
```bash
python src/build_features.py
```
- **Output:** Populates the `candidate_features` table in `data/candidates.db`.

---

## 🧠 3. Model Training & Distillation

### Step 6: Generate Distilled Training Data
Since candidate ranking lacks explicit relevance labels, we use **Knowledge Distillation** from a high-capacity **Neural Cross-Encoder Teacher** (`ms-marco-MiniLM-L-6-v2`).

To ensure robustness, the system constructs **4 distinct training groups (Queries)**:
1. **Main JD** (Senior ML Engineer)
2. **DevOps Engineer** (Synthetic profile)
3. **Data Engineer** (Synthetic profile)
4. **Full Stack Developer** (Synthetic profile)

For each query group, the top 1,500 candidates are retrieved using hybrid search (RRF). The Cross-Encoder scores the candidates relative to the query. These continuous teacher scores are binned into discrete relevance levels `[0, 1, 2, 3]` based on within-group quantiles (95th percentile = 3, 80th = 2, 50th = 1, rest = 0).
```bash
python src/generate_training_data.py
```
- **Output:** Distilled dataset saved to `data/training_data.csv`.

> [!NOTE]
> Candidates marked as **honeypots** (e.g., spam profiles or prompt-injected profiles) are automatically forced to a relevance score of `0` in this script to train the model to penalize adversarial candidates.

### Step 7: Train the XGBoost LTR Student
Train the dual-objective LTR ranker ensemble on the distilled training data using the engineered tabular features.
```bash
python src/train_ltr.py
```
This script trains and outputs two distinct rankers:
1. **Pairwise Ranker (`rank:pairwise`):** Minimizes swap errors between candidate pairs.
2. **NDCG Ranker (`rank:ndcg`):** Directly optimizes Normalized Discounted Cumulative Gain for top-of-list performance.

- **Outputs:**
  - `models/xgb_ranker_pairwise.json`
  - `models/xgb_ranker_ndcg.json`
  - `models/xgb_ranker.json` (Legacy/Fallback model)

---

## 📊 4. Evaluation & Walkthrough

### Step 8: Run Evaluation & Ablation Study
Evaluate the LTR models on an **80/20 train/validation split** on holdout candidates.
```bash
python src/evaluate_pipeline.py
```

This will output the **NDCG@100 validation metrics**:
- **Baseline Hybrid Search (RRF Score):** Baseline score before LTR.
- **XGBoost LTR Pairwise Model:** Score achieved by the pairwise objective.
- **XGBoost LTR NDCG Model:** Score achieved by the NDCG objective.
- **Blended Dual-Objective LTR Ensemble:** Performance when blending predictions from both models.
- **Neural Cross-Encoder (Teacher Gold):** The theoretical upper performance ceiling.

The evaluation will also list the **Top 15 Feature Importances** to demonstrate which signals (e.g., `skill_score`, `yoe`, `ai_continuity`) influence ranking the most.

---

## 🛠️ 5. Troubleshooting & RAM Optimization

- **OutOfMemory (OOM) Errors:** The pipeline automatically configures PyTorch to use 4 CPU threads and frees the BGE embedding model (`del embed_model; gc.collect()`) prior to launching the Cross-Encoder teacher to prevent RAM exhaustion.
- **Sorting Requirement:** XGBoost LTR training requires the input training dataframe to be grouped and sorted by `query_id` with a corresponding group size list. If you modify training data, ensure `df.sort_values(by=["query_id", "relevance"], ascending=[True, False])` is maintained.
