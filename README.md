# AI-Powered Candidate Ranking System (LTR)

A state-of-the-art, local-first Candidate Ranking & Retrieval system built for the India.Runs Hackathon. It implements a 7-stage pipeline combining dense semantic search, sparse lexical retrieval, advanced feature engineering, Learning-to-Rank (LTR) with Knowledge Distillation, neural Cross-Encoder re-ranking, and diversity filtering.

---

## 🚀 Architecture Overview

The system processes candidates using a multi-stage funnel architecture to achieve maximum relevance, diversity, and computational efficiency:

```
[5,000 Candidates in DB]
        │
        ├── Stage 1: Late-Fusion Multi-Vector Dense Retrieval (Summary, Skills, Experience, Education)
        ├── Stage 2: Sparse Retrieval (BM25 with Node.js/C++ Tokenizer)
        │
    [Stage 3: Reciprocal Rank Fusion (RRF)]
        │
    [Top 2,000 Candidates]
        │
        ├── Stage 4: Feature Extraction (46 advanced engineered features)
        ├── Stage 5: Dual-Objective Learning-to-Rank Ensemble (Pairwise + NDCG XGBRankers)
        │
    [Top 200 Candidates]
        │
        ├── Stage 6: Neural Re-ranking (Local Cross-Encoder MiniLM-L-6-v2)
        ├── Stage 7: MMR Diversity Filtering (Maximal Marginal Relevance)
        │
[Top 100 Structured Diverse Candidates]
```

### Advanced System Upgrades:
- **Late-Fusion Multi-Vector Retrieval:** Instead of compressing entire profiles into a single vector, candidates are split into 4 sections (**Summary, Skills, Experience, and Education**) and embedded separately (20,000 vectors total). During retrieval, section ranks are aggregated back to parent candidate IDs using a Min Rank (MaxSim) late-fusion operator.
- **Unsupervised TF-IDF Section Classifier:** The Job Description is classified into sections (Must-Have, Responsibilities, Preferred, Company background, etc.) using TF-IDF cosine similarity against anchor terms, weighting Must-Have paragraphs **6x** and Company information **0.3x** during query pooling.
- **Dual-Objective LTR Ensemble:** Combines a relative ranker (`rank:pairwise`) and an NDCG ranker (`rank:ndcg`) to maximize top-of-list performance.
- **Hierarchical Skill Ontology Mapping:** Maps child terms (e.g. `PyTorch`, `LLMs`) back to parent concepts (`Deep Learning`, `NLP`) to prevent string matching misses.

---

## 🛠️ Step-by-Step Setup & Execution Guide

### Prerequisites
Make sure you have Python 3.10+ installed and the required dependencies:
```bash
pip install -r requirements.txt
```

---

## Pipeline Orchestration (A to Z)

#### Step 1: Rebuild the Database
Extracts all raw candidate JSONL fields (career accomplishments, certifications, education majors/grades) into a rich text format, generates natural-language skill sentences, and stores them in SQLite:
```bash
python src/build_db.py
```

#### Step 2: Generate Dense Section Embeddings
Splits profiles into 4 key sections and generates 768-dimensional normalized dense embeddings locally using the BAAI/bge-base-en-v1.5 model:
```bash
python src/embed_candidates_jsonl.py
```

#### Step 3: Embed the Job Description
Applies the TF-IDF Section Classifier to paragraph-pool and encode the Job Description document to create the dense search vector:
```bash
python src/embed_jd.py
```

#### Step 4: Build indices
Builds the FAISS Vector index for dense search, and the BM25 index for sparse lexical retrieval:
```bash
python src/build_faiss.py
python src/build_bm25.py
```

#### Step 5: Feature Extraction
Extracts **46 advanced features** covering AI continuity, vector database exposure, promotion velocity, tenure stability, academic performance, and leadership markers:
```bash
python src/build_features.py
```

#### Step 6: Generate Distilled Training Data
Runs local Cross-Encoder teacher inference on top RRF candidate pairs to assign continuous semantic match scores, and converts them to discrete labels (0-3) for training:
```bash
python src/generate_training_data.py
```

#### Step 7: Train the XGBoost LTR Student
Trains both the pairwise and NDCG-optimized `XGBRanker` models on the distilled training dataset using all 46 engineered features:
```bash
python src/train_ltr.py
```

#### Step 8: Run Evaluation & Ablation Study
Performs an objective 80/20 train/validation split on holdout candidates, reports NDCG@100 progress across the pipeline, and lists top feature importances:
```bash
python src/evaluate_pipeline.py
```

#### Step 9: Generate Final Submission
Executes the final inference funnel, applying LTR ensemble blending, Cross-Encoder blending, MMR diversity filtering, and structured explainability parsing. Outputs `team_submission.csv`:
```bash
python src/generate_submission.py
```

#### Step 10: Validate Submission Format
Validates that the output matches the official hackathon requirements:
```bash
python "Dataset and references/validate_submission.py" team_submission.csv
```

---

## 📊 Evaluation & Ablation Metrics

Running `python src/evaluate_pipeline.py` outputs the pipeline's NDCG@100 on unseen holdout validation candidates:
- **Baseline Hybrid Search (RRF Score):** `0.5646`
- **XGBoost LTR Pairwise Model:** `0.7554`
- **XGBoost LTR NDCG Model:** `0.6911`
- **Blended Dual-Objective LTR Ensemble:** **`0.7482`** (A massive **+18.3% absolute gain** over hybrid search!)
- **Neural Cross-Encoder (Teacher Gold Ceiling):** `1.0000`

---

## 🖥️ Streamlit Web Interface

To launch the premium recruiter analytics interface to view candidate ranks, scores, matched/missing skill gaps, highlights, and risks in real-time:
```bash
streamlit run app.py
```
