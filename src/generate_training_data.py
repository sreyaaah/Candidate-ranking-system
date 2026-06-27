import pandas as pd
import sqlite3
import os
import sys
import numpy as np
import torch
import gc
from sentence_transformers import CrossEncoder, SentenceTransformer
import config

# Ensure src is in the path to import search_hybrid
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from search_hybrid import get_hybrid_scores
from extract_skills import extract_required_skills, match_skills_with_ontology
from docx import Document

def generate_training_data():
    torch.set_num_threads(4)
    print("Configured PyTorch to use 4 CPU threads.")
    
    # Define multi-query training groups (Fixes Weakness 2)
    print("Defining LTR Multi-Query Groups...")
    
    # Load main JD
    doc = Document(config.JD_PATH)
    main_jd_text = "\n".join([para.text for para in doc.paragraphs])
    
    queries = [
        {"id": 0, "text": main_jd_text, "name": "Main JD (Senior ML)"},
        {"id": 1, "text": "Senior DevOps Engineer. Required skills: docker, kubernetes, aws, terraform, ci/cd, linux, monitoring, ansible.", "name": "DevOps"},
        {"id": 2, "text": "Data Engineer. Required skills: spark, hadoop, kafka, postgresql, python, big data, airflow, mysql.", "name": "Data Engineering"},
        {"id": 3, "text": "Full Stack Developer. Required skills: python, react, javascript, django, sql, docker, typescript, html, css.", "name": "Full Stack"}
    ]
    
    # Step 1: Load BGE model, generate query embeddings, and immediately delete it to free RAM
    print(f"Loading local embedding model ({config.DENSE_MODEL_NAME}) for query generation...")
    embed_model = SentenceTransformer(config.DENSE_MODEL_NAME, local_files_only=True)
    embed_model.max_seq_length = config.MAX_SEQ_LENGTH
    
    # INT8 Quantization for BGE CPU speedup (Fixes Weakness 4)
    try:
        embed_model = torch.quantization.quantize_dynamic(
            embed_model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("BGE Query Model INT8 quantization enabled.")
    except Exception as e:
        print(f"BGE Quantization fallback: {e}")
        
    query_embs = {}
    print("Generating query embeddings...")
    for q in queries:
        query_embs[q["id"]] = embed_model.encode(
            "Represent this sentence for searching relevant passages: " + q["text"][:1200],
            normalize_embeddings=True
        )
        
    # Free memory (Fixes Windows Paging File OOM error 1455)
    del embed_model
    gc.collect()
    print("Embedding model successfully freed from RAM.")
        
    # Step 2: Load Cross-Encoder model
    print(f"\nLoading CrossEncoder ({config.CROSS_ENCODER_MODEL_NAME}) for Teacher Labeling...")
    ce_model = CrossEncoder(config.CROSS_ENCODER_MODEL_NAME, max_length=512, local_files_only=True)
        
    full_dataset = []
    
    for q in queries:
        print(f"\nProcessing Query Group {q['id']}: {q['name']}...")
        jd_text = q["text"]
        required_skills = extract_required_skills(jd_text)
        print(f"Dynamically extracted skills: {required_skills}")
        
        # Load pre-computed query embedding
        query_emb = query_embs[q["id"]]
        
        # We retrieve top 1,500 candidates per query to keep training dataset balanced and fast
        top_n = 1500
        hybrid_results = get_hybrid_scores(jd_text, top_n=top_n, query_emb=query_emb)
        
        candidate_ids = [res[0] for res in hybrid_results]
        
        conn = sqlite3.connect(config.DB_PATH)
        placeholders = ",".join(["?"] * len(candidate_ids))
        query_sql = f"""
            SELECT 
                c.id, c.skills, c.full_text, f.*
            FROM candidates c
            LEFT JOIN candidate_features f ON c.id = f.candidate_id
            WHERE c.id IN ({placeholders})
        """
        df = pd.read_sql_query(query_sql, conn, params=candidate_ids)
        df = df.loc[:, ~df.columns.duplicated()]
        conn.close()
        
        # Create truncation of JD for Cross-Encoder matching
        jd_trunc = jd_text[:300]
        if len(jd_text) > 1500:
            jd_trunc += "\n...[Requirements]...\n" + jd_text[-1200:]
            
        group_dataset = []
        for cand_id, score in hybrid_results:
            row = df[df["id"] == cand_id]
            if row.empty:
                continue
                
            row = row.iloc[0]
            
            # Skill Ontology check
            candidate_skills = [s.strip() for s in str(row["skills"]).split(",") if s.strip()]
            matched_skills, missing_skills = match_skills_with_ontology(candidate_skills, required_skills)
            skill_score = len(matched_skills) / len(required_skills) if required_skills else 0.0
            
            full_text = str(row["full_text"])[:1000]
            
            feat_dict = {
                "query_id": q["id"],
                "candidate_id": cand_id,
                "full_text": full_text,
                "rrf_score": score,
                "skill_score": skill_score,
            }
            
            # Copy all numerical features
            exclude_cols = {'id', 'skills', 'full_text', 'candidate_id'}
            for col in df.columns:
                if col not in exclude_cols:
                    val = row[col]
                    if pd.isna(val) or val is None:
                        val = 3.0 if col == 'education_tier' else 0.0
                    feat_dict[col] = val
                    
            group_dataset.append(feat_dict)
            
        group_df = pd.DataFrame(group_dataset)
        
        # Run Cross-Encoder on this query group
        print("Running Teacher inference...")
        ce_pairs = [(jd_trunc, txt) for txt in group_df["full_text"]]
        ce_scores = ce_model.predict(ce_pairs)
        norm_ce_scores = 1 / (1 + np.exp(-ce_scores))
        group_df["teacher_score"] = norm_ce_scores
        
        # Set honeypot tags to 0 relevance
        group_df.loc[group_df["honeypot_flag"] == 1, "teacher_score"] = 0.0
        
        # Assign relevance labels based on quantiles within this query group!
        q95 = group_df["teacher_score"].quantile(0.95)
        q80 = group_df["teacher_score"].quantile(0.80)
        q50 = group_df["teacher_score"].quantile(0.50)
        
        def assign_label(score):
            if score >= q95: return 3
            if score >= q80: return 2
            if score >= q50: return 1
            return 0
            
        group_df["relevance"] = group_df["teacher_score"].apply(assign_label)
        
        full_dataset.append(group_df)
        
    final_df = pd.concat(full_dataset, ignore_index=True)
    
    os.makedirs("data", exist_ok=True)
    final_df.to_csv("data/training_data.csv", index=False)
    print("\nSaved multi-query training data to data/training_data.csv successfully!")
    print(final_df.groupby("query_id")["relevance"].value_counts())

if __name__ == "__main__":
    generate_training_data()
