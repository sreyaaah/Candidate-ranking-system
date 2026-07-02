import sqlite3
import pickle
import os
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import re

DB_PATH = "data/candidates.db"

def tokenize(text):
    if not text:
        return []
    # simple tokenization: lowercase, split by non-alphanumeric
    return [word for word in re.split(r'\W+', text.lower()) if word]

def build_bm25():
    print("Loading candidate texts from database...")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, full_text FROM candidates")
    
    candidate_ids = []
    rows = cursor.fetchall()
    corpus = [row[1] for row in rows]
    candidate_ids = [row[0] for row in rows]
    
    print(f"Found {len(rows)} candidates. Tokenizing...")
    
    # Custom tokenizer that preserves things like C++ and Node.js
    def custom_tokenize(text):
        if not text: return []
        return re.findall(r"(?i)\b[a-z0-9_+#.]+\b", text.lower())

    tokenized_corpus = []
    for doc in tqdm(corpus, desc="Tokenizing documents"):
        tokenized_corpus.append(custom_tokenize(doc))
        
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save the index and the mapping
    os.makedirs("embeddings", exist_ok=True)
    with open("embeddings/bm25_index.pkl", "wb") as f:
        pickle.dump(bm25, f)
        
    with open("embeddings/bm25_names.pkl", "wb") as f:
        pickle.dump(candidate_ids, f)
        
    print("BM25 index saved successfully.")

if __name__ == "__main__":
    build_bm25()
