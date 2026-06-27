from sentence_transformers import SentenceTransformer
from docx import Document
import numpy as np

print("Loading Job Description...")

doc = Document("data/jobs/job_description.docx")

paragraphs = []
for para in doc.paragraphs:
    text = para.text.strip()
    if len(text) > 15:
        paragraphs.append(text)

print(f"Extracted {len(paragraphs)} meaningful paragraphs from JD.")

print("Loading BGE model...")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

queries = [
    "Represent this sentence for searching relevant passages: " + p 
    for p in paragraphs
]

print("Encoding paragraphs...")
embeddings = model.encode(
    queries,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Calculating document embedding (mean pooling)...")
final_jd_embedding = np.mean(embeddings, axis=0)

final_jd_embedding = final_jd_embedding / np.linalg.norm(final_jd_embedding)

print("Final Embedding Shape:", final_jd_embedding.shape)
print("Preview:", final_jd_embedding[:10])

np.save("embeddings/jd_embedding.npy", final_jd_embedding)

print("JD embedding saved successfully.")