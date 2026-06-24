from sentence_transformers import SentenceTransformer
from docx import Document
import numpy as np

print("Loading Job Description...")

doc = Document("data/jobs/job_description.docx")

job_description = ""

for para in doc.paragraphs:
    job_description += para.text + "\n"

print("Loading BGE model...")

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

query = (
    "Represent this sentence for searching relevant passages: "
    + job_description
)

embedding = model.encode(
    query,
    normalize_embeddings=True
)

print("Embedding Shape:", embedding.shape)
print(embedding[:10])
np.save(
    "embeddings/jd_embedding.npy",
    embedding
)

print("JD embedding saved successfully.")