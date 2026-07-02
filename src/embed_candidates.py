import os
import numpy as np
from sentence_transformers import SentenceTransformer
from extract_text import extract_text
import pickle

print("Loading BGE model...")

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

resume_folder = "data/resumes"

embeddings = []
resume_names = []

for file in os.listdir(resume_folder):

    if file.endswith(".pdf"):

        path = os.path.join(
            resume_folder,
            file
        )

        text = extract_text(path)

        query = (
            "Represent this sentence for searching relevant passages: "
            + text
        )

        embedding = model.encode(
            query,
            normalize_embeddings=True
        )

        embeddings.append(embedding)
        resume_names.append(file)

        print(file, "→", embedding.shape)

embeddings = np.array(embeddings)

np.save(
    "embeddings/resume_embeddings.npy",
    embeddings
)

with open(
    "embeddings/resume_names.pkl",
    "wb"
) as f:
    pickle.dump(
        resume_names,
        f
    )

print("\nFinal Shape:", embeddings.shape)
print("Resume embeddings saved successfully.")