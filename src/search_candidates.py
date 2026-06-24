import numpy as np
import faiss
import pickle

jd_embedding = np.load(
    "embeddings/jd_embedding.npy"
)

index = faiss.read_index(
    "embeddings/resume.index"
)

with open(
    "embeddings/resume_names.pkl",
    "rb"
) as f:
    resume_names = pickle.load(f)

scores, indices = index.search(
    jd_embedding.reshape(1, -1),
    3
)

print("\nCandidate Ranking:\n")

for i in range(len(indices[0])):

    idx = indices[0][i]

    print(
        f"{i+1}. "
        f"{resume_names[idx]} "
        f"→ {scores[0][i]:.4f}"
    )