import numpy as np
import faiss

# Load resume embeddings
embeddings = np.load(
    "embeddings/resume_embeddings.npy"
)

dimension = embeddings.shape[1]

# Create FAISS index
index = faiss.IndexFlatIP(
    dimension
)

# Add embeddings
index.add(embeddings)

# Save index
faiss.write_index(
    index,
    "embeddings/resume.index"
)

print("FAISS index created successfully.")
print("Total resumes:", index.ntotal)