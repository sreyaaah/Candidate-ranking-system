import os
import numpy as np
from sentence_transformers import SentenceTransformer
from extract_text import extract_text
from docx import Document

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

# Read Job Description
doc = Document(
    "data/jobs/job_description.docx"
)

job_description = ""

for para in doc.paragraphs:
    job_description += para.text + "\n"

jd_query = (
    "Represent this sentence for searching relevant passages: "
    + job_description
)

jd_embedding = model.encode(
    jd_query,
    normalize_embeddings=True
)

resume_folder = "data/resumes"

scores = []

for file in os.listdir(resume_folder):

    if file.endswith(".pdf"):

        text = extract_text(
            os.path.join(
                resume_folder,
                file
            )
        )

        resume_query = (
            "Represent this sentence for searching relevant passages: "
            + text
        )

        resume_embedding = model.encode(
            resume_query,
            normalize_embeddings=True
        )

        similarity = np.dot(
            jd_embedding,
            resume_embedding
        )

        scores.append(
            (file, similarity)
        )

scores.sort(
    key=lambda x: x[1],
    reverse=True
)

print("\nCandidate Ranking:\n")

for rank, (name, score) in enumerate(scores, 1):
    print(
        f"{rank}. {name} → {score:.4f}"
    )