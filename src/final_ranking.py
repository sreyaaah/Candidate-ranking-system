import numpy as np
import faiss
import pickle
import os

from extract_text import extract_text

required_skills = [
    "python",
    "react",
    "node.js",
    "docker",
    "sql"
]

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
    len(resume_names)
)

print("\nFinal Candidate Ranking:\n")

for i in range(len(indices[0])):

    idx = indices[0][i]

    resume_name = resume_names[idx]

    text = extract_text(
        os.path.join(
            "data/resumes",
            resume_name
        )
    ).lower()

    matched = 0

    for skill in required_skills:
        if skill in text:
            matched += 1

    skill_score = matched / len(required_skills)

    semantic_score = scores[0][i]

    final_score = (
        0.7 * semantic_score
        + 0.3 * skill_score
    )

    print(
        f"{resume_name}"
    )
    print(
        f"Semantic Score: {semantic_score:.4f}"
    )
    print(
        f"Skill Score: {skill_score:.2f}"
    )
    print(
        f"Final Score: {final_score:.4f}\n"
    )