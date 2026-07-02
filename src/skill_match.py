from extract_text import extract_text
import os

required_skills = [
    "python",
    "react",
    "node.js",
    "docker",
    "sql"
]

resume_folder = "data/resumes"

for file in os.listdir(resume_folder):

    if file.endswith(".pdf"):

        text = extract_text(
            os.path.join(
                resume_folder,
                file
            )
        ).lower()

        matched = []

        for skill in required_skills:

            if skill in text:
                matched.append(skill)

        print("\n", file)
        print("Matched Skills:", matched)
        print(
            "Score:",
            len(matched),
            "/",
            len(required_skills)
        )