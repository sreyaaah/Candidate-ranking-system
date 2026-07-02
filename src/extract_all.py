import os
from extract_text import extract_text

resume_folder = "data/resumes"

for file in os.listdir(resume_folder):

    if file.endswith(".pdf"):

        path = os.path.join(
            resume_folder,
            file
        )

        text = extract_text(path)

        print("\n" + "=" * 50)
        print("Resume:", file)
        print("=" * 50)
        print(text[:500])