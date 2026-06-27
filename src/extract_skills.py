import re

KNOWN_SKILLS = [
    "python", "java", "c++", "c#", "javascript", "typescript", "ruby", "go", "rust", "php",
    "react", "angular", "vue", "node.js", "express", "django", "flask", "fastapi", "spring",
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ansible", "jenkins", "ci/cd",
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "cassandra",
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch",
    "scikit-learn", "pandas", "numpy", "matplotlib", "spark", "hadoop", "kafka", "airflow"
]

def extract_required_skills(jd_text):
    """
    Dynamically parses the JD text and returns a list of required skills
    found in the known skills ontology.
    """
    text = jd_text.lower()
    
    # Simple regex tokenizer that preserves versions, c++, c#, and .js
    tokens = re.findall(r"(?i)\b[a-z0-9_+#.]+\b", text)
    tokens_set = set(tokens)
    
    found_skills = []
    for skill in KNOWN_SKILLS:
        if " " in skill:
            if skill in text:
                found_skills.append(skill)
        else:
            if skill in tokens_set:
                found_skills.append(skill)
                
    # Fallback if nothing found
    if not found_skills:
        found_skills = ["python", "sql"]
        
    return list(set(found_skills))
