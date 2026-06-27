import re

KNOWN_SKILLS = [
    "python", "java", "c++", "c#", "javascript", "typescript", "ruby", "go", "rust", "php",
    "react", "angular", "vue", "node.js", "express", "django", "flask", "fastapi", "spring",
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ansible", "jenkins", "ci/cd",
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch", "cassandra",
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch",
    "scikit-learn", "pandas", "numpy", "matplotlib", "spark", "hadoop", "kafka", "airflow"
]

# Parent-to-Child Skill Ontology Mapping
# If the JD requires a "parent" concept, and the candidate possesses a "child" technology, it counts as a match!
ONTOLOGY_MAP = {
    "machine learning": ["tensorflow", "pytorch", "scikit-learn", "deep learning", "statistical modeling", "gans", "reinforcement learning"],
    "deep learning": ["tensorflow", "pytorch", "keras", "neural networks", "lora", "fine-tuning llms"],
    "nlp": ["llm", "transformers", "langchain", "nltk", "spacy", "bert", "text classification", "speech recognition", "tts"],
    "computer vision": ["opencv", "image classification", "object detection", "yolo", "gans"],
    "sql": ["postgresql", "mysql", "sqlite", "oracle", "sql server", "nosql", "mongodb", "redis", "cassandra"],
    "cloud": ["aws", "azure", "gcp", "terraform", "serverless", "s3", "ec2"],
    "mlops": ["docker", "kubernetes", "airflow", "mlflow", "dvc", "jenkins", "ci/cd"],
    "big data": ["spark", "hadoop", "kafka", "flink", "hive", "mapreduce"]
}

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

def match_skills_with_ontology(candidate_skills_list, required_skills):
    """
    Checks candidate skills against required skills, using the ontology map
    to grant parent credit if a candidate has a child technology.
    """
    candidate_set = set([s.lower().strip() for s in candidate_skills_list])
    matched_skills = []
    missing_skills = []
    
    for req in required_skills:
        req_clean = req.lower().strip()
        # Direct Match
        if req_clean in candidate_set:
            matched_skills.append(req)
            continue
            
        # Ontology Parent-Child Match
        children = ONTOLOGY_MAP.get(req_clean, [])
        ontology_matched = False
        for child in children:
            if child in candidate_set:
                matched_skills.append(req)
                ontology_matched = True
                break
                
        if not ontology_matched:
            # Check if candidate set contains the required skill as a substring (e.g. "postgresql database" matching "postgresql")
            substring_matched = False
            for cand in candidate_set:
                if req_clean in cand:
                    matched_skills.append(req)
                    substring_matched = True
                    break
            if not substring_matched:
                missing_skills.append(req)
                
    return list(set(matched_skills)), list(set(missing_skills))
