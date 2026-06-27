from sentence_transformers import SentenceTransformer
from docx import Document
import numpy as np
import re

print("Loading Job Description...")

doc = Document("data/jobs/job_description.docx")

paragraphs = []
weights = []

# Rule-based Section Classification keywords
must_have_keywords = [r"must have", r"requirements", r"required skills", r"what we are looking for", r"what we need", r"experience required"]
responsibilities_keywords = [r"responsibilities", r"what you'll do", r"role", r"expectations"]
preferred_keywords = [r"preferred", r"nice to have", r"plus", r"desirable"]
company_keywords = [r"about us", r"company", r"we are", r"our team", r"founding team"]
benefits_keywords = [r"benefits", r"what we offer", r"perks", r"compensation"]
equal_opportunity_keywords = [r"equal opportunity", r"diversity", r"inclusion"]

def classify_paragraph_and_get_weight(text):
    text_lower = text.lower()
    
    # Priority 1: Must Have / Core Requirements
    if any(re.search(kw, text_lower) for kw in must_have_keywords):
        return 6.0
        
    # Priority 2: Responsibilities / Day-to-Day
    if any(re.search(kw, text_lower) for kw in responsibilities_keywords):
        return 4.0
        
    # Priority 3: Preferred / Nice to Have
    if any(re.search(kw, text_lower) for kw in preferred_keywords):
        return 2.0
        
    # Priority 4: Company Profile
    if any(re.search(kw, text_lower) for kw in company_keywords):
        return 0.3
        
    # Priority 5: Benefits
    if any(re.search(kw, text_lower) for kw in benefits_keywords):
        return 0.2
        
    # Priority 6: Equal Opportunity
    if any(re.search(kw, text_lower) for kw in equal_opportunity_keywords):
        return 0.1
        
    # Default fallback
    return 1.0

current_section_weight = 1.0

for para in doc.paragraphs:
    text = para.text.strip()
    if len(text) > 15:
        # Detect if paragraph is a section header (short text, ending with colon or bold)
        # If it's a section header, update the active section weight
        if len(text) < 40 and (text.endswith(":") or para.style.name.startswith("Heading")):
            current_section_weight = classify_paragraph_and_get_weight(text)
            
        paragraphs.append(text)
        weights.append(current_section_weight)

print(f"Extracted {len(paragraphs)} paragraphs from JD.")
print(f"Section weights preview (Max: {max(weights)}, Min: {min(weights)}, Mean: {np.mean(weights):.2f})")

print("Loading BGE model...")
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

queries = [
    "Represent this sentence for searching relevant passages: " + p 
    for p in paragraphs
]

print("Encoding paragraphs...")
embeddings = model.encode(
    queries,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Calculating document embedding (Section-Classified weighted pooling)...")
weights = np.array(weights, dtype=np.float32)
# Normalize to sum to 1.0
weights = weights / np.sum(weights)

final_jd_embedding = np.average(embeddings, axis=0, weights=weights)
final_jd_embedding = final_jd_embedding / np.linalg.norm(final_jd_embedding)

print("Final Embedding Shape:", final_jd_embedding.shape)
print("Preview:", final_jd_embedding[:10])

np.save("embeddings/jd_embedding.npy", final_jd_embedding)

print("JD embedding saved successfully.")