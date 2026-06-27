from sentence_transformers import SentenceTransformer
from docx import Document
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading Job Description...")

doc = Document("data/jobs/job_description.docx")

paragraphs = []
for para in doc.paragraphs:
    text = para.text.strip()
    if len(text) > 15:
        paragraphs.append(text)

print(f"Extracted {len(paragraphs)} paragraphs from JD.")

# --- Upgraded: Unsupervised TF-IDF Section Classifier ---
print("\nRunning TF-IDF Section Classifier...")

# Define anchor documents representing each section type
anchors = {
    "must_have": "required skills qualifications requirements minimum basic must have experience criteria eligibility guidelines",
    "responsibilities": "responsibilities duties role expectations what you will do tasks daily day to day project execution deliver",
    "preferred": "preferred desired nice to have plus optional advantages extra beneficial secondary asset",
    "company": "about us company profile overview culture mission vision team founding values background series startup",
    "benefits": "benefits compensation perks health insurance leave salary equity package retirement bonus",
    "equal_opportunity": "equal opportunity diversity inclusion gender race religion veteran disability gender identity sexual orientation"
}

anchor_names = list(anchors.keys())
anchor_texts = list(anchors.values())

# We vectorise the paragraphs and the anchors together
vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit(paragraphs + anchor_texts)

anchor_vectors = vectorizer.transform(anchor_texts)

weights_map = {
    "must_have": 6.0,
    "responsibilities": 4.0,
    "preferred": 2.0,
    "company": 0.3,
    "benefits": 0.2,
    "equal_opportunity": 0.1
}

weights = []
classified_counts = {name: 0 for name in anchor_names}
classified_counts["default"] = 0

for para in paragraphs:
    para_vector = vectorizer.transform([para])
    # Compute cosine similarity between paragraph and each section anchor
    similarities = cosine_similarity(para_vector, anchor_vectors)[0]
    
    max_idx = np.argmax(similarities)
    max_sim = similarities[max_idx]
    
    # Assign section if similarity is above a threshold, else default
    if max_sim > 0.05:
        section = anchor_names[max_idx]
        weight = weights_map[section]
        classified_counts[section] += 1
    else:
        weight = 1.0
        classified_counts["default"] += 1
        
    weights.append(weight)

print("Section Classification Results:")
for sec, count in classified_counts.items():
    print(f"  {sec:<18}: {count} paragraphs")

print("\nLoading BGE model...")
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
weights = weights / np.sum(weights)

final_jd_embedding = np.average(embeddings, axis=0, weights=weights)
final_jd_embedding = final_jd_embedding / np.linalg.norm(final_jd_embedding)

print("Final Embedding Shape:", final_jd_embedding.shape)
print("Preview:", final_jd_embedding[:10])

np.save("embeddings/jd_embedding.npy", final_jd_embedding)

print("JD embedding saved successfully.")