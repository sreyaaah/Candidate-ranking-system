import os
from docx import Document

def generate_optimized_jd():
    doc = Document()
    
    # Title
    doc.add_heading("Job Description: Lead AI & Retrieval Systems Engineer", 0)
    
    # Overview
    doc.add_heading("Company & Role Overview", level=1)
    doc.add_paragraph(
        "Company: Redrob AI (Series A AI-native talent intelligence platform)\n"
        "Location: Pune/Noida, India (Hybrid)\n"
        "Experience Required: 5–9 years\n\n"
        "We are seeking a Lead AI & Retrieval Systems Engineer to own the intelligence and search matching layer of our "
        "next-generation talent platform. In this role, you will lead the design, development, and optimization of "
        "large-scale search architectures, candidate matching engines, and dual-objective ranking systems. This is a founding "
        "team role requiring a blend of deep technical machine learning knowledge and product-oriented delivery."
    )
    
    # Required Core Technical Stack
    doc.add_heading("Required Core Technical Stack", level=1)
    doc.add_paragraph("Candidates must have hands-on experience and expertise in the following technologies:")
    
    tech_stack = [
        "Programming Languages: Python, SQL",
        "Deep Learning & Machine Learning: PyTorch, TensorFlow, Scikit-learn, Numpy, Pandas",
        "Natural Language Processing (NLP): Transformers, BERT, LLM fine-tuning, LangChain, text classification",
        "Computer Vision: OpenCV, image classification, object detection (YOLO, GANs)",
        "Search & Retrieval Infrastructure: FAISS, Elasticsearch, Redis, vector databases",
        "Big Data & Pipelines: Apache Spark, Hadoop, Kafka, Airflow",
        "MLOps & Cloud Infrastructure: Docker, Kubernetes, AWS, Azure, GCP, Terraform, CI/CD, Jenkins"
    ]
    for tech in tech_stack:
        doc.add_paragraph(tech, style='List Bullet')
        
    # Core Concepts & Frameworks
    doc.add_heading("Core Concepts & Responsibilities", level=1)
    doc.add_paragraph("You will design, develop, and maintain the following pipeline components:")
    
    responsibilities = [
        "Design and scale hybrid retrieval indices combining dense similarity search (FAISS) and sparse lexical matching (BM25).",
        "Build, validate, and train dual-objective Learning-to-Rank (LTR) ensembles using XGBoost to predict rank relevance.",
        "Implement and fine-tune neural Cross-Encoder teacher-student networks to perform deep semantic text matching.",
        "Optimize search query expansion models using domain-specific skill ontologies.",
        "Secure retrieval systems against prompt injection attacks, spam profiles, and adversarial keyword-stuffing (honeypots).",
        "Evaluate ranking precision using objectives such as NDCG@100, Precision@K, and Mean Reciprocal Rank (MRR)."
    ]
    for resp in responsibilities:
        doc.add_paragraph(resp, style='List Bullet')
        
    # Qualifications
    doc.add_heading("Preferred Profile & Qualifications", level=1)
    doc.add_paragraph(
        "• B.Tech, M.Tech, or Ph.D. in Computer Science, Data Science, or AI from a Tier 1 educational institution.\n"
        "• Proven track record of shipping production-grade recommendation or ranking systems at scale.\n"
        "• Leadership experience leading engineering sprints or mentoring junior machine learning engineers.\n"
        "• Upward career trajectory with a solid technical tenure, demonstrating strong ownership and problem-solving skills."
    )
    
    # Save the document
    output_dir = "data/jobs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "job_description.docx")
    doc.save(output_path)
    print(f"Optimized job description saved successfully to: {output_path}")

if __name__ == "__main__":
    generate_optimized_jd()
