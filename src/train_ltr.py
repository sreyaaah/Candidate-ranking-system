import pandas as pd
import xgboost as xgb
import os

def train_ranker():
    print("Loading training data...")
    df = pd.read_csv("data/training_data.csv")
    
    # Sort by relevance just to be safe, though group handles it
    df = df.sort_values(by="relevance", ascending=False).reset_index(drop=True)
    
    features = [
        "rrf_score", "skill_score", "yoe", "ai_years", 
        "job_hopping_index", "github_score", "education_tier", "notice_period",
        "company_fit_score", "career_trajectory_score", "behavioral_score",
        "location_match", "honeypot_flag"
    ]
    
    X = df[features]
    y = df["relevance"]
    
    # In pairwise ranking, we need a 'group' array that specifies how many items are in each query.
    # Since we only have 1 query (the job description) and all 4000 candidates belong to it:
    groups = [len(X)]
    
    print("Initializing XGBRanker...")
    ranker = xgb.XGBRanker(
        tree_method="hist",
        objective="rank:pairwise",
        learning_rate=0.1,
        n_estimators=100,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    print("Training model...")
    ranker.fit(X, y, group=groups)
    
    # Save the model
    os.makedirs("models", exist_ok=True)
    model_path = "models/xgb_ranker.json"
    ranker.save_model(model_path)
    
    print(f"Model successfully saved to {model_path}!")
    
    # Print Feature Importances
    importances = ranker.feature_importances_
    print("\nFeature Importances:")
    for feat, imp in zip(features, importances):
        print(f"  {feat}: {imp:.4f}")

if __name__ == "__main__":
    train_ranker()
