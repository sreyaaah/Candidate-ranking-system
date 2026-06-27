import pandas as pd
import xgboost as xgb
import os

def train_ranker():
    print("Loading training data...")
    df = pd.read_csv("data/training_data.csv")
    
    # Sort by relevance just to be safe, though group handles it
    df = df.sort_values(by="relevance", ascending=False).reset_index(drop=True)
    
    # Dynamically extract all features except metadata columns
    exclude = {"candidate_id", "full_text", "teacher_score", "relevance", "matched_skills", "missing_skills"}
    features = [col for col in df.columns if col not in exclude]
    
    X = df[features]
    y = df["relevance"]
    groups = [len(X)]
    
    os.makedirs("models", exist_ok=True)
    
    # --- Model A: Pairwise LTR Ranker ---
    print("\nTraining Model A: Pairwise XGBRanker...")
    ranker_pairwise = xgb.XGBRanker(
        tree_method="hist",
        objective="rank:pairwise",
        learning_rate=0.1,
        n_estimators=100,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    ranker_pairwise.fit(X, y, group=groups)
    pairwise_path = "models/xgb_ranker_pairwise.json"
    ranker_pairwise.save_model(pairwise_path)
    print(f"Model A saved to {pairwise_path}")
    
    # --- Model B: NDCG LTR Ranker ---
    print("\nTraining Model B: NDCG XGBRanker...")
    ranker_ndcg = xgb.XGBRanker(
        tree_method="hist",
        objective="rank:ndcg",
        learning_rate=0.1,
        n_estimators=100,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    ranker_ndcg.fit(X, y, group=groups)
    ndcg_path = "models/xgb_ranker_ndcg.json"
    ranker_ndcg.save_model(ndcg_path)
    print(f"Model B saved to {ndcg_path}")
    
    # Save a legacy fallback so standard inference doesn't break
    ranker_pairwise.save_model("models/xgb_ranker.json")
    
    # Print Feature Importances for Pairwise Model
    importances = ranker_pairwise.feature_importances_
    print("\nFeature Importances (Pairwise):")
    sorted_idx = importances.argsort()[::-1]
    for idx in sorted_idx[:15]:
        print(f"  {features[idx]:<25}: {importances[idx]:.4f}")

if __name__ == "__main__":
    train_ranker()
