import pandas as pd
import xgboost as xgb
import os
import config

def train_ranker():
    print("Loading training data...")
    df = pd.read_csv("data/training_data.csv")
    
    # Crucial for LTR: Sort data by query_id first, then relevance
    df = df.sort_values(by=["query_id", "relevance"], ascending=[True, False]).reset_index(drop=True)
    
    # Calculate group lengths (number of candidates per query group)
    groups = df.groupby("query_id").size().tolist()
    print(f"Loaded {len(groups)} query groups with sizes: {groups}")
    
    # Dynamically extract all features except metadata columns
    exclude = {"query_id", "candidate_id", "full_text", "teacher_score", "relevance", "matched_skills", "missing_skills"}
    features = [col for col in df.columns if col not in exclude]
    
    X = df[features]
    y = df["relevance"]
    
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
    ranker_pairwise.save_model(config.LTR_PAIRWISE_PATH)
    print(f"Model A saved to {config.LTR_PAIRWISE_PATH}")
    
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
    ranker_ndcg.save_model(config.LTR_NDCG_PATH)
    print(f"Model B saved to {config.LTR_NDCG_PATH}")
    
    # Save a legacy fallback so standard inference doesn't break
    ranker_pairwise.save_model(config.LTR_LEGACY_PATH)
    
    # Print Feature Importances for Pairwise Model
    importances = ranker_pairwise.feature_importances_
    print("\nFeature Importances (Pairwise):")
    sorted_idx = importances.argsort()[::-1]
    for idx in sorted_idx[:15]:
        print(f"  {features[idx]:<25}: {importances[idx]:.4f}")

if __name__ == "__main__":
    train_ranker()
