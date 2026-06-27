import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

def calculate_ndcg(true_labels, predicted_scores, k=20):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG) at K.
    Assuming true_labels are graded relevance (0, 1, 2, 3)
    """
    if len(true_labels) == 0:
        return 0.0
        
    df = pd.DataFrame({"label": true_labels, "score": predicted_scores})
    
    # Sort by predicted score
    df = df.sort_values(by="score", ascending=False).head(k)
    
    # DCG
    dcg = 0.0
    for i, label in enumerate(df["label"]):
        dcg += (2**label - 1) / np.log2(i + 2)
        
    # IDCG (Ideal DCG)
    ideal_df = df.sort_values(by="label", ascending=False).head(k)
    idcg = 0.0
    for i, label in enumerate(ideal_df["label"]):
        idcg += (2**label - 1) / np.log2(i + 2)
        
    if idcg == 0:
        return 0.0
        
    return dcg / idcg

def evaluate():
    print("==========================================")
    print("  OBJECTIVE PIPELINE EVALUATION (80/20 SPLIT) ")
    print("==========================================\n")
    
    print("Loading distilled training data...")
    try:
        df = pd.read_csv("data/training_data.csv")
    except FileNotFoundError:
        print("Training data not found. Please run generate_training_data.py first.")
        return
        
    # Split candidates into 80% Train, 20% Holdout Validation
    # We do a clean holdout split to ensure LTR generalizes objectively.
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Train Size: {len(train_df)} | Holdout Validation Size: {len(val_df)}")
    
    # Get features list dynamically
    exclude = {"candidate_id", "full_text", "teacher_score", "relevance"}
    features = [col for col in df.columns if col not in exclude]
    
    # Train LTR on train_df ONLY
    X_train = train_df[features]
    y_train = train_df["relevance"]
    
    # Train Ranker
    ranker = xgb.XGBRanker(
        tree_method="hist",
        objective="rank:pairwise",
        learning_rate=0.1,
        n_estimators=100,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    ranker.fit(X_train, y_train, group=[len(X_train)])
    
    # Evaluate on holdout validation set
    labels_val = val_df["relevance"].values
    
    # 1. Baseline Hybrid (Semantic + BM25)
    ndcg_hybrid = calculate_ndcg(labels_val, val_df["rrf_score"].values, k=100)
    
    # 2. XGBoost LTR on validation set (Unseen Candidates)
    X_val = val_df[features]
    predicted_val_scores = ranker.predict(X_val)
    ndcg_xgb = calculate_ndcg(labels_val, predicted_val_scores, k=100)
    
    # 3. CrossEncoder Teacher Score (Upper Ceiling / Semantic Gold)
    ndcg_ce = calculate_ndcg(labels_val, val_df["teacher_score"].values, k=100)
    
    print("\n------------------------------------------")
    print("      NDCG@100 ON HOLDOUT VALIDATION SET  ")
    print("------------------------------------------")
    print(f"1. Baseline Hybrid Search (RRF Score)     -> NDCG@100: {ndcg_hybrid:.4f}")
    print(f"2. XGBoost Learning-to-Rank (LTR Student) -> NDCG@100: {ndcg_xgb:.4f}")
    print(f"3. Neural Cross-Encoder (Teacher Gold)    -> NDCG@100: {ndcg_ce:.4f}")
    
    print("\n------------------------------------------")
    print("      STUDENT FEATURE IMPORTANCES         ")
    print("------------------------------------------")
    importances = ranker.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    
    # Print top 15 features for clarity
    print("Top 15 Most Predictive Tabular Features:")
    for i in range(min(15, len(features))):
        idx = sorted_idx[i]
        print(f"  {features[idx]:<25}: {importances[idx]:.4f}")
        
    print("\nObjective Evaluation Complete.")

if __name__ == "__main__":
    evaluate()
