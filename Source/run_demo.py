import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, confusion_matrix, classification_report
from xgboost import XGBClassifier
import shap
import os
import warnings

# Suppress version-specific warnings for cleaner demo output
warnings.filterwarnings('ignore')

DATA_PATH = '/Users/bhavika/Documents/fraud detection/Dataset/creditcard.csv'

def run_demo():
    print("[SYSTEM] Initializing Fraud Detection Demo Pipeline...")
    
    # Load and subset for demonstration efficiency
    df = pd.read_csv(DATA_PATH)
    df_sample = df.sample(n=50000, random_state=42)
    print(f"[DATA] Processing {len(df_sample)} transactions from ULB dataset.")
    
    X = df_sample.drop('Class', axis=1)
    y = df_sample['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("[MODEL] Training XGBoost with cost-sensitive weighting...")
    # scale_pos_weight used to handle extreme class imbalance (0.17% fraud)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = neg_count / pos_count if pos_count > 0 else 1
    
    model = XGBClassifier(scale_pos_weight=scale_weight, random_state=42)
    model.fit(X_train, y_train)
    
    print("[EVAL] Optimizing decision threshold for high recall (>80%)...")
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    
    # Target 80% recall to minimize missed fraud cases
    if len(recall[recall >= 0.8]) > 0:
        idx = np.where(recall >= 0.8)[0][-1]
        best_threshold = thresholds[min(idx, len(thresholds)-1)]
    else:
        best_threshold = 0.5
        
    y_pred_optimized = (y_prob >= best_threshold).astype(int)
    print(f"[INFO] Selected Threshold: {best_threshold:.4f}")
    print("\n--- Model Performance Summary ---")
    print(classification_report(y_test, y_pred_optimized))
    
    print("\n[SHAP] Computing local feature attribution for anomalies...")
    explainer = shap.TreeExplainer(model)
    fraud_indices = np.where(y_test == 1)[0]
    
    if len(fraud_indices) > 0:
        target_idx = fraud_indices[0]
        sample_txn = X_test.iloc[target_idx]
        shap_values = explainer.shap_values(sample_txn.values.reshape(1, -1))
        
        # Identify top drivers of the fraud classification
        top_indices = np.argsort(np.abs(shap_values))[0][-3:]
        top_features = X.columns[top_indices].tolist()
        print(f"[RESULT] Key Risk Drivers: {', '.join(top_features)}")
        
        print("\n[GEMINI] Generating Automated Suspicious Activity Report (SAR)...")
        # Template for automated compliance narrative
        mock_sar = (
            f"CASE SUMMARY: Transaction flagged due to anomalous variance in features {top_features[0]} and {top_features[1]}. "
            f"Transaction Amount: ${sample_txn['Amount']:.2f}. "
            f"RISK ASSESSMENT: HIGH. "
            f"ACTION: Flagged for manual review; temporary card restriction initiated."
        )
        print(f"\nReport Narrative:\n{mock_sar}")
    else:
        print("[WARN] No fraud samples in the current test subset. Skipping attribution demo.")

if __name__ == "__main__":
    run_demo()
