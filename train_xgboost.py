import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("labeled_dataset.csv")
print(f"Dataset: {df.shape[0]} rows | Fraud: {df['is_fraud'].sum()} | Clean: {(df['is_fraud']==0).sum()}")

feature_cols = ["roa", "roa_change", "leverage", "cash_to_assets", "asset_growth",
                "cash_profit_ratio", "income_change", "dsri", "rev_growth",
                "recv_growth", "rev_recv_divergence", "leverage_change", "accruals"]

X = df[feature_cols].values
y = df["is_fraud"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# XGBOOST + SMOTE
# ============================================
print("\n" + "="*50)
print("XGBOOST + SMOTE (oversampling minority class)")
print("="*50)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_scores = []
all_y_test = []
all_y_prob = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # SMOTE: create synthetic fraud examples to balance training data
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"  Fold {fold+1}: Train before SMOTE: {sum(y_train)} fraud / {len(y_train)} total")
    print(f"           Train after SMOTE:  {sum(y_train_smote)} fraud / {len(y_train_smote)} total")

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1),
        eval_metric="aucpr",
        random_state=42,
        use_label_encoder=False
    )
    xgb.fit(X_train_smote, y_train_smote)

    y_prob = xgb.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    xgb_scores.append(pr_auc)

    all_y_test.extend(y_test)
    all_y_prob.extend(y_prob)

    if fold == 0:
        y_pred = xgb.predict(X_test)
        print(f"\n  Fold 1 report:")
        print(classification_report(y_test, y_pred, target_names=["Clean", "Fraud"]))

print(f"\nAverage PR-AUC: {np.mean(xgb_scores):.4f} (+/- {np.std(xgb_scores):.4f})")

# Feature importance
importances = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop features (XGBoost):")
for feat, val in importances.items():
    print(f"  {feat}: {val:.4f}")

# ============================================
# FINAL MODEL: Train on all data, score everything
# ============================================
print("\n" + "="*50)
print("SCORING ALL COMPANIES")
print("="*50)

smote_full = SMOTE(random_state=42, k_neighbors=3)
X_full_smote, y_full_smote = smote_full.fit_resample(X_scaled, y)

xgb_final = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    scale_pos_weight=len(y[y==0]) / max(len(y[y==1]), 1),
    eval_metric="aucpr", random_state=42, use_label_encoder=False
)
xgb_final.fit(X_full_smote, y_full_smote)

df["fraud_score"] = xgb_final.predict_proba(X_scaled)[:, 1]

# Top 20 suspicious
print("\nTop 20 Most Suspicious Filings:")
suspicious = df.nlargest(20, "fraud_score")[["company", "ticker", "year", "fraud_score", "is_fraud"]]
print(suspicious.to_string(index=False))

# Hit rate in top 20
top20_hits = suspicious["is_fraud"].sum()
print(f"\nFraud cases in top 20: {top20_hits}/20 ({100*top20_hits/20:.0f}% hit rate)")

# Save
df.to_csv("scored_dataset.csv", index=False)
print("\nSaved to scored_dataset.csv")