import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve, auc
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("labeled_dataset.csv")
print(f"Dataset: {df.shape[0]} rows")
print(f"Fraud: {df['is_fraud'].sum()} | Clean: {(df['is_fraud']==0).sum()}")

# Updated feature list
feature_cols = ["roa", "roa_change", "leverage", "cash_to_assets", "asset_growth",
                "cash_profit_ratio", "income_change", "dsri", "rev_growth",
                "recv_growth", "rev_recv_divergence", "leverage_change", "accruals"]

X = df[feature_cols].values
y = df["is_fraud"].values

print(f"Features: {len(feature_cols)}")
print(f"Fraud cases in model: {y.sum()}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# MODEL 1: Logistic Regression
# ============================================
print("\n" + "="*50)
print("MODEL 1: Logistic Regression")
print("="*50)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)

    y_prob = lr.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    lr_scores.append(pr_auc)

    if fold == 0:
        y_pred = lr.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=["Clean", "Fraud"]))

print(f"Average PR-AUC: {np.mean(lr_scores):.4f} (+/- {np.std(lr_scores):.4f})")

coefs = pd.Series(lr.coef_[0], index=feature_cols).abs().sort_values(ascending=False)
print("\nTop features:")
for feat, val in coefs.head(5).items():
    print(f"  {feat}: {val:.4f}")

# ============================================
# MODEL 2: Random Forest
# ============================================
print("\n" + "="*50)
print("MODEL 2: Random Forest")
print("="*50)

rf_scores = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample",
        max_depth=8, min_samples_leaf=5, random_state=42
    )
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    rf_scores.append(pr_auc)

    if fold == 0:
        y_pred = rf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=["Clean", "Fraud"]))

print(f"Average PR-AUC: {np.mean(rf_scores):.4f} (+/- {np.std(rf_scores):.4f})")

importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop features:")
for feat, val in importances.head(5).items():
    print(f"  {feat}: {val:.4f}")

# ============================================
# FIND SUSPICIOUS COMPANIES
# ============================================
print("\n" + "="*50)
print("TOP SUSPICIOUS FILINGS (by model score)")
print("="*50)

# Train on full data, score everything
rf_full = RandomForestClassifier(
    n_estimators=300, class_weight="balanced_subsample",
    max_depth=8, min_samples_leaf=5, random_state=42
)
rf_full.fit(X_scaled, y)
df["fraud_score"] = rf_full.predict_proba(X_scaled)[:, 1]

# Show top 20 most suspicious filings
suspicious = df.nlargest(20, "fraud_score")[["company", "ticker", "year", "fraud_score", "is_fraud"]]
print(suspicious.to_string(index=False))

df.to_csv("scored_dataset.csv", index=False)
print("\nSaved scored dataset to scored_dataset.csv")

# ============================================
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Logistic Regression PR-AUC: {np.mean(lr_scores):.4f}")
print(f"Random Forest PR-AUC:       {np.mean(rf_scores):.4f}")