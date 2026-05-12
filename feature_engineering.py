import pandas as pd
import numpy as np

df = pd.read_csv("financial_data_clean.csv")
print(f"Loaded: {df.shape[0]} rows")

# Drop rows missing the core three (best coverage)
df = df.dropna(subset=["Assets", "NetIncomeLoss"])
df = df.sort_values(["company", "year"]).reset_index(drop=True)
print(f"After core filter: {df.shape[0]} rows")

# Previous year values
for col in ["Revenues", "AccountsReceivableNetCurrent", "Assets",
            "NetIncomeLoss", "Liabilities", "InventoryNet",
            "CashAndCashEquivalentsAtCarryingValue"]:
    df[f"prev_{col}"] = df.groupby("company")[col].shift(1)

# ============================================
# GROUP 1: HIGH COVERAGE FEATURES (use Assets, NetIncome, Cash)
# ============================================

# Return on Assets — profitability relative to size
df["roa"] = df["NetIncomeLoss"] / df["Assets"]

# Previous ROA and change
df["prev_roa"] = df["prev_NetIncomeLoss"] / df["prev_Assets"]
df["roa_change"] = df["roa"] - df["prev_roa"]

# Leverage ratio — how indebted is the company
df["leverage"] = df["Liabilities"] / df["Assets"]

# Cash to Assets — how much cash relative to size
df["cash_to_assets"] = df["CashAndCashEquivalentsAtCarryingValue"] / df["Assets"]

# Asset growth rate — rapid growth can hide problems
df["asset_growth"] = (df["Assets"] - df["prev_Assets"]) / df["prev_Assets"].abs()

# Cash vs Profit — profitable but no cash is suspicious
df["cash_profit_ratio"] = df["CashAndCashEquivalentsAtCarryingValue"] / df["NetIncomeLoss"].abs().clip(lower=1)

# Net Income volatility — big swings in profit are suspicious
df["income_change"] = (df["NetIncomeLoss"] - df["prev_NetIncomeLoss"]) / df["prev_NetIncomeLoss"].abs().clip(lower=1)

# ============================================
# GROUP 2: BENEISH RATIOS (lower coverage but high signal)
# ============================================

# DSRI — receivables growing faster than revenue
df["dsri"] = (df["AccountsReceivableNetCurrent"] / df["Revenues"]) / \
             (df["prev_AccountsReceivableNetCurrent"] / df["prev_Revenues"])

# Revenue growth
df["rev_growth"] = (df["Revenues"] - df["prev_Revenues"]) / df["prev_Revenues"].abs().clip(lower=1)

# Receivables vs Revenue divergence
df["recv_growth"] = (df["AccountsReceivableNetCurrent"] - df["prev_AccountsReceivableNetCurrent"]) / \
                     df["prev_AccountsReceivableNetCurrent"].abs().clip(lower=1)
df["rev_recv_divergence"] = df["recv_growth"] - df["rev_growth"]

# Leverage change
df["leverage_change"] = df["leverage"] - (df["prev_Liabilities"] / df["prev_Assets"])

# Accruals
df["accruals"] = (df["NetIncomeLoss"] - df["CashAndCashEquivalentsAtCarryingValue"]) / df["Assets"]

# ============================================
# CLEAN UP
# ============================================

# Replace infinities
df = df.replace([np.inf, -np.inf], np.nan)

# Drop prev_ columns
prev_cols = [c for c in df.columns if c.startswith("prev_")]
df = df.drop(columns=prev_cols)

# All feature columns
feature_cols = ["roa", "roa_change", "leverage", "cash_to_assets", "asset_growth",
                "cash_profit_ratio", "income_change", "dsri", "rev_growth",
                "recv_growth", "rev_recv_divergence", "leverage_change", "accruals"]

# Show coverage for each feature
print(f"\nFeature coverage:")
for col in feature_cols:
    valid = df[col].notna().sum()
    print(f"  {col}: {valid}/{len(df)} ({100*valid/len(df):.1f}%)")

# Fill missing feature values with median (not dropping anymore)
for col in feature_cols:
    df[col] = df[col].fillna(df[col].median())

print(f"\nFinal dataset: {df.shape[0]} rows, {df.shape[1]} columns")
df.to_csv("features_dataset.csv", index=False)
print("Saved to features_dataset.csv")