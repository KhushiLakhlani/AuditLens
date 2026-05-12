import pandas as pd
import numpy as np

df = pd.read_csv("financial_data_clean.csv")
print(f"Loaded: {df.shape[0]} rows")

# Drop rows missing the essentials — need at least Assets and NetIncome
df = df.dropna(subset=["Assets", "NetIncomeLoss"])
print(f"After dropping rows without Assets/NetIncome: {df.shape[0]} rows")

# Sort so we can calculate year-over-year changes
df = df.sort_values(["company", "year"]).reset_index(drop=True)

# ============================================
# FRAUD SIGNAL FEATURES (Beneish M-Score ratios)
# ============================================

# For each company, get prior year values
df["prev_revenues"] = df.groupby("company")["Revenues"].shift(1)
df["prev_receivables"] = df.groupby("company")["AccountsReceivableNetCurrent"].shift(1)
df["prev_assets"] = df.groupby("company")["Assets"].shift(1)
df["prev_net_income"] = df.groupby("company")["NetIncomeLoss"].shift(1)
df["prev_liabilities"] = df.groupby("company")["Liabilities"].shift(1)
df["prev_inventory"] = df.groupby("company")["InventoryNet"].shift(1)
df["prev_cash"] = df.groupby("company")["CashAndCashEquivalentsAtCarryingValue"].shift(1)

# --- FRAUD SIGNAL 1: Days Sales in Receivables Index (DSRI) ---
# If receivables grow faster than revenue, could mean fake sales
df["dsri"] = (df["AccountsReceivableNetCurrent"] / df["Revenues"]) / \
             (df["prev_receivables"] / df["prev_revenues"])

# --- FRAUD SIGNAL 2: Asset Quality Index (AQI) ---
# If non-current assets grow, company may be capitalizing expenses
hard_assets = df["AccountsReceivableNetCurrent"].fillna(0) + \
              df["InventoryNet"].fillna(0) + \
              df["CashAndCashEquivalentsAtCarryingValue"].fillna(0)
prev_hard = df["prev_receivables"].fillna(0) + \
            df["prev_inventory"].fillna(0) + \
            df["prev_cash"].fillna(0)
df["aqi"] = (1 - hard_assets / df["Assets"]) / \
            (1 - prev_hard / df["prev_assets"])

# --- FRAUD SIGNAL 3: Revenue Growth Index (SGI) ---
# Rapid revenue growth creates pressure to keep growing — motive for fraud
df["sgi"] = df["Revenues"] / df["prev_revenues"]

# --- FRAUD SIGNAL 4: Leverage Index (LVGI) ---
# Increasing debt creates pressure — another fraud motive
df["lvgi"] = (df["Liabilities"] / df["Assets"]) / \
             (df["prev_liabilities"] / df["prev_assets"])

# --- FRAUD SIGNAL 5: Accruals to Assets ---
# High accruals = big gap between reported profit and actual cash
df["accruals"] = (df["NetIncomeLoss"] - df["CashAndCashEquivalentsAtCarryingValue"]) / df["Assets"]

# --- FRAUD SIGNAL 6: Cash vs Profit Divergence ---
# Company reports profit but has no cash? Suspicious.
df["cash_profit_gap"] = df["CashAndCashEquivalentsAtCarryingValue"] / df["NetIncomeLoss"].abs()

# --- FRAUD SIGNAL 7: Revenue Growth vs Receivables Growth ---
df["rev_growth"] = (df["Revenues"] - df["prev_revenues"]) / df["prev_revenues"].abs()
df["recv_growth"] = (df["AccountsReceivableNetCurrent"] - df["prev_receivables"]) / df["prev_receivables"].abs()
df["rev_recv_divergence"] = df["recv_growth"] - df["rev_growth"]

# Clean up: replace infinities with NaN
df = df.replace([np.inf, -np.inf], np.nan)

# Drop the "prev_" helper columns
prev_cols = [c for c in df.columns if c.startswith("prev_")]
df = df.drop(columns=prev_cols)

# Show results
feature_cols = ["dsri", "aqi", "sgi", "lvgi", "accruals", 
                "cash_profit_gap", "rev_growth", "recv_growth", "rev_recv_divergence"]

print(f"\nFinal dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nFraud signal features — summary stats:")
print(df[feature_cols].describe().round(3))

df.to_csv("features_dataset.csv", index=False)
print("\nSaved to features_dataset.csv")