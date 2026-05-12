import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="AuditLens", page_icon="🔍", layout="wide")

st.title("🔍 AuditLens")
st.markdown("**ML-Powered Financial Fraud Detection from SEC EDGAR Filings**")
st.markdown("---")

@st.cache_data
def load_data():
    df = pd.read_csv("scored_dataset.csv")
    return df

df = load_data()

# Sidebar
# Sidebar
st.sidebar.header("Search")

all_tickers = sorted(df["ticker"].unique())

# Highlight interesting companies at the top
flagged = df[df["fraud_score"] > 0.9]["ticker"].unique().tolist()
clean = ["AAPL", "MSFT", "GOOGL", "AMZN"]

st.sidebar.markdown("**🚨 High Risk Companies**")
quick_pick = st.sidebar.selectbox(
    "Quick pick",
    options=[""] + sorted(flagged),
    format_func=lambda x: "Select a flagged company..." if x == "" else x
)

st.sidebar.markdown("**🔎 Search Any Company**")
search_pick = st.sidebar.selectbox(
    "All S&P 500",
    options=[""] + all_tickers,
    format_func=lambda x: "Select from full list..." if x == "" else f"{x} — {df[df['ticker']==x]['company'].iloc[0]}" if x != "" else x
)

# Use whichever they picked
ticker_input = (quick_pick or search_pick or "GE").upper()

company_data = df[df["ticker"] == ticker_input].sort_values("year")

if len(company_data) == 0:
    st.error(f"Ticker '{ticker_input}' not found in dataset.")
    st.stop()

company_name = company_data["company"].iloc[0]
latest = company_data.iloc[-1]
fraud_score = latest.get("fraud_score", 0)

# Header metrics
st.header(f"{company_name} ({ticker_input})")
col1, col2, col3, col4 = st.columns(4)

with col1:
    risk_label = "🔴 HIGH" if fraud_score > 0.7 else "🟡 MEDIUM" if fraud_score > 0.3 else "🟢 LOW"
    st.metric("Risk Level", risk_label)
with col2:
    st.metric("Fraud Score", f"{fraud_score:.4f}")
with col3:
    st.metric("Latest Year", int(latest["year"]))
with col4:
    st.metric("Years of Data", len(company_data))

st.markdown("---")

# Fraud score over time
st.subheader("Fraud Risk Score Over Time")
if "fraud_score" in company_data.columns:
    chart_data = company_data[["year", "fraud_score"]].set_index("year")
    st.line_chart(chart_data)

# Key financial metrics
st.subheader("Key Financial Metrics")
metric_cols = ["Revenues", "NetIncomeLoss", "Assets", "Liabilities",
               "CashAndCashEquivalentsAtCarryingValue"]
available_metrics = [c for c in metric_cols if c in company_data.columns]

if available_metrics:
    fig_data = company_data[["year"] + available_metrics].set_index("year")
    rename_map = {
        "Revenues": "Revenue",
        "NetIncomeLoss": "Net Income",
        "Assets": "Total Assets",
        "Liabilities": "Total Liabilities",
        "CashAndCashEquivalentsAtCarryingValue": "Cash"
    }
    fig_data = fig_data.rename(columns=rename_map)
    st.line_chart(fig_data)

# Fraud signal features
st.subheader("Fraud Signal Features (Latest Year)")
feature_cols = ["roa", "roa_change", "leverage", "cash_to_assets", "asset_growth",
                "cash_profit_ratio", "income_change", "accruals", "leverage_change"]
available_features = [c for c in feature_cols if c in latest.index]

col1, col2, col3 = st.columns(3)
for i, feat in enumerate(available_features):
    val = latest[feat]
    with [col1, col2, col3][i % 3]:
        display_name = feat.replace("_", " ").title()
        st.metric(display_name, f"{val:.4f}" if pd.notna(val) else "N/A")

# Top 20 most suspicious
st.markdown("---")
st.subheader("🚨 Top 20 Most Suspicious Filings (All Companies)")
if "fraud_score" in df.columns:
    top20 = df.nlargest(20, "fraud_score")[["company", "ticker", "year", "fraud_score", "is_fraud"]]
    top20["is_fraud"] = top20["is_fraud"].map({1: "✅ Confirmed", 0: "❌ Not confirmed"})
    top20 = top20.rename(columns={
        "company": "Company",
        "ticker": "Ticker",
        "year": "Year",
        "fraud_score": "Fraud Score",
        "is_fraud": "SEC Enforcement"
    })
    st.dataframe(top20, use_container_width=True, hide_index=True)

# About
st.markdown("---")
st.markdown("""
### About AuditLens
AuditLens analyzes SEC EDGAR financial filings using machine learning to detect potential accounting fraud.

**Data:** 500 S&P 500 companies, 8,900+ company-year records (2006-2026) from SEC EDGAR API  
**Models:** XGBoost with SMOTE (supervised), PyTorch Autoencoder (unsupervised), NLP Text Analysis  
**Features:** 13 engineered fraud signal features based on Beneish M-Score research  
**Results:** 100% hit rate identifying confirmed SEC enforcement cases in top 20 predictions  

Built by Khushi Lakhlani | [GitHub](https://github.com/KhushiLakhlani/AuditLens)
""")