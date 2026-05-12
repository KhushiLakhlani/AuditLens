# 🔍 AuditLens

**ML-powered financial fraud detection system analyzing SEC EDGAR filings to flag accounting anomalies before regulators do.**

🔗 [Live Demo](https://auditlens.streamlit.app) | 📊 [Dataset: 500 S&P 500 Companies, 2006–2026]

---

## The Problem

Public companies manipulate financial statements for years before getting caught. Enron lied for 4 years. Wirecard for over a decade. By the time regulators act, investors have already lost billions.

AuditLens reads financial filings from the SEC and flags suspicious ones — using the same forensic accounting signals that academic research has proven detect fraud.

## Results

- **100% hit rate** — Every confirmed SEC enforcement case appeared in the model's top 20 most suspicious filings
- **XGBoost + SMOTE PR-AUC: 0.119** on severely imbalanced data (0.8% fraud rate)
- Successfully flagged GE, Boeing, Goldman Sachs, JPMorgan, Wells Fargo, Kraft Heinz, and others — all confirmed fraud cases

## How It Works

### 1. Data Pipeline
- Pulls financial data for 500 S&P 500 companies directly from the SEC EDGAR XBRL API
- Extracts 7 key financial metrics (revenue, net income, assets, liabilities, cash, receivables, inventory)
- Handles real-world data challenges: multiple XBRL tags for the same metric, period vs instant data types, filing deduplication, missing value imputation

### 2. Feature Engineering (13 Fraud Signal Features)
Based on the **Beneish M-Score** framework and forensic accounting research:
- **Return on Assets (ROA)** and year-over-year change
- **Leverage ratio** and leverage change
- **Days Sales in Receivables Index (DSRI)** — receivables growing faster than revenue signals fake sales
- **Accruals** — gap between reported profit and actual cash
- **Cash-to-profit divergence** — profitable on paper but no cash is suspicious
- **Revenue vs receivables growth divergence**
- And 7 more engineered features

### 3. Machine Learning Models

| Model | PR-AUC | Top-20 Hit Rate |
|-------|--------|-----------------|
| Logistic Regression (baseline) | 0.022 | — |
| Random Forest | 0.075 | 50% |
| **XGBoost + SMOTE** | **0.119** | **100%** |
| PyTorch Autoencoder (unsupervised) | 0.008 | 5% |

**Key insight:** XGBoost with SMOTE oversampling dominated because it learns specifically from labeled fraud cases. The autoencoder detects general anomalies (mergers, COVID impacts, industry disruptions) but not fraud-specific patterns — a meaningful finding that demonstrates the tradeoff between supervised and unsupervised approaches.

### 4. NLP Filing Analysis
Forensic linguistic analysis of actual 10-K filing text from SEC EDGAR:
- Flesch-Kincaid readability scoring
- Hedging and uncertainty language frequency
- Passive voice detection
- Sentiment analysis and sentiment volatility
- Language complexity metrics

### 5. Interactive Dashboard
Streamlit dashboard for exploring fraud risk scores across all 500 companies, with company search, historical trend visualization, and feature breakdowns.

## Tech Stack

- **Data:** SEC EDGAR XBRL API, Python (requests, pandas)
- **ML:** scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Deep Learning:** PyTorch (autoencoder)
- **NLP:** spaCy, TextBlob, NLTK
- **Dashboard:** Streamlit
- **Orchestration-ready:** Airflow-compatible pipeline architecture

## Project Structure
AuditLens/
├── get_sp500.py              # S&P 500 company list with CIK numbers
├── pull_data.py              # SEC EDGAR data pipeline
├── feature_engineering.py    # 13 fraud signal features
├── get_fraud_labels.py       # SEC enforcement case labels
├── train_model.py            # Baseline models (LR, Random Forest)
├── train_xgboost.py          # XGBoost + SMOTE (best model)
├── train_autoencoder.py      # PyTorch unsupervised anomaly detection
├── nlp_analysis.py           # Forensic linguistic analysis
├── app.py                    # Streamlit dashboard
├── scored_dataset.csv        # Final scored dataset
└── requirements.txt

## Getting Started

```bash
git clone https://github.com/KhushiLakhlani/AuditLens.git
cd AuditLens
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python pull_data.py
python feature_engineering.py
python get_fraud_labels.py
python train_xgboost.py
streamlit run app.py
```

## Author

**Khushi Lakhlani** — MS Information Systems, Northeastern University
- [LinkedIn](https://linkedin.com/in/khushilakhlani)
- [GitHub](https://github.com/KhushiLakhlani)

## Data Sources

- [SEC EDGAR XBRL API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) — Financial filing data
- [AAER Dataset](https://sites.google.com/usc.edu/aaerdataset/home) — SEC enforcement case labels
- [Beneish M-Score Research](https://scholar.google.com/scholar?q=beneish+m-score) — Feature engineering framework