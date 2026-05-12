import requests
import pandas as pd
import time

headers = {
    "User-Agent": "Khushi Lakhlani khushilakhlani02@gmail.com"
}

# Multiple names companies use for revenue in their filings
revenue_tags = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "SalesRevenueNet", "SalesRevenueGoodsNet"]

# Other financial metrics (revenue handled separately above)
metrics = ["NetIncomeLoss", "Assets", "Liabilities",
           "CashAndCashEquivalentsAtCarryingValue",
           "AccountsReceivableNetCurrent", "InventoryNet"]

def get_company_facts(cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return None
    return response.json()

def extract_annual_data(data, company_name, ticker):
    rows = []

    # Handle revenue separately — try multiple tag names
    for tag in revenue_tags:
        try:
            entries = data["facts"]["us-gaap"][tag]["units"]["USD"]
            for entry in entries:
                if entry["form"] != "10-K":
                    continue
                end = pd.to_datetime(entry["end"])
                if "start" in entry:
                    start = pd.to_datetime(entry["start"])
                    days = (end - start).days
                    if days < 350 or days > 380:
                        continue
                rows.append({
                    "company": company_name,
                    "ticker": ticker,
                    "metric": "Revenues",
                    "value": entry["val"],
                    "year": end.year,
                    "period_end": entry["end"],
                    "filed": entry["filed"]
                })
            if rows:
                break
        except KeyError:
            continue

    # Handle all other metrics normally
    for metric in metrics:
        try:
            entries = data["facts"]["us-gaap"][metric]["units"]["USD"]
            for entry in entries:
                if entry["form"] != "10-K":
                    continue
                end = pd.to_datetime(entry["end"])
                if "start" in entry:
                    start = pd.to_datetime(entry["start"])
                    days = (end - start).days
                    if days < 350 or days > 380:
                        continue
                rows.append({
                    "company": company_name,
                    "ticker": ticker,
                    "metric": metric,
                    "value": entry["val"],
                    "year": end.year,
                    "period_end": entry["end"],
                    "filed": entry["filed"]
                })
        except KeyError:
            continue
    return rows

# Load S&P 500 list
sp500 = pd.read_csv("sp500_companies.csv", dtype={"cik_str": str})
sp500["cik_str"] = sp500["cik_str"].str.zfill(10)
print(f"Pulling data for {len(sp500)} companies...")
print(f"Sample CIK: {sp500['cik_str'].iloc[0]}")

all_rows = []
failed = []

for i, row in sp500.iterrows():
    cik = row["cik_str"]
    name = row["title"]
    ticker = row["ticker"]

    try:
        data = get_company_facts(cik)
        if data is None:
            failed.append(name)
            continue
        rows = extract_annual_data(data, name, ticker)
        all_rows.extend(rows)
    except Exception as e:
        failed.append(name)
        continue

    if (i + 1) % 50 == 0:
        print(f"  Pulled {i + 1}/{len(sp500)} companies... ({len(all_rows)} records so far)")

    time.sleep(0.15)

print(f"\nDone! Total records: {len(all_rows)}")
print(f"Failed: {len(failed)} companies")

# Create dataframe
df = pd.DataFrame(all_rows)

# Deduplicate
df = df.sort_values("filed").drop_duplicates(
    subset=["company", "metric", "year"],
    keep="last"
)

# Pivot
df_pivot = df.pivot_table(
    index=["company", "ticker", "year"],
    columns="metric",
    values="value"
).reset_index()
df_pivot.columns.name = None

print(f"\nFinal dataset: {df_pivot.shape[0]} rows, {df_pivot.shape[1]} columns")
print(f"Year range: {df_pivot['year'].min()} to {df_pivot['year'].max()}")
print(f"Companies: {df_pivot['company'].nunique()}")

# Show missing values
print(f"\nMissing values:")
for col in df_pivot.columns:
    missing = df_pivot[col].isnull().sum()
    if missing > 0:
        print(f"  {col}: {missing} ({100*missing/len(df_pivot):.1f}%)")

df.to_csv("raw_financial_data.csv", index=False)
df_pivot.to_csv("financial_data_clean.csv", index=False)
print("\nSaved both CSVs!")