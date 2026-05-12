import requests
import pandas as pd
import time

headers = {
    "User-Agent": "Khushi Lakhlani khushilakhlani02@gmail.com"
}

metrics = ["Revenues", "NetIncomeLoss", "Assets", "Liabilities",
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

sp500 = pd.read_csv("sp500_companies.csv", dtype={"cik_str": str})
# Ensure CIK is padded to 10 digits
sp500["cik_str"] = sp500["cik_str"].str.zfill(10)
print(f"Pulling data for {len(sp500)} companies...")
print(f"Sample CIK: {sp500['cik_str'].iloc[0]}")  # should look like 0001045810

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
    
    # Print progress every 50 companies
    if (i + 1) % 50 == 0:
        print(f"  Pulled {i + 1}/{len(sp500)} companies... ({len(all_rows)} records so far)")
    
    time.sleep(0.15)  # SEC rate limit: 10 requests/sec

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

df.to_csv("raw_financial_data.csv", index=False)
df_pivot.to_csv("financial_data_clean.csv", index=False)
print("\nSaved both CSVs!")