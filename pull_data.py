import requests
import pandas as pd
import time

headers = {
    "User-Agent": "Khushi Lakhlani khushilakhlani02@gmail.com"
}

companies = {
    "0000320193": "Apple",
    "0000789019": "Microsoft",
    "0001318605": "Tesla",
    "0000050863": "Intel",
    "0000200406": "Johnson & Johnson"
}

metrics = ["Revenues", "NetIncomeLoss", "Assets", "Liabilities",
           "CashAndCashEquivalentsAtCarryingValue",
           "AccountsReceivableNetCurrent", "InventoryNet"]

def get_company_facts(cik):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    response = requests.get(url, headers=headers)
    return response.json()

def extract_annual_data(data, company_name):
    rows = []
    for metric in metrics:
        try:
            entries = data["facts"]["us-gaap"][metric]["units"]["USD"]
            for entry in entries:
                if entry["form"] != "10-K":
                    continue
                
                end = pd.to_datetime(entry["end"])
                
                if "start" in entry:
                    # Period metric (revenue, net income) — keep full year only
                    start = pd.to_datetime(entry["start"])
                    days = (end - start).days
                    if days < 350 or days > 380:
                        continue
                
                # If no "start", it's a balance sheet snapshot — keep it
                rows.append({
                    "company": company_name,
                    "metric": metric,
                    "value": entry["val"],
                    "year": end.year,
                    "period_end": entry["end"],
                    "filed": entry["filed"]
                })
        except KeyError:
            continue
    return rows

# Pull data for all companies
all_rows = []
for cik, name in companies.items():
    print(f"Pulling data for {name}...")
    data = get_company_facts(cik)
    rows = extract_annual_data(data, name)
    all_rows.extend(rows)
    time.sleep(0.2)

# Create dataframe
df = pd.DataFrame(all_rows)

# Remove duplicates — keep the most recently filed number
df = df.sort_values("filed").drop_duplicates(
    subset=["company", "metric", "year"],
    keep="last"
)
df = df.sort_values(["company", "year", "metric"]).reset_index(drop=True)

print(f"\nTotal records: {len(df)}")
print(f"\nCompanies: {df['company'].unique()}")
print(f"\nSample data:")
print(df.head(10))


# Pivot: one row per company per year, metrics as columns
df_pivot = df.pivot_table(
    index=["company", "year"],
    columns="metric",
    values="value"
).reset_index()

# Flatten column names
df_pivot.columns.name = None

print(f"\nPivoted shape: {df_pivot.shape}")
print(f"\nColumns: {list(df_pivot.columns)}")
print(f"\n{df_pivot.head()}")

df_pivot.to_csv("financial_data_clean.csv", index=False)
print("\nSaved clean data to financial_data_clean.csv")

