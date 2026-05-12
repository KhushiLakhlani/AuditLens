import requests
import pandas as pd

headers = {
    "User-Agent": "Khushi Lakhlani khushilakhlani02@gmail.com"
}

# SEC's master list of all companies with tickers and CIK numbers
url = "https://www.sec.gov/files/company_tickers.json"
response = requests.get(url, headers=headers)
tickers_data = response.json()

# Convert to dataframe
df = pd.DataFrame.from_dict(tickers_data, orient="index")
print(f"Total companies in SEC database: {len(df)}")

# Fetch S&P 500 list from Wikipedia with proper headers
wiki_headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
}
wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
wiki_response = requests.get(wiki_url, headers=wiki_headers)
sp500_tables = pd.read_html(wiki_response.text)
sp500 = sp500_tables[0]
sp500_tickers = set(sp500["Symbol"].str.upper().str.replace(".", "-"))

print(f"S&P 500 companies: {len(sp500_tickers)}")

# Match S&P 500 tickers to CIK numbers
df["ticker"] = df["ticker"].str.upper()
sp500_ciks = df[df["ticker"].isin(sp500_tickers)].copy()

# Pad CIK numbers with leading zeros (SEC needs 10 digits)
sp500_ciks["cik_str"] = sp500_ciks["cik_str"].astype(str).str.zfill(10)

print(f"Matched: {len(sp500_ciks)} companies")
print(sp500_ciks.head(10))

sp500_ciks.to_csv("sp500_companies.csv", index=False)
print("\nSaved to sp500_companies.csv")