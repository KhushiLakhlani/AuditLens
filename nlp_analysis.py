import requests
import re
import pandas as pd
import numpy as np
from textblob import TextBlob
import spacy

nlp = spacy.load("en_core_web_sm")

SEC_HEADERS = {
    "User-Agent": "Khushi Lakhlani khushilakhlani02@gmail.com"
}

# Words that signal evasiveness and hedging (from forensic accounting research)
HEDGE_WORDS = {"approximately", "substantially", "may", "might", "could", "possibly",
               "potentially", "generally", "largely", "primarily", "certain",
               "believe", "estimate", "anticipate", "expect", "intend",
               "appear", "suggest", "seem", "likely", "unlikely", "perhaps"}

UNCERTAINTY_WORDS = {"uncertain", "risk", "contingent", "litigation", "pending",
                     "alleged", "impairment", "restatement", "adjustment",
                     "modification", "revised", "reclassified", "restated"}

STRONG_WORDS = {"confident", "strong", "robust", "excellent", "outstanding",
                "exceptional", "record", "significant growth", "momentum",
                "solid", "remarkable", "unprecedented"}


def get_filing_text(ticker):
    """Pull latest 10-K text from SEC EDGAR"""
    
    # Get CIK
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=SEC_HEADERS)
    tickers = response.json()
    
    cik = None
    company_name = None
    for entry in tickers.values():
        if entry["ticker"].upper() == ticker.upper():
            cik = str(entry["cik_str"]).zfill(10)
            company_name = entry["title"]
            break
    
    if not cik:
        return None, None, "Not found"
    
    # Get filing history
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=SEC_HEADERS)
    submissions = response.json()
    
    recent = submissions["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            accession = recent["accessionNumber"][i].replace("-", "")
            primary_doc = recent["primaryDocument"][i]
            filing_date = recent["filingDate"][i]
            
            doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"
            doc_response = requests.get(doc_url, headers=SEC_HEADERS)
            
            # Strip HTML
            text = re.sub(r'<[^>]+>', ' ', doc_response.text)
            text = re.sub(r'&[a-zA-Z]+;', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text, company_name, filing_date
    
    return None, company_name, "No 10-K found"


def compute_readability(text):
    """Flesch-Kincaid readability — higher = harder to read = more suspicious"""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 10]
    words = text.split()
    
    if len(sentences) == 0 or len(words) == 0:
        return 0, 0
    
    avg_sentence_length = len(words) / len(sentences)
    
    # Count syllables (rough approximation)
    def count_syllables(word):
        word = word.lower()
        count = 0
        vowels = "aeiou"
        if word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        return max(count, 1)
    
    syllable_count = sum(count_syllables(w) for w in words if len(w) > 0)
    avg_syllables = syllable_count / len(words)
    
    # Flesch-Kincaid Grade Level
    fk_grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables - 15.59
    
    return fk_grade, avg_sentence_length


def compute_linguistic_features(text):
    """Extract all NLP fraud signal features from filing text"""
    
    words = text.lower().split()
    total_words = len(words)
    word_set = set(words)
    
    if total_words == 0:
        return {}
    
    # 1. READABILITY
    fk_grade, avg_sentence_len = compute_readability(text)
    
    # 2. HEDGING LANGUAGE frequency
    hedge_count = sum(1 for w in words if w in HEDGE_WORDS)
    hedge_ratio = hedge_count / total_words
    
    # 3. UNCERTAINTY LANGUAGE frequency
    uncertainty_count = sum(1 for w in words if w in UNCERTAINTY_WORDS)
    uncertainty_ratio = uncertainty_count / total_words
    
    # 4. STRONG/POSITIVE LANGUAGE frequency
    strong_count = sum(1 for w in words if w in STRONG_WORDS)
    strong_ratio = strong_count / total_words
    
    # 5. CONTRAST: hedging vs confidence (big gap = possible deception)
    hedge_confidence_gap = hedge_ratio - strong_ratio
    
    # 6. SENTIMENT (TextBlob)
    # Analyze chunks (TextBlob is slow on huge text)
    chunk_size = 10000
    sentiments = []
    for i in range(0, min(len(text), 50000), chunk_size):
        chunk = text[i:i+chunk_size]
        blob = TextBlob(chunk)
        sentiments.append(blob.sentiment.polarity)
    
    avg_sentiment = np.mean(sentiments)
    sentiment_volatility = np.std(sentiments)  # High volatility = tone shifts
    
    # 7. PASSIVE VOICE detection (using spaCy on a sample)
    sample = text[:20000]
    doc = nlp(sample)
    passive_count = 0
    total_verbs = 0
    for token in doc:
        if token.dep_ in ("nsubjpass", "auxpass"):
            passive_count += 1
        if token.pos_ == "VERB":
            total_verbs += 1
    passive_ratio = passive_count / max(total_verbs, 1)
    
    # 8. WORD COMPLEXITY — average word length
    avg_word_length = np.mean([len(w) for w in words])
    
    # 9. UNIQUE WORDS ratio (low = repetitive/formulaic)
    unique_ratio = len(set(words)) / total_words
    
    return {
        "fk_grade_level": round(fk_grade, 2),
        "avg_sentence_length": round(avg_sentence_len, 2),
        "hedge_ratio": round(hedge_ratio, 6),
        "uncertainty_ratio": round(uncertainty_ratio, 6),
        "strong_language_ratio": round(strong_ratio, 6),
        "hedge_confidence_gap": round(hedge_confidence_gap, 6),
        "avg_sentiment": round(avg_sentiment, 4),
        "sentiment_volatility": round(sentiment_volatility, 4),
        "passive_voice_ratio": round(passive_ratio, 4),
        "avg_word_length": round(avg_word_length, 2),
        "unique_word_ratio": round(unique_ratio, 4),
        "total_words": total_words
    }


def analyze_company(ticker):
    """Full NLP analysis for one company"""
    
    print(f"\n{'='*60}")
    print(f"  AUDITLENS NLP ANALYSIS: {ticker.upper()}")
    print(f"{'='*60}")
    
    # Get ML score
    scores = pd.read_csv("scored_dataset.csv")
    company_scores = scores[scores["ticker"] == ticker.upper()].sort_values("year", ascending=False)
    
    financial_score = 0.5
    if len(company_scores) > 0:
        latest = company_scores.iloc[0]
        financial_score = latest.get("fraud_score", 0.5)
        print(f"\n  Financial Model Score: {financial_score:.4f} (Year: {int(latest['year'])})")
    
    # Pull filing
    print(f"  Pulling 10-K from SEC EDGAR...")
    text, company_name, filing_date = get_filing_text(ticker)
    
    if text is None:
        print(f"  Error: {filing_date}")
        return None
    
    print(f"  Company: {company_name}")
    print(f"  Filing Date: {filing_date}")
    print(f"  Document Size: {len(text):,} characters")
    
    # NLP analysis
    print(f"  Running NLP analysis...")
    features = compute_linguistic_features(text)
    
    # Display results
    print(f"\n  LINGUISTIC RED FLAGS:")
    print(f"  {'—'*40}")
    print(f"  Readability (FK Grade):    {features['fk_grade_level']} {'⚠️  COMPLEX' if features['fk_grade_level'] > 14 else '✓ Normal'}")
    print(f"  Avg Sentence Length:       {features['avg_sentence_length']} words")
    print(f"  Hedging Language:          {features['hedge_ratio']:.4%} {'⚠️  HIGH' if features['hedge_ratio'] > 0.02 else '✓ Normal'}")
    print(f"  Uncertainty Language:      {features['uncertainty_ratio']:.4%} {'⚠️  HIGH' if features['uncertainty_ratio'] > 0.005 else '✓ Normal'}")
    print(f"  Passive Voice:             {features['passive_voice_ratio']:.2%} {'⚠️  HIGH' if features['passive_voice_ratio'] > 0.15 else '✓ Normal'}")
    print(f"  Sentiment:                 {features['avg_sentiment']:.4f}")
    print(f"  Sentiment Volatility:      {features['sentiment_volatility']:.4f} {'⚠️  UNSTABLE' if features['sentiment_volatility'] > 0.05 else '✓ Stable'}")
    print(f"  Hedge vs Confidence Gap:   {features['hedge_confidence_gap']:.4%} {'⚠️  EVASIVE' if features['hedge_confidence_gap'] > 0.015 else '✓ Normal'}")
    
    # Compute NLP risk score
    nlp_score = 0
    if features["fk_grade_level"] > 14: nlp_score += 0.2
    if features["hedge_ratio"] > 0.02: nlp_score += 0.2
    if features["uncertainty_ratio"] > 0.005: nlp_score += 0.15
    if features["passive_voice_ratio"] > 0.15: nlp_score += 0.15
    if features["sentiment_volatility"] > 0.05: nlp_score += 0.15
    if features["hedge_confidence_gap"] > 0.015: nlp_score += 0.15
    
    # Combined score
    combined = (0.6 * financial_score) + (0.4 * nlp_score)
    
    print(f"\n  {'='*40}")
    print(f"  COMBINED RISK ASSESSMENT")
    print(f"  {'='*40}")
    print(f"  Financial Model:  {financial_score:.4f}")
    print(f"  NLP Analysis:     {nlp_score:.4f}")
    print(f"  Combined Score:   {combined:.4f}")
    
    if combined > 0.6:
        print(f"  Verdict:          HIGH RISK")
    elif combined > 0.3:
        print(f"  Verdict:          MEDIUM RISK")
    else:
        print(f"  Verdict:          LOW RISK")
    
    return {"ticker": ticker, "financial_score": financial_score,
            "nlp_score": nlp_score, "combined": combined, "features": features}


if __name__ == "__main__":
    # Compare a known fraud company vs a clean one
    result_ge = analyze_company("GE")
    result_aapl = analyze_company("AAPL")
    
    print(f"\n\n{'='*60}")
    print(f"  SIDE BY SIDE COMPARISON")
    print(f"{'='*60}")
    if result_ge and result_aapl:
        print(f"  {'Metric':<30} {'GE (fraud)':<15} {'AAPL (clean)':<15}")
        print(f"  {'—'*60}")
        for key in result_ge["features"]:
            ge_val = result_ge["features"][key]
            aapl_val = result_aapl["features"][key]
            print(f"  {key:<30} {str(ge_val):<15} {str(aapl_val):<15}")
        print(f"  {'—'*60}")
        print(f"  {'Combined Risk Score':<30} {result_ge['combined']:<15.4f} {result_aapl['combined']:<15.4f}")