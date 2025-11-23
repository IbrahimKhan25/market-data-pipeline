import yfinance as yf
import pandas as pd
from datetime import datetime

def extract_stock_data(tickers):
    all_data = []

    for ticker in tickers:
        df = yf.download(ticker, period="1mo", interval="1d")
        df = df.reset_index()

        df["ticker"] = ticker
        df["extracted_at"] = datetime.utcnow()

        all_data.append(df)

    final_df = pd.concat(all_data)
    return final_df

if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "TSLA"]
    df = extract_stock_data(tickers)
    df.to_csv("stage_raw.csv", index=False)
    print("Saved stage_raw.csv")
