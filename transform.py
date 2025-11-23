import pandas as pd

def transform():
    df = pd.read_csv("stage_raw.csv")

    # Normalize column names
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]

    # Ensure adj_close exists
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    # Convert numeric cols to floats
    numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Feature Engineering (handled per ticker)
    df["daily_return"] = df.groupby("ticker")["adj_close"].pct_change()
    df["price_change"] = df["close"] - df["open"]

    # Drop rows with missing or invalid values
    df = df.dropna(subset=["adj_close", "open", "close", "daily_return"])

    df.to_csv("stage_clean.csv", index=False)
    print("Saved stage_clean.csv")

if __name__ == "__main__":
    transform()
