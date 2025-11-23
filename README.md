# Data-Pipeline-Project
# Financial Market ETL Pipeline (Python + Azure)

This project is an end-to-end ETL pipeline that ingests daily stock market data from a public API, transforms it into an analytics-ready format, and loads it into Azure Blob Storage for downstream analytics.

## 🚀 Tech Stack

- **Language:** Python
- **Data Source:** Yahoo Finance (`yfinance`)
- **Processing:** `pandas`
- **Cloud Storage:** Azure Blob Storage
- **File Formats:** CSV (raw + cleaned)

## 🧠 Problem

Analysts and data teams often need fresh, reliable stock market data to support reporting, dashboards, and quantitative analysis. Manually downloading CSVs or copy-pasting from websites is slow, inconsistent, and not reproducible.

This pipeline automates that workflow.

## 🏗️ Pipeline Overview

1. **Extract (`extract.py`)**
   - Uses `yfinance` to pull daily OHLCV data for multiple tickers (e.g. AAPL, MSFT, TSLA) over the last 30 days.
   - Normalizes the structure and adds:
     - `ticker`
     - `extracted_at` (UTC timestamp)
   - Saves the raw data as `stage_raw.csv`.

2. **Transform (`transform.py`)**
   - Loads `stage_raw.csv` and standardizes column names.
   - Ensures pricing fields are numeric (handles string/CSV type issues).
   - Computes:
     - `daily_return` (per ticker, using adjusted close)
     - `price_change` (`close - open`)
   - Drops invalid/incomplete rows.
   - Writes the cleaned dataset to `stage_clean.csv`.

3. **Load (`load_azure.py`)**
   - Uploads `stage_clean.csv` to **Azure Blob Storage**.
   - Targets a specified container (e.g. `market-data`) in a given Storage Account.
   - Overwrites the blob on each run to keep a fresh version of the cleaned dataset in the cloud.

## 📂 File Structure

```text
market-data-pipeline/
├── extract.py          # Extract stock data from Yahoo Finance
├── transform.py        # Clean data and engineer features
├── load_azure.py       # Load cleaned data into Azure Blob Storage
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
