# StockSense — Full-Stack Random Forest Stock Direction Predictor

A portfolio-ready ML web app that predicts whether a stock is more likely to close
UP or DOWN on its next trading day using historical OHLCV data.

> Educational project only. This is not financial advice and is not a trading system.

## Stack
- Machine learning: Python, pandas, scikit-learn RandomForestClassifier
- Backend: FastAPI
- Frontend: HTML/CSS/vanilla JavaScript
- Model persistence: joblib
- Dataset: Kaggle "S&P 500 stock data" by Cam Nugent

Dataset:
https://www.kaggle.com/datasets/camnugent/sandp500

## 1. Download the Kaggle dataset

Download `all_stocks_5yr.csv` and place it here:

    data/all_stocks_5yr.csv

Expected columns:
`date, open, high, low, close, volume, Name`

## 2. Create a virtual environment

    python -m venv .venv

macOS/Linux:

    source .venv/bin/activate

Windows:

    .venv\Scripts\activate

Then:

    pip install -r requirements.txt

## 3. Train the model

    python train.py

This:
- sorts each ticker chronologically
- engineers return/momentum/volatility/moving-average/volume features
- creates a next-day UP/DOWN target
- holds out the newest 20% of observations
- trains a RandomForestClassifier
- reports accuracy, precision, recall, F1 and ROC-AUC
- saves the model and metadata in `model/`

## 4. Run the full-stack app

    uvicorn backend.main:app --reload

Open:

    http://127.0.0.1:8000

## API

### GET /api/health
Checks whether the trained model is available.

### GET /api/tickers
Returns tickers found in the trained dataset.

### POST /api/predict
Example body:

    {
      "ticker": "AAPL"
    }

Returns:
- predicted direction
- probability of UP
- confidence
- latest close
- latest feature values
- model test metrics

## Modeling notes

The model predicts direction, not an exact future price. This is deliberately a
classification problem because an exact next-day price forecast can imply more
precision than a simple historical-price model can justify.

The test set is chronological. Do not randomly shuffle financial time-series data:
doing so can train on observations from the future and evaluate on observations
from the past.

For a stronger v2:
- walk-forward validation
- probability calibration
- transaction-cost-aware backtesting
- macro/market-index features
- news/sentiment features
- live market-data provider
- PostgreSQL user/watchlist persistence
- React/Next.js frontend
