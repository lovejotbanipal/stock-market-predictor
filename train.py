from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report
)

DATA_PATH = Path("data/all_stocks_5yr.csv")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "random_forest.joblib"
META_PATH = MODEL_DIR / "metadata.json"
FEATURE_DATA_PATH = MODEL_DIR / "feature_data.csv"

FEATURES = [
    "return_1d",
    "return_2d",
    "return_5d",
    "range_pct",
    "open_close_pct",
    "ma5_ratio",
    "ma10_ratio",
    "ma20_ratio",
    "volatility_5d",
    "volatility_10d",
    "volume_ratio_5d",
    "volume_change_1d",
    "rsi_14",
]

def normalize_columns(df):
    df.columns = [c.strip().lower() for c in df.columns]
    if "name" not in df.columns:
        raise ValueError("Dataset must contain a Name/ticker column.")
    return df.rename(columns={"name": "ticker"})

def add_features(group):
    g = group.sort_values("date").copy()
    close = g["close"]
    volume = g["volume"].replace(0, np.nan)

    g["return_1d"] = close.pct_change(1)
    g["return_2d"] = close.pct_change(2)
    g["return_5d"] = close.pct_change(5)
    g["range_pct"] = (g["high"] - g["low"]) / close
    g["open_close_pct"] = (close - g["open"]) / g["open"]

    for n in (5, 10, 20):
        g[f"ma{n}_ratio"] = close / close.rolling(n).mean() - 1

    g["volatility_5d"] = g["return_1d"].rolling(5).std()
    g["volatility_10d"] = g["return_1d"].rolling(10).std()
    g["volume_ratio_5d"] = volume / volume.rolling(5).mean()
    g["volume_change_1d"] = volume.pct_change()

    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gains / losses.replace(0, np.nan)
    g["rsi_14"] = 100 - (100 / (1 + rs))

    # Label at date t uses only whether close(t+1) > close(t).
    next_close = close.shift(-1)
    g["target"] = np.where(next_close.notna(), (next_close > close).astype(int), np.nan)
    return g

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Download all_stocks_5yr.csv from the Kaggle "
            "S&P 500 stock data dataset and put it in data/."
        )

    df = pd.read_csv(DATA_PATH)
    df = normalize_columns(df)
    required = {"date", "open", "high", "low", "close", "volume", "ticker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    parts = [add_features(g) for _, g in df.groupby("ticker", sort=False)]
    featured = pd.concat(parts, ignore_index=True)
    featured = featured.replace([np.inf, -np.inf], np.nan)
    featured = featured.dropna(subset=FEATURES + ["target"]).copy()
    featured["target"] = featured["target"].astype(int)

    # Global chronological split: newest 20% of dates are held out.
    dates = np.array(sorted(featured["date"].unique()))
    cutoff = dates[int(len(dates) * 0.80)]
    train = featured[featured["date"] < cutoff].copy()
    test = featured[featured["date"] >= cutoff].copy()

    X_train, y_train = train[FEATURES], train["target"]
    X_test, y_test = test[FEATURES], test["target"]

    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=12,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_end": str(pd.Timestamp(cutoff).date()),
        "test_start": str(pd.Timestamp(cutoff).date()),
    }

    importances = sorted(
        [{"feature": f, "importance": float(v)}
         for f, v in zip(FEATURES, model.feature_importances_)],
        key=lambda x: x["importance"],
        reverse=True,
    )

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # Keep engineered history so the API can predict any ticker without recomputing.
    keep = ["date", "ticker", "close"] + FEATURES
    featured[keep].sort_values(["ticker", "date"]).to_csv(FEATURE_DATA_PATH, index=False)

    metadata = {
        "features": FEATURES,
        "metrics": metrics,
        "feature_importance": importances,
        "tickers": sorted(featured["ticker"].unique().tolist()),
        "model": "RandomForestClassifier",
        "target": "next_trading_day_close_is_higher",
    }
    META_PATH.write_text(json.dumps(metadata, indent=2))

    print("\nModel trained successfully")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:")
    print(classification_report(y_test, pred, digits=4))
    print("\nTop feature importances:")
    for row in importances[:8]:
        print(f"{row['feature']:20s} {row['importance']:.4f}")

if __name__ == "__main__":
    main()
