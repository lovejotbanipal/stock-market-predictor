from pathlib import Path
import re
import json

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

STOCK_UNIVERSE_PATH = BASE_DIR / "data" / "stock_universe.json"


def load_stock_universe():
    if not STOCK_UNIVERSE_PATH.exists():
        return []

    with open(
        STOCK_UNIVERSE_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


STOCK_UNIVERSE = load_stock_universe()

FRONTEND_DIR = BASE_DIR / "frontend"

MODEL_PATH = BASE_DIR / "model" / "v2" / "model.joblib"
METADATA_PATH = BASE_DIR / "model" / "v2" / "metadata.json"

BACKTEST_SUMMARY_PATH = (
    BASE_DIR
    / "model"
    / "v2"
    / "backtests"
    / "aapl_walk_forward_summary.json"
)

BACKTEST_CSV_PATH = (
    BASE_DIR
    / "model"
    / "v2"
    / "backtests"
    / "aapl_walk_forward.csv"
)


# --------------------------------------------------
# App
# --------------------------------------------------

app = FastAPI(
    title="StockSense API",
    version="2.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


app.mount(
    "/static",
    StaticFiles(
        directory=FRONTEND_DIR
    ),
    name="static",
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise RuntimeError(
        "V2 model not found. "
        "Run train_v2.py first."
    )


if not METADATA_PATH.exists():
    raise RuntimeError(
        "V2 metadata not found."
    )


model = joblib.load(
    MODEL_PATH
)


with open(
    METADATA_PATH
) as file:
    metadata = json.load(file)


FEATURES = metadata["features"]


# --------------------------------------------------
# Walk-forward validation results
#
# These come from evaluate_v2.py.
# They are fixed historical evaluation results,
# not live predictions.
# --------------------------------------------------

WALK_FORWARD_VALIDATION = [
    {
        "year": 2022,
        "accuracy": 0.511554,
        "roc_auc": 0.514477,
        "f1": 0.549154,
    },
    {
        "year": 2023,
        "accuracy": 0.501467,
        "roc_auc": 0.501776,
        "f1": 0.515234,
    },
    {
        "year": 2024,
        "accuracy": 0.510317,
        "roc_auc": 0.513730,
        "f1": 0.512253,
    },
    {
        "year": 2025,
        "accuracy": 0.505672,
        "roc_auc": 0.508245,
        "f1": 0.491767,
    },
    {
        "year": 2026,
        "accuracy": 0.503986,
        "roc_auc": 0.504817,
        "f1": 0.446869,
    },
]


VALIDATION_AVERAGE = {
    "accuracy": 0.506599,
    "roc_auc": 0.508609,
    "f1": 0.503055,
}


# --------------------------------------------------
# Request body
# --------------------------------------------------

class PredictionRequest(BaseModel):
    ticker: str


# --------------------------------------------------
# Data download
# --------------------------------------------------

def download_prices(
    ticker: str,
    period: str = "1y",
):
    # Yahoo Finance represents share classes with a hyphen
    # instead of the exchange-style period.
    yahoo_ticker = ticker.strip().upper().replace(".", "-")

    df = yf.download(
        yahoo_ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )


    if df.empty:
        raise ValueError(
            f"No market data found for {ticker}."
        )


    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):
        df.columns = (
            df.columns
            .get_level_values(0)
        )


    df.columns = [
        str(column).lower()
        for column in df.columns
    ]


    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


    missing = [
        column
        for column in required
        if column not in df.columns
    ]


    if missing:
        raise ValueError(
            "Market data is missing "
            f"columns: {missing}"
        )


    return df[
        required
    ].copy()


# --------------------------------------------------
# RSI
# --------------------------------------------------

def calculate_rsi(
    close,
    window=14,
):

    delta = close.diff()


    gains = (
        delta
        .clip(lower=0)
        .rolling(window)
        .mean()
    )


    losses = (
        -delta
        .clip(upper=0)
        .rolling(window)
        .mean()
    )


    rs = (
        gains
        /
        losses.replace(
            0,
            np.nan,
        )
    )


    return (
        100
        -
        (
            100
            /
            (1 + rs)
        )
    )


# --------------------------------------------------
# Feature engineering
#
# This intentionally mirrors train_v2.py.
# --------------------------------------------------

def add_features(
    df,
    spy,
):

    data = df.copy()

    close = data["close"]

    volume = (
        data["volume"]
        .replace(
            0,
            np.nan,
        )
    )


    # Returns

    for days in [
        1,
        2,
        5,
        10,
        20,
    ]:

        data[
            f"return_{days}d"
        ] = close.pct_change(
            days,
            fill_method=None,
        )


    # Overnight gap

    data["gap_return"] = (
        data["open"]
        /
        close.shift(1)
        - 1
    )


    # Daily range

    data["range_pct"] = (
        data["high"]
        -
        data["low"]
    ) / close


    # Intraday return

    data["open_close_pct"] = (
        close
        -
        data["open"]
    ) / data["open"]


    # Moving averages

    for window in [
        5,
        10,
        20,
        50,
    ]:

        ma = (
            close
            .rolling(window)
            .mean()
        )


        data[
            f"ma{window}_ratio"
        ] = (
            close
            /
            ma
            - 1
        )


    # Volatility

    for window in [
        5,
        10,
        20,
    ]:

        data[
            f"volatility_{window}d"
        ] = (
            data["return_1d"]
            .rolling(window)
            .std()
        )


    # Volume

    data["volume_ratio_5d"] = (
        volume
        /
        volume
        .rolling(5)
        .mean()
    )


    data["volume_ratio_20d"] = (
        volume
        /
        volume
        .rolling(20)
        .mean()
    )


    data["volume_change_1d"] = (
        volume.pct_change(
            fill_method=None
        )
    )


    # RSI

    data["rsi_14"] = (
        calculate_rsi(
            close
        )
    )


    # MACD

    ema12 = (
        close
        .ewm(
            span=12,
            adjust=False,
        )
        .mean()
    )


    ema26 = (
        close
        .ewm(
            span=26,
            adjust=False,
        )
        .mean()
    )


    macd_raw = (
        ema12
        -
        ema26
    )


    signal_raw = (
        macd_raw
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )


    data["macd"] = (
        macd_raw
        /
        close
    )


    data["macd_signal"] = (
        signal_raw
        /
        close
    )


    data["macd_hist"] = (
        macd_raw
        -
        signal_raw
    ) / close


    # SPY

    spy_close = (
        spy["close"]
        .reindex(
            data.index
        )
        .ffill()
    )


    spy_1 = (
        spy_close
        .pct_change(
            1,
            fill_method=None,
        )
    )


    spy_5 = (
        spy_close
        .pct_change(
            5,
            fill_method=None,
        )
    )


    spy_20 = (
        spy_close
        .pct_change(
            20,
            fill_method=None,
        )
    )


    data["spy_return_1d"] = spy_1

    data["spy_return_5d"] = spy_5

    data["spy_return_20d"] = spy_20


    # Relative performance

    data["relative_return_1d"] = (
        data["return_1d"]
        -
        spy_1
    )


    data["relative_return_5d"] = (
        data["return_5d"]
        -
        spy_5
    )


    data["relative_return_20d"] = (
        data["return_20d"]
        -
        spy_20
    )


    return data


# --------------------------------------------------
# Signal strength label
# --------------------------------------------------

def signal_strength(
    probability,
):

    distance = abs(
        probability - 0.5
    )


    if distance < 0.03:
        return "Weak"


    if distance < 0.08:
        return "Moderate"


    return "Strong"


# --------------------------------------------------
# Feature importance
# --------------------------------------------------

def get_feature_importance():

    if not hasattr(
        model,
        "feature_importances_",
    ):
        return []


    pairs = list(
        zip(
            FEATURES,
            model.feature_importances_,
        )
    )


    pairs.sort(
        key=lambda item: item[1],
        reverse=True,
    )


    return [
        {
            "feature": feature,
            "importance": float(
                importance
            ),
        }
        for feature, importance
        in pairs
    ]


# --------------------------------------------------
# Backtest information
# --------------------------------------------------

def get_backtest():

    if not BACKTEST_SUMMARY_PATH.exists():
        return None

    with open(BACKTEST_SUMMARY_PATH) as file:
        summary = json.load(file)

    # Add equity curve if available.
    if BACKTEST_CSV_PATH.exists():

        frame = pd.read_csv(BACKTEST_CSV_PATH)

        curve = []

        date_column = (
            "date"
            if "date" in frame.columns
            else frame.columns[0]
        )

        for _, row in frame.iterrows():

            curve.append({
                "date": str(row[date_column]),
                "strategy": float(row["strategy_equity"]),
                "buy_hold": float(row["buy_hold_equity"]),
            })

        summary["equity_curve"] = curve

    return summary

# --------------------------------------------------
# API
# --------------------------------------------------

@app.get("/")
def home():

    return FileResponse(
        FRONTEND_DIR
        / "index.html"
    )


@app.get(
    "/api/health"
)
def health():

    return {
        "status": "ok",
        "model":
            metadata.get(
                "model",
                "Unknown",
            ),
        "version": "2.0",
    }

@app.get("/api/search")
def search_stocks(
    q: str = Query(
        default="",
        max_length=100,
    ),
):
    query = q.strip().lower()

    if not query:
        return {"results": []}

    matches = []

    for stock in STOCK_UNIVERSE:
        ticker = stock["ticker"]
        name = stock["name"]
        exchange = stock["exchange"]

        ticker_lower = ticker.lower()
        name_lower = name.lower()

        if (
            query in ticker_lower
            or query in name_lower
        ):
            # Put exact ticker matches first,
            # then ticker-prefix matches,
            # then company-name matches.
            if ticker_lower == query:
                score = 0
            elif ticker_lower.startswith(query):
                score = 1
            elif name_lower.startswith(query):
                score = 2
            else:
                score = 3

            matches.append({
                "ticker": ticker,
                "name": name,
                "exchange": exchange,
                "type": stock.get("type", "Other"),
                "_score": score,
            })

    matches.sort(
        key=lambda stock: (
            stock["_score"],
            len(stock["ticker"]),
            stock["ticker"],
        )
    )

    results = []

    for stock in matches[:10]:
        results.append({
            "ticker": stock["ticker"],
            "name": stock["name"],
            "exchange": stock["exchange"],
            "type": stock.get("type", "Other"),
        })

    return {
        "query": q,
        "results": results,
    }

@app.post(
    "/api/predict"
)
def predict(
    request: PredictionRequest,
):

    ticker = (
        request.ticker
        .strip()
        .upper()
    )


    if not ticker:
        raise HTTPException(
            status_code=400,
            detail="Enter a ticker symbol.",
        )


    if len(ticker) > 10:
        raise HTTPException(
            status_code=400,
            detail="Invalid ticker symbol.",
        )


    # Allow standard U.S. ticker characters only.
    # Examples: AAPL, BRK.B, BRK-B
    if not re.fullmatch(
        r"[A-Z0-9][A-Z0-9.-]*",
        ticker,
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid ticker symbol.",
        )


    try:

        # Use enough history for rolling
        # technical indicators.

        stock = download_prices(
            ticker,
            period="1y",
        )


        spy = download_prices(
            "SPY",
            period="1y",
        )


        featured = add_features(
            stock,
            spy,
        )


        featured = (
            featured
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )


        usable = (
            featured
            .dropna(
                subset=FEATURES
            )
        )


        if usable.empty:

            raise ValueError(
                "Not enough price history "
                "to calculate features."
            )


        latest = (
            usable.iloc[-1]
        )


        latest_date = (
            usable.index[-1]
        )


        X = pd.DataFrame(
            [
                latest[
                    FEATURES
                ].values
            ],
            columns=FEATURES,
        )


        probability_up = float(
            model.predict_proba(
                X
            )[0][1]
        )


        probability_down = (
            1.0
            -
            probability_up
        )


        direction = (
            "UP"
            if probability_up >= 0.5
            else "DOWN"
        )


        model_probability = (
            probability_up
            if direction == "UP"
            else probability_down
        )


        # Last 60 trading sessions
        # for the frontend chart.

        price_history = []


        for date, row in (
            stock
            .tail(60)
            .iterrows()
        ):

            price_history.append({

                "date":
                    str(
                        date.date()
                    ),

                "open":
                    float(
                        row["open"]
                    ),

                "high":
                    float(
                        row["high"]
                    ),

                "low":
                    float(
                        row["low"]
                    ),

                "close":
                    float(
                        row["close"]
                    ),

                "volume":
                    int(
                        row["volume"]
                    ),
            })


        response = {

            "ticker":
                ticker,

            "as_of":
                str(
                    latest_date.date()
                ),

            "latest_close":
                float(
                    latest["close"]
                ),

            "direction":
                direction,

            "up_probability":
                probability_up,

            "down_probability":
                probability_down,

            "model_probability":
                model_probability,

            "signal_strength":
                signal_strength(
                    probability_up
                ),

            "model": {
                "name":
                    metadata.get(
                        "model",
                        "Random Forest",
                    ),

                "version":
                    "V2",

                "feature_count":
                    len(FEATURES),

                "training_start":
                    metadata.get(
                        "training_start"
                    ),

                "training_tickers":
                    len(
                        metadata.get(
                            "tickers",
                            [],
                        )
                    ),
            },

            "validation": {
                "years":
                    WALK_FORWARD_VALIDATION,

                "average":
                    VALIDATION_AVERAGE,

                "description":
                    (
                        "Expanding-window "
                        "walk-forward validation. "
                        "Each test year was excluded "
                        "from its training period."
                    ),
            },

            "feature_importance":
                get_feature_importance(),

            "price_history":
                price_history,

            "backtest":
                (
                    get_backtest()
                    if ticker == "AAPL"
                    else None
                ),

            "disclaimer":
                (
                    "Experimental machine-learning "
                    "research project. The model's "
                    "historical out-of-sample results "
                    "are close to chance and should "
                    "not be interpreted as financial "
                    "advice or a reliable trading signal."
                ),
        }


        return response


    except HTTPException:
        raise


    except Exception as error:

        security = next(
            (
                stock
                for stock in STOCK_UNIVERSE
                if stock.get("ticker") == ticker
            ),
            None,
        )

        if security:
            security_type = security.get(
                "type",
                "security",
            )

            error_text = str(error).lower()

            if (
                "no market data" in error_text
                or "no data" in error_text
                or "not enough price history" in error_text
            ):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Market data is not currently "
                        f"available for {ticker} "
                        f"({security_type})."
                    ),
                )

        # Do not expose internal exception details
        # to users in production.
        print(
            f"Prediction error for {ticker}: "
            f"{type(error).__name__}: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to complete the prediction "
                "right now. Please try again later."
            ),
        )
