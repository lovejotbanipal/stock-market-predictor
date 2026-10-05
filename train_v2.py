from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

MODEL_DIR = Path("model/v2")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# A reasonably diverse universe while keeping training practical on a laptop.
TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "JPM", "V", "MA",
    "UNH", "JNJ", "XOM", "CVX", "WMT",
    "COST", "HD", "PG", "KO", "PEP",
    "NFLX", "AMD", "INTC", "CRM", "ORCL",
    "BAC", "GS", "DIS", "NKE", "MCD",
]

MARKET_TICKER = "SPY"

START_DATE = "2015-01-01"


FEATURES = [
    "return_1d",
    "return_2d",
    "return_5d",
    "return_10d",
    "return_20d",

    "gap_return",

    "range_pct",
    "open_close_pct",

    "ma5_ratio",
    "ma10_ratio",
    "ma20_ratio",
    "ma50_ratio",

    "volatility_5d",
    "volatility_10d",
    "volatility_20d",

    "volume_ratio_5d",
    "volume_ratio_20d",
    "volume_change_1d",

    "rsi_14",

    "macd",
    "macd_signal",
    "macd_hist",

    "spy_return_1d",
    "spy_return_5d",
    "spy_return_20d",

    "relative_return_1d",
    "relative_return_5d",
    "relative_return_20d",
]


def download_prices(ticker):

    print(f"Downloading {ticker}...")

    df = yf.download(
        ticker,
        start=START_DATE,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        print(f"Skipping {ticker}: no data")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

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

    if not all(
        column in df.columns
        for column in required
    ):
        print(
            f"Skipping {ticker}: missing columns"
        )
        return None

    df = df[required].copy()

    df.index = pd.to_datetime(df.index)

    return df


def calculate_rsi(close, window=14):

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

    rs = gains / losses.replace(
        0,
        np.nan
    )

    return (
        100 -
        (100 / (1 + rs))
    )


def add_features(df, spy):

    data = df.copy()

    close = data["close"]

    volume = data["volume"].replace(
        0,
        np.nan
    )


    # -----------------------
    # Returns / momentum
    # -----------------------

    data["return_1d"] = close.pct_change(
        1,
        fill_method=None,
    )

    data["return_2d"] = close.pct_change(
        2,
        fill_method=None,
    )

    data["return_5d"] = close.pct_change(
        5,
        fill_method=None,
    )

    data["return_10d"] = close.pct_change(
        10,
        fill_method=None,
    )

    data["return_20d"] = close.pct_change(
        20,
        fill_method=None,
    )


    # Overnight move

    data["gap_return"] = (
        data["open"] /
        close.shift(1)
        - 1
    )


    # Daily range

    data["range_pct"] = (
        data["high"] -
        data["low"]
    ) / close


    # Intraday move

    data["open_close_pct"] = (
        close -
        data["open"]
    ) / data["open"]


    # -----------------------
    # Moving averages
    # -----------------------

    for window in [
        5,
        10,
        20,
        50,
    ]:

        moving_average = (
            close
            .rolling(window)
            .mean()
        )

        data[
            f"ma{window}_ratio"
        ] = (
            close /
            moving_average
            - 1
        )


    # -----------------------
    # Volatility
    # -----------------------

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


    # -----------------------
    # Volume
    # -----------------------

    data["volume_ratio_5d"] = (
        volume /
        volume
        .rolling(5)
        .mean()
    )

    data["volume_ratio_20d"] = (
        volume /
        volume
        .rolling(20)
        .mean()
    )

    data["volume_change_1d"] = (
        volume.pct_change(
            fill_method=None
        )
    )


    # -----------------------
    # RSI
    # -----------------------

    data["rsi_14"] = calculate_rsi(
        close,
        14,
    )


    # -----------------------
    # MACD
    # -----------------------

    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    data["macd"] = (
        ema12 -
        ema26
    ) / close

    macd_raw = (
        ema12 -
        ema26
    )

    signal_raw = macd_raw.ewm(
        span=9,
        adjust=False,
    ).mean()

    data["macd_signal"] = (
        signal_raw /
        close
    )

    data["macd_hist"] = (
        macd_raw -
        signal_raw
    ) / close


    # -----------------------
    # Market / SPY features
    # -----------------------

    aligned_spy = (
        spy["close"]
        .reindex(data.index)
        .ffill()
    )

    spy_return_1d = (
        aligned_spy
        .pct_change(
            1,
            fill_method=None,
        )
    )

    spy_return_5d = (
        aligned_spy
        .pct_change(
            5,
            fill_method=None,
        )
    )

    spy_return_20d = (
        aligned_spy
        .pct_change(
            20,
            fill_method=None,
        )
    )

    data["spy_return_1d"] = (
        spy_return_1d
    )

    data["spy_return_5d"] = (
        spy_return_5d
    )

    data["spy_return_20d"] = (
        spy_return_20d
    )


    # Stock relative to market

    data["relative_return_1d"] = (
        data["return_1d"] -
        spy_return_1d
    )

    data["relative_return_5d"] = (
        data["return_5d"] -
        spy_return_5d
    )

    data["relative_return_20d"] = (
        data["return_20d"] -
        spy_return_20d
    )


    # -----------------------
    # Prediction target
    # -----------------------

    next_close = close.shift(-1)

    data["target"] = np.where(
        next_close.notna(),
        (
            next_close >
            close
        ).astype(int),
        np.nan,
    )

    return data


def metrics_for(
    name,
    model,
    X_test,
    y_test,
):

    predictions = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    return {
        "model": name,

        "accuracy":
            float(
                accuracy_score(
                    y_test,
                    predictions,
                )
            ),

        "precision":
            float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    y_test,
                    probabilities,
                )
            ),
    }


def main():

    print()
    print(
        "Downloading modern market data..."
    )
    print()


    # SPY first because every stock uses it.

    spy = download_prices(
        MARKET_TICKER
    )

    if spy is None:
        raise RuntimeError(
            "Could not download SPY."
        )


    frames = []


    for ticker in TICKERS:

        stock = download_prices(
            ticker
        )

        if stock is None:
            continue

        featured = add_features(
            stock,
            spy,
        )

        featured["ticker"] = ticker

        featured["date"] = (
            featured.index
        )

        frames.append(
            featured.reset_index(
                drop=True
            )
        )


    if not frames:
        raise RuntimeError(
            "No stock data downloaded."
        )


    dataset = pd.concat(
        frames,
        ignore_index=True,
    )


    dataset = dataset.replace(
        [np.inf, -np.inf],
        np.nan,
    )


    dataset = dataset.dropna(
        subset=
            FEATURES +
            ["target"]
    ).copy()


    dataset["target"] = (
        dataset["target"]
        .astype(int)
    )


    dataset = dataset.sort_values(
        "date"
    ).reset_index(
        drop=True
    )


    # --------------------------------
    # Chronological holdout
    # --------------------------------

    dates = np.array(
        sorted(
            dataset[
                "date"
            ].unique()
        )
    )

    cutoff_index = int(
        len(dates) * 0.80
    )

    cutoff_date = (
        dates[
            cutoff_index
        ]
    )


    train = dataset[
        dataset["date"] <
        cutoff_date
    ].copy()


    test = dataset[
        dataset["date"] >=
        cutoff_date
    ].copy()


    X_train = train[
        FEATURES
    ]

    y_train = train[
        "target"
    ]

    X_test = test[
        FEATURES
    ]

    y_test = test[
        "target"
    ]


    print()
    print("-------------------------")
    print("Dataset")
    print("-------------------------")

    print(
        "Training rows:",
        len(train)
    )

    print(
        "Testing rows:",
        len(test)
    )

    print(
        "Training through:",
        pd.Timestamp(
            cutoff_date
        ).date()
    )

    print(
        "Testing from:",
        pd.Timestamp(
            cutoff_date
        ).date()
    )


    # --------------------------------
    # Models
    # --------------------------------

    models = {}


    models[
        "Logistic Regression"
    ] = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )
        ),
    ])


    models[
        "Random Forest"
    ] = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


    models[
        "HistGradientBoosting"
    ] = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=20,
        min_samples_leaf=30,
        l2_regularization=1.0,
        random_state=42,
    )


    results = []

    trained_models = {}


    # --------------------------------
    # Train + evaluate
    # --------------------------------

    for name, model in models.items():

        print()
        print(
            f"Training {name}..."
        )

        model.fit(
            X_train,
            y_train,
        )

        result = metrics_for(
            name,
            model,
            X_test,
            y_test,
        )

        results.append(
            result
        )

        trained_models[
            name
        ] = model


        print(
            f"Accuracy: "
            f"{result['accuracy']:.4f}"
        )

        print(
            f"ROC-AUC:  "
            f"{result['roc_auc']:.4f}"
        )

        print(
            f"F1:       "
            f"{result['f1']:.4f}"
        )


    # --------------------------------
    # Display comparison
    # --------------------------------

    results_df = (
        pd.DataFrame(
            results
        )
        .sort_values(
            "roc_auc",
            ascending=False,
        )
    )


    print()
    print("=========================")
    print("MODEL COMPARISON")
    print("=========================")
    print()

    print(
        results_df[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
            ]
        ].to_string(
            index=False
        )
    )


    # We use ROC-AUC as the automated
    # selection metric for this experiment.

    best_name = (
        results_df.iloc[0][
            "model"
        ]
    )

    best_model = (
        trained_models[
            best_name
        ]
    )


    print()
    print(
        "Selected model:",
        best_name
    )


    # --------------------------------
    # Save artifacts
    # --------------------------------

    joblib.dump(
        best_model,
        MODEL_DIR /
        "model.joblib",
    )


    metadata = {

        "model":
            best_name,

        "features":
            FEATURES,

        "metrics":
            results_df
            .iloc[0]
            .to_dict(),

        "comparison":
            results_df
            .to_dict(
                orient="records"
            ),

        "train_rows":
            int(
                len(train)
            ),

        "test_rows":
            int(
                len(test)
            ),

        "test_start":
            str(
                pd.Timestamp(
                    cutoff_date
                ).date()
            ),

        "training_start":
            START_DATE,

        "tickers":
            TICKERS,

        "target":
            "next_trading_day_close_is_higher",
    }


    with open(
        MODEL_DIR /
        "metadata.json",
        "w",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )


    print()
    print(
        "Saved:"
    )

    print(
        MODEL_DIR /
        "model.joblib"
    )

    print(
        MODEL_DIR /
        "metadata.json"
    )

    print()
    print(
        "V2 training complete."
    )


if __name__ == "__main__":
    main()
