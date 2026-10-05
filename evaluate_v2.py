import warnings

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
    f1_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "JPM", "V", "MA",
    "UNH", "JNJ", "XOM", "CVX", "WMT",
    "COST", "HD", "PG", "KO", "PEP",
    "NFLX", "AMD", "INTC", "CRM", "ORCL",
    "BAC", "GS", "DIS", "NKE", "MCD",
]

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

    return 100 - (
        100 / (1 + rs)
    )


def add_features(df, spy):

    data = df.copy()

    close = data["close"]

    volume = data["volume"].replace(
        0,
        np.nan
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
            close / ma
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
        volume /
        volume.rolling(5).mean()
    )

    data["volume_ratio_20d"] = (
        volume /
        volume.rolling(20).mean()
    )

    data["volume_change_1d"] = (
        volume.pct_change(
            fill_method=None
        )
    )


    # RSI

    data["rsi_14"] = calculate_rsi(
        close
    )


    # MACD

    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    macd_raw = ema12 - ema26

    signal_raw = (
        macd_raw
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    data["macd"] = (
        macd_raw / close
    )

    data["macd_signal"] = (
        signal_raw / close
    )

    data["macd_hist"] = (
        macd_raw -
        signal_raw
    ) / close


    # SPY market features

    spy_close = (
        spy["close"]
        .reindex(data.index)
        .ffill()
    )

    spy_1 = spy_close.pct_change(
        1,
        fill_method=None,
    )

    spy_5 = spy_close.pct_change(
        5,
        fill_method=None,
    )

    spy_20 = spy_close.pct_change(
        20,
        fill_method=None,
    )

    data["spy_return_1d"] = spy_1
    data["spy_return_5d"] = spy_5
    data["spy_return_20d"] = spy_20


    # Relative performance

    data["relative_return_1d"] = (
        data["return_1d"] -
        spy_1
    )

    data["relative_return_5d"] = (
        data["return_5d"] -
        spy_5
    )

    data["relative_return_20d"] = (
        data["return_20d"] -
        spy_20
    )


    # Next-session direction

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


def make_models():

    return {

        "Logistic Regression":
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=20,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),

        "HistGradientBoosting":
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=20,
                min_samples_leaf=30,
                l2_regularization=1.0,
                random_state=42,
            ),
    }


def main():

    print()
    print("Preparing walk-forward dataset...")
    print()


    spy = download_prices("SPY")

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

        data = add_features(
            stock,
            spy,
        )

        data["ticker"] = ticker

        data["date"] = data.index

        frames.append(
            data.reset_index(
                drop=True
            )
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

    dataset["year"] = (
        pd.to_datetime(
            dataset["date"]
        ).dt.year
    )


    test_years = [
        2022,
        2023,
        2024,
        2025,
        2026,
    ]


    results = []


    for year in test_years:

        print()
        print("=" * 55)
        print(
            f"TEST YEAR: {year}"
        )
        print("=" * 55)


        train = dataset[
            dataset["year"] <
            year
        ].copy()


        test = dataset[
            dataset["year"] ==
            year
        ].copy()


        if train.empty or test.empty:

            print(
                "Not enough data. Skipping."
            )

            continue


        print(
            f"Train observations: "
            f"{len(train):,}"
        )

        print(
            f"Test observations:  "
            f"{len(test):,}"
        )


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


        models = make_models()


        for name, model in models.items():

            print(
                f"\nTraining {name}..."
            )


            model.fit(
                X_train,
                y_train,
            )


            predictions = (
                model.predict(
                    X_test
                )
            )


            probabilities = (
                model.predict_proba(
                    X_test
                )[:, 1]
            )


            accuracy = (
                accuracy_score(
                    y_test,
                    predictions,
                )
            )


            f1 = (
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            )


            auc = (
                roc_auc_score(
                    y_test,
                    probabilities,
                )
            )


            results.append({
                "year": year,
                "model": name,
                "accuracy": accuracy,
                "f1": f1,
                "roc_auc": auc,
                "test_rows": len(test),
            })


            print(
                f"Accuracy: "
                f"{accuracy:.4f}"
            )

            print(
                f"ROC-AUC:  "
                f"{auc:.4f}"
            )

            print(
                f"F1:       "
                f"{f1:.4f}"
            )


    results_df = pd.DataFrame(
        results
    )


    print()
    print()
    print(
        "=" * 70
    )

    print(
        "WALK-FORWARD RESULTS"
    )

    print(
        "=" * 70
    )

    print()


    print(
        results_df.to_string(
            index=False
        )
    )


    print()
    print(
        "=" * 70
    )

    print(
        "AVERAGE PERFORMANCE"
    )

    print(
        "=" * 70
    )

    print()


    averages = (
        results_df
        .groupby("model")
        [
            [
                "accuracy",
                "f1",
                "roc_auc",
            ]
        ]
        .mean()
        .sort_values(
            "roc_auc",
            ascending=False,
        )
    )


    print(
        averages.to_string()
    )


    print()
    print(
        "=" * 70
    )

    print(
        "ROC-AUC BY YEAR"
    )

    print(
        "=" * 70
    )

    print()


    pivot = (
        results_df
        .pivot(
            index="year",
            columns="model",
            values="roc_auc",
        )
    )


    print(
        pivot.to_string()
    )


    print()
    print(
        "Walk-forward evaluation complete."
    )


if __name__ == "__main__":
    main()
