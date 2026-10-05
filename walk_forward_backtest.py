from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


# --------------------------------------------------
# Settings
# --------------------------------------------------

TICKER = "AAPL"

DATA_START = "2015-01-01"

TEST_YEARS = [
    2024,
    2025,
    2026,
]

UP_THRESHOLD = 0.55

INITIAL_VALUE = 10000.0

TRANSACTION_COST = 0.001


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


# --------------------------------------------------
# Download data
# --------------------------------------------------

def download_prices(ticker):

    print(
        f"Downloading {ticker}..."
    )

    df = yf.download(
        ticker,
        start=DATA_START,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(
            f"No data for {ticker}"
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

    return df[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
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
        gains /
        losses.replace(
            0,
            np.nan,
        )
    )

    return (
        100 -
        (
            100 /
            (1 + rs)
        )
    )


# --------------------------------------------------
# Feature engineering
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
        data["open"] /
        close.shift(1)
        - 1
    )


    # Intraday range

    data["range_pct"] = (
        data["high"] -
        data["low"]
    ) / close


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
            close /
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
        ema12 -
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
        macd_raw /
        close
    )


    data["macd_signal"] = (
        signal_raw /
        close
    )


    data["macd_hist"] = (
        (
            macd_raw -
            signal_raw
        )
        /
        close
    )


    # SPY features

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


    data["spy_return_1d"] = (
        spy_1
    )

    data["spy_return_5d"] = (
        spy_5
    )

    data["spy_return_20d"] = (
        spy_20
    )


    # Relative returns

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


    # Target

    next_close = (
        close.shift(-1)
    )


    data["target"] = np.where(
        next_close.notna(),
        (
            next_close >
            close
        ).astype(int),
        np.nan,
    )


    # Tradable return after the signal.
    # Features are observed after today's close.
    # The position is entered at the next trading day's open
    # and exited at that day's close.

    next_open = data["open"].shift(-1)

    data["next_return"] = (
        next_close /
        next_open -
        1
    )

    # Buy-and-hold benchmark:
    # today's close to the next trading day's close.
    data["benchmark_return"] = (
        next_close /
        close -
        1
    )


    return data


# --------------------------------------------------
# Model
# --------------------------------------------------

def create_model():

    return RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


# --------------------------------------------------
# Performance metrics
# --------------------------------------------------

def max_drawdown(
    equity,
):

    running_max = (
        equity.cummax()
    )

    drawdown = (
        equity /
        running_max -
        1
    )

    return float(
        drawdown.min()
    )


def sharpe_ratio(
    returns,
):

    returns = (
        returns
        .dropna()
    )

    if (
        returns.empty
        or returns.std() == 0
    ):
        return 0.0

    return float(
        np.sqrt(252)
        *
        returns.mean()
        /
        returns.std()
    )


# --------------------------------------------------
# Main walk-forward test
# --------------------------------------------------

def main():

    print()
    print(
        "Preparing walk-forward backtest..."
    )
    print()


    stock = download_prices(
        TICKER
    )


    spy = download_prices(
        "SPY"
    )


    data = add_features(
        stock,
        spy,
    )


    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )


    data = data.dropna(
        subset=(
            FEATURES +
            [
                "target",
                "next_return",
                "benchmark_return",
            ]
        )
    ).copy()


    data["target"] = (
        data["target"]
        .astype(int)
    )


    data["year"] = (
        data.index.year
    )


    all_predictions = []


    # --------------------------------------------------
    # Each year gets a newly trained model.
    #
    # Example:
    #
    # test 2024 -> train only before 2024
    # test 2025 -> train only before 2025
    # test 2026 -> train only before 2026
    # --------------------------------------------------

    for year in TEST_YEARS:

        print()
        print(
            "=" * 60
        )

        print(
            f"OUT-OF-SAMPLE YEAR: {year}"
        )

        print(
            "=" * 60
        )


        train = data[
            data["year"] <
            year
        ].copy()


        test = data[
            data["year"] ==
            year
        ].copy()


        if (
            train.empty
            or test.empty
        ):

            print(
                "Not enough data. Skipping."
            )

            continue


        print(
            f"Training rows: "
            f"{len(train):,}"
        )


        print(
            f"Testing rows:  "
            f"{len(test):,}"
        )


        print(
            f"Training through: "
            f"{train.index.max().date()}"
        )


        print(
            f"Testing from:     "
            f"{test.index.min().date()}"
        )


        model = create_model()


        model.fit(
            train[FEATURES],
            train["target"],
        )


        test["prob_up"] = (
            model.predict_proba(
                test[FEATURES]
            )[:, 1]
        )


        test["signal"] = (
            test["prob_up"] >=
            UP_THRESHOLD
        ).astype(int)


        test["test_year"] = (
            year
        )


        all_predictions.append(
            test
        )


        signal_days = int(
            test["signal"].sum()
        )


        if signal_days > 0:

            hit_rate = float(
                (
                    test.loc[
                        test["signal"] == 1,
                        "next_return",
                    ] > 0
                ).mean()
            )

        else:

            hit_rate = 0.0


        print(
            f"Signal days:      "
            f"{signal_days}"
        )


        print(
            f"Exposure:         "
            f"{test['signal'].mean():.1%}"
        )


        print(
            f"Signal hit rate:  "
            f"{hit_rate:.1%}"
        )


    # --------------------------------------------------
    # Combine ONLY out-of-sample predictions
    # --------------------------------------------------

    results = pd.concat(
        all_predictions
    ).sort_index()


    # Each signal represents a separate next-session
    # open-to-close trade, so charge the assumed
    # transaction cost on every executed trade.

    results["cost"] = (
        results["signal"]
        *
        TRANSACTION_COST
    )


    # Long/cash strategy

    results["strategy_return"] = (
        results["signal"]
        *
        results["next_return"]
        -
        results["cost"]
    )


    # Buy-and-hold benchmark

    results["buy_hold_return"] = (
        results["benchmark_return"]
    )


    # Equity curves

    results["strategy_equity"] = (
        INITIAL_VALUE
        *
        (
            1 +
            results["strategy_return"]
        ).cumprod()
    )


    results["buy_hold_equity"] = (
        INITIAL_VALUE
        *
        (
            1 +
            results["buy_hold_return"]
        ).cumprod()
    )


    # --------------------------------------------------
    # Overall metrics
    # --------------------------------------------------

    strategy_return = (
        results[
            "strategy_equity"
        ].iloc[-1]
        /
        INITIAL_VALUE
        - 1
    )


    buy_hold_return = (
        results[
            "buy_hold_equity"
        ].iloc[-1]
        /
        INITIAL_VALUE
        - 1
    )


    strategy_sharpe = (
        sharpe_ratio(
            results[
                "strategy_return"
            ]
        )
    )


    buy_hold_sharpe = (
        sharpe_ratio(
            results[
                "buy_hold_return"
            ]
        )
    )


    strategy_drawdown = (
        max_drawdown(
            results[
                "strategy_equity"
            ]
        )
    )


    buy_hold_drawdown = (
        max_drawdown(
            results[
                "buy_hold_equity"
            ]
        )
    )


    signal_days = int(
        results[
            "signal"
        ].sum()
    )


    exposure = float(
        results[
            "signal"
        ].mean()
    )




    if signal_days > 0:

        hit_rate = float(
            (
                results.loc[
                    results["signal"] == 1,
                    "next_return",
                ] > 0
            ).mean()
        )

    else:

        hit_rate = 0.0


    # --------------------------------------------------
    # Final output
    # --------------------------------------------------

    print()
    print()
    print(
        "=" * 65
    )

    print(
        "WALK-FORWARD BACKTEST RESULTS"
    )

    print(
        "=" * 65
    )

    print()


    print(
        f"Ticker:              "
        f"{TICKER}"
    )


    print(
        f"Period:              "
        f"{results.index.min().date()} "
        f"to "
        f"{results.index.max().date()}"
    )


    print(
        f"Observations:        "
        f"{len(results):,}"
    )


    print(
        f"Threshold:           "
        f"{UP_THRESHOLD:.1%}"
    )


    print(
        f"Signal days:         "
        f"{signal_days:,}"
    )


    print(
        f"Exposure:            "
        f"{exposure:.1%}"
    )


    print(
        f"Signal hit rate:     "
        f"{hit_rate:.1%}"
    )


    print()


    print(
        f"{'':24}"
        f"{'Strategy':>15}"
        f"{'Buy & Hold':>15}"
    )


    print(
        f"{'Final value':24}"
        f"${results['strategy_equity'].iloc[-1]:>14,.2f}"
        f"${results['buy_hold_equity'].iloc[-1]:>14,.2f}"
    )


    print(
        f"{'Total return':24}"
        f"{strategy_return:>14.1%}"
        f"{buy_hold_return:>14.1%}"
    )


    print(
        f"{'Sharpe-like ratio':24}"
        f"{strategy_sharpe:>15.2f}"
        f"{buy_hold_sharpe:>15.2f}"
    )


    print(
        f"{'Max drawdown':24}"
        f"{strategy_drawdown:>14.1%}"
        f"{buy_hold_drawdown:>14.1%}"
    )


    # --------------------------------------------------
    # Per-year results
    # --------------------------------------------------

    print()
    print(
        "=" * 65
    )

    print(
        "RESULTS BY YEAR"
    )

    print(
        "=" * 65
    )


    yearly_results = []


    for year in TEST_YEARS:

        year_data = results[
            results[
                "test_year"
            ] == year
        ].copy()


        if year_data.empty:
            continue


        year_strategy_return = (
            (
                1 +
                year_data[
                    "strategy_return"
                ]
            ).prod()
            - 1
        )


        year_buy_hold_return = (
            (
                1 +
                year_data[
                    "buy_hold_return"
                ]
            ).prod()
            - 1
        )


        year_signals = int(
            year_data[
                "signal"
            ].sum()
        )


        if year_signals > 0:

            year_hit_rate = float(
                (
                    year_data.loc[
                        year_data["signal"] == 1,
                        "next_return",
                    ] > 0
                ).mean()
            )

        else:

            year_hit_rate = 0.0


        yearly_results.append({

            "year":
                year,

            "strategy_return":
                year_strategy_return,

            "buy_hold_return":
                year_buy_hold_return,

            "signal_days":
                year_signals,

            "exposure":
                float(
                    year_data[
                        "signal"
                    ].mean()
                ),

            "hit_rate":
                year_hit_rate,
        })


    yearly_df = pd.DataFrame(
        yearly_results
    )


    print()

    for _, row in yearly_df.iterrows():

        print(
            f"{int(row['year'])}: "
            f"strategy "
            f"{row['strategy_return']:+.1%} | "
            f"buy & hold "
            f"{row['buy_hold_return']:+.1%} | "
            f"exposure "
            f"{row['exposure']:.1%} | "
            f"hit rate "
            f"{row['hit_rate']:.1%}"
        )


    # --------------------------------------------------
    # Save files
    # --------------------------------------------------

    output_dir = Path(
        "model/v2/backtests"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    csv_path = (
        output_dir /
        "aapl_walk_forward.csv"
    )


    json_path = (
        output_dir /
        "aapl_walk_forward_summary.json"
    )


    results[
        [
            "close",
            "prob_up",
            "signal",
            "next_return",
            "strategy_return",
            "buy_hold_return",
            "strategy_equity",
            "buy_hold_equity",
            "test_year",
        ]
    ].to_csv(
        csv_path
    )


    summary = {

        "ticker":
            TICKER,

        "start":
            str(
                results.index.min().date()
            ),

        "end":
            str(
                results.index.max().date()
            ),

        "threshold":
            UP_THRESHOLD,

        "observations":
            int(
                len(results)
            ),

        "signal_days":
            signal_days,

        "exposure":
            exposure,


        "hit_rate":
            hit_rate,

        "strategy_final_value":
            float(
                results[
                    "strategy_equity"
                ].iloc[-1]
            ),

        "buy_hold_final_value":
            float(
                results[
                    "buy_hold_equity"
                ].iloc[-1]
            ),

        "strategy_return":
            float(
                strategy_return
            ),

        "buy_hold_return":
            float(
                buy_hold_return
            ),

        "strategy_sharpe":
            strategy_sharpe,

        "buy_hold_sharpe":
            buy_hold_sharpe,

        "strategy_max_drawdown":
            strategy_drawdown,

        "buy_hold_max_drawdown":
            buy_hold_drawdown,

        "yearly":
            yearly_results,
    }


    with open(
        json_path,
        "w",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )


    print()
    print(
        "Saved:"
    )

    print(
        csv_path
    )

    print(
        json_path
    )

    print()

    print(
        "Walk-forward backtest complete."
    )


if __name__ == "__main__":
    main()
