import csv
import io
import json
import urllib.request
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "stock_universe.json"

NASDAQ_URL = (
    "https://www.nasdaqtrader.com/dynamic/"
    "symdir/nasdaqlisted.txt"
)

OTHER_URL = (
    "https://www.nasdaqtrader.com/dynamic/"
    "symdir/otherlisted.txt"
)


def download_text(url):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=30,
    ) as response:
        return response.read().decode("utf-8")

def classify_security(ticker, name):
    ticker_upper = ticker.upper()
    name_lower = name.lower()


    # Derivative / special listing types first.
   # Units come before warrants because unit descriptions
    # can also contain the word "warrant".
    if (
        "units" in name_lower
        or " unit" in name_lower
        or ticker_upper.endswith(".U")
    ):
        return "Unit"

    if (
        "warrant" in name_lower
        or ticker_upper.endswith(".W")
    ):
        return "Warrant"

    if (
        "rights" in name_lower
        or " right" in name_lower
    ):
        return "Right"

    if (
        "preferred stock" in name_lower
        or "preferred shares" in name_lower
        or "depositary shares" in name_lower
        and "preferred" in name_lower
    ):
        return "Preferred"

    # Funds.
    if (
        " etf" in name_lower
        or name_lower.endswith("etf")
        or "exchange traded fund" in name_lower
    ):
        return "ETF"

    if (
        "closed-end fund" in name_lower
        or "closed end fund" in name_lower
    ):
        return "Closed-End Fund"

    # Depositary receipts representing foreign companies.
    if (
        "american depositary" in name_lower
        or "american depository" in name_lower
        or "depositary receipt" in name_lower
        or "depositary shares" in name_lower
    ):
        return "ADR"

    # REITs are still operating equities, but useful to identify separately.
    if (
        "real estate investment trust" in name_lower
        or " reit" in name_lower
    ):
        return "REIT"

    # SPACs frequently appear as acquisition corporations.
    if (
        "acquisition corp" in name_lower
        or "acquisition corporation" in name_lower
        or "acquisition company" in name_lower
    ):
        return "SPAC"

    # Ordinary/common equity.
    if (
        "common stock" in name_lower
        or "common shares" in name_lower
        or "ordinary shares" in name_lower
        or "ordinary share" in name_lower
        or "class a shares" in name_lower
        or "class b shares" in name_lower
    ):
        return "Stock"

    return "Other"

def add_stock(
    universe,
    ticker,
    name,
    exchange,
):
    ticker = ticker.strip()
    name = name.strip()

    if not ticker or not name:
        return

    # Ignore Nasdaq metadata/footer rows.
    if ticker.startswith("File Creation Time"):
        return

    universe[ticker] = {
        "ticker": ticker,
        "name": name,
        "exchange": exchange,
        "type": classify_security(ticker, name),
    }


def load_nasdaq(universe):
    text = download_text(NASDAQ_URL)

    reader = csv.DictReader(
        io.StringIO(text),
        delimiter="|",
    )

    for row in reader:
        ticker = row.get("Symbol", "").strip()
        name = row.get("Security Name", "").strip()

        if not ticker or not name:
            continue

        # Skip test securities.
        if row.get("Test Issue", "").strip() == "Y":
            continue

        add_stock(
            universe,
            ticker,
            name,
            "NASDAQ",
        )


def load_other_exchanges(universe):
    text = download_text(OTHER_URL)

    reader = csv.DictReader(
        io.StringIO(text),
        delimiter="|",
    )

    exchange_names = {
        "A": "NYSE American",
        "N": "NYSE",
        "P": "NYSE Arca",
        "Z": "Cboe BZX",
        "V": "IEX",
    }

    for row in reader:
        ticker = row.get("ACT Symbol", "").strip()
        name = row.get("Security Name", "").strip()
        exchange_code = row.get("Exchange", "").strip()

        if not ticker or not name:
            continue

        if row.get("Test Issue", "").strip() == "Y":
            continue

        exchange = exchange_names.get(
            exchange_code,
            exchange_code,
        )

        add_stock(
            universe,
            ticker,
            name,
            exchange,
        )


def main():
    print("Building U.S. stock universe...")

    universe = {}

    print("Downloading Nasdaq listings...")
    load_nasdaq(universe)

    print("Downloading other U.S. exchange listings...")
    load_other_exchanges(universe)

    stocks = sorted(
        universe.values(),
        key=lambda stock: stock["ticker"],
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            stocks,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(f"Saved {len(stocks):,} listed securities.")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
