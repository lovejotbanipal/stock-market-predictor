const form = document.getElementById("ticker-form");
const input = document.getElementById("ticker-input");
const searchSuggestions =
    document.getElementById("search-suggestions");

const errorMessage = document.getElementById("error-message");

const results = document.getElementById("results");
const emptyState = document.getElementById("empty-state");

const tickerSymbol = document.getElementById("ticker-symbol");
const asOf = document.getElementById("as-of");
const latestClose = document.getElementById("latest-close");

const direction = document.getElementById("direction");
const modelProbability = document.getElementById("model-probability");
const signalStrength = document.getElementById("signal-strength");

const upProbability = document.getElementById("up-probability");
const downProbability = document.getElementById("down-probability");
const probabilityFill = document.getElementById("probability-fill");

const modelName = document.getElementById("model-name");
const modelVersion = document.getElementById("model-version");
const featureCount = document.getElementById("feature-count");
const trainingTickers = document.getElementById("training-tickers");
const trainingStart = document.getElementById("training-start");

const validationAccuracy =
    document.getElementById("validation-accuracy");

const validationAuc =
    document.getElementById("validation-auc");

const validationF1 =
    document.getElementById("validation-f1");

const validationTable =
    document.getElementById("validation-table");

const featureImportance =
    document.getElementById("feature-importance");

const disclaimer =
    document.getElementById("disclaimer");

const backtestSection =
    document.getElementById("backtest-section");

const analyzeButton =
    form.querySelector("button");


/* =========================================================
   HELPERS
   ========================================================= */
let searchTimer = null;


function hideSearchSuggestions() {
    searchSuggestions.hidden = true;
    searchSuggestions.innerHTML = "";
}


async function searchStocks(query) {

    const response = await fetch(
        `/api/search?q=${encodeURIComponent(query)}`
    );

    if (!response.ok) {
        throw new Error("Search failed.");
    }

    return response.json();
}


function showSearchSuggestions(stocks) {

    searchSuggestions.innerHTML = "";

    if (!stocks || stocks.length === 0) {
        hideSearchSuggestions();
        return;
    }

    for (const stock of stocks) {

        const item = document.createElement("button");

        item.type = "button";
        item.className = "search-suggestion";

        item.innerHTML = `
            <span class="suggestion-main">
                <strong>${escapeHTML(stock.ticker)}</strong>
                <span>${escapeHTML(stock.name)}</span>
            </span>

            <span class="suggestion-exchange">
                ${escapeHTML(stock.exchange)} · ${escapeHTML(stock.type || "Other")}
            </span>
        `;

        item.addEventListener("click", () => {
            input.value = stock.ticker;
            hideSearchSuggestions();
            form.requestSubmit();
        });

        searchSuggestions.appendChild(item);
    }

    searchSuggestions.hidden = false;
}


input.addEventListener("input", () => {

    clearTimeout(searchTimer);

    const query = input.value.trim();

    if (query.length < 1) {
        hideSearchSuggestions();
        return;
    }

    searchTimer = setTimeout(
        async () => {
            try {
                const data = await searchStocks(query);
                showSearchSuggestions(data.results);
            } catch (error) {
                hideSearchSuggestions();
            }
        },
        200
    );
});


input.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
        hideSearchSuggestions();
    }
});


document.addEventListener("click", (event) => {

    if (
        !input.contains(event.target) &&
        !searchSuggestions.contains(event.target)
    ) {
        hideSearchSuggestions();
    }
});

function percent(value, digits = 1) {
    return `${(Number(value) * 100).toFixed(digits)}%`;
}


function money(value) {
    return new Intl.NumberFormat(
        "en-US",
        {
            style: "currency",
            currency: "USD",
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }
    ).format(Number(value));
}


function escapeHTML(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


function niceFeatureName(name) {

    const replacements = {
        "return_1d": "1-day return",
        "return_2d": "2-day return",
        "return_5d": "5-day return",
        "return_10d": "10-day return",
        "return_20d": "20-day return",

        "gap_return": "Overnight gap",
        "range_pct": "Daily range",
        "open_close_pct": "Intraday return",

        "ma5_ratio": "5-day moving avg.",
        "ma10_ratio": "10-day moving avg.",
        "ma20_ratio": "20-day moving avg.",
        "ma50_ratio": "50-day moving avg.",

        "volatility_5d": "5-day volatility",
        "volatility_10d": "10-day volatility",
        "volatility_20d": "20-day volatility",

        "volume_ratio_5d": "5-day volume",
        "volume_ratio_20d": "20-day volume",
        "volume_change_1d": "Volume change",

        "rsi_14": "RSI (14)",

        "macd": "MACD",
        "macd_signal": "MACD signal",
        "macd_hist": "MACD histogram",

        "spy_return_1d": "SPY 1-day return",
        "spy_return_5d": "SPY 5-day return",
        "spy_return_20d": "SPY 20-day return",

        "relative_return_1d": "Relative 1-day",
        "relative_return_5d": "Relative 5-day",
        "relative_return_20d": "Relative 20-day"
    };

    return replacements[name] || name;
}


/* =========================================================
   SVG HELPERS
   ========================================================= */

function svgElement(name, attributes = {}) {

    const element =
        document.createElementNS(
            "http://www.w3.org/2000/svg",
            name
        );

    for (const [key, value] of Object.entries(attributes)) {
        element.setAttribute(key, value);
    }

    return element;
}


function makePoints(values, width, height, padding = 20) {

    if (!values || values.length === 0) {
        return [];
    }

    const min = Math.min(...values);
    const max = Math.max(...values);

    const range =
        max - min === 0
            ? 1
            : max - min;

    return values.map((value, index) => {

        const x =
            padding +
            (
                index /
                Math.max(values.length - 1, 1)
            ) *
            (width - padding * 2);

        const y =
            height -
            padding -
            (
                (value - min) /
                range
            ) *
            (height - padding * 2);

        return {
            x,
            y
        };
    });
}


function pathFromPoints(points) {

    if (points.length === 0) {
        return "";
    }

    return points
        .map(
            (point, index) =>
                `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`
        )
        .join(" ");
}


/* =========================================================
   PRICE CHART
   ========================================================= */

function drawPriceChart(history) {

    const svg =
        document.getElementById("price-chart");

    svg.innerHTML = "";

    if (!history || history.length < 2) {
        return;
    }

    const width = 900;
    const height = 300;
    const padding = 18;

    const closes =
        history.map(item => Number(item.close));

    const points =
        makePoints(
            closes,
            width,
            height,
            padding
        );


    /* horizontal grid */

    for (let i = 1; i <= 4; i++) {

        const y =
            (height / 5) * i;

        svg.appendChild(
            svgElement(
                "line",
                {
                    x1: 0,
                    y1: y,
                    x2: width,
                    y2: y,
                    class: "chart-grid-line"
                }
            )
        );
    }


    /* gradient */

    const defs =
        svgElement("defs");

    const gradient =
        svgElement(
            "linearGradient",
            {
                id: "priceGradient",
                x1: "0",
                y1: "0",
                x2: "0",
                y2: "1"
            }
        );

    gradient.appendChild(
        svgElement(
            "stop",
            {
                offset: "0%",
                "stop-color": "#76b900",
                "stop-opacity": "0.20"
            }
        )
    );

    gradient.appendChild(
        svgElement(
            "stop",
            {
                offset: "100%",
                "stop-color": "#76b900",
                "stop-opacity": "0"
            }
        )
    );

    defs.appendChild(gradient);
    svg.appendChild(defs);


    const linePath =
        pathFromPoints(points);

    const areaPath =
        `${linePath}
         L ${points[points.length - 1].x} ${height}
         L ${points[0].x} ${height}
         Z`;


    svg.appendChild(
        svgElement(
            "path",
            {
                d: areaPath,
                class: "price-area"
            }
        )
    );


    svg.appendChild(
        svgElement(
            "path",
            {
                d: linePath,
                class: "price-line"
            }
        )
    );


    const start =
        history[0].date;

    const end =
        history[history.length - 1].date;

    document.getElementById(
        "chart-range"
    ).textContent =
        `${start} — ${end}`;
}


/* =========================================================
   VALIDATION
   ========================================================= */

function renderValidation(validation) {

    if (!validation) {
        return;
    }

    const average =
        validation.average || {};

    validationAccuracy.textContent =
        percent(
            average.accuracy || 0
        );

    validationAuc.textContent =
        Number(
            average.roc_auc || 0
        ).toFixed(3);

    validationF1.textContent =
        Number(
            average.f1 || 0
        ).toFixed(3);


    validationTable.innerHTML = "";


    for (const row of validation.years || []) {

        const tr =
            document.createElement("tr");

        tr.innerHTML = `
            <td>${escapeHTML(row.year)}</td>
            <td>${percent(row.accuracy)}</td>
            <td>${Number(row.roc_auc).toFixed(3)}</td>
            <td>${Number(row.f1).toFixed(3)}</td>
        `;

        validationTable.appendChild(tr);
    }
}


/* =========================================================
   FEATURE IMPORTANCE
   ========================================================= */

function renderFeatureImportance(features) {

    featureImportance.innerHTML = "";

    if (!features || features.length === 0) {

        featureImportance.innerHTML =
            "<p>Feature importance unavailable.</p>";

        return;
    }


    const top =
        features.slice(0, 12);

    const maximum =
        Math.max(
            ...top.map(
                item =>
                    Number(item.importance)
            )
        );


    for (const item of top) {

        const row =
            document.createElement("div");

        row.className =
            "feature-row";


        const relativeWidth =
            maximum > 0
                ? (
                    Number(item.importance) /
                    maximum
                ) * 100
                : 0;


        row.innerHTML = `
            <span class="feature-name">
                ${escapeHTML(
                    niceFeatureName(
                        item.feature
                    )
                )}
            </span>

            <div class="feature-track">
                <div
                    class="feature-bar"
                    style="width: ${relativeWidth}%"
                ></div>
            </div>

            <span class="feature-value">
                ${(
                    Number(item.importance) *
                    100
                ).toFixed(1)}%
            </span>
        `;


        featureImportance.appendChild(row);
    }
}


/* =========================================================
   BACKTEST
   ========================================================= */

function findBacktestValue(
    backtest,
    possibleNames,
    fallback = null
) {

    for (const name of possibleNames) {

        if (
            Object.prototype.hasOwnProperty.call(
                backtest,
                name
            )
        ) {
            return backtest[name];
        }
    }

    return fallback;
}


function drawBacktestChart(curve) {

    const svg =
        document.getElementById(
            "backtest-chart"
        );

    svg.innerHTML = "";


    if (!curve || curve.length < 2) {
        return;
    }


    const width = 900;
    const height = 300;
    const padding = 18;


    const strategy =
        curve.map(
            item =>
                Number(item.strategy)
        );


    const buyHold =
        curve.map(
            item =>
                Number(item.buy_hold)
        );


    const combined =
        [
            ...strategy,
            ...buyHold
        ];


    const minimum =
        Math.min(...combined);

    const maximum =
        Math.max(...combined);

    const range =
        maximum - minimum === 0
            ? 1
            : maximum - minimum;


    function pointsFor(values) {

        return values.map(
            (value, index) => {

                const x =
                    padding +
                    (
                        index /
                        Math.max(
                            values.length - 1,
                            1
                        )
                    ) *
                    (
                        width -
                        padding * 2
                    );


                const y =
                    height -
                    padding -
                    (
                        (
                            value -
                            minimum
                        ) /
                        range
                    ) *
                    (
                        height -
                        padding * 2
                    );


                return {
                    x,
                    y
                };
            }
        );
    }


    for (let i = 1; i <= 4; i++) {

        const y =
            (height / 5) * i;

        svg.appendChild(
            svgElement(
                "line",
                {
                    x1: 0,
                    y1: y,
                    x2: width,
                    y2: y,
                    class: "chart-grid-line"
                }
            )
        );
    }


    const strategyPoints =
        pointsFor(strategy);

    const buyHoldPoints =
        pointsFor(buyHold);


    svg.appendChild(
        svgElement(
            "path",
            {
                d:
                    pathFromPoints(
                        strategyPoints
                    ),
                class:
                    "strategy-line"
            }
        )
    );


    svg.appendChild(
        svgElement(
            "path",
            {
                d:
                    pathFromPoints(
                        buyHoldPoints
                    ),
                class:
                    "buyhold-line"
            }
        )
    );
}


function renderBacktest(backtest) {

    if (!backtest) {

        backtestSection.classList.add(
            "hidden"
        );

        return;
    }


    backtestSection.classList.remove(
        "hidden"
    );


    const strategyReturn =
        findBacktestValue(
            backtest,
            [
                "strategy_return",
                "total_return",
                "strategy_total_return"
            ]
        );


    const buyHoldReturn =
        findBacktestValue(
            backtest,
            [
                "buy_hold_return",
                "buy_and_hold_return",
                "benchmark_return"
            ]
        );


    const exposure =
        findBacktestValue(
            backtest,
            [
                "exposure",
                "strategy_exposure"
            ]
        );


    const hitRate =
        findBacktestValue(
            backtest,
            [
                "signal_hit_rate",
                "hit_rate"
            ]
        );


    document.getElementById(
        "strategy-return"
    ).textContent =
        strategyReturn !== null
            ? percent(strategyReturn)
            : "—";


    document.getElementById(
        "buy-hold-return"
    ).textContent =
        buyHoldReturn !== null
            ? percent(buyHoldReturn)
            : "—";


    document.getElementById(
        "strategy-exposure"
    ).textContent =
        exposure !== null
            ? percent(exposure)
            : "—";


    document.getElementById(
        "signal-hit-rate"
    ).textContent =
        hitRate !== null
            ? percent(hitRate)
            : "—";


    drawBacktestChart(
        backtest.equity_curve || []
    );
}


/* =========================================================
   MAIN RESULT
   ========================================================= */

function renderResult(data) {

    emptyState.classList.add(
        "hidden"
    );

    results.classList.remove(
        "hidden"
    );


    tickerSymbol.textContent =
        data.ticker;


    asOf.textContent =
        `Data through ${data.as_of}`;


    latestClose.textContent =
        money(data.latest_close);


    direction.textContent =
        data.direction;


    direction.classList.remove(
        "up",
        "down"
    );


    direction.classList.add(
        data.direction === "UP"
            ? "up"
            : "down"
    );


    modelProbability.textContent =
        percent(
            data.model_probability
        );


    signalStrength.textContent =
        data.signal_strength;


    upProbability.textContent =
        percent(
            data.up_probability
        );


    downProbability.textContent =
        percent(
            data.down_probability
        );


    probabilityFill.style.width =
        `${Number(
            data.up_probability
        ) * 100}%`;


    if (data.model) {

        modelName.textContent =
            data.model.name || "Random Forest";

        modelVersion.textContent =
            data.model.version || "V2";

        featureCount.textContent =
            data.model.feature_count ?? "—";

        trainingTickers.textContent =
            data.model.training_tickers
                ? `${data.model.training_tickers} stocks`
                : "—";

        trainingStart.textContent =
            data.model.training_start || "—";
    }


    renderValidation(
        data.validation
    );


    renderFeatureImportance(
        data.feature_importance
    );


    drawPriceChart(
        data.price_history
    );


    renderBacktest(
        data.backtest
    );



    results.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


/* =========================================================
   REQUEST
   ========================================================= */

async function analyzeTicker(ticker) {

    errorMessage.textContent = "";

    analyzeButton.disabled = true;
    analyzeButton.textContent =
        "Analyzing...";


    try {

        const response =
            await fetch(
                "/api/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        ticker
                    })
                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Unable to analyze ticker."
            );
        }


        renderResult(data);

    }

    catch (error) {

        errorMessage.textContent =
            error.message ||
            "Something went wrong.";

    }

    finally {

        analyzeButton.disabled = false;
        analyzeButton.textContent =
            "Analyze";
    }
}


/* =========================================================
   EVENTS
   ========================================================= */

form.addEventListener(
    "submit",
    event => {

        event.preventDefault();

        const ticker =
            input.value
                .trim()
                .toUpperCase();

        if (!ticker) {

            errorMessage.textContent =
                "Enter a ticker symbol.";

            return;
        }

        input.value =
            ticker;

        analyzeTicker(
            ticker
        );
    }
);
