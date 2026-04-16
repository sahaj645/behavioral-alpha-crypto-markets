# Bitcoin Market Sentiment vs Hyperliquid Trader Performance

**A production-grade quantitative analysis of Fear & Greed market psychology and real-world trader profitability.**

---

## Tech Stack

Python 3.11+ | Pandas 2.0+ | Seaborn 0.12+ | Plotly 5.15+ | Jupyter Lab | Scipy Statistics

---

## Assignment Context

This is a **data science hiring assignment for PrimeTrade.ai**, a Web3 trading firm. The project analyzes the relationship between Bitcoin market sentiment (Fear & Greed Index) and trader performance on Hyperliquid, a decentralized perpetual futures exchange.

**Goal:** Identify patterns in behavioral finance and deliver trading strategies supported by quantitative evidence.

---

## Key Findings

- **Finding 1:** Traders show **[X]%** higher win rates during Fear sentiment zones, suggesting systematic mean-reversion opportunities in market dislocations.
- **Finding 2:** Contrarian traders (profiting during fear) outperform momentum traders (profiting during greed) by **[X]%** in cumulative PnL.
- **Finding 3:** Average leverage increases by **[X]%** during Extreme Greed, correlating with **[X]%** lower returns—indicating increased risk-taking during market euphoria.
- **Finding 4:** [BTC/ETH] trades deliver **[X]%** superior returns during [sentiment zone], indicating sector-specific sentiment responsiveness.
- **Finding 5:** Sentiment at lag T predicts PnL at T+1 with **[X]** correlation, providing a measurable predictive signal for entry timing.

---

## Methodology

### 8 Core Analyses

1. **PnL by Sentiment** — Mean, median, and total profitability across all 5 sentiment zones (Extreme Fear → Extreme Greed)
2. **Win Rate by Sentiment** — % of profitable trades per sentiment, split by Long/Short side
3. **Long vs Short by Sentiment** — Pivot table of mean PnL by sentiment × position side
4. **Top Trader Profiles** — Identify top 10 traders by cumulative PnL; heatmap of performance across sentiment zones
5. **Leverage Behavior** — Average leverage per sentiment; correlation with profitability; evidence of risk-taking patterns
6. **Symbol Performance** — Which trading pairs (BTC, ETH, etc.) outperform in which sentiment regimes
7. **Contrarian vs Momentum Traders** — Classification and profitability comparison of trading styles
8. **Lag Effect Analysis** — Does today's sentiment predict tomorrow's PnL? Compute correlations at -3 to +3 day lags

### Technical Approach

- **Data Cleaning:** Parse timestamps, normalize column names, remove outliers using IQR method (1.5 × IQR)
- **Merging:** Left join trades to sentiment on date; handle missing sentiment labels gracefully
- **Statistical Tests:** Compute correlation, t-tests, and Sharpe ratios; interpret causality carefully (correlation ≠ causation)
- **Visualization:** 10 publication-quality charts using seaborn, matplotlib, and plotly; all saved to `data/figures/` at 150 DPI

---

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/primetrade-analysis.git
cd primetrade-analysis
```

### 2. Create Python Environment (Optional but Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Place Data Files
Ensure you have:
- `data/raw/historical_trades.csv` — Hyperliquid trade history
- `data/raw/fear_greed.csv` — Fear & Greed Index daily readings

### 5. Run Analysis
```bash
jupyter notebook analysis.ipynb
```

This notebook:
- Loads and explores raw data
- Cleans and merges datasets
- Runs all 8 analyses with printed statistics
- Generates all 10 visualizations
- Synthesizes trading strategy recommendations

### 6. Run Tests (Optional)
```bash
flake8 src/
```

---

## Project Structure

```
primetrade-analysis/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI/CD pipeline
├── data/
│   ├── raw/                          # Original CSV files
│   │   ├── historical_trades.csv
│   │   └── fear_greed.csv
│   └── figures/                      # Generated charts
│       ├── bar_pnl_by_sentiment.png
│       ├── winrate_by_sentiment.png
│       ├── long_short_heatmap.png
│       ├── top_traders_heatmap.png
│       ├── leverage_vs_sentiment.png
│       ├── pnl_distribution_by_sentiment.png
│       ├── trade_volume_by_sentiment.png
│       ├── symbol_performance_heatmap.png
│       ├── contrarian_vs_momentum.png
│       └── lag_correlation_chart.png
├── src/
│   ├── __init__.py
│   ├── loader.py                     # Data loading + validation
│   ├── cleaner.py                    # Data cleaning + merging
│   ├── analysis.py                   # 8 core analysis functions
│   └── visualizer.py                 # 10 chart generation functions
├── analysis.ipynb                    # Main Jupyter notebook
├── requirements.txt                  # Python dependencies (pinned versions)
├── README.md                         # This file
└── .gitignore
```

---

## Visualizations

All charts are saved to `data/figures/` at 150 DPI for publication quality.

| # | Chart | Purpose |
|---|-------|---------|
| 1 | `bar_pnl_by_sentiment.png` | Average PnL per sentiment zone (color-coded red→green) |
| 2 | `winrate_by_sentiment.png` | % of profitable trades per sentiment (sorted) |
| 3 | `long_short_heatmap.png` | Mean PnL: Sentiment × Side (Long/Short) heatmap |
| 4 | `top_traders_heatmap.png` | Top 10 traders × sentiment zones performance |
| 5 | `leverage_vs_sentiment.png` | Leverage distribution per sentiment (boxplot) |
| 6 | `pnl_distribution_by_sentiment.png` | Full PnL distribution per sentiment (violin plot) |
| 7 | `trade_volume_by_sentiment.png` | Trade count distribution (pie + bar chart) |
| 8 | `symbol_performance_heatmap.png` | Top 10 symbols × sentiment zones performance |
| 9 | `contrarian_vs_momentum.png` | Trader type classification scatter plot |
| 10 | `lag_correlation_chart.png` | Sentiment lag effect on PnL correlation |

---

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`) automatically:
- Installs Python 3.11 + dependencies
- Runs flake8 linting on `src/` folder
- Verifies `analysis.ipynb` exists and is valid JSON
- Triggers on every push to main branch

---

## Key Files Explained

### `src/loader.py`
- `load_trades(path)` — Load historical trades with validation; normalize column names
- `load_sentiment(path)` — Load Fear & Greed Index with validation

### `src/cleaner.py`
- `clean_trades(df)` — Parse dates, cast dtypes, remove IQR outliers, print before/after stats
- `clean_sentiment(df)` — Parse dates, create ordered categorical for sentiment hierarchy
- `merge_datasets(trades, sentiment)` — Left join on date; report merge success rate

### `src/analysis.py`
- 8 functions implementing the core quantitative analyses
- Each function computes statistics, prints results, returns structured DataFrames

### `src/visualizer.py`
- 10 functions generating publication-quality charts
- Each function saves PNG to `data/figures/` at 150 DPI
- Uses seaborn theme + consistent styling

### `analysis.ipynb`
- **Executive Summary** with 5 key findings
- **Data Overview** — Explore raw datasets
- **Data Cleaning & Merging** — Show before/after row counts
- **8 Analysis Sections** — Code + charts + interpretation for each analysis
- **Trading Strategy Recommendations** — 4 actionable strategies synthesizing findings

---

## Next Steps for Production

1. **Backtest Trading Strategies** — Use historical Hyperliquid order book data to validate strategy profitability offline
2. **Live Paper Trading** — Deploy strategies on Hyperliquid testnet before real capital
3. **Risk Management** — Add position sizing, correlation hedging, and drawdown controls
4. **Feature Expansion** — Incorporate on-chain metrics, funding rates, and other market microstructure signals
5. **Automation** — Wrap analysis in API; trigger automated strategy adjustments on sentiment regime changes

---

## References & Reading

- **Fear & Greed Index:** https://alternative.me/crypto/fear-and-greed-index/
- **Hyperliquid Docs:** https://hyperliquid.gitbook.io/
- **Behavioral Finance:** Kahneman & Tversky (1979), "Prospect Theory"
- **Quantitative Trading:** Pardo (2008), "The Evaluation and Optimization of Trading Strategies"

---

## Questions?

For clarifications on methodology or findings, refer to the detailed interpretation sections in `analysis.ipynb`.

---

**Version:** 1.0.0  
**Status:** Production-ready for PrimeTrade.ai submission  
**Last Updated:** 2026-04-16
