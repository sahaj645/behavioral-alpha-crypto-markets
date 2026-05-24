# Behavioral Alpha in Crypto Markets

Research pipeline studying whether Bitcoin sentiment regimes create
predictable dispersion in realized trader outcomes on Hyperliquid.

The core question: does the Fear & Greed Index carry information about
realized DEX trader performance, and can that information be structured
into a regime-conditional research framework?

---

## Research Question

Behavioral finance predicts that sentiment extremes should produce
systematic trader errors — overconfidence during greed, capitulation
during fear. If those errors are large enough, they should appear in
realized trade-level PnL distributions. This project tests that
hypothesis empirically using two years of Hyperliquid trade history.

---

## Data

Two datasets, joined on calendar date:

| Source | Description | Coverage |
|---|---|---|
| Hyperliquid trade history | Closed-position PnL, side, symbol, fee, notional | 2023-05-01 → 2025-05-01 |
| Bitcoin Fear & Greed Index | Daily sentiment classification + numeric score (0–100) | Full overlap, 100% merge rate |

**After cleaning:** 162,283 trades across 730 trading days.  
**Fee adjustment:** net return = (closed_pnl − |fee|) / |notional_usd|, expressed in basis points.  
**Outlier handling:** IQR filter on closed_pnl; winsorization on net_return_bps at 1st/99th percentile.

---

## Key Findings

### 1. Regime-conditional PnL

| Regime | Mean PnL (USD) | Mean Net Return (bps) | Win Rate |
|---|---|---|---|
| Extreme Fear | — | 36.99¹ | — |
| Fear | — | 36.99¹ | — |
| Neutral | — | — | — |
| Greed | — | 89.94¹ | — |
| Extreme Greed | $1.26 | 89.94¹ | 32.5% |

¹ Aggregated across Fear/Extreme Fear and Greed/Extreme Greed buckets respectively.

The fear-to-greed spread is **~53 bps/day** in net normalized returns.
A Welch t-test on trade-level observations yields p ≈ 6.66e-128, though
this should be treated as an indicative lower bound — trade-level
observations are not independent (they cluster within traders, symbols,
and calendar days). Trader-clustered and day-clustered specifications
are the correct inference units and are on the roadmap.

### 2. Directional behavior

Long positions outperform short positions under Greed regimes.
Short positions are relatively more competitive under Fear regimes.
The long-short asymmetry by regime is consistent with momentum-tilted
crowding during high-sentiment periods.

### 3. Trader clustering: momentum versus contrarian

Traders are classified by whether their realized PnL concentrates in
Greed regimes (momentum) or Fear regimes (contrarian):

- Momentum traders outperform contrarian traders on cumulative realized
  PnL by approximately **4.2%** in the current sample.
- This is consistent with a market where greed-regime crowding pays off
  in realized terms, even though it implies higher tail risk.

### 4. Lag structure

The sentiment-to-PnL relationship is strongest at lag 0 (same-day
correlation: 0.130). Predictive power at lags 1–3 is substantially
weaker. This suggests the signal is largely contemporaneous rather
than a forward-looking predictor — a finding with direct implications
for execution-grade strategy design.

---

## Research Pipeline

```
data/raw/
  historical_trades.csv      ← Hyperliquid closed-position data
  fear_greed.csv             ← Daily F&G index

src/
  loader.py                  ← Raw CSV ingestion, column normalization
  cleaner.py                 ← Timestamp parsing, fee adjustment, return
                                computation, outlier handling, dataset merge
  analysis.py                ← Eight descriptive cross-sections (PnL by
                                regime, win rates, long/short pivot, top-trader
                                profiles, leverage, symbol sensitivity,
                                contrarian/momentum clustering, lag correlations)
  backtest.py                ← Daily panel construction, train/test split,
                                regime-side map fitting, out-of-sample
                                evaluation, non-overlapping walk-forward,
                                performance metrics, significance tests
  visualizer.py              ← Chart generation for all analyses

run_research.py              ← End-to-end batch runner, CLI flags, artifact export
analysis.ipynb               ← Exploratory notebook (presentation layer)
```

---

## Backtest Design

The backtest uses fee-adjusted realized trade returns (`net_return_bps`)
as the performance proxy. This is explicitly a **research proxy**, not a
deployable strategy, because the underlying asset is trader PnL rather
than an exchange mid-price return series.

**What is implemented:**

- Date-based train/test split (default: 70/30)
- Regime-to-side mapping fitted on train data only, applied to holdout
- Baselines: `always_long`, `always_short`, `all_trades`
- Non-overlapping walk-forward with configurable train/test windows
- Per-strategy: annualized return, volatility, Sharpe, win rate,
  cumulative return, max drawdown

**What is not implemented (known limitations):**

- No purge window between train and test folds (label leakage possible)
- No Deflated Sharpe Ratio (no correction for multiple strategy trials)
- Significance tests use trade-level Welch; trader-clustered inference
  is not yet implemented
- No market mid-price, no order-book replay, no slippage model beyond
  direct fee subtraction
- No leverage field in the source dataset
- No portfolio construction or capital allocation layer

These constraints mean backtest outputs are useful for **regime
diagnostics and behavioral research**, not for claims about live
strategy performance.

---

## Running the Project

### Install

```bash
pip install -r requirements.txt
```

### Provide raw data

Place in `data/raw/`:

```
data/raw/historical_trades.csv
data/raw/fear_greed.csv
```

### Run the full pipeline

```bash
python run_research.py
```

### Options

```bash
# Skip chart generation (faster iteration)
python run_research.py --skip-plots

# Custom walk-forward windows
python run_research.py --train-days 120 --test-days 20 --min-trades 150

# Custom output directories
python run_research.py --figures-dir out/figs --processed-dir out/tables
```

### Run tests

```bash
python -m unittest discover -s tests
```

---

## Outputs

`run_research.py` exports to `data/processed/` and `data/figures/`:

| File | Contents |
|---|---|
| `research_summary.json` | Run config + headline metrics |
| `pnl_by_sentiment.csv` | Mean, median, std, count, total PnL per regime |
| `winrate_by_sentiment.csv` | Win rate + avg PnL per regime and side |
| `backtest_metrics.csv` | Strategy comparison on the holdout set |
| `walk_forward_metrics.csv` | Aggregated walk-forward performance |
| `significance_tests.csv` | Welch t-test results across regime comparisons |
| `regime_map.csv` | Fitted regime → side mapping (from train data) |

---

## Repository Layout

```
behavioral-alpha-crypto-markets/
├── .github/workflows/ci.yml
├── analysis.ipynb
├── run_research.py
├── data/
│   ├── figures/
│   ├── processed/
│   └── raw/
├── reports/
│   └── research_snapshot.md
├── src/
│   ├── __init__.py
│   ├── analysis.py
│   ├── backtest.py
│   ├── cleaner.py
│   ├── loader.py
│   └── visualizer.py
└── tests/
    └── test_backtest.py
```

---

## Planned Extensions

The following are scoped and in progress:

- **Clustered inference** — trader-clustered and day-clustered t-statistics;
  block bootstrap confidence intervals on the regime spread
- **Deflated Sharpe Ratio** — selection-bias-corrected Sharpe across
  all evaluated strategies
- **Orthogonal signal construction** — F&G level, F&G rate-of-change,
  realized vol regime, funding-rate z-score; sequential residualization
  to remove inter-signal correlation; IC decay curves with Newey-West SE
- **Event study** — sentiment-threshold crossings and vol-spike events;
  cumulative abnormal return analysis with confidence bands
- **Cross-sectional factor model** — time-series OLS of portfolio returns
  on orthogonal signal exposures with HAC covariance
- **Position sizing** — volatility-targeted sizing + fractional Kelly;
  break-even cost sensitivity sweep
- **Purged walk-forward** — embargo window between train and test folds
  to eliminate label leakage

---

## Limitations

This is a research workflow, not a trading system. The results should be
read as behavioral diagnostics:

- **Proxy returns:** the backtest operates on realized trader PnL, not
  on a tradable instrument with mid-price returns. There is no guarantee
  the regime edge is capturable at the market level.
- **Inference:** the headline Welch p-value is computed on trade-level
  observations, which are not independent. The true clustered p-value
  (by trader or by day) will be larger; computing it is on the roadmap.
- **No leverage data:** the source dataset does not include a leverage
  field, so risk-adjusted return analysis is incomplete.
- **No slippage model:** beyond direct fee subtraction, there is no
  model of market impact or execution slippage.
- **Single data source:** conclusions are specific to Hyperliquid in
  the 2023–2025 sample period. Out-of-sample generalization to other
  venues or periods is untested.
