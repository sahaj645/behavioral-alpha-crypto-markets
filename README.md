# Behavioral Alpha in Crypto Markets

Quant-style research project on how Bitcoin Fear & Greed regimes line up with
realized Hyperliquid trader outcomes.

This repo is built as a portfolio project for internship applications. It is
not a live trading system and it does not claim deployable alpha from sentiment
alone. The goal is to show a clean research workflow: data ingestion, feature
engineering, regime analysis, out-of-sample evaluation, and reproducible
artifacts.

## What the project does

- Loads Hyperliquid trade history and Fear & Greed daily sentiment data.
- Cleans and normalizes the raw data into a merged research dataset.
- Measures PnL, win rate, side bias, trader behavior, symbol behavior, and lag
  structure across sentiment regimes.
- Builds fee-adjusted trade-return features in basis points.
- Runs a regime-aware side-selection backtest with train/test and walk-forward
  evaluation.
- Exports figures plus processed tables to `data/processed/`.

## Current dataset snapshot

- Cleaned trades: `162,283`
- Coverage window: `2023-05-01` to `2025-05-01`
- Sentiment merge coverage: `100%`
- Mean gross trade return: `63.73 bps`
- Mean net trade return after fees: `59.87 bps`

## Measured findings from the current dataset

- `Extreme Greed` had the highest average trade PnL at `$1.26` and the highest
  win rate at `32.5%`.
- `Fear` had higher average trade PnL than `Greed` (`$1.07` vs `$0.89`), but
  not the highest win rate.
- On normalized net returns, `Fear + Extreme Fear` averaged `36.99 bps` versus
  `89.94 bps` for `Greed + Extreme Greed`.
- The fear-vs-greed return difference is statistically strong in this sample
  with Welch test `p ~= 6.66e-128`.
- The lag analysis peaks at same-day correlation (`0.130`), which suggests the
  relationship is mostly contemporaneous rather than strongly predictive.
- In the current sample, momentum-style traders slightly outperform contrarian
  traders on cumulative realized PnL (`+4.2%`).

## Research layer

The project now includes a backtest module that works on fee-adjusted realized
trade returns (`net_return_bps`).

Implemented research checks:

- Static train/test split by date.
- Regime-to-side mapping learned on train data only.
- Out-of-sample comparison against `always_long`, `always_short`, and
  `all_trades` baselines.
- Non-overlapping walk-forward evaluation.
- Risk summary: daily mean return, volatility, Sharpe, win rate, cumulative
  return, and max drawdown.

Important caveat:

The backtest is a research proxy built from realized trader outcomes, not a
market-executable strategy replay. Because the source data is already trader
PnL, the performance metrics should be treated as exploratory diagnostics, not
production trading claims.

## Project structure

```text
behavioral-alpha-crypto-markets/
|-- .github/workflows/ci.yml
|-- analysis.ipynb
|-- run_research.py
|-- data/
|   |-- figures/
|   |-- processed/
|   `-- raw/
|-- src/
|   |-- __init__.py
|   |-- loader.py
|   |-- cleaner.py
|   |-- analysis.py
|   |-- backtest.py
|   `-- visualizer.py
`-- tests/
    `-- test_backtest.py
```

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add raw data

Place these files in `data/raw/`:

- `historical_trades.csv`
- `fear_greed.csv`

### 3. Run the research pipeline

```bash
python run_research.py
```

Optional batch flags:

```bash
python run_research.py --skip-plots --train-days 120 --test-days 20 --min-trades 150
```

This produces:

- figures in `data/figures/`
- summary tables in `data/processed/`
- a compact JSON snapshot in `data/processed/research_summary.json`

### 4. Run tests

```bash
python -m unittest discover -s tests
```

## Notebook

`analysis.ipynb` remains the presentation notebook for exploratory review. The
more production-like path is `run_research.py`, which gives a repeatable batch
execution path for the same workflow.

## Production gaps

This repo is much closer to a quant research project than it was initially, but
it still has clear limits:

- No market mid-price or order-book data.
- No slippage model beyond fee adjustment.
- No portfolio sizing or capital constraints.
- No live execution layer or API integration.
- No leverage analysis in the shipped dataset because the source CSV does not
  contain a leverage field.

Those are deliberate boundaries, and they are better stated explicitly than
hidden behind inflated claims.
