# Behavioral Alpha in Crypto Markets

Research pipeline for analyzing how Bitcoin Fear & Greed regimes align with
realized trader outcomes on Hyperliquid.

This repository packages the workflow as a reproducible Python project rather
than a notebook-only analysis. It covers data ingestion, normalization,
sentiment regime analysis, fee-adjusted return features, backtest-style regime
evaluation, chart generation, and export of processed research artifacts.

## Overview

The project joins two datasets:

- Hyperliquid trade history
- Daily Bitcoin Fear & Greed Index readings

The merged dataset is used to study:

- PnL dispersion across sentiment regimes
- Win-rate behavior by regime and direction
- Long-versus-short behavior under different sentiment states
- Symbol-level regime sensitivity
- Trader clustering into contrarian versus momentum behavior
- Same-day and lagged sentiment relationships
- Out-of-sample regime-side selection using fee-adjusted proxy returns

## Dataset Snapshot

Latest measured run on the current raw files:

- Cleaned trades: `162,283`
- Coverage window: `2023-05-01` to `2025-05-01`
- Sentiment coverage after merge: `100.0%`
- Mean gross trade return: `63.73 bps`
- Mean net trade return after fees: `59.87 bps`

## Key Findings

- `Extreme Greed` produced the highest average trade PnL at `$1.26`.
- `Extreme Greed` also produced the highest win rate at `32.5%`.
- `Fear + Extreme Fear` averaged `36.99 bps` in net normalized return, versus
  `89.94 bps` for `Greed + Extreme Greed`.
- The fear-versus-greed return spread is statistically significant in the
  current sample with Welch test `p ~= 6.66e-128`.
- The strongest sentiment/PnL relationship is contemporaneous, with lag `0`
  correlation of `0.130`.
- In the current sample, momentum-style traders exceed contrarian traders on
  cumulative realized PnL by roughly `4.2%`.

## Research Pipeline

The codebase is organized as a batch research workflow:

- `src/loader.py`
  Loads raw datasets and normalizes column names.
- `src/cleaner.py`
  Parses timestamps, derives position side, computes fee-adjusted returns, and
  merges trade data with sentiment regimes.
- `src/analysis.py`
  Runs the core descriptive and cross-sectional analyses.
- `src/backtest.py`
  Builds regime-aware daily panels and evaluates static and walk-forward
  regime-side strategies on fee-adjusted proxy returns.
- `src/visualizer.py`
  Produces the chart set used by the notebook and batch pipeline.
- `run_research.py`
  Executes the end-to-end workflow and exports processed outputs.

## Backtest Scope

The backtest layer uses fee-adjusted realized trade returns (`net_return_bps`)
as a research proxy. It is intended for regime evaluation, not for claiming a
deployable live strategy.

Implemented checks:

- Date-based train/test split
- Regime-to-side mapping fitted on train data only
- Out-of-sample comparison against `always_long`, `always_short`, and
  `all_trades`
- Non-overlapping walk-forward evaluation
- Risk summary including return, volatility, Sharpe, win rate, cumulative
  return, and max drawdown

## Repository Layout

```text
behavioral-alpha-crypto-markets/
|-- .github/workflows/ci.yml
|-- analysis.ipynb
|-- run_research.py
|-- data/
|   |-- figures/
|   |-- processed/
|   `-- raw/
|-- reports/
|   `-- research_snapshot.md
|-- src/
|   |-- __init__.py
|   |-- analysis.py
|   |-- backtest.py
|   |-- cleaner.py
|   |-- loader.py
|   `-- visualizer.py
`-- tests/
    `-- test_backtest.py
```

## Running the Project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Provide raw data

Place the following files in `data/raw/`:

- `historical_trades.csv`
- `fear_greed.csv`

### Run the batch research workflow

```bash
python run_research.py
```

### Useful runtime options

```bash
python run_research.py --skip-plots
python run_research.py --train-days 120 --test-days 20 --min-trades 150
```

### Run tests

```bash
python -m unittest discover -s tests
```

## Outputs

Running `run_research.py` produces:

- figures in `data/figures/`
- exported tables in `data/processed/`
- a run summary in `data/processed/research_summary.json`

Representative exported tables:

- `backtest_metrics.csv`
- `walk_forward_metrics.csv`
- `significance_tests.csv`
- `pnl_by_sentiment.csv`
- `winrate_by_sentiment.csv`

## Notebook

`analysis.ipynb` remains available for exploratory review and presentation.
`run_research.py` is the reproducible batch entrypoint for repeatable research
runs.

## Limitations

This repository is intentionally scoped as a research workflow, not a trading
system. Current limitations include:

- No market mid-price or order-book replay
- No slippage model beyond direct fee adjustment
- No portfolio construction or capital allocation layer
- No live execution or monitoring stack
- No leverage analysis in the shipped dataset because the source file does not
  contain a leverage field

These constraints matter when interpreting the backtest outputs. The current
results are useful for behavioral regime analysis and research iteration, but
they should not be presented as production trading performance.
