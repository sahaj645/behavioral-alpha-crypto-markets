# Research Snapshot

Latest local run from `run_research.py` on the bundled raw files.

## Data scope

- Cleaned trades: `162,283`
- Date range: `2023-05-01` to `2025-05-01`
- Sentiment merge coverage: `100%`
- Mean gross trade return: `63.73 bps`
- Mean net trade return: `59.87 bps`

## Core findings

- Best average trade PnL: `Extreme Greed` at `$1.26`
- Best win rate: `Extreme Greed` at `32.5%`
- Fear-plus-extreme-fear mean net return: `36.99 bps`
- Greed-plus-extreme-greed mean net return: `89.94 bps`
- Fear vs greed Welch test p-value: `6.66e-128`
- Strongest lag relationship: same day, correlation `0.130`

## Backtest notes

- Research mode uses fee-adjusted realized trade returns.
- Static out-of-sample summary and walk-forward summary are exported to
  `data/processed/backtest_metrics.csv` and
  `data/processed/walk_forward_metrics.csv`.
- These should be read as regime diagnostics, not live-trading claims, because
  the source data is trader PnL rather than market return series.
