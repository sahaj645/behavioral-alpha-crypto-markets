"""
Research and backtest helpers for regime-aware sentiment strategies.

The strategies in this module are proxies built from realized trader PnL rather
than exchange mid-price returns. They are useful for exploratory research and
portfolio projects, but they should not be presented as deployable execution
models without market-level data, fees, and slippage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


SENTIMENT_ORDER = [
    "Extreme Fear",
    "Fear",
    "Neutral",
    "Greed",
    "Extreme Greed",
]


@dataclass
class BacktestArtifacts:
    """Container for backtest outputs."""

    metrics: pd.DataFrame
    daily_results: pd.DataFrame
    regime_map: pd.DataFrame
    walk_forward_metrics: pd.DataFrame
    walk_forward_daily: pd.DataFrame
    significance: pd.DataFrame


def build_daily_side_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate realized trade returns to a daily sentiment x side panel."""
    required_columns = {
        "date",
        "classification",
        "side",
        "closed_pnl",
        "net_closed_pnl",
        "net_return_bps",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_str}")

    panel = df.dropna(subset=["date", "classification", "side", "net_return_bps"]).copy()
    panel = panel[panel["side"].isin(["Long", "Short"])]
    panel["date"] = pd.to_datetime(panel["date"])

    grouped = (
        panel.groupby(["date", "classification", "side"], observed=True)
        .agg(
            mean_return_bps=("net_return_bps", "mean"),
            median_return_bps=("net_return_bps", "median"),
            mean_pnl=("net_closed_pnl", "mean"),
            total_pnl=("net_closed_pnl", "sum"),
            trade_count=("net_closed_pnl", "count"),
        )
        .reset_index()
        .sort_values(["date", "classification", "side"])
        .reset_index(drop=True)
    )

    return grouped


def train_test_split_by_date(
    panel: pd.DataFrame,
    test_fraction: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the daily panel into train and test segments by date."""
    unique_dates = np.array(sorted(panel["date"].dropna().unique()))
    split_idx = max(1, int(len(unique_dates) * (1 - test_fraction)))
    split_date = unique_dates[split_idx - 1]

    train = panel[panel["date"] <= split_date].copy()
    test = panel[panel["date"] > split_date].copy()
    return train, test


def fit_regime_side_map(
    train_panel: pd.DataFrame,
    min_trades: int = 100,
) -> pd.DataFrame:
    """Choose the better side for each sentiment regime using train data."""
    eligible = train_panel[train_panel["trade_count"] >= min_trades].copy()
    if eligible.empty:
        raise ValueError("No train observations satisfy min_trades.")

    summary = (
        eligible.groupby(["classification", "side"], observed=True)
        .agg(
            avg_return_bps=("mean_return_bps", "mean"),
            avg_pnl=("mean_pnl", "mean"),
            observations=("date", "nunique"),
            total_trades=("trade_count", "sum"),
        )
        .reset_index()
    )

    winners = (
        summary.sort_values(
            ["classification", "avg_return_bps", "avg_pnl", "observations"],
            ascending=[True, False, False, False],
        )
        .groupby("classification", observed=True)
        .head(1)
        .reset_index(drop=True)
    )

    return winners.rename(columns={"side": "selected_side"})


def evaluate_regime_strategy(
    test_panel: pd.DataFrame,
    regime_map: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the fitted sentiment-side map to an out-of-sample test panel."""
    selected = test_panel.merge(
        regime_map[["classification", "selected_side"]],
        on="classification",
        how="inner",
    )
    selected = selected[selected["side"] == selected["selected_side"]].copy()
    selected["strategy"] = "regime_side"

    baseline_long = test_panel[test_panel["side"] == "Long"].copy()
    baseline_long["strategy"] = "always_long"

    baseline_short = test_panel[test_panel["side"] == "Short"].copy()
    baseline_short["strategy"] = "always_short"

    baseline_all = (
        test_panel.groupby(["date", "classification"], observed=True)
        .agg(
            mean_return_bps=("mean_return_bps", "mean"),
            mean_pnl=("mean_pnl", "mean"),
            trade_count=("trade_count", "sum"),
        )
        .reset_index()
    )
    baseline_all["strategy"] = "all_trades"

    combined = pd.concat(
        [
            selected[["date", "classification", "strategy", "mean_return_bps", "mean_pnl", "trade_count"]],
            baseline_long[["date", "classification", "strategy", "mean_return_bps", "mean_pnl", "trade_count"]],
            baseline_short[["date", "classification", "strategy", "mean_return_bps", "mean_pnl", "trade_count"]],
            baseline_all[["date", "classification", "strategy", "mean_return_bps", "mean_pnl", "trade_count"]],
        ],
        ignore_index=True,
    )

    daily = (
        combined.groupby(["strategy", "date"], observed=True)
        .agg(
            daily_return_bps=("mean_return_bps", "mean"),
            daily_pnl=("mean_pnl", "mean"),
            contributing_regimes=("classification", "nunique"),
            trade_count=("trade_count", "sum"),
        )
        .reset_index()
        .sort_values(["strategy", "date"])
        .reset_index(drop=True)
    )
    return daily


def compute_performance_metrics(daily_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize strategy performance from daily proxy returns."""
    metrics = []
    for strategy, frame in daily_results.groupby("strategy", observed=True):
        returns_decimal = frame["daily_return_bps"] / 10000.0
        mean_return = returns_decimal.mean()
        vol = returns_decimal.std(ddof=1)
        sharpe = np.nan
        if vol and not np.isnan(vol):
            sharpe = (mean_return / vol) * np.sqrt(252)

        equity_curve = (1.0 + returns_decimal.fillna(0)).cumprod()
        running_max = equity_curve.cummax()
        drawdown = (equity_curve / running_max) - 1.0

        metrics.append(
            {
                "strategy": strategy,
                "days": int(len(frame)),
                "avg_daily_return_bps": frame["daily_return_bps"].mean(),
                "vol_daily_return_bps": frame["daily_return_bps"].std(ddof=1),
                "sharpe": sharpe,
                "win_rate": (frame["daily_return_bps"] > 0).mean(),
                "cumulative_return_pct": (equity_curve.iloc[-1] - 1.0) * 100,
                "max_drawdown_pct": drawdown.min() * 100,
                "avg_trade_count": frame["trade_count"].mean(),
            }
        )

    return pd.DataFrame(metrics).sort_values("sharpe", ascending=False).reset_index(drop=True)


def walk_forward_regime_strategy(
    panel: pd.DataFrame,
    train_days: int = 90,
    test_days: int = 30,
    min_trades: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a non-overlapping walk-forward regime strategy."""
    unique_dates = np.array(sorted(panel["date"].dropna().unique()))
    results = []

    start_idx = 0
    while start_idx + train_days + test_days <= len(unique_dates):
        train_dates = unique_dates[start_idx : start_idx + train_days]
        test_dates = unique_dates[start_idx + train_days : start_idx + train_days + test_days]

        train = panel[panel["date"].isin(train_dates)]
        test = panel[panel["date"].isin(test_dates)]
        regime_map = fit_regime_side_map(train, min_trades=min_trades)
        daily = evaluate_regime_strategy(test, regime_map)
        daily["window_start"] = pd.Timestamp(train_dates[0])
        daily["window_end"] = pd.Timestamp(test_dates[-1])
        results.append(daily[daily["strategy"] == "regime_side"])
        start_idx += test_days

    if not results:
        raise ValueError("Insufficient history for the configured walk-forward windows.")

    combined = pd.concat(results, ignore_index=True)
    metrics = compute_performance_metrics(combined)
    return metrics, combined


def sentiment_significance_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run regime-level Welch tests on normalized trade returns."""
    frame = df.dropna(subset=["classification", "net_return_bps"]).copy()
    frame["net_return_bps"] = pd.to_numeric(frame["net_return_bps"], errors="coerce")
    fear = frame[frame["classification"].isin(["Extreme Fear", "Fear"])]["net_return_bps"].dropna().to_numpy(dtype=float)
    greed = frame[frame["classification"].isin(["Greed", "Extreme Greed"])]["net_return_bps"].dropna().to_numpy(dtype=float)
    neutral = frame[frame["classification"] == "Neutral"]["net_return_bps"].dropna().to_numpy(dtype=float)

    tests = []
    comparisons = [
        ("fear_vs_greed", fear, greed),
        ("fear_vs_neutral", fear, neutral),
        ("greed_vs_neutral", greed, neutral),
    ]

    for label, left, right in comparisons:
        stat = stats.ttest_ind(left, right, equal_var=False, nan_policy="omit")
        tests.append(
            {
                "comparison": label,
                "left_mean_bps": left.mean(),
                "right_mean_bps": right.mean(),
                "difference_bps": left.mean() - right.mean(),
                "t_stat": stat.statistic,
                "p_value": stat.pvalue,
            }
        )

    return pd.DataFrame(tests)


def run_backtest_suite(
    merged_df: pd.DataFrame,
    test_fraction: float = 0.30,
    train_days: int = 90,
    test_days: int = 30,
    min_trades: int = 100,
) -> BacktestArtifacts:
    """Run the full research suite on the cleaned merged trade dataset."""
    panel = build_daily_side_panel(merged_df)
    train_panel, test_panel = train_test_split_by_date(panel, test_fraction=test_fraction)
    regime_map = fit_regime_side_map(train_panel, min_trades=min_trades)
    daily_results = evaluate_regime_strategy(test_panel, regime_map)
    metrics = compute_performance_metrics(daily_results)
    walk_metrics, walk_daily = walk_forward_regime_strategy(
        panel,
        train_days=train_days,
        test_days=test_days,
        min_trades=min_trades,
    )
    significance = sentiment_significance_tests(merged_df)

    return BacktestArtifacts(
        metrics=metrics,
        daily_results=daily_results,
        regime_map=regime_map,
        walk_forward_metrics=walk_metrics,
        walk_forward_daily=walk_daily,
        significance=significance,
    )
