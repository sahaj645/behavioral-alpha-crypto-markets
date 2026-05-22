"""Run the full sentiment research pipeline and export artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import analysis, backtest, cleaner, loader, visualizer


RAW_DIR = Path("data/raw")
FIGURES_DIR = Path("data/figures")
PROCESSED_DIR = Path("data/processed")


def export_summary(summary: dict) -> None:
    """Persist a compact JSON summary for downstream reporting."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = PROCESSED_DIR / "research_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def export_tables(tables: dict[str, pd.DataFrame]) -> None:
    """Persist research tables as CSV files."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)


def main() -> None:
    trades_raw = loader.load_trades(RAW_DIR / "historical_trades.csv")
    sentiment_raw = loader.load_sentiment(RAW_DIR / "fear_greed.csv")

    trades_clean = cleaner.clean_trades(trades_raw)
    sentiment_clean = cleaner.clean_sentiment(sentiment_raw)
    merged = cleaner.merge_datasets(trades_clean, sentiment_clean)

    pnl_results = analysis.pnl_by_sentiment(merged)
    winrate_results = analysis.win_rate_by_sentiment(merged)
    long_short_results = analysis.long_short_sentiment_analysis(merged)
    top_traders, top_trader_heatmap = analysis.top_trader_analysis(merged)
    leverage_results = analysis.leverage_sentiment_analysis(merged)
    symbol_heatmap = analysis.symbol_sentiment_analysis(merged)
    trader_classification = analysis.contrarian_vs_momentum_analysis(merged)
    lag_results = analysis.lag_effect_analysis(merged)

    visualizer.bar_pnl_by_sentiment(pnl_results)
    visualizer.winrate_by_sentiment(winrate_results)
    visualizer.long_short_heatmap(long_short_results)
    visualizer.top_traders_heatmap(top_trader_heatmap)
    visualizer.leverage_vs_sentiment(merged)
    visualizer.pnl_distribution_by_sentiment(merged)
    visualizer.trade_volume_by_sentiment(merged)
    visualizer.symbol_performance_heatmap(symbol_heatmap)
    visualizer.contrarian_vs_momentum(trader_classification)
    visualizer.lag_correlation_chart(lag_results)

    suite = backtest.run_backtest_suite(merged)
    summary = {
        "merged_rows": int(len(merged)),
        "date_range": {
            "start": str(pd.to_datetime(merged["date"]).min().date()),
            "end": str(pd.to_datetime(merged["date"]).max().date()),
        },
        "sentiment_coverage_pct": round(100 * merged["classification"].notna().mean(), 2),
        "avg_trade_return_bps": round(float(merged["return_bps"].mean()), 4),
        "avg_net_trade_return_bps": round(float(merged["net_return_bps"].mean()), 4),
        "best_sentiment_avg_pnl": {
            "classification": str(pnl_results["mean"].idxmax()),
            "avg_pnl": round(float(pnl_results["mean"].max()), 4),
        },
        "best_sentiment_win_rate": {
            "classification": str(winrate_results["win_rate"].idxmax()),
            "win_rate": round(float(winrate_results["win_rate"].max()), 4),
        },
        "out_of_sample_best_strategy": suite.metrics.iloc[0].to_dict(),
        "walk_forward_best_strategy": suite.walk_forward_metrics.iloc[0].to_dict(),
    }
    export_summary(summary)

    export_tables(
        {
            "backtest_metrics": suite.metrics,
            "backtest_daily_results": suite.daily_results,
            "regime_map": suite.regime_map,
            "walk_forward_metrics": suite.walk_forward_metrics,
            "walk_forward_daily": suite.walk_forward_daily,
            "significance_tests": suite.significance,
            "pnl_by_sentiment": pnl_results.reset_index(),
            "winrate_by_sentiment": winrate_results.reset_index(),
        }
    )

    print("\nResearch pipeline complete.")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Tables saved to: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
