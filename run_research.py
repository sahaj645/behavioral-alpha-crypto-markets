"""Run the full sentiment research pipeline and export artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src import analysis, backtest, cleaner, loader, visualizer


RAW_DIR = Path("data/raw")
DEFAULT_FIGURES_DIR = Path("data/figures")
DEFAULT_PROCESSED_DIR = Path("data/processed")


def export_summary(summary: dict, processed_dir: Path) -> None:
    """Persist a compact JSON summary for downstream reporting."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    summary_path = processed_dir / "research_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def export_tables(tables: dict[str, pd.DataFrame], processed_dir: Path) -> None:
    """Persist research tables as CSV files."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(processed_dir / f"{name}.csv", index=False)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for batch execution."""
    parser = argparse.ArgumentParser(
        description="Run the crypto sentiment research pipeline."
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip chart generation.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.30,
        help="Holdout fraction for the static train/test split.",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=90,
        help="Train window length for walk-forward research.",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=30,
        help="Test window length for walk-forward research.",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=100,
        help="Minimum train-period trade count required for a regime-side pair.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=DEFAULT_FIGURES_DIR,
        help="Directory for saved figures.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory for exported tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    trades_raw = loader.load_trades(RAW_DIR / "historical_trades.csv")
    sentiment_raw = loader.load_sentiment(RAW_DIR / "fear_greed.csv")

    trades_clean = cleaner.clean_trades(trades_raw)
    sentiment_clean = cleaner.clean_sentiment(sentiment_raw)
    merged = cleaner.merge_datasets(trades_clean, sentiment_clean)

    pnl_results = analysis.pnl_by_sentiment(merged)
    winrate_results = analysis.win_rate_by_sentiment(merged)
    long_short_results = analysis.long_short_sentiment_analysis(merged)
    top_traders, top_trader_heatmap = analysis.top_trader_analysis(merged)
    analysis.leverage_sentiment_analysis(merged)
    symbol_heatmap = analysis.symbol_sentiment_analysis(merged)
    trader_classification = analysis.contrarian_vs_momentum_analysis(merged)
    lag_results = analysis.lag_effect_analysis(merged)

    visualizer.FIGURES_DIR = args.figures_dir
    visualizer.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_plots:
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

    suite = backtest.run_backtest_suite(
        merged,
        test_fraction=args.test_fraction,
        train_days=args.train_days,
        test_days=args.test_days,
        min_trades=args.min_trades,
    )
    summary = {
        "config": {
            "skip_plots": args.skip_plots,
            "test_fraction": args.test_fraction,
            "train_days": args.train_days,
            "test_days": args.test_days,
            "min_trades": args.min_trades,
        },
        "merged_rows": int(len(merged)),
        "date_range": {
            "start": str(pd.to_datetime(merged["date"]).min().date()),
            "end": str(pd.to_datetime(merged["date"]).max().date()),
        },
        "sentiment_coverage_pct": round(
            100 * merged["classification"].notna().mean(),
            2,
        ),
        "avg_trade_return_bps": round(float(merged["return_bps"].mean()), 4),
        "avg_net_trade_return_bps": round(
            float(merged["net_return_bps"].mean()),
            4,
        ),
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
    export_summary(summary, args.processed_dir)

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
        },
        args.processed_dir,
    )

    print("\nResearch pipeline complete.")
    print(f"Figures saved to: {args.figures_dir}")
    print(f"Tables saved to: {args.processed_dir}")


if __name__ == "__main__":
    main()
