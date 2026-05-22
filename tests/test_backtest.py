import unittest

import pandas as pd

from src import backtest


class BacktestTests(unittest.TestCase):
    def setUp(self):
        self.merged = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-01-01",
                        "2024-01-02",
                        "2024-01-02",
                        "2024-01-03",
                        "2024-01-03",
                        "2024-01-04",
                        "2024-01-04",
                    ]
                ),
                "classification": [
                    "Fear",
                    "Fear",
                    "Fear",
                    "Fear",
                    "Greed",
                    "Greed",
                    "Greed",
                    "Greed",
                ],
                "side": ["Long", "Short", "Long", "Short", "Long", "Short", "Long", "Short"],
                "closed_pnl": [10, 5, 8, 3, 4, 9, 2, 7],
                "return_bps": [20, 10, 18, 8, 5, 15, 4, 12],
            }
        )

    def test_build_daily_side_panel(self):
        panel = backtest.build_daily_side_panel(self.merged)
        self.assertEqual(len(panel), 8)
        self.assertIn("mean_return_bps", panel.columns)
        self.assertIn("trade_count", panel.columns)

    def test_fit_regime_side_map(self):
        panel = backtest.build_daily_side_panel(self.merged)
        regime_map = backtest.fit_regime_side_map(panel, min_trades=1)
        mapping = dict(zip(regime_map["classification"], regime_map["selected_side"]))
        self.assertEqual(mapping["Fear"], "Long")
        self.assertEqual(mapping["Greed"], "Short")

    def test_compute_performance_metrics(self):
        daily = pd.DataFrame(
            {
                "strategy": ["regime_side", "regime_side", "always_long", "always_long"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
                "daily_return_bps": [10.0, -5.0, 2.0, 3.0],
                "daily_pnl": [1.0, -0.5, 0.2, 0.3],
                "contributing_regimes": [1, 1, 1, 1],
                "trade_count": [10, 12, 8, 9],
            }
        )
        metrics = backtest.compute_performance_metrics(daily)
        self.assertEqual(set(metrics["strategy"]), {"regime_side", "always_long"})
        self.assertIn("sharpe", metrics.columns)


if __name__ == "__main__":
    unittest.main()
