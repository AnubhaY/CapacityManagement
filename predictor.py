"""
CPU Usage Predictor.

Inherits the Multi-STL + AutoARIMA + ETS forecasting engine from
BaseCapacityPredictor (base_predictor.py) and specialises it for CPU metrics.

Algorithm summary (see base_predictor.py for full detail)
----------------------------------------------------------
  1. STL(period=24)  — strip daily seasonality from hourly CPU series
  2. STL(period=168) — strip weekly seasonality from residual
  3. pmdarima.auto_arima — AICc-stepwise ARIMA on de-seasonalised remainder
  4. statsmodels ExponentialSmoothing — ensemble partner (equal weight)

Libraries: statsmodels + pmdarima  (pure Python, no C++ compiler on Windows)
"""

import logging
import pandas as pd

from base_predictor import BaseCapacityPredictor, ForecastResult   # noqa: F401 (re-export)
from config import LOGGING_CONFIG

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class CPUPredictor(BaseCapacityPredictor):
    """
    Trains STL+AutoARIMA+ETS models for each CPU metric and generates forecasts.

    Usage:
        predictor = CPUPredictor()
        predictor.train(df)          # df must have @timestamp + cpu metric columns
        predictor.predict()
        future = predictor.get_future_forecast()
    """

    METRICS = [
        "cpu.overall.pct",
        "cpu.user.pct",
        "cpu.kernel.pct",
        "cpu.wait.pct",
    ]

    # Short IDs used only for logging clarity (not required by algorithm)
    METRIC_ID_MAP = {
        "cpu.overall.pct": "overall",
        "cpu.user.pct":    "user",
        "cpu.kernel.pct":  "kernel",
        "cpu.wait.pct":    "wait",
    }

    def train(self, df: pd.DataFrame, run_cv: bool = False) -> dict:
        logger.info("=" * 50)
        logger.info("CPU Predictor — STL + AutoARIMA + ETS")
        logger.info("=" * 50)
        return super().train(df, run_cv=run_cv)
