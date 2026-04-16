"""
Memory Usage Predictor.

Inherits the Multi-STL + AutoARIMA + ETS forecasting engine from
BaseCapacityPredictor (base_predictor.py) and specialises it for memory metrics.

Algorithm summary (see base_predictor.py for full detail)
----------------------------------------------------------
  1. STL(period=24)  — strip daily seasonality from hourly memory series
  2. STL(period=168) — strip weekly seasonality from residual
  3. pmdarima.auto_arima — AICc-stepwise ARIMA on de-seasonalised remainder
  4. statsmodels ExponentialSmoothing — ensemble partner (equal weight)

Libraries: statsmodels + pmdarima  (pure Python, no C++ compiler on Windows)

Metrics handled
---------------
  system.memory.actual.used.pct  — primary metric; used for 80% threshold alert
  memory.application.pct         — derived: application bytes as % of total RAM
"""

import logging
import numpy as np
import pandas as pd

from base_predictor import BaseCapacityPredictor, ForecastResult   # noqa: F401 (re-export)
from config import LOGGING_CONFIG

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class MemoryPredictor(BaseCapacityPredictor):
    """
    Trains STL+AutoARIMA+ETS models for each memory metric and generates forecasts.

    Usage:
        predictor = MemoryPredictor()
        predictor.train(df)          # df must have @timestamp + memory columns
        predictor.predict()
        future = predictor.get_future_forecast()
    """

    METRICS = [
        "system.memory.actual.used.pct",
        "memory.application.pct",       # derived inside _build_series_map
    ]

    METRIC_ID_MAP = {
        "system.memory.actual.used.pct": "used_pct",
        "memory.application.pct":        "app_pct",
    }

    def train(self, df: pd.DataFrame, run_cv: bool = False) -> dict:
        logger.info("=" * 50)
        logger.info("Memory Predictor — STL + AutoARIMA + ETS")
        logger.info("=" * 50)
        return super().train(df, run_cv=run_cv)

    # ------------------------------------------------------------------
    # Override to derive memory.application.pct before resampling
    # ------------------------------------------------------------------

    def _build_series_map(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """
        Extend the base method: derive memory.application.pct from raw bytes
        before handing off to the standard hourly-resampling pipeline.
        """
        df = df.copy()

        if (
            "memory.application.pct" not in df.columns
            and "memory.application" in df.columns
            and "system.memory.total" in df.columns
        ):
            total = df["system.memory.total"].replace(0, np.nan)
            df["memory.application.pct"] = (
                (df["memory.application"] / total) * 100
            ).clip(0, 100)
            logger.debug("Derived memory.application.pct from byte columns")

        return super()._build_series_map(df)
