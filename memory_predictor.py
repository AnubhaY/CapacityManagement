"""
Memory Usage Predictor using Nixtla's statsforecast library.

Algorithm: MSTL + AutoARIMA ensemble (same choice as CPUPredictor)
-------------------------------------------------------------------
Memory usage has characteristics that make MSTL particularly effective:
  - Slow upward trend (memory leaks, growing workloads)
  - Daily seasonality: applications allocate more during business hours
  - Mild weekly seasonality: weekday vs. weekend workloads
  - Smoother signal than CPU (less spiky) → AutoARIMA captures residuals well

Metrics predicted:
  - system.memory.actual.used.pct  (primary — used for alerting)
  - memory.application             (derived bytes metric, normalised to %)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, MSTL
import warnings

warnings.filterwarnings("ignore")

from config import PREDICTION_CONFIG, MEMORY_ALERT_CONFIG, LOGGING_CONFIG

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# 1 GB in bytes — used to normalise application bytes to a 0-100 % scale
GB = 1024 ** 3


@dataclass
class MemoryForecastResult:
    metric: str
    forecast_df: pd.DataFrame       # columns: ds, yhat, yhat_lower, yhat_upper
    model_name: str = "MSTL+AutoARIMA"
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None


class MemoryPredictor:
    """
    Trains MSTL + AutoARIMA models for each memory metric and generates
    48-hour-ahead forecasts.

    Usage:
        predictor = MemoryPredictor()
        predictor.train(df)          # df must have @timestamp + memory columns
        predictor.predict()
        future = predictor.get_future_forecast()
    """

    # Primary metric (used for alerting) + secondary byte metric
    METRICS = [
        "system.memory.actual.used.pct",
        "memory.application.pct",       # derived: app_bytes / total_bytes * 100
    ]

    METRIC_ID_MAP = {
        "system.memory.actual.used.pct": "used_pct",
        "memory.application.pct":        "app_pct",
    }

    def __init__(self, config: dict = None):
        self.config = config or PREDICTION_CONFIG
        self.models: dict[str, MemoryForecastResult] = {}
        self._sf: Optional[StatsForecast] = None
        self._train_df: Optional[pd.DataFrame] = None
        self._last_training_dates: dict[str, pd.Timestamp] = {}
        # Store total memory (bytes) from training data for display purposes
        self._total_memory_bytes: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        df: pd.DataFrame,
        run_cv: bool = False,
    ) -> dict[str, "MemoryForecastResult"]:
        """
        Prepare data and fit MSTL + AutoARIMA models for all memory metrics.

        Args:
            df:      DataFrame with @timestamp, memory.application,
                     system.memory.total, system.memory.actual.used.pct
            run_cv:  Whether to compute accuracy metrics via train/test split
        """
        logger.info("Preparing memory training data...")
        df = self._add_derived_columns(df)
        stacked = self._build_stacked_frame(df)
        self._train_df = stacked

        for metric in self.METRICS:
            uid = self.METRIC_ID_MAP[metric]
            rows = stacked[stacked["unique_id"] == uid]
            if not rows.empty:
                self._last_training_dates[metric] = rows["ds"].max()

        logger.info("Fitting MSTL + AutoARIMA (daily=24, weekly=168)...")
        self._sf = self._build_statsforecast()
        self._sf.fit(stacked)

        for metric in self.METRICS:
            uid = self.METRIC_ID_MAP[metric]
            if uid in stacked["unique_id"].values:
                self.models[metric] = MemoryForecastResult(
                    metric=metric,
                    forecast_df=pd.DataFrame(),
                    model_name="MSTL+AutoARIMA",
                )
                logger.info(f"  Trained model for {metric}")

        if run_cv:
            self._compute_cv_metrics(stacked)

        return self.models

    def predict(self, periods: int = None, freq: str = None) -> dict[str, "MemoryForecastResult"]:
        """Generate forecasts for all trained metrics."""
        if self._sf is None:
            raise RuntimeError("Models not trained. Call train() first.")

        periods = periods or self.config["forecast_periods"]
        level = [int(self.config["interval_width"] * 100)]

        logger.info(f"Generating {periods}-period memory forecast with {level[0]}% CI...")
        raw = self._sf.predict(h=periods, level=level).reset_index()

        self._distribute_forecasts(raw, level[0])
        return self.models

    def get_future_forecast(self) -> dict[str, pd.DataFrame]:
        """Return only post-training forecast rows for each metric."""
        future = {}
        for metric, result in self.models.items():
            if result.forecast_df.empty:
                continue
            last = self._last_training_dates.get(metric, pd.Timestamp.min)
            fdf = result.forecast_df.copy()
            fdf["ds"] = pd.to_datetime(fdf["ds"])
            if fdf["ds"].dt.tz is None:
                last = last.tz_localize(None) if hasattr(last, "tzinfo") and last.tzinfo else last
            else:
                if not (hasattr(last, "tzinfo") and last.tzinfo):
                    last = last.tz_localize("UTC")
            future[metric] = fdf[fdf["ds"] > last]
        return future

    def get_model_summary(self) -> pd.DataFrame:
        rows = []
        for metric, result in self.models.items():
            rows.append({
                "metric":       metric,
                "model":        result.model_name,
                "mae":          result.mae,
                "rmse":         result.rmse,
                "mape_pct":     result.mape,
                "has_forecast": not result.forecast_df.empty,
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add memory.application.pct column (app bytes as % of total RAM)."""
        df = df.copy()
        if "system.memory.total" in df.columns and "memory.application" in df.columns:
            total = df["system.memory.total"].replace(0, np.nan)
            df["memory.application.pct"] = (df["memory.application"] / total * 100).clip(0, 100)
            self._total_memory_bytes = int(df["system.memory.total"].iloc[0])
        else:
            df["memory.application.pct"] = np.nan
        return df

    def _build_stacked_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert wide DataFrame to statsforecast long format, resampled hourly."""
        frames = []
        for metric in self.METRICS:
            if metric not in df.columns:
                logger.warning(f"Metric {metric} missing from data, skipping")
                continue

            series = df[["@timestamp", metric]].copy()
            series.columns = ["ds", "y"]
            series["ds"] = pd.to_datetime(series["ds"])
            series = series.dropna()
            series["y"] = series["y"].clip(0, 100)

            # Resample to hourly mean
            series = series.set_index("ds").resample("h").mean().reset_index()
            series.columns = ["ds", "y"]
            series = series.dropna()
            series["unique_id"] = self.METRIC_ID_MAP[metric]
            frames.append(series)

        stacked = pd.concat(frames, ignore_index=True)
        stacked = stacked.sort_values(["unique_id", "ds"])
        logger.info(
            f"Stacked frame: {len(stacked)} rows, "
            f"{stacked['unique_id'].nunique()} metrics, "
            f"{stacked['ds'].min()} → {stacked['ds'].max()}"
        )
        return stacked

    def _build_statsforecast(self) -> StatsForecast:
        """MSTL + AutoARIMA ensemble for hourly data with daily+weekly seasonality."""
        models = [
            MSTL(
                season_length=[24, 168],
                trend_forecaster=AutoARIMA(
                    approximation=True,
                    stepwise=True,
                    season_length=1,
                    max_p=5,
                    max_q=5,
                    ic="aicc",
                ),
            ),
            AutoETS(season_length=24),
        ]
        return StatsForecast(
            models=models,
            freq="h",
            n_jobs=-1,
            verbose=False,
        )

    def _distribute_forecasts(self, raw: pd.DataFrame, level: int) -> None:
        """Parse statsforecast output and populate each MemoryForecastResult."""
        mstl_col = next(
            (c for c in raw.columns if "MSTL" in c and "-lo-" not in c and "-hi-" not in c),
            None,
        )
        lo_col = next((c for c in raw.columns if "MSTL" in c and f"-lo-{level}" in c), None)
        hi_col = next((c for c in raw.columns if "MSTL" in c and f"-hi-{level}" in c), None)

        non_id = [c for c in raw.columns if c not in ("unique_id", "ds")]
        if mstl_col is None and non_id:
            mstl_col = non_id[0]

        for metric in self.METRICS:
            uid = self.METRIC_ID_MAP.get(metric)
            if uid is None or metric not in self.models:
                continue

            rows = raw[raw["unique_id"] == uid].copy()
            if rows.empty:
                continue

            fdf = pd.DataFrame()
            fdf["ds"] = pd.to_datetime(rows["ds"])

            if mstl_col and mstl_col in rows.columns:
                fdf["yhat"] = rows[mstl_col].values.clip(0, 100)
            else:
                fdf["yhat"] = np.nan

            if lo_col and lo_col in rows.columns:
                fdf["yhat_lower"] = rows[lo_col].values.clip(0, 100)
            else:
                fdf["yhat_lower"] = fdf["yhat"] * 0.92

            if hi_col and hi_col in rows.columns:
                fdf["yhat_upper"] = rows[hi_col].values.clip(0, 100)
            else:
                fdf["yhat_upper"] = fdf["yhat"] * 1.08

            # AutoETS ensemble: simple equal-weight average
            ets_col = next(
                (c for c in raw.columns if "AutoETS" in c and "-lo-" not in c and "-hi-" not in c),
                None,
            )
            if ets_col and ets_col in rows.columns:
                ets_vals = rows[ets_col].values.clip(0, 100)
                fdf["yhat"] = (fdf["yhat"] + ets_vals) / 2

            self.models[metric].forecast_df = fdf.reset_index(drop=True)

    def _compute_cv_metrics(self, stacked: pd.DataFrame) -> None:
        logger.info("Computing memory accuracy metrics (last 7 days as test set)...")
        for metric in self.METRICS:
            uid = self.METRIC_ID_MAP.get(metric)
            series = stacked[stacked["unique_id"] == uid].sort_values("ds")
            if len(series) < 200:
                continue

            split = series["ds"].max() - pd.Timedelta(days=7)
            train = series[series["ds"] <= split]
            test  = series[series["ds"] >  split]
            if train.empty or test.empty:
                continue

            try:
                sf_eval = self._build_statsforecast()
                sf_eval.fit(train)
                fc = sf_eval.predict(h=len(test)).reset_index()
                mstl_col = next(
                    (c for c in fc.columns if "MSTL" in c and "-lo-" not in c and "-hi-" not in c),
                    fc.columns[-1],
                )
                preds   = fc[fc["unique_id"] == uid][mstl_col].values
                actuals = test["y"].values[: len(preds)]

                mae  = np.mean(np.abs(actuals - preds))
                rmse = np.sqrt(np.mean((actuals - preds) ** 2))
                mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-9))) * 100

                self.models[metric].mae  = round(mae,  3)
                self.models[metric].rmse = round(rmse, 3)
                self.models[metric].mape = round(mape, 3)
                logger.info(f"  {metric}: MAE={mae:.2f}% RMSE={rmse:.2f}% MAPE={mape:.2f}%")
            except Exception as e:
                logger.warning(f"CV failed for {metric}: {e}")
