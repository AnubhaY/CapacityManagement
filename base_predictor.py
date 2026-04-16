"""
BaseCapacityPredictor — shared MSTL-equivalent forecasting engine.

Algorithm: Multi-STL decomposition + AutoARIMA + ExponentialSmoothing ensemble
-------------------------------------------------------------------------------
Replaces statsforecast (not available on all Windows environments) with
pure-Python libraries that install anywhere without a C++ compiler:

  statsmodels  — STL seasonal decomposition, ExponentialSmoothing (ETS)
  pmdarima     — auto_arima: automatic ARIMA order selection via AICc

How it mirrors the original MSTL + AutoARIMA approach
------------------------------------------------------
  Original (statsforecast)        This implementation
  ──────────────────────────────  ───────────────────────────────────────────
  MSTL(season_length=[24,168])    STL(period=24) then STL(period=168) on resid
  AutoARIMA on remainder          pmdarima.auto_arima on de-seasonalised series
  AutoETS ensemble                statsmodels ExponentialSmoothing ensemble
  StatsForecast.predict(level=95) arima.predict(return_conf_int=True, alpha=.05)

Public API (identical for CPUPredictor and MemoryPredictor)
-----------------------------------------------------------
  predictor.train(df, run_cv=False)  → dict[metric, ForecastResult]
  predictor.predict(periods, freq)   → dict[metric, ForecastResult]
  predictor.get_future_forecast()    → dict[metric, pd.DataFrame]
  predictor.get_model_summary()      → pd.DataFrame
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# STL decomposition + ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Auto-ARIMA — pure Python, no compiler required
import pmdarima as pm

warnings.filterwarnings("ignore")

from config import PREDICTION_CONFIG, LOGGING_CONFIG

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    """Holds the forecast output for a single metric."""
    metric: str
    forecast_df: pd.DataFrame       # columns: ds, yhat, yhat_lower, yhat_upper
    model_name: str = "STL+AutoARIMA+ETS"
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseCapacityPredictor:
    """
    Shared time-series forecasting engine used by CPUPredictor and MemoryPredictor.

    Subclasses must define:
        METRICS      : list[str]   — column names in the input DataFrame
        METRIC_ID_MAP: dict[str,str] — metric name → short unique_id string

    Subclasses may override:
        _build_stacked_frame()  — if their DataFrame needs special pre-processing
    """

    METRICS: list[str] = []
    METRIC_ID_MAP: dict[str, str] = {}

    # Hourly seasonality periods
    PERIOD_DAILY  = 24     # 24 h cycle
    PERIOD_WEEKLY = 168    # 7 × 24 h cycle
    MIN_WEEKLY_POINTS = 2 * 168   # need ≥2 full weeks for weekly STL

    def __init__(self, config: dict = None):
        self.config = config or PREDICTION_CONFIG
        self.models: dict[str, ForecastResult] = {}
        self._fitted: dict[str, dict] = {}          # internal model state per metric
        self._train_series: dict[str, pd.Series] = {}  # hourly series used for training
        self._last_training_dates: dict[str, pd.Timestamp] = {}
        self._forecast_index: dict[str, pd.DatetimeIndex] = {}  # future timestamps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, run_cv: bool = False) -> dict:
        """
        Resample data to hourly, apply STL decomposition, fit AutoARIMA + ETS
        for every metric.

        Args:
            df:     DataFrame with @timestamp + metric columns
            run_cv: Compute accuracy via 7-day hold-out split
        """
        logger.info("Preparing training data (resampling to hourly)...")
        series_map = self._build_series_map(df)

        for metric, series in series_map.items():
            logger.info(f"  Fitting STL+AutoARIMA for {metric}  ({len(series)} hourly points)")
            try:
                self._fitted[metric] = self._fit_series(series)
                self._train_series[metric] = series
                self._last_training_dates[metric] = series.index[-1]
                self.models[metric] = ForecastResult(
                    metric=metric,
                    forecast_df=pd.DataFrame(),
                    model_name="STL+AutoARIMA+ETS",
                )
            except Exception as exc:
                logger.error(f"  Training failed for {metric}: {exc}")

        if run_cv:
            self._compute_cv_metrics(series_map)

        return self.models

    def predict(self, periods: int = None, freq: str = None) -> dict:
        """
        Generate h-step-ahead forecasts for all trained metrics.

        Args:
            periods: Number of future hourly periods (default from config)
            freq:    Ignored (kept for API compatibility)
        """
        if not self._fitted:
            raise RuntimeError("Models not trained. Call train() first.")

        periods = periods or self.config["forecast_periods"]
        level   = int(self.config["interval_width"] * 100)  # e.g. 95

        for metric, fitted in self._fitted.items():
            if metric not in self.models:
                continue
            logger.info(f"  Forecasting {metric} → {periods} periods ahead")
            try:
                fdf = self._forecast_series(fitted, h=periods, level=level)

                # Build future timestamp index starting one hour after training end
                last_ts = self._last_training_dates[metric]
                future_idx = pd.date_range(
                    start=last_ts + pd.Timedelta(hours=1),
                    periods=periods,
                    freq="h",
                )
                fdf["ds"] = future_idx
                self.models[metric].forecast_df = fdf[["ds", "yhat", "yhat_lower", "yhat_upper"]]
            except Exception as exc:
                logger.error(f"  Forecast failed for {metric}: {exc}")

        return self.models

    def get_future_forecast(self) -> dict[str, pd.DataFrame]:
        """Return only post-training (future) forecast rows per metric."""
        future = {}
        for metric, result in self.models.items():
            if result.forecast_df.empty:
                continue
            last = self._last_training_dates.get(metric, pd.Timestamp.min)
            fdf = result.forecast_df.copy()
            fdf["ds"] = pd.to_datetime(fdf["ds"])

            # Align timezone awareness
            if fdf["ds"].dt.tz is None:
                if hasattr(last, "tzinfo") and last.tzinfo:
                    last = last.tz_localize(None)
            else:
                if not (hasattr(last, "tzinfo") and last.tzinfo):
                    last = last.tz_localize("UTC")

            future[metric] = fdf[fdf["ds"] > last].reset_index(drop=True)
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
    # Core algorithm — fit one series
    # ------------------------------------------------------------------

    def _fit_series(self, series: pd.Series) -> dict:
        """
        Apply multi-STL decomposition + AutoARIMA + ETS on a single hourly series.

        Steps:
          1. STL(period=24)  → extract daily seasonal component
          2. STL(period=168) → extract weekly seasonal component (if enough data)
          3. auto_arima      → fit on de-seasonalised remainder
          4. ExponentialSmoothing → fit on original series for ensemble

        Returns a dict that _forecast_series() can use to reconstruct the forecast.
        """
        values = series.values.astype(float)
        n = len(values)

        # ---- Step 1: daily STL (period=24) ----
        stl_daily = STL(values, period=self.PERIOD_DAILY, robust=True).fit()
        seasonal_24 = stl_daily.seasonal            # shape (n,)
        detrended_daily = values - seasonal_24      # trend + weekly + residual

        # ---- Step 2: weekly STL (period=168) ----
        seasonal_168 = np.zeros(n)
        if n >= self.MIN_WEEKLY_POINTS:
            try:
                stl_weekly = STL(detrended_daily, period=self.PERIOD_WEEKLY, robust=True).fit()
                seasonal_168 = stl_weekly.seasonal
            except Exception as exc:
                logger.debug(f"Weekly STL skipped: {exc}")

        # ---- Step 3: AutoARIMA on de-seasonalised remainder ----
        remainder = values - seasonal_24 - seasonal_168
        arima = pm.auto_arima(
            remainder,
            seasonal=False,          # seasonality already removed by STL
            stepwise=True,           # fast AICc-based search
            information_criterion="aicc",
            max_p=5, max_q=5,
            max_d=2,
            error_action="ignore",
            suppress_warnings=True,
            n_jobs=1,
        )
        logger.debug(f"    AutoARIMA order: {arima.order}")

        # ---- Step 4: ExponentialSmoothing (ensemble partner) ----
        ets_model = None
        try:
            ets_model = ExponentialSmoothing(
                values,
                trend="add",
                seasonal="add",
                seasonal_periods=self.PERIOD_DAILY,
                initialization_method="estimated",
            ).fit(optimized=True, disp=False)
        except Exception as exc:
            logger.debug(f"    ETS fitting failed, will skip ensemble: {exc}")

        return {
            "arima":        arima,
            "ets":          ets_model,
            "seasonal_24":  seasonal_24,
            "seasonal_168": seasonal_168,
            "n":            n,
        }

    # ------------------------------------------------------------------
    # Core algorithm — forecast one series
    # ------------------------------------------------------------------

    def _forecast_series(self, fitted: dict, h: int, level: int = 95) -> pd.DataFrame:
        """
        Reconstruct an h-step forecast with confidence intervals.

        Returns a DataFrame with columns: yhat, yhat_lower, yhat_upper
        (no 'ds' column — caller adds the timestamp index).
        """
        alpha = 1.0 - level / 100.0      # e.g. 0.05 for 95% CI

        seasonal_24  = fitted["seasonal_24"]
        seasonal_168 = fitted["seasonal_168"]

        # Repeat last seasonal cycles forward
        season_24_fc = np.tile(
            seasonal_24[-self.PERIOD_DAILY:],
            h // self.PERIOD_DAILY + 1
        )[:h]

        has_weekly = np.any(seasonal_168 != 0)
        season_168_fc = np.zeros(h)
        if has_weekly:
            season_168_fc = np.tile(
                seasonal_168[-self.PERIOD_WEEKLY:],
                h // self.PERIOD_WEEKLY + 1
            )[:h]

        # ---- ARIMA forecast on de-seasonalised remainder ----
        arima_fc, conf_int = fitted["arima"].predict(
            n_periods=h,
            return_conf_int=True,
            alpha=alpha,
        )

        # Re-add seasonal components to point forecast and CI bounds
        yhat_arima = arima_fc     + season_24_fc + season_168_fc
        lo_arima   = conf_int[:, 0] + season_24_fc + season_168_fc
        hi_arima   = conf_int[:, 1] + season_24_fc + season_168_fc

        # ---- ETS ensemble ----
        yhat = yhat_arima.copy()
        lo   = lo_arima.copy()
        hi   = hi_arima.copy()

        if fitted["ets"] is not None:
            try:
                ets_fc = fitted["ets"].forecast(h)
                # Equal-weight ensemble for point forecast
                yhat = (yhat_arima + np.asarray(ets_fc)) / 2.0
                # Keep CI half-width from ARIMA, re-centre on ensemble mean
                half_width = (hi_arima - lo_arima) / 2.0
                lo = yhat - half_width
                hi = yhat + half_width
            except Exception as exc:
                logger.debug(f"ETS forecast failed, using ARIMA only: {exc}")

        # Clip to valid [0, 100] range
        yhat = np.clip(yhat, 0.0, 100.0)
        lo   = np.clip(lo,   0.0, 100.0)
        hi   = np.clip(hi,   0.0, 100.0)

        return pd.DataFrame({
            "yhat":       np.round(yhat, 3),
            "yhat_lower": np.round(lo,   3),
            "yhat_upper": np.round(hi,   3),
        })

    # ------------------------------------------------------------------
    # Data preparation helpers
    # ------------------------------------------------------------------

    def _build_series_map(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """
        Convert wide DataFrame to a dict of {metric: hourly pd.Series}.
        Subclasses can override to add derived columns first.
        """
        df = df.copy()
        df["@timestamp"] = pd.to_datetime(df["@timestamp"])
        df = df.set_index("@timestamp")

        series_map = {}
        for metric in self.METRICS:
            if metric not in df.columns:
                logger.warning(f"Metric '{metric}' not in DataFrame — skipping")
                continue
            s = df[metric].dropna().clip(0, 100)
            s = s.resample("h").mean().dropna()
            if len(s) < self.PERIOD_DAILY * 3:   # need at least 3 days
                logger.warning(f"Not enough data for {metric} ({len(s)} hourly pts) — skipping")
                continue
            series_map[metric] = s

        logger.info(
            f"Training data: {len(series_map)} metrics | "
            f"window: {min(s.index[0] for s in series_map.values())} → "
            f"{max(s.index[-1] for s in series_map.values())}"
        )
        return series_map

    # ------------------------------------------------------------------
    # Optional: cross-validation accuracy
    # ------------------------------------------------------------------

    def _compute_cv_metrics(self, series_map: dict[str, pd.Series]) -> None:
        """7-day hold-out accuracy evaluation."""
        logger.info("Computing accuracy (7-day hold-out split)...")

        for metric, series in series_map.items():
            if len(series) < 200:
                continue
            split_ts = series.index[-1] - pd.Timedelta(days=7)
            train = series[series.index <= split_ts]
            test  = series[series.index >  split_ts]
            if train.empty or test.empty:
                continue
            try:
                fitted_cv = self._fit_series(train)
                fdf = self._forecast_series(fitted_cv, h=len(test))
                preds   = fdf["yhat"].values
                actuals = test.values[: len(preds)]

                mae  = float(np.mean(np.abs(actuals - preds)))
                rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
                mape = float(np.mean(np.abs((actuals - preds) / (actuals + 1e-9))) * 100)

                self.models[metric].mae  = round(mae,  3)
                self.models[metric].rmse = round(rmse, 3)
                self.models[metric].mape = round(mape, 3)
                logger.info(
                    f"  {metric}: MAE={mae:.2f}% RMSE={rmse:.2f}% MAPE={mape:.2f}%"
                )
            except Exception as exc:
                logger.warning(f"  CV failed for {metric}: {exc}")
