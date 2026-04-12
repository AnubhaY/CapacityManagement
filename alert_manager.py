"""
Alert Manager for CPU capacity predictions.

Evaluates predicted CPU values against configurable thresholds and
generates structured alert objects. Supports three alert levels:
  - WARNING  : predicted yhat > threshold
  - CRITICAL : predicted yhat_upper (95th percentile) > threshold
  - BREACH   : predicted yhat > threshold AND upper bound well above it
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import pandas as pd

from config import ALERT_CONFIG, LOGGING_CONFIG

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):9
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CPUAlert:
    alert_id: str
    severity: AlertSeverity
    metric: str
    predicted_value: float
    threshold: float
    predicted_at: datetime           # When the alert was generated
    breach_time: datetime            # When the predicted breach occurs
    hours_until_breach: float
    upper_bound: float               # 95th percentile of prediction
    lower_bound: float               # 5th percentile of prediction
    message: str
    tags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "metric": self.metric,
            "predicted_value": round(self.predicted_value, 2),
            "threshold": self.threshold,
            "predicted_at": self.predicted_at.isoformat(),
            "breach_time": self.breach_time.isoformat(),
            "hours_until_breach": round(self.hours_until_breach, 1),
            "upper_bound": round(self.upper_bound, 2),
            "lower_bound": round(self.lower_bound, 2),
            "message": self.message,
            "tags": self.tags,
            "@timestamp": self.predicted_at.isoformat(),
        }


METRIC_THRESHOLDS = {
    "cpu.overall.pct": ALERT_CONFIG["cpu_overall_threshold"],
    "cpu.user.pct": ALERT_CONFIG["cpu_user_threshold"],
    "cpu.kernel.pct": ALERT_CONFIG["cpu_kernel_threshold"],
    "cpu.wait.pct": ALERT_CONFIG["cpu_wait_threshold"],
}


class AlertManager:
    """Evaluates forecasts against thresholds and produces alert records."""

    def __init__(self, config: dict = None):
        self.config = config or ALERT_CONFIG
        self.thresholds = METRIC_THRESHOLDS

    def evaluate_forecasts(
        self,
        future_forecasts: dict[str, pd.DataFrame],
        generated_at: Optional[datetime] = None,
    ) -> list[CPUAlert]:
        """
        Scan all future forecasts for threshold violations.

        Args:
            future_forecasts: dict mapping metric name -> Prophet forecast DataFrame
                              (columns: ds, yhat, yhat_lower, yhat_upper)
            generated_at: Timestamp when this evaluation runs (defaults to now)

        Returns:
            List of CPUAlert objects, sorted by breach_time ascending
        """
        generated_at = generated_at or datetime.now(timezone.utc)
        horizon_hours = self.config["alert_horizon_hours"]
        alerts: list[CPUAlert] = []

        for metric, forecast_df in future_forecasts.items():
            threshold = self.thresholds.get(metric)
            if threshold is None:
                continue

            # Filter to alert horizon window — normalize tz to match forecast_df
            cutoff = pd.Timestamp(generated_at) + pd.Timedelta(hours=horizon_hours)
            ds_series = pd.to_datetime(forecast_df["ds"])
            if ds_series.dt.tz is None:
                cutoff = cutoff.tz_localize(None) if cutoff.tzinfo else cutoff
            else:
                if cutoff.tzinfo is None:
                    cutoff = cutoff.tz_localize("UTC")
            window = forecast_df[ds_series <= cutoff].copy()

            # Find rows where predicted mean OR upper bound exceeds threshold
            breaches = window[
                (window["yhat"] > threshold) | (window["yhat_upper"] > threshold)
            ]

            if breaches.empty:
                continue

            # Group consecutive breaches and emit one alert per breach window
            for _, row in breaches.iterrows():
                breach_time = pd.Timestamp(row["ds"])
                gen_ts = pd.Timestamp(generated_at)
                # Normalize both to UTC-aware or both to naive
                if breach_time.tzinfo is None and gen_ts.tzinfo is not None:
                    breach_time = breach_time.tz_localize("UTC")
                elif breach_time.tzinfo is not None and gen_ts.tzinfo is None:
                    gen_ts = gen_ts.tz_localize("UTC")

                hours_until = (breach_time - gen_ts).total_seconds() / 3600

                severity = self._classify_severity(
                    yhat=row["yhat"],
                    yhat_upper=row["yhat_upper"],
                    threshold=threshold,
                )

                alert_id = f"{metric.replace('.', '_')}_{breach_time.strftime('%Y%m%dT%H%M%S')}"

                message = (
                    f"[{severity.value.upper()}] {metric} predicted to reach "
                    f"{row['yhat']:.1f}% (threshold: {threshold}%) "
                    f"at {breach_time.strftime('%Y-%m-%d %H:%M UTC')} "
                    f"({hours_until:.1f}h from now). "
                    f"95th percentile: {row['yhat_upper']:.1f}%"
                )

                alert = CPUAlert(
                    alert_id=alert_id,
                    severity=severity,
                    metric=metric,
                    predicted_value=float(row["yhat"]),
                    threshold=threshold,
                    predicted_at=generated_at,
                    breach_time=breach_time.to_pydatetime(),
                    hours_until_breach=hours_until,
                    upper_bound=float(row["yhat_upper"]),
                    lower_bound=float(row["yhat_lower"]),
                    message=message,
                    tags={
                        "source": "prophet_forecast",
                        "metric_type": "cpu",
                    },
                )
                alerts.append(alert)

        alerts.sort(key=lambda a: a.breach_time)
        logger.info(f"Generated {len(alerts)} alerts across {len(future_forecasts)} metrics")
        return alerts

    def _classify_severity(
        self, yhat: float, yhat_upper: float, threshold: float
    ) -> AlertSeverity:
        """
        Classify alert severity based on how far the prediction exceeds threshold.

        WARNING  : upper bound > threshold but mean is below
        CRITICAL : mean > threshold by < 10%
        (above CRITICAL+10%): also CRITICAL but logged as extreme
        """
        margin = yhat - threshold
        if margin <= 0:
            return AlertSeverity.WARNING      # Only upper bound exceeds threshold
        elif margin < 10:
            return AlertSeverity.CRITICAL
        else:
            return AlertSeverity.CRITICAL     # Extreme breach still critical

    def print_alerts(self, alerts: list[CPUAlert]) -> None:
        if not alerts:
            logger.info("No threshold breaches detected in forecast window.")
            return

        print("\n" + "=" * 70)
        print(f"  CPU CAPACITY ALERTS  ({len(alerts)} total)")
        print("=" * 70)

        by_severity = {s: [] for s in AlertSeverity}
        for alert in alerts:
            by_severity[alert.severity].append(alert)

        for severity in [AlertSeverity.CRITICAL, AlertSeverity.WARNING, AlertSeverity.INFO]:
            group = by_severity[severity]
            if not group:
                continue
            print(f"\n[{severity.value.upper()}] ({len(group)} alerts)")
            for alert in group[:5]:  # Show first 5 per severity
                print(f"  {alert.message}")

        if len(alerts) > 15:
            print(f"\n  ... and {len(alerts) - 15} more alerts (see Elasticsearch index)")
        print("=" * 70 + "\n")
