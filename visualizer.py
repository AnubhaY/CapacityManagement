"""
Visualization utilities for CPU and Memory capacity predictions.

Generates matplotlib plots showing:
  - Historical usage vs MSTL+AutoARIMA forecast
  - Confidence intervals (yhat_lower / yhat_upper)
  - Threshold breach lines
  - Rolling-mean trend
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe in headless environments)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from config import ALERT_CONFIG, MEMORY_ALERT_CONFIG, LOGGING_CONFIG

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

THRESHOLD_MAP = {
    # CPU
    "cpu.overall.pct": ALERT_CONFIG["cpu_overall_threshold"],
    "cpu.user.pct":    ALERT_CONFIG["cpu_user_threshold"],
    "cpu.kernel.pct":  ALERT_CONFIG["cpu_kernel_threshold"],
    "cpu.wait.pct":    ALERT_CONFIG["cpu_wait_threshold"],
    # Memory
    "system.memory.actual.used.pct": MEMORY_ALERT_CONFIG["memory_used_pct_threshold"],
    "memory.application.pct":        MEMORY_ALERT_CONFIG["memory_application_threshold_pct"],
}

COLORS = {
    # CPU
    "cpu.overall.pct": "#E74C3C",
    "cpu.user.pct":    "#3498DB",
    "cpu.kernel.pct":  "#F39C12",
    "cpu.wait.pct":    "#2ECC71",
    # Memory
    "system.memory.actual.used.pct": "#9B59B6",
    "memory.application.pct":        "#1ABC9C",
}

# Human-readable y-axis labels per metric
YLABEL_MAP = {
    "cpu.overall.pct":               "CPU %",
    "cpu.user.pct":                  "CPU %",
    "cpu.kernel.pct":                "CPU %",
    "cpu.wait.pct":                  "CPU %",
    "system.memory.actual.used.pct": "Memory used %",
    "memory.application.pct":        "App memory %",
}


def plot_forecast(
    historical_df: pd.DataFrame,
    forecast_results: dict,
    output_dir: str = "plots",
    lookback_days: int = 14,
) -> list[str]:
    """
    Generate one PNG per metric showing historical data + forecast + threshold.

    Args:
        historical_df: Raw CPU data (hourly resampled ideally)
        forecast_results: dict metric -> ForecastResult
        output_dir: Directory to write PNGs
        lookback_days: How many past days to show in the chart

    Returns:
        List of file paths written
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    written = []

    for metric, result in forecast_results.items():
        if result.forecast_df.empty:
            continue

        color = COLORS.get(metric, "#7F8C8D")
        threshold = THRESHOLD_MAP.get(metric)

        fig, axes = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1])
        ax_main, ax_trend = axes

        # ---- Historical data ----
        hist = historical_df[["@timestamp", metric]].copy()
        hist["@timestamp"] = pd.to_datetime(hist["@timestamp"])
        hist = hist.set_index("@timestamp").resample("h").mean()
        cutoff = hist.index.max() - pd.Timedelta(days=lookback_days)
        hist_window = hist[hist.index >= cutoff]

        ax_main.plot(
            hist_window.index,
            hist_window[metric],
            color=color,
            alpha=0.6,
            linewidth=1.0,
            label="Historical",
        )

        # ---- Forecast ----
        forecast = result.forecast_df.copy()
        forecast["ds"] = pd.to_datetime(forecast["ds"])
        last_hist = hist.index.max()
        future = forecast[forecast["ds"] > last_hist]

        ax_main.plot(
            future["ds"],
            future["yhat"],
            color=color,
            linewidth=2.0,
            linestyle="--",
            label="Forecast (yhat)",
        )
        ax_main.fill_between(
            future["ds"],
            future["yhat_lower"],
            future["yhat_upper"],
            color=color,
            alpha=0.15,
            label="95% CI",
        )

        # ---- Threshold line ----
        if threshold is not None:
            all_dates = list(hist_window.index) + list(future["ds"])
            ax_main.axhline(
                threshold,
                color="red",
                linestyle=":",
                linewidth=1.5,
                label=f"Alert threshold ({threshold}%)",
            )

            # Shade predicted breaches
            breach = future[future["yhat"] > threshold]
            if not breach.empty:
                ax_main.fill_between(
                    breach["ds"],
                    threshold,
                    breach["yhat"],
                    color="red",
                    alpha=0.25,
                    label="Predicted breach",
                )

        ax_main.set_ylabel(YLABEL_MAP.get(metric, "Usage %"), fontsize=11)
        ax_main.set_ylim(0, 105)
        ax_main.set_title(f"{metric} — Historical + 48h Forecast", fontsize=13, fontweight="bold")
        ax_main.legend(loc="upper left", fontsize=9)
        ax_main.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %H:%M"))
        ax_main.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax_main.grid(True, alpha=0.3)

        # ---- Rolling mean as trend proxy ----
        combined = pd.concat([
            hist_window.rename(columns={metric: "y"}),
            future[["ds", "yhat"]].rename(columns={"ds": "@timestamp", "yhat": "y"}).set_index("@timestamp"),
        ])
        window_pts = min(48, len(hist_window) // 4)
        if window_pts > 1:
            rolling_mean = hist_window[metric].rolling(window=window_pts, center=True).mean()
            ax_trend.plot(hist_window.index, rolling_mean, color="#7F8C8D", linewidth=1.5, label=f"{window_pts}h rolling mean")
            ax_trend.axvline(last_hist, color="gray", linestyle="--", alpha=0.7, label="Forecast start")
            ax_trend.plot(future["ds"], future["yhat"], color=color, linewidth=1.2, linestyle="--", alpha=0.7)
        ax_trend.set_ylabel("Rolling avg", fontsize=10)
        ax_trend.set_title("Trend (rolling mean)", fontsize=10)
        ax_trend.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax_trend.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        plt.setp(ax_trend.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax_trend.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = f"{output_dir}/{metric.replace('.', '_')}_forecast.png"
        plt.savefig(filename, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(filename)
        logger.info(f"Plot saved: {filename}")

    return written


def plot_memory_forecast(
    historical_df: pd.DataFrame,
    forecast_results: dict,
    output_dir: str = "plots",
    lookback_days: int = 14,
) -> list[str]:
    """
    Generate one PNG per memory metric showing historical + forecast + threshold.

    historical_df must contain: @timestamp, system.memory.actual.used.pct,
    and optionally memory.application.pct.
    """
    # Derive application.pct column if raw bytes are present but pct is missing
    if (
        "memory.application.pct" not in historical_df.columns
        and "memory.application" in historical_df.columns
        and "system.memory.total" in historical_df.columns
    ):
        total = historical_df["system.memory.total"].replace(0, np.nan)
        historical_df = historical_df.copy()
        historical_df["memory.application.pct"] = (
            historical_df["memory.application"] / total * 100
        ).clip(0, 100)

    return plot_forecast(historical_df, forecast_results, output_dir, lookback_days)


def plot_alert_summary(alerts: list, output_dir: str = "plots") -> str:
    """Bar chart showing alert count by metric and severity."""
    if not alerts:
        return ""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metrics = list({a.metric for a in alerts})
    severities = ["warning", "critical"]
    counts = {
        sev: [sum(1 for a in alerts if a.metric == m and a.severity.value == sev) for m in metrics]
        for sev in severities
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax.bar(x - width / 2, counts["warning"], width, label="Warning", color="#F39C12")
    bars2 = ax.bar(x + width / 2, counts["critical"], width, label="Critical", color="#E74C3C")

    ax.set_xticks(x)
    ax.set_xticklabels([
        m.replace("cpu.", "").replace("system.memory.", "mem.").replace(".pct", "")
        for m in metrics
    ])
    ax.set_ylabel("Alert count")
    ax.set_title("Predicted Threshold Breaches — Next 24 Hours")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if h > 0:
            ax.annotate(str(int(h)), xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    path = f"{output_dir}/alert_summary.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Alert summary plot saved: {path}")
    return path
