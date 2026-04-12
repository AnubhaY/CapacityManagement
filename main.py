"""
Capacity Management Pipeline — CPU + Memory

Flow (per mode):
  1. Load (or generate) historical data
  2. Train MSTL + AutoARIMA models for each metric
  3. Generate 48-hour forecast
  4. Evaluate predictions against alert thresholds (>80%)
  5. Store metrics, predictions, and alerts in Elasticsearch
  6. Produce forecast plots

Usage examples:
  .venv/Scripts/python main.py                        # run both CPU and memory
  .venv/Scripts/python main.py --mode cpu
  .venv/Scripts/python main.py --mode memory
  .venv/Scripts/python main.py --mode all --store-es
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import LOGGING_CONFIG, PREDICTION_CONFIG

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


# ===========================================================================
# CPU pipeline
# ===========================================================================

def _load_cpu_data(args) -> pd.DataFrame:
    from config import SAMPLE_DATA_CONFIG
    from data_generator import generate_cpu_data

    if args.cpu_data_file and Path(args.cpu_data_file).exists():
        logger.info(f"Loading CPU data from {args.cpu_data_file}")
        df = pd.read_csv(args.cpu_data_file, parse_dates=["@timestamp"])
    else:
        logger.info("Generating synthetic CPU sample data...")
        df = generate_cpu_data()
        df.to_csv("sample_cpu_data.csv", index=False)
        logger.info("Saved to sample_cpu_data.csv")

    if args.historical_days:
        cutoff = df["@timestamp"].max() - pd.Timedelta(days=args.historical_days)
        df = df[df["@timestamp"] >= cutoff]

    logger.info(
        f"CPU data: {len(df)} records | "
        f"{df['@timestamp'].min()} → {df['@timestamp'].max()}"
    )
    return df


def _run_cpu_pipeline(args) -> tuple:
    """Returns (predictor, future_forecasts, alerts)."""
    from predictor import CPUPredictor
    from alert_manager import AlertManager

    df = _load_cpu_data(args)

    PREDICTION_CONFIG["forecast_periods"] = args.forecast_periods
    predictor = CPUPredictor()
    predictor.train(df, run_cv=args.cross_validate)
    predictor.predict()
    future_forecasts = predictor.get_future_forecast()

    for metric, fdf in future_forecasts.items():
        if not fdf.empty:
            logger.info(
                f"  CPU {metric}: mean={fdf['yhat'].mean():.1f}% "
                f"max={fdf['yhat'].max():.1f}%"
            )

    alerts = AlertManager().evaluate_forecasts(
        future_forecasts, generated_at=datetime.now(timezone.utc)
    )
    AlertManager().print_alerts(alerts)

    if args.cross_validate:
        print("\nCPU Model Accuracy:")
        print(predictor.get_model_summary().to_string(index=False))

    return df, predictor, future_forecasts, alerts


# ===========================================================================
# Memory pipeline
# ===========================================================================

def _load_memory_data(args) -> pd.DataFrame:
    from memory_data_generator import generate_memory_data

    if args.memory_data_file and Path(args.memory_data_file).exists():
        logger.info(f"Loading memory data from {args.memory_data_file}")
        df = pd.read_csv(args.memory_data_file, parse_dates=["@timestamp"])
    else:
        logger.info("Generating synthetic memory sample data...")
        df = generate_memory_data()
        df.to_csv("sample_memory_data.csv", index=False)
        logger.info("Saved to sample_memory_data.csv")

    if args.historical_days:
        cutoff = df["@timestamp"].max() - pd.Timedelta(days=args.historical_days)
        df = df[df["@timestamp"] >= cutoff]

    logger.info(
        f"Memory data: {len(df)} records | "
        f"{df['@timestamp'].min()} → {df['@timestamp'].max()}"
    )
    return df


def _run_memory_pipeline(args) -> tuple:
    """Returns (df, predictor, future_forecasts, alerts)."""
    from memory_predictor import MemoryPredictor
    from memory_alert_manager import MemoryAlertManager

    df = _load_memory_data(args)

    PREDICTION_CONFIG["forecast_periods"] = args.forecast_periods
    predictor = MemoryPredictor()
    predictor.train(df, run_cv=args.cross_validate)
    predictor.predict()
    future_forecasts = predictor.get_future_forecast()

    for metric, fdf in future_forecasts.items():
        if not fdf.empty:
            logger.info(
                f"  Memory {metric}: mean={fdf['yhat'].mean():.1f}% "
                f"max={fdf['yhat'].max():.1f}%"
            )

    alerts = MemoryAlertManager().evaluate_forecasts(
        future_forecasts, generated_at=datetime.now(timezone.utc)
    )
    MemoryAlertManager().print_alerts(alerts)

    if args.cross_validate:
        print("\nMemory Model Accuracy:")
        print(predictor.get_model_summary().to_string(index=False))

    return df, predictor, future_forecasts, alerts


# ===========================================================================
# Elasticsearch storage
# ===========================================================================

def _store_cpu_in_es(es_client, df, predictor, alerts, skip_metrics):
    from config import SAMPLE_DATA_CONFIG

    if not skip_metrics:
        df_hourly = df.set_index("@timestamp").resample("h").mean().reset_index()
        logger.info(f"Indexing {len(df_hourly)} CPU hourly metric records...")
        es_client.index_metrics_data(df_hourly)

    logger.info("Indexing CPU predictions...")
    es_client.index_predictions_data(predictor.models, predictor._last_training_dates)

    logger.info(f"Indexing {len(alerts)} CPU alerts...")
    es_client.index_alerts_data(alerts)


def _store_memory_in_es(es_client, df, predictor, alerts, skip_metrics):
    if not skip_metrics:
        df_hourly = df.set_index("@timestamp").resample("h").mean().reset_index()
        logger.info(f"Indexing {len(df_hourly)} memory hourly metric records...")
        es_client.index_memory_metrics_data(df_hourly)

    logger.info("Indexing memory predictions...")
    es_client.index_memory_predictions_data(predictor.models, predictor._last_training_dates)

    logger.info(f"Indexing {len(alerts)} memory alerts...")
    es_client.index_memory_alerts_data(alerts)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Capacity Management — MSTL+AutoARIMA forecasting for CPU and Memory"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["cpu", "memory", "all"],
        default="all",
        help="Which pipeline to run: cpu, memory, or all (default: all)",
    )
    parser.add_argument(
        "--cpu-data-file",
        default=None,
        help="Path to CSV with historical CPU data. Omit to generate synthetic data.",
    )
    parser.add_argument(
        "--memory-data-file",
        default=None,
        help="Path to CSV with historical memory data "
             "(columns: @timestamp, memory.application, system.memory.total, "
             "system.memory.actual.used.pct). Omit to generate synthetic data.",
    )
    parser.add_argument(
        "--historical-days",
        type=int,
        default=PREDICTION_CONFIG["historical_days"],
        help=f"Days of history to use for training (default: {PREDICTION_CONFIG['historical_days']})",
    )
    parser.add_argument(
        "--forecast-periods",
        type=int,
        default=PREDICTION_CONFIG["forecast_periods"],
        help="Number of future hourly periods to forecast (default: 48)",
    )
    parser.add_argument(
        "--store-es",
        action="store_true",
        default=False,
        help="Index results into Elasticsearch",
    )
    parser.add_argument(
        "--skip-metrics-index",
        action="store_true",
        default=False,
        help="Skip indexing historical metrics (faster — only predictions + alerts)",
    )
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        default=False,
        help="Run train/test split accuracy evaluation (slower)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Skip plot generation",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info(f"  Capacity Management Pipeline  [mode={args.mode}]")
    logger.info("=" * 60)

    cpu_result    = None
    memory_result = None

    # ---- Run pipelines ----
    if args.mode in ("cpu", "all"):
        logger.info("\n--- CPU Pipeline ---")
        cpu_result = _run_cpu_pipeline(args)

    if args.mode in ("memory", "all"):
        logger.info("\n--- Memory Pipeline ---")
        memory_result = _run_memory_pipeline(args)

    # ---- Elasticsearch ----
    if args.store_es:
        from elasticsearch_client import ESClient
        es_client = ESClient()
        if not es_client.ping():
            logger.warning("Elasticsearch not reachable — skipping indexing.")
        else:
            es_client.setup_indices(mode=args.mode)
            if cpu_result:
                df_cpu, cpu_pred, _, cpu_alerts = cpu_result
                _store_cpu_in_es(es_client, df_cpu, cpu_pred, cpu_alerts, args.skip_metrics_index)
            if memory_result:
                df_mem, mem_pred, _, mem_alerts = memory_result
                _store_memory_in_es(es_client, df_mem, mem_pred, mem_alerts, args.skip_metrics_index)
            logger.info("Elasticsearch indexing complete.")
    else:
        logger.info("Skipping Elasticsearch (pass --store-es to enable)")

    # ---- Plots ----
    if not args.no_plots:
        from visualizer import plot_forecast, plot_memory_forecast, plot_alert_summary

        if cpu_result:
            df_cpu, cpu_pred, _, cpu_alerts = cpu_result
            files = plot_forecast(df_cpu, cpu_pred.models, output_dir="plots")
            for f in files:
                logger.info(f"  CPU plot: {f}")
            if cpu_alerts:
                plot_alert_summary(cpu_alerts, output_dir="plots")

        if memory_result:
            df_mem, mem_pred, _, mem_alerts = memory_result
            files = plot_memory_forecast(df_mem, mem_pred.models, output_dir="plots")
            for f in files:
                logger.info(f"  Memory plot: {f}")
            if mem_alerts:
                plot_alert_summary(mem_alerts, output_dir="plots")

    logger.info("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
