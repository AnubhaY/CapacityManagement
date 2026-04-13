"""
Configuration settings for Capacity Management (CPU + Memory)
"""

# Elasticsearch settings
ELASTICSEARCH_CONFIG = {
    "hosts": ["http://localhost:9200"],
    "username": "elastic",
    "password": "changeme",
    "verify_certs": False,
    # CPU indices
    "index_metrics": "cpu-metrics",
    "index_predictions": "cpu-predictions",
    "index_alerts": "cpu-alerts",
    # Memory indices
    "index_memory_metrics": "memory-metrics",
    "index_memory_predictions": "memory-predictions",
    "index_memory_alerts": "memory-alerts",
}

# Alert thresholds
ALERT_CONFIG = {
    "cpu_overall_threshold": 80.0,   # Alert when predicted CPU > 80%
    "cpu_user_threshold": 70.0,
    "cpu_kernel_threshold": 30.0,
    "cpu_wait_threshold": 20.0,
    "alert_horizon_hours": 24,       # How far ahead to check for threshold breaches
}

# Prediction settings
PREDICTION_CONFIG = {
    "forecast_periods": 48,          # Predict 48 hours ahead (in hourly intervals)
    "forecast_freq": "h",            # Hourly forecasts
    "historical_days": 90,           # Use 90 days of historical data for training
    "seasonality_mode": "multiplicative",  # multiplicative or additive
    "changepoint_prior_scale": 0.05, # Flexibility of trend changepoints
    "seasonality_prior_scale": 10.0, # Strength of seasonality
    "interval_width": 0.95,          # Confidence interval width (95%)
    "uncertainty_samples": 1000,
}

# Sample data generation settings
SAMPLE_DATA_CONFIG = {
    "start_date": "2025-01-01",
    "end_date": "2026-04-11",
    "freq": "5min",                  # 5-minute intervals
    "num_cores": 32,
    "base_cpu_pct": 35.0,
    "daily_peak_hour": 14,           # Peak usage at 2 PM
    "weekly_peak_day": 1,            # Peak on Tuesday (0=Monday)
    "noise_std": 5.0,
    "spike_probability": 0.01,       # 1% chance of usage spike
    "spike_magnitude": 30.0,
}

# Memory alert thresholds
MEMORY_ALERT_CONFIG = {
    "memory_used_pct_threshold": 80.0,       # Alert when predicted used% > 80%
    "memory_application_threshold_pct": 70.0, # App memory > 70% of total
    "alert_horizon_hours": 24,
}

# Memory sample data generation settings
MEMORY_SAMPLE_DATA_CONFIG = {
    "start_date": "2025-01-01",
    "end_date": "2026-04-11",
    "freq": "5min",
    "total_memory_gb": 32,          # 32 GB server RAM
    "base_used_pct": 45.0,          # Base memory utilisation
    "daily_peak_hour": 14,          # Peak usage at 2 PM
    "weekly_peak_day": 1,           # Peak on Tuesday
    "noise_std": 3.0,               # Smaller noise than CPU (memory is smoother)
    "spike_probability": 0.005,     # 0.5% chance of memory spike (OOM event)
    "spike_magnitude": 20.0,        # Size of spike in %
    "leak_rate_per_day": 0.05,      # Slow memory-leak trend: +0.05% per day
}

# Logging
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    "datefmt": "%Y-%m-%d %H:%M:%S",
}
