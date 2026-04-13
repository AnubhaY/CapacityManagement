"""
Sample memory timeseries data generator.

Generates realistic server memory usage data with columns matching
the Elastic APM / Metricbeat schema:
  - @timestamp
  - memory.application       bytes used by application processes
  - system.memory.total      total installed RAM in bytes
  - system.memory.actual.used.pct   actual used percentage (0-100)

Patterns modelled:
  - Slow upward trend (memory leak / growing workload)
  - Daily seasonality (applications allocate more during business hours)
  - Mild weekly pattern (weekday vs. weekend)
  - Occasional OOM/restart events (sharp drop followed by re-accumulation)
  - Random noise (smaller variance than CPU — memory is smoother)
"""

import logging
import numpy as np
import pandas as pd

from config import MEMORY_SAMPLE_DATA_CONFIG, LOGGING_CONFIG

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# 1 GB in bytes
GB = 1024 ** 3


def generate_memory_data(
    start_date: str = None,
    end_date: str = None,
    freq: str = None,
    total_memory_gb: int = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic memory usage timeseries data.

    Returns a DataFrame with columns:
        @timestamp,
        memory.application,           (bytes)
        system.memory.total,          (bytes)
        system.memory.actual.used.pct (%)
    """
    cfg = MEMORY_SAMPLE_DATA_CONFIG
    start_date       = start_date       or cfg["start_date"]
    end_date         = end_date         or cfg["end_date"]
    freq             = freq             or cfg["freq"]
    total_memory_gb  = total_memory_gb  or cfg["total_memory_gb"]

    np.random.seed(seed)

    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(timestamps)
    total_memory_bytes = total_memory_gb * GB

    logger.info(
        f"Generating {n} memory data points from {start_date} to {end_date} "
        f"at {freq} intervals  |  RAM: {total_memory_gb} GB"
    )

    # ------------------------------------------------------------------ #
    # 1. Long-term trend: slow memory leak / workload growth              #
    # ------------------------------------------------------------------ #
    # +leak_rate_per_day per day across the full window
    days_total = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
    leak_per_point = cfg["leak_rate_per_day"] / (24 * 60 / _freq_minutes(freq))
    trend = np.linspace(0, days_total * cfg["leak_rate_per_day"], n)

    # ------------------------------------------------------------------ #
    # 2. Daily seasonality (business-hours peak)                          #
    # ------------------------------------------------------------------ #
    hour = timestamps.hour + timestamps.minute / 60.0
    peak_h = cfg["daily_peak_hour"]
    daily_pattern = (
        8.0  * np.exp(-0.5 * ((hour - peak_h)   / 3.5) ** 2)   # afternoon peak
        + 4.0  * np.exp(-0.5 * ((hour - 10.0)    / 2.0) ** 2)   # morning ramp
        - 3.5  * np.exp(-0.5 * ((hour - 4.0)     / 2.0) ** 2)   # night-time low
    )

    # ------------------------------------------------------------------ #
    # 3. Weekly seasonality                                               #
    # ------------------------------------------------------------------ #
    day_of_week = timestamps.dayofweek
    weekly_multiplier = np.where(day_of_week < 5, 1.0, 0.70)   # weekends ~70%

    # ------------------------------------------------------------------ #
    # 4. Random noise (Gaussian — tighter than CPU)                       #
    # ------------------------------------------------------------------ #
    noise = np.random.normal(0, cfg["noise_std"], n)

    # ------------------------------------------------------------------ #
    # 5. OOM-style memory spikes (sudden jump, then restart drop)         #
    # ------------------------------------------------------------------ #
    spike_mask = np.random.random(n) < cfg["spike_probability"]
    spikes = spike_mask * np.random.uniform(
        cfg["spike_magnitude"] * 0.5,
        cfg["spike_magnitude"],
        n,
    )

    # After a spike there is often a restart (sharp drop back to base)
    restart_drops = np.zeros(n)
    spike_indices = np.where(spike_mask)[0]
    for idx in spike_indices:
        drop_at = min(idx + int(10 / _freq_minutes(freq)), n - 1)  # ~10 min later
        restart_drops[drop_at] = -cfg["spike_magnitude"] * 1.2

    # ------------------------------------------------------------------ #
    # 6. Compose used percentage                                          #
    # ------------------------------------------------------------------ #
    used_pct = (
        cfg["base_used_pct"]
        + trend
        + daily_pattern * weekly_multiplier
        + noise
        + spikes
        + restart_drops
    )
    used_pct = np.clip(used_pct, 5.0, 99.0)

    # ------------------------------------------------------------------ #
    # 7. Derive byte-level fields                                         #
    # ------------------------------------------------------------------ #
    used_bytes = (used_pct / 100.0) * total_memory_bytes

    # Application memory ≈ 65-75% of total used (OS/kernel uses the rest)
    app_fraction = np.random.uniform(0.65, 0.75, n)
    app_bytes = used_bytes * app_fraction + np.random.normal(0, 50 * 1024 * 1024, n)
    app_bytes = np.clip(app_bytes, 0, used_bytes)

    df = pd.DataFrame({
        "@timestamp":                    timestamps,
        "memory.application":            np.round(app_bytes).astype(np.int64),
        "system.memory.total":           int(total_memory_bytes),
        "system.memory.actual.used.pct": np.round(used_pct, 2),
    })

    logger.info(
        f"Generated memory data | used_pct mean={df['system.memory.actual.used.pct'].mean():.1f}%  "
        f"max={df['system.memory.actual.used.pct'].max():.1f}%"
    )
    return df


def _freq_minutes(freq: str) -> float:
    """Return the number of minutes represented by a pandas frequency string."""
    mapping = {
        "1min": 1, "5min": 5, "10min": 10, "15min": 15,
        "30min": 30, "h": 60, "1h": 60, "2h": 120,
    }
    return mapping.get(freq, 5)


def save_sample_data(path: str = "sample_memory_data.csv") -> pd.DataFrame:
    df = generate_memory_data()
    df.to_csv(path, index=False)
    logger.info(f"Sample memory data saved to {path}")
    return df


if __name__ == "__main__":
    df = save_sample_data()
    print(df.head(10).to_string())
    print(f"\nShape: {df.shape}")
    print(f"\nStats:\n{df.describe()}")
