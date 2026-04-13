"""
Sample CPU timeseries data generator.

Generates realistic CPU usage data with:
- Daily seasonality (business hours peak)
- Weekly seasonality (weekday vs weekend)
- Long-term trend (gradual growth)
- Random spikes (unexpected load)
- Correlated metrics (user, kernel, wait, overall)
"""

import logging
import numpy as np
import pandas as pd

from config import SAMPLE_DATA_CONFIG, LOGGING_CONFIG

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


def generate_cpu_data(
    start_date: str = None,
    end_date: str = None,
    freq: str = None,
    num_cores: int = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic CPU usage timeseries data.

    Returns a DataFrame with columns:
        @timestamp, cpu.overall.pct, cpu.user.pct,
        cpu.kernel.pct, cpu.wait.pct, cpu.cores
    """
    cfg = SAMPLE_DATA_CONFIG
    start_date = start_date or cfg["start_date"]
    end_date = end_date or cfg["end_date"]
    freq = freq or cfg["freq"]
    num_cores = num_cores or cfg["num_cores"]

    np.random.seed(seed)

    timestamps = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(timestamps)
    logger.info(f"Generating {n} data points from {start_date} to {end_date} at {freq} intervals")

    # --- Base trend: gradual CPU growth over time (capacity filling up) ---
    trend = np.linspace(0, 15, n)  # +15% over the full period

    # --- Daily seasonality ---
    hour = timestamps.hour + timestamps.minute / 60.0
    # Gaussian peak at 14:00 with width ~3 hours
    daily_pattern = (
        12.0 * np.exp(-0.5 * ((hour - cfg["daily_peak_hour"]) / 3.0) ** 2)
        + 5.0 * np.exp(-0.5 * ((hour - 10.0) / 2.0) ** 2)   # morning ramp
        - 5.0 * np.exp(-0.5 * ((hour - 3.0) / 2.0) ** 2)    # night-time low
    )

    # --- Weekly seasonality ---
    day_of_week = timestamps.dayofweek  # 0=Monday
    weekly_multiplier = np.where(day_of_week < 5, 1.0, 0.55)  # weekends ~55%

    # --- Random noise ---
    noise = np.random.normal(0, cfg["noise_std"], n)

    # --- Random spikes (e.g., batch jobs, deployments) ---
    spike_mask = np.random.random(n) < cfg["spike_probability"]
    spikes = spike_mask * np.random.uniform(
        cfg["spike_magnitude"], cfg["spike_magnitude"] * 2, n
    )

    # --- Compose overall CPU ---
    overall = (
        cfg["base_cpu_pct"]
        + trend
        + daily_pattern * weekly_multiplier
        + noise
        + spikes
    )
    overall = np.clip(overall, 1.0, 99.0)

    # --- Derive correlated metrics ---
    # user.pct: majority of overall (applications)
    user_fraction = np.random.uniform(0.60, 0.75, n)
    user_pct = overall * user_fraction + np.random.normal(0, 1.5, n)

    # kernel.pct: system calls, I/O interrupts
    kernel_fraction = np.random.uniform(0.08, 0.18, n)
    kernel_pct = overall * kernel_fraction + np.random.normal(0, 0.8, n)

    # wait.pct: I/O wait (inversely correlated with fast SSDs, spikes during disk ops)
    wait_base = np.random.uniform(0.5, 3.0, n)
    wait_spikes = (np.random.random(n) < 0.02) * np.random.uniform(5, 20, n)
    wait_pct = wait_base + wait_spikes

    # Clamp all metrics
    user_pct = np.clip(user_pct, 0.0, overall)
    kernel_pct = np.clip(kernel_pct, 0.0, overall - user_pct)
    wait_pct = np.clip(wait_pct, 0.0, 25.0)

    df = pd.DataFrame({
        "@timestamp": timestamps,
        "cpu.overall.pct": np.round(overall, 2),
        "cpu.user.pct": np.round(user_pct, 2),
        "cpu.kernel.pct": np.round(kernel_pct, 2),
        "cpu.wait.pct": np.round(wait_pct, 2),
        "cpu.cores": num_cores,
    })

    logger.info(
        f"Generated data | overall mean={df['cpu.overall.pct'].mean():.1f}% "
        f"max={df['cpu.overall.pct'].max():.1f}%"
    )
    return df


def save_sample_data(path: str = "sample_cpu_data.csv") -> pd.DataFrame:
    df = generate_cpu_data()
    df.to_csv(path, index=False)
    logger.info(f"Sample data saved to {path}")
    return df


if __name__ == "__main__":
    df = save_sample_data()
    print(df.head(10).to_string())
    print(f"\nShape: {df.shape}")
    print(f"\nStats:\n{df.describe()}")
