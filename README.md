# Capacity Management

Predicts server **CPU and memory** usage up to 48 hours ahead and alerts when forecasted load is expected to exceed configurable thresholds (default: 80%). Predictions and alerts are stored in Elasticsearch for dashboarding and incident response.

---

## Table of Contents

- [Overview](#overview)
- [Data Flow](#data-flow)
- [ML Algorithm](#ml-algorithm)
- [Project Structure](#project-structure)
- [Input Data Schema](#input-data-schema)
- [Elasticsearch Indices](#elasticsearch-indices)
- [Alert Severity Model](#alert-severity-model)
- [Quick Start](#quick-start)
- [Configuration Reference](#configuration-reference)
- [CLI Reference](#cli-reference)
- [Output](#output)

---

## Overview

```
Historical data  →  MSTL + AutoARIMA  →  48h forecast  →  Alert evaluation  →  Elasticsearch
  (CSV / ES)            (train)           (per metric)     (vs thresholds)       + PNG plots
```

Two independent pipelines share the same codebase, configuration, and Elasticsearch client:

### CPU metrics forecasted

| Metric | Description |
|---|---|
| `cpu.overall.pct` | Total CPU utilisation (all cores averaged) |
| `cpu.user.pct` | CPU time in user space (application code) |
| `cpu.kernel.pct` | CPU time in kernel space (syscalls, drivers) |
| `cpu.wait.pct` | CPU time waiting for I/O |

### Memory metrics forecasted

| Metric | Description |
|---|---|
| `system.memory.actual.used.pct` | Actual used memory as a percentage of total RAM **(primary alert metric)** |
| `memory.application.pct` | Application process memory as a percentage of total RAM (derived from bytes) |

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA SOURCES                                     │
│                                                                             │
│   CPU CSV / synthetic data          Memory CSV / synthetic data             │
│   --cpu-data-file                   --memory-data-file                      │
│   Columns:                          Columns:                                │
│   - @timestamp                      - @timestamp                            │
│   - cpu.overall.pct                 - memory.application      (bytes)       │
│   - cpu.user.pct                    - system.memory.total     (bytes)       │
│   - cpu.kernel.pct                  - system.memory.actual.used.pct (%)     │
│   - cpu.wait.pct                                                            │
└─────────────────┬───────────────────────────────┬───────────────────────────┘
                  │                               │
                  ▼                               ▼
        predictor.py                    memory_predictor.py
        CPUPredictor                    MemoryPredictor
                  │                               │
                  └───────────────┬───────────────┘
                                  │
               PREPROCESSING (both predictors, same steps)
               1. Filter to --historical-days window (default 90 days)
               2. Resample 5-min → hourly mean
               3. Clip values to [0, 100]
               4. Convert wide → long (unique_id, ds, y)
                                  │
                                  ▼
               MODEL TRAINING  (StatsForecast — parallel fit)
               ┌────────────────────────────────────────────┐
               │  MSTL  season_length=[24, 168]             │
               │    ├── STL(24)  → daily seasonal component  │
               │    ├── STL(168) → weekly seasonal component │
               │    └── Remainder → AutoARIMA(AICc stepwise) │
               └────────────────────────────────────────────┘
               ┌──────────────────────────────┐
               │  AutoETS  season_length=24   │  ← ensemble model
               └──────────────────────────────┘

               Final yhat = (MSTL+AutoARIMA + AutoETS) / 2
                                  │
               ┌──────────────────┴──────────────────┐
               │                                     │
               ▼                                     ▼
    ALERT EVALUATION                       ELASTICSEARCH INDEXING
    alert_manager.py                       elasticsearch_client.py
    memory_alert_manager.py
                                           CPU:
    For each forecast row                  - cpu-metrics
    in next 24h window:                    - cpu-predictions
                                           - cpu-alerts
    yhat_upper > threshold → WARNING
    yhat > threshold       → CRITICAL      Memory:
                                           - memory-metrics
                                           - memory-predictions
                                           - memory-alerts

               ▼
    VISUALISATION  (visualizer.py)
    - Per metric PNG:
      historical (solid) + forecast (dashed)
      + 95% CI band + threshold line + breach region
      + rolling mean trend panel
```

---

## ML Algorithm

**MSTL + AutoARIMA** (from [Nixtla statsforecast](https://github.com/Nixtla/statsforecast))

### Why this combination for capacity management

| Requirement | How it is met |
|---|---|
| Daily cycle (business hours peak) | MSTL `season_length=24` strips the 24h seasonal component |
| Weekly cycle (weekday vs weekend) | MSTL `season_length=168` strips the 7×24h seasonal component |
| Long-term capacity growth / memory-leak trend | AutoARIMA models the de-seasonalized trend |
| Confidence intervals for alerting | 95% prediction interval from MSTL+AutoARIMA residual variance |
| No compiler toolchain required | Pure Python/C — no Stan, no MinGW, no RTools |
| Multiple correlated metrics | StatsForecast fits all metrics in parallel (`n_jobs=-1`) |
| Automatic hyperparameter selection | AutoARIMA AICc stepwise search — no manual p,d,q tuning |

### Why not Prophet?

Prophet requires CmdStan (a C++ compiler toolchain). On Windows this means installing MinGW or Rtools, which is error-prone in CI/CD and containerised environments. MSTL+AutoARIMA achieves equal or better accuracy with zero native dependencies.

### How MSTL decomposition works

```
Original hourly series (CPU or Memory)
    │
    ├── STL decomposition (period=24)
    │     ├── Seasonal₂₄   (daily pattern — e.g. peak at 14:00)
    │     └── Trend₂₄ + Remainder
    │               │
    │               └── STL decomposition (period=168)
    │                     ├── Seasonal₁₆₈  (weekly pattern — weekdays vs weekend)
    │                     └── Trend + Residual
    │                               │
    │                               └── AutoARIMA fits this residual
    │
    Forecast = AutoARIMA_forecast + Seasonal₂₄ + Seasonal₁₆₈
```

### Ensemble

```
Final yhat = (MSTL+AutoARIMA  +  AutoETS) / 2
```

AutoETS (Exponential Smoothing) adds robustness: it weights recent observations more heavily, which helps when there are sudden shifts (e.g. a new deployment, a memory-intensive batch job) that the ARIMA trend has not yet adapted to.

---

## Project Structure

```
.
├── config.py                   # All thresholds, ES config, model hyperparameters (CPU + Memory)
│
├── data_generator.py           # Synthetic CPU data (15 months, 5-min, daily+weekly seasonality)
├── predictor.py                # CPUPredictor — MSTL+AutoARIMA training and forecasting
├── alert_manager.py            # CPUAlert objects; threshold evaluation
│
├── memory_data_generator.py    # Synthetic memory data (same period, same frequency)
├── memory_predictor.py         # MemoryPredictor — MSTL+AutoARIMA for memory metrics
├── memory_alert_manager.py     # MemoryAlert objects; threshold evaluation
│
├── elasticsearch_client.py     # Bulk indexing — all 6 indices (CPU + Memory)
├── visualizer.py               # Forecast PNG plots for CPU and Memory (matplotlib)
├── main.py                     # CLI pipeline — --mode cpu | memory | all
│
├── requirements.txt            # Python dependencies
│
├── sample_cpu_data.csv         # Auto-generated synthetic CPU data (created on first run)
├── sample_memory_data.csv      # Auto-generated synthetic memory data (created on first run)
│
└── plots/                      # Forecast PNGs (created on first run)
    ├── cpu_overall_pct_forecast.png
    ├── cpu_user_pct_forecast.png
    ├── cpu_kernel_pct_forecast.png
    ├── cpu_wait_pct_forecast.png
    ├── system_memory_actual_used_pct_forecast.png
    ├── memory_application_pct_forecast.png
    └── alert_summary.png
```

---

## Input Data Schema

### CPU data

| Column | Type | Description |
|---|---|---|
| `@timestamp` | ISO 8601 datetime | Measurement timestamp (UTC recommended) |
| `cpu.overall.pct` | float [0–100] | Total CPU usage percentage |
| `cpu.user.pct` | float [0–100] | User-space CPU percentage |
| `cpu.kernel.pct` | float [0–100] | Kernel-space CPU percentage |
| `cpu.wait.pct` | float [0–100] | I/O wait CPU percentage |
| `cpu.cores` | int | Number of CPU cores (informational, not forecasted) |

### Memory data

| Column | Type | Description |
|---|---|---|
| `@timestamp` | ISO 8601 datetime | Measurement timestamp (UTC recommended) |
| `memory.application` | int (bytes) | Memory used by application processes |
| `system.memory.total` | int (bytes) | Total installed RAM |
| `system.memory.actual.used.pct` | float [0–100] | Actual used memory percentage |

Any interval frequency is accepted (5 min, 1 min, 1 hour). Data is resampled to hourly means internally before training.

---

## Elasticsearch Indices

### CPU indices

#### `cpu-metrics`
Raw historical CPU data (hourly downsampled).

```json
{
  "@timestamp": "2026-04-10T14:00:00Z",
  "cpu.overall.pct": 58.4,
  "cpu.user.pct": 38.1,
  "cpu.kernel.pct": 9.2,
  "cpu.wait.pct": 1.8,
  "cpu.cores": 32
}
```

#### `cpu-predictions`
Forecast rows for all CPU metrics, including fitted history and future forecast.

```json
{
  "@timestamp": "2026-04-11T22:00:00Z",
  "forecast_time": "2026-04-12T14:00:00Z",
  "metric": "cpu.overall.pct",
  "yhat": 61.3,
  "yhat_lower": 54.2,
  "yhat_upper": 68.5,
  "model": "MSTL+AutoARIMA",
  "is_future": true,
  "threshold": 80.0,
  "threshold_breached": false
}
```

#### `cpu-alerts`
One document per predicted CPU threshold breach within the alert horizon window.

```json
{
  "@timestamp": "2026-04-11T22:00:00Z",
  "alert_id": "cpu_overall_pct_20260412T140000",
  "severity": "critical",
  "metric": "cpu.overall.pct",
  "predicted_value": 83.7,
  "threshold": 80.0,
  "breach_time": "2026-04-12T14:00:00Z",
  "hours_until_breach": 16.0,
  "upper_bound": 91.2,
  "lower_bound": 76.4,
  "message": "[CRITICAL] cpu.overall.pct predicted to reach 83.7% (threshold: 80.0%) at 2026-04-12 14:00 UTC (16.0h from now). 95th percentile: 91.2%",
  "tags": { "source": "mstl_autoarima_forecast", "metric_type": "cpu" }
}
```

---

### Memory indices

#### `memory-metrics`
Raw historical memory data (hourly downsampled).

```json
{
  "@timestamp": "2026-04-10T14:00:00Z",
  "memory.application": 12345678912,
  "system.memory.total": 34359738368,
  "system.memory.actual.used.pct": 62.4
}
```

#### `memory-predictions`
Forecast rows for all memory metrics.

```json
{
  "@timestamp": "2026-04-11T22:00:00Z",
  "forecast_time": "2026-04-12T14:00:00Z",
  "metric": "system.memory.actual.used.pct",
  "yhat": 78.6,
  "yhat_lower": 73.1,
  "yhat_upper": 84.2,
  "model": "MSTL+AutoARIMA",
  "is_future": true,
  "threshold": 80.0,
  "threshold_breached": false
}
```

#### `memory-alerts`
One document per predicted memory threshold breach.

```json
{
  "@timestamp": "2026-04-11T22:00:00Z",
  "alert_id": "mem_system_memory_actual_used_pct_20260412T140000",
  "severity": "warning",
  "metric": "system.memory.actual.used.pct",
  "predicted_value": 78.6,
  "threshold": 80.0,
  "breach_time": "2026-04-12T14:00:00Z",
  "hours_until_breach": 16.0,
  "upper_bound": 84.2,
  "lower_bound": 73.1,
  "message": "[WARNING] system.memory.actual.used.pct predicted to reach 78.6% (threshold: 80.0%) at 2026-04-12 14:00 UTC (16.0h from now). 95th percentile: 84.2%",
  "tags": { "source": "mstl_autoarima_forecast", "metric_type": "memory" }
}
```

---

## Alert Severity Model

The same severity model applies to both CPU and memory:

```
                     yhat_upper > threshold?
                              │
              ┌────── No ─────┤
              │               │
        No alert         yhat > threshold?
                              │
              ┌───── No ──────┴────── Yes ──────┐
              │                                 │
          WARNING                        yhat − threshold < 10%?
     (upper bound only                         │
      exceeds threshold)         ┌─── Yes ─────┴──── No ───┐
                                 │                          │
                              CRITICAL                   CRITICAL
                           (mean > threshold)       (extreme breach,
                                                    >10% over limit)
```

| Severity | Condition | Recommended action |
|---|---|---|
| `warning` | Only the 95th percentile (`yhat_upper`) exceeds threshold — mean is still below | Investigate; may be noise or a transient spike |
| `critical` | Predicted mean (`yhat`) exceeds threshold | Escalate; capacity action required within the breach window |

---

## Quick Start

### 1. Set up environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run both pipelines with synthetic data (no Elasticsearch needed)

```bash
python main.py
```

Generates `sample_cpu_data.csv` and `sample_memory_data.csv`, trains models, forecasts 48 hours ahead, prints any alerts, and saves plots to `plots/`.

### 3. Run a single pipeline

```bash
# CPU only
python main.py --mode cpu

# Memory only
python main.py --mode memory
```

### 4. Run with your own data

```bash
# CPU data
python main.py --mode cpu --cpu-data-file /path/to/cpu_data.csv

# Memory data
python main.py --mode memory --memory-data-file /path/to/memory_data.csv

# Both
python main.py --mode all \
    --cpu-data-file /path/to/cpu_data.csv \
    --memory-data-file /path/to/memory_data.csv
```

### 5. Store results in Elasticsearch

Configure the connection in `config.py`, then:

```bash
python main.py --store-es
```

### 6. Run accuracy evaluation

```bash
python main.py --cross-validate
```

Reports MAE, RMSE, and MAPE per metric using a 7-day hold-out test set.

---

## Configuration Reference

All settings live in [config.py](config.py).

### Elasticsearch

```python
ELASTICSEARCH_CONFIG = {
    "hosts":           ["http://localhost:9200"],
    "username":        "elastic",
    "password":        "changeme",
    "verify_certs":    False,
    # CPU indices
    "index_metrics":              "cpu-metrics",
    "index_predictions":          "cpu-predictions",
    "index_alerts":               "cpu-alerts",
    # Memory indices
    "index_memory_metrics":       "memory-metrics",
    "index_memory_predictions":   "memory-predictions",
    "index_memory_alerts":        "memory-alerts",
}
```

### CPU Alert Thresholds

```python
ALERT_CONFIG = {
    "cpu_overall_threshold": 80.0,   # %
    "cpu_user_threshold":    70.0,   # %
    "cpu_kernel_threshold":  30.0,   # %
    "cpu_wait_threshold":    20.0,   # %
    "alert_horizon_hours":   24,
}
```

### Memory Alert Thresholds

```python
MEMORY_ALERT_CONFIG = {
    "memory_used_pct_threshold":          80.0,   # system.memory.actual.used.pct %
    "memory_application_threshold_pct":   70.0,   # application memory as % of total
    "alert_horizon_hours":                24,
}
```

### Prediction Settings (shared)

```python
PREDICTION_CONFIG = {
    "forecast_periods":  48,      # hourly periods to forecast
    "historical_days":   90,      # days of history used for training
    "interval_width":    0.95,    # confidence interval (95%)
}
```

### Memory Sample Data

```python
MEMORY_SAMPLE_DATA_CONFIG = {
    "start_date":         "2025-01-01",
    "end_date":           "2026-04-11",
    "freq":               "5min",
    "total_memory_gb":    32,
    "base_used_pct":      45.0,
    "leak_rate_per_day":  0.05,   # slow memory-leak trend (+0.05% per day)
    "spike_probability":  0.005,
    "spike_magnitude":    20.0,
}
```

---

## CLI Reference

```
python main.py [OPTIONS]

Options:
  -m, --mode {cpu,memory,all}
                        Which pipeline(s) to run. Default: all.

  --cpu-data-file FILE  CSV with historical CPU data.
                        If omitted, synthetic data is generated.

  --memory-data-file FILE
                        CSV with historical memory data
                        (columns: @timestamp, memory.application,
                        system.memory.total, system.memory.actual.used.pct).
                        If omitted, synthetic data is generated.

  --historical-days N   Days of history used for training. Default: 90.

  --forecast-periods N  Future hourly periods to forecast. Default: 48.

  --store-es            Index metrics, predictions, and alerts into
                        Elasticsearch.

  --skip-metrics-index  Skip re-indexing historical metrics. Only index
                        predictions and alerts. Useful for re-runs.

  --cross-validate      Compute MAE / RMSE / MAPE using a 7-day hold-out
                        test. Roughly doubles runtime.

  --no-plots            Skip PNG plot generation.
```

---

## Output

### Console

```
============================================================
  Capacity Management Pipeline  [mode=all]
============================================================

--- CPU Pipeline ---
[INFO] Generating synthetic CPU sample data...
[INFO] Stacked frame: 5764 rows, 4 metrics, 2026-02-10 → 2026-04-11
[INFO] Fitting MSTL + AutoARIMA (daily=24, weekly=168)...
[INFO] Generating 48-period forecast with 95% CI...
  CPU cpu.overall.pct: mean=53.4%  max=60.7%
  CPU cpu.user.pct:    mean=35.9%  max=41.1%
  ...

======================================================================
  CPU CAPACITY ALERTS  (2 total)
======================================================================
[CRITICAL] (2 alerts)
  [CRITICAL] cpu.overall.pct predicted to reach 83.7% ...

--- Memory Pipeline ---
[INFO] Generating synthetic memory sample data...
[INFO] Stacked frame: 2882 rows, 2 metrics, 2026-02-10 → 2026-04-11
[INFO] Fitting MSTL + AutoARIMA ...
  Memory system.memory.actual.used.pct: mean=71.3%  max=82.1%
  ...

======================================================================
  MEMORY CAPACITY ALERTS  (6 total)
======================================================================
[WARNING] (4 alerts)
  [WARNING] system.memory.actual.used.pct predicted to reach 78.8% ...
[CRITICAL] (2 alerts)
  [CRITICAL] system.memory.actual.used.pct predicted to reach 81.4% ...
```

### Plots

One PNG per metric saved to `plots/`:

- **Top panel** — historical data (solid line) + 48h forecast (dashed) + 95% CI band + alert threshold line + breach region (red shaded)
- **Bottom panel** — rolling mean trend

### Elasticsearch

After `--store-es`:

| Index | Contents |
|---|---|
| `cpu-metrics` | Hourly historical CPU data |
| `cpu-predictions` | All CPU forecast rows with `is_future` and `threshold_breached` flags |
| `cpu-alerts` | Structured CPU alert documents |
| `memory-metrics` | Hourly historical memory data |
| `memory-predictions` | All memory forecast rows with `is_future` and `threshold_breached` flags |
| `memory-alerts` | Structured memory alert documents ready for Kibana alerting rules |
