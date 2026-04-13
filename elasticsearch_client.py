"""
Elasticsearch client for indexing CPU and Memory metrics, predictions, and alerts.

Index layout:
  cpu-metrics          : raw/historical CPU data points
  cpu-predictions      : MSTL+AutoARIMA forecast output (yhat, bounds)
  cpu-alerts           : generated CPU threshold-breach alerts
  memory-metrics       : raw/historical memory data points
  memory-predictions   : MSTL+AutoARIMA memory forecast output
  memory-alerts        : generated memory threshold-breach alerts
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from elasticsearch import Elasticsearch, helpers
from elasticsearch.exceptions import ConnectionError, NotFoundError

from config import ELASTICSEARCH_CONFIG, LOGGING_CONFIG

logging.basicConfig(**LOGGING_CONFIG)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index mappings
# ---------------------------------------------------------------------------

METRICS_MAPPING = {
    "mappings": {
        "properties": {
            "@timestamp":      {"type": "date"},
            "cpu.overall.pct": {"type": "float"},
            "cpu.user.pct":    {"type": "float"},
            "cpu.kernel.pct":  {"type": "float"},
            "cpu.wait.pct":    {"type": "float"},
            "cpu.cores":       {"type": "integer"},
        }
    },
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
}

PREDICTIONS_MAPPING = {
    "mappings": {
        "properties": {
            "@timestamp":        {"type": "date"},
            "forecast_time":     {"type": "date"},
            "metric":            {"type": "keyword"},
            "yhat":              {"type": "float"},
            "yhat_lower":        {"type": "float"},
            "yhat_upper":        {"type": "float"},
            "model":             {"type": "keyword"},
            "is_future":         {"type": "boolean"},
            "threshold":         {"type": "float"},
            "threshold_breached": {"type": "boolean"},
        }
    },
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
}

ALERTS_MAPPING = {
    "mappings": {
        "properties": {
            "@timestamp":        {"type": "date"},
            "alert_id":          {"type": "keyword"},
            "severity":          {"type": "keyword"},
            "metric":            {"type": "keyword"},
            "predicted_value":   {"type": "float"},
            "threshold":         {"type": "float"},
            "breach_time":       {"type": "date"},
            "hours_until_breach": {"type": "float"},
            "upper_bound":       {"type": "float"},
            "lower_bound":       {"type": "float"},
            "message":           {"type": "text"},
            "tags":              {"type": "object"},
        }
    },
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
}


MEMORY_METRICS_MAPPING = {
    "mappings": {
        "properties": {
            "@timestamp":                    {"type": "date"},
            "memory.application":            {"type": "long"},
            "system.memory.total":           {"type": "long"},
            "system.memory.actual.used.pct": {"type": "float"},
        }
    },
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
}

MEMORY_PREDICTIONS_MAPPING = {
    "mappings": {
        "properties": {
            "@timestamp":        {"type": "date"},
            "forecast_time":     {"type": "date"},
            "metric":            {"type": "keyword"},
            "yhat":              {"type": "float"},
            "yhat_lower":        {"type": "float"},
            "yhat_upper":        {"type": "float"},
            "model":             {"type": "keyword"},
            "is_future":         {"type": "boolean"},
            "threshold":         {"type": "float"},
            "threshold_breached": {"type": "boolean"},
        }
    },
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
}

MEMORY_ALERTS_MAPPING = {
    "mappings": {
        "properties": {
            "@timestamp":          {"type": "date"},
            "alert_id":            {"type": "keyword"},
            "severity":            {"type": "keyword"},
            "metric":              {"type": "keyword"},
            "predicted_value":     {"type": "float"},
            "threshold":           {"type": "float"},
            "breach_time":         {"type": "date"},
            "hours_until_breach":  {"type": "float"},
            "upper_bound":         {"type": "float"},
            "lower_bound":         {"type": "float"},
            "message":             {"type": "text"},
            "tags":                {"type": "object"},
        }
    },
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
}


class ESClient:
    """Thin wrapper around the Elasticsearch client with bulk-indexing helpers."""

    def __init__(self, config: dict = None):
        cfg = config or ELASTICSEARCH_CONFIG
        # CPU indices
        self.index_metrics      = cfg["index_metrics"]
        self.index_predictions  = cfg["index_predictions"]
        self.index_alerts       = cfg["index_alerts"]
        # Memory indices
        self.index_memory_metrics     = cfg["index_memory_metrics"]
        self.index_memory_predictions = cfg["index_memory_predictions"]
        self.index_memory_alerts      = cfg["index_memory_alerts"]

        self.es = Elasticsearch(
            hosts=cfg["hosts"],
            basic_auth=(cfg.get("username", ""), cfg.get("password", "")),
            verify_certs=cfg.get("verify_certs", False),
            ssl_show_warn=False,
        )

    def ping(self) -> bool:
        try:
            return self.es.ping()
        except ConnectionError as e:
            logger.error(f"Elasticsearch connection failed: {e}")
            return False

    def setup_indices(self, recreate: bool = False, mode: str = "all") -> None:
        """Create indices with mappings (idempotent by default).

        Args:
            recreate: Delete and recreate existing indices
            mode: 'cpu', 'memory', or 'all'
        """
        indices = {}
        if mode in ("cpu", "all"):
            indices.update({
                self.index_metrics:     METRICS_MAPPING,
                self.index_predictions: PREDICTIONS_MAPPING,
                self.index_alerts:      ALERTS_MAPPING,
            })
        if mode in ("memory", "all"):
            indices.update({
                self.index_memory_metrics:     MEMORY_METRICS_MAPPING,
                self.index_memory_predictions: MEMORY_PREDICTIONS_MAPPING,
                self.index_memory_alerts:      MEMORY_ALERTS_MAPPING,
            })
        for index_name, mapping in indices.items():
            if recreate and self.es.indices.exists(index=index_name):
                self.es.indices.delete(index=index_name)
                logger.info(f"Deleted existing index: {index_name}")

            if not self.es.indices.exists(index=index_name):
                self.es.indices.create(index=index_name, body=mapping)
                logger.info(f"Created index: {index_name}")
            else:
                logger.info(f"Index already exists: {index_name}")

    # ------------------------------------------------------------------
    # Bulk indexing helpers
    # ------------------------------------------------------------------

    def index_metrics_data(self, df: pd.DataFrame, chunk_size: int = 2000) -> int:
        """Bulk index raw CPU metrics into cpu-metrics."""
        docs = self._metrics_df_to_docs(df)
        return self._bulk_index(docs, self.index_metrics, chunk_size)

    def index_predictions_data(
        self,
        forecasts: dict,         # metric -> ForecastResult
        last_training_dates: dict,  # metric -> pd.Timestamp
        chunk_size: int = 2000,
    ) -> int:
        """Bulk index MSTL+AutoARIMA forecast output into cpu-predictions."""
        from config import ALERT_CONFIG

        thresholds = {
            "cpu.overall.pct": ALERT_CONFIG["cpu_overall_threshold"],
            "cpu.user.pct":    ALERT_CONFIG["cpu_user_threshold"],
            "cpu.kernel.pct":  ALERT_CONFIG["cpu_kernel_threshold"],
            "cpu.wait.pct":    ALERT_CONFIG["cpu_wait_threshold"],
        }

        docs = []
        indexed_at = datetime.now(timezone.utc).isoformat()

        for metric, result in forecasts.items():
            if result.forecast_df.empty:
                continue
            last_train = last_training_dates.get(metric, pd.Timestamp.min)
            threshold = thresholds.get(metric, 80.0)

            for _, row in result.forecast_df.iterrows():
                ds = pd.Timestamp(row["ds"])
                if ds.tzinfo is None:
                    ds = ds.tz_localize("UTC")

                doc = {
                    "@timestamp":        indexed_at,
                    "forecast_time":     ds.isoformat(),
                    "metric":            metric,
                    "yhat":              round(float(row["yhat"]), 3),
                    "yhat_lower":        round(float(row["yhat_lower"]), 3),
                    "yhat_upper":        round(float(row["yhat_upper"]), 3),
                    "model":             "MSTL+AutoARIMA",
                    "is_future":         ds > last_train,
                    "threshold":         threshold,
                    "threshold_breached": float(row["yhat"]) > threshold,
                }
                docs.append(doc)

        return self._bulk_index(docs, self.index_predictions, chunk_size)

    def index_alerts_data(self, alerts: list, chunk_size: int = 500) -> int:
        """Bulk index CPUAlert objects into cpu-alerts."""
        docs = [a.to_dict() for a in alerts]
        return self._bulk_index(docs, self.index_alerts, chunk_size)

    # ------------------------------------------------------------------
    # Memory indexing methods
    # ------------------------------------------------------------------

    def index_memory_metrics_data(self, df: pd.DataFrame, chunk_size: int = 2000) -> int:
        """Bulk index raw memory metrics into memory-metrics."""
        docs = self._memory_metrics_df_to_docs(df)
        return self._bulk_index(docs, self.index_memory_metrics, chunk_size)

    def index_memory_predictions_data(
        self,
        forecasts: dict,
        last_training_dates: dict,
        chunk_size: int = 2000,
    ) -> int:
        """Bulk index memory forecast output into memory-predictions."""
        from config import MEMORY_ALERT_CONFIG
        thresholds = {
            "system.memory.actual.used.pct": MEMORY_ALERT_CONFIG["memory_used_pct_threshold"],
            "memory.application.pct":        MEMORY_ALERT_CONFIG["memory_application_threshold_pct"],
        }

        docs = []
        indexed_at = datetime.now(timezone.utc).isoformat()

        for metric, result in forecasts.items():
            if result.forecast_df.empty:
                continue
            last_train = last_training_dates.get(metric, pd.Timestamp.min)
            threshold  = thresholds.get(metric, 80.0)

            for _, row in result.forecast_df.iterrows():
                ds = pd.Timestamp(row["ds"])
                if ds.tzinfo is None:
                    ds = ds.tz_localize("UTC")
                docs.append({
                    "@timestamp":         indexed_at,
                    "forecast_time":      ds.isoformat(),
                    "metric":             metric,
                    "yhat":               round(float(row["yhat"]), 3),
                    "yhat_lower":         round(float(row["yhat_lower"]), 3),
                    "yhat_upper":         round(float(row["yhat_upper"]), 3),
                    "model":              "MSTL+AutoARIMA",
                    "is_future":          ds > last_train,
                    "threshold":          threshold,
                    "threshold_breached": float(row["yhat"]) > threshold,
                })

        return self._bulk_index(docs, self.index_memory_predictions, chunk_size)

    def index_memory_alerts_data(self, alerts: list, chunk_size: int = 500) -> int:
        """Bulk index MemoryAlert objects into memory-alerts."""
        docs = [a.to_dict() for a in alerts]
        return self._bulk_index(docs, self.index_memory_alerts, chunk_size)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _metrics_df_to_docs(self, df: pd.DataFrame) -> list[dict]:
        docs = []
        for _, row in df.iterrows():
            ts = pd.Timestamp(row["@timestamp"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            docs.append({
                "@timestamp":      ts.isoformat(),
                "cpu.overall.pct": float(row["cpu.overall.pct"]),
                "cpu.user.pct":    float(row["cpu.user.pct"]),
                "cpu.kernel.pct":  float(row["cpu.kernel.pct"]),
                "cpu.wait.pct":    float(row["cpu.wait.pct"]),
                "cpu.cores":       int(row["cpu.cores"]),
            })
        return docs

    def _memory_metrics_df_to_docs(self, df: pd.DataFrame) -> list[dict]:
        docs = []
        for _, row in df.iterrows():
            ts = pd.Timestamp(row["@timestamp"])
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            docs.append({
                "@timestamp":                    ts.isoformat(),
                "memory.application":            int(row["memory.application"]),
                "system.memory.total":           int(row["system.memory.total"]),
                "system.memory.actual.used.pct": float(row["system.memory.actual.used.pct"]),
            })
        return docs

    def _bulk_index(self, docs: list[dict], index: str, chunk_size: int) -> int:
        if not docs:
            logger.warning(f"No documents to index into {index}")
            return 0

        def actions():
            for doc in docs:
                yield {"_index": index, "_source": doc}

        total = 0
        errors = 0
        for success, info in helpers.parallel_bulk(
            self.es,
            actions(),
            chunk_size=chunk_size,
            raise_on_error=False,
        ):
            if success:
                total += 1
            else:
                errors += 1
                logger.debug(f"Index error: {info}")

        logger.info(f"Indexed {total} docs into '{index}' ({errors} errors)")
        return total

    def search_alerts(self, severity: Optional[str] = None, hours: int = 24) -> list[dict]:
        """Query recent alerts from Elasticsearch."""
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"range": {"@timestamp": {"gte": f"now-{hours}h"}}}
                    ]
                }
            },
            "sort": [{"breach_time": {"order": "asc"}}],
            "size": 100,
        }
        if severity:
            query["query"]["bool"]["must"].append({"term": {"severity": severity}})

        try:
            resp = self.es.search(index=self.index_alerts, body=query)
            return [hit["_source"] for hit in resp["hits"]["hits"]]
        except NotFoundError:
            return []

    def get_latest_prediction(self, metric: str) -> Optional[dict]:
        """Get the most recent future prediction for a metric."""
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"metric": metric}},
                        {"term": {"is_future": True}},
                    ]
                }
            },
            "sort": [{"forecast_time": {"order": "desc"}}],
            "size": 1,
        }
        try:
            resp = self.es.search(index=self.index_predictions, body=query)
            hits = resp["hits"]["hits"]
            return hits[0]["_source"] if hits else None
        except NotFoundError:
            return None
