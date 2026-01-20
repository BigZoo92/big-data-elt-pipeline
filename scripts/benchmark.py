from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Dict, Tuple

import pandas as pd

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flows.bronze_ingestion import bronze_ingestion_flow
from flows.gold_transformation import gold_transformation_flow
from flows.silver_transformation import silver_transformation_flow
from flows_spark.bronze_ingestion_spark import bronze_ingestion_spark
from flows_spark.gold_transformation_spark import gold_transformation_spark
from flows_spark.silver_transformation_spark import silver_transformation_spark


def _duplicate_sources(scale: int) -> Tuple[str, int]:
    if scale <= 1:
        base = Path("data") / "sources"
        achats_rows = sum(1 for _ in open(base / "achats.csv")) - 1
        return str(base), achats_rows

    base = Path("data") / "sources"
    tmp_dir = Path(tempfile.mkdtemp(prefix="benchmark_sources_"))
    clients = pd.read_csv(base / "clients.csv")
    achats = pd.read_csv(base / "achats.csv")
    achats_list = []
    offset = 0
    for _ in range(scale):
        df = achats.copy()
        df["id_achat"] = df["id_achat"] + offset
        offset += len(achats)
        achats_list.append(df)
    achats_scaled = pd.concat(achats_list, ignore_index=True)

    clients.to_csv(tmp_dir / "clients.csv", index=False)
    achats_scaled.to_csv(tmp_dir / "achats.csv", index=False)
    return str(tmp_dir), len(achats_scaled)


def _timed(label: str, fn, *args, **kwargs) -> Tuple[float, object]:
    start = perf_counter()
    result = fn(*args, **kwargs)
    duration = perf_counter() - start
    print(f"{label} done in {duration:.2f}s")
    return duration, result


def run_pandas(scale: int) -> Dict[str, float]:
    data_dir, dataset_size = _duplicate_sources(scale)
    durations: Dict[str, float] = {}
    durations["bronze"], _ = _timed("pandas bronze", bronze_ingestion_flow, data_dir=data_dir)
    durations["silver"], _ = _timed("pandas silver", silver_transformation_flow)
    durations["gold"], _ = _timed("pandas gold", gold_transformation_flow)
    durations["total"] = sum(durations.values())
    durations["dataset_size"] = dataset_size
    return durations


def run_spark(scale: int) -> Dict[str, float]:
    data_dir, dataset_size = _duplicate_sources(scale)
    durations: Dict[str, float] = {}
    durations["bronze"], _ = _timed("spark bronze", bronze_ingestion_spark, data_dir=data_dir)
    durations["silver"], _ = _timed("spark silver", silver_transformation_spark)
    durations["gold"], _ = _timed("spark gold", gold_transformation_spark)
    durations["total"] = sum(durations.values())
    durations["dataset_size"] = dataset_size
    return durations


def append_metrics(out_path: Path, entry: Dict[str, object]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if out_path.exists():
        try:
            history = json.loads(out_path.read_text())
        except Exception:
            history = []
    history.append(entry)
    out_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Benchmark saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Pandas vs Spark")
    parser.add_argument("--scale", type=int, default=1, help="Dupliquer achats pour stress-test (>=1)")
    parser.add_argument("--output", type=Path, default=Path("data/metrics/benchmark.json"))
    args = parser.parse_args()

    pandas_metrics = run_pandas(args.scale)
    spark_metrics = run_spark(args.scale)

    entry = {
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_size": pandas_metrics["dataset_size"],
        "scale": args.scale,
        "durations": {
            "pandas": {k: v for k, v in pandas_metrics.items() if k != "dataset_size"},
            "spark": {k: v for k, v in spark_metrics.items() if k != "dataset_size"},
        },
        "environment": {
            "machine": platform.node(),
            "os": platform.system(),
            "spark_master": os.getenv("SPARK_MASTER_URL", "local[*]"),
        },
    }
    append_metrics(args.output, entry)


if __name__ == "__main__":
    main()
