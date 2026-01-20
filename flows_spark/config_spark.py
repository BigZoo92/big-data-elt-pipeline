"""
Configuration Spark (MinIO S3A avec fallback local).
- Par defaut on tente S3A (SPARK_STORAGE_MODE=s3) puis fallback local si echec.
- Local fallback : data/spark_lake/<layer>.
"""
from __future__ import annotations

import os
os.environ.setdefault("HADOOP_HOME", r"C:\hadoop")
os.environ.setdefault("hadoop.home.dir", r"C:\hadoop")
from pathlib import Path
from typing import Tuple

from pyspark.sql import SparkSession

tmp = Path("data") / "spark_tmp"
tmp.mkdir(parents=True, exist_ok=True)

# Buckets identiques a la pipeline Pandas
BUCKET_SOURCES = "sources"
BUCKET_BRONZE = "bronze"
BUCKET_SILVER = "silver"
BUCKET_GOLD = "gold"

STORAGE_MODE = os.getenv("SPARK_STORAGE_MODE", "s3")  # "s3" ou "local"
LOCAL_BASE = Path("data") / "spark_lake"
SPARK_PREFIX = "spark"  # prefix dans les buckets pour eviter collision

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
SPARK_MASTER_URL = os.getenv("SPARK_MASTER_URL", "local[*]")

HADOOP_VERSION = "3.3.4"

AWS_PACKAGES = (
    f"org.apache.hadoop:hadoop-aws:{HADOOP_VERSION},"
    "com.amazonaws:aws-java-sdk-bundle:1.12.262"
)


def create_spark_session(app: str = "spark_pipeline") -> SparkSession:
    """
    Cree une SparkSession configuree pour MinIO S3A avec fallback local.
    """
    secure = "true" if MINIO_SECURE else "false"
    spark = (
        SparkSession.builder.master(SPARK_MASTER_URL)
        .appName(app)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.jars.packages", AWS_PACKAGES)
        .config("spark.hadoop.fs.s3a.endpoint", f"http://{MINIO_ENDPOINT}")
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", secure)
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.analytics.accelerator.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl.disable.cache", "true")
        .config("spark.hadoop.io.native.lib.available", "false")
        .config("spark.hadoop.util.Shell.useDeprecatedWinUtils", "true")
        .config("spark.local.dir", str(tmp.resolve()))
    ).getOrCreate()
    print("Spark:", spark.version)
    print("Hadoop:", spark._jvm.org.apache.hadoop.util.VersionInfo.getVersion())
    return spark


def _s3_path(bucket: str, object_name: str) -> str:
    return f"s3a://{bucket}/{SPARK_PREFIX}/{object_name}"


def _local_path(layer: str, object_name: str) -> str:
    return str((LOCAL_BASE / layer / object_name))


def pick_path(layer: str, object_name: str) -> Tuple[str, str]:
    """
    Retourne (primary, fallback) paths pour un object.
    primary = S3 si mode s3, sinon local.
    """
    primary = _s3_path({BUCKET_BRONZE: BUCKET_BRONZE, BUCKET_SILVER: BUCKET_SILVER, BUCKET_GOLD: BUCKET_GOLD, BUCKET_SOURCES: BUCKET_SOURCES}.get(layer, layer), object_name)
    fallback = _local_path(layer, object_name)
    if STORAGE_MODE == "local":
        primary, fallback = fallback, fallback
    return primary, fallback


def ensure_local_dirs() -> None:
    for layer in [BUCKET_SOURCES, BUCKET_BRONZE, BUCKET_SILVER, BUCKET_GOLD]:
        (LOCAL_BASE / layer).mkdir(parents=True, exist_ok=True)


def write_with_fallback(df, primary_path: str, fallback_path: str, fmt: str = "parquet", mode: str = "overwrite") -> str:
    """
    Tente d'ecrire sur primary, sinon fallback local.
    """
    try:
        df.write.format(fmt).mode(mode).save(primary_path)
        print(f"Written to {primary_path}")
        return primary_path
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"[WARN] Primary write failed ({exc}); fallback to {fallback_path}")
        ensure_local_dirs()
        df.write.format(fmt).mode(mode).save(fallback_path)
        return fallback_path


def read_with_fallback(spark, primary_path: str, fallback_path: str, fmt: str = "parquet"):
    try:
        return spark.read.format(fmt).load(primary_path)
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"[WARN] Read primary failed ({exc}); trying fallback {fallback_path}")
        if fmt == "csv":
            return spark.read.format(fmt).option("header", True).load(fallback_path)
        return spark.read.format(fmt).load(fallback_path)
