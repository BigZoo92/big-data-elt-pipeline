from pathlib import Path
from typing import Dict

from pyspark.sql import SparkSession

from flows_spark.config_spark import (
    BUCKET_BRONZE,
    BUCKET_SOURCES,
    create_spark_session,
    ensure_local_dirs,
    pick_path,
    write_with_fallback,
)


def bronze_ingestion_spark(data_dir: str = "./data/sources") -> Dict[str, str]:
    spark: SparkSession = create_spark_session("bronze_spark")
    ensure_local_dirs()
    src = Path(data_dir)

    clients_path = src / "clients.csv"
    achats_path = src / "achats.csv"

    clients_df = spark.read.csv(str(clients_path), header=True, inferSchema=True)
    achats_df = spark.read.csv(str(achats_path), header=True, inferSchema=True)

    clients_primary, clients_fb = pick_path(BUCKET_BRONZE, "clients")
    achats_primary, achats_fb = pick_path(BUCKET_BRONZE, "achats")

    out_clients = write_with_fallback(clients_df, clients_primary, clients_fb, fmt="parquet")
    out_achats = write_with_fallback(achats_df, achats_primary, achats_fb, fmt="parquet")

    return {"clients": out_clients, "achats": out_achats}


if __name__ == "__main__":
    res = bronze_ingestion_spark()
    print(res)
