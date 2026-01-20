from typing import Dict, List, Set, Tuple

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from flows_spark.config_spark import (
    BUCKET_BRONZE,
    BUCKET_SILVER,
    create_spark_session,
    pick_path,
    read_with_fallback,
    write_with_fallback,
)

DATE_LOWER_BOUND = "2000-01-01"
MAX_PURCHASE_AMOUNT = 10_000.0
REQUIRED_CLIENT_COLS: List[str] = ["id_client", "nom", "email", "date_inscription", "pays"]
REQUIRED_ACHAT_COLS: List[str] = ["id_achat", "id_client", "date_achat", "montant", "produit"]


def _validate_columns(df: DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: colonnes manquantes {missing}; present {df.columns}")


def clean_clients(df: DataFrame) -> Tuple[DataFrame, Dict[str, int]]:
    q: Dict[str, int] = {"initial_rows": df.count()}
    _validate_columns(df, REQUIRED_CLIENT_COLS, "clients")

    out = (
        df.select(*REQUIRED_CLIENT_COLS)
        .withColumn("id_client", F.col("id_client").cast("long"))
        .withColumn("email", F.lower(F.trim(F.col("email"))))
        .withColumn("nom", F.trim(F.col("nom")))
        .withColumn("pays", F.initcap(F.trim(F.col("pays"))))
        .withColumn("date_inscription", F.to_date("date_inscription"))
    )

    out = out.filter(F.col("id_client").isNotNull()).filter(F.col("date_inscription").isNotNull())
    out = out.filter(F.col("email").contains("@"))
    out = out.filter(F.col("date_inscription") >= F.lit(DATE_LOWER_BOUND))
    out = out.dropDuplicates(["id_client"]).orderBy("id_client")

    q["final_rows"] = out.count()
    return out, q


def clean_achats(df: DataFrame, valid_ids: Set[int]) -> Tuple[DataFrame, Dict[str, int]]:
    q: Dict[str, int] = {"initial_rows": df.count()}
    _validate_columns(df, REQUIRED_ACHAT_COLS, "achats")

    out = (
        df.select(*REQUIRED_ACHAT_COLS)
        .withColumn("id_achat", F.col("id_achat").cast("long"))
        .withColumn("id_client", F.col("id_client").cast("long"))
        .withColumn("montant", F.col("montant").cast("double"))
        .withColumn("date_achat", F.to_timestamp("date_achat"))
        .withColumn("produit", F.initcap(F.trim(F.col("produit"))))
    )

    out = out.filter(
        F.col("id_achat").isNotNull()
        & F.col("id_client").isNotNull()
        & F.col("montant").isNotNull()
        & F.col("date_achat").isNotNull()
        & F.col("produit").isNotNull()
    )
    out = out.filter((F.col("montant") > 0) & (F.col("montant") <= MAX_PURCHASE_AMOUNT))
    if valid_ids:
        session = df.sparkSession
        valid_df = session.createDataFrame([(i,) for i in valid_ids], ["id_client"])
        out = out.join(F.broadcast(valid_df), "id_client", "inner")
    out = out.dropDuplicates(["id_achat"])
    q["final_rows"] = out.count()
    return out.orderBy("id_achat"), q


def silver_transformation_spark(
    bronze_clients_object: str = "clients", bronze_achats_object: str = "achats"
) -> Dict[str, Dict[str, object]]:
    spark = create_spark_session("silver_spark")

    bronze_clients_path, bronze_clients_fb = pick_path(BUCKET_BRONZE, bronze_clients_object)
    bronze_achats_path, bronze_achats_fb = pick_path(BUCKET_BRONZE, bronze_achats_object)

    raw_clients = read_with_fallback(spark, bronze_clients_path, bronze_clients_fb, fmt="parquet")
    raw_achats = read_with_fallback(spark, bronze_achats_path, bronze_achats_fb, fmt="parquet")

    clients_df, clients_q = clean_clients(raw_clients)
    client_ids = set(row["id_client"] for row in clients_df.select("id_client").distinct().collect())
    achats_df, achats_q = clean_achats(raw_achats, client_ids)

    clients_primary, clients_fb = pick_path(BUCKET_SILVER, "clients")
    achats_primary, achats_fb = pick_path(BUCKET_SILVER, "achats")

    silver_clients = write_with_fallback(clients_df, clients_primary, clients_fb, fmt="parquet")
    silver_achats = write_with_fallback(achats_df, achats_primary, achats_fb, fmt="parquet")

    print("Quality clients:", clients_q)
    print("Quality achats:", achats_q)
    return {
        "clients": {"object_name": silver_clients, "quality": clients_q},
        "achats": {"object_name": silver_achats, "quality": achats_q},
    }


if __name__ == "__main__":
    res = silver_transformation_spark()
    print(res)
