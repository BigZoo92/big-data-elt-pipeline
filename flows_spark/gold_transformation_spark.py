from typing import Dict, Tuple

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from flows_spark.config_spark import (
    BUCKET_GOLD,
    BUCKET_SILVER,
    create_spark_session,
    pick_path,
    read_with_fallback,
    write_with_fallback,
)

HORIZON_DAYS = 365
MAX_PURCHASE_AMOUNT = 10_000.0


def _read_silver() -> Tuple[DataFrame, DataFrame]:
    spark = create_spark_session("gold_spark")
    clients_path, clients_fb = pick_path(BUCKET_SILVER, "clients")
    achats_path, achats_fb = pick_path(BUCKET_SILVER, "achats")
    return read_with_fallback(spark, clients_path, clients_fb, fmt="parquet"), read_with_fallback(
        spark, achats_path, achats_fb, fmt="parquet"
    )


def build_fact(achats: DataFrame, clients: DataFrame) -> DataFrame:
    fact = achats.join(clients.select("id_client", "pays"), on="id_client", how="left")
    fact = (
        fact.withColumn("pays", F.coalesce(F.col("pays"), F.lit("Inconnu")))
        .withColumn("mois", F.date_format("date_achat", "yyyy-MM"))
        .withColumn("jour", F.to_date("date_achat"))
        .withColumn("annee", F.year("date_achat"))
    )
    return fact.select(
        "id_achat",
        "id_client",
        "date_achat",
        "montant",
        "produit",
        "pays",
        "jour",
        "mois",
        "annee",
    )


def build_dim_clients(clients: DataFrame, fact: DataFrame, reference_date) -> DataFrame:
    agg = (
        fact.groupBy("id_client")
        .agg(
            F.min("date_achat").alias("first_purchase"),
            F.max("date_achat").alias("last_purchase"),
            F.count("id_achat").alias("total_orders"),
            F.sum("montant").alias("total_spend"),
            F.countDistinct("produit").alias("product_count"),
        )
    )
    dim = clients.join(agg, on="id_client", how="left")
    dim = dim.withColumn("recency_days", F.coalesce(F.datediff(F.lit(reference_date), F.col("last_purchase")), F.lit(HORIZON_DAYS)))
    dim = dim.withColumn("tenure_days", F.coalesce(F.datediff(F.lit(reference_date), F.to_timestamp("date_inscription")), F.lit(0)))
    dim = dim.withColumn(
        "avg_order_value",
        F.when(F.col("total_orders") > 0, F.col("total_spend") / F.col("total_orders")).otherwise(F.lit(0.0)),
    )
    return dim.select(
        "id_client",
        "nom",
        "email",
        "pays",
        "date_inscription",
        "first_purchase",
        "last_purchase",
        "recency_days",
        "tenure_days",
        "total_orders",
        "total_spend",
        "avg_order_value",
        "product_count",
    )


def build_features(fact: DataFrame, reference_date) -> DataFrame:
    window_start = F.lit(reference_date) - F.expr(f"INTERVAL {HORIZON_DAYS} DAYS")
    window = fact.filter(F.col("date_achat") >= window_start)

    per_window = window.groupBy("id_client").agg(
        F.count("id_achat").alias("freq_12m"),
        F.sum("montant").alias("monetary_12m"),
        F.avg("montant").alias("monetary_avg_12m"),
        F.countDistinct("produit").alias("product_diversity_12m"),
        F.max("date_achat").alias("last_purchase"),
        F.min("date_achat").alias("first_purchase"),
    )

    per_all = fact.groupBy("id_client").agg(
        F.count("id_achat").alias("total_orders_all"),
        F.sum("montant").alias("total_spend_all"),
    )

    base = fact.select("id_client").distinct()
    features = (
        base.join(per_window, on="id_client", how="left")
        .join(per_all, on="id_client", how="left")
        .fillna({"freq_12m": 0, "monetary_12m": 0.0, "monetary_avg_12m": 0.0, "product_diversity_12m": 0})
    )

    features = features.withColumn(
        "recency_days",
        F.when(F.col("last_purchase").isNotNull(), F.datediff(F.lit(reference_date), F.col("last_purchase"))).otherwise(F.lit(HORIZON_DAYS)),
    )
    features = features.withColumn(
        "tenure_days",
        F.when(F.col("first_purchase").isNotNull(), F.datediff(F.lit(reference_date), F.col("first_purchase"))).otherwise(F.lit(0)),
    )
    features = features.withColumn(
        "avg_order_value_all",
        F.when(F.col("total_orders_all") > 0, F.col("total_spend_all") / F.col("total_orders_all")).otherwise(F.lit(0.0)),
    )
    features = features.withColumn("reference_date", F.lit(reference_date))
    return features


def _quantiles(df: DataFrame, col: str, probs) -> Dict[float, float]:
    values = df.approxQuantile(col, probs, 0.01)
    return {p: v for p, v in zip(probs, values)}


def score_clients(features: DataFrame) -> DataFrame:
    freq_q = _quantiles(features, "freq_12m", [0.95, 0.75])
    mon_q = _quantiles(features, "monetary_12m", [0.95, 0.75])
    rec_q = _quantiles(features, "recency_days", [0.75, 0.25])

    freq95 = freq_q.get(0.95, 1.0) or 1.0
    mon95 = mon_q.get(0.95, 1.0) or 1.0
    mon75 = mon_q.get(0.75, mon95) or mon95
    rec75 = rec_q.get(0.75, float(HORIZON_DAYS)) or float(HORIZON_DAYS)
    rec25 = rec_q.get(0.25, float(HORIZON_DAYS / 4)) or float(HORIZON_DAYS / 4)
    freq75 = freq_q.get(0.75, freq95) or freq95

    max_div = features.agg(F.max("product_diversity_12m")).first()[0] or 1.0

    scored = features
    scored = scored.withColumn("freq_clip", F.least(F.col("freq_12m"), F.lit(freq95)))
    scored = scored.withColumn("mon_clip", F.least(F.col("monetary_12m"), F.lit(mon95)))
    scored = scored.withColumn("rec_clip", F.least(F.col("recency_days"), F.lit(float(HORIZON_DAYS))))

    scored = scored.withColumn(
        "freq_norm",
        F.when(F.lit(freq95) > 0, F.col("freq_clip") / F.lit(freq95)).otherwise(F.lit(0.0)),
    )
    scored = scored.withColumn(
        "rec_norm",
        F.lit(1.0) - (F.col("rec_clip") / F.lit(float(HORIZON_DAYS))),
    )
    scored = scored.withColumn(
        "mon_norm",
        F.when(F.lit(mon95) > 0, F.col("mon_clip") / F.lit(mon95)).otherwise(F.lit(0.0)),
    )
    scored = scored.withColumn(
        "div_norm",
        F.when(F.lit(max_div) > 0, F.col("product_diversity_12m") / F.lit(max_div)).otherwise(F.lit(0.0)),
    )
    scored = scored.fillna({"freq_norm": 0.0, "rec_norm": 0.0, "mon_norm": 0.0, "div_norm": 0.0})

    scored = scored.withColumn(
        "prob_reachat_12m",
        (F.lit(0.45) * F.col("freq_norm") + F.lit(0.30) * F.col("rec_norm") + F.lit(0.15) * F.col("mon_norm") + F.lit(0.10) * F.col("div_norm")),
    )

    scored = scored.withColumn(
        "months_active",
        F.least(F.greatest(F.col("tenure_days") / F.lit(30.0), F.lit(1.0)), F.lit(HORIZON_DAYS / 30.0)),
    )
    scored = scored.withColumn("monthly_baseline", F.col("monetary_12m") / F.col("months_active"))
    scored = scored.withColumn("expected_value_12m", F.col("monthly_baseline") * F.lit(12.0) * F.col("prob_reachat_12m"))
    scored = scored.withColumn("value_at_risk_12m", F.col("monetary_12m") * (F.lit(1.0) - F.col("prob_reachat_12m")))

    scored = scored.withColumn(
        "segment_label",
        F.when((F.col("prob_reachat_12m") >= 0.65) & (F.col("monetary_12m") >= mon75), F.lit("VIP"))
        .when((F.col("prob_reachat_12m") >= 0.55) & (F.col("recency_days") <= rec25), F.lit("Actifs"))
        .when((F.col("monetary_12m") >= mon75) & (F.col("recency_days") > rec75), F.lit("A relancer"))
        .when((F.col("freq_12m") <= F.lit(max(1.0, freq75 * 0.5))) & (F.col("recency_days") > rec75), F.lit("Dormants"))
        .otherwise(F.lit("A potentiel")),
    )
    return scored


def summarize_segments(scored: DataFrame) -> DataFrame:
    return (
        scored.groupBy("segment_label")
        .agg(
            F.count("id_client").alias("clients"),
            F.sum("monetary_12m").alias("ca_12m"),
            F.sum("expected_value_12m").alias("expected_value_12m"),
            F.sum("value_at_risk_12m").alias("value_at_risk_12m"),
            F.expr("percentile_approx(freq_12m, 0.5)").alias("freq_med"),
            F.expr("percentile_approx(recency_days, 0.5)").alias("recency_med"),
        )
        .orderBy(F.desc("expected_value_12m"))
    )


def aggregate_sales(fact: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
    ca_monthly = fact.groupBy("mois").agg(F.sum("montant").alias("ca")).orderBy("mois")
    ca_country = fact.groupBy("pays").agg(F.sum("montant").alias("ca")).orderBy(F.desc("ca"))
    ca_product = fact.groupBy("produit").agg(F.sum("montant").alias("ca")).orderBy(F.desc("ca"))
    return ca_monthly, ca_country, ca_product


def build_cohort(fact: DataFrame) -> DataFrame:
    first_purchase = fact.groupBy("id_client").agg(F.min("date_achat").alias("fp"))
    fact_fp = fact.join(first_purchase, on="id_client", how="left").withColumn("first_purchase_month", F.date_format("fp", "yyyy-MM"))
    return (
        fact_fp.groupBy("first_purchase_month")
        .agg(F.countDistinct("id_client").alias("clients"), F.sum("montant").alias("ca"))
        .orderBy("first_purchase_month")
    )


def gold_transformation_spark() -> Dict[str, str]:
    clients, achats = _read_silver()
    achats = achats.filter((F.col("montant") > 0) & (F.col("montant") <= MAX_PURCHASE_AMOUNT))
    ref_date = achats.agg(F.max("date_achat")).first()[0]
    if ref_date is None:
        raise ValueError("Pas de date_achat valide en silver")

    fact = build_fact(achats, clients)
    dim_clients = build_dim_clients(clients, fact, ref_date)
    features = build_features(fact, ref_date)
    scores = score_clients(features)
    segments = summarize_segments(scores)
    ca_monthly, ca_country, ca_product = aggregate_sales(fact)
    cohort = build_cohort(fact)

    outputs = {
        "fact_achats.parquet": fact,
        "dim_clients.parquet": dim_clients,
        "client_features.parquet": features,
        "client_scores.parquet": scores,
        "segment_summary.parquet": segments,
        "ca_monthly.parquet": ca_monthly,
        "ca_country.parquet": ca_country,
        "ca_product.parquet": ca_product,
        "cohort_first_purchase.parquet": cohort,
    }

    written: Dict[str, str] = {}
    for name, df in outputs.items():
        primary, fallback = pick_path(BUCKET_GOLD, name)
        written[name] = write_with_fallback(df, primary, fallback, fmt="parquet")

    print("Gold Spark done.")
    return written


if __name__ == "__main__":
    res = gold_transformation_spark()
    print(res)
