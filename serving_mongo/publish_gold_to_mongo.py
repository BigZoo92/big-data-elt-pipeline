from __future__ import annotations

import json
import os
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List

import pandas as pd
from dotenv import load_dotenv
from pymongo import ASCENDING, ReplaceOne
from pymongo.collection import Collection
from pymongo.database import Database

from flows.config import BUCKET_GOLD, get_minio_client
from serving_mongo.mongo_config import get_database

load_dotenv()

LOCAL_GOLD_DIRS = [Path("data") / "spark_lake" / "gold", Path("data") / "gold"]

GOLD_FILES = {
    "fact_achats.parquet": "gold_fact_achats",
    "dim_clients.parquet": "gold_dim_clients",
    "client_features.parquet": "gold_client_features",
    "client_scores.parquet": "gold_client_scores",
    "segment_summary.parquet": "gold_segment_summary",
    "ca_monthly.parquet": "gold_monthly",
    "ca_country.parquet": "gold_by_country",
    "ca_product.parquet": "gold_by_product",
    "cohort_first_purchase.parquet": "gold_cohort_first_purchase",
}

INDEXES = {
    "gold_fact_achats": [
        ("id_achat", True),
        ("id_client", False),
        ("date_achat", False),
        ("mois", False),
        ("pays", False),
    ],
    "gold_dim_clients": [("id_client", True), ("pays", False)],
    "gold_client_features": [("id_client", True)],
    "gold_client_scores": [("id_client", True), ("segment_label", False)],
    "gold_monthly": [("mois", True)],
    "gold_by_country": [("pays", True)],
    "gold_by_product": [("produit", True)],
    "gold_segment_summary": [("segment_label", True)],
    "gold_daily": [("jour", True)],
    "gold_weekly": [("semaine", True)],
    "gold_distribution": [("bucket", True)],
    "gold_monthly_growth": [("mois", True)],
    "gold_cohort_first_purchase": [("first_purchase_month", True)],
}


def _read_parquet(object_name: str) -> pd.DataFrame | None:
    try:
        client = get_minio_client()
        obj = client.get_object(BUCKET_GOLD, object_name)
        data = obj.read()
        obj.close()
        obj.release_conn()
        return pd.read_parquet(BytesIO(data))
    except Exception as exc_minio:
        for base in LOCAL_GOLD_DIRS:
            candidate = base / object_name
            if candidate.exists():
                print(f"[WARN] MinIO read failed ({exc_minio}); fallback to {candidate}")
                return pd.read_parquet(candidate)
        print(f"[WARN] Impossible de lire {object_name} (MinIO et fallback échouent)")
        return None


def _records(df: pd.DataFrame | None) -> List[dict]:
    if df is None or df.empty:
        return []
    clean = df.copy()
    clean = clean.where(pd.notna(clean), None)
    return json.loads(clean.to_json(orient="records", date_format="iso"))


def _ensure_indexes(db: Database) -> None:
    for coll_name, fields in INDEXES.items():
        coll = db[coll_name]
        for field, unique in fields:
            coll.create_index([(field, ASCENDING)], unique=unique, background=True)


def _upsert_by_key(collection: Collection, records: Iterable[dict], key: str) -> int:
    ops = []
    for rec in records:
        if key not in rec or rec[key] is None:
            continue
        ops.append(ReplaceOne({key: rec[key]}, rec, upsert=True))
    if not ops:
        return 0
    res = collection.bulk_write(ops, ordered=False)
    return res.upserted_count + res.modified_count + res.matched_count


def _replace_all(collection: Collection, records: List[dict]) -> int:
    collection.delete_many({})
    if not records:
        return 0
    result = collection.insert_many(records)
    return len(result.inserted_ids)


def _build_daily(fact: pd.DataFrame | None) -> pd.DataFrame:
    if fact is None or fact.empty:
        return pd.DataFrame()
    df = fact.copy()
    df["date_achat"] = pd.to_datetime(df["date_achat"], errors="coerce")
    df = df.dropna(subset=["date_achat"])
    daily = (
        df.groupby(df["date_achat"].dt.date)
        .agg(ca=("montant", "sum"), achats=("id_achat", "count"))
        .reset_index()
    )
    daily["jour"] = pd.to_datetime(daily["date_achat"]).dt.strftime("%Y-%m-%d")
    return daily[["jour", "ca", "achats"]].sort_values("jour")


def _build_weekly(fact: pd.DataFrame | None) -> pd.DataFrame:
    if fact is None or fact.empty:
        return pd.DataFrame()
    df = fact.copy()
    df["date_achat"] = pd.to_datetime(df["date_achat"], errors="coerce")
    df = df.dropna(subset=["date_achat"])
    df["semaine"] = df["date_achat"].dt.to_period("W").apply(lambda p: str(p.start_time.date()))
    weekly = (
        df.groupby("semaine")
        .agg(ca=("montant", "sum"), achats=("id_achat", "count"))
        .reset_index()
        .sort_values("semaine")
    )
    return weekly


def _build_distribution(fact: pd.DataFrame | None, bins: int = 12) -> pd.DataFrame:
    if fact is None or fact.empty:
        return pd.DataFrame()
    df = fact.copy()
    df["montant"] = pd.to_numeric(df["montant"], errors="coerce")
    df = df.dropna(subset=["montant"])
    cuts = pd.cut(df["montant"], bins=bins)
    distrib = cuts.value_counts().sort_index().reset_index()
    distrib.columns = ["bucket", "count"]
    distrib["bucket"] = distrib["bucket"].astype(str)
    return distrib.sort_values("bucket")


def _build_monthly_growth(ca_monthly: pd.DataFrame | None) -> pd.DataFrame:
    if ca_monthly is None or ca_monthly.empty:
        return pd.DataFrame()
    df = ca_monthly.copy()
    df["mois"] = pd.to_datetime(df["mois"].astype(str), errors="coerce").dt.to_period("M")
    df = df.dropna(subset=["mois"])
    df["mois"] = df["mois"].astype(str)
    df = df.sort_values("mois")
    df["prev_ca"] = df["ca"].shift(1)
    df["growth_abs"] = df["ca"] - df["prev_ca"]
    df["growth_pct"] = df.apply(
        lambda row: (row["growth_abs"] / row["prev_ca"]) if pd.notna(row["prev_ca"]) and row["prev_ca"] != 0 else None,
        axis=1,
    )
    return df[["mois", "ca", "prev_ca", "growth_abs", "growth_pct"]]


def load_gold_tables() -> Dict[str, pd.DataFrame]:
    tables: Dict[str, pd.DataFrame] = {}
    for object_name in GOLD_FILES.keys():
        tables[object_name] = _read_parquet(object_name)
    return tables


def normalize_fact(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    df = df.copy()
    df["id_achat"] = pd.to_numeric(df.get("id_achat"), errors="coerce").astype("Int64")
    df = df.dropna(subset=["id_achat"])
    df["id_client"] = pd.to_numeric(df.get("id_client"), errors="coerce").astype("Int64")
    df["montant"] = pd.to_numeric(df.get("montant"), errors="coerce")
    df["date_achat"] = pd.to_datetime(df.get("date_achat"), errors="coerce")
    df["jour"] = pd.to_datetime(df.get("jour"), errors="coerce").dt.strftime("%Y-%m-%d")
    df["mois"] = df["mois"].astype(str)
    df["pays"] = df["pays"].astype(str)
    return df.dropna(subset=["date_achat"]).reset_index(drop=True)


def normalize_scores(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None:
        return None
    df = df.copy()
    df["id_client"] = pd.to_numeric(df.get("id_client"), errors="coerce").astype("Int64")
    df["prob_reachat_12m"] = pd.to_numeric(df.get("prob_reachat_12m"), errors="coerce")
    df["expected_value_12m"] = pd.to_numeric(df.get("expected_value_12m"), errors="coerce")
    df["value_at_risk_12m"] = pd.to_numeric(df.get("value_at_risk_12m"), errors="coerce")
    df["segment_label"] = df.get("segment_label", "").astype(str)
    df = df.dropna(subset=["id_client"])
    return df.reset_index(drop=True)


def publish() -> None:
    db = get_database()
    gold = load_gold_tables()

    fact = normalize_fact(gold.get("fact_achats.parquet"))
    scores = normalize_scores(gold.get("client_scores.parquet"))
    ca_monthly = gold.get("ca_monthly.parquet")

    derived = {
        "daily": _build_daily(fact),
        "weekly": _build_weekly(fact),
        "distribution": _build_distribution(fact),
        "monthly_growth": _build_monthly_growth(ca_monthly),
    }

    heavy_tables = {
        "gold_fact_achats": (fact, "id_achat"),
        "gold_dim_clients": (gold.get("dim_clients.parquet"), "id_client"),
        "gold_client_features": (gold.get("client_features.parquet"), "id_client"),
        "gold_client_scores": (scores, "id_client"),
    }
    for coll_name, (df, key) in heavy_tables.items():
        records = _records(df)
        count = _upsert_by_key(db[coll_name], records, key)
        print(f"{coll_name}: upserté {count} documents (key={key})")

    replace_tables = {
        "gold_segment_summary": gold.get("segment_summary.parquet"),
        "gold_monthly": ca_monthly,
        "gold_by_country": gold.get("ca_country.parquet"),
        "gold_by_product": gold.get("ca_product.parquet"),
        "gold_cohort_first_purchase": gold.get("cohort_first_purchase.parquet"),
        "gold_daily": derived["daily"],
        "gold_weekly": derived["weekly"],
        "gold_distribution": derived["distribution"],
        "gold_monthly_growth": derived["monthly_growth"],
    }
    for coll_name, df in replace_tables.items():
        records = _records(df)
        if df is None:
            print(f"[WARN] {coll_name}: source absente, skip")
            continue
        written = _replace_all(db[coll_name], records)
        print(f"{coll_name}: inséré {written} documents (replace)")

    _ensure_indexes(db)
    total_achats = len(fact) if fact is not None else 0
    print(f"Publish done. Dataset fact_achats={total_achats} lignes.")


if __name__ == "__main__":
    start = perf_counter()
    publish()
    print(f"Temps total publish: {(perf_counter() - start)*1000:.0f} ms")
