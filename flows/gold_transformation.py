"""Transformation gold : agrégations actionnables pour l'électroménager.
- Objectifs : propension de ré-achat 12 mois + valeur attendue 12 mois.
- Tables : fact_achats, dim_clients, client_features, client_scores, segment_summary,
  ca_monthly, ca_country, ca_product, cohort_first_purchase.
"""
from io import BytesIO
from typing import Dict, List, Set, Tuple

import pandas as pd
from prefect import flow, task

from flows.config import BUCKET_GOLD, BUCKET_SILVER, get_minio_client

REQUIRED_CLIENT_COLS: List[str] = ["id_client", "nom", "email", "date_inscription", "pays"]
REQUIRED_ACHAT_COLS: List[str] = ["id_achat", "id_client", "date_achat", "montant", "produit"]
HORIZON_DAYS = 365
MAX_PURCHASE_AMOUNT = 10_000.0


def _read_parquet(bucket: str, object_name: str) -> pd.DataFrame:
    client = get_minio_client()
    obj = client.get_object(bucket, object_name)
    data = obj.read()
    obj.close()
    obj.release_conn()
    return pd.read_parquet(BytesIO(data))


def _write_parquet(df: pd.DataFrame, object_name: str) -> str:
    client = get_minio_client()
    if not client.bucket_exists(BUCKET_GOLD):
        client.make_bucket(BUCKET_GOLD)

    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)

    client.put_object(
        BUCKET_GOLD,
        object_name,
        buf,
        length=buf.getbuffer().nbytes,
        content_type="application/octet-stream",
    )
    print(f"Written {object_name} to gold ({len(df)} rows)")
    return object_name


def _validate_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name}: colonnes manquantes {missing}; présent {list(df.columns)}")


def _normalize_zero_one(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    min_v, max_v = series.min(), series.max()
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series(0.0, index=series.index)
    return series.sub(min_v).div(max_v - min_v).clip(0, 1)


@task
def load_clients(object_name: str = "clients.parquet") -> pd.DataFrame:
    return _read_parquet(BUCKET_SILVER, object_name)


@task
def load_achats(object_name: str = "achats.parquet") -> pd.DataFrame:
    return _read_parquet(BUCKET_SILVER, object_name)


def prepare_clients(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    _validate_columns(df, REQUIRED_CLIENT_COLS, "clients_silver")

    df["id_client"] = pd.to_numeric(df["id_client"], errors="coerce")
    df["date_inscription"] = pd.to_datetime(df["date_inscription"], errors="coerce", utc=True).dt.tz_localize(None)

    df = df.dropna(subset=["id_client", "date_inscription"])
    df["id_client"] = df["id_client"].astype(int)
    df["nom"] = df["nom"].astype(str).str.strip()
    df["email"] = df["email"].astype(str).str.strip().str.lower()
    df["pays"] = df["pays"].astype(str).str.strip().str.title()
    df = df.drop_duplicates("id_client", keep="first").sort_values("id_client")
    return df.reset_index(drop=True)


def prepare_achats(df: pd.DataFrame, valid_client_ids: Set[int]) -> pd.DataFrame:
    df = df.copy()
    _validate_columns(df, REQUIRED_ACHAT_COLS, "achats_silver")

    df["id_achat"] = pd.to_numeric(df["id_achat"], errors="coerce")
    df["id_client"] = pd.to_numeric(df["id_client"], errors="coerce")
    df["montant"] = pd.to_numeric(df["montant"], errors="coerce")
    df["date_achat"] = pd.to_datetime(df["date_achat"], errors="coerce", utc=True).dt.tz_localize(None)

    df = df.dropna(subset=["id_achat", "id_client", "montant", "date_achat", "produit"])
    df = df[(df["montant"] > 0) & (df["montant"] <= MAX_PURCHASE_AMOUNT)]

    if valid_client_ids:
        df = df[df["id_client"].isin(valid_client_ids)]

    df["produit"] = df["produit"].astype(str).str.strip().str.title()
    df = df.drop_duplicates("id_achat", keep="last").sort_values("date_achat")

    if (df["montant"] < 0).any():
        raise ValueError("Montant négatif détecté dans achats")

    return df.reset_index(drop=True)


def build_fact(achats_df: pd.DataFrame, clients_df: pd.DataFrame) -> pd.DataFrame:
    fact = achats_df.merge(clients_df[["id_client", "pays"]], on="id_client", how="left")
    fact["pays"] = fact["pays"].fillna("Inconnu")
    fact["mois"] = fact["date_achat"].dt.to_period("M").astype(str)
    fact["jour"] = fact["date_achat"].dt.date.astype(str)
    fact["annee"] = fact["date_achat"].dt.year.astype(int)
    fact = fact[
        [
            "id_achat",
            "id_client",
            "date_achat",
            "montant",
            "produit",
            "pays",
            "jour",
            "mois",
            "annee",
        ]
    ]
    return fact.reset_index(drop=True)


def build_dim_clients(clients_df: pd.DataFrame, fact_df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    agg = (
        fact_df.groupby("id_client")
        .agg(
            first_purchase=("date_achat", "min"),
            last_purchase=("date_achat", "max"),
            total_orders=("id_achat", "count"),
            total_spend=("montant", "sum"),
            product_count=("produit", "nunique"),
        )
        .reset_index()
    )

    dim = clients_df.merge(agg, on="id_client", how="left")
    dim["first_purchase"] = pd.to_datetime(dim["first_purchase"])
    dim["last_purchase"] = pd.to_datetime(dim["last_purchase"])

    dim["recency_days"] = dim["last_purchase"].map(
        lambda d: (reference_date - d).days if pd.notna(d) else HORIZON_DAYS
    )
    dim["tenure_days"] = dim["date_inscription"].map(lambda d: (reference_date - d).days if pd.notna(d) else 0)
    dim["avg_order_value"] = (
        dim["total_spend"] / dim["total_orders"].replace({0: pd.NA})
    ).fillna(0.0)

    return dim[
        [
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
        ]
    ].reset_index(drop=True)


def build_client_features(
    fact_df: pd.DataFrame, clients_df: pd.DataFrame, horizon_days: int
) -> Tuple[pd.DataFrame, pd.Timestamp]:
    reference_date = fact_df["date_achat"].max()
    if pd.isna(reference_date):
        raise ValueError("Impossible de calculer les features : aucune date d'achat valide")

    window_start = reference_date - pd.Timedelta(days=horizon_days)
    window = fact_df[fact_df["date_achat"] >= window_start].copy()

    per_client_window = window.groupby("id_client")
    per_client_all = fact_df.groupby("id_client")

    freq_12m = per_client_window.size()
    monetary_12m = per_client_window["montant"].sum()
    monetary_avg_12m = per_client_window["montant"].mean()
    diversity_12m = per_client_window["produit"].nunique()

    last_purchase = per_client_all["date_achat"].max()
    first_purchase = per_client_all["date_achat"].min()

    features = pd.DataFrame({"id_client": clients_df["id_client"].unique()})
    features["freq_12m"] = features["id_client"].map(freq_12m).fillna(0).astype(int)
    features["monetary_12m"] = features["id_client"].map(monetary_12m).fillna(0.0)
    features["monetary_avg_12m"] = features["id_client"].map(monetary_avg_12m).fillna(0.0)
    features["product_diversity_12m"] = features["id_client"].map(diversity_12m).fillna(0).astype(int)
    features["last_purchase"] = features["id_client"].map(last_purchase)
    features["first_purchase"] = features["id_client"].map(first_purchase)
    features["recency_days"] = features["last_purchase"].map(
        lambda d: (reference_date - d).days if pd.notna(d) else horizon_days
    )
    features["tenure_days"] = features["first_purchase"].map(
        lambda d: (reference_date - d).days if pd.notna(d) else 0
    )
    features["total_orders_all"] = features["id_client"].map(per_client_all.size()).fillna(0).astype(int)
    features["total_spend_all"] = features["id_client"].map(per_client_all["montant"].sum()).fillna(0.0)
    features["avg_order_value_all"] = (
        features["total_spend_all"] / features["total_orders_all"].replace(0, pd.NA)
    ).fillna(0.0)
    features["reference_date"] = reference_date.normalize()

    return features.reset_index(drop=True), reference_date


def score_clients(features: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    scored = features.copy()

    freq_norm = _normalize_zero_one(scored["freq_12m"].clip(upper=scored["freq_12m"].quantile(0.95)))
    rec_norm = 1 - _normalize_zero_one(scored["recency_days"].fillna(horizon_days).clip(upper=horizon_days))
    mon_norm = _normalize_zero_one(scored["monetary_12m"].clip(lower=0))
    div_norm = _normalize_zero_one(scored["product_diversity_12m"])

    scored["prob_reachat_12m"] = (0.45 * freq_norm + 0.30 * rec_norm + 0.15 * mon_norm + 0.10 * div_norm).fillna(
        0.0
    ).clip(0, 1)

    months_active = (scored["tenure_days"].clip(lower=1) / 30).clip(lower=1, upper=horizon_days / 30)
    monthly_baseline = scored["monetary_12m"] / months_active
    scored["expected_value_12m"] = (monthly_baseline * 12 * scored["prob_reachat_12m"]).fillna(0.0)
    scored["value_at_risk_12m"] = (scored["monetary_12m"] * (1 - scored["prob_reachat_12m"])).fillna(0.0)

    mon_hi = float(scored["monetary_12m"].quantile(0.75)) if not scored.empty else 0.0
    rec_hi = float(scored["recency_days"].quantile(0.75)) if not scored.empty else horizon_days
    rec_lo = float(scored["recency_days"].quantile(0.25)) if not scored.empty else horizon_days / 4
    freq_hi = float(scored["freq_12m"].quantile(0.75)) if not scored.empty else 0.0

    def _segment(row: pd.Series) -> str:
        if row["prob_reachat_12m"] >= 0.65 and row["monetary_12m"] >= mon_hi:
            return "VIP"
        if row["prob_reachat_12m"] >= 0.55 and row["recency_days"] <= rec_lo:
            return "Actifs"
        if row["monetary_12m"] >= mon_hi and row["recency_days"] > rec_hi:
            return "A relancer"
        if row["freq_12m"] <= max(1.0, freq_hi * 0.5) and row["recency_days"] > rec_hi:
            return "Dormants"
        return "A potentiel"

    scored["segment_label"] = scored.apply(_segment, axis=1)
    return scored.reset_index(drop=True)


def summarize_segments(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.groupby("segment_label")
        .agg(
            clients=("id_client", "count"),
            ca_12m=("monetary_12m", "sum"),
            expected_value_12m=("expected_value_12m", "sum"),
            value_at_risk_12m=("value_at_risk_12m", "sum"),
            freq_med=("freq_12m", "median"),
            recency_med=("recency_days", "median"),
        )
        .reset_index()
        .sort_values("expected_value_12m", ascending=False)
    )


def aggregate_sales(fact_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ca_monthly = (
        fact_df.groupby("mois")["montant"]
        .sum()
        .reset_index(name="ca")
        .sort_values("mois")
    )
    ca_country = (
        fact_df.groupby("pays")["montant"]
        .sum()
        .reset_index(name="ca")
        .sort_values("ca", ascending=False)
    )
    ca_product = (
        fact_df.groupby("produit")["montant"]
        .sum()
        .reset_index(name="ca")
        .sort_values("ca", ascending=False)
    )
    return ca_monthly, ca_country, ca_product


def build_cohort_first_purchase(fact_df: pd.DataFrame) -> pd.DataFrame:
    first_purchase = (
        fact_df.groupby("id_client")["date_achat"]
        .min()
        .dt.to_period("M")
        .astype(str)
        .rename("first_purchase_month")
    )
    fact_with_cohort = fact_df.merge(
        first_purchase.reset_index(), on="id_client", how="left"
    )
    cohort = (
        fact_with_cohort.groupby("first_purchase_month")
        .agg(clients=("id_client", "nunique"), ca=("montant", "sum"))
        .reset_index()
        .sort_values("first_purchase_month")
    )
    return cohort


@task
def write_gold_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    return {name: _write_parquet(df, name) for name, df in tables.items()}


@flow(name="Gold Transformation Flow")
def gold_transformation_flow(
    clients_object: str = "clients.parquet",
    achats_object: str = "achats.parquet",
    horizon_days: int = HORIZON_DAYS,
) -> Dict[str, Dict[str, str]]:
    clients_silver = load_clients(clients_object)
    achats_silver = load_achats(achats_object)

    clients = prepare_clients(clients_silver)
    achats = prepare_achats(achats_silver, set(clients["id_client"]))

    if achats.empty:
        raise ValueError("Aucun achat valide en entrée gold")

    fact_df = build_fact(achats, clients)
    features_df, reference_date = build_client_features(fact_df, clients, horizon_days)
    scores_df = score_clients(features_df, horizon_days)
    segments_df = summarize_segments(scores_df)
    dim_clients_df = build_dim_clients(clients, fact_df, reference_date)
    ca_monthly_df, ca_country_df, ca_product_df = aggregate_sales(fact_df)
    cohort_df = build_cohort_first_purchase(fact_df)

    tables = {
        "fact_achats.parquet": fact_df,
        "dim_clients.parquet": dim_clients_df,
        "client_features.parquet": features_df,
        "client_scores.parquet": scores_df,
        "segment_summary.parquet": segments_df,
        "ca_monthly.parquet": ca_monthly_df,
        "ca_country.parquet": ca_country_df,
        "ca_product.parquet": ca_product_df,
        "cohort_first_purchase.parquet": cohort_df,
    }

    written = write_gold_tables(tables)
    print("Gold generation done.")
    return {"written": written, "reference_date": {"value": reference_date.strftime('%Y-%m-%d')}}


if __name__ == "__main__":
    print(gold_transformation_flow())
