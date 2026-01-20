"""
Sanity check minimal de la couche gold (MinIO).
- Verifie la presence des fichiers Parquet attendus.
- Controle quelques colonnes cles, montants et dates.
"""
from io import BytesIO

import pandas as pd

from flows.config import BUCKET_GOLD, get_minio_client

EXPECTED_COLUMNS = {
    "dim_clients.parquet": [
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
    ],
    "fact_achats.parquet": ["id_achat", "id_client", "date_achat", "montant", "produit", "mois", "pays"],
    "client_features.parquet": [
        "id_client",
        "freq_12m",
        "monetary_12m",
        "monetary_avg_12m",
        "recency_days",
        "tenure_days",
        "product_diversity_12m",
    ],
    "client_scores.parquet": [
        "id_client",
        "prob_reachat_12m",
        "expected_value_12m",
        "value_at_risk_12m",
        "segment_label",
        "recency_days",
        "freq_12m",
        "monetary_12m",
    ],
    "segment_summary.parquet": ["segment_label", "expected_value_12m", "clients"],
    "ca_monthly.parquet": ["mois", "ca"],
    "ca_country.parquet": ["pays", "ca"],
    "ca_product.parquet": ["produit", "ca"],
    "cohort_first_purchase.parquet": ["first_purchase_month", "clients", "ca"],
}


def _read_parquet(client, object_name: str) -> pd.DataFrame:
    obj = client.get_object(BUCKET_GOLD, object_name)
    data = obj.read()
    obj.close()
    obj.release_conn()
    return pd.read_parquet(BytesIO(data))


def main() -> None:
    client = get_minio_client()

    statuses = []
    for object_name, required_cols in EXPECTED_COLUMNS.items():
        try:
            df = _read_parquet(client, object_name)
        except Exception as exc:  # pragma: no cover - quick check script
            statuses.append((object_name, False, f"absent ou illisible ({exc})"))
            continue

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            statuses.append((object_name, False, f"colonnes manquantes {missing}"))
            continue

        if "montant" in df.columns and (df["montant"] < 0).any():
            statuses.append((object_name, False, "montant negatif detecte"))
            continue

        if "date_achat" in df.columns:
            pd.to_datetime(df["date_achat"], errors="raise")

        statuses.append((object_name, True, f"ok ({len(df)} lignes)"))

    print("=== Verification gold ===")
    for name, ok, msg in statuses:
        status = "OK" if ok else "KO"
        print(f"{status} - {name} - {msg}")


if __name__ == "__main__":
    main()
