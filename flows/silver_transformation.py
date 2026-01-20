from io import BytesIO
from typing import Dict, List, Set, Tuple

import pandas as pd
from prefect import flow, task

from flows.config import BUCKET_BRONZE, BUCKET_SILVER, get_minio_client

DATE_LOWER_BOUND = pd.Timestamp("2000-01-01")
DATE_UPPER_BOUND = pd.Timestamp.utcnow().tz_localize(None)
MAX_PURCHASE_AMOUNT = 10_000.0
REQUIRED_CLIENT_COLS: List[str] = ["id_client", "nom", "email", "date_inscription", "pays"]
REQUIRED_ACHAT_COLS: List[str] = ["id_achat", "id_client", "date_achat", "montant", "produit"]


def _read_csv(bucket: str, object_name: str) -> pd.DataFrame:
    client = get_minio_client()
    obj = client.get_object(bucket, object_name)
    data = obj.read()
    obj.close()
    obj.release_conn()
    return pd.read_csv(BytesIO(data))


def _validate_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name}: colonnes manquantes {missing}; présent {list(df.columns)}")


def _write_parquet(df: pd.DataFrame, object_name: str) -> str:
    client = get_minio_client()
    if not client.bucket_exists(BUCKET_SILVER):
        client.make_bucket(BUCKET_SILVER)

    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)

    client.put_object(
        BUCKET_SILVER,
        object_name,
        buf,
        length=buf.getbuffer().nbytes,
        content_type="application/octet-stream",
    )
    return object_name


@task
def load_bronze(object_name: str) -> pd.DataFrame:
    return _read_csv(BUCKET_BRONZE, object_name)


@task
def clean_clients(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = df.copy()
    q: Dict[str, int] = {"initial_rows": len(df)}

    _validate_columns(df, REQUIRED_CLIENT_COLS, "clients")

    df["id_client"] = pd.to_numeric(df.get("id_client"), errors="coerce")
    q["dropped_invalid_id"] = int(df["id_client"].isna().sum())
    df = df.dropna(subset=["id_client"])

    base_missing = df[["nom", "email", "pays"]].isna().any(axis=1)
    q["dropped_missing"] = int(base_missing.sum())
    df = df[~base_missing]

    df["date_inscription"] = pd.to_datetime(
        df["date_inscription"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    q["dropped_invalid_date"] = int(df["date_inscription"].isna().sum())
    df = df.dropna(subset=["date_inscription"])

    out_of_range = (df["date_inscription"] < DATE_LOWER_BOUND) | (
        df["date_inscription"] > DATE_UPPER_BOUND
    )
    q["dropped_out_of_range"] = int(out_of_range.sum())
    df = df[~out_of_range]

    invalid_email = ~df["email"].astype(str).str.contains("@")
    q["dropped_invalid_email"] = int(invalid_email.sum())
    df = df[~invalid_email]

    df["nom"] = df["nom"].astype(str).str.strip()
    df["email"] = df["email"].astype(str).str.strip().str.lower()
    df["pays"] = df["pays"].astype(str).str.strip().str.title()

    df = df.sort_values("id_client")
    q["dropped_duplicates"] = int(df.duplicated("id_client").sum())
    df = df.drop_duplicates("id_client", keep="first")

    df["id_client"] = df["id_client"].astype(int)
    df["date_inscription"] = df["date_inscription"].dt.strftime("%Y-%m-%d")

    df = df[REQUIRED_CLIENT_COLS]
    q["final_rows"] = len(df)
    return df.reset_index(drop=True), q


@task
def clean_achats(
    df: pd.DataFrame, valid_client_ids: Set[int]
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = df.copy()
    q: Dict[str, int] = {"initial_rows": len(df)}

    _validate_columns(df, REQUIRED_ACHAT_COLS, "achats")

    df["id_achat"] = pd.to_numeric(df.get("id_achat"), errors="coerce")
    df["id_client"] = pd.to_numeric(df.get("id_client"), errors="coerce")
    df["montant"] = pd.to_numeric(df.get("montant"), errors="coerce")

    req_missing = df[
        ["id_achat", "id_client", "montant", "date_achat", "produit"]
    ].isna().any(axis=1)
    q["dropped_missing"] = int(req_missing.sum())
    df = df[~req_missing]

    df["date_achat"] = pd.to_datetime(
        df["date_achat"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    q["dropped_invalid_date"] = int(df["date_achat"].isna().sum())
    df = df.dropna(subset=["date_achat"])

    out_of_range = (df["date_achat"] < DATE_LOWER_BOUND) | (
        df["date_achat"] > DATE_UPPER_BOUND
    )
    q["dropped_out_of_range"] = int(out_of_range.sum())
    df = df[~out_of_range]

    bad_amount = (df["montant"] <= 0) | (df["montant"] > MAX_PURCHASE_AMOUNT)
    q["dropped_bad_amount"] = int(bad_amount.sum())
    df = df[~bad_amount]

    if valid_client_ids:
        orphan = ~df["id_client"].isin(valid_client_ids)
        q["dropped_orphan_client"] = int(orphan.sum())
        df = df[~orphan]

    df["produit"] = df["produit"].astype(str).str.strip().str.title()

    df = df.sort_values("id_achat")
    q["dropped_duplicates"] = int(df.duplicated("id_achat").sum())
    df = df.drop_duplicates("id_achat", keep="last")

    df["id_achat"] = df["id_achat"].astype(int)
    df["id_client"] = df["id_client"].astype(int)
    df["montant"] = df["montant"].astype(float)
    df["date_achat"] = df["date_achat"].dt.strftime("%Y-%m-%d %H:%M:%S")

    df = df[REQUIRED_ACHAT_COLS]
    q["final_rows"] = len(df)
    return df.reset_index(drop=True), q


@task
def write_silver(df: pd.DataFrame, object_name: str) -> str:
    return _write_parquet(df, object_name)


@flow(name="Silver Transformation Flow")
def silver_transformation_flow(
    bronze_clients_object: str = "clients.csv", bronze_achats_object: str = "achats.csv"
) -> Dict[str, Dict[str, object]]:
    raw_clients = load_bronze(bronze_clients_object)
    raw_achats = load_bronze(bronze_achats_object)

    clients_df, clients_q = clean_clients(raw_clients)
    achats_df, achats_q = clean_achats(raw_achats, set(clients_df["id_client"]))

    silver_clients = write_silver(clients_df, "clients.parquet")
    silver_achats = write_silver(achats_df, "achats.parquet")

    print("Quality clients:", clients_q)
    print("Quality achats:", achats_q)

    return {
        "clients": {"object_name": silver_clients, "quality": clients_q},
        "achats": {"object_name": silver_achats, "quality": achats_q},
    }


if __name__ == "__main__":
    print(silver_transformation_flow())
