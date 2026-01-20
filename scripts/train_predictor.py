import argparse
import json
import logging
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

LOG = logging.getLogger("predictor")
HORIZON_DAYS = 365


def detect_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    if required:
        raise ValueError(f"Colonne manquante. Attendu {candidates}, trouvé {list(df.columns)}")
    return None


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def detect_default_paths() -> Tuple[Path, Path]:
    silver_dir = Path("data") / "silver"
    gold_dir = Path("data") / "gold"
    sources_dir = Path("data") / "sources"
    if (silver_dir / "clients.parquet").exists():
        return silver_dir / "clients.parquet", silver_dir / "achats.parquet"
    if (gold_dir / "clients.parquet").exists():
        return gold_dir / "clients.parquet", gold_dir / "achats.parquet"
    return sources_dir / "clients.csv", sources_dir / "achats.csv"


def load_and_standardize(clients_path: Path, achats_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clients = load_table(clients_path)
    achats = load_table(achats_path)

    client_col = detect_column(clients, ["id_client", "client_id", "customer_id"])
    clients = clients.rename(columns={client_col: "id_client"})

    client_col_achats = detect_column(achats, ["id_client", "client_id", "customer_id", "id"])
    amount_col = detect_column(achats, ["montant", "amount", "price", "value", "total"])
    date_col = detect_column(achats, ["date_achat", "date", "purchase_date", "created_at", "timestamp"])
    product_col = detect_column(achats, ["produit", "product", "item"], required=False)

    achats = achats.rename(
        columns={
            client_col_achats: "id_client",
            amount_col: "montant",
            date_col: "date_achat",
            **({product_col: "produit"} if product_col else {}),
        }
    )
    achats["date_achat"] = pd.to_datetime(achats["date_achat"], errors="coerce")
    achats["montant"] = pd.to_numeric(achats["montant"], errors="coerce")
    achats = achats.dropna(subset=["id_client", "montant", "date_achat"])
    achats["id_client"] = achats["id_client"].astype(int)
    achats["montant"] = achats["montant"].astype(float)
    achats["produit"] = achats.get("produit", "NA").astype(str).str.title()

    LOG.info(
        "Colonnes détectées achats -> id_client=%s montant=%s date_achat=%s produit=%s",
        client_col_achats,
        amount_col,
        date_col,
        product_col or "None",
    )
    return clients, achats


def _normalize_zero_one(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    min_v, max_v = series.min(), series.max()
    if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
        return pd.Series(0.0, index=series.index)
    return series.sub(min_v).div(max_v - min_v).clip(0, 1)


def build_features(achats: pd.DataFrame, horizon_days: int = HORIZON_DAYS) -> Tuple[pd.DataFrame, pd.Timestamp]:
    reference_date = achats["date_achat"].max()
    window_start = reference_date - timedelta(days=horizon_days)
    window = achats[achats["date_achat"] >= window_start].copy()

    per_client_window = window.groupby("id_client")
    per_client_all = achats.groupby("id_client")

    freq_12m = per_client_window["id_achat"].count() if "id_achat" in achats.columns else per_client_window.size()
    monetary_12m = per_client_window["montant"].sum()
    monetary_avg_12m = per_client_window["montant"].mean()
    diversity_12m = per_client_window["produit"].nunique() if "produit" in window.columns else pd.Series(dtype=float)

    last_purchase = per_client_all["date_achat"].max()
    first_purchase = per_client_all["date_achat"].min()

    features = pd.DataFrame({"id_client": achats["id_client"].unique()})
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
    features["total_orders_all"] = features["id_client"].map(per_client_all["date_achat"].count()).fillna(0).astype(int)
    features["total_spend_all"] = features["id_client"].map(per_client_all["montant"].sum()).fillna(0.0)
    features["avg_order_value_all"] = (
        features["total_spend_all"] / features["total_orders_all"].replace(0, pd.NA)
    ).fillna(0.0)

    return features, reference_date


def score_clients(features: pd.DataFrame, horizon_days: int = HORIZON_DAYS) -> pd.DataFrame:
    scored = features.copy()

    freq_norm = _normalize_zero_one(scored["freq_12m"].clip(upper=scored["freq_12m"].quantile(0.95)))
    rec_norm = 1 - _normalize_zero_one(scored["recency_days"].fillna(horizon_days).clip(upper=horizon_days))
    mon_norm = _normalize_zero_one(scored["monetary_12m"].clip(lower=0))
    div_norm = _normalize_zero_one(scored["product_diversity_12m"])

    scored["prob_reachat_12m"] = (
        0.45 * freq_norm + 0.30 * rec_norm + 0.15 * mon_norm + 0.10 * div_norm
    ).fillna(0.0).clip(0, 1)

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
    return scored


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


def select_opportunities(scored: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    top_potential = scored.sort_values("expected_value_12m", ascending=False).head(20)
    to_reactivate = scored.sort_values(
        ["value_at_risk_12m", "recency_days"], ascending=[False, False]
    ).head(20)
    return top_potential, to_reactivate


def write_outputs(
    scored: pd.DataFrame,
    summary: pd.DataFrame,
    out_dir: Path,
) -> Tuple[Path, Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    preds_path = out_dir / "predictions.csv"
    summary_path = out_dir / "segment_summary.csv"
    metrics_path = out_dir / "metrics.json"
    report_path = out_dir / "report.md"

    scored.to_csv(preds_path, index=False)
    summary.to_csv(summary_path, index=False)

    metrics = {
        "total_clients": int(scored["id_client"].nunique()),
        "expected_value_12m_total": float(scored["expected_value_12m"].sum()),
        "value_at_risk_12m_total": float(scored["value_at_risk_12m"].sum()),
        "median_prob_reachat_12m": float(scored["prob_reachat_12m"].median()),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    top_potential, to_reactivate = select_opportunities(scored)

    lines = [
        "# Rapport scoring (12 mois)",
        f"- Clients scorés : {len(scored)}",
        f"- Valeur attendue totale 12m : {metrics['expected_value_12m_total']:.2f}",
        f"- Valeur à risque 12m : {metrics['value_at_risk_12m_total']:.2f}",
        "",
        "## Top 20 potentiel (valeur attendue)",
        "id_client | prob_reachat_12m | expected_value_12m | recency_days | freq_12m | monetary_12m",
        "--- | --- | --- | --- | --- | ---",
    ]
    for _, row in top_potential.iterrows():
        lines.append(
            f"{row['id_client']} | {row['prob_reachat_12m']:.3f} | {row['expected_value_12m']:.2f} | "
            f"{row['recency_days']:.0f} | {row['freq_12m']:.0f} | {row['monetary_12m']:.2f}"
        )
    lines.extend(
        [
            "",
            "## Top 20 à réactiver (valeur à risque)",
            "id_client | value_at_risk_12m | recency_days | freq_12m | monetary_12m",
            "--- | --- | --- | --- | ---",
        ]
    )
    for _, row in to_reactivate.iterrows():
        lines.append(
            f"{row['id_client']} | {row['value_at_risk_12m']:.2f} | {row['recency_days']:.0f} | "
            f"{row['freq_12m']:.0f} | {row['monetary_12m']:.2f}"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    plt.figure(figsize=(7, 4))
    summary.plot.bar(x="segment_label", y="expected_value_12m", legend=False)
    plt.title("Valeur attendue 12m par segment")
    plt.tight_layout()
    plt.savefig(plots_dir / "expected_value_by_segment.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.scatter(scored["recency_days"], scored["expected_value_12m"], alpha=0.5)
    plt.xlabel("Recency (jours)")
    plt.ylabel("Valeur attendue 12m")
    plt.title("Recency vs valeur attendue")
    plt.tight_layout()
    plt.savefig(plots_dir / "recency_vs_value.png", dpi=150)
    plt.close()

    return preds_path, summary_path, metrics_path, report_path


def log_mlflow(
    tracking_uri: str,
    experiment: str,
    params: Dict[str, object],
    metrics: Dict[str, float],
    artifacts_dir: Path,
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="predictor_run") as run:
        mlflow.log_params(params)
        metrics_to_log = {k: float(v) for k, v in metrics.items() if v is not None and not pd.isna(v)}
        mlflow.log_metrics(metrics_to_log)
        mlflow.log_artifacts(str(artifacts_dir))
        LOG.info("MLflow run logged: %s", run.info.run_id)


def parse_args() -> argparse.Namespace:
    default_clients, default_achats = detect_default_paths()
    parser = argparse.ArgumentParser(description="Scoring clients (propension & valeur 12 mois)")
    parser.add_argument("--clients-path", type=Path, default=default_clients)
    parser.add_argument("--achats-path", type=Path, default=default_achats)
    parser.add_argument("--tracking-uri", type=str, default="file:./mlruns")
    parser.add_argument("--experiment-name", type=str, default="predictor")
    parser.add_argument("--horizon-days", type=int, default=HORIZON_DAYS)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args = parse_args()
    np.random.seed(args.random_state)

    clients, achats = load_and_standardize(args.clients_path, args.achats_path)
    features, reference_date = build_features(achats, horizon_days=args.horizon_days)
    scored = score_clients(features, horizon_days=args.horizon_days)
    summary = summarize_segments(scored)

    run_id = uuid.uuid4().hex[:8]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data") / "advisor" / f"{ts}_{run_id}"

    preds_path, summary_path, metrics_path, report_path = write_outputs(scored, summary, out_dir)

    global_metrics = {
        "total_revenue": float(achats["montant"].sum()),
        "aov": float(achats["montant"].mean()),
        "repeat_rate": float((achats.groupby("id_client")["montant"].count() > 1).mean()),
    }
    mlflow_metrics = {
        **json.loads(metrics_path.read_text()),
        **global_metrics,
    }
    mlflow_params = {
        "horizon_days": args.horizon_days,
        "clients_path": str(args.clients_path),
        "achats_path": str(args.achats_path),
        "random_state": args.random_state,
        "reference_date": reference_date.strftime("%Y-%m-%d"),
    }

    log_mlflow(args.tracking_uri, args.experiment_name, mlflow_params, mlflow_metrics, out_dir)

    LOG.info("Artifacts: %s", out_dir)
    LOG.info("Predictions: %s", preds_path)
    LOG.info("Segments: %s", summary_path)
    LOG.info("Report: %s", report_path)
    LOG.info("Done.")


if __name__ == "__main__":
    main()
