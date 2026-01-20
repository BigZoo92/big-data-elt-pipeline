"""
Client segmentation and business advice generator.
 - Loads clients/purchases data (auto-detect columns, CSV/Parquet).
 - Builds RFM features, clusters with KMeans.
 - Produces segment summaries, advice, plots, and logs everything to MLflow.
"""

import argparse
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger("advisor")
RANDOM_STATE = 42


# -----------------------
# Path detection helpers
# -----------------------
def detect_default_paths() -> Tuple[Path, Path]:
    silver_dir = Path("data") / "silver"
    gold_dir = Path("data") / "gold"
    sources_dir = Path("data") / "sources"

    if silver_dir.exists():
        return silver_dir / "clients.parquet", silver_dir / "achats.parquet"
    if gold_dir.exists():
        return gold_dir / "clients.parquet", gold_dir / "achats.parquet"
    return sources_dir / "clients.csv", sources_dir / "achats.csv"


# -----------------------
# Data loading utilities
# -----------------------
def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def detect_column(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    if required:
        raise ValueError(
            f"Column not found. Expected one of {candidates}, got {list(df.columns)}"
        )
    return None


# -----------------------
# Feature engineering
# -----------------------
def build_rfm(df: pd.DataFrame, client_col: str, amount_col: str, date_col: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    df[client_col] = df[client_col].astype(str)
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
    df = df.dropna(subset=[client_col, amount_col])

    recency_series = None
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        if not df.empty:
            ref_date = df[date_col].max()
            recency_series = (ref_date - df.groupby(client_col)[date_col].max()).dt.days

    grouped = df.groupby(client_col)[amount_col]
    rfm = pd.DataFrame(
        {
            "frequency": grouped.count(),
            "monetary_total": grouped.sum(),
            "monetary_avg": grouped.mean(),
        }
    )

    if recency_series is not None:
        rfm["recency_days"] = recency_series
    else:
        rfm["recency_days"] = 0
        LOG.warning("No usable date column found; recency_days set to 0.")

    return rfm.reset_index().rename(columns={client_col: "client_id"})


# -----------------------
# Modeling & scoring
# -----------------------
def train_model(features: pd.DataFrame, n_clusters: int) -> Tuple[Pipeline, np.ndarray, Optional[float]]:
    numeric_cols = ["frequency", "monetary_total", "monetary_avg", "recency_days"]
    X = features[numeric_cols].values

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)),
        ]
    )
    pipe.fit(X)
    labels = pipe.predict(X)

    sil = None
    if len(features) > n_clusters > 1:
        try:
            sil = float(silhouette_score(X, labels))
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("Silhouette score failed: %s", exc)
    return pipe, labels, sil


def label_segments(features: pd.DataFrame) -> pd.DataFrame:
    q_amount = features["monetary_total"].quantile(0.75)
    q_recency = features["recency_days"].quantile(0.75)
    q_recency_low = features["recency_days"].quantile(0.25)
    q_freq = features["frequency"].quantile(0.75)

    def _label(row: pd.Series) -> str:
        if row["monetary_total"] >= q_amount and row["recency_days"] <= q_recency_low:
            return "VIP"
        if row["monetary_total"] >= q_amount and row["recency_days"] > q_recency:
            return "A relancer"
        if row["frequency"] >= q_freq:
            return "Fideles"
        if row["frequency"] <= 1 and row["recency_days"] <= q_recency_low:
            return "Nouveaux"
        return "Dormants"

    features["segment_label"] = features.apply(_label, axis=1)
    return features


# -----------------------
# Reporting & advice
# -----------------------
def summarize_segments(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.groupby(["segment", "segment_label"])
        .agg(
            size=("client_id", "count"),
            revenue_total=("monetary_total", "sum"),
            revenue_avg=("monetary_total", "mean"),
            frequency_avg=("frequency", "mean"),
            recency_avg=("recency_days", "mean"),
        )
        .reset_index()
        .sort_values("revenue_total", ascending=False)
    )


def select_top_clients(scored: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    to_reactivate = scored.sort_values(
        ["recency_days", "monetary_total"], ascending=[False, False]
    ).head(10)
    vip = scored.sort_values(
        ["monetary_total", "recency_days"], ascending=[False, True]
    ).head(10)
    return to_reactivate, vip


def write_advice(advice_path: Path, segments: pd.DataFrame, to_reactivate: pd.DataFrame, vip: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# Conseils business\n")
    lines.append("## Actions rapides (winback / upsell / fidélisation)\n")
    lines.extend(
        [
            "- Relancer les segments \"A relancer\" avec offres limitées dans le temps.",
            "- Proposer un programme VIP aux segments \"VIP\" (early access, support dédié).",
            "- Offrir un bon de réduction ciblé aux clients Dormants à forte valeur passée.",
            "- Tester des bundles produit pour les Fidèles à fréquence élevée.",
            "- Automatiser des emails post-achat pour encourager un deuxième achat.",
            "- Activer des reminders SMS aux clients à haute récence (Nouveaux) pour augmenter la fréquence.",
            "- Mettre en place un parrainage pour fidéliser et recruter via les VIP/Fidèles.",
        ]
    )

    def _fmt_clients(df: pd.DataFrame, title: str) -> None:
        lines.append(f"\n## {title}\n")
        if df.empty:
            lines.append("Aucun client identifié pour cette catégorie.\n")
            return
        lines.append("client_id | monetary_total | recency_days | frequency")
        lines.append("--- | --- | --- | ---")
        for _, row in df.iterrows():
            lines.append(
                f"{row['client_id']} | {row['monetary_total']:.2f} | {row['recency_days']:.0f} | {row['frequency']:.0f}"
            )
        lines.append("")

    _fmt_clients(to_reactivate, "Top 10 clients à relancer")
    _fmt_clients(vip, "Top 10 VIP à chouchouter")

    advice_path.write_text("\n".join(lines), encoding="utf-8")


def plot_segment_revenue(summary: pd.DataFrame, plot_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(summary["segment_label"], summary["revenue_total"])
    plt.title("CA par segment")
    plt.ylabel("CA")
    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()


# -----------------------
# MLflow logging
# -----------------------
def log_to_mlflow(
    params: Dict[str, object],
    metrics: Dict[str, float],
    artifacts_dir: Path,
    model: Pipeline,
) -> None:
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("advisor")
    with mlflow.start_run(run_name="advisor_run") as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(artifacts_dir))
        mlflow.sklearn.log_model(model, artifact_path="model")
        LOG.info("Logged to MLflow run_id=%s", run.info.run_id)


# -----------------------
# CLI and main workflow
# -----------------------
def parse_args() -> argparse.Namespace:
    default_clients, default_achats = detect_default_paths()
    parser = argparse.ArgumentParser(description="Client segmentation advisor")
    parser.add_argument("--clients-path", type=Path, default=default_clients, help="Path to clients file (CSV/Parquet)")
    parser.add_argument("--achats-path", type=Path, default=default_achats, help="Path to purchases file (CSV/Parquet)")
    parser.add_argument("--n-clusters", type=int, default=5, help="Number of KMeans clusters")
    parser.add_argument("--experiment", type=str, default="advisor", help="MLflow experiment name")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    args = parse_args()
    LOG.info("Clients path: %s", args.clients_path)
    LOG.info("Achats path: %s", args.achats_path)

    mlflow.set_experiment(args.experiment)

    clients_df = load_table(args.clients_path)
    achats_df = load_table(args.achats_path)

    client_col = detect_column(
        achats_df,
        ["client_id", "customer_id", "id_client", "id", "client"],
    )
    amount_col = detect_column(
        achats_df,
        ["amount", "total", "montant", "price", "value"],
    )
    date_col = detect_column(
        achats_df,
        ["date", "purchase_date", "created_at", "timestamp", "date_achat"],
        required=False,
    )

    LOG.info(
        "Detected columns -> client: %s, amount: %s, date: %s",
        client_col,
        amount_col,
        date_col or "None",
    )

    rfm = build_rfm(achats_df, client_col, amount_col, date_col)
    nb_clients = rfm["client_id"].nunique()
    nb_achats = len(achats_df)
    LOG.info("Computed RFM for %s clients (nb achats: %s)", nb_clients, nb_achats)

    model, labels, sil = train_model(rfm, args.n_clusters)
    rfm["segment"] = labels
    rfm = label_segments(rfm)

    summary = summarize_segments(rfm)
    to_reactivate, vip = select_top_clients(rfm)

    run_id = uuid.uuid4().hex[:8]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data") / "advisor" / f"{ts}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"

    clients_scored_path = out_dir / "clients_scored.csv"
    segment_summary_path = out_dir / "segment_summary.csv"
    advice_path = out_dir / "advice.md"
    plot_path = plots_dir / "segment_revenue.png"

    rfm.to_csv(clients_scored_path, index=False)
    summary.to_csv(segment_summary_path, index=False)
    write_advice(advice_path, summary, to_reactivate, vip)
    plot_segment_revenue(summary, plot_path)

    metrics = {
        "silhouette": sil if sil is not None else 0.0,
        "total_revenue": float(rfm["monetary_total"].sum()),
        "aov_global": float(rfm["monetary_total"].sum() / max(nb_achats, 1)),
        "repeat_rate": float((rfm["frequency"] > 1).mean()),
    }
    params = {
        "n_clusters": args.n_clusters,
        "clients_path": str(args.clients_path),
        "achats_path": str(args.achats_path),
        "nb_clients": nb_clients,
        "nb_achats": nb_achats,
    }

    log_to_mlflow(params, metrics, out_dir, model)

    LOG.info("Artifacts saved to %s", out_dir)
    LOG.info("Done.")


if __name__ == "__main__":
    main()
