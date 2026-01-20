from __future__ import annotations

import json
import os
import sys
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from flows.config import BUCKET_GOLD, get_minio_client

API_DEFAULT = os.getenv("SERVING_API_URL", "http://localhost:5000")
REFRESH_BENCH_PATH = Path("data/metrics/refresh_benchmark.json")
PIPELINE_BENCH_PATH = Path("data/metrics/benchmark.json")


@st.cache_data(show_spinner=False)
def load_parquet_from_minio(object_name: str) -> Optional[pd.DataFrame]:
    client = get_minio_client()
    try:
        obj = client.get_object(BUCKET_GOLD, object_name)
        data = obj.read()
        obj.close()
        obj.release_conn()
        return pd.read_parquet(BytesIO(data))
    except Exception:
        return None


def _read_parquet_uncached(object_name: str) -> Optional[pd.DataFrame]:
    client = get_minio_client()
    try:
        obj = client.get_object(BUCKET_GOLD, object_name)
        data = obj.read()
        obj.close()
        obj.release_conn()
        return pd.read_parquet(BytesIO(data))
    except Exception:
        return None


def load_gold_parquet() -> Dict[str, Optional[pd.DataFrame]]:
    files = {
        "fact": "fact_achats.parquet",
        "scores": "client_scores.parquet",
        "segments": "segment_summary.parquet",
        "ca_monthly": "ca_monthly.parquet",
        "ca_country": "ca_country.parquet",
        "ca_product": "ca_product.parquet",
        "cohort": "cohort_first_purchase.parquet",
    }
    return {key: load_parquet_from_minio(path) for key, path in files.items()}


def build_distribution_from_fact(fact: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    if fact is None or fact.empty:
        return pd.DataFrame()
    df = fact.copy()
    df["montant"] = pd.to_numeric(df["montant"], errors="coerce")
    df = df.dropna(subset=["montant"])
    cuts = pd.cut(df["montant"], bins=bins)
    distrib = cuts.value_counts().sort_index().reset_index()
    distrib.columns = ["bucket", "count"]
    distrib["bucket"] = distrib["bucket"].astype(str)
    return distrib


@st.cache_data(show_spinner=False)
def fetch_api_json(endpoint: str, base_url: str) -> dict:
    url = f"{base_url.rstrip('/')}{endpoint}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()


def fetch_api_json_uncached(endpoint: str, base_url: str) -> dict:
    url = f"{base_url.rstrip('/')}{endpoint}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()


def load_api_data(base_url: str) -> Tuple[Optional[Dict[str, pd.DataFrame]], Optional[dict], Optional[str]]:
    endpoints = {
        "kpis": "/kpis",
        "monthly": "/monthly",
        "by_country": "/by_country",
        "by_product": "/by_product",
        "segments": "/segments",
        "scores": "/scores",
        "distribution": "/distribution",
        "cohort": "/cohort",
    }
    payload: Dict[str, dict] = {}
    for key, endpoint in endpoints.items():
        try:
            payload[key] = fetch_api_json(endpoint, base_url)
        except Exception as exc:
            return None, None, f"{endpoint} : {exc}"

    frames = {
        "ca_monthly": pd.DataFrame(payload["monthly"].get("data", [])),
        "ca_country": pd.DataFrame(payload["by_country"].get("data", [])),
        "ca_product": pd.DataFrame(payload["by_product"].get("data", [])),
        "segments": pd.DataFrame(payload["segments"].get("data", [])),
        "scores": pd.DataFrame(payload["scores"].get("data", [])),
        "distribution": pd.DataFrame(payload["distribution"].get("data", [])),
        "cohort": pd.DataFrame(payload["cohort"].get("data", [])),
    }
    return frames, payload.get("kpis"), None


def _fmt_eur(value: float) -> str:
    return f"{value:,.0f} EUR".replace(",", " ")


def render_kpis(kpis: Optional[dict], fact: Optional[pd.DataFrame], scores: Optional[pd.DataFrame]) -> None:
    if kpis:
        total_ca = kpis.get("total_ca", 0.0) or 0.0
        total_achats = kpis.get("total_achats", 0) or 0
        total_clients = kpis.get("total_clients", 0) or 0
        panier_moyen = kpis.get("panier_moyen", 0.0) or 0.0
        expected_total = kpis.get("expected_total")
    elif fact is not None:
        total_ca = fact["montant"].sum()
        total_achats = len(fact)
        total_clients = fact["id_client"].nunique()
        panier_moyen = fact["montant"].mean()
        expected_total = scores["expected_value_12m"].sum() if scores is not None else None
    else:
        st.warning("Aucune donnée KPI disponible.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CA total", _fmt_eur(total_ca))
    col2.metric("Nb achats", f"{total_achats:,}".replace(",", " "))
    col3.metric("Nb clients", f"{total_clients:,}".replace(",", " "))
    col4.metric("Panier moyen", _fmt_eur(panier_moyen))
    if expected_total is not None:
        st.caption(f"Valeur attendue 12m (scoring heuristique) : {_fmt_eur(expected_total)}")


def render_sales_charts(
    fact: Optional[pd.DataFrame],
    ca_monthly: pd.DataFrame,
    ca_country: pd.DataFrame,
    ca_product: pd.DataFrame,
    distribution: Optional[pd.DataFrame] = None,
) -> None:
    st.subheader("CA par mois")
    st.plotly_chart(px.line(ca_monthly, x="mois", y="ca", markers=True), use_container_width=True)

    st.subheader("CA par pays")
    st.plotly_chart(px.bar(ca_country.sort_values("ca", ascending=False), x="pays", y="ca"), use_container_width=True)

    st.subheader("Top produits (CA)")
    top_products = ca_product.sort_values("ca", ascending=False).head(15)
    st.plotly_chart(px.bar(top_products, x="produit", y="ca"), use_container_width=True)

    st.subheader("Distribution des montants panier")
    if distribution is not None and not distribution.empty:
        st.plotly_chart(px.bar(distribution, x="bucket", y="count"), use_container_width=True)
    elif fact is not None:
        st.plotly_chart(px.histogram(fact, x="montant", nbins=30), use_container_width=True)
    else:
        st.info("Distribution indisponible (ni API ni Parquet).")


def render_segments(scores: pd.DataFrame, segments: pd.DataFrame) -> None:
    st.subheader("Segmentation RFM (12 mois)")
    st.plotly_chart(
        px.bar(
            segments,
            x="segment_label",
            y="expected_value_12m",
            color="segment_label",
            title="Valeur attendue 12m par segment",
        ),
        use_container_width=True,
    )

    st.subheader("Distribution propension de re-achat (12m)")
    st.plotly_chart(px.histogram(scores, x="prob_reachat_12m", nbins=30), use_container_width=True)


def render_opportunities(scores: pd.DataFrame) -> None:
    st.subheader("Top opportunités")
    top_value = scores.sort_values("expected_value_12m", ascending=False).head(20)
    to_reactivate = scores.sort_values(["value_at_risk_12m", "recency_days"], ascending=[False, False]).head(20)

    col1, col2 = st.columns(2)
    col1.caption("Plus forte valeur attendue (12 mois)")
    col1.dataframe(
        top_value[
            [
                "id_client",
                "prob_reachat_12m",
                "expected_value_12m",
                "recency_days",
                "freq_12m",
                "monetary_12m",
                "segment_label",
            ]
        ],
        use_container_width=True,
    )

    col2.caption("A réactiver (valeur à risque)")
    col2.dataframe(
        to_reactivate[
            [
                "id_client",
                "value_at_risk_12m",
                "recency_days",
                "freq_12m",
                "monetary_12m",
                "segment_label",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Recency vs valeur attendue")
    st.plotly_chart(
        px.scatter(
            scores,
            x="recency_days",
            y="expected_value_12m",
            color="segment_label",
            hover_data=["id_client", "freq_12m", "monetary_12m", "prob_reachat_12m"],
        ),
        use_container_width=True,
    )


def render_cohorts(cohort: pd.DataFrame) -> None:
    st.subheader("Cohortes (mois du 1er achat)")
    st.plotly_chart(
        px.bar(cohort, x="first_purchase_month", y="clients", title="Nouveaux clients par cohorte"),
        use_container_width=True,
    )
    st.plotly_chart(
        px.bar(cohort, x="first_purchase_month", y="ca", title="CA par cohorte (mois 1er achat)"),
        use_container_width=True,
    )


def render_pipeline_benchmark_tab() -> None:
    if not PIPELINE_BENCH_PATH.exists():
        st.info("Pas encore de benchmark pipeline. Lance `python scripts/benchmark.py`.")
        return
    try:
        history = json.loads(PIPELINE_BENCH_PATH.read_text())
    except Exception:
        history = []
    if not history:
        st.info("Benchmark vide. Relancer le script.")
        return

    df = pd.json_normalize(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["pandas_total"] = df.get("durations.pandas.total", pd.Series(dtype=float))
    df["spark_total"] = df.get("durations.spark.total", pd.Series(dtype=float))
    latest = df.sort_values("timestamp").tail(1).iloc[0]
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset size (achats)", int(latest.get("dataset_size", 0)))
    col2.metric("Temps total Pandas (s)", f"{latest['pandas_total']:.2f}")
    col3.metric("Temps total Spark (s)", f"{latest['spark_total']:.2f}")

    st.subheader("Historique")
    st.dataframe(
        df[["timestamp", "dataset_size", "scale", "pandas_total", "spark_total"]],
        use_container_width=True,
    )

    st.subheader("Temps total vs temps")
    st.plotly_chart(
        px.line(df, x="timestamp", y=["pandas_total", "spark_total"], markers=True),
        use_container_width=True,
    )


def benchmark_parquet_once() -> Optional[dict]:
    files = {
        "fact": "fact_achats.parquet",
        "scores": "client_scores.parquet",
        "segments": "segment_summary.parquet",
        "ca_monthly": "ca_monthly.parquet",
        "ca_country": "ca_country.parquet",
        "ca_product": "ca_product.parquet",
        "cohort": "cohort_first_purchase.parquet",
    }
    timings: Dict[str, float] = {}
    data: Dict[str, pd.DataFrame] = {}
    start = perf_counter()
    for key, path in files.items():
        t0 = perf_counter()
        df = _read_parquet_uncached(path)
        if df is None:
            return None
        timings[key] = (perf_counter() - t0) * 1000
        data[key] = df
    total_ms = (perf_counter() - start) * 1000
    dataset_size = len(data["fact"]) if data.get("fact") is not None else 0
    return {
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "parquet",
        "total_ms": total_ms,
        "breakdown_ms": timings,
        "dataset_size": dataset_size,
    }


def benchmark_api_once(base_url: str) -> Optional[dict]:
    endpoints = {
        "kpis": "/kpis",
        "monthly": "/monthly",
        "by_country": "/by_country",
        "by_product": "/by_product",
        "segments": "/segments",
        "scores": "/scores",
        "distribution": "/distribution",
        "cohort": "/cohort",
    }
    timings: Dict[str, float] = {}
    start = perf_counter()
    dataset_size = None
    for key, endpoint in endpoints.items():
        t0 = perf_counter()
        try:
            payload = fetch_api_json_uncached(endpoint, base_url)
        except Exception:
            return None
        timings[key] = (perf_counter() - t0) * 1000
        if key == "kpis":
            dataset_size = payload.get("dataset_size")
    total_ms = (perf_counter() - start) * 1000
    return {
        "timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "api",
        "total_ms": total_ms,
        "breakdown_ms": timings,
        "dataset_size": dataset_size,
    }


def append_refresh_history(entries: list) -> None:
    REFRESH_BENCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    history = []
    if REFRESH_BENCH_PATH.exists():
        try:
            history = json.loads(REFRESH_BENCH_PATH.read_text())
        except Exception:
            history = []
    history.extend(entries)
    REFRESH_BENCH_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def load_refresh_history() -> list:
    if not REFRESH_BENCH_PATH.exists():
        return []
    try:
        return json.loads(REFRESH_BENCH_PATH.read_text())
    except Exception:
        return []


def render_refresh_benchmark_tab(api_base_url: str) -> None:
    st.write("Compare temps de refresh Parquet direct vs API+Mongo (runs successifs).")
    runs = st.number_input("Nombre de runs (par source)", min_value=1, max_value=5, value=3, step=1)
    if st.button("Run benchmark refresh"):
        entries = []
        for _ in range(runs):
            parquet_res = benchmark_parquet_once()
            api_res = benchmark_api_once(api_base_url)
            if parquet_res:
                entries.append(parquet_res)
            if api_res:
                entries.append(api_res)
        if entries:
            append_refresh_history(entries)
            st.success(f"Benchmarks ajoutés ({len(entries)} entrées).")
        else:
            st.error("Impossible de collecter les benchmarks (API ou Parquet indisponible).")

    history = load_refresh_history()
    if not history:
        st.info("Pas encore de mesure. Lance le benchmark ci-dessus.")
        return

    df = pd.json_normalize(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    latest = df.sort_values("timestamp").groupby("source").tail(1)
    col1, col2 = st.columns(2)
    if not latest.empty:
        for _, row in latest.iterrows():
            ds_val = row.get("dataset_size", 0)
            dataset_size = 0
            if pd.notna(ds_val):
                try:
                    dataset_size = int(ds_val)
                except Exception:
                    dataset_size = 0
            target_col = col1 if row["source"] == "api" else col2
            target_col.metric(
                f"Dernier {row['source']}",
                f"{float(row['total_ms']):.0f} ms",
                help=f"dataset_size={dataset_size}",
            )

    st.subheader("Historique")
    st.dataframe(
        df[["timestamp", "source", "total_ms", "dataset_size"]],
        use_container_width=True,
    )

    st.subheader("Durée totale vs temps")
    st.plotly_chart(
        px.line(df.sort_values("timestamp"), x="timestamp", y="total_ms", color="source", markers=True),
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(page_title="Gold KPIs & Scoring (API Mongo)", layout="wide")
    st.title("Gold layer | KPIs & scoring 12m")

    source_mode = st.sidebar.radio("Source des données", ["API (Mongo)", "Parquet direct"], index=0)
    api_base_url = st.sidebar.text_input("API base URL", API_DEFAULT)
    st.sidebar.caption("API = source principale. Parquet direct garde le fallback et le benchmark.")

    data: Dict[str, Optional[pd.DataFrame]] = {}
    kpis: Optional[dict] = None
    active_source = source_mode
    api_error = None

    if source_mode == "API (Mongo)":
        api_data, api_kpis, api_error = load_api_data(api_base_url)
        if api_data:
            data = api_data
            kpis = api_kpis
            data["fact"] = None
        else:
            active_source = "Parquet direct"

    if active_source == "Parquet direct":
        gold = load_gold_parquet()
        data.update(gold)
        kpis = None
        data["distribution"] = build_distribution_from_fact(data.get("fact")) if gold.get("fact") is not None else None

    tab_kpi, tab_bench, tab_refresh = st.tabs(
        ["KPIs / Scoring", "Benchmark pipeline", "Benchmark refresh"]
    )

    with tab_kpi:
        if api_error and source_mode == "API (Mongo)":
            st.warning(f"API indisponible ({api_error}). Fallback Parquet.")
        required = [data.get("ca_monthly"), data.get("ca_country"), data.get("ca_product")]
        if any(df is None for df in required):
            st.warning("Données manquantes. Relancer publish gold->mongo ou vérifier MinIO.")
            st.stop()
        if data.get("scores") is None or data.get("segments") is None:
            st.warning("Le scoring client est absent. Relancer le flow gold/publish.")
            st.stop()

        render_kpis(kpis, data.get("fact"), data.get("scores"))
        st.divider()
        render_sales_charts(
            data.get("fact"),
            data["ca_monthly"],
            data["ca_country"],
            data["ca_product"],
            distribution=data.get("distribution"),
        )
        st.divider()
        render_segments(data["scores"], data["segments"])
        st.divider()
        render_opportunities(data["scores"])
        cohort_df = data.get("cohort")
        if cohort_df is not None and not cohort_df.empty:
            st.divider()
            render_cohorts(cohort_df)

    with tab_bench:
        render_pipeline_benchmark_tab()

    with tab_refresh:
        render_refresh_benchmark_tab(api_base_url)


if __name__ == "__main__":
    main()
