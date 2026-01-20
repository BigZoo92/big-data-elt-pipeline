from __future__ import annotations

from time import perf_counter
from typing import Optional

from flask import Blueprint, current_app, jsonify, request

from serving_api.repository import (
    MissingDataError,
    compute_kpis,
    fetch_collection,
    get_db_from_app,
)

api_bp = Blueprint("api", __name__)


def _db():
    return get_db_from_app(current_app)


def _response(data, start_ts: float):
    elapsed_ms = (perf_counter() - start_ts) * 1000
    return jsonify({"data": data, "meta": {"count": len(data), "query_ms": elapsed_ms, "total_ms": elapsed_ms}})


def _fetch_and_respond(collection: str, sort: Optional[str] = None, default_limit: Optional[int] = None):
    start = perf_counter()
    limit = request.args.get("limit", type=int, default=default_limit)
    sort_expr = request.args.get("sort", default=sort)
    try:
        data = fetch_collection(_db(), collection, sort=sort_expr, limit=limit)
    except MissingDataError as exc:
        return jsonify({"error": str(exc)}), 503
    return _response(data, start)


@api_bp.route("/health")
def health():
    try:
        _db().command("ping")
        status = "ok"
    except Exception as exc:
        return jsonify({"status": "down", "error": str(exc)}), 503
    return jsonify({"status": status})


@api_bp.route("/kpis")
def kpis():
    start = perf_counter()
    try:
        data = compute_kpis(_db())
    except MissingDataError as exc:
        return jsonify({"error": str(exc)}), 503
    elapsed_ms = (perf_counter() - start) * 1000
    data["meta"] = {"query_ms": elapsed_ms, "total_ms": elapsed_ms}
    return jsonify(data)


@api_bp.route("/monthly")
def monthly():
    return _fetch_and_respond("gold_monthly", sort="mois:asc")


@api_bp.route("/by_country")
def by_country():
    return _fetch_and_respond("gold_by_country", sort="ca:desc")


@api_bp.route("/by_product")
def by_product():
    return _fetch_and_respond("gold_by_product", sort="ca:desc")


@api_bp.route("/distribution")
def distribution():
    return _fetch_and_respond("gold_distribution", sort="bucket:asc")


@api_bp.route("/daily")
def daily():
    return _fetch_and_respond("gold_daily", sort="jour:asc")


@api_bp.route("/weekly")
def weekly():
    return _fetch_and_respond("gold_weekly", sort="semaine:asc")


@api_bp.route("/monthly_growth")
def monthly_growth():
    return _fetch_and_respond("gold_monthly_growth", sort="mois:asc")


@api_bp.route("/segments")
def segments():
    return _fetch_and_respond("gold_segment_summary", sort="expected_value_12m:desc")


@api_bp.route("/scores")
def scores():
    sort_default = request.args.get("sort", default="expected_value_12m:desc")
    return _fetch_and_respond("gold_client_scores", sort=sort_default, default_limit=5000)


@api_bp.route("/cohort")
def cohort():
    return _fetch_and_respond("gold_cohort_first_purchase", sort="first_purchase_month:asc")
