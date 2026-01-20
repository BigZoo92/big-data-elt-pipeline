from __future__ import annotations

from typing import List, Optional, Tuple

from pymongo import ASCENDING, DESCENDING
from pymongo.database import Database

from serving_mongo.mongo_config import get_database


class MissingDataError(RuntimeError):
    """Raised when a collection is empty or missing."""


def _parse_sort(sort_expr: Optional[str]) -> Optional[Tuple[str, int]]:
    if not sort_expr:
        return None
    if ":" in sort_expr:
        field, direction = sort_expr.split(":", 1)
    else:
        field, direction = sort_expr, "asc"
    direction_flag = DESCENDING if direction.lower() == "desc" else ASCENDING
    return field, direction_flag


def fetch_collection(
    db: Database,
    name: str,
    sort: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[dict]:
    projection = {"_id": False}
    cursor = db[name].find({}, projection=projection)
    sort_clause = _parse_sort(sort)
    if sort_clause:
        cursor = cursor.sort([sort_clause])
    if limit:
        cursor = cursor.limit(limit)
    data = list(cursor)
    if not data:
        raise MissingDataError(f"{name} vide")
    return data


def compute_kpis(db: Database) -> dict:
    agg = list(
        db["gold_fact_achats"].aggregate(
            [
                {
                    "$group": {
                        "_id": None,
                        "total_ca": {"$sum": "$montant"},
                        "total_achats": {"$sum": 1},
                        "clients": {"$addToSet": "$id_client"},
                    }
                }
            ]
        )
    )
    if not agg:
        raise MissingDataError("gold_fact_achats vide")
    base = agg[0]
    total_ca = float(base.get("total_ca", 0.0) or 0.0)
    total_achats = int(base.get("total_achats", 0) or 0)
    total_clients = len(base.get("clients", []))
    panier_moyen = total_ca / total_achats if total_achats else 0.0

    expected = list(
        db["gold_client_scores"].aggregate(
            [{"$group": {"_id": None, "expected_total": {"$sum": "$expected_value_12m"}}}]
        )
    )
    expected_total = float(expected[0]["expected_total"]) if expected else None

    return {
        "total_ca": total_ca,
        "total_achats": total_achats,
        "total_clients": total_clients,
        "panier_moyen": panier_moyen,
        "expected_total": expected_total,
        "dataset_size": total_achats,
    }


def get_db_from_app(app) -> Database:
    db = app.config.get("MONGO_DB")
    return db if db is not None else get_database()
