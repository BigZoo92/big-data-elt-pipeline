import os
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


def _build_uri() -> str:
    return os.getenv("MONGO_URI", "mongodb://localhost:27017")


def _db_name() -> str:
    return os.getenv("MONGO_DB", "gold_serving")


@lru_cache(maxsize=1)
def get_mongo_client() -> MongoClient:
    uri = _build_uri()
    return MongoClient(uri, uuidRepresentation="standard")


def get_database():
    client = get_mongo_client()
    return client[_db_name()]
