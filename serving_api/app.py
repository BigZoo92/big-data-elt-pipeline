from __future__ import annotations

import os

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

from serving_api.routes import api_bp
from serving_mongo.mongo_config import get_mongo_client, get_database


def create_app() -> Flask:
    load_dotenv()
    app = Flask(__name__)
    app.config["MONGO_CLIENT"] = get_mongo_client()
    app.config["MONGO_DB"] = get_database()
    CORS(app)
    app.register_blueprint(api_bp)
    return app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", "5000"))
    app.run(host=host, port=port)
