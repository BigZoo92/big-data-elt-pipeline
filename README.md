## Screen résultat

<img width="1907" height="862" alt="Capture d&#39;écran 2026-01-20 161155" src="https://github.com/user-attachments/assets/7b45131e-92e5-4cff-90a2-bbccb678bdba" />
<img width="1904" height="1061" alt="fb7f9cf2-1ae1-4af4-b5da-ff3a1585c863" src="https://github.com/user-attachments/assets/985b8475-b3cb-43ab-aafd-18e830a0f6f4" />
<img width="1918" height="867" alt="Capture d&#39;écran 2026-01-20 161535" src="https://github.com/user-attachments/assets/1c2309b9-3727-4740-b9e8-836255a5405a" />
<img width="1906" height="857" alt="Capture d&#39;écran 2026-01-20 161549" src="https://github.com/user-attachments/assets/c858c98a-7967-46a9-8599-416183899a1e" />
<img width="1918" height="866" alt="Capture d&#39;écran 2026-01-20 161727" src="https://github.com/user-attachments/assets/664d74cb-b806-4d2e-bc08-77ebda0762d8" />

## Résumé rapide

- Pipeline ELT bronze/silver/gold en Pandas + variante Spark (Parquet MinIO).
- Scoring clients et agrégats CA avec publication dans MongoDB (serving layer).
- API Flask (5000) expose les collections Mongo pour le dashboard Streamlit (8501).
- Metabase (3000) branchée sur Mongo pour les dashboards libres.
- MLflow UI sur 5001 pour suivre les runs d'entraînement (advisor/predictor).
- Benchmarks pipeline (Pandas vs Spark) + refresh (Parquet vs API) stockés dans data/metrics.

## Architecture (services Docker)

- MinIO + minio-setup (buckets bronze/silver/gold/sources).
- Prefect server + Postgres (pour orchestration existante).
- Spark master/workers (optionnel) + spark-client.
- MongoDB (mongo-serving) + Flask API (gold-flask-api) + option mongo-express.
- Streamlit dashboard + MLflow UI + runner (python) pour pipelines/benchmarks.
- Metabase pour BI.

## Quickstart (commande maître)

- `python -m tools.run up`
- URLs attendues : Streamlit http://localhost:8501, API http://localhost:5000/health, Metabase http://localhost:3000, MLflow http://localhost:5001, MinIO http://localhost:9001.
- Idempotent : la commande crée .env si absent, (re)build les images, lance les pipelines, publie dans Mongo et lance les services.

## Commandes manuelles

- Tout lancer : `docker compose -f docker-compose.yml -f docker-compose.serving.yml -f docker-compose.dev.yml up -d --build`
- Génération + pipelines + publish (runner) :
  - `docker compose -f docker-compose.yml -f docker-compose.serving.yml -f docker-compose.dev.yml run --rm runner python scripts/generate_data.py`
  - `docker compose ... run --rm runner python -m flows.bronze_ingestion`
  - `docker compose ... run --rm runner python -m flows.silver_transformation`
  - `docker compose ... run --rm runner python -m flows.gold_transformation`
  - `docker compose ... run --rm runner python -m flows_spark.bronze_ingestion_spark`
  - `docker compose ... run --rm runner python -m flows_spark.silver_transformation_spark`
  - `docker compose ... run --rm runner python -m flows_spark.gold_transformation_spark`
  - `docker compose ... run --rm runner python -m serving_mongo.publish_gold_to_mongo`
  - `docker compose ... run --rm runner python scripts/benchmark.py`
- Arrêt : `python -m tools.run down` (ou `reset` pour supprimer les volumes).
- Logs : `python -m tools.run logs` ; statut : `python -m tools.run status`.

## ML & entraînement

- Scripts : `python scripts/train_predictor.py` et `python scripts/train_advisor.py` (via runner : `docker compose ... run --rm runner python scripts/train_predictor.py`).
- Les runs/artefacts sont stockés dans `./mlruns` (monté dans le conteneur MLflow). UI : http://localhost:5001.

## Benchmarks

- Pipeline Pandas/Spark : `python scripts/benchmark.py` écrit `data/metrics/benchmark.json` (onglet Benchmark dans Streamlit).
- Refresh API vs Parquet : onglet "Benchmark refresh" dans Streamlit écrit `data/metrics/refresh_benchmark.json`.

## API Flask (port 5000)

- `/health`, `/kpis`, `/monthly`, `/by_country`, `/by_product`, `/distribution`, `/daily`, `/weekly`, `/monthly_growth`, `/segments`, `/scores`, `/cohort`.
- Réponses JSON simples avec `meta.query_ms` et 503 si collection vide.

## Notes gold -> Mongo -> API

- Parquet -> collections : fact_achats -> gold_fact_achats, dim_clients -> gold_dim_clients, client_features -> gold_client_features, client_scores -> gold_client_scores, segment_summary -> gold_segment_summary, ca_monthly -> gold_monthly (+ gold_monthly_growth), ca_country -> gold_by_country, ca_product -> gold_by_product, cohort_first_purchase -> gold_cohort_first_purchase, dérivés -> gold_daily/gold_weekly/gold_distribution.
- Idempotence : upsert sur fact/dim/features/scores, replace sur agrégats.

## Difficultés rencontrées

- Spark/Hadoop sous Windows (winutils/NativeIO) : contourné via images Docker et OpenJDK dans l'image Python.
- Alignement versions PySpark/driver : image unique construite avec requirements (pyspark 3.5).
- Conflit de ports (API 5000 vs MLflow) : MLflow déplacé sur 5001.
- Résolution des hostnames Docker : l'API/Mongo utilisent les noms de services (`mongo-serving`, `flask-api`, `minio`).
