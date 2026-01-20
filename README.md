# Pipeline analytique – électroménager (bronze/silver/gold)

## Décisions KEEP / REMOVE / REPLACE

### Features

- [KEEP] `id_client`, `nom`, `email`, `pays`, `date_inscription`
- [KEEP] Transactions `id_achat`, `id_client`, `date_achat`, `montant`, `produit`
- [KEEP] RFM 12 mois : `freq_12m`, `monetary_12m`, `monetary_avg_12m`, `recency_days`, `tenure_days`, `product_diversity_12m`
- [KEEP] Santé relation client : `total_orders_all`, `total_spend_all`, `avg_order_value_all`
- [REPLACE] `churn_risk_90j` → `prob_reachat_12m` (propension réaliste vs cycles longs)
- [REPLACE] `spend_pred_90d` → `expected_value_12m` + `value_at_risk_12m`
- [REMOVE] Toute métrique churn court terme (30/90j), croissance hebdo et ratios non actionnables

### Tables gold

- [KEEP] `fact_achats.parquet` (achats normalisés + mois/pays)
- [KEEP] `dim_clients.parquet` (identité + first/last purchase, recency, tenure, total_orders/spend)
- [KEEP] `client_features.parquet` (RFM 12m + tenure/diversité)
- [KEEP] `client_scores.parquet` (features + prob_reachat_12m, expected_value_12m, value_at_risk_12m, segment_label)
- [KEEP] `segment_summary.parquet` (agrégats par segment)
- [KEEP] `ca_monthly.parquet`, `ca_country.parquet`, `ca_product.parquet`
- [KEEP] `cohort_first_purchase.parquet` (cohortes simples par 1er achat)
- [REMOVE] Artefacts churn ou fichiers de croissance court terme

## Objectifs prédictifs retenus (max 2)

1. Propension de ré-achat à 12 mois (`prob_reachat_12m`) via heuristique RFM (fréquence + récence + valeur + diversité produit).
2. Valeur attendue 12 mois (`expected_value_12m`) et valeur à risque (`value_at_risk_12m`) pour prioriser réactivation/upsell.

## Pipeline à lancer (depuis la racine)

1. Démarrer MinIO  
   `docker-compose up -d minio minio-setup`
2. (Optionnel) Générer des données synthétiques  
   `python scripts/generate_data.py`
3. Ingestion & nettoyage
   - Bronze : `python flows/bronze_ingestion.py`
   - Silver : `python flows/silver_transformation.py`
4. Gold (features + scoring + agrégats)  
   `python flows/gold_transformation.py`
5. **Spark (nouvelle, en parallèle)**  
   `python flows_spark/bronze_ingestion_spark.py`  
   `python flows_spark/silver_transformation_spark.py`  
   `python flows_spark/gold_transformation_spark.py`
6. Before benchmark
   `$py = (Get-Command python).Source`
   `$py`
   `$env:PYSPARK_PYTHON = $py`
   `$env:PYSPARK_DRIVER_PYTHON = $py`

7. Benchmark
   `python -m script.benchmark`
8. Dashboard Streamlit  
   `streamlit run scripts/dashboard.py`

## Gold attendu dans MinIO/gold

- `fact_achats.parquet` : id_achat, id_client, date_achat, montant, produit, pays, mois
- `dim_clients.parquet` : identité + first/last purchase, recency_days, tenure_days, total_orders, total_spend, avg_order_value
- `client_features.parquet` : RFM 12m + product_diversity_12m + totals
- `client_scores.parquet` : features + prob_reachat_12m, expected_value_12m, value_at_risk_12m, segment_label
- `segment_summary.parquet` : expected_value_12m, value_at_risk_12m, freq_med, recency_med par segment
- `ca_monthly.parquet`, `ca_country.parquet`, `ca_product.parquet` (CA agrégé)
- `cohort_first_purchase.parquet` : nouveaux clients et CA par mois de premier achat

## Notes métier (électroménager)

- Cycles d’achat longs et one-shot fréquents → le churn 30/90j n’est pas pertinent.
- La valeur vient des réachats rares mais élevés : on suit prob_reachat_12m + valeur attendue pour prioriser les relances.
- Segmentation RFM simple (VIP, Actifs, A relancer, Dormants, A potentiel) basée sur récence/fréquence/valeur.

## Validation légère

- `python scripts/check_gold.py` liste les fichiers gold et valide colonnes + montants non négatifs + dates parsées.
- Assertions dans les flows : montants > 0 et < 10k, colonnes obligatoires, présence de dates achetées.

## Références rapides

- HORIZON scoring : 365 jours (12 mois)
- Buckets MinIO : sources → bronze → silver → gold
- Parquet pour toutes les tables silver/gold
- RFM : recency = jours depuis le dernier achat, frequency = nb d’achats, monetary = CA cumulé/12m
- `prob_reachat_12m` = score heuristique RFM normalisé ; `expected_value_12m` = valeur mensuelle moyenne x 12 x probabilité

## Spark (optionnel) et benchmark

- Cluster Spark : `docker-compose -f docker-compose.spark.yml up -d spark-master spark-worker-1 spark-worker-2`
- Pipeline Spark parallèle (ne remplace pas Pandas) : `python flows_spark/bronze_ingestion_spark.py`, puis silver/gold équivalents.
- Benchmark Pandas vs Spark : `python scripts/benchmark.py --scale 1` → résultats dans `data/metrics/benchmark.json` (onglet Benchmark du dashboard).
## Serving Mongo + API + Metabase

- Compose d�di� : `docker-compose -f docker-compose.yml -f docker-compose.serving.yml up -d minio minio-setup mongodb flask-api metabase` (optionnel: `mongo-express`). Tout tourne sur le network `elt-network` (le compose principal le cr�e).
- Publication gold -> Mongo (idempotent) : `python serving_mongo/publish_gold_to_mongo.py` (lit MinIO en priorit�, fallback local `data/spark_lake/gold` si pr�sent).
- Mapping Parquet -> Collections -> API :

| Parquet gold | Collection Mongo | Endpoint API |
| --- | --- | --- |
| fact_achats.parquet | gold_fact_achats | /kpis (agg), /daily, /weekly, /distribution |
| dim_clients.parquet | gold_dim_clients | (Metabase) |
| client_features.parquet | gold_client_features | (Metabase) |
| client_scores.parquet | gold_client_scores | /kpis (expected), /scores |
| segment_summary.parquet | gold_segment_summary | /segments |
| ca_monthly.parquet | gold_monthly | /monthly, /monthly_growth |
| ca_country.parquet | gold_by_country | /by_country |
| ca_product.parquet | gold_by_product | /by_product |
| cohort_first_purchase.parquet | gold_cohort_first_purchase | /cohort |
| d�riv�s (daily/weekly/distribution/monthly_growth) | gold_daily, gold_weekly, gold_distribution, gold_monthly_growth | /daily, /weekly, /distribution, /monthly_growth |

- API Flask (port 5000) : `/health`, `/kpis`, `/monthly`, `/by_country`, `/by_product`, `/distribution`, `/daily`, `/weekly`, `/monthly_growth`, `/segments`, `/scores`, `/cohort`. R�ponses JSON simples avec `meta.query_ms` et 503 si collection vide.
- Dashboard Streamlit : mode par d�faut "API (Mongo)" avec fallback "Parquet direct". Nouvel onglet "Benchmark refresh" (Parquet direct vs API+Mongo) qui alimente `data/metrics/refresh_benchmark.json`.
- Metabase : http://localhost:3000 (volume persistant). Connexion Mongo : host `mongodb`, port `27017`, database `MONGO_DB` (d�faut `gold_serving`), user/pass vides (selon votre config). Collections utiles : gold_fact_achats, gold_monthly, gold_by_country, gold_by_product, gold_client_scores, gold_segment_summary.
