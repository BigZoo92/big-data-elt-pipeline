# Pipeline analytique – Électroménager (bronze / silver / gold)

Ce projet met en place un mini pipeline ELT sur des données d’achats d’électroménager, avec :
- stockage en “lake” (bronze/silver/gold) en Parquet,
- une version Pandas + une version Spark (pour comparer),
- un “serving” MongoDB + API Flask,
- un dashboard Streamlit (KPIs + benchmarks),
- Metabase pour explorer les collections,
- MLflow pour suivre les runs d’entraînement (advisor / predictor).

---
## Données et features
<img width="1907" height="862" alt="Capture d&#39;écran 2026-01-20 161155" src="https://github.com/user-attachments/assets/7b45131e-92e5-4cff-90a2-bbccb678bdba" />
<img width="1904" height="1061" alt="fb7f9cf2-1ae1-4af4-b5da-ff3a1585c863" src="https://github.com/user-attachments/assets/985b8475-b3cb-43ab-aafd-18e830a0f6f4" />
<img width="1918" height="867" alt="Capture d&#39;écran 2026-01-20 161535" src="https://github.com/user-attachments/assets/1c2309b9-3727-4740-b9e8-836255a5405a" />
<img width="1906" height="857" alt="Capture d&#39;écran 2026-01-20 161549" src="https://github.com/user-attachments/assets/c858c98a-7967-46a9-8599-416183899a1e" />
<img width="1918" height="866" alt="Capture d&#39;écran 2026-01-20 161727" src="https://github.com/user-attachments/assets/664d74cb-b806-4d2e-bc08-77ebda0762d8" />

## Données et features

### Données sources (keep)
- Clients : `id_client`, `nom`, `email`, `pays`, `date_inscription`
- Achats : `id_achat`, `id_client`, `date_achat`, `montant`, `produit`

### Features (keep)
- RFM 12 mois : `freq_12m`, `monetary_12m`, `monetary_avg_12m`, `recency_days`, `tenure_days`, `product_diversity_12m`
- Santé relation client : `total_orders_all`, `total_spend_all`, `avg_order_value_all`

### Changement (replace)
- `churn_risk_90j` → `prob_reachat_12m` (plus cohérent sur des cycles longs)
- `spend_pred_90d` → `expected_value_12m` + `value_at_risk_12m`

### Suppressions (remove)
- métriques churn court terme (30/90j) + artefacts de “croissance hebdo” non actionnables

---

## Tables Gold (outputs attendus)

- `fact_achats.parquet` : achats normalisés + `pays`, `mois`
- `dim_clients.parquet` : identité + first/last purchase, recency, tenure, totals
- `client_features.parquet` : RFM 12m + diversité + totaux
- `client_scores.parquet` : features + `prob_reachat_12m`, `expected_value_12m`, `value_at_risk_12m`, `segment_label`
- `segment_summary.parquet` : agrégats par segment
- `ca_monthly.parquet`, `ca_country.parquet`, `ca_product.parquet`
- `cohort_first_purchase.parquet` : cohortes simples par mois de 1er achat

---

## Objectifs prédictifs (max 2)

1. **Propension de ré-achat à 12 mois** (`prob_reachat_12m`) via heuristique RFM.
2. **Valeur attendue à 12 mois** (`expected_value_12m`) et **valeur à risque** (`value_at_risk_12m`) pour prioriser relance/upsell.

---

## Lancer le pipeline (manuel, depuis la racine)

### 0) MinIO (buckets bronze/silver/gold + sources)
```bash
docker compose up -d minio minio-setup
````

### 1) Génération de données (optionnel)

```bash
python scripts/generate_data.py
```

### 2) Pipeline Pandas

```bash
python flows/bronze_ingestion.py
python flows/silver_transformation.py
python flows/gold_transformation.py
```

### 3) Pipeline Spark (en parallèle)

```bash
python flows_spark/bronze_ingestion_spark.py
python flows_spark/silver_transformation_spark.py
python flows_spark/gold_transformation_spark.py
```

### 4) Benchmark Pandas vs Spark

```bash
python scripts/benchmark.py --scale 1
```

Résultats : `data/metrics/benchmark.json` (visible aussi dans le dashboard Streamlit).

### 5) Dashboard Streamlit

```bash
streamlit run scripts/dashboard.py
```

---

## Serving Mongo + API + Metabase

### Démarrage (stack serving)

```bash
docker compose -f docker-compose.yml -f docker-compose.serving.yml up -d \
  minio minio-setup mongodb flask-api metabase
```

### Publication Gold → Mongo (idempotent)

```bash
python serving_mongo/publish_gold_to_mongo.py
```

Le script lit MinIO en priorité, avec fallback local si besoin.

### Endpoints API Flask (port 5000)

* `/health`
* `/kpis`
* `/monthly`, `/monthly_growth`
* `/by_country`, `/by_product`
* `/distribution`, `/daily`, `/weekly`
* `/segments`, `/scores`, `/cohort`

Les réponses sont en JSON avec `meta.query_ms`. Si une collection est vide : `503`.

---

## Mapping Gold → Mongo → API

| Parquet gold                                       | Collection Mongo                                                | Endpoint API                                    |
| -------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------- |
| fact_achats.parquet                                | gold_fact_achats                                                | /kpis (agg), /daily, /weekly, /distribution     |
| dim_clients.parquet                                | gold_dim_clients                                                | (Metabase)                                      |
| client_features.parquet                            | gold_client_features                                            | (Metabase)                                      |
| client_scores.parquet                              | gold_client_scores                                              | /kpis (expected), /scores                       |
| segment_summary.parquet                            | gold_segment_summary                                            | /segments                                       |
| ca_monthly.parquet                                 | gold_monthly                                                    | /monthly, /monthly_growth                       |
| ca_country.parquet                                 | gold_by_country                                                 | /by_country                                     |
| ca_product.parquet                                 | gold_by_product                                                 | /by_product                                     |
| cohort_first_purchase.parquet                      | gold_cohort_first_purchase                                      | /cohort                                         |
| dérivés (daily/weekly/distribution/monthly_growth) | gold_daily, gold_weekly, gold_distribution, gold_monthly_growth | /daily, /weekly, /distribution, /monthly_growth |

---

## Metabase

* URL : [http://localhost:3000](http://localhost:3000)
* Connexion Mongo :

  * **Hôte** : `mongodb` (si Metabase est dans Docker)
  * **Port** : `27017`
  * **Database** : `gold_serving` (ou `MONGO_DB` si configuré)
  * **User/Password** : vides (selon config)

Collections utiles : `gold_fact_achats`, `gold_monthly`, `gold_by_country`, `gold_by_product`, `gold_client_scores`, `gold_segment_summary`.

---

## Notes “métier” (électroménager)

* Cycles d’achat longs + achats one-shot fréquents → churn 30/90j peu pertinent.
* On suit plutôt `prob_reachat_12m` + valeur attendue pour prioriser les relances.
* Segmentation RFM simple (VIP / Actifs / À relancer / Dormants / À potentiel).

---

## Validation légère

```bash
python scripts/check_gold.py
```

Le script liste les fichiers gold et vérifie : colonnes attendues, dates parsées, montants non négatifs.

---

## Spark & benchmark (rappel)

* La version Spark ne remplace pas Pandas : elle sert surtout à comparer.
* Les résultats du benchmark sont affichés dans l’onglet “Benchmark” du dashboard.

---

## Difficultés rencontrées (résumé)

* Spark/Hadoop sous Windows : setup natif fragile → passage à Docker pour stabiliser.
* Mismatch de versions Python driver/worker avec PySpark → alignement des versions.
* Problèmes de réseau Docker (localhost vs nom de service) → correction via `MONGO_URI` côté API.
* Conflit de ports (API 5000 / MLflow) → MLflow déplacé en **5001**.
