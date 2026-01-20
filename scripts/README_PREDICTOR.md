# Predictor (propension/valeur 12 mois)

Heuristique RFM (pas de churn 90j, pas de modèle lourd) pour scorer la probabilité de ré-achat à 12 mois et la valeur attendue associée.

## Prérequis

```
pip install -r requirements.txt
```

## Scoring

```
python scripts/train_predictor.py
```

Options utiles :

- `--clients-path` : chemin vers clients (détection auto data/silver -> data/gold -> data/sources)
- `--achats-path` : chemin vers achats (même logique)
- `--tracking-uri` : MLflow tracking URI (défaut file:./mlruns)
- `--experiment-name` : nom d'expérience MLflow (défaut predictor)
- `--horizon-days` : horizon en jours pour le scoring (défaut 365)

Sorties : `data/advisor/<timestamp>_<runid>/` avec `predictions.csv` (client_scores), `segment_summary.csv`, `metrics.json`, `report.md`, `plots/`, artefacts loggés dans MLflow.

## Dashboard Streamlit

```
streamlit run scripts/dashboard.py
```

## MLflow UI

```
mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port $env:MLFLOW_PORT
```
