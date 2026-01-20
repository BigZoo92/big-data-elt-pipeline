# Advisor (segmentation RFM)

Script : `scripts/train_advisor.py`

## Lancer
```
python scripts/train_advisor.py
```

Options utiles :
- `--clients-path` : chemin vers le fichier clients (CSV/Parquet). Par défaut détection (data/silver, data/gold, sinon data/sources/clients.csv).
- `--achats-path` : chemin vers le fichier achats (CSV/Parquet). Par défaut détection (data/silver, data/gold, sinon data/sources/achats.csv).
- `--n-clusters` : nombre de clusters KMeans (défaut 5).
- `--experiment` : nom d'expérience MLflow (défaut `advisor`).

## Sorties
Générées dans `data/advisor/<TIMESTAMP>_<RUNID>/` :
- `clients_scored.csv` : RFM + segment + label business.
- `segment_summary.csv` : agrégats par segment.
- `advice.md` : actions business, top clients à relancer / VIP.
- `plots/segment_revenue.png` : CA par segment.

## MLflow
```
mlflow ui --backend-store-uri ./mlruns
```

Le script logge params, métriques, artefacts et le modèle KMeans (pipeline sklearn).***
