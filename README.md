# Anvil — Forge a Model (Streamlit prototype)

Single-file, single-user demo of Anvil's core loop: upload a CSV, pick a
target column, AutoML trains 5 algorithms and picks a winner, test
predictions live. No auth, no Postgres, no Docker required — this is the
5-minute-demo version, not the production app.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

It opens at `http://localhost:8501`. If you want to show it to someone
remotely without deploying anywhere, the same Cloudflare Tunnel approach
from the main Anvil app works here too — just point the tunnel's Public
Hostname service at `localhost:8501` instead of `5000`.

## What it does

1. **Upload & Forge** — drop in a CSV, choose the column to predict. Anvil
   auto-detects classification vs. regression and trains 5 scikit-learn
   models (Logistic/Linear Regression, Random Forest, Gradient Boosting,
   KNN, SVM).
2. **Leaderboard** — ranks all 5 by accuracy (classification) or R²
   (regression), plus a feature-importance chart for the winner.
3. **Test a Prediction** — a live form built from your dataset's columns;
   pick any trained model and get an instant prediction.
4. **Deploy** — download the winning model as a `.pkl`, plus a code
   snippet showing how to load and call it, and an illustrative sketch of
   what a production REST endpoint would look like (not a live endpoint —
   Streamlit can't serve one; the real Flask version of Anvil does).

## Notes for going from prototype to production

- Training runs synchronously in the browser session — fine for demo-sized
  CSVs, would need a background job queue for anything large.
- No team/auth system — everything lives in Streamlit's `session_state`
  for the current browser tab only; refreshing loses your trained models.
- Tabular data only (no image classification) — keeps the demo tight.
