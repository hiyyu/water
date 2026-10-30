# Vercel static dashboard

Deploy this folder as the **Root Directory** (`vercel-site`) in Vercel. Framework preset: **Other**; Build Command: **none**; Output Directory: **.** (the root of this directory). This is a static site and must not be configured as a Python Function.

The CSV files under `data/` are copies of saved model outputs. After retraining, run `python sync_vercel_data.py` from the repository root and commit the changed CSVs.

This frontend shows all saved models, station-level next-sampling predictions, model evaluation, and a clearly labelled experimental 7-day carry-forward visualization. It does **not** run Python models on Vercel and does **not** claim daily WQI forecast validity.
