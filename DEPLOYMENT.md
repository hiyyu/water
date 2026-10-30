# Deploy Water Quality Dashboard — GitHub + Render + Vercel

This repository supports **two independent deployment targets** sharing the same saved DOE/model outputs.

## A. GitHub

1. Extract this archive into your existing local `water` repository (merge folders; don't delete your own `.git` directory).
2. In VS Code terminal from the `water` root:

   ```powershell
   python sync_vercel_data.py
   git add -A
   git commit -m "Prepare Render and Vercel deployments"
   git push origin main
   ```

3. Confirm `render.yaml`, `app.py`, `vercel-site/index.html`, and `vercel-site/data/*.csv` are visible on GitHub.

## B. Render — full original Streamlit application

1. On [render.com](https://render.com), choose **New → Blueprint** and connect the GitHub repository `hiyyu/water`.
2. Render will read the root `render.yaml` and create a Python web service.
3. The build command installs only the dashboard dependencies (`requirements-deploy.txt`), **not** TensorFlow or XGBoost training packages.
4. The start command binds Streamlit to `0.0.0.0:$PORT`.
5. Render provides a `.onrender.com` URL. The free instance may sleep when idle.

If using **New → Web Service** instead of Blueprint, set Runtime `Python`, Build Command `pip install -r requirements-deploy.txt`, Start Command `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`, Python version `3.12.8`.

## C. Vercel — static companion dashboard

**Important:** Vercel does not directly host this repository's Streamlit `app.py`. Deploy only the **`vercel-site`** folder as a separate Vercel project.

1. In Vercel, choose **Add New → Project**, import `hiyyu/water`.
2. Set **Root Directory** to `vercel-site` (not the repository root).
3. Framework Preset: **Other**. Build Command: leave blank/override off. Output Directory: leave blank (static files are served from the selected root). Do not choose Python.
4. Deploy. The site loads `data/*.csv` already committed in `vercel-site/data`.
5. If an existing Vercel project is still throwing `PYTHON_ENTRYPOINT_NOT_FOUND`, change its Root Directory to `vercel-site`, or create a new Vercel project with that root. Redeploy the latest commit.

The Vercel site is a lightweight **read-only** companion with saved model results, a forecast overview, and evaluation metrics. The Render site preserves the full Streamlit interactions and map views.

## D. Update predictions later

After running your local Python training/data-preparation scripts, run:

```powershell
python sync_vercel_data.py
git add -A
git commit -m "Update model dashboard data"
git push origin main
```

Both services redeploy on GitHub updates. The Vercel frontend never trains models or calls TensorFlow.

## Scientific limitations

DOE observations currently end in 2024 and the sampling cadence is approximately every 2 months. The 7-day Vercel panel is explicitly labelled as an **experimental carry-forward illustration**, not genuine daily predictions or live readings. Dates in the dashboard header use the browser's current date, not a newly collected water sample.
