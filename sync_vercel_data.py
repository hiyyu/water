"""Copy saved model outputs into the static Vercel site."""
from pathlib import Path
from shutil import copy2
root = Path(__file__).resolve().parent
names = ["all_models_latest_forecasts.csv", "all_models_metrics.csv", "all_models_wqi_predictions.csv", "latest_station_observations.csv"]
target = root / "vercel-site" / "data"
target.mkdir(parents=True, exist_ok=True)
for name in names:
    source = root / "outputs" / "dashboard" / name
    if not source.is_file():
        raise FileNotFoundError(source)
    copy2(source, target / name)
    print(f"Copied {name}")
