#!/usr/bin/env python3
"""Fetch DOE continuous river water quality (WQI) data for the dashboard's Live tab.

Keeps only the states listed in WANTED_STATES (Johor, Kedah, Kelantan, Penang, Selangor).

Source: the public JSON endpoints behind https://eqms.doe.gov.my/RQIMS/conti_river
  - mapcrwqm                     latest WQI for every continuous station (national)
  - datatrendchartcrwqm?stateid  roughly the last 7 days of hourly WQI for one state

Writes into vercel-site/data/ (override with the LIVE_DATA_DIR environment variable):
  live_latest.csv   current snapshot, overwritten every run
  live_history.csv  hourly history, MERGED with the existing file (nothing is lost)
  live_meta.json    when the data was fetched and the newest reading time

Safety: if the main request fails or returns no usable rows, the script exits with
an error and does not touch any file, so a bad run can never wipe good data.

Run:  python scripts/fetch_live_wqi.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "https://eqms.doe.gov.my/api3/publicportalrqims"
# Only these states are kept. DOE writes Penang as "P.Pinang"; the aliases cover other spellings.
# To add a state later, add its name here (lower case, no spaces).
WANTED_STATES = {"johor", "kedah", "kelantan", "ppinang", "pulaupinang", "penang", "selangor"}
HEADERS = {
    "User-Agent": "water-quality-fyp-dashboard/1.0 (university final-year project; hourly polling)",
    "Accept": "application/json",
}
TIMEOUT = 20      # seconds per request
RETRIES = 3
PAUSE = 1.0       # seconds between state requests, to stay polite

LATEST_COLS = ["station_id", "river", "location", "state", "datetime", "wqi", "status", "lat", "lon"]  # state_id is internal
HISTORY_COLS = ["station_id", "datetime", "wqi"]


def state_key(name) -> str:
    """'P.Pinang' -> 'ppinang', 'Pulau Pinang' -> 'pulaupinang' (for matching WANTED_STATES)."""
    return "".join(ch for ch in clean_text(name).lower() if ch.isalnum())


def default_data_dir() -> Path:
    env = os.environ.get("LIVE_DATA_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[1] / "vercel-site" / "data"


# ---------------------------------------------------------------- fetching
def get_json(path: str, session: requests.Session):
    url = f"{BASE_URL}/{path}"
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            response = session.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            print(f"  attempt {attempt}/{RETRIES} failed for {path}: {error}", file=sys.stderr)
            if attempt < RETRIES:
                time.sleep(2 * attempt)
    raise RuntimeError(f"{url} failed after {RETRIES} attempts: {last_error}")


# ----------------------------------------------------------------- parsing
def clean_text(value) -> str:
    """Trim and collapse whitespace ('Sg. Moyog ' -> 'Sg. Moyog')."""
    return " ".join(str(value).split()) if value is not None else ""


def to_wqi(value):
    """WQI arrives as a number in one endpoint and a string in another. Returns None if unusable."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number < 0 or number > 100:
        return None
    return int(number) if number.is_integer() else round(number, 1)


def status_from(value, wqi) -> str:
    text = clean_text(value).title()  # 'SLIGHTLY POLLUTED' -> 'Slightly Polluted'
    if text:
        return text
    return "Clean" if wqi >= 81 else "Slightly Polluted" if wqi >= 60 else "Polluted"


def as_rows(payload) -> list:
    """Accept a bare list, or a dict that wraps one list (e.g. {'highcharts_bar': [...]})."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        for value in payload.values():
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
    return []


def parse_latest(payload) -> list[dict]:
    rows = []
    for r in as_rows(payload):
        station = clean_text(r.get("STATION_ID"))
        stamp = clean_text(r.get("DATETIME"))
        wqi = to_wqi(r.get("WQI"))
        if not station or not stamp or wqi is None:
            continue  # e.g. the blank row with no station / null WQI
        rows.append({
            "station_id": station,
            "river": clean_text(r.get("RIVER_NAME")),
            "location": clean_text(r.get("LOCATION")),
            "state": clean_text(r.get("STATE_NAME")),
            "state_id": clean_text(r.get("STATE_ID")),
            "datetime": stamp,
            "wqi": wqi,
            "status": status_from(r.get("WQI_RANGE"), wqi),
            "lat": r.get("LATITUDE"),
            "lon": r.get("LONGITUDE"),
        })
    rows.sort(key=lambda x: (x["state"], x["river"], x["station_id"]))
    return rows


def parse_trend(payload) -> list[dict]:
    rows = []
    for r in as_rows(payload):
        station = clean_text(r.get("STATION_ID"))
        stamp = clean_text(r.get("DATETIME"))
        wqi = to_wqi(r.get("WQI"))
        if station and stamp and wqi is not None:
            rows.append({"station_id": station, "datetime": stamp, "wqi": wqi})
    return rows


# ----------------------------------------------------------------- storing
def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    """Write via a temp file so an interrupted run can't leave a half-written CSV."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def merge_history(path: Path, new_rows: list[dict], drop_stations=frozenset()) -> list[dict]:
    """Old rows + new rows, one row per (station_id, datetime); newest value wins.
    Old rows of stations in `drop_stations` (states no longer wanted) are removed."""
    merged = {}
    old_rows = [r for r in read_csv(path) if r["station_id"] not in drop_stations]
    for r in old_rows + new_rows:
        merged[(r["station_id"], r["datetime"])] = {
            "station_id": r["station_id"], "datetime": r["datetime"], "wqi": r["wqi"],
        }
    return [merged[k] for k in sorted(merged)]  # sorted by station, then time


def main() -> int:
    data_dir = default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    print("Fetching latest readings ...")
    try:
        latest = parse_latest(get_json("mapcrwqm", session))
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if not latest:
        print("ERROR: no usable rows in mapcrwqm; leaving existing files untouched.", file=sys.stderr)
        return 1

    # Keep only the wanted states (the national feed includes all of Malaysia).
    excluded = {r["station_id"] for r in latest if state_key(r["state"]) not in WANTED_STATES}
    latest = [r for r in latest if state_key(r["state"]) in WANTED_STATES]
    if not latest:
        print("ERROR: none of the wanted states appear in mapcrwqm; leaving existing files untouched.", file=sys.stderr)
        return 1
    state_ids = sorted({r["state_id"] for r in latest if r["state_id"]}, key=lambda x: (len(x), x))
    print(f"Keeping {len(latest)} stations in {len({r['state'] for r in latest})} states (state ids {', '.join(state_ids)}).")

    # The latest snapshot is also a history point (duplicates are removed on merge).
    new_history = [{k: r[k] for k in HISTORY_COLS} for r in latest]

    states_ok = 0
    for state_id in state_ids:
        try:
            rows = [x for x in parse_trend(get_json(f"datatrendchartcrwqm?stateid={state_id}", session))
                    if x["station_id"] not in excluded]
        except RuntimeError as error:
            print(f"  warning: state {state_id} skipped: {error}", file=sys.stderr)
            continue
        if rows:
            states_ok += 1
            new_history.extend(rows)
            print(f"  state {state_id}: {len(rows)} hourly rows")
        time.sleep(PAUSE)

    history = merge_history(data_dir / "live_history.csv", new_history, excluded)

    # All parsing and merging succeeded -> now it is safe to write.
    write_csv(data_dir / "live_latest.csv", LATEST_COLS, latest)
    write_csv(data_dir / "live_history.csv", HISTORY_COLS, history)
    meta = {
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "newest_reading": max(r["datetime"] for r in latest),
        "stations": len(latest),
        "history_rows": len(history),
        "states_with_trend": states_ok,
    }
    (data_dir / "live_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Done: {len(latest)} stations, {len(history)} history rows, newest reading {meta['newest_reading']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
