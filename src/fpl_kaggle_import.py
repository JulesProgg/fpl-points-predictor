from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import pandas as pd
import kagglehub
import re


# Paths for your project (same logic as in data_pipeline.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = DATA_RAW_DIR / "player_gameweeks_raw.csv"

# Mapping from Kaggle-style column names → your internal schema
RENAME_MAP = {
    "element": "player_id",
    "id": "player_id",            # fallback if used
    "web_name": "name",
    "name": "name",
    "team": "team",
    "team_name": "team",
    "position": "position",
    "round": "gameweek",
    "gw": "gameweek",
    "gameweek": "gameweek",
    "minutes": "minutes",
    "goals_scored": "goals_scored",
    "assists": "assists",
    "expected_goals": "expected_goals",
    "xG": "expected_goals",
    "expected_assists": "expected_assists",
    "xA": "expected_assists",
    "total_points": "points",
    "points": "points",
    "season": "season",
    "season_name": "season",
}


def _looks_like_gw_file(sample_df: pd.DataFrame) -> bool:
    """
    Heuristic: check if a small sample of a CSV looks like
    a player×gameweek dataset.
    """
    cols = set(sample_df.columns)

    # Need something like player id
    has_player = any(c in cols for c in ["element", "player_id", "id"])
    # Need something like gameweek / round
    has_gw = any(c in cols for c in ["round", "gw", "gameweek"])
    # Need some notion of points
    has_points = any(c in cols for c in ["total_points", "points"])

    return has_player and has_gw and has_points


def build_player_gameweeks_raw_from_kaggle() -> Path:
    """
    Download the Kaggle dataset:
        'reevebarreto/fantasy-premier-league-player-data-2016-2024'
    and convert all CSVs that look like player×gameweek data into a single
    player_gameweeks_raw.csv in data/raw/.

    Each row in the final CSV corresponds to:
        one player × one season × one gameweek.
    """

    # 1) Download (or load from cache) the Kaggle dataset
    kaggle_path_str = kagglehub.dataset_download(
        "reevebarreto/fantasy-premier-league-player-data-2016-2024"
    )
    kaggle_root = Path(kaggle_path_str)
    print(f"Kaggle dataset downloaded to: {kaggle_root}")

    dataframes: List[pd.DataFrame] = []

    # 2) Find ALL CSV files under that root
    csv_files = sorted(kaggle_root.rglob("*.csv"))

    if not csv_files:
        raise RuntimeError(
            f"No CSV files found under {kaggle_root}. Check the dataset structure."
        )

    print("Found CSV files:")
    for f in csv_files:
        print("  -", f)

    for csv_path in csv_files:
        # Quick sample to decide if it looks like a GW file
        try:
            sample = pd.read_csv(csv_path, nrows=10)
        except Exception as e:
            print(f"Skipping {csv_path} (cannot read sample): {e}")
            continue

        if not _looks_like_gw_file(sample):
            print(f"Skipping {csv_path} (does not look like GW data)")
            continue

        print(f"Using {csv_path} as GW data")
        df = pd.read_csv(csv_path)

        # 3) Rename columns to your standard schema
        rename_map = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
        df = df.rename(columns=rename_map)

        # ⚠️ Remove duplicate column names (e.g. two sources mapped to 'gameweek')
        df = df.loc[:, ~df.columns.duplicated()]


        # 4) Ensure required columns exist
        required = ["player_id", "name", "team", "position", "gameweek", "points"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"Skipping {csv_path}, missing columns after rename: {missing}")
            continue

        # 5) Season column: from data if possible, else from folder name
        if "season" not in df.columns:
            # Try to infer from parent folder or filename, ex: "2018-19"
            season_guess = None
            for p in [csv_path.parent, csv_path]:
                m = re.search(r"(20\d{2}-\d{2})", str(p))
                if m:
                    season_guess = m.group(1)
                    break
            if season_guess is None:
                season_guess = "unknown"
            df["season"] = season_guess

        # 6) Keep only useful columns
        useful_cols = [
            "player_id",
            "name",
            "team",
            "position",
            "season",
            "gameweek",
            "minutes",
            "goals_scored",
            "assists",
            "expected_goals",
            "expected_assists",
            "points",
        ]
        existing_useful = [c for c in useful_cols if c in df.columns]
        df = df[existing_useful].copy()

        dataframes.append(df)

    if not dataframes:
        raise RuntimeError(
            "No CSV in the Kaggle dataset looked like GW-level player data. "
            "Check the dataset content or adjust _looks_like_gw_file/RENAME_MAP."
        )

    # 7) Concatenate everything
    full = pd.concat(dataframes, ignore_index=True)

    # Sort for clarity
    if {"season", "gameweek", "player_id"}.issubset(full.columns):
        full["gameweek"] = full["gameweek"].astype(int)
        full = full.sort_values(["season", "gameweek", "player_id"]).reset_index(
            drop=True
        )

    # 8) Save to data/raw/
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    full.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved combined GW dataset to: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    build_player_gameweeks_raw_from_kaggle()

