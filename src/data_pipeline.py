"""
Gameweek-level data pipeline for the FPL Points Predictor.

This pipeline:

1. Loads a raw per-player-per-gameweek CSV from data/raw/player_gameweeks_raw.csv
   (Kaggle Fantasy Premier League Player Data 2016-2024).

2. Keeps ONLY the seasons 2016/17 to 2022/23 (project scope).
   Seasons like 2023/24, 2024/25, ... are dropped.

3. Selects and standardises the relevant columns:
   id, name, team, position, minutes, goals_scored, assists,
   goals_conceded, clean_sheets, saves, penalties_saved, penalties_missed,
   yellow_cards, red_cards, own_goals, starts, bonus, bps,
   expected_goals, expected_assists, expected_goal_involvements,
   expected_goals_conceded, influence, creativity, threat, ict_index,
   total_points, season, gameweek

4. Renames:
   - total_points -> points    (target for GW models)
   - id -> player_id (keeps id as well, but models group by player_id)

5. Sorts the dataset by (player_id, season, gameweek).

6. Saves the final dataset to:
   - data/processed/player_gameweeks.csv
   - data/processed/player_gameweeks_lagged.csv

   The models then build lag features in model.py (no lagging here).
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# 👉 Adapte ce nom si ton fichier brut a un autre nom
RAW_GW_FILE = DATA_RAW_DIR / "player_gameweeks_raw.csv"

PLAYER_GW_FILE = DATA_PROCESSED_DIR / "player_gameweeks.csv"
PLAYER_GW_LAGGED_FILE = DATA_PROCESSED_DIR / "player_gameweeks_lagged.csv"

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Seasons kept in the project (Kaggle FPL 2016/17 → 2022/23 only)
ALLOWED_SEASONS = [
    "2016/17",
    "2017/18",
    "2018/19",
    "2019/20",
    "2020/21",
    "2021/22",
    "2022/23",
]

# Columns expected from the raw Kaggle file
GAMEWEEK_COLS = [
    "id",
    "name",
    "team",
    "position",
    "minutes",
    "goals_scored",
    "assists",
    "goals_conceded",
    "clean_sheets",
    "saves",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "starts",
    "bonus",
    "bps",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "total_points",
    "season",
    "gameweek",
]


# ---------------------------------------------------------------------
# CORE HELPERS
# ---------------------------------------------------------------------


def filter_allowed_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the seasons defined in ALLOWED_SEASONS.
    This ensures that 2023/24 and later seasons are always excluded.
    """
    if "season" not in df.columns:
        raise ValueError("Column 'season' is missing from dataframe.")
    return df[df["season"].isin(ALLOWED_SEASONS)].copy()


def load_raw_gameweeks(path: Path | str = RAW_GW_FILE) -> pd.DataFrame:
    """
    Load the raw per-player-per-gameweek dataset from Kaggle.

    The file is expected to contain at least the columns listed in GAMEWEEK_COLS.
    """
    df = pd.read_csv(path)

    # Optional sanity check: ensure required columns are present
    missing = [c for c in GAMEWEEK_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in raw gameweek file {path!r}: {missing}"
        )

    return df


def clean_gameweeks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise the gameweek-level dataset:

    - Keep only GAMEWEEK_COLS
    - Filter to ALLOWED_SEASONS (2016/17 → 2022/23)
    - Rename:
        * total_points -> points (target for GW models)
        * id -> player_id (keep id as well)
    - Ensure proper dtypes and sorting.
    """
    # Keep only the relevant columns
    df = df[GAMEWEEK_COLS].copy()

    # Filter seasons
    df = filter_allowed_seasons(df)

    # Rename target column
    df = df.rename(columns={"total_points": "points"})

    # Create player_id column (models group by player_id)
    if "player_id" not in df.columns:
        df["player_id"] = df["id"]

    # Enforce integer type for gameweek if possible
    try:
        df["gameweek"] = df["gameweek"].astype(int)
    except Exception:
        # If conversion fails, we leave as-is but it's a red flag
        pass

    # Sort chronologically by player, season, gameweek
    df = df.sort_values(["player_id", "season", "gameweek"]).reset_index(drop=True)

    # Optional: reorder columns to put player_id first (nice to read)
    # keep original id as well
    cols = ["player_id"] + [c for c in df.columns if c != "player_id"]
    df = df[cols]

    return df


# ---------------------------------------------------------------------
# PIPELINE STEPS
# ---------------------------------------------------------------------


def build_player_gameweeks(raw_path: Path | str = RAW_GW_FILE) -> Path:
    """
    Build a clean per-player-per-gameweek dataset:

    1. Load raw Kaggle CSV (player_gameweeks_raw).
    2. Clean and filter seasons (2016/17 → 2022/23).
    3. Standardise columns.
    4. Save to data/processed/player_gameweeks.csv.

    Returns
    -------
    Path
        Path to data/processed/player_gameweeks.csv
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_gameweeks(raw_path)
    df_clean = clean_gameweeks(df_raw)

    df_clean.to_csv(PLAYER_GW_FILE, index=False)
    return PLAYER_GW_FILE


def build_player_gameweeks_lagged(raw_path: Path | str = RAW_GW_FILE) -> Path:
    """
    Build the dataset used by GW-level models.

    For now, this function simply:

    - Rebuilds player_gameweeks.csv (cleaned GW data).
    - Copies it to player_gameweeks_lagged.csv.

    The lag features (points_lag_1, points_lag_2, etc.) are computed
    dynamically inside model.py (functions _add_anytime_lags,
    _add_seasonal_lags_with_prev5, prepare_gw_lag_dataset).
    """
    gw_path = build_player_gameweeks(raw_path)
    df = pd.read_csv(gw_path)

    df.to_csv(PLAYER_GW_LAGGED_FILE, index=False)
    return PLAYER_GW_LAGGED_FILE


def run_pipeline() -> Path:
    """
    Main entry point used by the CLI.

    Runs the gameweek-level pipeline and returns the path to
    player_gameweeks_lagged.csv (the file used in model.py).
    """
    return build_player_gameweeks_lagged()




# When we run this script directly, launch the SEASON pipeline and display the output file path
if __name__ == "__main__":
    output_path = run_pipeline()
    print(f"Data pipeline completed. File saved to: {output_path}")

