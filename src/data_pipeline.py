"""
Gameweek-level data pipeline for the FPL Points Predictor.

This pipeline:

1. Loads a raw per-player-per-gameweek CSV from data/raw/player_gameweeks_raw.csv
   (built from the Kaggle dataset "Fantasy Premier League Player Data 2016-2024").

2. Keeps ONLY the seasons 2016/17 to 2022/23 (project scope).
   Seasons like 2023/24, 2024/25, ... are dropped.

3. Selects and standardises the relevant columns:
   id, name, team, opponent (optional), was_home (optional), position, minutes,
   goals_scored, assists, goals_conceded, clean_sheets, saves, penalties_saved,
   penalties_missed, yellow_cards, red_cards, own_goals, starts, bonus, bps,
   expected_goals, expected_assists, expected_goal_involvements,
   expected_goals_conceded, influence, creativity, threat, ict_index,
   total_points, season, gameweek

4. Renames:
   - total_points -> points    (target for GW models)
   - id -> player_id           (keeps id as well)
   - opponent -> opponent_team (if present)

5. Sorts the dataset by (player_id, season, gameweek).

6. Saves the final dataset to:
   - data/processed/player_gameweeks.csv

   Lag features are NOT built here; they are constructed dynamically
   in model.py for GW-level models.
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_GW_FILE = DATA_RAW_DIR / "player_gameweeks_raw.csv"
PLAYER_GW_FILE = DATA_PROCESSED_DIR / "player_gameweeks.csv"

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

ALLOWED_SEASONS = [
    "2016/17",
    "2017/18",
    "2018/19",
    "2019/20",
    "2020/21",
    "2021/22",
    "2022/23",
]

# Colonnes obligatoires
GAMEWEEK_BASE_COLS = [
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

# Colonnes optionnelles (si présentes dans le brut, on les garde)
OPTIONAL_COLS = ["opponent", "was_home"]


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------


def filter_allowed_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only seasons in ALLOWED_SEASONS."""
    if "season" not in df.columns:
        raise ValueError("Column 'season' is missing from dataframe.")
    return df[df["season"].isin(ALLOWED_SEASONS)].copy()


def load_raw_gameweeks(path: Path | str = RAW_GW_FILE) -> pd.DataFrame:
    """
    Load the raw per-player-per-gameweek dataset.

    Required columns: GAMEWEEK_BASE_COLS
    Optional columns: OPTIONAL_COLS
    """
    df = pd.read_csv(path)

    missing_required = [c for c in GAMEWEEK_BASE_COLS if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Missing REQUIRED columns in raw gameweek file {path!r}: {missing_required}"
        )

    return df


def clean_gameweeks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise the gameweek-level dataset:

    - Keep required columns + optional ones (if present)
    - Filter to ALLOWED_SEASONS (2016/17 → 2022/23)
    - Rename:
        * total_points -> points
        * opponent -> opponent_team (if present)
    - Create player_id from id
    - Sort by player_id, season, gameweek
    """
    # Colonnes disponibles = obligatoires + optionnelles présentes
    available_optional = [c for c in OPTIONAL_COLS if c in df.columns]
    cols_to_keep = GAMEWEEK_BASE_COLS + available_optional

    df = df[cols_to_keep].copy()

    # Filtrer les saisons
    df = filter_allowed_seasons(df)

    # Renommer la cible
    df = df.rename(columns={"total_points": "points"})

    # Renommer opponent -> opponent_team si présent
    if "opponent" in df.columns:
        df = df.rename(columns={"opponent": "opponent_team"})

    # Ajouter player_id
    if "player_id" not in df.columns:
        df["player_id"] = df["id"]

    # gameweek en int si possible
    try:
        df["gameweek"] = df["gameweek"].astype(int)
    except Exception:
        pass

    # Trier chronologiquement par joueur
    df = df.sort_values(["player_id", "season", "gameweek"]).reset_index(drop=True)

    # Mettre player_id en première colonne pour la lisibilité
    cols = ["player_id"] + [c for c in df.columns if c != "player_id"]
    df = df[cols]

    return df


# ---------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------


def build_player_gameweeks(raw_path: Path | str = RAW_GW_FILE) -> Path:
    """
    Build a clean per-player-per-gameweek dataset:

    1. Load raw Kaggle-based CSV (player_gameweeks_raw.csv).
    2. Clean and filter seasons (2016/17 → 2022/23).
    3. Standardise columns.
    4. Save to data/processed/player_gameweeks.csv.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_gameweeks(raw_path)
    df_clean = clean_gameweeks(df_raw)  # <--- df_clean est bien un DataFrame

    df_clean.to_csv(PLAYER_GW_FILE, index=False)
    return PLAYER_GW_FILE


def run_pipeline() -> Path:
    """
    Main entry point used by the CLI.

    Runs the gameweek-level pipeline and returns the path to
    player_gameweeks.csv (the file consumed by GW models).
    """
    return build_player_gameweeks()


if __name__ == "__main__":
    output_path = run_pipeline()
    print(f"Data pipeline completed. File saved to: {output_path}")
