"""
data_pipeline.py

Simple data cleaning pipeline for the FPL Points Predictor project.

Season-level pipeline:
- Load the three raw CSV files (one per season) from data/raw/
- Normalise the 2024-25 dataset so columns match the others
- Keep only the relevant columns (USEFUL_COLS)
- Add a 'season' column so each row knows which season it belongs to
- Concatenate all three seasons
- Save the final dataset to data/processed/players_all_seasons.csv

Gameweek-level pipeline (for GW-by-GW models):
- Load a raw per-player-per-gameweek CSV from data/raw/player_gameweeks_raw.csv
- Standardise column names (id, name, team, position, season, gameweek, points, etc.)
- Keep only relevant columns (GAMEWEEK_COLS)
- Sort and save to data/processed/player_gameweeks.csv
"""

from pathlib import Path
import pandas as pd

# Path to the project root folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw and processed data folders
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# SEASON-LEVEL PIPELINE 

# Relevant columns from the raw datasets
USEFUL_COLS = [
    # identification
    "id", "name", "team", "position",

    # raw statistics
    "minutes", "goals_scored", "assists",
    "goals_conceded", "clean_sheets", "saves",
    "penalties_saved", "penalties_missed",
    "yellow_cards", "red_cards", "own_goals",
    "starts", "bonus", "bps",

    # advanced cumulative statistics
    "expected_goals", "expected_assists",
    "expected_goal_involvements",  # not available for 24-25
    "expected_goals_conceded",
    "influence", "creativity", "threat", "ict_index",

    # target
    "total_points",
]


def select_useful(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a new and independent DataFrame containing only the relevant columns
    defined in USEFUL_COLS (keeping only those that exist in df).
    """
    cols = [c for c in USEFUL_COLS if c in df.columns]
    return df[cols].copy()


def run_pipeline() -> Path:
    """
    Run the full SEASON-LEVEL pipeline:
    - load the three raw datasets
    - normalise the 2024-25 dataset
    - select useful columns
    - add a 'season' column
    - concatenate all seasons
    - save the final dataset to data/processed/

    Returns:
        Path to the saved CSV file.
    """

    # Load the 3 raw season datasets
    df22 = pd.read_csv(DATA_RAW_DIR / "season22-23.csv")
    df23 = pd.read_csv(DATA_RAW_DIR / "season23-24.csv")
    df24 = pd.read_csv(DATA_RAW_DIR / "season24-25.csv")

    # Normalise the 2024-25 dataset so that it matches the others

    # 1) Rename columns to match the other seasons
    rename_24 = {
        "first_name": "first",
        "second_name": "second",
        "player_position": "position",
        "team_name": "team",
    }
    df24 = df24.rename(columns=rename_24)

    # 2) Build a "name" column from "first" and "second"
    df24["name"] = df24["first"].str.strip() + " " + df24["second"].str.strip()

    # 3) Drop the temporary columns
    df24 = df24.drop(columns=["first", "second"])

    # Keep only useful columns for each season
    df22_useful = select_useful(df22)
    df23_useful = select_useful(df23)
    df24_useful = select_useful(df24)

    # Add season information (kept outside USEFUL_COLS on purpose)
    df22_useful["season"] = "2022-23"
    df23_useful["season"] = "2023-24"
    df24_useful["season"] = "2024-25"

    # Concatenate the three seasons
    full = pd.concat([df22_useful, df23_useful, df24_useful], ignore_index=True)

    # Create the output folder, save the cleaned dataset into it, and return the file path
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "players_all_seasons.csv"
    full.to_csv(out_path, index=False)

    return out_path


# GAMEWEEK-LEVEL PIPELINE 

# Raw gameweek file (one line = player, season, gameweek)
# You can create this CSV file from Kaggle (via src.fpl_kaggle_import)
RAW_GAMEWEEK_FILE = DATA_RAW_DIR / "player_gameweeks_raw.csv"

# Cleaned gameweek file
PROCESSED_GAMEWEEK_FILE = DATA_PROCESSED_DIR / "player_gameweeks.csv"

# Lagged gameweek file (with lag features)
LAGGED_GAMEWEEK_FILE = DATA_PROCESSED_DIR / "player_gameweeks_lagged.csv"

# Useful columns at gameweek level
GAMEWEEK_COLS = [
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


def build_gameweek_dataset() -> Path:
    """
    Build a tidy per-player-per-gameweek dataset from raw FPL data.

    Expected raw file
    -----------------
    RAW_GAMEWEEK_FILE should point to a CSV with (at least) columns similar to:
    - id / element (player id)
    - name / web_name
    - team / team_name
    - position
    - season / season_name
    - gameweek / round
    - minutes, goals_scored, assists, xG, xA, total_points, ...

    Returns
    -------
    Path
        Path to the processed CSV file (player_gameweeks.csv).
    """
    if not RAW_GAMEWEEK_FILE.exists():
        raise FileNotFoundError(
            f"Raw gameweek file not found: {RAW_GAMEWEEK_FILE}. "
            "Please create it or update RAW_GAMEWEEK_FILE in data_pipeline.py."
        )

    df = pd.read_csv(RAW_GAMEWEEK_FILE)

    # Rename raw columns to our standardised names
    # We build a flexible mapping: only the keys present will be used.
    rename_map = {
        "id": "player_id",
        "element": "player_id",
        "web_name": "name",
        "name": "name",
        "team": "team",
        "team_name": "team",
        "position": "position",
        "season_name": "season",
        "season": "season",
        "round": "gameweek",
        "gw": "gameweek",
        "gameweek": "gameweek",
        "minutes": "minutes",
        "goals_scored": "goals_scored",
        "assists": "assists",
        "xG": "expected_goals",
        "expected_goals": "expected_goals",
        "xA": "expected_assists",
        "expected_assists": "expected_assists",
        "total_points": "points",
        "points": "points",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Keep only the useful columns that actually exist
    existing_cols = [c for c in GAMEWEEK_COLS if c in df.columns]
    df = df[existing_cols].copy()

    # Minimal cleaning: remove lines without id/season/gameweek
    for col in ["player_id", "season", "gameweek"]:
        if col in df.columns:
            df = df.dropna(subset=[col])

    # Logical sorting
    sort_cols = [c for c in ["season", "gameweek", "player_id"] if c in df.columns]
    if sort_cols:
        df["gameweek"] = df["gameweek"].astype(int)
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # Save
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_GAMEWEEK_FILE
    df.to_csv(out_path, index=False)

    return out_path


def load_gameweek_data() -> pd.DataFrame:
    """
    Load the per-player-per-gameweek dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per (player, season, gameweek).

    Raises
    ------
    FileNotFoundError
        If the processed file does not exist (run build_gameweek_dataset() first).
    """
    if not PROCESSED_GAMEWEEK_FILE.exists():
        raise FileNotFoundError(
            f"Processed gameweek file not found: {PROCESSED_GAMEWEEK_FILE}. "
            "Run build_gameweek_dataset() first."
        )

    return pd.read_csv(PROCESSED_GAMEWEEK_FILE)


def build_lagged_gameweek_dataset(
    max_lag: int = 3,
    rolling_window: int = 3,
) -> Path:
    """
    Build a gameweek-level dataset with lagged features for each player.

    For each (season, player_id), the rows are sorted by gameweek and we create:
    - points_lag_1, ..., points_lag_{max_lag}
    - points_rolling_mean_<rolling_window>

    Returns
    -------
    Path
        Path to the processed CSV file (player_gameweeks_lagged.csv).
    """
    df = load_gameweek_data()

    required_cols = ["season", "player_id", "gameweek", "points"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in gameweek data: {missing}. "
            "Check build_gameweek_dataset() or your raw files."
        )

    # Sort by season, player, gameweek
    df["gameweek"] = df["gameweek"].astype(int)
    df = df.sort_values(["season", "player_id", "gameweek"]).reset_index(drop=True)

    group = df.groupby(["season", "player_id"], group_keys=False)

    # Lag features on points
    for lag in range(1, max_lag + 1):
        df[f"points_lag_{lag}"] = group["points"].shift(lag)

    # Rolling mean over last `rolling_window` gameweeks
    df[f"points_rolling_mean_{rolling_window}"] = (
        group["points"]
        .rolling(window=rolling_window, min_periods=rolling_window)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # Drop rows without full lag history
    lag_cols = [f"points_lag_{lag}" for lag in range(1, max_lag + 1)]
    lag_cols.append(f"points_rolling_mean_{rolling_window}")
    df = df.dropna(subset=lag_cols).reset_index(drop=True)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(LAGGED_GAMEWEEK_FILE, index=False)

    return LAGGED_GAMEWEEK_FILE


# When we run this script directly, launch the SEASON pipeline and display the output file path
if __name__ == "__main__":
    output_path = run_pipeline()
    print(f"Data pipeline completed. File saved to: {output_path}")

