from __future__ import annotations

from pathlib import Path
import pandas as pd
import kagglehub


# =============================================================================
# data_loader.py
# =============================================================================
# Single source of truth for:
#   1) Project paths to raw/processed datasets
#   2) Standardisation utilities (season strings, team names)
#   3) Public data loading functions (raw + processed CSVs)
#   4) Pipeline helpers previously living in separate modules:
#        - cleaning gameweeks
#        - building fixtures
#        - running odds pipeline
#        - building raw gameweeks from Kaggle (download + conversion)
#
# IMPORTANT:
# - This file mixes "loaders" and "pipelines" utilities on purpose (project choice).
# - Some pipeline utilities depend on external constants / imports which must exist
#   in the runtime context (see notes in their section). Code is kept identical.
# =============================================================================


# ---------------------------------------------------------------------
# PATHS (single source of truth)
# ---------------------------------------------------------------------
# File location: <project_root>/src/data/data_loader.py

PROJECT_ROOT = Path.cwd()

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Raw inputs
RAW_GW_FILE = DATA_RAW_DIR / "player_gameweeks_raw.csv"
RAW_ODDS_FILE = DATA_RAW_DIR / "oddsdataset.csv"

# Kaggle import output (raw GW dataset path)
OUTPUT_PATH = RAW_GW_FILE


# Processed outputs (produced by pipelines)
PLAYER_GW_FILE = DATA_PROCESSED_DIR / "player_gameweeks.csv"
EPL_FIXTURES_FILE = DATA_PROCESSED_DIR / "epl_fixtures_2016_23.csv"
ODDS_FILE = DATA_PROCESSED_DIR / "bet365odds_epl_2016_23.csv"

# ---------------------------------------------------------------------
# ODDS PIPELINE CONSTANTS (required by run_odds_pipeline / assign_season_column)
# ---------------------------------------------------------------------

EPL_CODE = "E0"
OUT_ODDS_FILE = ODDS_FILE

SEASON_RANGES = [
    ("2016-08-01", "2017-06-30", "2016/17"),
    ("2017-08-01", "2018-06-30", "2017/18"),
    ("2018-08-01", "2019-06-30", "2018/19"),
    ("2019-08-01", "2020-06-30", "2019/20"),
    ("2020-08-01", "2021-06-30", "2020/21"),
    ("2021-08-01", "2022-06-30", "2021/22"),
    ("2022-08-01", "2023-06-30", "2022/23"),
]


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
# Seasons supported end-to-end by the project (raw -> processed -> models)
ALLOWED_SEASONS = [
    "2016/17",
    "2017/18",
    "2018/19",
    "2019/20",
    "2020/21",
    "2021/22",
    "2022/23",
]

# Base columns expected in the raw player-gameweeks dataset
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

# Optional columns sometimes present in raw data sources
OPTIONAL_COLS = ["opponent", "was_home"]


# ---------------------------------------------------------------------
# NAME MAPPINGS
# ---------------------------------------------------------------------
# Mapping Bet365 -> FPL/Kaggle names (for consistent merges on team names)
TEAM_NAME_MAP_B365_TO_FPL = {
    "Man City": "Manchester City",
    "Man Utd": "Manchester United",
    "Spurs": "Tottenham Hotspur",
    "Wolves": "Wolverhampton Wanderers",
    "Newcastle Utd": "Newcastle United",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United",
    "Huddersfield": "Huddersfield Town",
    "Cardiff": "Cardiff City",
    "Norwich": "Norwich City",
    "Sheff Utd": "Sheffield United",
    "Leeds": "Leeds United",
}

# ---------------------------------------------------------------------
# FPL Kaggle import mapping (public API via __init__.py)
# ---------------------------------------------------------------------
# Column mapping to harmonise Kaggle dataset fields into project conventions.
RENAME_MAP = {
    "element": "id",
    "name": "name",
    "team": "team",
    "position": "position",
    "minutes_played": "minutes",
    "goals_scored": "goals_scored",
    "assists": "assists",
    "goals_conceded": "goals_conceded",
    "clean_sheets": "clean_sheets",
    "saves": "saves",
    "penalties_saved": "penalties_saved",
    "penalties_missed": "penalties_missed",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "own_goals": "own_goals",
    "bonus": "bonus",
    "bps": "bps",
    "creativity": "creativity",
    "threat": "threat",
    "influence": "influence",
    "ict_index": "ict_index",
    "points": "total_points",
    "season": "season",
    "gameweek": "gameweek",
    "round": "gameweek",  # au cas où certains fichiers utilisent "round"
    "was_home": "was_home",
    "opponent": "opponent",
}

# Target schema for the combined raw GW dataset generated from Kaggle import.
TARGET_COLUMNS = [
    "id",
    "name",
    "team",
    "opponent",
    "was_home",
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
# NORMALISATION HELPERS
# ---------------------------------------------------------------------
def normalise_season_str(s: str) -> str:
    """
    Normalise season strings to a single format, e.g. '2016/17'.

    Accepts:
        '2016-17', '2016_17', '2016/2017' -> '2016/17'
    """
    s = str(s).strip()
    s = s.replace("_", "/").replace("-", "/")

    # Typical case '2016/2017' -> '2016/17'
    if len(s) == 9 and s[4] == "/" and s[7:9].isdigit():
        return f"{s[:4]}/{s[7:]}"
    return s


def normalise_team_names(
    df: pd.DataFrame,
    home_col: str = "home_team",
    away_col: str = "away_team",
) -> pd.DataFrame:
    """Harmonise Bet365 team names toward the names used in FPL/Kaggle datasets."""
    df = df.copy()
    df[home_col] = df[home_col].replace(TEAM_NAME_MAP_B365_TO_FPL)
    df[away_col] = df[away_col].replace(TEAM_NAME_MAP_B365_TO_FPL)
    return df


# ---------------------------------------------------------------------
# LOADERS (public API via src/data/__init__.py)
# ---------------------------------------------------------------------
# These functions are pure "I/O + validation + normalisation" helpers.
# They do not build/transform datasets beyond minimal standardisation.
def load_clean_odds(path: Path | str = ODDS_FILE) -> pd.DataFrame:
    """
    Load the cleaned Bet365 odds dataset.

    Expected columns (at least):
        home_team, away_team,
        pnorm_home_win, pnorm_draw, pnorm_away_win,
        match_date, season
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Clean odds file not found: {path}")

    df = pd.read_csv(path)

    if "season" not in df.columns:
        raise ValueError(f"Missing required column 'season' in odds file {path}")

    df["season"] = df["season"].map(normalise_season_str)
    df = normalise_team_names(df, home_col="home_team", away_col="away_team")

    required_cols = {
        "home_team",
        "away_team",
        "pnorm_home_win",
        "pnorm_draw",
        "pnorm_away_win",
        "match_date",
        "season",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in odds file {path}: {sorted(missing)}")

    return df


def load_fixtures(path: Path | str = EPL_FIXTURES_FILE) -> pd.DataFrame:
    """
    Load EPL fixtures (season, gameweek, home_team, away_team).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Fixtures file not found: {path}. "
            "Expected a processed fixtures dataset with columns: season, gameweek, home_team, away_team."
        )

    df = pd.read_csv(path)

    if "season" not in df.columns:
        raise ValueError(f"Missing required column 'season' in fixtures file {path}")

    df["season"] = df["season"].map(normalise_season_str)
    df = normalise_team_names(df, home_col="home_team", away_col="away_team")

    required_cols = {"season", "gameweek", "home_team", "away_team"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in fixtures file {path}: {sorted(missing)}")

    return df


def load_raw_gameweeks(path: Path | str = RAW_GW_FILE) -> pd.DataFrame:
    """
    Load raw per-player-per-gameweek dataset.

    Required columns: GAMEWEEK_BASE_COLS
    Optional columns: OPTIONAL_COLS
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw gameweeks file not found: {path}")

    df = pd.read_csv(path)

    missing_required = [c for c in GAMEWEEK_BASE_COLS if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Missing REQUIRED columns in raw gameweeks file {path}: {missing_required}"
        )

    return df


def load_raw_fixtures_source(path: Path | str = RAW_GW_FILE) -> pd.DataFrame:
    """
    Load raw columns needed to build fixtures from player-gameweeks.

    Required columns:
      - season, gameweek, team, opponent, was_home
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw gameweeks file not found: {path}")

    use_cols = ["season", "gameweek", "team", "opponent", "was_home"]

    # pandas will raise if any is missing, but this yields a clearer message
    df = pd.read_csv(path)
    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw gameweeks file {path}: {missing}")

    return df[use_cols].copy()


def load_raw_odds(path: Path | str = RAW_ODDS_FILE) -> pd.DataFrame:
    """
    Load the raw odds dataset with only useful columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw odds file not found: {path}")

    use_cols = [
        "Division",
        "MatchDate",
        "HomeTeam",
        "AwayTeam",
        "OddHome",
        "OddDraw",
        "OddAway",
    ]
    return pd.read_csv(path, usecols=use_cols, low_memory=False)


def load_player_gameweeks(path: Path | str = PLAYER_GW_FILE) -> pd.DataFrame:
    """
    Load the cleaned per-player-per-gameweek dataset (processed).

    Canonical target column: 'points'
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Processed player gameweeks file not found: {path}")

    df = pd.read_csv(path)

    # Normalise season format
    if "season" in df.columns:
        df["season"] = df["season"].map(normalise_season_str)

    required = {"id", "name", "team", "position", "season", "gameweek", "points"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in processed gameweeks file {path}: {sorted(missing)}"
        )

    return df


# =============================================================================
# PIPELINE UTILITIES (formerly separate modules)
# =============================================================================
# The following functions are "data preparation" steps that produce datasets
# stored under data/processed/. They rely on the loaders above.
# =============================================================================


# ---------------------------------------------------------------------
# GAMEWEEKS CLEANING (formerly DATAPIPELINE.py)
# ---------------------------------------------------------------------
def filter_allowed_seasons(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only seasons in ALLOWED_SEASONS."""
    if "season" not in df.columns:
        raise ValueError("Column 'season' is missing from dataframe.")
    return df[df["season"].isin(ALLOWED_SEASONS)].copy()


def clean_gameweeks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise the gameweek-level dataset:
    - Keep required + optional columns
    - Filter seasons (2016/17 → 2022/23)
    - Rename total_points → points
    - Rename opponent → opponent_team (if present)
    - Create player_id
    - Sort by (player_id, season, gameweek)
    """
    available_optional = [c for c in OPTIONAL_COLS if c in df.columns]
    cols_to_keep = GAMEWEEK_BASE_COLS + available_optional
    df = df[cols_to_keep].copy()

    df = filter_allowed_seasons(df)

    df = df.rename(columns={"total_points": "points"})

    if "opponent" in df.columns:
        df = df.rename(columns={"opponent": "opponent_team"})

    if "player_id" not in df.columns:
        df["player_id"] = df["id"]

    try:
        df["gameweek"] = df["gameweek"].astype(int)
    except Exception:
        pass

    df = (
        df.sort_values(["player_id", "season", "gameweek"])
        .reset_index(drop=True)
    )

    cols = ["player_id"] + [c for c in df.columns if c != "player_id"]
    df = df[cols]

    return df


def build_player_gameweeks(raw_path: Path | str | None = None) -> Path:
    """
    Build the clean per-player-per-gameweek dataset:
    - Load raw Kaggle CSV
    - Clean & filter seasons
    - Save to data/processed/player_gameweeks.csv
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_gameweeks(raw_path) if raw_path else load_raw_gameweeks()
    df_clean = clean_gameweeks(df_raw)

    df_clean.to_csv(PLAYER_GW_FILE, index=False)
    return PLAYER_GW_FILE


# ---------------------------------------------------------------------
# FIXTURES BUILDING (formerly FIXTUREPIPELINE.py)
# ---------------------------------------------------------------------
def build_fixtures(raw_path: Path | str | None = None) -> Path:
    """
    Build a clean fixtures file for EPL seasons 2016/17 → 2022/23.

    Logic:
    - Load the raw player-gameweeks dataset (fixtures-relevant columns only).
    - Filter to ALLOWED_SEASONS.
    - Keep only rows where the player's team is playing at HOME.
    - Drop duplicates (one row per match).
    - Save to data/processed/epl_fixtures_2016_23.csv
      with columns: season, gameweek, home_team, away_team
    """
    df = load_raw_fixtures_source(raw_path) if raw_path else load_raw_fixtures_source()

    # Filter seasons
    df = df[df["season"].isin(ALLOWED_SEASONS)].copy()

    # Ensure gameweek is int
    try:
        df["gameweek"] = df["gameweek"].astype(int)
    except Exception:
        pass

    # Normalise was_home to boolean
    df["was_home"] = df["was_home"].astype(str).str.lower().isin(["true", "1", "yes"])

    # Keep only home matches
    df_home = df[df["was_home"]].copy()

    # Build fixtures: one row per (season, GW, home_team, away_team)
    fixtures = (
        df_home[["season", "gameweek", "team", "opponent"]]
        .drop_duplicates()
        .rename(columns={"team": "home_team", "opponent": "away_team"})
        .sort_values(["season", "gameweek", "home_team", "away_team"])
        .reset_index(drop=True)
    )

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fixtures.to_csv(EPL_FIXTURES_FILE, index=False)

    return EPL_FIXTURES_FILE


# ---------------------------------------------------------------------
# ODDS PIPELINE (formerly ODDSPIPELINE.py)
# ---------------------------------------------------------------------
# NOTE:
# - This code expects certain constants to exist:
#     SEASON_RANGES, EPL_CODE, OUT_ODDS_FILE
# - The module keeps the code identical; ensure those constants are defined
#   somewhere in your project scope before running this pipeline.
def assign_season_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'season' column to the odds dataframe based on MatchDate.

    Season labels match the FPL format:
    "2016/17", ..., "2022/23".
    Rows outside these date ranges are dropped.
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["MatchDate"]):
        df["MatchDate"] = pd.to_datetime(df["MatchDate"])

    df["season"] = pd.NA

    for start, end, season_label in SEASON_RANGES:
        mask = df["MatchDate"].between(start, end)
        df.loc[mask, "season"] = season_label

    return df[df["season"].notna()].copy()


def run_odds_pipeline(raw_path: Path | str = RAW_ODDS_FILE) -> Path:
    """
    Run the full odds pipeline and save to data/processed/bet365odds_epl_2016_23.csv.
    """
    odds = load_raw_odds(raw_path)

    # Date conversion
    odds["MatchDate"] = pd.to_datetime(odds["MatchDate"])

    # EPL only
    odds = odds[odds["Division"] == EPL_CODE].copy()

    # Assign seasons and drop out-of-scope matches
    odds = assign_season_column(odds)

    # Rename columns
    odds = odds.rename(
        columns={
            "MatchDate": "match_date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "OddHome": "home_win_odds",
            "OddDraw": "draw_odds",
            "OddAway": "away_win_odds",
        }
    )

    # Team name normalisation
    team_fix = {
        "Nott'm Forest": "Nottingham Forest",
        "Nottm Forest": "Nottingham Forest",
    }
    odds["home_team"] = odds["home_team"].replace(team_fix)
    odds["away_team"] = odds["away_team"].replace(team_fix)

    # Implied probabilities
    odds["p_home_implied"] = 1.0 / odds["home_win_odds"]
    odds["p_draw_implied"] = 1.0 / odds["draw_odds"]
    odds["p_away_implied"] = 1.0 / odds["away_win_odds"]

    total = odds["p_home_implied"] + odds["p_draw_implied"] + odds["p_away_implied"]

    # Normalised probabilities
    odds["pnorm_home_win"] = odds["p_home_implied"] / total
    odds["pnorm_draw"] = odds["p_draw_implied"] / total
    odds["pnorm_away_win"] = odds["p_away_implied"] / total

    # Save
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    odds.to_csv(OUT_ODDS_FILE, index=False)

    return OUT_ODDS_FILE


# ---------------------------------------------------------------------
# KAGGLE RAW GAMEWEEKS BUILD (formerly FPLKAGGLEIMPORT)
# ---------------------------------------------------------------------
# NOTE:
# - This code expects external elements to exist:
#     kagglehub, OUTPUT_PATH
# - Code is kept identical: ensure these names are available at runtime.
def build_player_gameweeks_raw_from_kaggle() -> Path:
    """
    Download the Kaggle dataset:
        'reevebarreto/fantasy-premier-league-player-data-2016-2024'
    and convert:
        'FPL Player Stats(2016-2024).csv'
    into:
        data/raw/player_gameweeks_raw.csv

    Note:
    - expected_* columns do not exist in this dataset → created as NaN.
    - starts is approximated as (minutes >= 60).
    """
    kaggle_path_str = kagglehub.dataset_download(
        "reevebarreto/fantasy-premier-league-player-data-2016-2024"
    )
    kaggle_root = Path(kaggle_path_str)
    print(f"Kaggle dataset downloaded to: {kaggle_root}")

    csv_path = kaggle_root / "FPL Player Stats(2016-2024).csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Rename columns to standard names
    rename_map = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Remove duplicated columns (e.g., two 'gameweek')
    df = df.loc[:, ~df.columns.duplicated()]

    # Create starts if missing: starts=1 if minutes >= 60
    if "starts" not in df.columns:
        if "minutes" in df.columns:
            df["starts"] = (df["minutes"] >= 60).astype(int)
        else:
            df["starts"] = pd.NA

    # Create expected_* columns if absent
    for col in [
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    # Ensure all target columns exist
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Keep only target columns in target order
    df = df[TARGET_COLUMNS].copy()

    # Save to data/raw/player_gameweeks_raw.csv
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved combined GW dataset to: {OUTPUT_PATH}")
    return OUTPUT_PATH
