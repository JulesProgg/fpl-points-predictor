from __future__ import annotations

from pathlib import Path
import pandas as pd

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

# Processed outputs (produced by pipelines)
PLAYER_GW_FILE = DATA_PROCESSED_DIR / "player_gameweeks.csv"
EPL_FIXTURES_FILE = DATA_PROCESSED_DIR / "epl_fixtures_2016_23.csv"
ODDS_FILE = DATA_PROCESSED_DIR / "bet365odds_epl_2016_23.csv"

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

OPTIONAL_COLS = ["opponent", "was_home"]

# Mapping Bet365 -> FPL/Kaggle names
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
