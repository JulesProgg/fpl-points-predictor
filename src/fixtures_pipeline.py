"""
Fixtures pipeline for the FPL Points Predictor project.

Goal:
- Build a clean fixtures file for the English Premier League
  covering seasons 2016/17 to 2022/23, consistent with the FPL data.

Input:
- data/raw/player_gameweeks_raw.csv  (Kaggle FPL player-gameweeks dataset)

Expected columns (at least):
- season        (e.g. "2016/17")
- gameweek      (int)
- team          (name of the player's team)
- opponent      (name of the opponent team)
- was_home      (boolean or 0/1)

Output:
- data/processed/epl_fixtures_2016_23.csv
  with columns: season, gameweek, home_team, away_team
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_GW_FILE = DATA_RAW_DIR / "player_gameweeks_raw.csv"
EPL_FIXTURES_FILE = DATA_PROCESSED_DIR / "epl_fixtures_2016_23.csv"

ALLOWED_SEASONS = [
    "2016/17",
    "2017/18",
    "2018/19",
    "2019/20",
    "2020/21",
    "2021/22",
    "2022/23",
]


def build_fixtures(raw_path: Path | str = RAW_GW_FILE) -> Path:
    """
    Build a clean fixtures file for EPL seasons 2016/17 → 2022/23.

    Logic:
    - Load the raw player-gameweeks dataset.
    - Keep only the columns needed to identify matches.
    - Filter to ALLOWED_SEASONS.
    - Keep only rows where the player's team is playing at HOME (was_home == True).
      Each such row corresponds to a (season, GW, home_team, away_team) match.
    - Drop duplicates (one row per match).
    - Save to data/processed/epl_fixtures_2016_23.csv.
    """
    use_cols = ["season", "gameweek", "team", "opponent", "was_home"]

    df = pd.read_csv(raw_path, usecols=use_cols)

    missing = [c for c in use_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in raw file {raw_path!r}: {missing}"
        )

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


if __name__ == "__main__":
    out = build_fixtures()
    print(f"Fixtures pipeline completed. File saved to: {out}")

