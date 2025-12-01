"""
Utilities to work with bookmaker (Bet365) data for the FPL Points Predictor project.

Now supports multiple seasons:
- Seasons 2016-17 to 2022-23

Main features:
- Load all cleaned Bet365 odds files for 7 seasons
- Concatenate into a single dataset
- Compute Bet365 strength per team based on normalised win probabilities
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

# Project structure
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ---- NEW ----
# List of seasons we now support
SEASONS = [
    "2016_17",
    "2017_18",
    "2018_19",
    "2019_20",
    "2020_21",
    "2021_22",
    "2022_23",
]

def load_clean_odds_all_seasons() -> pd.DataFrame:
    """
    Load all cleaned Bet365 odds files for seasons 2016-17 to 2022-23.
    
    Expected filenames:
        bet365odds_season_2016_17.csv
        bet365odds_season_2017_18.csv
        ...
        bet365odds_season_2022_23.csv
    """
    frames = []
    for season in SEASONS:
        file = DATA_PROCESSED_DIR / f"bet365odds_season_{season}.csv"

        if not file.exists():
            raise FileNotFoundError(f"Missing odds file: {file}")

        df = pd.read_csv(file)
        df["season"] = season
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def compute_team_strength(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute "Bet365 strength" per team across ALL seasons:
    - pnorm_home_win when playing at home
    - pnorm_away_win when playing away
    """
    df = odds_df.copy()

    required_cols = {"home_team", "away_team", "pnorm_home_win", "pnorm_away_win"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in odds_df: {missing}")

    home_strength = (
        df.groupby("home_team")["pnorm_home_win"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "home_win_prob_mean", "count": "home_matches"})
    )

    away_strength = (
        df.groupby("away_team")["pnorm_away_win"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "away_win_prob_mean", "count": "away_matches"})
    )

    combined = home_strength.join(away_strength, how="outer").fillna(0)
    combined.index.name = "team"

    total_matches = combined["home_matches"] + combined["away_matches"]
    total_matches = total_matches.replace(0, pd.NA)

    weighted_sum = (
        combined["home_win_prob_mean"] * combined["home_matches"]
        + combined["away_win_prob_mean"] * combined["away_matches"]
    )

    combined["bet365_strength"] = weighted_sum / total_matches
    combined["n_matches"] = total_matches

    result = (
        combined.reset_index()[["team", "bet365_strength", "n_matches"]]
        .dropna(subset=["bet365_strength"])
        .sort_values("bet365_strength", ascending=False)
        .reset_index(drop=True)
    )

    return result


def build_team_strength_table() -> pd.DataFrame:
    """
    Load all seasons 2016-2023 and compute team strength.
    """
    odds = load_clean_odds_all_seasons()
    strength = compute_team_strength(odds)
    return strength


if __name__ == "__main__":
    table = build_team_strength_table()
    print(table)

