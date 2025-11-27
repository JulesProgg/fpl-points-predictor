"""
Utilities to work with bookmaker (Bet365) data for the FPL Points Predictor project.

Current features:
- Load the cleaned EPL 2024-25 odds (produced by odds_pipeline.py)
- Compute a "Bet365 strength" per team, based on average win probabilities

This module does NOT depend on the FPL models.
It can later be extended to compare:
- bookmaker strength vs actual FPL results
- bookmaker strength vs model predictions
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

# Project paths (same logic as in data_pipeline.py / odds_pipeline.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# File produced by src/odds_pipeline.py
ODDS_CLEAN_FILE = DATA_PROCESSED_DIR / "bet365odds_season_24_25.csv"


def load_clean_odds(path: Path | None = None) -> pd.DataFrame:
    """
    Load the cleaned EPL 2024-25 odds dataset.

    Expected columns (created by src.odds_pipeline.run_odds_pipeline):
    - match_date
    - home_team, away_team
    - home_win_odds, draw_odds, away_win_odds
    - p_home_implied, p_draw_implied, p_away_implied
    - pnorm_home_win, pnorm_draw, pnorm_away_win
    """
    if path is None:
        path = ODDS_CLEAN_FILE

    df = pd.read_csv(path)
    return df


def compute_team_strength(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a "Bet365 strength" per team, based on normalised win probabilities.

    For each team, we use:
    - pnorm_home_win when the team plays at home
    - pnorm_away_win when the team plays away

    Then we average those probabilities across all matches.

    Returns
    -------
    DataFrame with columns:
    - team
    - bet365_strength   (average win probability)
    - n_matches         (number of matches used)
    """
    df = odds_df.copy()

    required_cols = {"home_team", "away_team", "pnorm_home_win", "pnorm_away_win"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in odds_df: {missing}")

    # View from home side
    home_strength = (
        df.groupby("home_team")["pnorm_home_win"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "home_win_prob_mean", "count": "home_matches"})
    )

    # View from away side
    away_strength = (
        df.groupby("away_team")["pnorm_away_win"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "away_win_prob_mean", "count": "away_matches"})
    )

    # Combine both
    combined = home_strength.join(away_strength, how="outer").fillna(0)

    # we give the index an explicit name
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



def build_team_strength_table(path: Path | None = None) -> pd.DataFrame:
    """
    Convenience function:
    - load cleaned odds
    - compute team strength
    - return the strength table
    """
    odds = load_clean_odds(path)
    strength = compute_team_strength(odds)
    return strength


if __name__ == "__main__":
    strength_table = build_team_strength_table()
    print(strength_table)
