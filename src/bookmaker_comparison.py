"""
Comparison utilities between the FPL model and Bet365 bookmaker odds.

Goal
----
For a given TEST season, we build a match-level table that contains,
for each match:

- Bookmaker win probabilities:
    p_home_book, p_away_book

- Model-based "strength" for each team:
    P_home_model, P_away_model  (sum of predicted FPL points for all players)

- Normalised model probabilities:
    p_home_model = P_home_model / (P_home_model + P_away_model)
    p_away_model = P_away_model / (P_home_model + P_away_model)

This table can then be used to:
- Compare model vs bookmaker on predicting match outcomes
- Compute MAE / Brier Score between probabilities and actual results
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .model import predict_gw_all_players  # make sure this exists and is imported correctly


# Project structure (same logic as other modules)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_clean_odds_for_season(season: str) -> pd.DataFrame:
    """
    Load the cleaned Bet365 odds file for a single season.

    Expected path:
        data/processed/bet365odds_season_<season>.csv
    For example:
        bet365odds_season_2016_17.csv

    Expected columns:
        - season
        - gameweek
        - match_date
        - home_team, away_team
        - pnorm_home_win, pnorm_away_win

    If 'season' is missing in the file, we add it from the argument.
    """
    file = DATA_PROCESSED_DIR / f"bet365odds_season_{season}.csv"
    if not file.exists():
        raise FileNotFoundError(f"Missing odds file for season {season}: {file}")

    odds = pd.read_csv(file)

    if "season" not in odds.columns:
        odds["season"] = season

    required_cols = {
        "season",
        "gameweek",
        "match_date",
        "home_team",
        "away_team",
        "pnorm_home_win",
        "pnorm_away_win",
    }
    missing = required_cols - set(odds.columns)
    if missing:
        raise ValueError(f"Odds file for season {season} is missing columns: {missing}")

    return odds


def _build_match_rows_for_gw(
    odds_gw: pd.DataFrame,
    preds_gw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Internal helper: build match-level rows for a single gameweek.

    Parameters
    ----------
    odds_gw : DataFrame
        One row per match (home_team, away_team, probs...) for a given GW.
    preds_gw : DataFrame
        One row per player with predicted points for the same GW.

    Returns
    -------
    DataFrame with one row per match containing:
        - season, gameweek, match_date
        - home_team, away_team
        - p_home_book, p_away_book
        - P_home_model, P_away_model
        - p_home_model, p_away_model
    """
    # Sum predicted points per team
    if "team" not in preds_gw.columns:
        raise ValueError("Expected column 'team' in player predictions for the GW.")

    if "predicted_points" not in preds_gw.columns:
        raise ValueError("Expected column 'predicted_points' in player predictions for the GW.")

    team_points = (
        preds_gw.groupby("team")["predicted_points"]
        .sum()
        .rename("P_team_model")
    )

    rows = []

    for _, m in odds_gw.iterrows():
        season = m["season"]
        gameweek = m["gameweek"]
        match_date = m["match_date"]
        home_team = m["home_team"]
        away_team = m["away_team"]
        p_home_book = m["pnorm_home_win"]
        p_away_book = m["pnorm_away_win"]

        # Model-side aggregated points
        P_home = team_points.get(home_team, np.nan)
        P_away = team_points.get(away_team, np.nan)

        if np.isnan(P_home) or np.isnan(P_away):
            # If for some reason we are missing one of the teams in model predictions,
            # we keep NaNs for the model quantities.
            p_home_model = np.nan
            p_away_model = np.nan
        else:
            total = P_home + P_away
            if total <= 0:
                p_home_model = np.nan
                p_away_model = np.nan
            else:
                p_home_model = float(P_home / total)
                p_away_model = float(P_away / total)

        rows.append(
            {
                "season": season,
                "gameweek": gameweek,
                "match_date": match_date,
                "home_team": home_team,
                "away_team": away_team,
                "p_home_book": p_home_book,
                "p_away_book": p_away_book,
                "P_home_model": P_home,
                "P_away_model": P_away,
                "p_home_model": p_home_model,
                "p_away_model": p_away_model,
            }
        )

    return pd.DataFrame(rows)


def build_match_level_comparison(
    season_test: str,
    model: str = "gw_seasonal_gbm",
) -> pd.DataFrame:
    """
    Build a match-level comparison table between:
        - Bet365 bookmaker probabilities
        - Model-based "probabilities" derived from team total predicted points

    For each match in the TEST season, we:
        1) Load bookmaker odds and implied probabilities.
        2) For each gameweek:
            - Predict player points for ALL players (predict_gw_all_players).
            - Aggregate to team-level total predicted points.
            - Convert to normalised model "probabilities".
        3) Assemble a single DataFrame with one row per match.

    Parameters
    ----------
    season_test : str
        Season we use as TEST season for the comparison
        (e.g. "2022_23" or "2022-23" depending on your naming convention).
        This must match the season used in the odds filenames.
    model : str, optional
        Name of the GW model to use, passed directly to predict_gw_all_players.

    Returns
    -------
    DataFrame
        Columns:
        - season, gameweek, match_date
        - home_team, away_team
        - p_home_book, p_away_book
        - P_home_model, P_away_model
        - p_home_model, p_away_model
    """
    odds = load_clean_odds_for_season(season_test)

    # Ensure match_date is a datetime for consistency
    odds["match_date"] = pd.to_datetime(odds["match_date"])

    all_rows: list[pd.DataFrame] = []

    gameweeks = sorted(odds["gameweek"].unique())

    for gw in gameweeks:
        odds_gw = odds[odds["gameweek"] == gw]

        # Predict player-level points for this GW and season
        preds_gw = predict_gw_all_players(
            season=season_test,
            gameweek=int(gw),
            model=model,
        )

        rows_gw = _build_match_rows_for_gw(odds_gw, preds_gw)
        all_rows.append(rows_gw)

    if not all_rows:
        raise ValueError(f"No matches found for season {season_test} in odds data.")

    result = pd.concat(all_rows, ignore_index=True)
    return result


def simple_mae_book_vs_model(
    match_df: pd.DataFrame,
    side: Literal["home", "away"] = "home",
) -> float:
    """
    Optional helper: compute a simple MAE between bookmaker and model
    probabilities on one side (home or away), ignoring matches where
    model probabilities are NaN.

    This is just an example metric; you can adapt it or add others.
    """
    if side not in {"home", "away"}:
        raise ValueError("side must be 'home' or 'away'.")

    p_book_col = f"p_{side}_book"
    p_model_col = f"p_{side}_model"

    if p_book_col not in match_df.columns or p_model_col not in match_df.columns:
        raise ValueError("Missing required columns in match_df for MAE computation.")

    df = match_df[[p_book_col, p_model_col]].dropna()
    if df.empty:
        raise ValueError("No valid rows for MAE computation (all NaNs?).")

    mae = float(np.mean(np.abs(df[p_book_col] - df[p_model_col])))
    return mae
