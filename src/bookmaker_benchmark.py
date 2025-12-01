"""
Utilities to work with bookmaker (Bet365) data for the FPL Points Predictor project.

Current setup (multi-season):
- Uses a single cleaned odds file with all Premier League matches
  from 2016-17 to 2022-23:
    data/processed/bet365odds_epl_2016_23.csv

Main features:
- Load the cleaned Bet365 odds dataset (7 seasons)
- Compute a Bet365 "strength" per team based on normalised win probabilities
- Compare model-implied team strength (per match) vs Bet365 probabilities,
  match by match, for a given season.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

# Project structure
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Files produced by other pipelines
ODDS_FILE = DATA_PROCESSED_DIR / "bet365odds_epl_2016_23.csv"
EPL_FIXTURES_FILE = DATA_PROCESSED_DIR / "epl_fixtures_2016_23.csv"


# ----------------------------------------------------------------------
# LOADERS
# ----------------------------------------------------------------------
def load_clean_odds() -> pd.DataFrame:
    """
    Load the cleaned Bet365 odds dataset for all EPL matches
    between seasons 2016-17 and 2022-23.

    Expected columns (at least):
        home_team, away_team,
        pnorm_home_win, pnorm_draw, pnorm_away_win,
        match_date, season ("2016/17", ..., "2022/23")
    """
    if not ODDS_FILE.exists():
        raise FileNotFoundError(f"Clean odds file not found: {ODDS_FILE}")

    df = pd.read_csv(ODDS_FILE)

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
        raise ValueError(f"Missing required columns in odds file {ODDS_FILE}: {missing}")

    return df


def load_fixtures() -> pd.DataFrame:
    """
    Load Premier League fixtures with mapping (season, gameweek, home_team, away_team).

    Expected CSV:
        data/processed/epl_fixtures_2016_23.csv

    Expected columns:
        season (e.g., "2016/17"),
        gameweek (int),
        home_team,
        away_team
    """
    if not EPL_FIXTURES_FILE.exists():
        raise FileNotFoundError(
            f"Fixtures file not found: {EPL_FIXTURES_FILE}. "
            "You need a processed fixtures dataset with "
            "season, gameweek, home_team, away_team."
        )

    df = pd.read_csv(EPL_FIXTURES_FILE)

    required_cols = {"season", "gameweek", "home_team", "away_team"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in fixtures file {EPL_FIXTURES_FILE}: {missing}"
        )

    return df


# ----------------------------------------------------------------------
# TEAM STRENGTH (AGGREGATE, POUR SHOW_BOOKMAKERS)
# ----------------------------------------------------------------------
def compute_team_strength(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute "Bet365 strength" per team across ALL seasons (2016-17 -> 2022-23).

    For each team, we use:
    - average normalised home win probability when playing at home
    - average normalised away win probability when playing away

    Then we compute a single aggregated strength per team by weighting
    home and away averages by the number of matches.
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

    combined = home_strength.join(away_strength, how="outer").fillna(0.0)
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
    Main entry point used by the CLI:

    - Load the multi-season odds dataset
    - Compute per-team Bet365 strength across all 7 seasons
    """
    odds = load_clean_odds()
    strength = compute_team_strength(odds)
    return strength


# ----------------------------------------------------------------------
# MATCH-BY-MATCH COMPARISON: MODEL vs BOOKMAKER
# ----------------------------------------------------------------------
def compare_model_vs_bookmakers(
    model: str,
    test_season: str,
) -> Tuple[pd.DataFrame, float]:
    """
    Compare, MATCH BY MATCH, the model's implied home-win probability
    vs Bet365 home-win probability, for a given season.

    Steps:
    - Load odds and fixtures.
    - Keep only matches for `test_season`.
    - For each gameweek of that season:
        * use predict_gw_all_players(...) to get per-player predictions
        * aggregate to team-level strength (sum of predicted points)
        * merge with fixtures + odds for that GW
        * compute model-implied p(home win)
    - Compute MAE between model p(home win) and Bet365 pnorm_home_win.

    Returns
    -------
    comparison_df : DataFrame
        One row per match with:
            season, gameweek, home_team, away_team,
            pnorm_home_win (Bet365),
            p_model_home_win,
            abs_error
    mae : float
        Mean absolute error over all matches.
    """
    odds = load_clean_odds()
    fixtures = load_fixtures()

    # Filtrer sur la saison de test
    odds_season = odds[odds["season"] == test_season].copy()
    fixtures_season = fixtures[fixtures["season"] == test_season].copy()

    if odds_season.empty or fixtures_season.empty:
        raise ValueError(f"No odds or fixtures found for season {test_season!r}.")

    # Join odds <-> fixtures sur (season, home_team, away_team)
    matches = pd.merge(
        fixtures_season,
        odds_season,
        on=["season", "home_team", "away_team"],
        how="inner",
        suffixes=("", "_odds"),
    )

    if matches.empty:
        raise ValueError(
            f"No overlapping matches between odds and fixtures for season {test_season!r}."
        )

    # On va boucler sur les gameweeks de la saison
    gameweeks = sorted(matches["gameweek"].unique())

    from src.model import predict_gw_all_players  # import local pour éviter les cycles

    all_rows = []

    for gw in gameweeks:
        # 1) Prédictions par joueur pour cette GW
        preds = predict_gw_all_players(
            model=model,
            test_season=test_season,
            gameweek=int(gw),
        )

        if preds.empty:
            # Pas assez d'historique ou autre problème → on saute ce GW
            continue

        # 2) Force d'équipe = somme des points prédits par équipe
        team_strength = (
            preds.groupby("team")["predicted_points"]
            .sum()
            .rename("predicted_team_points")
            .reset_index()
        )
        team_strength["gameweek"] = gw

        # 3) Récupérer les matches de cette GW
        matches_gw = matches[matches["gameweek"] == gw].copy()

        # Join pour avoir pred_home_points
        matches_gw = matches_gw.merge(
            team_strength.rename(
                columns={
                    "team": "home_team",
                    "predicted_team_points": "pred_home_points",
                }
            ),
            on=["gameweek", "home_team"],
            how="left",
        )

        # Join pour avoir pred_away_points
        matches_gw = matches_gw.merge(
            team_strength.rename(
                columns={
                    "team": "away_team",
                    "predicted_team_points": "pred_away_points",
                }
            ),
            on=["gameweek", "away_team"],
            how="left",
        )

        all_rows.append(matches_gw)

    if not all_rows:
        raise ValueError(
            "No matches could be compared – model returned empty predictions "
            "for all gameweeks in that season."
        )

    comp = pd.concat(all_rows, ignore_index=True)

    # 4) Proba "home win" impliquée par le modèle
    comp["strength_sum"] = comp["pred_home_points"] + comp["pred_away_points"]
    comp = comp[comp["strength_sum"] > 0].copy()

    comp["p_model_home_win"] = comp["pred_home_points"] / comp["strength_sum"]

    # 5) Erreur absolue vs Bet365
    comp["abs_error"] = (comp["p_model_home_win"] - comp["pnorm_home_win"]).abs()
    mae = float(comp["abs_error"].mean())

    # Garder seulement les colonnes les plus parlantes pour l'analyse
    cols_to_keep = [
        "season",
        "gameweek",
        "home_team",
        "away_team",
        "pnorm_home_win",
        "p_model_home_win",
        "abs_error",
    ]
    comp = comp[cols_to_keep].sort_values(["gameweek", "home_team", "away_team"])

    return comp, mae


if __name__ == "__main__":
    # Petit test manuel : afficher le ranking Bet365
    table = build_team_strength_table()
    print(table)
