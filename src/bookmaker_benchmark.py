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


from pathlib import Path
from typing import Tuple

import pandas as pd

# Project structure
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Files produced by other pipelines
ODDS_FILE = DATA_PROCESSED_DIR / "bet365odds_epl_2016_23.csv"
EPL_FIXTURES_FILE = DATA_PROCESSED_DIR / "epl_fixtures_2016_23.csv"

def _normalise_season_str(s: str) -> str:
    """
    Normalise season strings to a single format, e.g. '2016/17'.

    Accepts:
        '2016-17', '2016_17', '2016/2017' → '2016/17'
    """
    s = str(s).strip()
    s = s.replace("_", "/").replace("-", "/")

    # Cas typique '2016/2017' -> '2016/17'
    if len(s) == 9 and s[4] == "/" and s[7:9].isdigit():
        return f"{s[:4]}/{s[7:]}"
    return s

# Mapping des noms Bet365 -> noms FPL/Kaggle
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
    # 👉 complète / adapte avec les vrais noms que tu vois dans ton CSV Bet365
}


def _normalise_team_names(
    df: pd.DataFrame,
    home_col: str = "home_team",
    away_col: str = "away_team",
) -> pd.DataFrame:
    """
    Harmonise les noms d'équipes Bet365 vers les noms utilisés
    dans les données FPL (player_gameweeks / fixtures).
    """
    df = df.copy()
    df[home_col] = df[home_col].replace(TEAM_NAME_MAP_B365_TO_FPL)
    df[away_col] = df[away_col].replace(TEAM_NAME_MAP_B365_TO_FPL)
    return df



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

    # Normalisation saison + noms d'équipes
    df["season"] = df["season"].map(_normalise_season_str)
    df = _normalise_team_names(df, home_col="home_team", away_col="away_team")

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

    # Normalisation saison + noms d'équipes
    df["season"] = df["season"].map(_normalise_season_str)
    df = _normalise_team_names(df, home_col="home_team", away_col="away_team")

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
) -> Tuple[pd.DataFrame, float, float]:
    """
    Match-level comparison between the model's implied home-win probability
    and Bet365 home-win probability, for a given TEST season.

    Vectorised + calibrated version:
    - The GW model is trained ONCE for the whole season.
    - Player predictions are computed ONCE for all GWs.
    - Team strength is aggregated per (team, gameweek).
    - A logistic transformation is calibrated so that
      strength_diff -> probability matches Bet365 as closely as possible.
    """
    from sklearn.linear_model import LinearRegression
    import numpy as np

    odds = load_clean_odds()
    fixtures = load_fixtures()

    # Filter odds and fixtures to the selected season
    odds_season = odds[odds["season"] == test_season].copy()
    fixtures_season = fixtures[fixtures["season"] == test_season].copy()

    if odds_season.empty or fixtures_season.empty:
        raise ValueError(f"No odds or fixtures found for season {test_season!r}.")

    # Merge fixtures with bookmaker probabilities (inner join)
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

    # ------------------------------------------------------
    # 1) Predict ALL player-gameweeks for the entire season
    # ------------------------------------------------------
    from src.model import predict_gw_all_players  # local import to avoid circular deps

    preds_all = predict_gw_all_players(
        model=model,
        test_season=test_season,
        gameweek=None,   # compute predictions for ALL gameweeks
    )

    if preds_all.empty:
        raise ValueError(
            f"Model {model!r} returned no predictions for season {test_season!r}."
        )

    # ------------------------------------------------------
    # 2) Approximate the probability of playing based on recent minutes
    # ------------------------------------------------------
    minute_cols = [c for c in preds_all.columns if c.startswith("minutes_lag_")]

    if minute_cols:
        preds_all["recent_minutes_mean"] = preds_all[minute_cols].mean(axis=1)

        # Poids entre 0 et 1 (0 = ne joue jamais, 1 ≈ 90 minutes)
        preds_all["playing_weight"] = (
            preds_all["recent_minutes_mean"] / 90.0
        ).clip(0.0, 1.0)
    else:
        # Si jamais les minutes_lag_* n'existent pas, tout le monde compte pareil
        preds_all["playing_weight"] = 1.0

    # Points pondérés par probabilité de jouer
    preds_all["weighted_points"] = (
        preds_all["predicted_points"] * preds_all["playing_weight"]
    )

    # Ne garder que les 11 joueurs les plus importants par équipe & gameweek
    preds_all = (
        preds_all
        .sort_values(
            ["season", "gameweek", "team", "weighted_points"],
            ascending=[True, True, True, False],
        )
        .groupby(["season", "gameweek", "team"])
        .head(11)   # XI probable
        .reset_index(drop=True)
    )

    # ------------------------------------------------------
    # 3) Aggregate to team-level predicted strength per gameweek
    # ------------------------------------------------------
    team_strength = (
        preds_all.groupby(["season", "gameweek", "team"])["weighted_points"]
        .sum()
        .rename("predicted_team_points")
        .reset_index()
    )


    # ------------------------------------------------------
    # 4) Merge home team predicted strength
    # ------------------------------------------------------
    matches = matches.merge(
        team_strength.rename(
            columns={
                "team": "home_team",
                "predicted_team_points": "pred_home_points",
            }
        ),
        on=["season", "gameweek", "home_team"],
        how="left",
    )

    # ------------------------------------------------------
    # 5) Merge away team predicted strength
    # ------------------------------------------------------
    matches = matches.merge(
        team_strength.rename(
            columns={
                "team": "away_team",
                "predicted_team_points": "pred_away_points",
            }
        ),
        on=["season", "gameweek", "away_team"],
        how="left",
    )

    # ------------------------------------------------------
    # 6) Build comparison table and clean invalid rows
    # ------------------------------------------------------
    comp = matches.copy()
    comp["strength_sum"] = comp["pred_home_points"] + comp["pred_away_points"]

    # Remove invalid / infinite values
    comp = comp.replace([np.inf, -np.inf], np.nan)
    comp = comp.dropna(
        subset=["pred_home_points", "pred_away_points", "pnorm_home_win"]
    )

    if comp.empty:
        raise ValueError(
            "No matches with valid team strengths to compare "
            "(check team name mappings or model predictions)."
        )
    
        # ------------------------------------------------------
    # 7) Non-linear amplification of team strength (gamma calibration)
    # ------------------------------------------------------
    candidate_gammas = [1.0, 1.1, 1.2, 1.3, 1.4]

    best_gamma = 1.0
    best_mae = float("inf")

    for g in candidate_gammas:
        S_home = comp["pred_home_points"] ** g
        S_away = comp["pred_away_points"] ** g

        # Approximate model probability (ratio form)
        p_model = S_home / (S_home + S_away)

        mae_g = np.mean(np.abs(p_model - comp["pnorm_home_win"]))
        if mae_g < best_mae:
            best_mae = mae_g
            best_gamma = g

    # Apply best gamma exponent
    gamma = best_gamma
    comp["pred_home_points"] = comp["pred_home_points"] ** gamma
    comp["pred_away_points"] = comp["pred_away_points"] ** gamma

    print(f"[Gamma calibration] Using gamma = {gamma}, pre-logistic MAE={best_mae:.3f}")


    # ------------------------------------------------------
    # 8) Logistic calibration: strength_diff -> probability
    # ------------------------------------------------------
    comp["strength_diff"] = comp["pred_home_points"] - comp["pred_away_points"]

    X = comp["strength_diff"].to_numpy(dtype=float).reshape(-1, 1)
    y = comp["pnorm_home_win"].to_numpy(dtype=float)

    # Avoid infinities in the logit
    eps = 1e-6
    y_clip = np.clip(y, eps, 1.0 - eps)
    logit_y = np.log(y_clip / (1.0 - y_clip))

    # Fit: logit(p_book) ≈ a + b·strength_diff
    reg = LinearRegression()
    reg.fit(X, logit_y)

    # Predict calibrated logit & convert to probabilities
    logit_pred = reg.predict(X)
    comp["p_model_home_win"] = 1.0 / (1.0 + np.exp(-logit_pred))


    # ------------------------------------------------------
    # 9) Final error metrics
    # ------------------------------------------------------
    comp["abs_error"] = (comp["p_model_home_win"] - comp["pnorm_home_win"]).abs()
    mae = float(comp["abs_error"].mean())
    corr = float(comp["p_model_home_win"].corr(comp["pnorm_home_win"]))

    # Columns to keep for output
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

    return comp, mae, corr




if __name__ == "__main__":
    # Petit test manuel : afficher le ranking Bet365
    table = build_team_strength_table()
    print(table)
