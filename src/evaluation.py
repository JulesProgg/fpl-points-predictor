from __future__ import annotations

from typing import Tuple
from sklearn.linear_model import LinearRegression
from pathlib import Path
import numpy as np
import pandas as pd
import kagglehub

from src.data_loader import  

from src.data_loader import (
    load_clean_odds,
    load_fixtures,
    ALLOWED_SEASONS,
    GAMEWEEK_BASE_COLS,
    OPTIONAL_COLS,
    DATA_PROCESSED_DIR,
    PLAYER_GW_FILE,
    load_raw_gameweeks,
    EPL_FIXTURES_FILE,
    load_raw_fixtures_source,
    DATA_RAW_DIR,
    OUTPUT_PATH,
    RENAME_MAP,
    TARGET_COLUMNS,
    RAW_ODDS_FILE,
    OUT_ODDS_FILE,
    EPL_CODE,
    SEASON_RANGES,
    load_raw_odds,
)



# ----------------------------------------------------------------------
# TEAM STRENGTH (AGGREGATE, POUR SHOW_BOOKMAKERS) ((((BOOKMAKERBENCHMARK.py))))
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
    """
    odds = load_clean_odds()
    fixtures = load_fixtures()

    odds_season = odds[odds["season"] == test_season].copy()
    fixtures_season = fixtures[fixtures["season"] == test_season].copy()

    if odds_season.empty or fixtures_season.empty:
        raise ValueError(f"No odds or fixtures found for season {test_season!r}.")

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
    from src.models import predict_gw_all_players  # local import to avoid circular deps

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
        preds_all["playing_weight"] = (preds_all["recent_minutes_mean"] / 90.0).clip(0.0, 1.0)
    else:
        preds_all["playing_weight"] = 1.0

    preds_all["weighted_points"] = preds_all["predicted_points"] * preds_all["playing_weight"]

    preds_all = (
        preds_all
        .sort_values(
            ["season", "gameweek", "team", "weighted_points"],
            ascending=[True, True, True, False],
        )
        .groupby(["season", "gameweek", "team"])
        .head(11)
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
    comp = matches.merge(
        team_strength.rename(columns={"team": "home_team", "predicted_team_points": "pred_home_points"}),
        on=["season", "gameweek", "home_team"],
        how="left",
    )

    # ------------------------------------------------------
    # 5) Merge away team predicted strength
    # ------------------------------------------------------
    comp = comp.merge(
        team_strength.rename(columns={"team": "away_team", "predicted_team_points": "pred_away_points"}),
        on=["season", "gameweek", "away_team"],
        how="left",
    )

    # ------------------------------------------------------
    # 6) Build comparison table and clean invalid rows
    # ------------------------------------------------------
    comp = comp.replace([np.inf, -np.inf], np.nan)
    comp = comp.dropna(subset=["pred_home_points", "pred_away_points", "pnorm_home_win"])

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
        p_model = S_home / (S_home + S_away)

        mae_g = float(np.mean(np.abs(p_model - comp["pnorm_home_win"])))
        if mae_g < best_mae:
            best_mae = mae_g
            best_gamma = g

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

    eps = 1e-6
    y_clip = np.clip(y, eps, 1.0 - eps)
    logit_y = np.log(y_clip / (1.0 - y_clip))

    reg = LinearRegression()
    reg.fit(X, logit_y)

    logit_pred = reg.predict(X)
    comp["p_model_home_win"] = 1.0 / (1.0 + np.exp(-logit_pred))

    # ------------------------------------------------------
    # 9) Final error metrics
    # ------------------------------------------------------
    comp["abs_error"] = (comp["p_model_home_win"] - comp["pnorm_home_win"]).abs()
    mae = float(comp["abs_error"].mean())
    corr = float(comp["p_model_home_win"].corr(comp["pnorm_home_win"]))

    cols_to_keep = [
        "season",
        "gameweek",
        "home_team",
        "away_team",
        "pnorm_home_win",
        "p_model_home_win",
        "abs_error",
    ]

    comp = comp[cols_to_keep].sort_values(["gameweek", "home_team", "away_team"]).reset_index(drop=True)

    return comp, mae, corr


def print_example_matches(comp: pd.DataFrame, n: int = 10) -> None:
    """
    Displays a few sample matches, sorted by the largest absolute discrepancy
    between the model and Bet365.
    """
    if comp.empty:
        print("No matches to display.")
        return

    examples = comp.sort_values("abs_error", ascending=False).head(n)

    print("Example matches (sorted by largest absolute disagreement):")
    print("-" * 80)
    for _, row in examples.iterrows():
        season = row["season"]
        gw = int(row["gameweek"])
        home = row["home_team"]
        away = row["away_team"]
        p_bookie = row["pnorm_home_win"]
        p_model = row["p_model_home_win"]
        diff = p_model - p_bookie

        print(f"{season} GW{gw}: {home} vs {away}")
        print(f"  Bet365 home-win prob : {p_bookie:.3f}")
        print(f"  Model home-win prob  : {p_model:.3f}")
        print(f"  Difference (model - Bet365): {diff:+.3f}")
        print()





# ---------------------------------------------------------------------
# ((((DATAPIPELINE.py))))
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


# ---------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------
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
# ((((FIXTUREPIPELINE.py))))
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
# ((((FPLKAGGLEIMPORT))))
# ---------------------------------------------------------------------    



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


# ---------------------------------------------------------------------
# ((((ODDSPIPELINE.py))))
# --------------------------------------------------------------------- 

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

