from __future__ import annotations

"""
Evaluation module for the FPL Points Predictor.

This module is STRICTLY EVALUATIVE.

It contains:
- Bookmaker-related benchmarks (Bet365-based team strength, match-by-match comparison)
- Gameweek-level model evaluation utilities (MAE-based backtesting on a held-out season)
- Simple baselines and legacy wrappers kept for compatibility

"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from src import load_player_gameweeks
from src.data_loader import load_clean_odds, load_fixtures

# NOTE:
# The evaluation helpers below reuse the same lag logic as the predictive module.
# We import the lag builders from src.models to avoid duplicating feature logic.
from src.models import _add_anytime_lags, _add_seasonal_lags_with_prev5


# ----------------------------------------------------------------------
# BOOKMAKER BENCHMARKS (BET365-BASED)
# ----------------------------------------------------------------------


def compute_team_strength(odds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute an aggregate "Bet365 strength" per team across ALL seasons.

    For each team, we use:
    - average normalized home-win probability when playing at home
    - average normalized away-win probability when playing away

    A single strength score is then computed as a match-count-weighted average.
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
    CLI-friendly entry point:
    - Load the multi-season cleaned odds dataset
    - Compute per-team Bet365 strength across all seasons in scope
    """
    odds = load_clean_odds()
    strength = compute_team_strength(odds)
    return strength


def compare_model_vs_bookmakers(
    model: str,
    test_season: str,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Match-level comparison: model-implied home-win probability vs Bet365 probability.

    Method overview
    ---------------
    1) Predict all player-gameweeks for the season (via src.models.predict_gw_all_players)
    2) Approximate probability of playing using recent minutes lags (if available)
    3) Build team-level strengths by summing predicted points for the top 11 players
    4) Calibrate team-strength-to-probability mapping:
       - gamma amplification (grid search)
       - logistic calibration via linear regression in logit space
    5) Return a comparison table + summary metrics (MAE and correlation)
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
        gameweek=None,
    )

    if preds_all.empty:
        raise ValueError(
            f"Model {model!r} returned no predictions for season {test_season!r}."
        )

    # ------------------------------------------------------
    # 2) Approximate the probability of playing using recent minutes
    # ------------------------------------------------------
    minute_cols = [c for c in preds_all.columns if c.startswith("minutes_lag_")]

    if minute_cols:
        preds_all["recent_minutes_mean"] = preds_all[minute_cols].mean(axis=1)
        preds_all["playing_weight"] = (
            (preds_all["recent_minutes_mean"] / 90.0).clip(0.0, 1.0)
        )
    else:
        preds_all["playing_weight"] = 1.0

    preds_all["weighted_points"] = preds_all["predicted_points"] * preds_all["playing_weight"]

    # Keep top-11 predicted contributors per team/gameweek
    preds_all = (
        preds_all.sort_values(
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
        team_strength.rename(
            columns={"team": "home_team", "predicted_team_points": "pred_home_points"}
        ),
        on=["season", "gameweek", "home_team"],
        how="left",
    )

    # ------------------------------------------------------
    # 5) Merge away team predicted strength
    # ------------------------------------------------------
    comp = comp.merge(
        team_strength.rename(
            columns={"team": "away_team", "predicted_team_points": "pred_away_points"}
        ),
        on=["season", "gameweek", "away_team"],
        how="left",
    )

    # ------------------------------------------------------
    # 6) Clean invalid rows
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
    # 9) Final summary metrics
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

    comp = comp[cols_to_keep].sort_values(
        ["gameweek", "home_team", "away_team"]
    ).reset_index(drop=True)

    return comp, mae, corr


def print_example_matches(comp: pd.DataFrame, n: int = 10) -> None:
    """
    Print a few sample matches, sorted by largest absolute discrepancy
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
# GAMEWEEK-LEVEL MODEL EVALUATION 
# ---------------------------------------------------------------------
# The following functions are used to measure predictive performance on a held-out season.
# They do NOT belong in src.models because src.models is strictly predictive.


def _compute_mae(y_true, y_pred) -> float:
    """
    Compute Mean Absolute Error (MAE), ignoring any pair where either value is NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        raise ValueError("No valid (y_true, y_pred) pairs to compute MAE (all NaN).")

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def _build_gw_features_and_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "points",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Utility used by evaluation functions:
    - drop rows with missing required features/target
    - sanitize inf/NaN
    - return (X, y)
    """
    cols_needed = feature_cols + [target_col]
    clean_df = df.dropna(subset=cols_needed).copy()

    X = (
        clean_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    y = clean_df[target_col].astype(float)

    return X, y


# Columns that are NOT features (identifiers / metadata / target)
NON_FEATURE_COLS_GW: set[str] = {
    "id",
    "player_id",
    "name",
    "team",
    "position",
    "season",
    "gameweek",
    "points",
}


def _evaluate_anytime_linear_gw(
    max_lag: int,
    test_season: str = "2022/23",
) -> float:
    """
    Evaluate a linear regression GW model using "anytime" lags.
    """
    df = load_player_gameweeks()
    df = _add_anytime_lags(df, max_lag=max_lag, rolling_window=max_lag)

    feature_cols = [f"points_lag_{k}" for k in range(1, max_lag + 1)]
    feature_cols.append(f"points_rolling_mean_{max_lag}")

    train_df = df[df["season"] != test_season].copy()
    test_df = df[df["season"] == test_season].copy()

    X_train, y_train = _build_gw_features_and_target(train_df, feature_cols)
    X_test, y_test = _build_gw_features_and_target(test_df, feature_cols)

    reg = LinearRegression()
    reg.fit(X_train.to_numpy(dtype=float), y_train.to_numpy(dtype=float))

    y_pred = reg.predict(X_test.to_numpy(dtype=float))
    mae = _compute_mae(y_test, y_pred)
    return float(mae)


def evaluate_linear_gw_model_lag3(test_season: str = "2022/23") -> float:
    """Anytime linear GW model using the last 3 player appearances as features."""
    return _evaluate_anytime_linear_gw(max_lag=3, test_season=test_season)


def evaluate_linear_gw_model_lag5(test_season: str = "2022/23") -> float:
    """Anytime linear GW model using the last 5 player appearances as features."""
    return _evaluate_anytime_linear_gw(max_lag=5, test_season=test_season)


def evaluate_linear_gw_model_lag10(test_season: str = "2022/23") -> float:
    """Anytime linear GW model using the last 10 player appearances as features."""
    return _evaluate_anytime_linear_gw(max_lag=10, test_season=test_season)


def evaluate_linear_gw_model_seasonal(test_season: str = "2022/23") -> float:
    """
    Seasonal linear GW model evaluation using seasonal lags + fallback.
    """
    df = load_player_gameweeks()
    df = _add_seasonal_lags_with_prev5(df, max_lag=5)

    feature_cols = [f"points_lag_{k}" for k in range(1, 4)]
    feature_cols.append("points_lag_mean")

    train_df = df[df["season"] != test_season].copy()
    test_df = df[df["season"] == test_season].copy()

    X_train, y_train = _build_gw_features_and_target(train_df, feature_cols)
    X_test, y_test = _build_gw_features_and_target(test_df, feature_cols)

    reg = LinearRegression()
    reg.fit(X_train.to_numpy(dtype=float), y_train.to_numpy(dtype=float))

    y_pred = reg.predict(X_test.to_numpy(dtype=float))
    mae = _compute_mae(y_test, y_pred)
    return float(mae)


def evaluate_gbm_gw_model_seasonal(test_season: str = "2022/23") -> float:
    """
    Seasonal GradientBoostingRegressor GW model evaluation using seasonal lags + fallback.
    """
    df = load_player_gameweeks()
    df = _add_seasonal_lags_with_prev5(df, max_lag=5)

    feature_cols = [f"points_lag_{k}" for k in range(1, 4)]
    feature_cols.append("points_lag_mean")

    train_df = df[df["season"] != test_season].copy()
    train_df = train_df.dropna(subset=feature_cols + ["points"]).copy()

    if train_df.empty:
        raise ValueError(
            f"No training data available for GW model when excluding season {test_season!r}."
        )

    X_train, y_train = _build_gw_features_and_target(train_df, feature_cols)
    X_test, y_test = _build_gw_features_and_target(
        df[df["season"] == test_season].dropna(subset=feature_cols), feature_cols
    )

    gbm = GradientBoostingRegressor(
        random_state=42,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
    )
    gbm.fit(X_train.to_numpy(dtype=float), y_train.to_numpy(dtype=float))

    y_pred = gbm.predict(X_test.to_numpy(dtype=float))
    mae = _compute_mae(y_test, y_pred)
    return float(mae)


def evaluate_gw_baseline_lag1(test_season: str = "2022/23") -> float:
    """
    Naive persistence baseline for GW points:
        y_hat_t = points_lag_1
    """
    df = load_player_gameweeks()
    df = _add_anytime_lags(df, max_lag=1, rolling_window=None)

    feature_col = "points_lag_1"

    test_df = df[df["season"] == test_season].dropna(subset=["points", feature_col]).copy()

    y_true = test_df["points"].astype(float).to_numpy()
    y_pred = test_df[feature_col].astype(float).to_numpy()

    mae = _compute_mae(y_true, y_pred)
    return float(mae)


# ---------------------------------------------------------------------
# WITHIN-SEASON (STRICT) LINEAR MODEL EVALUATION (ALTERNATIVE)
# ---------------------------------------------------------------------


def prepare_gw_lag_dataset(
    df: pd.DataFrame,
    test_season: str = "2022/23",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare a within-season lag dataset (no cross-season lags).
    """
    df = df.sort_values(["player_id", "season", "gameweek"]).copy()

    group_cols = ["player_id", "season"]

    df["points_lag_1"] = df.groupby(group_cols)["points"].shift(1)
    df["points_lag_2"] = df.groupby(group_cols)["points"].shift(2)
    df["points_lag_3"] = df.groupby(group_cols)["points"].shift(3)

    prev_points = df.groupby(group_cols)["points"].shift(1)
    df["points_rolling_mean_3"] = prev_points.rolling(3).mean()

    train_df = df[df["season"] != test_season].copy()
    test_df = df[df["season"] == test_season].copy()

    return train_df, test_df


def evaluate_linear_gw_model(test_season: str = "2022/23") -> float:
    """
    Evaluate a within-season linear GW model (3 lags + rolling mean).
    """
    df = load_player_gameweeks()
    train_df, test_df = prepare_gw_lag_dataset(df, test_season=test_season)

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Train or test set is empty in evaluate_linear_gw_model "
            f"(check that season {test_season!r} has enough data)."
        )

    feature_cols = ["points_lag_1", "points_lag_2", "points_lag_3", "points_rolling_mean_3"]

    X_train, y_train = _build_gw_features_and_target(train_df, feature_cols, target_col="points")
    X_test, y_test = _build_gw_features_and_target(test_df, feature_cols, target_col="points")

    reg = LinearRegression()
    reg.fit(X_train.to_numpy(dtype=float), y_train.to_numpy(dtype=float))

    y_pred = reg.predict(X_test.to_numpy(dtype=float))
    mae = _compute_mae(y_test, y_pred)
    return float(mae)


# ---------------------------------------------------------------------
# LEGACY WRAPPERS (BACKWARD COMPATIBILITY)
# ---------------------------------------------------------------------


def evaluate_linear_model(test_season: str = "2022/23") -> float:
    """
    Legacy alias kept for backward compatibility.
    """
    return evaluate_linear_gw_model(test_season=test_season)


def evaluate_gradient_boosting_model(test_season: str = "2022/23") -> float:
    """
    Legacy alias kept for backward compatibility.
    """
    return evaluate_gbm_gw_model_seasonal(test_season=test_season)
