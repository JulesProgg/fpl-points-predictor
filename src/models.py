"""
Model module for the FPL Points Predictor – GAMEWEEK-LEVEL ONLY.

This module is STRICTLY PREDICTIVE.

It provides player-level, gameweek-by-gameweek point predictions using
lagged historical information. No evaluation, no metrics, no benchmarking
is performed here.

Scope of this module
--------------------
- Feature engineering required for prediction
- On-the-fly model training
- Point prediction for player-gameweeks

Out of scope (handled elsewhere)
--------------------------------
- Model evaluation (MAE, RMSE, etc.)
- Train/test performance comparison
- Baselines and benchmarks
- Bookmaker comparisons
- Data ingestion and pipeline logic

Implemented predictive model families
-------------------------------------
1) "Anytime" linear GW models
   - Linear regression trained on the last K player appearances
   - K ∈ {3, 5, 10}
   - Cross-season continuity is allowed

2) Seasonal GW models with fallback
   - Linear regression (seasonal lags)
   - Gradient Boosting Regressor (seasonal lags)
   - Seasonal lags are computed within (player_id, season)
   - Missing early-season lags are filled using the player’s most recent
     historical appearances from previous seasons

General design choice
---------------------
Models are trained on-the-fly when predictions are requested.
This keeps the module stateless and avoids persisting fitted models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from src import load_player_gameweeks


# ---------------------------------------------------------------------
# FEATURE ENGINEERING: LAG CONSTRUCTION
# ---------------------------------------------------------------------


def _add_anytime_lags(
    df: pd.DataFrame,
    max_lag: int,
    rolling_window: int | None = None,
) -> pd.DataFrame:
    """
    Add cross-season ("anytime") lag features for player points.

    Lags are computed across the player’s entire match history, sorted by
    (player_id, season, gameweek). This allows information from previous
    seasons to be used naturally.

    Generated features
    ------------------
    - points_lag_1 .. points_lag_{max_lag}
    - Optional rolling mean over the previous appearances:
        points_rolling_mean_{rolling_window}

    This function is used by the anytime linear GW models.
    """
    df = df.sort_values(["player_id", "season", "gameweek"]).copy()
    grouped = df.groupby("player_id")

    for lag in range(1, max_lag + 1):
        df[f"points_lag_{lag}"] = grouped["points"].shift(lag)

    if rolling_window is not None:
        df[f"points_rolling_mean_{rolling_window}"] = (
            grouped["points"].shift(1).rolling(rolling_window).mean()
        )

    return df


def _add_seasonal_lags_with_prev5(df: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    """
    Add within-season lag features with a previous-season fallback.

    Lag construction logic
    ----------------------
    - Seasonal lags are computed within (player_id, season)
    - If early-season lags are missing, they are filled using the player’s
      most recent historical appearances (cross-season continuity)

    Generated features
    ------------------
    - points_lag_1 .. points_lag_{max_lag}
    - points_lag_mean : mean of the seasonal lag features

    This function supports the seasonal linear and seasonal GBM models.
    """
    df = df.sort_values(["player_id", "season", "gameweek"]).copy()

    # Global player history (anytime lags)
    grouped_player = df.groupby("player_id")
    for lag in range(1, max_lag + 1):
        df[f"points_lag_any_{lag}"] = grouped_player["points"].shift(lag)

    # Within-season lags with fallback
    grouped_season = df.groupby(["player_id", "season"])
    for lag in range(1, max_lag + 1):
        seasonal = f"points_lag_{lag}"
        anytime = f"points_lag_any_{lag}"
        df[seasonal] = grouped_season["points"].shift(lag)
        df[seasonal] = df[seasonal].fillna(df[anytime])

    lag_cols = [f"points_lag_{k}" for k in range(1, max_lag + 1)]
    df["points_lag_mean"] = df[lag_cols].mean(axis=1)

    return df.drop(columns=[f"points_lag_any_{k}" for k in range(1, max_lag + 1)])


def _add_minute_lags(df: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    """
    Add lagged minutes-played features for contextual information.

    These features are NOT used directly by the predictive models in this module.
    They are included in the output to support downstream logic (e.g. estimating
    likelihood of playing).

    Generated features
    ------------------
    - minutes_lag_1 .. minutes_lag_{max_lag}
    """
    if "minutes" not in df.columns:
        return df

    df = df.sort_values(["player_id", "season", "gameweek"])
    group = df.groupby(["player_id", "season"], group_keys=False)

    for k in range(1, max_lag + 1):
        df[f"minutes_lag_{k}"] = group["minutes"].shift(k)

    return df


# ---------------------------------------------------------------------
# MAIN PREDICTION ENTRY POINT
# ---------------------------------------------------------------------


def predict_gw_all_players(
    model: str = "gw_seasonal_gbm",
    test_season: str = "2022/23",
    gameweek: int | None = None,
) -> pd.DataFrame:
    """
    Predict FPL points for player-gameweeks using a specified GW-level model.

    This function:
    - loads the prepared player-gameweek dataset,
    - builds the required lag features,
    - trains the chosen model on all seasons except `test_season`,
    - returns predictions for the requested season/gameweek.

    Parameters
    ----------
    model : str
        Predictive model identifier:
        - "gw_lag3", "gw_lag5", "gw_lag10"
        - "gw_seasonal_linear" / "gw_seasonal"
        - "gw_seasonal_gbm"

    test_season : str
        Season for which predictions are generated.

    gameweek : int | None
        If None, predictions are generated for all gameweeks of the season.
        If specified, predictions are restricted to a single gameweek.

    Returns
    -------
    pd.DataFrame
        Player-gameweek predictions with optional contextual minute lags.
    """
    df = load_player_gameweeks()
    df = _add_minute_lags(df, max_lag=3)

    cols_out = [
        "player_id",
        "name",
        "team",
        "position",
        "season",
        "gameweek",
        "predicted_points",
    ]

    # ------------------------------------------------------------------
    # ANYTIME LINEAR GW MODELS
    # ------------------------------------------------------------------
    if model in {"gw_lag3", "gw_lag5", "gw_lag10"}:
        max_lag = int(model.split("lag")[1])
        df = _add_anytime_lags(df, max_lag=max_lag, rolling_window=max_lag)

        feature_cols = [f"points_lag_{k}" for k in range(1, max_lag + 1)]
        feature_cols.append(f"points_rolling_mean_{max_lag}")

        train_df = df[df["season"] != test_season].dropna(subset=feature_cols + ["points"])

        X_train = (
            train_df[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        y_train = train_df["points"].astype(float).to_numpy()

        reg = LinearRegression()
        reg.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # SEASONAL LINEAR GW MODEL
    # ------------------------------------------------------------------
    elif model in {"gw_seasonal_linear", "gw_seasonal"}:
        df = _add_seasonal_lags_with_prev5(df, max_lag=5)

        feature_cols = ["points_lag_1", "points_lag_2", "points_lag_3", "points_lag_mean"]

        train_df = df[df["season"] != test_season].dropna(subset=feature_cols + ["points"])

        X_train = (
            train_df[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        y_train = train_df["points"].astype(float).to_numpy()

        reg = LinearRegression()
        reg.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # SEASONAL GBM GW MODEL
    # ------------------------------------------------------------------
    elif model == "gw_seasonal_gbm":
        df = _add_seasonal_lags_with_prev5(df, max_lag=5)

        feature_cols = ["points_lag_1", "points_lag_2", "points_lag_3", "points_lag_mean"]

        train_df = df[df["season"] != test_season].dropna(subset=feature_cols + ["points"])

        X_train = (
            train_df[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        # -----------------------------
        # BIG-IMPACT CHANGES (simple)
        # 1) log-transform target to reduce skew and "mean regression"
        # 2) squared_error to emphasize large errors (haul games)
        # 3) sample_weight to upweight high-point outcomes
        # -----------------------------
        y_points = train_df["points"].astype(float).to_numpy()
        y_train = np.log1p(y_points)

        sample_weight = 1.0 + 0.15 * y_points  # simple, stable weighting

        reg = GradientBoostingRegressor(
            random_state=42,
            n_estimators=400,        # a bit more capacity (still fast)
            learning_rate=0.05,
            max_depth=3,
            loss="squared_error",    # RMSE-oriented
        )
        reg.fit(X_train, y_train, sample_weight=sample_weight)


    else:
        raise ValueError(f"Unknown GW model: {model!r}")

    # ------------------------------------------------------------------
    # PREDICTION PHASE
    # ------------------------------------------------------------------
    if gameweek is None:
        test_df = df[df["season"] == test_season].copy()
    else:
        test_df = df[(df["season"] == test_season) & (df["gameweek"] == gameweek)].copy()

    test_df = test_df.dropna(subset=feature_cols)

    if test_df.empty:
        return pd.DataFrame(columns=cols_out)

    X_test = (
        test_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    y_pred = reg.predict(X_test)

    # If GBM: model was trained on log1p(points) -> invert transform
    if model == "gw_seasonal_gbm":
        y_pred = np.expm1(y_pred)
        y_pred = np.clip(y_pred, 0.0, 25.0)  # keep predictions in a plausible range

    test_df["predicted_points"] = y_pred


    minute_cols = [c for c in test_df.columns if c.startswith("minutes_lag_")]

    return test_df[
        cols_out + minute_cols
    ].sort_values(["season", "gameweek", "team", "name"])
