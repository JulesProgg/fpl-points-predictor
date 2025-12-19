"""
Model module for the FPL Points Predictor – GAMEWEEK-LEVEL ONLY.

We provide gameweek-level models (per player per gameweek) using lagged points:

- Baseline:
    • GW Naive baseline: points_t = points_{t-1} (persistence)

- Linear models with "anytime" lags:
    • Last 3 matches  (anytime, across seasons)
    • Last 5 matches  (anytime, across seasons)
    • Last 10 matches (anytime, across seasons)

- Seasonal lag models:
    • Linear GW model (seasonal, except GW1)
    • Gradient Boosting GW model (seasonal, except GW1)

Lag logic:
    - Within a season: lags come only from the same season.
    - For GW1 of a season: lags come from the last few gameweeks
      of the previous season (if available).

Default test season:
    - "2022/23" (last season in the project scope 2016/17 → 2022/23)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression



from src.data_pipeline import DATA_PROCESSED_DIR

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

# Gameweek-level data (no lags in file; lags are computed here)
# We now directly use the clean GW file produced by data_pipeline.py
PLAYER_GW_LAGGED_DATA = DATA_PROCESSED_DIR / "player_gameweeks.csv"

# ---------------------------------------------------------------------
# DATA LOADING & UTILS
# ---------------------------------------------------------------------


def load_player_gameweeks_lagged(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load the per-player-per-gameweek dataset (one row per player-gameweek).

    The CSV should contain:
    - identifiers (player_id, id, name, team, position)
    - 'season', 'gameweek'
    - 'points' (target for this GW)
    """
    if path is None:
        path = PLAYER_GW_LAGGED_DATA

    df = pd.read_csv(path)
    return df


def _compute_mae(y_true, y_pred) -> float:
    """
    Compute Mean Absolute Error (MAE), ignoring pairs (y_true, y_pred)
    that contain NaN.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        raise ValueError("No valid (y_true, y_pred) pairs to compute MAE (all NaN).")

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


# Columns that are NOT features (identifiers / metadata / target)
NON_FEATURE_COLS_GW: set[str] = {
    "id",
    "player_id",
    "name",
    "team",
    "position",
    "season",
    "gameweek",
    "points",  # target
}

# ---------------------------------------------------------------------
# LAG FEATURE BUILDERS (ANYTIME & SEASONAL)
# ---------------------------------------------------------------------


def _add_anytime_lags(
    df: pd.DataFrame,
    max_lag: int,
    rolling_window: int | None = None,
) -> pd.DataFrame:
    """
    Add "anytime" lag features for points, across all seasons.

    For each player:
        points_lag_1  = points at previous gameweek (whatever the season)
        points_lag_2  = points two games ago
        ...
        points_lag_k  = points k games ago

    Optionally adds a rolling mean over the last `rolling_window` matches.
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
    Version rapide :
    - Crée d'abord des lags "anytime" (toute l’histoire du joueur).
    - Crée ensuite des lags "season-only".
    - Pour les débuts de saison, remplace les NaN saisonniers par les lags anytime.
    - Vectorisé, aucun loop Python => extrêmement rapide.
    """
    df = df.sort_values(["player_id", "season", "gameweek"]).copy()

    # ANYTIME lags (global history)
    grouped_player = df.groupby("player_id")
    for lag in range(1, max_lag + 1):
        df[f"points_lag_any_{lag}"] = grouped_player["points"].shift(lag)

    # SEASON-only lags (intrablock)
    grouped_season = df.groupby(["player_id", "season"])
    for lag in range(1, max_lag + 1):
        seasonal = f"points_lag_{lag}"
        anytime = f"points_lag_any_{lag}"
        df[seasonal] = grouped_season["points"].shift(lag)

        # BEGIN OF SEASON = fallback vers anytime
        df[seasonal] = df[seasonal].fillna(df[anytime])

    # average lag
    lag_cols = [f"points_lag_{k}" for k in range(1, max_lag + 1)]
    df["points_lag_mean"] = df[lag_cols].mean(axis=1)

    # clean intermediate columns
    df = df.drop(columns=[f"points_lag_any_{k}" for k in range(1, max_lag + 1)])

    return df


def _add_minute_lags(df: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    """
    Add lagged minutes columns per player and season:
    minutes_lag_1, minutes_lag_2, ..., minutes_lag_{max_lag}.
    """
    if "minutes" not in df.columns:
        return df  # do nothing

    df = df.sort_values(["player_id", "season", "gameweek"])
    group = df.groupby(["player_id", "season"], group_keys=False)

    for k in range(1, max_lag + 1):
        df[f"minutes_lag_{k}"] = group["minutes"].shift(k)

    return df






def _build_gw_features_and_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "points",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build X, y for gameweek-level models, given a list of feature columns.
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


# ---------------------------------------------------------------------
# "ANYTIME" LINEAR GW MODELS (LAST K MATCHES)
# ---------------------------------------------------------------------


def _evaluate_anytime_linear_gw(
    max_lag: int,
    test_season: str = "2022/23",
) -> float:
    """
    Helper to evaluate a linear GW model using "anytime" lags over the
    last `max_lag` matches of each player, regardless of season.

    Default test_season = "2022/23" (last season in project scope).
    """
    df = load_player_gameweeks_lagged()
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


def evaluate_linear_gw_model_lag3(
    test_season: str = "2022/23",
) -> float:
    """
    Linear GW model using the last 3 matches of each player
    (anytime lags across all seasons).
    """
    return _evaluate_anytime_linear_gw(max_lag=3, test_season=test_season)


def evaluate_linear_gw_model_lag5(
    test_season: str = "2022/23",
) -> float:
    """
    Linear GW model using the last 5 matches of each player
    (anytime lags across all seasons).
    """
    return _evaluate_anytime_linear_gw(max_lag=5, test_season=test_season)


def evaluate_linear_gw_model_lag10(
    test_season: str = "2022/23",
) -> float:
    """
    Linear GW model using the last 10 matches of each player
    (anytime lags across all seasons).
    """
    return _evaluate_anytime_linear_gw(max_lag=10, test_season=test_season)


# ---------------------------------------------------------------------
# SEASONAL LINEAR & GBM GW MODELS
# ---------------------------------------------------------------------


def evaluate_linear_gw_model_seasonal(
    test_season: str = "2022/23",
) -> float:
    """
    Linear GW model that:
    - within a season: only uses lags from the same season,
    - for the first GW of a season: uses the last 5 gameweeks
      of the previous season.

    Features:
        points_lag_1 .. points_lag_5
        points_lag_mean
    """
    df = load_player_gameweeks_lagged()
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


def evaluate_gbm_gw_model_seasonal(
    test_season: str = "2022/23",
) -> float:
    """
    Gameweek-level GradientBoosting model:

    - within a season: only uses lags from the same season,
    - for the first GW of a season: uses the last 5 gameweeks
      of the previous season (via _add_seasonal_lags_with_prev5).

    Features:
        points_lag_1 .. points_lag_5
        points_lag_mean
    """
    df = load_player_gameweeks_lagged()
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


# ---------------------------------------------------------------------
# PREDICT ALL PLAYERS FOR A GIVEN SEASON / GAMEWEEK
# ---------------------------------------------------------------------


def predict_gw_all_players(
    model: str = "gw_seasonal_gbm",
    test_season: str = "2022/23",
    gameweek: int | None = None,
) -> pd.DataFrame:
    """
    Predict FPL points for EACH player-gameweek using a chosen GW-level model.

    Parameters
    ----------
    model : str
        One of:
        - "gw_lag3"
        - "gw_lag5"
        - "gw_lag10"
        - "gw_seasonal_linear"
        - "gw_seasonal_gbm"
    test_season : str
        Season for which we want predictions (e.g. "2022/23").
    gameweek : int | None
        If None: predict for ALL gameweeks of test_season.
        If an int (e.g. 15): predict only for that GW.

    Returns
    -------
    pd.DataFrame with columns:
        player_id, name, team, position, season, gameweek, predicted_points
    """
    df = load_player_gameweeks_lagged()

    # Add minute lags (to approximate the probability of playing)
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
    # ANYTIME LAG MODELS
    # ------------------------------------------------------------------
    if model in {"gw_lag3", "gw_lag5", "gw_lag10"}:
        max_lag = int(model.split("lag")[1])

        # Add lags across all seasons
        df = _add_anytime_lags(df, max_lag=max_lag, rolling_window=max_lag)

        feature_cols = [f"points_lag_{k}" for k in range(1, max_lag + 1)]
        feature_cols.append(f"points_rolling_mean_{max_lag}")

        # Train set: all seasons except test_season
        train_df = df[df["season"] != test_season].copy()
        train_df = train_df.dropna(subset=feature_cols + ["points"]).copy()

        if train_df.empty:
            raise ValueError(
                f"No training data available for GW model {model!r} "
                f"when excluding season {test_season!r}."
            )

        X_train = (
            train_df[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        y_train = train_df["points"].astype(float).to_numpy()

        reg = LinearRegression()
        reg.fit(X_train, y_train)

        # Test set
        if gameweek is None:
            test_df = df[df["season"].eq(test_season)].copy()
        else:
            test_df = df[
                df["season"].eq(test_season) & df["gameweek"].eq(gameweek)
            ].copy()

        test_df = test_df.dropna(subset=feature_cols).copy()

        if test_df.empty:
            return pd.DataFrame(columns=cols_out)

        X_test = (
            test_df[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        y_pred = reg.predict(X_test)

    # ------------------------------------------------------------------
    # SEASONAL LINEAR MODEL
    # ------------------------------------------------------------------
    elif model in {"gw_seasonal_linear", "gw_seasonal"}:
        df = _add_seasonal_lags_with_prev5(df, max_lag=5)

        feature_cols = [f"points_lag_{k}" for k in range(1, 4)]
        feature_cols.append("points_lag_mean")

        train_df = df[df["season"] != test_season].copy()
        train_df = train_df.dropna(subset=feature_cols + ["points"]).copy()

        if train_df.empty:
            raise ValueError(
                f"No training data available for GW model {model!r} "
                f"when excluding season {test_season!r}."
            )

        X_train = (
            train_df[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        y_train = train_df["points"].astype(float).to_numpy()

        reg = LinearRegression()
        reg.fit(X_train, y_train)

        if gameweek is None:
            test_df = df[df["season"].eq(test_season)].copy()
        else:
            test_df = df[
                df["season"].eq(test_season) & df["gameweek"].eq(gameweek)
            ].copy()

        test_df = test_df.dropna(subset=feature_cols).copy()

        if test_df.empty:
            return pd.DataFrame(columns=cols_out)

        X_test = (
            test_df[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        y_pred = reg.predict(X_test)

    # ------------------------------------------------------------------
    # SEASONAL GBM MODEL
    # ------------------------------------------------------------------
    elif model == "gw_seasonal_gbm":
        df = _add_seasonal_lags_with_prev5(df, max_lag=5)

        feature_cols = [f"points_lag_{k}" for k in range(1, 4)]
        feature_cols.append("points_lag_mean")

        train_df = df[df["season"] != test_season].copy()
        train_df = train_df.dropna(subset=feature_cols + ["points"]).copy()

        if train_df.empty:
            raise ValueError(
                f"No training data available for GW model {model!r} "
                f"when excluding season {test_season!r}."
            )

        X_train = (
            train_df[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        y_train = train_df["points"].astype(float).to_numpy()

        gbm = GradientBoostingRegressor(
            random_state=42,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
        )
        gbm.fit(X_train, y_train)

        if gameweek is None:
            test_df = df[df["season"].eq(test_season)].copy()
        else:
            test_df = df[
                df["season"].eq(test_season) & df["gameweek"].eq(gameweek)
            ].copy()

        test_df = test_df.dropna(subset=feature_cols).copy()

        if test_df.empty:
            return pd.DataFrame(columns=cols_out)

        X_test = (
            test_df[feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        y_pred = gbm.predict(X_test)

    else:
        raise ValueError(
            f"Unknown GW model: {model!r}. "
            "Expected one of: gw_lag3, gw_lag5, gw_lag10, "
            "gw_seasonal_linear, gw_seasonal_gbm."
        )

    test_df["predicted_points"] = y_pred

    # Also allow for any delays of several minutes, if they occur
    minute_cols = [c for c in test_df.columns if c.startswith("minutes_lag_")]

    cols_out = [
        "player_id",
        "name",
        "team",
        "position",
        "season",
        "gameweek",
        "predicted_points",
    ]
    cols_out_extended = cols_out + minute_cols

    return test_df[cols_out_extended].sort_values(
        ["season", "gameweek", "team", "name"]
    )





# ---------------------------------------------------------------------
# SIMPLE GW-LEVEL BASELINE
# ---------------------------------------------------------------------


def evaluate_gw_baseline_lag1(test_season: str = "2022/23") -> float:
    """
    Naive gameweek-level baseline:

    Predict points_t = points_{t-1} (points_lag_1) for each player.

    This is the standard persistence baseline:
        "next gameweek = last gameweek".
    """
    df = load_player_gameweeks_lagged()

    # Ensures that points_lag_1 exists (anytime lags across the entire history)
    df = _add_anytime_lags(df, max_lag=1, rolling_window=None)

    feature_col = "points_lag_1"

    test_df = (
        df[df["season"] == test_season]
        .dropna(subset=["points", feature_col])
        .copy()
    )

    y_true = test_df["points"].astype(float).to_numpy()
    y_pred = test_df[feature_col].astype(float).to_numpy()

    mae = _compute_mae(y_true, y_pred)
    return float(mae)


# ---------------------------------------------------------------------
# GW-LAG DATASET FOR SIMPLE LINEAR MODEL
# ---------------------------------------------------------------------


def prepare_gw_lag_dataset(
    df: pd.DataFrame,
    test_season: str = "2022/23",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a predictive gameweek-level dataset with clean lagged features.

    For each player and season, rows are sorted by gameweek and we create:
        - points_lag_1  = points at GW-1
        - points_lag_2  = points at GW-2
        - points_lag_3  = points at GW-3
        - points_rolling_mean_3 = mean(points at GW-1, GW-2, GW-3)

    Then we split by season:
        TRAIN = all seasons except test_season
        TEST  = only test_season
    """
    df = df.sort_values(["player_id", "season", "gameweek"]).copy()

    group_cols = ["player_id", "season"]

    # Lag features based only on past gameweeks (no leakage)
    df["points_lag_1"] = df.groupby(group_cols)["points"].shift(1)
    df["points_lag_2"] = df.groupby(group_cols)["points"].shift(2)
    df["points_lag_3"] = df.groupby(group_cols)["points"].shift(3)

    # Rolling mean over the last 3 past gameweeks
    prev_points = df.groupby(group_cols)["points"].shift(1)
    df["points_rolling_mean_3"] = prev_points.rolling(3).mean()

    train_df = df[df["season"] != test_season].copy()
    test_df = df[df["season"] == test_season].copy()

    return train_df, test_df


def evaluate_linear_gw_model(test_season: str = "2022/23") -> float:
    """
    Evaluate a linear regression model on gameweek-level lagged data.

    Predictive setup (no data leakage):

        TRAIN = all seasons except `test_season`
        TEST  = only `test_season`

    Target: 'points' (per player-gameweek).
    Features: lagged points from previous gameweeks (points_lag_1/2/3
    and points_rolling_mean_3).
    """
    df = load_player_gameweeks_lagged()

    # Build clean lag features and split by season
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
# BACKWARD-COMPATIBLE WRAPPERS (KEEP OLD NAMES, NOW GW-BASED)
# ---------------------------------------------------------------------


def evaluate_linear_model(
    test_season: str = "2022/23",
) -> float:
    """
    Backward-compatible wrapper.

    Historically: season-level pre-season LinearRegressionModel.
    Now: alias to evaluate_linear_gw_model (gameweek-level).
    """
    return evaluate_linear_gw_model(test_season=test_season)


def evaluate_gradient_boosting_model(
    test_season: str = "2022/23",
) -> float:
    """
    Backward-compatible wrapper.

    Historically: season-level pre-season GradientBoostingModel.
    Now: alias to evaluate_gbm_gw_model_seasonal (gameweek-level).
    """
    return evaluate_gbm_gw_model_seasonal(test_season=test_season)


