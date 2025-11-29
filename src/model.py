"""
Model module for the FPL Points Predictor.

We provide:
- Baseline models (position mean) using a season-based split:
    TRAIN = seasons 2022-23 + 2023-24
    TEST  = season 2024-25

- Pre-season models (linear regression, gradient boosting) using lagged features:
    For each player and season t, we use stats from season t-1 as predictors.
    TRAIN = season 2023-24  (features = stats 2022-23, target = points 2023-24)
    TEST  = season 2024-25  (features = stats 2023-24, target = points 2024-25)
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor  # CHANGEMENT: GBM au lieu de RF
from src.data_pipeline import DATA_PROCESSED_DIR
from pathlib import Path
from sklearn.linear_model import LinearRegression


DEFAULT_PROCESSED_DATA = DATA_PROCESSED_DIR / "players_all_seasons.csv"

PLAYER_GW_LAGGED_DATA = DATA_PROCESSED_DIR / "player_gameweeks_lagged.csv"


#  DATA LOADING & UTILS


def load_data(path=None) -> pd.DataFrame:
    """
    Load the dataset used for training and prediction.

    Parameters
    ----------
    path : Path | str | None
        Optional path to the processed CSV file. If None, a default
        location will be used.

    Returns
    -------
    pd.DataFrame
        The dataset with one row per player-season ready for modelling.
    """
    if path is None:
        path = DEFAULT_PROCESSED_DATA

    df = pd.read_csv(path)
    return df


def load_player_gameweeks_lagged(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load the per-player-per-gameweek dataset with lagged features.

    Returns
    -------
    pd.DataFrame
        One row per player-gameweek, with:
        - identifiers (id, name, team, position)
        - 'season', 'gameweek'
        - 'points' (target for this GW)
        - lagged features from previous gameweeks
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


def season_train_test_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the full dataset by season for baseline models:

    TRAIN = seasons 2022-23 + 2023-24
    TEST  = season 2024-25
    """
    df = load_data()

    train_df = df[df["season"].isin(["2022-23", "2023-24"])].copy()
    test_df = df[df["season"] == "2024-25"].copy()

    return train_df, test_df


#  FEATURE LISTS (used for lagged models)

LINEAR_FEATURE_COLUMNS: list[str] = [
    "minutes",
    "goals_scored",
    "assists",
    "expected_goals",
    "expected_assists",
    "ict_index",
]

GBM_FEATURE_COLUMNS: list[str] = [
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "yellow_cards",
    "red_cards",
    "bonus",
    "bps",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "influence",
    "creativity",
    "threat",
    "ict_index",
]
# GAMEWEEK-LEVEL FEATURES (automatically inferred)

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


def build_gw_features_and_target(
    df: pd.DataFrame,
    target_col: str = "points",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target y for the gameweek-level model.

    Features = all numeric columns except:
        - identifiers
        - season, gameweek
        - the target
        - any column containing 'point' (to avoid leakage)
    """
    numeric_cols = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ]

    # Strict exclusion of any column containing 'point'
    feature_cols = [
        c
        for c in numeric_cols
        if (
            c not in NON_FEATURE_COLS_GW
            and c != target_col
            and "point" not in c.lower()   # anti-leakage
        )
    ]

    if not feature_cols:
        raise ValueError("No feature columns found for the gameweek model.")

    clean_df = df.dropna(subset=feature_cols + [target_col]).copy()

    X = (
        clean_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    y = clean_df[target_col].astype(float)

    return X, y



#  LAGGED DATASET (X_{t-1} -> y_t) FOR PRE-SEASON MODELS

def prepare_lagged_dataset(feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a dataset with lagged features (X_{t-1}), following the
    time-series methodology (no leakage, strict past→future split).

    For each player (id), seasons are sorted chronologically.
    For season t, we use stats from season t-1 as predictors.
    The target variable is total_points_t.

    Temporal split:
    - TRAIN = season 2023-24  (features = stats from 2022-23)
    - TEST  = season 2024-25  (features = stats from 2023-24)
    """
    df = load_data().copy()
    df = df.sort_values(["id", "season"])

    lagged = df.copy()
    for col in feature_cols:
        lagged[col + "_prev"] = lagged.groupby("id")[col].shift(1)

    lagged["target"] = lagged["total_points"]

    needed_cols = [c + "_prev" for c in feature_cols] + ["target"]
    lagged = lagged.dropna(subset=needed_cols)

    train_df = lagged[lagged["season"] == "2023-24"].copy()
    test_df = lagged[lagged["season"] == "2024-25"].copy()

    return train_df, test_df


#  BASELINE MODELS


class PositionMeanModel:
    """
    Baseline model:
    - average total points per position on the training seasons
    - fallback to global average if position is unknown
    """

    def __init__(self, means_by_pos: dict[str, float], global_mean: float):
        self.means_by_pos = means_by_pos
        self.global_mean = global_mean

    def predict_for_position(self, position: str) -> float:
        return float(self.means_by_pos.get(position, self.global_mean))


def train_position_mean_model(train_df: pd.DataFrame) -> PositionMeanModel:
    """
    Train a model that predicts the average points per position.
    """
    grouped = train_df.groupby("position")["total_points"].mean()
    means_by_pos = grouped.to_dict()
    global_mean = float(train_df["total_points"].mean())
    return PositionMeanModel(means_by_pos, global_mean)


def evaluate_position_mean_model(
    test_size: float = 0.2,
    random_state: int = 42,
) -> float:
    """
    Evaluate PositionMeanModel on a season-based split:

    TRAIN = seasons 2022-23 + 2023-24
    TEST  = season 2024-25

    Parameters test_size and random_state are kept for API compatibility,
    but not used (no random splitting).
    """
    train_df, test_df = season_train_test_split()

    model = train_position_mean_model(train_df)

    y_true = test_df["total_points"].to_numpy()
    positions = test_df["position"].tolist()
    y_pred = np.array([model.predict_for_position(pos) for pos in positions])

    mae = _compute_mae(y_true, y_pred)
    return mae


#  LINEAR REGRESSION MODEL (PRE-SEASON, LAGGED FEATURES)

class LinearRegressionModel:
    """
    Multivariate linear regression model:
    y = intercept + sum_j coef_j * x_j

    - coef_: feature coefficients
    - intercept_: bias term
    - feature_names: the columns used for training
    """

    def __init__(self, coef_: np.ndarray, intercept_: float, feature_names: list[str]):
        self.coef_ = np.asarray(coef_, dtype=float)
        self.intercept_ = float(intercept_)
        self.feature_names = list(feature_names)

    def predict_from_array(self, X: np.ndarray) -> np.ndarray:
        """
        Take an X matrix (n_samples, n_features) and return the predictions.
        """
        return X @ self.coef_ + self.intercept_

    def predict_for_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract the features in df according to self.feature_names and predict y.
        NaN and +/-inf in the features are handled safely.
        """
        features = (
            df[self.feature_names]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        X = features.to_numpy(dtype=float)
        return self.predict_from_array(X)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Wrapper so the model matches the API of GradientBoostingModel.
        """
        return self.predict_for_dataframe(df)


def train_linear_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> LinearRegressionModel:
    """
    Train a linear regression model using lagged predictors (X_{t-1}),
    following the OLS approach:
        θ = argmin ||Xθ - y||²

    train_df must contain:
    - '<feature>_prev' columns = lagged predictors
    - 'target'                 = response (total_points_t)
    """
    lag_feature_cols = [c + "_prev" for c in feature_cols if c + "_prev" in train_df.columns]
    cols_needed = lag_feature_cols + ["target"]

    df = train_df.dropna(subset=cols_needed)
    if df.empty:
        raise ValueError("No training data available after dropping NaNs.")

    X = df[lag_feature_cols].to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=float)

    X_aug = np.c_[np.ones(X.shape[0]), X]

    theta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)

    intercept = float(theta[0])
    coef = theta[1:]

    return LinearRegressionModel(
        coef_=coef,
        intercept_=intercept,
        feature_names=lag_feature_cols,
    )


def evaluate_linear_model(
    test_size: float = 0.2,
    random_state: int = 42,
    feature_cols: list[str] | None = None,
) -> float:
    """
    Pre-season evaluation for the LinearRegressionModel using a temporal split:

    TRAIN:
        season = 2023-24
        predictors: stats from 2022-23 (columns *_prev)
        target: total_points_2023_24

    TEST:
        season = 2024-25
        predictors: stats from 2023-24 (columns *_prev)
        target: total_points_2024_25

    No shuffling, no leakage. Parameters test_size and random_state are
    kept only for API compatibility.
    """
    if feature_cols is None:
        feature_cols = LINEAR_FEATURE_COLUMNS

    train_df, test_df = prepare_lagged_dataset(feature_cols)

    model = train_linear_model(train_df, feature_cols=feature_cols)

    X_test_df = test_df[model.feature_names].copy()
    X_test_df = X_test_df.fillna(0.0)

    y_true = test_df["target"].to_numpy(dtype=float)
    y_pred = model.predict_for_dataframe(X_test_df)

    mae = _compute_mae(y_true, y_pred)
    return mae


#  GRADIENT BOOSTING MODEL (PRE-SEASON, LAGGED FEATURES)

class GradientBoostingModel:
    """
    Wrapper around sklearn's GradientBoostingRegressor.

    We keep a similar interface as LinearRegressionModel:
    - feature_columns: columns used as predictors
    """

    def __init__(self, regressor: GradientBoostingRegressor, feature_columns: list[str]):
        self.regressor = regressor
        self.feature_columns = feature_columns

    def predict_for_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features in df and predict y.
        NaN and +/-inf in the features are handled safely.
        """
        features = (
            df[self.feature_columns]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        X = features.to_numpy(dtype=float)
        return self.regressor.predict(X)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Wrapper for consistency with other model classes.
        """
        return self.predict_for_dataframe(df)


def train_gradient_boosting_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> GradientBoostingModel:
    """
    Train a GradientBoostingRegressor on lagged features.

    train_df must contain:
    - '<feature>_prev' columns
    - 'target' = total_points_t
    """
    lag_feature_cols = [c + "_prev" for c in feature_cols if c + "_prev" in train_df.columns]
    cols_needed = lag_feature_cols + ["target"]

    df = train_df.dropna(subset=cols_needed)
    if df.empty:
        raise ValueError("No training data available for Gradient Boosting after dropping NaNs.")

    X_train = df[lag_feature_cols].to_numpy(dtype=float)
    y_train = df["target"].to_numpy(dtype=float)

    gbm = GradientBoostingRegressor(
        random_state=42,
        # Tu peux ajuster ces hyperparamètres si tu veux:
        # n_estimators=200,
        # learning_rate=0.1,
        # max_depth=3,
    )
    gbm.fit(X_train, y_train)

    return GradientBoostingModel(regressor=gbm, feature_columns=lag_feature_cols)


def evaluate_gradient_boosting_model(
    test_size: float = 0.2,
    random_state: int = 42,
) -> float:
    """
    Pre-season evaluation for the GradientBoostingModel using temporal split:

    TRAIN:
        season = 2023-24
        predictors: stats from 2022-23 (columns *_prev)
        target: total_points_2023_24

    TEST:
        season = 2024-25
        predictors: stats from 2023-24 (columns *_prev)
        target: total_points_2024_25

    No shuffling, no leakage. Parameters test_size and random_state are
    kept only for API compatibility.
    """
    train_df, test_df = prepare_lagged_dataset(GBM_FEATURE_COLUMNS)

    model = train_gradient_boosting_model(train_df, feature_cols=GBM_FEATURE_COLUMNS)

    y_true = test_df["target"].to_numpy(dtype=float)
    y_pred = model.predict_for_dataframe(test_df)

    mae = _compute_mae(y_true, y_pred)
    return float(mae)

def evaluate_linear_gw_model(test_season: str = "2023/24") -> float:
    """
    Evaluate a linear regression model on gameweek-level lagged data.

    TRAIN = all seasons except `test_season`
    TEST  = only `test_season`

    Target: 'points' (per player-gameweek).
    Features: all numeric lagged columns, excluding identifiers and 'points'.
    """
    df = load_player_gameweeks_lagged()

    # Sanity check: does the test season exist?
    available_seasons = sorted(df["season"].unique())
    if test_season not in available_seasons:
        raise ValueError(
            f"Season {test_season!r} not found in gameweek data. "
            f"Available seasons: {available_seasons}"
        )

    train_df = df[df["season"] != test_season].copy()
    test_df = df[df["season"] == test_season].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Train or test set is empty in evaluate_linear_gw_model "
            f"(check that season {test_season!r} has enough data)."
        )

    # Build features/targets
    X_train, y_train = build_gw_features_and_target(train_df, target_col="points")
    X_test, y_test = build_gw_features_and_target(test_df, target_col="points")

    # Train linear regression
    reg = LinearRegression()
    reg.fit(X_train.to_numpy(dtype=float), y_train.to_numpy(dtype=float))

    # Predict on test set
    y_pred = reg.predict(X_test.to_numpy(dtype=float))

    mae = _compute_mae(y_test, y_pred)
    return float(mae)



#  CLI-FACING PREDICTION FUNCTION

def predict_points(n_gameweeks: int = 5, model: str = "linear") -> list[float]:
    """
    Predict FPL points for a given number of future gameweeks.

    Parameters
    ----------
    n_gameweeks : int
        Number of future gameweeks to predict.
    model : str
        Which model to use:
        - "position" -> PositionMeanModel (season-based split)
        - "linear"   -> LinearRegressionModel (pre-season, lagged)
        - "gbm"      -> GradientBoostingModel (pre-season, lagged)

    For linear and gbm in this CLI function, we:
    - train on the lagged TRAIN set (season 2023-24),
    - compute the average predicted target,
    - repeat this value for n_gameweeks.
    This is a simple way to expose season-level models via a GW-level CLI.
    """
    if model == "position":
        train_df, _ = season_train_test_split()
        model_obj = train_position_mean_model(train_df)
        # Use the average of position-specific means as a simple GW-level proxy
        avg_points = float(np.mean(list(model_obj.means_by_pos.values())))
        return [avg_points] * n_gameweeks

    elif model == "linear":
        train_df, _ = prepare_lagged_dataset(LINEAR_FEATURE_COLUMNS)
        model_obj = train_linear_model(train_df, feature_cols=LINEAR_FEATURE_COLUMNS)
        per_player_preds = model_obj.predict_for_dataframe(train_df)
        avg_points = float(np.mean(per_player_preds))
        return [avg_points] * n_gameweeks

    elif model == "gbm":
        train_df, _ = prepare_lagged_dataset(GBM_FEATURE_COLUMNS)
        model_obj = train_gradient_boosting_model(train_df, feature_cols=GBM_FEATURE_COLUMNS)
        per_player_preds = model_obj.predict_for_dataframe(train_df)
        avg_points = float(np.mean(per_player_preds))
        return [avg_points] * n_gameweeks

    else:
        raise ValueError(
            f"Unknown model: {model!r}. "
            "Expected 'position', 'linear', or 'gbm'."
        )
