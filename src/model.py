"""
Model module for the FPL Points Predictor.

We start with a very simple baseline model:
- load_data() reads the processed CSV from the data pipeline
- train_model() returns a simple mean-based model
- predict_points() uses that model to predict future gameweeks
"""


import pandas as pd
import numpy as np

from src.data_pipeline import DATA_PROCESSED_DIR  # important: reuse the same folder as the pipeline
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error



# Use the SAME filename as in data_pipeline.run_pipeline()
DEFAULT_PROCESSED_DATA = DATA_PROCESSED_DIR / "players_all_seasons.csv"


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
        The dataset with one row per player-gameweek ready for modelling.
    """
    if path is None:
        path = DEFAULT_PROCESSED_DATA

    df = pd.read_csv(path)
    return df

def _compute_mae(y_true, y_pred) -> float:
    """
    Compute Mean Absolute Error (MAE) ignoring pairs (y_true, y_pred)
    that contain NaN
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Only keep indices where neither y_true nor y_pred are NaN
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)

    if not np.any(mask):
        raise ValueError("No valid (y_true, y_pred) pairs to compute MAE (all NaN).")

    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))




def train_test_split_data(test_size: float = 0.2, random_state: int = 42):
    """
    Split the full processed dataset into train and test DataFrames.
    """
    df = load_data()

    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    return df_train, df_test



class SimpleMeanModel:
    """
    Very simple baseline model: predicts the mean of historical points
    for each future gameweek.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def predict(self, n_gameweeks: int) -> list[float]:
        # Use the target produced by your pipeline: 'total_points'
        if "total_points" in self.data.columns:
            baseline = float(self.data["total_points"].mean())
        elif "points" in self.data.columns:
            baseline = float(self.data["points"].mean())
        else:
            baseline = 0.0
        return [baseline for _ in range(n_gameweeks)]

class PositionMeanModel:
    """
    Slightly more advanced baseline:
- average total points per position
- if a position is unknown, we use the overall average
    """
    def __init__(self, means_by_pos: dict[str, float], global_mean: float):
        self.means_by_pos = means_by_pos
        self.global_mean = global_mean

    def predict_for_position(self, position: str) -> float:
        return float(self.means_by_pos.get(position, self.global_mean))

def train_position_mean_model(train_df: pd.DataFrame) -> PositionMeanModel:
    """
    Train a model that predicts the average points per position
    """
    grouped = train_df.groupby("position")["total_points"].mean()
    means_by_pos = grouped.to_dict()
    global_mean = float(train_df["total_points"].mean())
    return PositionMeanModel(means_by_pos, global_mean)

def evaluate_position_mean_model(test_size: float = 0.2, random_state: int = 42) -> float:
    """
    Evaluates PositionMeanModel on a separate test set 
    then returns the MAE on this test set
    """
    df = load_data()
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    model = train_position_mean_model(train_df)

    y_true = test_df["total_points"].to_numpy()
    positions = test_df["position"].tolist()

    y_pred = np.array([model.predict_for_position(pos) for pos in positions])
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return mae



def train_model(data: pd.DataFrame) -> Any:
    """
    Train a prediction model on the provided dataset.

    For now, we return a very simple baseline model that always predicts
    the mean historical points.
    """
    return SimpleMeanModel(data)


def predict_points(n_gameweeks: int = 5) -> list[float]:
    """
    Predict FPL points for a given number of future gameweeks.

    Steps:
    - load the dataset
    - train the baseline model
    - return its predictions
    """
    data = load_data()
    model_obj = train_model(data)
    return model_obj.predict(n_gameweeks)

import numpy as np  # en haut du fichier si pas déjà importé

def evaluate_model() -> float:
    """
    Trains the SimpleMeanModel on TRAIN data
    and evaluates it on TEST data using MAE.
    """
    df_train, df_test = train_test_split_data()
    model = train_model(df_train)

    y_true = df_test["total_points"].to_numpy()
    y_pred = np.array(model.predict(len(y_true)))

    # MAE
    mae = _compute_mae(y_true, y_pred)
    return mae


LINEAR_FEATURE_COLUMNS: list[str] = [
    "minutes",
    "goals_scored",
    "assists",
    "expected_goals",               # ex-xG
    "expected_assists",            # ex-xA
    "expected_goal_involvements",  # xG + xA combiné
    "ict_index",
]
class LinearRegressionModel:
    """
    Multivariate linear regression model:
    y = intercept + sum_j coef_j * x_j

    - coef_: feature coefficients
    - intercept_: bias
    - feature_names: order of columns used for training
    """

    def __init__(self, coef_: np.ndarray, intercept_: float, feature_names: list[str]):
        self.coef_ = np.asarray(coef_, dtype=float)
        self.intercept_ = float(intercept_)
        self.feature_names = list(feature_names)

    def predict_from_array(self, X: np.ndarray) -> np.ndarray:
        """
        Takes an X matrix (n_samples, n_features) and returns the predictions
        """
        return X @ self.coef_ + self.intercept_

    def predict_for_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extracts the features in df according to self.feature_names and predicts y
        """
        X = df[self.feature_names].to_numpy(dtype=float)
        return self.predict_from_array(X)

def train_linear_model(
    train_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> LinearRegressionModel:
    """
    Trains a linear model (least squares) on the specified features.
    Rows containing NaN in the features OR the target are removed
    """
    if feature_cols is None:
        feature_cols = LINEAR_FEATURE_COLUMNS

    # mandatory columns = features + target
    cols_needed = list(feature_cols) + ["total_points"]

    # all lines containing NaN in these columns are deleted
    df = train_df.dropna(subset=cols_needed)

    # security if everything is NaN
    if df.empty:
        raise ValueError(
            "No data left after dropping NaNs. "
            "Check that your feature columns exist and are not all NaN."
        )

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["total_points"].to_numpy(dtype=float)

    # Add column “1” for the intercept
    X_aug = np.c_[np.ones(X.shape[0]), X]

    # Least squares
    theta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)

    intercept = float(theta[0])
    coef = theta[1:]

    return LinearRegressionModel(
        coef_=coef,
        intercept_=intercept,
        feature_names=feature_cols
    )




def evaluate_linear_model(
    test_size: float = 0.2,
    random_state: int = 42,
    feature_cols: list[str] | None = None,
) -> float:
    """
    Split TRAIN/TEST, trains a LinearRegressionModel
    and returns the MAE on the TEST set
    """
    train_df, test_df = train_test_split_data(
        test_size=test_size,
        random_state=random_state,
    )

    model = train_linear_model(train_df, feature_cols=feature_cols)

    y_true = test_df["total_points"].to_numpy(dtype=float)

    # test features (NaN -> 0.0 to avoid NaN in predictions)
    X_test_df = test_df[model.feature_names].copy()
    X_test_df = X_test_df.fillna(0.0)
    X_test = X_test_df.to_numpy(dtype=float)

    y_pred = model.predict_from_array(X_test)

    mae = _compute_mae(y_true, y_pred)
    return mae
