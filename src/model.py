"""
Model module for the FPL Points Predictor.

We start with a very simple baseline model:
- load_data() reads the processed CSV from the data pipeline
- train_model() returns a simple mean-based model
- predict_points() uses that model to predict future gameweeks
"""

from typing import Any

import pandas as pd
import numpy as np
from src.data_pipeline import DATA_PROCESSED_DIR  # important: reuse the same folder as the pipeline

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

def evaluate_model() -> float:
    """
    Trains the SimpleMeanModel and returns an error metric (MAE)
    on the same dataset (simple baseline).
    """

    df = load_data()
    y_true = df["total_points"].to_numpy()

    model = train_model(df)

    # Predict for all observations
    y_pred = np.array(model.predict(len(y_true)))

    mae = float(np.mean(np.abs(y_true - y_pred)))
    return mae



