from pathlib import Path

import pandas as pd
import pytest
import numpy as np

import src.model as model
from src.model import load_data, predict_points, evaluate_model


def test_predict_points_returns_list_of_floats(monkeypatch):
    """
    predict_points(n) must return a list of n floats.
    We monkeypatch load_data and train_model so as not to depend
    on a real implementation.
    """

    # Fake load_data: returns a small dummy DataFrame
    def fake_load_data(path: Path | None = None) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "player": ["A", "B", "C"],
                "points": [5, 3, 7],
            }
        )

    # Fake model object with a predict method
    class FakeModel:
        def __init__(self, data: pd.DataFrame):
            self.data = data

        def predict(self, n_gameweeks: int) -> list[float]:
            # return a constant float for each gameweek
            return [1.0 for _ in range(n_gameweeks)]

    # Fake train_model: returns the fake model
    def fake_train_model(data: pd.DataFrame):
        return FakeModel(data)

    # Monkeypatch the real functions with the fake ones
    monkeypatch.setattr(model, "load_data", fake_load_data)
    monkeypatch.setattr(model, "train_model", fake_train_model)

    # Call the function under test
    n = 4
    preds = model.predict_points(n)

    # Assertions
    assert isinstance(preds, list)
    assert len(preds) == n
    assert all(isinstance(p, float) for p in preds)


def test_load_data_accepts_optional_path_argument(monkeypatch):
    """
    load_data(path) must call pd.read_csv(path) and return the DataFrame.
    Here we test the contract without relying on a real file.
    """

    fake_path = Path("some/path/to/data.csv")
    fake_df = pd.DataFrame({"points": [1, 2, 3]})
    called = {}

    def fake_read_csv(path):
        # We check that we read the correct path
        called["path"] = path
        return fake_df

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    result = model.load_data(fake_path)

    assert called["path"] == fake_path
    assert result is fake_df


def test_train_model_returns_simple_model_object():
    """
    train_model(data) must return a simple model object
    that stores the data and has a predict(n) method.
    """
    data = pd.DataFrame({"points": [1, 2, 3]})

    model_obj = model.train_model(data)

    assert model_obj is not None
    assert hasattr(model_obj, "data")
    assert hasattr(model_obj, "predict")

    preds = model_obj.predict(3)
    assert isinstance(preds, list)
    assert len(preds) == 3
    assert all(isinstance(p, float) for p in preds)


def evaluate_model() -> float:
    """
    Trains the SimpleMeanModel and returns an error metric (MAE)
    on the same dataset (simple baseline).
    """
    df = load_data()
    y_true = df["total_points"].to_numpy()

    model = train_model(df)

    # We make predictions for all lines
    y_pred = np.array(model.predict(len(y_true)))

    # Then the MAE is calculated
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return mae
