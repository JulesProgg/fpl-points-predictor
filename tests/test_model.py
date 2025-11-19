from pathlib import Path

import pandas as pd
import pytest
import numpy as np

import src.model as model
from src.model import (
    load_data,
    predict_points,
    evaluate_model,
    train_test_split_data,
    train_test_split,
    train_position_mean_model,
    evaluate_position_mean_model,
)



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


def test_train_test_split_data_covers_all_rows():
    """
    train_test_split_data must return two DataFrames
    that cover all rows without loss or duplication
    """
    df = load_data()
    n_total = len(df)

    df_train, df_test = train_test_split_data(test_size=0.2, random_state=42)

    assert len(df_train) > 0
    assert len(df_test) > 0
    assert len(df_train) + len(df_test) == n_total

    # no index overlap between train and test
    assert set(df_train.index).isdisjoint(set(df_test.index))


def test_evaluate_model_returns_non_negative_float():
    """
    evaluate_model must return a float (MAE) >= 0
    """
    mae = evaluate_model()

    assert isinstance(mae, float)
    assert mae >= 0


def test_train_position_mean_model_builds_non_empty_dict():
    df = load_data()
    train_df, _ = train_test_split(df, test_size=0.2, random_state=0)
    model = train_position_mean_model(train_df)
    assert hasattr(model, "means_by_pos")
    assert isinstance(model.means_by_pos, dict)
    assert len(model.means_by_pos) > 0


def test_evaluate_position_mean_model_returns_positive_float():
    mae = evaluate_position_mean_model()
    assert isinstance(mae, float)
    assert mae >= 0



