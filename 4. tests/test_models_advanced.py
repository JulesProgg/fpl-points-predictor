from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.model as model


def make_fake_fpl_df() -> pd.DataFrame:
    """
    Small dummy DataFrame compatible with:
    - LINEAR_FEATURE_COLUMNS
    - _get_rf_feature_columns
    """
    data = {
        # cible
        "total_points": [10, 20, 30, 40, 50],
        "position": ["MID", "MID", "FWD", "DEF", "GK"],
        # features linéaires + RF
        "minutes": [1500, 2000, 1800, 2200, 3400],
        "goals_scored": [5, 7, 15, 2, 0],
        "assists": [6, 8, 3, 2, 0],
        "clean_sheets": [5, 4, 0, 12, 18],
        "goals_conceded": [30, 25, 40, 20, 10],
        "saves": [0, 0, 0, 0, 120],
        "yellow_cards": [3, 5, 4, 1, 0],
        "red_cards": [0, 0, 1, 0, 0],
        "bonus": [10, 15, 25, 5, 2],
        "bps": [200, 250, 300, 180, 150],
        "expected_goals": [4.5, 6.5, 14.0, 1.5, 0.0],
        "expected_assists": [5.0, 7.0, 2.5, 1.5, 0.0],
        "expected_goal_involvements": [9.5, 13.5, 16.5, 3.0, 0.0],
        "expected_goals_conceded": [32.0, 27.0, 45.0, 22.0, 12.0],
        "influence": [400.0, 450.0, 500.0, 380.0, 300.0],
        "creativity": [300.0, 350.0, 200.0, 250.0, 150.0],
        "threat": [500.0, 550.0, 650.0, 300.0, 100.0],
        "ict_index": [50.0, 55.0, 60.0, 45.0, 35.0],
    }
    return pd.DataFrame(data)


# Simple unit tests 


def test__compute_mae_basic():
    """
    _compute_mae must return a float >= 0 and correctly ignore NaNs
    """
    y_true = [10, 20, np.nan, 40]
    y_pred = [12, 18, 30, np.nan]

    mae = model._compute_mae(y_true, y_pred)

    assert isinstance(mae, float)
    # only the first two elements are kept (the others have NaN values)
    expected = (abs(10 - 12) + abs(20 - 18)) / 2
    assert mae == pytest.approx(expected)


def test_simple_mean_model_predict():
    """
    SimpleMeanModel must predict the mean of total_points
    """
    df = make_fake_fpl_df()
    m = model.SimpleMeanModel(df)
    preds = m.predict(3)

    assert len(preds) == 3
    expected_mean = float(df["total_points"].mean())
    for p in preds:
        assert p == pytest.approx(expected_mean)


def test_position_mean_model_predict_for_position():
    """
    PositionMeanModel must use the average per position,
    and revert to the overall average if the position is unknown.
    """
    means_by_pos = {"MID": 25.0, "DEF": 15.0}
    global_mean = 20.0
    m = model.PositionMeanModel(means_by_pos, global_mean)

    assert m.predict_for_position("MID") == pytest.approx(25.0)
    assert m.predict_for_position("DEF") == pytest.approx(15.0)
    # poste inconnu -> global_mean
    assert m.predict_for_position("GK") == pytest.approx(20.0)


# Fixture for patching load_data


@pytest.fixture
def patch_load_data(monkeypatch):
    """
    We force load_data() to return a small dummy DataFrame.
    All evaluate_* functions will use this controlled dataset
    """

    fake_df = make_fake_fpl_df()

    def fake_load_data(path=None):
        return fake_df

    monkeypatch.setattr(model, "load_data", fake_load_data)


# Model evaluation function tests


def test_evaluate_model_with_fake_data_returns_non_negative_float(patch_load_data):
    mae = model.evaluate_model()
    assert isinstance(mae, float)
    assert mae >= 0.0


def test_evaluate_position_mean_model_with_fake_data_returns_non_negative_float(
    patch_load_data,
):
    mae = model.evaluate_position_mean_model()
    assert isinstance(mae, float)
    assert mae >= 0.0


def test_evaluate_linear_model_with_fake_data_returns_non_negative_float(
    patch_load_data,
):
    mae = model.evaluate_linear_model()
    assert isinstance(mae, float)
    assert mae >= 0.0


def test_evaluate_random_forest_model_with_fake_data_returns_non_negative_float(
    patch_load_data,
):
    mae = model.evaluate_random_forest_model()
    assert isinstance(mae, float)
    assert mae >= 0.0
