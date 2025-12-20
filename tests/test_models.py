from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import predict_gw_all_players


class _FastRegressor:
    """
    Tiny deterministic regressor used to avoid slow sklearn training in unit tests.

    It learns a single scalar (mean of y) and always predicts that constant.
    This keeps the test focused on the pipeline contract (lags -> features -> output schema)
    rather than on ML performance.
    """

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self._mean = float(np.mean(y)) if y.size else 0.0
        return self

    def predict(self, X):
        return np.full(shape=(len(X),), fill_value=self._mean, dtype=float)


def _toy_player_gameweeks() -> pd.DataFrame:
    """
    Small deterministic dataset used for unit testing.

    The dataset is deliberately constructed to:
    - provide enough historical depth to train an "anytime" GW model (gw_lag3),
    - include both a training season and a test season,
    - contain minute information so that minute lags are generated.
    """
    rows = []

    # Player 1
    for gw, pts, mins in [(1, 1.0, 90), (2, 2.0, 80), (3, 3.0, 90), (4, 4.0, 70), (5, 5.0, 90)]:
        rows.append(
            {
                "player_id": 1,
                "name": "Alpha",
                "team": "TeamA",
                "position": "MID",
                "season": "2021/22",
                "gameweek": gw,
                "points": pts,
                "minutes": mins,
            }
        )

    for gw, pts, mins in [(1, 2.0, 90), (2, 4.0, 90)]:
        rows.append(
            {
                "player_id": 1,
                "name": "Alpha",
                "team": "TeamA",
                "position": "MID",
                "season": "2022/23",
                "gameweek": gw,
                "points": pts,
                "minutes": mins,
            }
        )

    # Player 2 (included to validate multi-player behavior and sorting)
    for gw, pts, mins in [(1, 0.0, 0), (2, 1.0, 30), (3, 1.0, 20), (4, 2.0, 60), (5, 2.0, 75)]:
        rows.append(
            {
                "player_id": 2,
                "name": "Beta",
                "team": "TeamA",
                "position": "DEF",
                "season": "2021/22",
                "gameweek": gw,
                "points": pts,
                "minutes": mins,
            }
        )

    for gw, pts, mins in [(1, 1.0, 90), (2, 1.0, 45)]:
        rows.append(
            {
                "player_id": 2,
                "name": "Beta",
                "team": "TeamA",
                "position": "DEF",
                "season": "2022/23",
                "gameweek": gw,
                "points": pts,
                "minutes": mins,
            }
        )

    return pd.DataFrame(rows)


def test_predict_gw_all_players_contract_gw_lag3_gameweek_filter(monkeypatch):
    """
    Core contract test for predict_gw_all_players (gw_lag3).

    This test verifies that:
    - the function executes without error,
    - the gameweek filter is correctly applied,
    - the output schema is respected,
    - predicted points are finite numeric values,
    - minute lag features are present for downstream usage.

    We patch LinearRegression with a tiny fast regressor to keep the test < 1s.
    """
    import src.models as models  # monkeypatch at the correct import location

    monkeypatch.setattr(models, "load_player_gameweeks", _toy_player_gameweeks)
    monkeypatch.setattr(models, "LinearRegression", _FastRegressor)

    out = predict_gw_all_players(
        model="gw_lag3",
        test_season="2022/23",
        gameweek=1,
    )

    assert isinstance(out, pd.DataFrame)
    assert not out.empty

    # Correct filtering on season and gameweek
    assert set(out["season"]) == {"2022/23"}
    assert set(out["gameweek"]) == {1}

    # Minimal output contract
    required_cols = {
        "player_id",
        "name",
        "team",
        "position",
        "season",
        "gameweek",
        "predicted_points",
    }
    assert required_cols.issubset(set(out.columns))

    # Predicted points must be numeric and finite
    assert out["predicted_points"].dtype.kind in "fi"
    assert np.isfinite(out["predicted_points"].to_numpy(dtype=float)).all()

    # Minute lag columns should be present when minutes are available
    minute_cols = [c for c in out.columns if c.startswith("minutes_lag_")]
    assert minute_cols  # non-empty list


def test_predict_gw_all_players_unknown_model_raises(monkeypatch):
    """
    Defensive test: an explicit error must be raised
    when an unknown model identifier is provided.
    """
    import src.models as models

    monkeypatch.setattr(models, "load_player_gameweeks", _toy_player_gameweeks)

    with pytest.raises(ValueError) as exc:
        predict_gw_all_players(
            model="not_a_model",
            test_season="2022/23",
            gameweek=1,
        )

    assert "Unknown GW model" in str(exc.value)
