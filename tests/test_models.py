from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.models as m


# ---------------------------------------------------------------------
# Fixtures: small deterministic datasets
# ---------------------------------------------------------------------
def _make_base_df(
    *,
    seasons: list[str],
    gws: list[int],
    include_minutes: bool = True,
    negative_points: bool = False,
) -> pd.DataFrame:
    """
    Build a synthetic per-player per-gameweek dataset that matches models.py expectations.
    - Must include: player_id, name, team, position, season, gameweek, points
    - Optional: minutes (to exercise _add_minute_lags)
    """
    rows = []
    for season in seasons:
        for pid, name, team, pos in [
            (1, "P1", "Manchester City", "MID"),
            (2, "P2", "Tottenham Hotspur", "DEF"),
        ]:
            for gw in gws:
                # deterministic point signal; allow negatives for GBM fallback branch
                pts = float(pid + gw)
                if negative_points and (season == seasons[0]) and (gw in {1, 2}):
                    pts = -2.0  # ensures min(points) <= -1.0

                row = {
                    "player_id": pid,
                    "name": name,
                    "team": team,
                    "position": pos,
                    "season": season,
                    "gameweek": gw,
                    "points": pts,
                }
                if include_minutes:
                    row["minutes"] = 90
                rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def df_long_positive() -> pd.DataFrame:
    # enough history for lag10 and seasonal fallback; all points >= 0 (log1p branch)
    return _make_base_df(
        seasons=["2021/22", "2022/23"],
        gws=list(range(1, 13)),  # 12 GWs
        include_minutes=True,
        negative_points=False,
    )


@pytest.fixture()
def df_long_with_negatives() -> pd.DataFrame:
    # enough history but includes points <= -1 to force GBM "raw points" branch
    return _make_base_df(
        seasons=["2021/22", "2022/23"],
        gws=list(range(1, 13)),
        include_minutes=True,
        negative_points=True,
    )


@pytest.fixture()
def df_no_minutes() -> pd.DataFrame:
    # exercise _add_minute_lags early return branch
    return _make_base_df(
        seasons=["2021/22", "2022/23"],
        gws=list(range(1, 8)),
        include_minutes=False,
        negative_points=False,
    )


# ---------------------------------------------------------------------
# Unit tests: feature engineering helpers
# ---------------------------------------------------------------------
def test_add_minute_lags_no_minutes_returns_unchanged(df_no_minutes):
    out = m._add_minute_lags(df_no_minutes.copy(), max_lag=3)
    assert isinstance(out, pd.DataFrame)
    assert not any(c.startswith("minutes_lag_") for c in out.columns)


def test_add_minute_lags_creates_lags(df_long_positive):
    out = m._add_minute_lags(df_long_positive.copy(), max_lag=3)
    for k in (1, 2, 3):
        assert f"minutes_lag_{k}" in out.columns

    # within-season: first GW of a season must be NaN for minutes_lag_1
    first = out[(out["player_id"] == 1) & (out["season"] == "2022/23") & (out["gameweek"] == 1)].iloc[0]
    assert pd.isna(first["minutes_lag_1"])


def test_add_anytime_lags_and_rolling(df_long_positive):
    out = m._add_anytime_lags(df_long_positive.copy(), max_lag=3, rolling_window=3)
    assert {"points_lag_1", "points_lag_2", "points_lag_3", "points_rolling_mean_3"}.issubset(out.columns)

    # Anytime grouping is only by player_id (cross-season continuity)
    # For player 1, season 2022/23 gw1, lag1 should be last gw of 2021/22 (gw12)
    last_prev = df_long_positive[(df_long_positive["player_id"] == 1) & (df_long_positive["season"] == "2021/22") & (df_long_positive["gameweek"] == 12)].iloc[0]["points"]
    val = out[(out["player_id"] == 1) & (out["season"] == "2022/23") & (out["gameweek"] == 1)].iloc[0]["points_lag_1"]
    assert float(val) == float(last_prev)


def test_add_seasonal_lags_with_prev5_fallback(df_long_positive):
    out = m._add_seasonal_lags_with_prev5(df_long_positive.copy(), max_lag=5)

    # Seasonal lags + mean exist
    for k in range(1, 6):
        assert f"points_lag_{k}" in out.columns
    assert "points_lag_mean" in out.columns

    # Internal anytime columns must be dropped
    assert not any(c.startswith("points_lag_any_") for c in out.columns)

    # Fallback check: season 2022/23 gw1 has no within-season lag,
    # so lag_1 should be filled with last appearance (cross-season)
    last_prev = df_long_positive[(df_long_positive["player_id"] == 1) & (df_long_positive["season"] == "2021/22") & (df_long_positive["gameweek"] == 12)].iloc[0]["points"]
    val = out[(out["player_id"] == 1) & (out["season"] == "2022/23") & (out["gameweek"] == 1)].iloc[0]["points_lag_1"]
    assert float(val) == float(last_prev)


# ---------------------------------------------------------------------
# Predict: dispatch, outputs, and branches
# ---------------------------------------------------------------------
def test_predict_unknown_model_raises(monkeypatch, df_long_positive):
    monkeypatch.setattr(m, "load_player_gameweeks", lambda: df_long_positive)
    with pytest.raises(ValueError):
        m.predict_gw_all_players(model="nope", test_season="2022/23", gameweek=1)


@pytest.mark.parametrize("model_name", ["gw_lag3", "gw_lag5", "gw_lag10"])
def test_predict_anytime_models_work(monkeypatch, df_long_positive, model_name):
    monkeypatch.setattr(m, "load_player_gameweeks", lambda: df_long_positive)

    out = m.predict_gw_all_players(model=model_name, test_season="2022/23", gameweek=5)

    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0
    assert set(out["season"].unique()) == {"2022/23"}
    assert set(out["gameweek"].unique()) == {5}
    assert "predicted_points" in out.columns

    # minute lags should be present (since minutes exist)
    assert any(c.startswith("minutes_lag_") for c in out.columns)


@pytest.mark.parametrize("model_name", ["gw_seasonal", "gw_seasonal_linear"])
def test_predict_seasonal_linear_models_work(monkeypatch, df_long_positive, model_name):
    monkeypatch.setattr(m, "load_player_gameweeks", lambda: df_long_positive)

    out = m.predict_gw_all_players(model=model_name, test_season="2022/23", gameweek=6)

    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0
    assert set(out["season"].unique()) == {"2022/23"}
    assert set(out["gameweek"].unique()) == {6}
    assert "predicted_points" in out.columns


def test_predict_seasonal_gbm_log1p_branch_and_clip(monkeypatch, df_long_positive):
    """
    Covers:
    - GBM training with log1p(y) when min(y) > -1
    - expm1 inversion
    - clipping [0, 25]
    """
    monkeypatch.setattr(m, "load_player_gameweeks", lambda: df_long_positive)

    out = m.predict_gw_all_players(model="gw_seasonal_gbm", test_season="2022/23", gameweek=7)

    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0
    assert "predicted_points" in out.columns

    # Should be clipped to [0, 25] for GBM
    assert out["predicted_points"].min() >= 0.0
    assert out["predicted_points"].max() <= 25.0


def test_predict_seasonal_gbm_raw_points_branch(monkeypatch, df_long_with_negatives):
    """
    Covers GBM fallback branch when y contains values <= -1.0:
    - train on raw y (no log1p)
    - still clips predictions
    """
    monkeypatch.setattr(m, "load_player_gameweeks", lambda: df_long_with_negatives)

    out = m.predict_gw_all_players(model="gw_seasonal_gbm", test_season="2022/23", gameweek=7)

    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0
    assert out["predicted_points"].min() >= 0.0
    assert out["predicted_points"].max() <= 25.0


def test_predict_gameweek_none_returns_multiple(monkeypatch, df_long_positive):
    monkeypatch.setattr(m, "load_player_gameweeks", lambda: df_long_positive)

    out = m.predict_gw_all_players(model="gw_seasonal_gbm", test_season="2022/23", gameweek=None)

    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0
    assert set(out["season"].unique()) == {"2022/23"}
    assert len(set(out["gameweek"].unique())) > 1


def test_predict_returns_empty_dataframe_when_no_rows_after_dropna(monkeypatch, df_long_positive):
    """
    Force the 'empty after dropna(feature_cols)' branch:
    Make test season have no usable rows by setting points to NaN in test season.
    """
    df = df_long_positive.copy()
    df.loc[df["season"] == "2022/23", "points"] = np.nan

    monkeypatch.setattr(m, "load_player_gameweeks", lambda: df)

    out = m.predict_gw_all_players(model="gw_seasonal_gbm", test_season="2022/23", gameweek=3)

    assert isinstance(out, pd.DataFrame)
    # function returns an empty df with specific columns (cols_out)
    assert list(out.columns) == [
        "player_id",
        "name",
        "team",
        "position",
        "season",
        "gameweek",
        "predicted_points",
    ]
    assert len(out) == 0


def test_predict_without_minutes_has_no_minutes_lags(monkeypatch, df_no_minutes):
    monkeypatch.setattr(m, "load_player_gameweeks", lambda: df_no_minutes)

    out = m.predict_gw_all_players(model="gw_seasonal", test_season="2022/23", gameweek=4)

    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0
    assert "predicted_points" in out.columns
    assert not any(c.startswith("minutes_lag_") for c in out.columns)