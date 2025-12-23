from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.evaluation as ev


# ---------------------------------------------------------------------
# Fixtures: small deterministic datasets
# ---------------------------------------------------------------------
@pytest.fixture()
def mini_gw_df() -> pd.DataFrame:
    """
    Small player-gameweeks dataset spanning 2 seasons.
    Includes points + metadata required by evaluation module.
    """
    rows = []
    # 2 players, 2 seasons, 4 gameweeks each
    for season in ["2021/22", "2022/23"]:
        for pid, name, team, pos in [
            (1, "P1", "Manchester City", "MID"),
            (2, "P2", "Tottenham Hotspur", "DEF"),
        ]:
            for gw in [1, 2, 3, 4, 5, 6, 7]:
                rows.append(
                    {
                        "player_id": pid,
                        "id": pid,
                        "name": name,
                        "team": team,
                        "position": pos,
                        "season": season,
                        "gameweek": gw,
                        "points": float(pid + gw),  # deterministic signal
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def mini_odds_df() -> pd.DataFrame:
    """
    Minimal cleaned odds dataset expected by compute_team_strength / compare_model_vs_bookmakers.
    """
    return pd.DataFrame(
        {
            "season": ["2022/23", "2022/23"],
            "home_team": ["Manchester City", "Tottenham Hotspur"],
            "away_team": ["Tottenham Hotspur", "Manchester City"],
            "pnorm_home_win": [0.60, 0.40],
            "pnorm_draw": [0.20, 0.30],
            "pnorm_away_win": [0.20, 0.30],
            "match_date": ["2022-08-01", "2022-08-02"],
        }
    )


@pytest.fixture()
def mini_fixtures_df() -> pd.DataFrame:
    """
    Minimal fixtures dataset aligned with mini_odds_df so the merge succeeds.
    """
    return pd.DataFrame(
        {
            "season": ["2022/23", "2022/23"],
            "gameweek": [1, 2],
            "home_team": ["Manchester City", "Tottenham Hotspur"],
            "away_team": ["Tottenham Hotspur", "Manchester City"],
        }
    )


# ---------------------------------------------------------------------
# Unit tests: metrics & helpers
# ---------------------------------------------------------------------
def test_compute_metrics_ignore_nans_and_raise_on_all_nan():
    y_true = np.array([1.0, np.nan, 3.0])
    y_pred = np.array([2.0, 5.0, np.nan])

    # Only first pair is valid: (1.0, 2.0)
    assert ev._compute_mae(y_true, y_pred) == pytest.approx(1.0)
    assert ev._compute_rmse(y_true, y_pred) == pytest.approx(1.0)

    # R2 on a single point: ss_tot=0 -> 0.0 by your implementation
    assert ev._compute_r2(y_true, y_pred) == pytest.approx(0.0)

    # Spearman on a single point is nan-ish in pandas; your code returns float(...),
    # but with one point corr is nan. We just check it doesn't crash when there is 1 valid.
    s = ev._compute_spearman(y_true, y_pred)
    assert np.isnan(s) or (-1.0 <= s <= 1.0)

    # all NaN pairs -> must raise
    with pytest.raises(ValueError):
        ev._compute_mae([np.nan], [np.nan])
    with pytest.raises(ValueError):
        ev._compute_rmse([np.nan], [np.nan])
    with pytest.raises(ValueError):
        ev._compute_r2([np.nan], [np.nan])
    with pytest.raises(ValueError):
        ev._compute_spearman([np.nan], [np.nan])


def test_build_gw_features_and_target_sanitizes_inf_and_drops_nans():
    df = pd.DataFrame(
        {
            "points": [1.0, 2.0, np.nan],
            "f1": [1.0, np.inf, 3.0],
            "f2": [0.0, -np.inf, 1.0],
        }
    )

    X, y = ev._build_gw_features_and_target(df, feature_cols=["f1", "f2"], target_col="points")
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    # third row dropped due to points NaN
    assert len(X) == 2
    assert len(y) == 2

    # inf values must be converted -> NaN -> filled with 0.0
    assert float(X.loc[X.index[1], "f1"]) == 0.0
    assert float(X.loc[X.index[1], "f2"]) == 0.0


# ---------------------------------------------------------------------
# Bookmaker: compute_team_strength & build_team_strength_table
# ---------------------------------------------------------------------
def test_compute_team_strength_happy_path(mini_odds_df):
    strength = ev.compute_team_strength(mini_odds_df)
    assert {"team", "bet365_strength", "n_matches"}.issubset(strength.columns)
    assert len(strength) == 2
    assert strength["bet365_strength"].between(0.0, 1.0).all()


def test_compute_team_strength_raises_on_missing_cols(mini_odds_df):
    bad = mini_odds_df.drop(columns=["pnorm_away_win"])
    with pytest.raises(ValueError):
        ev.compute_team_strength(bad)


def test_build_team_strength_table_uses_loader(monkeypatch, mini_odds_df):
    monkeypatch.setattr(ev, "load_clean_odds", lambda: mini_odds_df)
    out = ev.build_team_strength_table()
    assert len(out) == 2


# ---------------------------------------------------------------------
# GW evaluation functions
# We monkeypatch:
# - load_player_gameweeks to return small df
# - lag builders to add the needed columns deterministically
# ---------------------------------------------------------------------
def _fake_add_anytime_lags(df: pd.DataFrame, max_lag: int, rolling_window: int | None):
    df = df.sort_values(["player_id", "season", "gameweek"]).copy()
    group_cols = ["player_id", "season"]

    for k in range(1, max_lag + 1):
        df[f"points_lag_{k}"] = df.groupby(group_cols)["points"].shift(k)

    if rolling_window is not None:
        prev = df.groupby(group_cols)["points"].shift(1)
        df[f"points_rolling_mean_{max_lag}"] = prev.rolling(max_lag).mean()

    return df


def _fake_add_seasonal_lags_with_prev5(df: pd.DataFrame, max_lag: int):
    # Minimal seasonal lag scheme: provide points_lag_1..3 and points_lag_mean
    df = df.sort_values(["player_id", "season", "gameweek"]).copy()
    group_cols = ["player_id", "season"]

    for k in range(1, max_lag + 1):
        df[f"points_lag_{k}"] = df.groupby(group_cols)["points"].shift(k)

    # Use mean of last 3 lags as a stable proxy
    df["points_lag_mean"] = df[[f"points_lag_{k}" for k in range(1, 4)]].mean(axis=1)
    return df


def test_anytime_linear_eval_variants(monkeypatch, mini_gw_df):
    # Patch data + lag builder
    monkeypatch.setattr(ev, "load_player_gameweeks", lambda: mini_gw_df)
    monkeypatch.setattr(ev, "_add_anytime_lags", _fake_add_anytime_lags)

    mae3 = ev.evaluate_linear_gw_model_lag3(test_season="2022/23")
    mae5 = ev.evaluate_linear_gw_model_lag5(test_season="2022/23")

    assert isinstance(mae3, float)
    assert isinstance(mae5, float)
    assert mae3 >= 0.0
    assert mae5 >= 0.0


def test_seasonal_linear_and_baseline(monkeypatch, mini_gw_df):
    monkeypatch.setattr(ev, "load_player_gameweeks", lambda: mini_gw_df)
    monkeypatch.setattr(ev, "_add_seasonal_lags_with_prev5", _fake_add_seasonal_lags_with_prev5)
    monkeypatch.setattr(ev, "_add_anytime_lags", _fake_add_anytime_lags)

    mae_seasonal = ev.evaluate_linear_gw_model_seasonal(test_season="2022/23")
    mae_base = ev.evaluate_gw_baseline_lag1(test_season="2022/23")

    assert isinstance(mae_seasonal, float)
    assert isinstance(mae_base, float)
    assert mae_seasonal >= 0.0
    assert mae_base >= 0.0


def test_prepare_gw_lag_dataset_and_within_season_linear(monkeypatch, mini_gw_df):
    monkeypatch.setattr(ev, "load_player_gameweeks", lambda: mini_gw_df)

    train_df, test_df = ev.prepare_gw_lag_dataset(mini_gw_df, test_season="2022/23")
    assert not train_df.empty
    assert not test_df.empty
    assert "points_lag_1" in train_df.columns

    mae = ev.evaluate_linear_gw_model(test_season="2022/23")
    assert isinstance(mae, float)
    assert mae >= 0.0


def test_seasonal_gbm_eval_and_predictions(monkeypatch, mini_gw_df):
    monkeypatch.setattr(ev, "load_player_gameweeks", lambda: mini_gw_df)
    monkeypatch.setattr(ev, "_add_seasonal_lags_with_prev5", _fake_add_seasonal_lags_with_prev5)

    mae = ev.evaluate_gbm_gw_model_seasonal(test_season="2022/23")
    assert isinstance(mae, float)
    assert mae >= 0.0

    preds = ev.get_test_predictions_seasonal_gbm_gw(test_season="2022/23")
    assert isinstance(preds, pd.DataFrame)
    assert {"player_id", "season", "gameweek", "points", "predicted_points", "abs_error"}.issubset(preds.columns)


def test_get_ytrue_ypred_helpers(monkeypatch, mini_gw_df):
    monkeypatch.setattr(ev, "load_player_gameweeks", lambda: mini_gw_df)
    monkeypatch.setattr(ev, "_add_anytime_lags", _fake_add_anytime_lags)
    monkeypatch.setattr(ev, "_add_seasonal_lags_with_prev5", _fake_add_seasonal_lags_with_prev5)

    y_true, y_pred = ev.get_ytrue_ypred_anytime_linear_gw(3, test_season="2022/23")
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert len(y_true) == len(y_pred)

    y_true2, y_pred2 = ev.get_ytrue_ypred_seasonal_linear_gw(test_season="2022/23")
    assert len(y_true2) == len(y_pred2)

    y_true3, y_pred3 = ev.get_ytrue_ypred_seasonal_gbm_gw(test_season="2022/23")
    assert len(y_true3) == len(y_pred3)


# ---------------------------------------------------------------------
# compare_model_vs_bookmakers (offline)
# Patch:
# - load_clean_odds / load_fixtures to small DFs
# - src.models.predict_gw_all_players (imported locally inside function)
#   by injecting a fake module attribute via sys.modules patching is overkill;
#   easiest is monkeypatching src.models.predict_gw_all_players directly.
# ---------------------------------------------------------------------
def test_compare_model_vs_bookmakers_offline(monkeypatch, mini_odds_df, mini_fixtures_df):
    monkeypatch.setattr(ev, "load_clean_odds", lambda: mini_odds_df)
    monkeypatch.setattr(ev, "load_fixtures", lambda: mini_fixtures_df)

    # Fake predict_gw_all_players returns predictions for the teams in fixtures
    def _fake_predict_gw_all_players(model: str, test_season: str, gameweek=None):
        return pd.DataFrame(
            {
                "season": ["2022/23"] * 44,
                "gameweek": [1] * 22 + [2] * 22,
                "team": (["Manchester City"] * 11 + ["Tottenham Hotspur"] * 11) * 2,
                "predicted_points": np.linspace(1.0, 5.0, 44),
                # include minutes lags so playing_weight branch is exercised
                "minutes_lag_1": [90.0] * 44,
                "minutes_lag_2": [80.0] * 44,
                "minutes_lag_3": [70.0] * 44,
            }
        )

    import src.models as models
    monkeypatch.setattr(models, "predict_gw_all_players", _fake_predict_gw_all_players)

    comp, mae, corr = ev.compare_model_vs_bookmakers(
        model="gw_seasonal_gbm",
        test_season="2022/23",
        verbose=False,
    )

    assert isinstance(comp, pd.DataFrame)
    assert {"season", "gameweek", "home_team", "away_team", "pnorm_home_win", "p_model_home_win", "abs_error"}.issubset(
        comp.columns
    )
    assert isinstance(mae, float)
    assert isinstance(corr, float)
    assert len(comp) > 0
