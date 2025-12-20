from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.evaluation as evaluation


# ---------------------------------------------------------------------
# 1) BOOKMAKER BENCHMARKS: compute_team_strength
# ---------------------------------------------------------------------


def test_compute_team_strength_missing_columns_raises():
    odds_df = pd.DataFrame(
        {
            "home_team": ["A"],
            "away_team": ["B"],
            # missing pnorm_home_win / pnorm_away_win
        }
    )

    with pytest.raises(ValueError) as exc:
        evaluation.compute_team_strength(odds_df)

    # Verify that the error message explicitly refers to missing required columns
    msg = str(exc.value)
    assert "Missing required columns" in msg


def test_compute_team_strength_basic_schema_and_order():
    """
    Contract-level deterministic test:
    - output schema
    - no NaN values in bet365_strength
    - descending order by bet365_strength
    - consistent match counts
    """
    odds_df = pd.DataFrame(
        {
            "home_team": ["A", "A", "B", "C"],
            "away_team": ["B", "C", "A", "A"],
            "pnorm_home_win": [0.60, 0.70, 0.50, 0.40],
            "pnorm_away_win": [0.30, 0.20, 0.35, 0.45],
        }
    )

    out = evaluation.compute_team_strength(odds_df)

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["team", "bet365_strength", "n_matches"]
    assert len(out) >= 1

    assert out["bet365_strength"].notna().all()
    assert out["n_matches"].notna().all()

    # n_matches must be strictly positive for all returned teams
    assert (out["n_matches"] > 0).all()

    # Output must be sorted in descending order of strength (as implemented)
    strengths = out["bet365_strength"].to_numpy()
    assert np.all(strengths[:-1] >= strengths[1:])

    # All teams appearing in the input must be present (A, B, C)
    assert set(out["team"]) == {"A", "B", "C"}


def test_compute_team_strength_exact_value_single_team_manual():
    """
    Exact-value test on a simple manually computable case.

    Team A:
    - plays 2 home matches
    - plays 1 away match
    """
    odds_df = pd.DataFrame(
        {
            "home_team": ["A", "A", "B"],
            "away_team": ["B", "C", "A"],
            "pnorm_home_win": [0.60, 0.80, 0.50],  # A-home: 0.60, 0.80 (mean 0.70)
            "pnorm_away_win": [0.30, 0.20, 0.40],  # A-away: 0.40
        }
    )

    out = evaluation.compute_team_strength(odds_df)
    a_row = out[out["team"] == "A"].iloc[0]

    # A: home mean=0.70 with 2 matches, away mean=0.40 with 1 match
    expected = (0.70 * 2 + 0.40 * 1) / 3
    assert a_row["n_matches"] == 3
    assert a_row["bet365_strength"] == pytest.approx(expected, rel=0, abs=1e-12)


# ---------------------------------------------------------------------
# 2) CORE EVALUATION UTILITIES: _compute_mae, _build_gw_features_and_target
# ---------------------------------------------------------------------


def test__compute_mae_ignores_nans_and_computes_correctly():
    y_true = np.array([1.0, np.nan, 3.0, 4.0])
    y_pred = np.array([2.0, 10.0, np.nan, 1.0])

    # Valid pairs are indices 0 and 3:
    # |1-2| = 1, |4-1| = 3 => MAE = (1 + 3) / 2 = 2
    mae = evaluation._compute_mae(y_true, y_pred)
    assert mae == pytest.approx(2.0, rel=0, abs=1e-12)


def test__compute_mae_all_nan_raises():
    y_true = np.array([np.nan, np.nan])
    y_pred = np.array([np.nan, np.nan])

    with pytest.raises(ValueError) as exc:
        evaluation._compute_mae(y_true, y_pred)

    assert "No valid" in str(exc.value)


def test__build_gw_features_and_target_drops_missing_and_sanitizes_inf():
    df = pd.DataFrame(
        {
            "f1": [1.0, np.nan, np.inf, 4.0],
            "f2": [0.0, 2.0, 3.0, -np.inf],
            "points": [10.0, 20.0, 30.0, np.nan],
        }
    )

    X, y = evaluation._build_gw_features_and_target(
        df,
        feature_cols=["f1", "f2"],
        target_col="points",
    )

    # dropna subset=["f1","f2","points"] => rows kept: indices 0 and 2 only
    assert len(X) == 2
    assert len(y) == 2

    # inf -> NaN -> filled with 0.0
    assert np.isfinite(X.to_numpy()).all()

    # Target must be floating point
    assert y.dtype.kind == "f"


# ---------------------------------------------------------------------
# 3) STRICT WITHIN-SEASON LAG DATASET: prepare_gw_lag_dataset
# ---------------------------------------------------------------------


def test_prepare_gw_lag_dataset_creates_expected_lags_and_splits():
    df = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 2, 2],
            "season": ["2021/22", "2021/22", "2022/23", "2021/22", "2022/23"],
            "gameweek": [1, 2, 1, 1, 1],
            "points": [2.0, 5.0, 1.0, 10.0, 7.0],
        }
    )

    train_df, test_df = evaluation.prepare_gw_lag_dataset(
        df,
        test_season="2022/23",
    )

    # Correct train / test split by season
    assert set(train_df["season"]) == {"2021/22"}
    assert set(test_df["season"]) == {"2022/23"}

    # Lag features must be present
    for col in [
        "points_lag_1",
        "points_lag_2",
        "points_lag_3",
        "points_rolling_mean_3",
    ]:
        assert col in train_df.columns
        assert col in test_df.columns

    # Within-season lag check:
    # player 1, season 2021/22, GW2 -> lag_1 should equal points at GW1 = 2.0
    row = train_df[
        (train_df["player_id"] == 1)
        & (train_df["season"] == "2021/22")
        & (train_df["gameweek"] == 2)
    ].iloc[0]

    assert row["points_lag_1"] == pytest.approx(2.0, rel=0, abs=1e-12)

