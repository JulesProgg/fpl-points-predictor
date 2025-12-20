from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import src.data_loader as data_loader


def test_load_player_gameweeks_normalizes_season_and_validates_required_columns(tmp_path):
    """
    Minimal unit test for the main processed loader.

    It verifies that:
    - the loader reads a CSV file from a provided path,
    - season strings are normalized (e.g., '2016-2017' -> '2016/17'),
    - required columns are validated.
    """
    p = tmp_path / "player_gameweeks.csv"

    df_in = pd.DataFrame(
        {
            "id": [101, 102],
            "name": ["A", "B"],
            "team": ["TeamA", "TeamB"],
            "position": ["MID", "DEF"],
            "season": ["2016-2017", "2016/17"],
            "gameweek": [1, 1],
            "points": [5.0, 2.0],
        }
    )
    df_in.to_csv(p, index=False)

    df_out = data_loader.load_player_gameweeks(p)

    assert isinstance(df_out, pd.DataFrame)
    assert len(df_out) == 2
    assert set(df_out["season"]) == {"2016/17"}  # normalized
    assert {"id", "name", "team", "position", "season", "gameweek", "points"}.issubset(df_out.columns)


def test_load_clean_odds_raises_file_not_found(tmp_path):
    """
    Defensive unit test: the loader must fail fast when the file does not exist.
    """
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        data_loader.load_clean_odds(missing_path)


def test_load_clean_odds_normalizes_team_names_and_validates_schema(tmp_path):
    """
    Unit test for odds loader:
    - validates required schema
    - normalizes season string format
    - normalizes Bet365 team names to FPL names
    """
    p = tmp_path / "odds.csv"

    df_in = pd.DataFrame(
        {
            "home_team": ["Man City"],
            "away_team": ["Spurs"],
            "pnorm_home_win": [0.6],
            "pnorm_draw": [0.2],
            "pnorm_away_win": [0.2],
            "match_date": ["2022-08-01"],
            "season": ["2022-2023"],
        }
    )
    df_in.to_csv(p, index=False)

    df_out = data_loader.load_clean_odds(p)

    assert df_out.loc[0, "season"] == "2022/23"
    assert df_out.loc[0, "home_team"] == "Manchester City"
    assert df_out.loc[0, "away_team"] == "Tottenham Hotspur"

    required_cols = {
        "home_team",
        "away_team",
        "pnorm_home_win",
        "pnorm_draw",
        "pnorm_away_win",
        "match_date",
        "season",
    }
    assert required_cols.issubset(df_out.columns)


def test_normalise_season_str_examples():
    """
    Unit test for the pure helper normalise_season_str with representative formats.
    """
    assert data_loader.normalise_season_str("2016-17") == "2016/17"
    assert data_loader.normalise_season_str("2016_17") == "2016/17"
    assert data_loader.normalise_season_str("2016/2017") == "2016/17"
    assert data_loader.normalise_season_str("2022/23") == "2022/23"
