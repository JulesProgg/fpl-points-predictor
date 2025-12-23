from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import src.data_loader as dl


# ---------------------------------------------------------------------
# Helpers (tests only)
# ---------------------------------------------------------------------
def _write_csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def _minimal_raw_gw_df(*, season: str = "2022/23", gw: int = 1) -> pd.DataFrame:
    """
    Build a minimal raw gameweeks dataframe that satisfies GAMEWEEK_BASE_COLS.
    Many columns can be set to 0/NA; schema completeness is what matters.
    """
    base = {c: [0] for c in dl.GAMEWEEK_BASE_COLS}

    # overwrite some meaningful fields
    base["id"] = [101]
    base["name"] = ["Player A"]
    base["team"] = ["Man City"]  # to exercise name normalisation in some places
    base["position"] = ["MID"]
    base["minutes"] = [90]
    base["total_points"] = [7]
    base["season"] = [season]
    base["gameweek"] = [gw]

    # add optional columns to exercise branches
    base["opponent"] = ["Spurs"]
    base["was_home"] = ["True"]  # string form to exercise boolean coercion later (build_fixtures)

    return pd.DataFrame(base)


def _patch_project_paths(monkeypatch, tmp_path: Path) -> None:
    """
    Redirect all module-level paths to tmp_path so build_* pipelines write locally.
    """
    raw_dir = tmp_path / "data" / "raw"
    proc_dir = tmp_path / "data" / "processed"

    monkeypatch.setattr(dl, "DATA_RAW_DIR", raw_dir, raising=True)
    monkeypatch.setattr(dl, "DATA_PROCESSED_DIR", proc_dir, raising=True)

    # Rebind files derived from those dirs
    monkeypatch.setattr(dl, "RAW_GW_FILE", raw_dir / "player_gameweeks_raw.csv", raising=True)
    monkeypatch.setattr(dl, "RAW_ODDS_FILE", raw_dir / "oddsdataset.csv", raising=True)

    monkeypatch.setattr(dl, "OUTPUT_PATH", dl.RAW_GW_FILE, raising=True)

    monkeypatch.setattr(dl, "PLAYER_GW_FILE", proc_dir / "player_gameweeks.csv", raising=True)
    monkeypatch.setattr(dl, "EPL_FIXTURES_FILE", proc_dir / "epl_fixtures_2016_23.csv", raising=True)
    monkeypatch.setattr(dl, "ODDS_FILE", proc_dir / "bet365odds_epl_2016_23.csv", raising=True)
    monkeypatch.setattr(dl, "OUT_ODDS_FILE", dl.ODDS_FILE, raising=True)


# ---------------------------------------------------------------------
# Helpers / Normalisation
# ---------------------------------------------------------------------
def test_normalise_season_str_examples():
    assert dl.normalise_season_str("2016-17") == "2016/17"
    assert dl.normalise_season_str("2016_17") == "2016/17"
    assert dl.normalise_season_str("2016/2017") == "2016/17"
    assert dl.normalise_season_str("2022/23") == "2022/23"


def test_normalise_team_names_maps_b365_to_fpl():
    df = pd.DataFrame({"home_team": ["Man City"], "away_team": ["Spurs"]})
    out = dl.normalise_team_names(df)
    assert out.loc[0, "home_team"] == "Manchester City"
    assert out.loc[0, "away_team"] == "Tottenham Hotspur"


# ---------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------
def test_load_clean_odds_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        dl.load_clean_odds(tmp_path / "missing.csv")


def test_load_clean_odds_validates_schema_and_normalises(tmp_path):
    df = pd.DataFrame(
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
    p = _write_csv(tmp_path, "odds.csv", df)

    out = dl.load_clean_odds(p)
    assert out.loc[0, "season"] == "2022/23"
    assert out.loc[0, "home_team"] == "Manchester City"
    assert out.loc[0, "away_team"] == "Tottenham Hotspur"


def test_load_fixtures_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        dl.load_fixtures(tmp_path / "missing.csv")


def test_load_fixtures_validates_schema_and_normalises(tmp_path):
    df = pd.DataFrame(
        {
            "season": ["2022-2023", "2022/23"],
            "gameweek": [1, 1],
            "home_team": ["Man City", "Spurs"],
            "away_team": ["Spurs", "Man City"],
        }
    )
    p = _write_csv(tmp_path, "fixtures.csv", df)

    out = dl.load_fixtures(p)
    assert set(out["season"]) == {"2022/23"}
    assert out.loc[0, "home_team"] in {"Manchester City", "Tottenham Hotspur"}


def test_load_raw_gameweeks_validates_required_columns(tmp_path):
    df = _minimal_raw_gw_df(season="2022/23", gw=1)
    p = _write_csv(tmp_path, "raw_gw.csv", df)

    out = dl.load_raw_gameweeks(p)
    assert len(out) == 1
    # schema enforcement is the key property here
    for c in dl.GAMEWEEK_BASE_COLS:
        assert c in out.columns


def test_load_raw_fixtures_source_extracts_cols(tmp_path):
    df = _minimal_raw_gw_df(season="2022/23", gw=1)
    p = _write_csv(tmp_path, "raw_gw.csv", df)

    out = dl.load_raw_fixtures_source(p)
    assert list(out.columns) == ["season", "gameweek", "team", "opponent", "was_home"]
    assert out.loc[0, "team"] == "Man City"
    assert out.loc[0, "opponent"] == "Spurs"


def test_load_raw_odds_uses_expected_columns(tmp_path):
    df = pd.DataFrame(
        {
            "Division": ["E0"],
            "MatchDate": ["2022-08-06"],
            "HomeTeam": ["Man City"],
            "AwayTeam": ["Spurs"],
            "OddHome": [1.80],
            "OddDraw": [3.60],
            "OddAway": [4.40],
            # extra columns should be ignored by usecols
            "Junk": ["x"],
        }
    )
    p = _write_csv(tmp_path, "raw_odds.csv", df)

    out = dl.load_raw_odds(p)
    assert set(out.columns) == {
        "Division",
        "MatchDate",
        "HomeTeam",
        "AwayTeam",
        "OddHome",
        "OddDraw",
        "OddAway",
    }


def test_load_player_gameweeks_validates_schema_and_normalises(tmp_path):
    df = pd.DataFrame(
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
    p = _write_csv(tmp_path, "player_gameweeks.csv", df)

    out = dl.load_player_gameweeks(p)
    assert len(out) == 2
    assert set(out["season"]) == {"2016/17"}


# ---------------------------------------------------------------------
# Pipeline utilities (pure / offline)
# ---------------------------------------------------------------------
def test_filter_allowed_seasons_keeps_only_allowed():
    df = pd.DataFrame({"season": ["2016/17", "2015/16"], "x": [1, 2]})
    out = dl.filter_allowed_seasons(df)
    assert set(out["season"]) == {"2016/17"}
    assert len(out) == 1


def test_clean_gameweeks_renames_creates_player_id_sorts():
    df = pd.concat(
        [
            _minimal_raw_gw_df(season="2022/23", gw=2),
            _minimal_raw_gw_df(season="2022/23", gw=1),
        ],
        ignore_index=True,
    )

    out = dl.clean_gameweeks(df)

    # total_points -> points
    assert "points" in out.columns
    assert "total_points" not in out.columns

    # opponent -> opponent_team (branch exercised because opponent exists)
    assert "opponent_team" in out.columns

    # player_id created from id
    assert "player_id" in out.columns
    assert out.loc[0, "player_id"] == out.loc[0, "id"]

    # sorted by (player_id, season, gameweek)
    assert list(out["gameweek"]) == sorted(out["gameweek"].tolist())


def test_build_player_gameweeks_writes_processed_csv(tmp_path, monkeypatch):
    _patch_project_paths(monkeypatch, tmp_path)

    raw = _minimal_raw_gw_df(season="2022/23", gw=1)
    raw_path = _write_csv(tmp_path, "raw_gameweeks.csv", raw)

    out_path = dl.build_player_gameweeks(raw_path=raw_path)
    assert out_path.exists()

    df_out = pd.read_csv(out_path)
    assert "points" in df_out.columns
    assert "player_id" in df_out.columns


def test_build_fixtures_writes_processed_csv(tmp_path, monkeypatch):
    _patch_project_paths(monkeypatch, tmp_path)

    raw = _minimal_raw_gw_df(season="2022/23", gw=1)
    raw_path = _write_csv(tmp_path, "raw_gameweeks.csv", raw)

    out_path = dl.build_fixtures(raw_path=raw_path)
    assert out_path.exists()

    fixtures = pd.read_csv(out_path)
    assert {"season", "gameweek", "home_team", "away_team"}.issubset(fixtures.columns)
    # was_home True -> team becomes home_team
    assert fixtures.loc[0, "home_team"] == "Man City"
    assert fixtures.loc[0, "away_team"] == "Spurs"


def test_assign_season_column_filters_out_of_range_and_labels():
    df = pd.DataFrame(
        {
            "MatchDate": ["2022-08-10", "2015-08-10"],  # one in range, one out
            "Division": ["E0", "E0"],
            "HomeTeam": ["A", "B"],
            "AwayTeam": ["C", "D"],
            "OddHome": [2.0, 2.0],
            "OddDraw": [3.0, 3.0],
            "OddAway": [4.0, 4.0],
        }
    )

    out = dl.assign_season_column(df)
    assert "season" in out.columns
    assert set(out["season"]) == {"2022/23"}
    assert len(out) == 1


def test_run_odds_pipeline_end_to_end_offline(tmp_path, monkeypatch):
    _patch_project_paths(monkeypatch, tmp_path)

    raw_odds = pd.DataFrame(
        {
            "Division": ["E0", "E1"],  # E1 should be filtered out
            "MatchDate": ["2022-08-06", "2022-08-06"],
            "HomeTeam": ["Man City", "Man City"],
            "AwayTeam": ["Spurs", "Spurs"],
            "OddHome": [1.80, 1.80],
            "OddDraw": [3.60, 3.60],
            "OddAway": [4.40, 4.40],
        }
    )
    raw_path = _write_csv(tmp_path, "oddsdataset.csv", raw_odds)

    out_path = dl.run_odds_pipeline(raw_path=raw_path)
    assert out_path.exists()

    odds = pd.read_csv(out_path)
    # only E0 remains, season assigned, implied probs computed + normalised
    assert len(odds) == 1
    assert odds.loc[0, "season"] == "2022/23"
    assert {"pnorm_home_win", "pnorm_draw", "pnorm_away_win"}.issubset(odds.columns)

    # quick sanity: normalised probabilities sum to ~1
    s = float(odds.loc[0, "pnorm_home_win"] + odds.loc[0, "pnorm_draw"] + odds.loc[0, "pnorm_away_win"])
    assert abs(s - 1.0) < 1e-9

