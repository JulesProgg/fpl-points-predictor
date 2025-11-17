import sys
from pathlib import Path

# ajoute la racine du projet (/files/fpl-points-predictor) au PYTHONPATH
PROJECT_ROOT = Path("/files/fpl-points-predictor")
sys.path.append(str(PROJECT_ROOT))

import src.data_pipeline as dp

def test_select_useful_drops_unwanted_columns():
    """select_useful must drop unuseful columns and keep the ordrer of USEFUL_COLS."""
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["Player A", "Player B"],
            "team": ["Team X", "Team Y"],
            "position": ["FWD", "MID"],
            "minutes": [1000, 2000],
            "junk": ["foo", "bar"],  # unuseful column
        }
    )

    result = dp.select_useful(df)

    # 1) the columns"junk" must disappear 
    assert "junk" not in result.columns

    # 2) the existing columns of df should be in the order defined in USEFUL_COLS,
    expected_cols = [c for c in dp.USEFUL_COLS if c in df.columns]
    assert list(result.columns) == expected_cols

    def test_select_useful_handles_missing_useful_columns():
    """select_useful should work even if some columns don't exist"""
    df = pd.DataFrame(
        {
            "name": ["Player A"],
            "team": ["Team X"],
            # only this two columns are fulfilled
        }
    )

    result = dp.select_useful(df)

    # it must keep ONLY the defined columns in USEFUL_COLS
    assert list(result.columns) == ["name", "team"]

    def test_run_pipeline_reads_three_csvs_and_writes_output(tmp_path, monkeypatch):
    """
    run_pipeline must :
    - read 3 CSV (one season each),
    - normalise the dataset 24-25 (first/second → name, etc.),
    - concatenate 3 of them,
    - write a CSSV in DATA_PROCESSED_DIR,
    - return the path of the file.
    We mock pd.read_csv and redirect DATA_PROCESSED_DIR to a temporary directory.
    """

    # 1) Prepare fake DataFrames for each season
    df22 = pd.DataFrame(
        {
            "id": [1],
            "name": ["Player 22"],
            "team": ["Team 22"],
            "position": ["FWD"],
            "minutes": [1000],
            "total_points": [150],
        }
    )

    df23 = pd.DataFrame(
        {
            "id": [2],
            "name": ["Player 23"],
            "team": ["Team 23"],
            "position": ["MID"],
            "minutes": [2000],
            "total_points": [160],
        }
    )

    # Dataset 24-25 with the special structure which we normalise with run_pipeline
    df24 = pd.DataFrame(
        {
            "id": [3],
            "first_name": ["Erling"],
            "second_name": ["Haaland"],
            "player_position": ["FWD"],
            "team_name": ["Team 24"],
            "minutes": [2500],
            "total_points": [200],
        }
    )

    # 2) Mock of pd.read_csv to return the above dataFrames
    def fake_read_csv(path, *args, **kwargs):
        path_str = str(path)
        if "season22-23.csv" in path_str:
            return df22.copy()
        if "season23-24.csv" in path_str:
            return df23.copy()
        if "season24-25.csv" in path_str:
            return df24.copy()
        raise ValueError(f"Unexpected path in fake_read_csv: {path_str}")

    monkeypatch.setattr(dp.pd, "read_csv", fake_read_csv)

# 3) Redirect DATA_PROCESSED_DIR to a temporary pytest folder
monkeypatch.setattr(dp, "DATA_PROCESSED_DIR", tmp_path)

# 4) Run the pipeline
out_path = dp.run_pipeline()

# The return path must match the file in tmp_path
expected_path = tmp_path / "players_all_seasons.csv"
assert out_path == expected_path
assert out_path.exists()

# 5) Check the contents of the written CSV file
full = pd.read_csv(out_path)

# There should be 3 lines (one per season)
assert len(full) == 3

# All columns in the CSV must be a subset of USEFUL_COLS
assert all(col in dp.USEFUL_COLS for col in full.columns)

# Player 24-25 must have a constructed name ‘Erling Haaland’ for example
assert "name" in full.columns
assert "Erling Haaland" in full["name"].values