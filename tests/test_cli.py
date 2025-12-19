from __future__ import annotations

import pandas as pd
import pytest
from pathlib import Path
from typer.testing import CliRunner

import src.cli as cli




runner = CliRunner()

# build data

def test_cli_build_data(monkeypatch, tmp_path):
    """
    `build_data` must:
    - call run_pipeline
    - display the returned path
    """

    fake_output = tmp_path / "players_all_seasons.csv"

    def fake_run_pipeline():
        return fake_output

    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    # command name Typer: build-data
    result = runner.invoke(cli.app, ["build-data"])

    assert result.exit_code == 0
    assert "Data pipeline completed" in result.stdout
    assert str(fake_output) in result.stdout


# run


def test_cli_run_baseline(monkeypatch):
    """
    `run --model baseline`:
    - exit_code == 0
    - displays “GW1” and 'pts'
    """

    def fake_predict_points(n_gameweeks: int = 5, model: str = "baseline"):
        return [22.73] * n_gameweeks

    monkeypatch.setattr(cli, "predict_points", fake_predict_points)

    result = runner.invoke(cli.app, ["run", "--model", "baseline"])

    assert result.exit_code == 0
    assert "GW1" in result.stdout
    assert "22.73" in result.stdout
    assert "pts" in result.stdout


def test_cli_run_linear(monkeypatch):
    """
    The same applies to --model linear, with another value to verify
    that the correct model name is being passed
    """

    def fake_predict_points(n_gameweeks: int = 5, model: str = "baseline"):
        if model == "linear":
            return [33.33] * n_gameweeks
        return [0.0] * n_gameweeks

    monkeypatch.setattr(cli, "predict_points", fake_predict_points)

    result = runner.invoke(cli.app, ["run", "--model", "linear"])

    assert result.exit_code == 0
    assert "33.33" in result.stdout

def test_cli_show_bookmakers(monkeypatch):
    """
    `show-bookmakers` must:
    - call build_team_strength_table
    - exit with code 0
    - display team names and strengths
    """

    fake_df = pd.DataFrame({
        "team": ["Arsenal", "Chelsea"],
        "bet365_strength": [0.6, 0.5],
        "n_matches": [38, 38],
    })

    # We replace build_team_strength_table in cli with our fake one.
    monkeypatch.setattr(cli, "build_team_strength_table", lambda: fake_df)

    result = runner.invoke(cli.app, ["show-bookmakers"])

    assert result.exit_code == 0
    assert "Bookmaker (Bet365) Team Strength" in result.stdout
    assert "Arsenal" in result.stdout
    assert "Chelsea" in result.stdout
    # 0.6 formatté en 3 décimales → 0.600
    assert "0.600" in result.stdout
    assert "0.500" in result.stdout


# evaluate


def test_cli_evaluate_uses_all_models(monkeypatch):
    """
    `evaluate` must:
    - complete without errors
    - display one line for each model
We monkey-patch the evaluation functions so that the real models are not launched
    """

    monkeypatch.setattr(cli, "evaluate_model", lambda: 10.0)
    monkeypatch.setattr(cli, "evaluate_position_mean_model", lambda: 9.5)
    monkeypatch.setattr(cli, "evaluate_linear_model", lambda: 8.0)
    monkeypatch.setattr(cli, "evaluate_random_forest_model", lambda: 4.0)

    result = runner.invoke(cli.app, ["evaluate"])

    assert result.exit_code == 0

    # model names

    assert "SimpleMeanModel (baseline)" in result.stdout
    assert "PositionMeanModel (by position)" in result.stdout
    assert "LinearRegressionModel" in result.stdout
    assert "RandomForestModel" in result.stdout

    # verify that our dummy MAEs appear correctly
    
    for v in ["10.0", "9.5", "8.0", "4.0"]:
        assert any(v in line for line in result.stdout.splitlines())
