
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
import os

from src.data_loader import (
    ALLOWED_SEASONS,
    normalise_season_str,
    build_player_gameweeks,
    build_fixtures,
    run_odds_pipeline,
    build_player_gameweeks_raw_from_kaggle,
)
from src.models import predict_gw_all_players

from src.evaluation import (
    build_team_strength_table,
    compare_model_vs_bookmakers,
    print_example_matches,
    evaluate_linear_gw_model_lag3,
    evaluate_linear_gw_model_lag5,
    evaluate_linear_gw_model_lag10,
    evaluate_linear_gw_model_seasonal,
    evaluate_gbm_gw_model_seasonal,
    evaluate_gw_baseline_lag1,
)

from src.reporting import export_gw_results


app = typer.Typer(add_completion=False, help="FPL Points Predictor – Single entry point")

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def _norm_season_or_exit(season: str) -> str:
    s = normalise_season_str(season)
    if s not in ALLOWED_SEASONS:
        typer.echo(
            f"ERROR: season {season!r} normalized to {s!r} is not supported.\n"
            f"Supported seasons: {', '.join(ALLOWED_SEASONS)}"
        )
        raise typer.Exit(code=1)
    return s


def _print_df(df: pd.DataFrame, head: int = 10) -> None:
    if df is None or df.empty:
        typer.echo("No rows to display.")
        return
    with pd.option_context("display.max_columns", 200, "display.width", 200):
        typer.echo(df.head(head).to_string(index=False))


def _save_df(df: pd.DataFrame, out: Optional[str]) -> None:
    if not out:
        return
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    typer.echo(f"Saved: {out_path}")


# ---------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------
@app.command()
def predict(
    season: str = typer.Option(..., "--season", help='Season, e.g. "2022/23".'),
    gw: Optional[int] = typer.Option(None, "--gw", min=1, help="Gameweek number (omit for full season)."),
    model: str = typer.Option("gw_seasonal_gbm", "--model", help="Model key."),
    out: Optional[str] = typer.Option(None, "--out", help="Output CSV path."),
    head: int = typer.Option(10, "--head", min=1, help="Rows to display."),
):
    """
    Predict FPL points for all players for a given season (and optional gameweek).
    """
    test_season = _norm_season_or_exit(season)

    try:
        df = predict_gw_all_players(model=model, test_season=test_season, gameweek=gw)
    except Exception as e:
        typer.echo(f"ERROR: prediction failed: {e}")
        raise typer.Exit(code=1)

    _print_df(df, head=head)
    _save_df(df, out)


@app.command("evaluate")
def evaluate(
    season: str = typer.Option("2022/23", "--season", help='Held-out season for evaluation, e.g. "2022/23".'),
    model: str = typer.Option(
        "gw_seasonal_gbm",
        "--model",
        help="Model key to evaluate. Supported: gw_baseline_lag1, gw_lag3, gw_lag5, gw_lag10, gw_seasonal_linear, gw_seasonal_gbm",
    ),
):
    """
    Evaluate a GW-level model on a held-out season using MAE.
    """
    test_season = _norm_season_or_exit(season)

    try:
        if model == "gw_baseline_lag1":
            mae = evaluate_gw_baseline_lag1(test_season=test_season)
        elif model == "gw_lag3":
            mae = evaluate_linear_gw_model_lag3(test_season=test_season)
        elif model == "gw_lag5":
            mae = evaluate_linear_gw_model_lag5(test_season=test_season)
        elif model == "gw_lag10":
            mae = evaluate_linear_gw_model_lag10(test_season=test_season)
        elif model in {"gw_seasonal_linear", "gw_seasonal"}:
            mae = evaluate_linear_gw_model_seasonal(test_season=test_season)
        elif model == "gw_seasonal_gbm":
            mae = evaluate_gbm_gw_model_seasonal(test_season=test_season)
        else:
            raise ValueError(f"Unknown evaluation model: {model!r}")
    except Exception as e:
        typer.echo(f"ERROR: evaluation failed: {e}")
        raise typer.Exit(code=1)

    typer.echo(f"MAE ({model}, test_season={test_season}): {mae:.4f}")


@app.command("compare-bookmakers")
def compare_bookmakers(
    season: str = typer.Option(..., "--season", help='Season, e.g. "2022/23".'),
    model: str = typer.Option("gw_seasonal_gbm", "--model", help="Predictive model key."),
    out: Optional[str] = typer.Option(None, "--out", help="Output CSV path for match table."),
    examples: int = typer.Option(10, "--examples", min=0, help="Number of example matches to print."),
):
    """
    Compare model-implied home-win probabilities vs Bet365 (match-by-match) for a season.
    """
    test_season = _norm_season_or_exit(season)

    try:
        comp, mae, corr = compare_model_vs_bookmakers(model=model, test_season=test_season)
    except Exception as e:
        typer.echo(f"ERROR: bookmaker comparison failed: {e}")
        raise typer.Exit(code=1)

    typer.echo(f"Match-by-match comparison vs Bet365 – Season {test_season}, model={model}")
    typer.echo(f"Mean Absolute Error (model vs Bet365 home-win prob): {mae:.3f}")
    typer.echo(f"Correlation (model vs Bet365 home-win prob): {corr:.3f}")

    if examples > 0:
        print_example_matches(comp, n=examples)

    _save_df(comp, out)


@app.command("team-strength")
def team_strength(
    out: Optional[str] = typer.Option(None, "--out", help="Output CSV path."),
    head: int = typer.Option(20, "--head", min=1, help="Rows to display."),
):
    """
    Build and display the aggregated Bet365 team strength table across all seasons.
    """
    try:
        df = build_team_strength_table()
    except Exception as e:
        typer.echo(f"ERROR: team strength failed: {e}")
        raise typer.Exit(code=1)

    _print_df(df, head=head)
    _save_df(df, out)


@app.command("build-data")
def build_data(
    kaggle: bool = typer.Option(False, "--kaggle", help="Download and build raw GW dataset from Kaggle."),
    player_gameweeks: bool = typer.Option(False, "--player-gameweeks", help="Build processed player_gameweeks.csv from raw."),
    fixtures: bool = typer.Option(False, "--fixtures", help="Build processed fixtures file from raw."),
    odds: bool = typer.Option(False, "--odds", help="Run odds pipeline to build cleaned odds dataset."),
):
    """
    Build datasets under data/raw and data/processed.

    Examples:
      - Build everything (typical first run):
          python main.py build-data --kaggle --player-gameweeks --fixtures --odds

      - Rebuild only processed datasets (assuming raw already exists):
          python main.py build-data --player-gameweeks --fixtures --odds
    """
    # If user passes no flags, do the sensible default: build processed datasets
    if not any([kaggle, player_gameweeks, fixtures, odds]):
        player_gameweeks = True
        fixtures = True
        odds = True

    try:
        if kaggle:
            p = build_player_gameweeks_raw_from_kaggle()
            typer.echo(f"Built raw gameweeks from Kaggle: {p}")

        if player_gameweeks:
            p = build_player_gameweeks()
            typer.echo(f"Built processed player_gameweeks: {p}")

        if fixtures:
            p = build_fixtures()
            typer.echo(f"Built processed fixtures: {p}")

        if odds:
            p = run_odds_pipeline()
            typer.echo(f"Built processed odds: {p}")

    except Exception as e:
        typer.echo(f"ERROR: build-data failed: {e}")
        raise typer.Exit(code=1)


@app.command("list")
def list_info(
    what: str = typer.Argument(..., help="What to list: seasons or models"),
):
    """
    List supported seasons and model keys (CLI discovery).
    """
    if what == "seasons":
        typer.echo("\n".join(ALLOWED_SEASONS))
        return

    if what == "models":
        typer.echo(
            "\n".join(
                [
                    # predictive keys (src.models)
                    "gw_lag3",
                    "gw_lag5",
                    "gw_lag10",
                    "gw_seasonal_linear (alias: gw_seasonal)",
                    "gw_seasonal_gbm",
                    # evaluation-only keys (main.py mapping)
                    "gw_baseline_lag1 (evaluation only)",
                ]
            )
        )
        return

    typer.echo("ERROR: unknown list target. Use: seasons or models")
    raise typer.Exit(code=1)


from pathlib import Path

# (tu as déjà ajouté ces imports)
# from src.reporting import (
#     export_gw_results,
#     export_gbm_results_by_position,
#     export_gbm_error_tables,
# )

@app.command("export-gw-results")
def export_gw_results_cmd(
    season: str = typer.Option("2022/23", "--season", help='Held-out season for evaluation, e.g. "2022/23".'),
    output_dir: str = typer.Option("results", "--output-dir", help="Output directory for metrics/figures/tables."),
    with_position_breakdown: bool = typer.Option(
        True, "--with-position-breakdown/--no-position-breakdown", help="Also export GBM metrics/figures by position."
    ),
    with_error_tables: bool = typer.Option(
        True, "--with-error-tables/--no-error-tables", help="Also export top over/under-predicted tables for GBM."
    ),
) -> None:
    """
    Export evaluation metrics + figures to the results/ folder (pred vs actual, residuals, summary bars, etc.).
    """
    test_season = _norm_season_or_exit(season)
    out_dir = Path(output_dir)

    try:
        export_gw_results(test_season=test_season, output_dir=out_dir)
    except Exception as e:
        typer.echo(f"ERROR: export failed: {e}")
        raise typer.Exit(code=1)

    typer.echo(f"Exported metrics/figures to: {out_dir.resolve()}")




if __name__ == "__main__":
    app()

