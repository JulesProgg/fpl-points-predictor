import typer

from src.data_pipeline import run_pipeline
from src.model import (
    predict_gw_all_players,
    evaluate_gw_baseline_lag1,
    evaluate_linear_gw_model_lag3,
    evaluate_linear_gw_model_lag5,
    evaluate_linear_gw_model_lag10,
    evaluate_linear_gw_model_seasonal,
    evaluate_gbm_gw_model_seasonal,
)

from src.bookmaker_benchmark import build_team_strength_table

app = typer.Typer(help="FPL Points Predictor CLI – gameweek-level models only")


@app.command()
def build_data() -> None:
    """
    Run the full gameweek-level data pipeline and display the output file path.

    Output: data/processed/player_gameweeks_lagged.csv
    (one row per player-gameweek, seasons 2016/17 → 2022/23 only).
    """
    output_path = run_pipeline()
    typer.echo(f"Data pipeline completed. File saved to: {output_path}")


@app.command()
def run(
    model: str = typer.Option(
        "gw_seasonal_gbm",
        help=(
            "GW model to use for predictions: "
            "gw_lag3, gw_lag5, gw_lag10, "
            "gw_seasonal_linear, gw_seasonal_gbm"
        ),
    ),
    test_season: str = typer.Option(
        "2022/23",
        help="Season for GW-level predictions (e.g. '2022/23').",
    ),
    gameweek: int = typer.Option(
        1,
        help="Gameweek to predict for (e.g. 1, 2, 3, ...).",
    ),
    top_n: int = typer.Option(
        20,
        help="Number of top players to display (sorted by predicted points).",
    ),
) -> None:
    """
    Predict FPL points for EACH player for a given season and gameweek,
    using a chosen gameweek-level model (per-player per-gameweek).

    Examples:
        python -m src.cli run --model gw_seasonal_gbm --test-season '2022/23' --gameweek 10
        python -m src.cli run --model gw_lag5 --test-season '2021/22' --gameweek 5
    """
    gw_models = {
        "gw_lag3",
        "gw_lag5",
        "gw_lag10",
        "gw_seasonal_linear",
        "gw_seasonal_gbm",
    }

    if model not in gw_models:
        raise typer.BadParameter(
            f"Unknown model: {model!r}. "
            "Expected one of: gw_lag3, gw_lag5, gw_lag10, "
            "gw_seasonal_linear, gw_seasonal_gbm."
        )

    df_preds = predict_gw_all_players(
        model=model,
        test_season=test_season,
        gameweek=gameweek,
    )

    if df_preds.empty:
        typer.echo(
            f"No predictions available for season={test_season!r}, "
            f"gameweek={gameweek}, model={model!r} "
            "(not enough history or no data)."
        )
        return

    # Trier par points prédits décroissants pour afficher les top_n joueurs
    df_sorted = df_preds.sort_values("predicted_points", ascending=False).head(top_n)

    typer.echo(
        f"Predicted FPL points per player – "
        f"season {test_season}, GW{gameweek}, model={model}"
    )
    typer.echo("-" * 80)
    for _, row in df_sorted.iterrows():
        name = str(row["name"])
        team = str(row["team"])
        position = str(row["position"])
        gw = int(row["gameweek"])
        pred = float(row["predicted_points"])
        typer.echo(f"{name:<25} {team:<18} {position:<4} GW{gw:>2}  ->  {pred:5.2f} pts")


@app.command()
def show_bookmakers() -> None:
    """
    Display Bet365 team strength for the 2024-25 Premier League season.

    This is used as an external benchmark to compare model predictions
    aggregated at team level vs bookmaker expectations.
    """
    df = build_team_strength_table()

    typer.echo("Bookmaker (Bet365) Team Strength – Season 2024/25")
    typer.echo("-" * 55)
    for _, row in df.iterrows():
        team = str(row["team"])
        strength = float(row["bet365_strength"])
        typer.echo(f"{team:<20} {strength:.3f}")


@app.command()
def evaluate_gw(test_season: str = "2022/23") -> None:
    """
    Compare several gameweek-level models on a TEST season (MAE).

    TRAIN = all seasons except `test_season` (2016/17 → 2021/22 by default)
    TEST  = season `test_season` (by default: 2022/23)

    Target: points per player-gameweek.
    """

    mae_baseline = evaluate_gw_baseline_lag1(test_season=test_season)
    mae_gw_lag3 = evaluate_linear_gw_model_lag3(test_season=test_season)
    mae_gw_lag5 = evaluate_linear_gw_model_lag5(test_season=test_season)
    mae_gw_lag10 = evaluate_linear_gw_model_lag10(test_season=test_season)
    mae_gw_seasonal = evaluate_linear_gw_model_seasonal(test_season=test_season)
    mae_gw_seasonal_gbm = evaluate_gbm_gw_model_seasonal(test_season=test_season)

    typer.echo(f"Gameweek-level model comparison on TEST season {test_season} (MAE):")
    typer.echo(f"- GW Naive baseline (points_lag_1):             {mae_baseline:8.3f}")
    typer.echo(f"- GW Linear (last 3 matches, anytime):         {mae_gw_lag3:8.3f}")
    typer.echo(f"- GW Linear (last 5 matches, anytime):         {mae_gw_lag5:8.3f}")
    typer.echo(f"- GW Linear (last 10 matches, anytime):        {mae_gw_lag10:8.3f}")
    typer.echo(f"- GW Linear (seasonal, except GW1):            {mae_gw_seasonal:8.3f}")
    typer.echo(f"- GW GBM   (seasonal, except GW1):             {mae_gw_seasonal_gbm:8.3f}")

    maes = {
        "GW Naive baseline (lag1)": mae_baseline,
        "GW Linear (lag3, anytime)": mae_gw_lag3,
        "GW Linear (lag5, anytime)": mae_gw_lag5,
        "GW Linear (lag10, anytime)": mae_gw_lag10,
        "GW Linear (seasonal)": mae_gw_seasonal,
        "GW GBM (seasonal)": mae_gw_seasonal_gbm,
    }
    best_name = min(maes, key=maes.get)
    typer.echo(f"→ Best gameweek model: {best_name}")


if __name__ == "__main__":
    app()
