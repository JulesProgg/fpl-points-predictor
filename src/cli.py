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

from src.bookmaker_benchmark import (
    build_team_strength_table,
    compare_model_vs_bookmakers,
)

app = typer.Typer(help="FPL Points Predictor CLI – Gameweek-level models (2016–2023)")


# ----------------------------------------------------------------------
# 1) BUILD DATA
# ----------------------------------------------------------------------
@app.command()
def build_data() -> None:
    """
    Run the full gameweek-level data pipeline and display the output file path.

    Output:
        data/processed/player_gameweeks.csv

    Content:
        One row per (player, gameweek), seasons 2016/17 → 2022/23.
        No lag features (constructed dynamically in model.py).
    """
    output_path = run_pipeline()
    typer.echo(f"Data pipeline completed. Clean GW file saved to: {output_path}")


# ----------------------------------------------------------------------
# 2) RUN PREDICTIONS FOR ONE GW
# ----------------------------------------------------------------------
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
        help="Season to predict (e.g. '2021/22', '2022/23').",
    ),
    gameweek: int = typer.Option(
        1,
        help="Gameweek to predict (e.g. 1, 5, 12).",
    ),
    top_n: int = typer.Option(
        20,
        help="Display top-N players sorted by predicted points.",
    ),
) -> None:
    """
    Predict FPL points for EVERY PLAYER for a chosen season and gameweek.

    Example:
        python -m src.cli run --model gw_seasonal_gbm --test-season '2022/23' --gameweek 10
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
            "Expected: gw_lag3, gw_lag5, gw_lag10, gw_seasonal_linear, gw_seasonal_gbm."
        )

    df_preds = predict_gw_all_players(
        model=model,
        test_season=test_season,
        gameweek=gameweek,
    )

    if df_preds.empty:
        typer.echo(
            f"No predictions for season={test_season}, GW={gameweek}, model={model}. "
            "Likely due to insufficient lag history."
        )
        return

    df_sorted = df_preds.sort_values("predicted_points", ascending=False).head(top_n)

    typer.echo(
        f"Predicted FPL points per player – Season {test_season}, GW{gameweek}, model={model}"
    )
    typer.echo("-" * 80)

    for _, row in df_sorted.iterrows():
        name = str(row["name"])
        team = str(row["team"])
        position = str(row["position"])
        gw = int(row["gameweek"])
        pred = float(row["predicted_points"])
        typer.echo(f"{name:<25} {team:<18} {position:<4} GW{gw:>2} → {pred:5.2f} pts")


# ----------------------------------------------------------------------
# 3) SHOW BOOKMAKER STRENGTH (MULTI-SEASON)
# ----------------------------------------------------------------------
@app.command()
def show_bookmakers() -> None:
    """
    Display Bet365 team strength aggregated across all EPL matches
    from seasons 2016/17 to 2022/23.

    Data source:
        data/processed/bet365odds_epl_2016_23.csv
    """
    df = build_team_strength_table()

    typer.echo("Bet365 Team Strength – EPL (2016–2023 Aggregate)")
    typer.echo("-" * 55)

    for _, row in df.iterrows():
        team = str(row["team"])
        strength = float(row["bet365_strength"])
        typer.echo(f"{team:<20} {strength:.3f}")


# ----------------------------------------------------------------------
# 4) EVALUATE GAMEWEEK MODELS
# ----------------------------------------------------------------------
@app.command()
def evaluate_gw(test_season: str = "2022/23") -> None:
    """
    Compare several GW-level models on a TEST season (MAE).

    TRAIN = all seasons except test_season (2016/17 → 2022/23)
    TEST  = the provided test_season.

    Target:
        points per player per gameweek.
    """
    mae_baseline = evaluate_gw_baseline_lag1(test_season=test_season)
    mae_gw_lag3 = evaluate_linear_gw_model_lag3(test_season=test_season)
    mae_gw_lag5 = evaluate_linear_gw_model_lag5(test_season=test_season)
    mae_gw_lag10 = evaluate_linear_gw_model_lag10(test_season=test_season)
    mae_gw_seasonal = evaluate_linear_gw_model_seasonal(test_season=test_season)
    mae_gw_seasonal_gbm = evaluate_gbm_gw_model_seasonal(test_season=test_season)

    typer.echo(f"GW model comparison (MAE) – TEST season {test_season}:")
    typer.echo(f"- Baseline lag1:                         {mae_baseline:8.3f}")
    typer.echo(f"- Linear lag3 (anytime):                 {mae_gw_lag3:8.3f}")
    typer.echo(f"- Linear lag5 (anytime):                 {mae_gw_lag5:8.3f}")
    typer.echo(f"- Linear lag10 (anytime):                {mae_gw_lag10:8.3f}")
    typer.echo(f"- Linear (seasonal except GW1):          {mae_gw_seasonal:8.3f}")
    typer.echo(f"- GBM (seasonal except GW1):             {mae_gw_seasonal_gbm:8.3f}")

    maes = {
        "Baseline lag1": mae_baseline,
        "Linear lag3": mae_gw_lag3,
        "Linear lag5": mae_gw_lag5,
        "Linear lag10": mae_gw_lag10,
        "Linear seasonal": mae_gw_seasonal,
        "GBM seasonal": mae_gw_seasonal_gbm,
    }
    best_name = min(maes, key=maes.get)
    typer.echo(f"→ Best GW model: {best_name}")



# ----------------------------------------------------------------------
# COMPARISON BETWEEN GBM AND BOOKMAKER
# ----------------------------------------------------------------------

@app.command()
def compare_vs_bookmakers(
    model: str = typer.Option(
        "gw_seasonal_gbm",
        help=(
            "GW model used to derive team strength per match: "
            "gw_lag3, gw_lag5, gw_lag10, gw_seasonal_linear, gw_seasonal_gbm."
        ),
    ),
    test_season: str = typer.Option(
        "2022/23",
        help="Season to compare against Bet365 (e.g. '2021/22', '2022/23').",
    ),
    max_rows: int = typer.Option(
        20,
        help="How many matches to display in the console (for inspection).",
    ),
) -> None:
    """
    Compare, MATCH BY MATCH, the model-implied home-win probability
    vs Bet365 home-win probability for a given season.

    The model's team strength is computed by summing predicted player points
    per team and gameweek, then turning it into a probability:
        p_model_home_win = home_strength / (home_strength + away_strength)

    Output:
        - Prints the MAE between model and Bet365 probabilities.
        - Displays a few example matches with both probabilities.
    """
    valid_models = {
        "gw_lag3",
        "gw_lag5",
        "gw_lag10",
        "gw_seasonal_linear",
        "gw_seasonal_gbm",
    }
    if model not in valid_models:
        raise typer.BadParameter(
            f"Unknown model: {model!r}. "
            "Expected one of: gw_lag3, gw_lag5, gw_lag10, "
            "gw_seasonal_linear, gw_seasonal_gbm."
        )

    comp, mae = compare_model_vs_bookmakers(model=model, test_season=test_season)

    typer.echo(
        f"Match-by-match comparison vs Bet365 – Season {test_season}, model={model}"
    )
    typer.echo(f"Mean Absolute Error (model vs Bet365 home-win prob): {mae:.3f}\n")

    typer.echo("Example matches:")
    typer.echo("-" * 80)
    for _, row in comp.head(max_rows).iterrows():
        gw = int(row["gameweek"])
        home = str(row["home_team"])
        away = str(row["away_team"])
        p_b365 = float(row["pnorm_home_win"])
        p_model = float(row["p_model_home_win"])
        err = float(row["abs_error"])
        typer.echo(
            f"GW{gw:>2}  {home:<20} vs {away:<20}  "
            f"Bet365: {p_b365:.3f}  |  Model: {p_model:.3f}  |  |Δ|={err:.3f}"
        )


if __name__ == "__main__":
    app()
