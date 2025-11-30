import typer

from src.data_pipeline import run_pipeline
from src.model import (
    predict_points,
    predict_gw_all_players,
    evaluate_position_mean_model,
    evaluate_linear_model,
    evaluate_gradient_boosting_model,
    evaluate_gw_baseline_lag1, 
    evaluate_linear_gw_model_lag3,
    evaluate_linear_gw_model_lag5,
    evaluate_linear_gw_model_lag10,
    evaluate_linear_gw_model_seasonal,
    evaluate_gbm_gw_model_seasonal,
    
)

from src.bookmaker_benchmark import build_team_strength_table

app = typer.Typer(help="FPL Points Predictor CLI")


@app.command()
def build_data():
    """
    Run the full data pipeline and display the output file path.
    """
    output_path = run_pipeline()
    typer.echo(f"Data pipeline completed. File saved to: {output_path}")


@app.command()
def run(
    n: int = 5,
    model: str = typer.Option(
        "position",
        help=(
            "Model to use for predictions: "
            "position, linear, gbm, "
            "gw_lag3, gw_lag5, gw_lag10, "
            "gw_seasonal_linear, gw_seasonal_gbm"
        ),
    ),
    test_season: str = typer.Option(
        "2023/24",
        help="Season to use for GW-level models (e.g. '2023/24').",
    ),
    gameweek: int = typer.Option(
        1,
        help="Gameweek to predict for (GW-level models). Ignored for season-level models.",
    ),
) -> None:
    """
    For season-level models (position, linear, gbm):
        - Perform test predictions for n future gameweeks (simple average proxy).

    For gameweek-level models (gw_*):
        - Predict points for EACH player for the given test_season and gameweek.
    """
    # Season-level models
    if model in {"position", "linear", "gbm"}:
        preds = predict_points(n_gameweeks=n, model=model)
        for i, p in enumerate(preds, start=1):
            print(f"GW{i}: {p:.2f} pts")
        return

    # Gameweek-level models: per-player predictions 
    gw_models = {
        "gw_lag3",
        "gw_lag5",
        "gw_lag10",
        "gw_seasonal_linear",
        "gw_seasonal_gbm",
    }

    if model not in gw_models:
        raise typer.BadParameter(f"Unknown model: {model}")

    df_preds = predict_gw_all_players(
        model=model,
        test_season=test_season,
        gameweek=gameweek,
    )

    if df_preds.empty:
        print(
            f"No predictions available for season={test_season!r}, "
            f"gameweek={gameweek}, model={model!r} "
            "(not enough history or no data)."
        )
        return

    print(
        f"Predicted FPL points per player – "
        f"season {test_season}, GW{gameweek}, model={model}"
    )
    print("-" * 80)
    for _, row in df_preds.iterrows():
        name = str(row["name"])
        team = str(row["team"])
        position = str(row["position"])
        gw = int(row["gameweek"])
        pred = float(row["predicted_points"])
        print(f"{name:<25} {team:<18} {position:<4} GW{gw:>2}  ->  {pred:5.2f} pts")



@app.command()
def show_bookmakers() -> None:
    """
    Display Bet365 team strength for the 2024-25 Premier League season.
    """
    df = build_team_strength_table()

    print("Bookmaker (Bet365) Team Strength – Season 2024/25")
    print("-" * 55)
    for _, row in df.iterrows():
        team = str(row["team"])
        strength = float(row["bet365_strength"])
        print(f"{team:<20} {strength:.3f}")


@app.command()
def evaluate() -> None:
    """
    Compare several models on the TEST set (MAE).

    TRAIN = seasons 2022-23 + 2023-24
    TEST  = season 2024-25
    """
    mae_position = evaluate_position_mean_model()        # PositionMeanModel
    mae_linear = evaluate_linear_model()                 # LinearRegressionModel
    mae_gbm = evaluate_gradient_boosting_model()         # GradientBoostingModel

    print("Model comparison on TEST set (MAE – season 2024-25):")
    print(f"- PositionMeanModel (by position): {mae_position:8.3f}")
    print(f"- LinearRegressionModel:           {mae_linear:8.3f}")
    print(f"- GradientBoostingModel:           {mae_gbm:8.3f}")

    maes = {
        "PositionMeanModel (by position)": mae_position,
        "LinearRegressionModel": mae_linear,
        "GradientBoostingModel": mae_gbm,
    }
    best_name = min(maes, key=maes.get)
    print(f"→ Best model: {best_name} ")

@app.command()
def evaluate_gw(test_season: str = "2023/24") -> None:
    """
    Compare several gameweek-level models on a TEST season (MAE).

    TRAIN = all seasons except `test_season`
    TEST  = season `test_season`
    Target: points per player-gameweek.
    """

    mae_baseline = evaluate_gw_baseline_lag1(test_season=test_season)
    mae_gw_lag3 = evaluate_linear_gw_model_lag3(test_season=test_season)
    mae_gw_lag5 = evaluate_linear_gw_model_lag5(test_season=test_season)
    mae_gw_lag10 = evaluate_linear_gw_model_lag10(test_season=test_season)
    mae_gw_seasonal = evaluate_linear_gw_model_seasonal(test_season=test_season)
    mae_gw_seasonal_gbm = evaluate_gbm_gw_model_seasonal(test_season=test_season)

    print(f"Gameweek-level model comparison on TEST season {test_season} (MAE):")
    print(f"- GW Naive baseline (points_lag_1):          {mae_baseline:8.3f}")
    print(f"- GW Linear (last 3 matches, anytime):      {mae_gw_lag3:8.3f}")
    print(f"- GW Linear (last 5 matches, anytime):      {mae_gw_lag5:8.3f}")
    print(f"- GW Linear (last 10 matches, anytime):     {mae_gw_lag10:8.3f}")
    print(f"- GW Linear (seasonal, except GW1):         {mae_gw_seasonal:8.3f}")
    print(f"- GW GBM   (seasonal, except GW1):          {mae_gw_seasonal_gbm:8.3f}")

    maes = {
        "GW Naive baseline (lag1)": mae_baseline,
        "GW Linear (lag3, anytime)": mae_gw_lag3,
        "GW Linear (lag5, anytime)": mae_gw_lag5,
        "GW Linear (lag10, anytime)": mae_gw_lag10,
        "GW Linear (seasonal)": mae_gw_seasonal,
        "GW GBM (seasonal)": mae_gw_seasonal_gbm,
    }
    best_name = min(maes, key=maes.get)
    print(f"→ Best gameweek model: {best_name}")




if __name__ == "__main__":
    app()
