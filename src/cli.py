import typer

from src.data_pipeline import run_pipeline
from src.model import (
    predict_points,
    evaluate_model,               
    evaluate_position_mean_model,
    evaluate_linear_model,
    evaluate_random_forest_model,
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
        "baseline",
        help="Model to use for predictions: baseline, linear, random_forest",
    ),
) -> None:
    """
    Performs test predictions for n future gameweeks using the chosen model.

    Models are trained on seasons 2022-23 and 2023-24.
    """
    preds = predict_points(n_gameweeks=n, model=model)
    for i, p in enumerate(preds, start=1):
        print(f"GW{i}: {p:.2f} pts")


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
    mae_baseline = evaluate_model()                 # SimpleMeanModel
    mae_position = evaluate_position_mean_model()   # PositionMeanModel
    mae_linear = evaluate_linear_model()            # LinearRegressionModel
    mae_rf = evaluate_random_forest_model()         # RandomForestModel

    print("Model comparison on TEST set (MAE – season 2024-25):")
    print(f"- SimpleMeanModel (baseline):      {mae_baseline:8.3f}")
    print(f"- PositionMeanModel (by position): {mae_position:8.3f}")
    print(f"- LinearRegressionModel:           {mae_linear:8.3f}")
    print(f"- RandomForestModel:               {mae_rf:8.3f}")

    maes = {
        "SimpleMeanModel (baseline)": mae_baseline,
        "PositionMeanModel (by position)": mae_position,
        "LinearRegressionModel": mae_linear,
        "RandomForestModel": mae_rf,
    }
    best_name = min(maes, key=maes.get)
    print(f"→ Best model: {best_name} ")


if __name__ == "__main__":
    app()
