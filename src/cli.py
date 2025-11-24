import typer

from src.data_pipeline import run_pipeline
from src.model import (
    predict_points,
    evaluate_model,               
    evaluate_position_mean_model,
    evaluate_linear_model,
    evaluate_random_forest_model
)

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
    """
    preds = predict_points(n_gameweeks=n, model=model)
    for i, p in enumerate(preds, start=1):
        print(f"GW{i}: {p:.2f} pts")




@app.command()
def evaluate() -> None:
    """
    Compare several models on the TEST set (MAE).
    """
    mae_baseline = evaluate_model()  # SimpleMeanModel
    mae_position = evaluate_position_mean_model()  # PositionMeanModel
    mae_linear = evaluate_linear_model()  # LinearRegressionModel
    mae_rf = evaluate_random_forest_model()

    print("Compare several models on the TEST set (MAE):")
    print(f"- SimpleMeanModel (baseline):      {mae_baseline:8.3f}")
    print(f"- PositionMeanModel (by position): {mae_position:8.3f}")
    print(f"- LinearRegressionModel:           {mae_linear:8.3f}")
    print(f"- RandomForestModel:                 {mae_rf:8.3f}")

    # Determining the best model
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
