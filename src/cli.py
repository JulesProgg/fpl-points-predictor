import typer

from src.data_pipeline import run_pipeline
from src.model import predict_points, evaluate_model



app = typer.Typer(help="FPL Points Predictor CLI")


@app.command()
def build_data():
    """
    Run the full data pipeline and display the output file path.
    """
    output_path = run_pipeline()
    typer.echo(f"Data pipeline completed. File saved to: {output_path}")


@app.command()
def run(n: int = 5):
    """
    Performs test predictions
    """
    preds = predict_points(n)
    for i, p in enumerate(preds, start=1):
        print(f"GW{i}: {p:.2f} pts")


@app.command()
def evaluate():
    """
    Evaluates the SimpleMeanModel model and displays the MAE
    """
    mae = evaluate_model()
    typer.echo(f"MAE of SimpleMeanModel: {mae:.2f} points")


if __name__ == "__main__":
    app()
