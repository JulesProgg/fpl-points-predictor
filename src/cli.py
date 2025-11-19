import typer

from src.model import predict_points
from src.data_pipeline import run_pipeline

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


if __name__ == "__main__":
    app()
