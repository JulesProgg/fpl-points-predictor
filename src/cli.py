import typer
from src.model import predict_points

app = typer.Typer(help="FPL Points Predictor CLI")

@app.command()
def run(n: int = 5):
    """
    Exécute des prédictions de test.
    """
    preds = predict_points(n)
    for i, p in enumerate(preds, start=1):
        print(f"GW{i}: {p:.2f} pts")

if __name__ == "__main__":
    app()