import pandas as pd
import pytest
from pathlib import Path
from src import odds_pipeline


def test_run_odds_pipeline_creates_output(tmp_path, monkeypatch):
    """
    run_odds_pipeline must :
    - read the raw odds CSV
    - filter EPL + 24-25
    - compute implied & normalised probabilities
    - write a CSV into data/processed/
    - return the output path
    """

    # 1) Fake input dataframe (2 matches, only one EPL)
    fake_df = pd.DataFrame({
        "Division": ["E0", "F1"],
        "MatchDate": ["2024-08-20", "2024-08-20"],
        "HomeTeam": ["Arsenal", "Lyon"],
        "AwayTeam": ["Chelsea", "PSG"],
        "OddHome": [2.0, 3.0],
        "OddDraw": [3.5, 3.2],
        "OddAway": [3.8, 2.1],
    })

    # 2) Écrire ce fake_df dans un vrai CSV temporaire
    raw_path = tmp_path / "odds_fake.csv"
    fake_df.to_csv(raw_path, index=False)

    # 3) Rediriger RAW_ODDS_FILE vers ce fichier temporaire
    monkeypatch.setattr(odds_pipeline, "RAW_ODDS_FILE", raw_path)

    # 4) Rediriger DATA_PROCESSED_DIR vers un dossier temporaire
    monkeypatch.setattr(odds_pipeline, "DATA_PROCESSED_DIR", tmp_path)

    # 5) Lancer le pipeline
    out_path = odds_pipeline.run_odds_pipeline()

    # 6) Vérifier que le chemin retourné existe bien et est un CSV
    assert out_path.exists()
    assert out_path.suffix == ".csv"

    # 7) Charger le fichier écrit et inspecter les colonnes
    written = pd.read_csv(out_path)

    expected_cols = {
        "match_date", "home_team", "away_team",
        "home_win_odds", "draw_odds", "away_win_odds",
        "p_home_implied", "p_draw_implied", "p_away_implied",
        "pnorm_home_win", "pnorm_draw", "pnorm_away_win",
    }

    assert expected_cols.issubset(set(written.columns))

    # 8) Vérifier que les probabilités normalisées somment à 1
    sums = (
        written["pnorm_home_win"]
        + written["pnorm_draw"]
        + written["pnorm_away_win"]
    )
    # We only have one EPL match (Division == ‘E0’), so only one line in “written”.
    assert pytest.approx(float(sums.iloc[0]), rel=1e-5) == 1.0
