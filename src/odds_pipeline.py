"""
Odds data cleaning pipeline for the FPL Points Predictor project.

Steps:
- Load the raw odds dataset from data/raw/oddsdataset.csv
- Keep only the relevant columns (Division, date, teams, Bet365 1N2 odds)
- Filter to Premier League (EPL) and season 2024–25
- Rename columns to a clean, consistent naming scheme
- Compute implied and normalised probabilities from odds
- Save the cleaned dataset to data/processed/bet365_epl_24_25.csv
"""

from pathlib import Path
import pandas as pd

# Path to the project root folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Raw and processed data folders
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Input / output files
RAW_ODDS_FILE = DATA_RAW_DIR / "oddsdataset.csv"
OUT_FILE = DATA_PROCESSED_DIR / "bet365odds_season_24_25.csv"

# Premier League code in the ‘Division’ column
EPL_CODE = "E0"


def run_odds_pipeline() -> Path:
    """
    Run the full odds pipeline:
    - load the raw odds dataset
    - select useful columns
    - filter Premier League 2024-25
    - rename columns
    - compute implied & normalised probabilities
    - save to data/processed/

    Returns
    -------
    Path
        Path to the saved CSV file.
    """

    # 1) Load only the useful columns
    use_cols = [
        "Division",
        "MatchDate",
        "HomeTeam",
        "AwayTeam",
        "OddHome",
        "OddDraw",
        "OddAway",
    ]

    odds = pd.read_csv(RAW_ODDS_FILE, usecols=use_cols, low_memory=False)

    # 2) Date conversion
    odds["MatchDate"] = pd.to_datetime(odds["MatchDate"])

    # 3) Filter : Premier League + season 2024-25
    mask_div = odds["Division"] == EPL_CODE
    mask_date = odds["MatchDate"].between("2024-08-01", "2025-06-30")
    epl = odds[mask_div & mask_date].copy()

    # 4) Rename the columns to have something clean
    epl = epl.rename(
        columns={
            "MatchDate": "match_date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "OddHome": "home_win_odds",
            "OddDraw": "draw_odds",
            "OddAway": "away_win_odds",
        }
    )
    # Normalise team names (e.g. Nottingham Forest variants)
    team_fix = {
        "Nott'm Forest": "Nottingham Forest",
        "Nottm Forest": "Nottingham Forest",
    }
    epl["home_team"] = epl["home_team"].replace(team_fix)
    epl["away_team"] = epl["away_team"].replace(team_fix)

    # 5) Implicit raw probabilities (based on odds)
    epl["p_home_implied"] = 1.0 / epl["home_win_odds"]
    epl["p_draw_implied"] = 1.0 / epl["draw_odds"]
    epl["p_away_implied"] = 1.0 / epl["away_win_odds"]

    total = epl["p_home_implied"] + epl["p_draw_implied"] + epl["p_away_implied"]

    # 6) Normalised probabilities (which sum exactly to 1)
    epl["pnorm_home_win"] = epl["p_home_implied"] / total
    epl["pnorm_draw"] = epl["p_draw_implied"] / total
    epl["pnorm_away_win"] = epl["p_away_implied"] / total

    # 7) Save to data/processed
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    epl.to_csv(OUT_FILE, index=False)

    return OUT_FILE


if __name__ == "__main__":
    output_path = run_odds_pipeline()
    print(f"Odds pipeline completed. File saved to: {output_path}")
