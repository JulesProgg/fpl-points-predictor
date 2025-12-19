"""
Odds data cleaning pipeline for the FPL Points Predictor project.

Steps:
- Load the raw odds dataset from data/raw/oddsdataset.csv
- Keep only the relevant columns (Division, date, teams, Bet365 1N2 odds)
- Filter to Premier League (EPL) and seasons 2016-17 to 2022-23
- Assign a 'season' label consistent with the FPL data (e.g. "2016/17")
- Rename columns to a clean, consistent naming scheme
- Compute implied and normalised probabilities from odds
- Save the cleaned dataset to data/processed/bet365odds_epl_2016_23.csv
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
OUT_FILE = DATA_PROCESSED_DIR / "bet365odds_epl_2016_23.csv"

# Premier League code in the ‘Division’ column
EPL_CODE = "E0"

# Season ranges (inclusive) to match your FPL seasons
SEASON_RANGES = [
    ("2016-08-01", "2017-06-30", "2016/17"),
    ("2017-08-01", "2018-06-30", "2017/18"),
    ("2018-08-01", "2019-06-30", "2018/19"),
    ("2019-08-01", "2020-06-30", "2019/20"),
    ("2020-08-01", "2021-06-30", "2020/21"),
    ("2021-08-01", "2022-06-30", "2021/22"),
    ("2022-08-01", "2023-06-30", "2022/23"),
]


def _assign_season_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'season' column to the odds dataframe based on MatchDate.

    The season labels are consistent with the FPL data:
    "2016/17", "2017/18", ..., "2022/23".
    Rows outside these date ranges get season = NaN and are dropped later.
    """
    df = df.copy()

    # Ensure MatchDate is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["MatchDate"]):
        df["MatchDate"] = pd.to_datetime(df["MatchDate"])

    df["season"] = pd.NA

    for start, end, season_label in SEASON_RANGES:
        mask = df["MatchDate"].between(start, end)
        df.loc[mask, "season"] = season_label

    # Keep only rows within our 7 seasons
    df = df[df["season"].notna()].copy()

    return df


def run_odds_pipeline() -> Path:
    """
    Run the full odds pipeline:
    - load the raw odds dataset
    - select useful columns
    - filter Premier League
    - restrict to seasons 2016-17 to 2022-23
    - assign a FPL-style 'season' label
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

    # 3) Filter : Premier League only
    odds = odds[odds["Division"] == EPL_CODE].copy()

    # 4) Assign season labels (2016/17 -> 2022/23) and drop others
    odds = _assign_season_column(odds)

    # 5) Rename the columns to have something clean
    odds = odds.rename(
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
    odds["home_team"] = odds["home_team"].replace(team_fix)
    odds["away_team"] = odds["away_team"].replace(team_fix)

    # 6) Implied raw probabilities (based on odds)
    odds["p_home_implied"] = 1.0 / odds["home_win_odds"]
    odds["p_draw_implied"] = 1.0 / odds["draw_odds"]
    odds["p_away_implied"] = 1.0 / odds["away_win_odds"]

    total = odds["p_home_implied"] + odds["p_draw_implied"] + odds["p_away_implied"]

    # 7) Normalised probabilities (which sum exactly to 1)
    odds["pnorm_home_win"] = odds["p_home_implied"] / total
    odds["pnorm_draw"] = odds["p_draw_implied"] / total
    odds["pnorm_away_win"] = odds["p_away_implied"] / total

    # 8) Save to data/processed
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    odds.to_csv(OUT_FILE, index=False)

    return OUT_FILE


if __name__ == "__main__":
    output_path = run_odds_pipeline()
    print(f"Odds pipeline completed. File saved to: {output_path}")

