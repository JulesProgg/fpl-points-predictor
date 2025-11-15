"""
data_pipeline.py

Simple data cleaning pipeline for the FPL Points Predictor project.

Steps:
- Load the three raw CSV files (one per season) from data/raw/
- Normalise the 2024-25 dataset so columns match the others
- Keep only the relevant columns (USEFUL_COLS)
- Concatenate all three seasons
- Save the final dataset to data/processed/players_all_seasons.csv
"""

from pathlib import Path
import pandas as pd

# Path to the project root folder
PROJECT_ROOT = Path("/files/fpl-points-predictor")

# Raw and processed data folders
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Relevant columns from the raw datasets
USEFUL_COLS = [
    # identification
    "id", "name", "team", "position",

    # raw statistics
    "minutes", "goals_scored", "assists",
    "goals_conceded", "clean_sheets", "saves",
    "penalties_saved", "penalties_missed",
    "yellow_cards", "red_cards", "own_goals",
    "starts", "bonus", "bps",

    # advanced cumulative statistics
    "expected_goals", "expected_assists",
    "expected_goal_involvements",  # not available for 24-25
    "expected_goals_conceded",
    "influence", "creativity", "threat", "ict_index",

    # target
    "total_points",
]

# create useful helper functions
def select_useful(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a new and independent DataFrame containing only the relevant columns
    defined in USEFUL_COLS (keeping only those that exist in df).
    """
    cols = [c for c in USEFUL_COLS if c in df.columns]
    return df[cols].copy()


def run_pipeline() -> Path:
    """
    Run the full pipeline:
    - load the three raw datasets
    - normalise the 2024-25 dataset
    - select useful columns
    - concatenate all seasons
    - save the final dataset to data/processed/

    Returns:
        Path to the saved CSV file.
    """

    # Load the 3 raw season datasets
    df22 = pd.read_csv(DATA_RAW_DIR / "season22-23.csv")
    df23 = pd.read_csv(DATA_RAW_DIR / "season23-24.csv")
    df24 = pd.read_csv(DATA_RAW_DIR / "season24-25.csv")

    # Normalise the 2024-25 dataset so that it matches the others 

    # 1) Rename columns to match the other seasons
    rename_24 = {
        "first_name": "first",
        "second_name": "second",
        "player_position": "position",
        "team_name": "team",
    }
    df24 = df24.rename(columns=rename_24)

    # 2) Build a "name" column from "first" and "second"
    df24["name"] = df24["first"].str.strip() + " " + df24["second"].str.strip()

    # 3) Drop the temporary columns
    df24 = df24.drop(columns=["first", "second"])

    # Keep only useful columns for each season
    df22_useful = select_useful(df22)
    df23_useful = select_useful(df23)
    df24_useful = select_useful(df24)

    # Concatenate the three seasons 
    full = pd.concat([df22_useful, df23_useful, df24_useful], ignore_index=True)

    # Create the output folder, save the cleaned dataset into it, and return the file path
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "players_all_seasons.csv"
    full.to_csv(out_path, index=False)

    return out_path

# When we run this script directly, launch the pipeline and display the output file path
if __name__ == "__main__":
    output_path = run_pipeline()
    print(f"Data pipeline completed. File saved to: {output_path}")





