from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import kagglehub


# Paths for your project (same logic as in data_pipeline.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_PATH = DATA_RAW_DIR / "player_gameweeks_raw.csv"

# Mapping from Kaggle-style column names → your internal schema
# On part exactement des colonnes du fichier brut :
# ,name,assists,bonus,bps,clean_sheets,creativity,element,fixture,goals_conceded,
# goals_scored,ict_index,influence,kickoff_time,minutes_played,own_goals,
# penalties_missed,penalties_saved,red_cards,round,saves,selected,
# team_a_score,team_h_score,threat,points,transfers_balance,transfers_in,
# transfers_out,value,was_home,yellow_cards,gameweek,season,position,team,opponent,result
RENAME_MAP = {
    "element": "id",
    "name": "name",
    "team": "team",
    "position": "position",

    "minutes_played": "minutes",

    "goals_scored": "goals_scored",
    "assists": "assists",
    "goals_conceded": "goals_conceded",
    "clean_sheets": "clean_sheets",
    "saves": "saves",
    "penalties_saved": "penalties_saved",
    "penalties_missed": "penalties_missed",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "own_goals": "own_goals",

    "bonus": "bonus",
    "bps": "bps",

    "creativity": "creativity",
    "threat": "threat",
    "influence": "influence",
    "ict_index": "ict_index",

    "points": "total_points",

    "season": "season",
    "gameweek": "gameweek",
    "round": "gameweek",  # au cas où certains fichiers utilisent round
}

# Colonnes cibles que tu veux dans player_gameweeks_raw.csv
# (on ajoute gameweek pour garder l’info de la GW)
TARGET_COLUMNS = [
    "id",
    "name",
    "team",
    "position",
    "minutes",
    "goals_scored",
    "assists",
    "goals_conceded",
    "clean_sheets",
    "saves",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "starts",
    "bonus",
    "bps",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "total_points",
    "season",
    "gameweek",
]


def build_player_gameweeks_raw_from_kaggle() -> Path:
    """
    Download the Kaggle dataset:
        'reevebarreto/fantasy-premier-league-player-data-2016-2024'
    and convert the single CSV
        'FPL Player Stats(2016-2024).csv'
    into a standardised player_gameweeks_raw.csv in data/raw/.

    Each row in the final CSV corresponds to:
        one player × one season × one gameweek.

    Columns are:
        id,name,team,position,minutes,goals_scored,assists,goals_conceded,
        clean_sheets,saves,penalties_saved,penalties_missed,yellow_cards,
        red_cards,own_goals,starts,bonus,bps,expected_goals,
        expected_assists,expected_goal_involvements,expected_goals_conceded,
        influence,creativity,threat,ict_index,total_points,season,gameweek

    Les colonnes expected_* sont absentes du fichier brut → elles seront créées
    et remplies avec NaN (aucune information disponible dans ce dataset).
    """

    # 1) Download (or load from cache) the Kaggle dataset
    kaggle_path_str = kagglehub.dataset_download(
        "reevebarreto/fantasy-premier-league-player-data-2016-2024"
    )
    kaggle_root = Path(kaggle_path_str)
    print(f"Kaggle dataset downloaded to: {kaggle_root}")

    # 2) Path to the single CSV file
    csv_path = kaggle_root / "FPL Player Stats(2016-2024).csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {csv_path}")

    # 3) Read CSV
    df = pd.read_csv(csv_path)

    # 4) Rename columns to standard names
    rename_map = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Enlever les éventuelles colonnes dupliquées (par ex. deux 'gameweek')
    df = df.loc[:, ~df.columns.duplicated()]

    # 5) Créer la colonne starts : un joueur commence un match s'il joue >= 60 minutes
    if "starts" not in df.columns:
        if "minutes" in df.columns:
            df["starts"] = (df["minutes"] >= 60).astype(int)
        else:
            df["starts"] = pd.NA

    # 6) Colonnes expected_* n'existent pas dans ce dataset → on les crée vides
    for col in ["expected_goals", "expected_assists",
                "expected_goal_involvements", "expected_goals_conceded"]:
        if col not in df.columns:
            df[col] = pd.NA

    # 7) S'assurer que toutes les colonnes cibles existent (même si NaN)
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # 8) Ne garder que les colonnes dans l'ordre voulu
    df = df[TARGET_COLUMNS].copy()

    # 9) Sauvegarder dans data/raw/player_gameweeks_raw.csv
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved combined GW dataset to: {OUTPUT_PATH}")
    return OUTPUT_PATH



if __name__ == "__main__":
    build_player_gameweeks_raw_from_kaggle()


