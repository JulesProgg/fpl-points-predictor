To ensure full reproducibility, both datasets can be downloaded using the Kaggle API as follows:

1) Create a Kaggle account at https://www.kaggle.com and generate an API token.

2) Install the Kaggle API and authenticate according to Kaggle’s official documentation.

3) Download the datasets via the command line:
kaggle datasets download -d reevebarreto/fantasy-premier-league-player-data-2016-2024
kaggle datasets download -d adamgbor/club-football-match-data-2000-2025 
* (Note that from the Club Football Match Data (2000–2025) dataset, only the file matches.csv is required for this project.)

4) Extract the downloaded archives and place the raw CSV files in the project’s data/raw/ directory.

All raw data are subsequently cleaned, normalised, and transformed into processed datasets stored under data/processed/. 

These processed files are the sole inputs used by the modelling and evaluation pipeline, ensuring consistent and reproducible experimental results.