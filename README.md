

### FPL Points Predictor README ###



# RESEARCH QUESTION :

To what extent are point predictions for a game such as Fantasy Premier League (FPL), made using supervised machine learning, good indicators of performance? Also, can we draw a comparison with those made by bookmakers?



# SETUP :

1. Clone the repository : 
git clone <repository-url>
cd fpl-points-predictor

2. Create and activate a Python environment : 
conda create -n fpl python=3.10
conda activate fpl

3. Install dependencies : 
pip install -r requirements.txt

The project has been developed and tested in the Nuvolos environment used at HEC Lausanne, but it is fully compatible with a standard local setup.



# USAGE 

- The project is designed to be executed from a single entry point: main.py. 
The basic command execution is : python main.py
- As you will see, the fourth part is optional (does not display automatically), it means you have to use a specific command to see the results, the command is : python main.py --run-bookmakers
- You can skip some section by using : python main.py --skip-*the section you want to skip here*



# STRUCTURE 

fpl-points-predictor/
│
├── data/
│   ├── raw/                # Original raw datasets (FPL, odds)
│   └── processed/          # Cleaned and feature-engineered datasets
│
├── src/
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── features.py         # Feature engineering (lags, rolling stats)
│   ├── model.py            # Predictive models
│   ├── evaluation.py       # Model evaluation & metrics (incl. bookmaker benchmark)
│   └── reporting.py        # Results reporting: tables, figures, and exports
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_model.py
│   └── test_evaluation.py
│
├── results/
│   ├── metrics/            # CSV, TXT, MD evaluation summaries
│   ├── figures/            # Plots (pred vs actual, residuals)
│   ├── tables/             # Exported tables (top players, squads)
│   └── predictions/        # Gameweek-level predictions
│
├── main.py                 # Lightweight orchestration script
├── requirements.txt
├── README.md
└── PROPOSAL.md



# RESULTS

The project produces reproducible and interpretable outputs at three levels:  
model evaluation, player-level predictions, and team-level summaries.

All results are generated automatically by running `main.py` and are stored
under the `results/` directory.

1) Model evaluation (Gameweek-level)

Predictive performance is evaluated on a *hold-out season* using several
standard regression metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R²
- Spearman rank correlation

The evaluation compares multiple models, including linear baselines and a
seasonal Gradient Boosting model. 
 
Results are exported as:

- `results/metrics/gw_metrics_detailed.csv`
- `results/metrics/gw_metrics_detailed.md`
- `results/metrics/gw_leaderboard.json`

In addition, diagnostic figures are generated for each model:
- predicted vs actual points
- residual distributions

These figures are saved under:
```text
results/figures/


2) Player-level predictions with gbm

Predictions are produced at the player–Gameweek level and rely only on
information available prior to the match (historical points and lagged features).
They can be directly used for decision-making and exploratory analysis. 

Outputs include:
- the best predicted player per position for selected Gameweeks,
- top-N player rankings based on predicted points,
- random prediction samples used as sanity checks.

All tables are exported in:
- **CSV format** (full fidelity, suitable for further analysis),
- **Markdown format** (compact, human-readable summaries).

Files are written under:
```text
results/predictions/
results/tables/


3) Team-level summaries and bookmaker comparison

Player-level predictions are aggregated to the team level by summing the
predicted contributions of the **top 11 players** per team and Gameweek.
This aggregation yields an interpretable measure of predicted team strength.

These team-level estimates are used to:
- construct match-level team strength summaries,
- compare model-implied probabilities with Bet365 normalized home-win probabilities,
- highlight illustrative examples of the best and worst model predictions.

The bookmaker comparison serves as an **informative benchmark**, enabling the
model’s outputs to be assessed against market-implied expectations rather than
used as a betting strategy.

All related tables and samples are exported under:
```text
results/predictions/
results/tables/




# REQUIREMENTS

All required packages are listed in:
environment.yml



