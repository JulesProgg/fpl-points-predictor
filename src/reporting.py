# =============================================================================
# reporting.py
# =============================================================================
# Reporting / exports module.
#
# Purpose:
# - Produce figures (pred vs actual, residuals) and evaluation tables under results/
# - Export additional tables (players / teams) and match samples (team strengths)
#
# Important constraint:
# - This file focuses on exports + presentation (CSV/Markdown/PNG).
# - It must preserve identical outputs for the same inputs/config.
# =============================================================================

from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation import (
    _compute_mae,
    _compute_rmse,
    _compute_r2,
    _compute_spearman,
    get_ytrue_ypred_anytime_linear_gw,
    get_ytrue_ypred_seasonal_linear_gw,
    get_ytrue_ypred_seasonal_gbm_gw,
    get_test_predictions_seasonal_gbm_gw,
)

from src.data_loader import load_fixtures


# =============================================================================
# FILESYSTEM HELPERS
# =============================================================================


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, path: Path) -> None:
    _ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=120,
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)


def summarize_bookmaker_comparison(comp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create display-ready worst5/best5 tables from compare_model_vs_bookmakers output.
    - abs_error computed directly from probabilities
    """
    df = comp.copy()

    required = {"pnorm_home_win", "p_model_home_win"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Bookmaker comparison missing columns: {sorted(missing)}")

    # Ensure numeric
    df["pnorm_home_win"] = pd.to_numeric(df["pnorm_home_win"], errors="coerce")
    df["p_model_home_win"] = pd.to_numeric(df["p_model_home_win"], errors="coerce")

    df = df.dropna(subset=["pnorm_home_win", "p_model_home_win"]).copy()

    # --------------------------------------------------
    # Compute absolute error 
    # --------------------------------------------------
    df["abs_error"] = (df["p_model_home_win"] - df["pnorm_home_win"]).abs()

    # --------------------------------------------------
    # Display-ready table (single source of truth)
    # --------------------------------------------------
    display_cols = [
        c for c in [
            "gameweek",          
            "home_team",
            "away_team",
            "pnorm_home_win",
            "p_model_home_win",
            "abs_error",
        ]
        if c in df.columns
    ]

    df_display = (
        df[display_cols]
        .rename(
            columns={
                "gameweek": "gw",
                "pnorm_home_win": "p_home_win_b365",
                "p_model_home_win": "p_home_win_model",
            }
        )
        .round(3)
    )

    worst5 = df_display.sort_values("abs_error", ascending=False).head(5)
    best5 = df_display.sort_values("abs_error", ascending=True).head(5)

    return worst5, best5



def console_print_bookmaker_stats(mae: float, corr: float, n: int) -> None:
    """Console-only: print a compact stats block for bookmaker comparison."""
    print("-" * 80)
    print("Statistics summary")
    print("-" * 80)
    print(f"MAE  : {mae:.3f}")
    print(f"Corr : {corr:.3f}")
    print(f"n    : {n}")
    print("-" * 80)




def _write_table_csv_md(df: pd.DataFrame, out_base: Path) -> None:
    """Write df to CSV + compact Markdown with base path (no extension).

    Design:
      - CSV: full fidelity (no truncation)
      - MD: compact display (short headers, compact team names, optional truncation)
    """
    _ensure_dir(out_base.parent)

    # 1) CSV full fidelity
    df.to_csv(out_base.with_suffix(".csv"), index=False, encoding="utf-8")

    # 2) Markdown compact view
    md_df = df.copy()

    # Short headers to prevent wrapping
    col_rename = {
        "player_id": "pid",
        "position": "pos",
        "gameweek": "gw",
        "predicted_points": "pred_points",
        "actual_points": "act_points",
    }
    md_df = md_df.rename(columns={k: v for k, v in col_rename.items() if k in md_df.columns})

    # Compact team/opponent names for MD only
    def _compact_team_name(x: object) -> str:
        if pd.isna(x):
            return ""
        s = str(x)
        repl = {
            "Brighton & Hove Albion": "Brighton",
            "Tottenham Hotspur": "Spurs",
            "Manchester City": "Man City",
            "Manchester United": "Man United",
            "Newcastle United": "Newcastle",
            "Wolverhampton Wanderers": "Wolves",
            "Nottingham Forest": "Nott'm Forest",
            "West Ham United": "West Ham",
            "AFC Bournemouth": "Bournemouth",
            "Sheffield United": "Sheff Utd",
            "Crystal Palace": "Palace",
        }
        return repl.get(s, s)

    for c in ["team", "opponent"]:
        if c in md_df.columns:
            md_df[c] = md_df[c].map(_compact_team_name)

    # Optional: truncate long player names in MD only (keeps table on one line)
    def _truncate(s: object, n: int = 22) -> str:
        if pd.isna(s):
            return ""
        txt = str(s)
        return (txt[: n - 1] + "…") if len(txt) > n else txt

    if "name" in md_df.columns:
        md_df["name"] = md_df["name"].map(lambda x: _truncate(x, n=22))

    # Format numeric columns to reduce width
    for c in ["pred_points", "act_points", "error", "abs_error", "top11_sum", "top11_avg"]:
        if c in md_df.columns:
            md_df[c] = pd.to_numeric(md_df[c], errors="coerce").map(
                lambda x: f"{x:.2f}" if pd.notna(x) else ""
            )

    try:
        md = md_df.to_markdown(index=False, tablefmt="github")
    except Exception:
        md = md_df.to_string(index=False)

    out_base.with_suffix(".md").write_text(md + "\n", encoding="utf-8")


# =============================================================================
# CONSOLE PRESENTATION HELPERS (used by main.py)
# =============================================================================

POS_ORDER = ["GK", "DEF", "MID", "FWD"]


def _console_print_table_fixed_width(
    df: pd.DataFrame,
    cols: list[str],
    *,
    indent: str = "   ",
    float_round: int = 2,
    col_space: dict[str, int] | None = None,
) -> None:
    """Print a fixed-width console table (headers + values aligned)."""
    if df is None or len(df) == 0:
        print(indent + "(empty)")
        return

    out = df.copy()
    cols_existing = [c for c in cols if c in out.columns]
    out = out[cols_existing].copy()

    # Round numeric columns for display
    for c in out.select_dtypes(include="number").columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(float_round)

    W = col_space or {
        "gameweek": 8,
        "position": 8,
        "name": 24,
        "team": 22,
        "predicted_points": 16,
    }

    def _fit(text: object, w: int) -> str:
        s = "" if pd.isna(text) else str(text)
        if len(s) > w:
            return (s[: w - 1] + "…") if w >= 2 else s[:w]
        return s.ljust(w)

    def _fit_right(text: object, w: int) -> str:
        s = "" if pd.isna(text) else str(text)
        if len(s) > w:
            return s[-w:]
        return s.rjust(w)

    # Header
    header_parts: list[str] = []
    for c in cols_existing:
        w = int(W.get(c, max(len(c), 10)))
        header_parts.append(c.rjust(w) if c == "predicted_points" else c.ljust(w))
    print(indent + " ".join(header_parts).rstrip())

    # Rows
    for _, r in out.iterrows():
        row_parts: list[str] = []
        for c in cols_existing:
            w = int(W.get(c, max(len(c), 10)))
            v = r[c]
            if c == "predicted_points":
                v = "" if pd.isna(v) else f"{float(v):.{float_round}f}"
                row_parts.append(_fit_right(v, w))
            else:
                row_parts.append(_fit(v, w))
        print(indent + " ".join(row_parts).rstrip())


def console_print_prediction_demo(
    preds: pd.DataFrame,
    *,
    one_line: bool = False,
    indent: str = "   ",
) -> None:
    """
    Console-only: prints (1) best buy per position, (2) top 3 per position.
    Keeps all transformation / ordering / formatting out of main.py.
    """
    if preds is None or len(preds) == 0:
        print(indent + "(empty)")
        return

    tmp = preds.copy()

    if "position" not in tmp.columns:
        print(indent + "WARNING: 'position' column missing")
        return
    if "predicted_points" not in tmp.columns:
        print(indent + "WARNING: 'predicted_points' column missing")
        return

    tmp["predicted_points"] = pd.to_numeric(tmp["predicted_points"], errors="coerce")
    tmp = tmp.dropna(subset=["predicted_points"]).copy()

    tmp["position"] = pd.Categorical(tmp["position"], categories=POS_ORDER, ordered=True)

    best_by_pos = (
        tmp.sort_values(["position", "predicted_points"], ascending=[True, False])
        .groupby("position", as_index=False, observed=True)
        .head(1)
        .sort_values("position")
        .reset_index(drop=True)
    )

    top3_by_pos = (
        tmp.sort_values(["position", "predicted_points"], ascending=[True, False])
        .groupby("position", as_index=False, observed=True)
        .head(3)
        .sort_values(["position", "predicted_points"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # If you want to use the compact one-line mode, reuse your markdown compacting logic idea,
    # but keep it console-only: simplest is to just print fixed-width for both modes
    # (one_line flag can be used later if you want a different print style).
    
    print("\n" + indent + "Best buy per position:")
    _console_print_table_fixed_width(
        best_by_pos,
        cols=["gameweek", "position", "name", "team", "predicted_points"],
        indent=indent,
    )

    print("\n" + indent + "Top 3 options per position:")
    _console_print_table_fixed_width(
        top3_by_pos,
        cols=["gameweek", "position", "name", "team", "predicted_points"],
        indent=indent,
    )

def print_example_matches(comp: pd.DataFrame, n: int = 5) -> None:
    """
    Print example matches:
    - n worst predictions (largest absolute error)
    - n best predictions (smallest absolute error)
    """
    if comp.empty:
        print("No matches to display.")
        return

    comp = comp.sort_values("abs_error")

    best = comp.head(n)
    worst = comp.tail(n).sort_values("abs_error", ascending=False)

    def _print_block(df: pd.DataFrame, title: str) -> None:
        print(title)
        print("-" * 80)
        for _, row in df.iterrows():
            season = row["season"]
            gw = int(row["gameweek"])
            home = row["home_team"]
            away = row["away_team"]
            p_bookie = row["pnorm_home_win"]
            p_model = row["p_model_home_win"]
            diff = p_model - p_bookie

            print(f"{season} GW{gw}: {home} vs {away}")
            print(f"  Bet365 home-win prob : {p_bookie:.3f}")
            print(f"  Model home-win prob  : {p_model:.3f}")
            print(f"  Difference (model - Bet365): {diff:+.3f}")
            print()

    _print_block(worst, f"Example matches – WORST {n} predictions")
    _print_block(best, f"Example matches – BEST {n} predictions")


# =============================================================================
# PLOTTING HELPERS
# =============================================================================


def _plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    fig = plt.figure()
    ax = plt.gca()

    hb = ax.hexbin(
        y_true,
        y_pred,
        gridsize=35,
        mincnt=1,
    )

    # -----------------------------
    # 1) Linear trend line (central tendency)
    # -----------------------------
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() >= 2:
        a, b = np.polyfit(y_true[mask], y_pred[mask], 1)
        x_line = np.array([float(np.nanmin(y_true[mask])), float(np.nanmax(y_true[mask]))])
        y_line = a * x_line + b
        ax.plot(x_line, y_line, linewidth=2)

    # -----------------------------
    # 2) Make hexes look "round"
    # -----------------------------
    ax.set_aspect("equal", adjustable="box")

    plt.colorbar(hb, ax=ax, label="Count")

    ax.set_xlabel("Actual points")
    ax.set_ylabel("Predicted points")
    ax.set_title(title)

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

    return fig


def _plot_residuals_hist(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    resid = y_pred - y_true

    fig = plt.figure()
    ax = plt.gca()

    counts, bins, patches = ax.hist(resid, bins=60)

    max_idx = int(np.argmax(counts))
    bar = patches[max_idx]

    left = float(bins[max_idx])
    right = float(bins[max_idx + 1])
    label = f"[{left:.2f}; {right:.2f}]"

    x = bar.get_x() + bar.get_width() / 2
    y = bar.get_height() / 2

    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        rotation=90,
        color="black",
        fontsize=9,
        fontweight="bold",
    )

    ax.set_xlabel("Residual (pred - actual)")
    ax.set_ylabel("Count")
    ax.set_title(title)

    return fig


def _plot_metric_bar(df: pd.DataFrame, metric: str, title: str):
    fig = plt.figure()
    ax = plt.gca()
    ax.bar(df["model_key"], df[metric])
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    return fig


# =============================================================================
# EXPORTS: MATCH TEAM STRENGTH SAMPLE
# =============================================================================


def export_sample_match_team_strength(
    test_season: str = "2022/23",
    output_dir: str | Path = "results",
    top_n_players: int = 11,
    sample_n_matches: int = 10,
    seed: int = 42,
    candidate_gammas: list[float] | None = None,
) -> pd.DataFrame:
    """
    Export a random sample of matches for a given season, with predicted team strength
    for both teams (home/away), aligned with src.evaluation.compare_model_vs_bookmakers.

    Team strength definition (aligned):
      - playing_weight from minutes_lag_* if available: mean(minutes_lag_*)/90 clipped to [0,1]
      - weighted_points = predicted_points * playing_weight
      - team strength = sum of top-N weighted_points per (season, gameweek, team)

    Additionally exports:
      - gamma-calibrated naive probability: S_home/(S_home+S_away) (after gamma)
      - (optional) logistic calibrated probability in logit space, same as evaluation.py,
        if Bet365 pnorm_home_win is available via load_clean_odds.

    Writes:
      - results/predictions/sample_match_team_strength__<season>__gbm_seasonal_top<N>.csv
      - results/predictions/sample_match_team_strength__<season>__gbm_seasonal_top<N>.md
    """
    from sklearn.linear_model import LinearRegression
    from src.data_loader import load_clean_odds  # local import to avoid circular deps

    output_dir = Path(output_dir)
    predictions_dir = output_dir / "predictions"
    _ensure_dir(predictions_dir)

    if candidate_gammas is None:
        candidate_gammas = [1.0, 1.1, 1.2, 1.3, 1.4]

    # ------------------------------------------------------
    # 1) Load player-level predictions for the season (GBM seasonal)
    # ------------------------------------------------------
    preds = get_test_predictions_seasonal_gbm_gw(test_season=test_season).copy()

    required = {"team", "season", "gameweek", "predicted_points"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(
            f"get_test_predictions_seasonal_gbm_gw is missing required columns: {sorted(missing)}"
        )

    preds["season"] = preds["season"].astype(str)
    preds = preds[preds["season"] == str(test_season)].copy()

    preds["gameweek"] = pd.to_numeric(preds["gameweek"], errors="coerce").astype("Int64")
    preds["predicted_points"] = pd.to_numeric(preds["predicted_points"], errors="coerce")

    preds = preds.dropna(subset=["team", "season", "gameweek", "predicted_points"]).copy()
    preds["gameweek"] = preds["gameweek"].astype(int)

    if preds.empty:
        raise ValueError(f"No predictions available for season {test_season!r}.")

    # ------------------------------------------------------
    # 2) Align with evaluation.py: playing_weight from minutes_lag_*
    # ------------------------------------------------------
    minute_cols = [c for c in preds.columns if str(c).startswith("minutes_lag_")]

    if minute_cols:
        preds["recent_minutes_mean"] = preds[minute_cols].mean(axis=1)
        preds["playing_weight"] = (preds["recent_minutes_mean"] / 90.0).clip(0.0, 1.0)
    else:
        preds["recent_minutes_mean"] = np.nan
        preds["playing_weight"] = 1.0

    preds["weighted_points"] = preds["predicted_points"] * preds["playing_weight"]

    # ------------------------------------------------------
    # 3) Top-N per team/GW based on weighted_points (not predicted_points)
    # ------------------------------------------------------
    preds = preds.sort_values(
        ["season", "gameweek", "team", "weighted_points"],
        ascending=[True, True, True, False],
    )

    top = (
        preds.groupby(["season", "gameweek", "team"], as_index=False, sort=False)
        .head(int(top_n_players))
        .copy()
    )

    team_strength = (
        top.groupby(["season", "gameweek", "team"], as_index=False)["weighted_points"]
        .sum()
        .rename(columns={"weighted_points": "team_strength"})
    )

    # ------------------------------------------------------
    # 4) Load fixtures and (optionally) odds, then merge like evaluation.py
    # ------------------------------------------------------
    fixtures = load_fixtures().copy()
    fixtures["season"] = fixtures["season"].astype(str)
    fixtures = fixtures[fixtures["season"] == str(test_season)].copy()
    fixtures["gameweek"] = pd.to_numeric(fixtures["gameweek"], errors="coerce").astype("Int64")
    fixtures = fixtures.dropna(subset=["gameweek"]).copy()
    fixtures["gameweek"] = fixtures["gameweek"].astype(int)

    if fixtures.empty:
        raise ValueError(f"No fixtures available for season {test_season!r}.")

    out = fixtures.merge(
        team_strength.rename(columns={"team": "home_team", "team_strength": "home_strength"}),
        on=["season", "gameweek", "home_team"],
        how="left",
    ).merge(
        team_strength.rename(columns={"team": "away_team", "team_strength": "away_strength"}),
        on=["season", "gameweek", "away_team"],
        how="left",
    )

    # Try to add Bet365 normalized probability if possible
    try:
        odds = load_clean_odds().copy()
        odds["season"] = odds["season"].astype(str)
        odds_season = odds[odds["season"] == str(test_season)].copy()

        if "pnorm_home_win" in odds_season.columns:
            out = out.merge(
                odds_season[["season", "home_team", "away_team", "pnorm_home_win"]],
                on=["season", "home_team", "away_team"],
                how="left",
            )
        else:
            out["pnorm_home_win"] = np.nan
    except Exception:
        out["pnorm_home_win"] = np.nan

    out = out.replace([np.inf, -np.inf], np.nan)

    out["top_n"] = int(top_n_players)
    out["strength_diff"] = out["home_strength"] - out["away_strength"]

    # ------------------------------------------------------
    # 5) Gamma calibration (only if we have pnorm_home_win)
    # ------------------------------------------------------
    gamma = 1.0
    best_mae = np.nan

    valid_for_gamma = out.dropna(subset=["home_strength", "away_strength", "pnorm_home_win"]).copy()
    if len(valid_for_gamma) > 0:
        best_gamma = 1.0
        best_mae_val = float("inf")

        for g in candidate_gammas:
            S_home = valid_for_gamma["home_strength"] ** g
            S_away = valid_for_gamma["away_strength"] ** g
            p_naive = S_home / (S_home + S_away)

            mae_g = float(np.mean(np.abs(p_naive - valid_for_gamma["pnorm_home_win"])))
            if mae_g < best_mae_val:
                best_mae_val = mae_g
                best_gamma = g

        gamma = float(best_gamma)
        best_mae = float(best_mae_val)

    out["gamma"] = gamma
    out["home_strength_gamma"] = out["home_strength"] ** gamma
    out["away_strength_gamma"] = out["away_strength"] ** gamma
    out["p_model_naive"] = out["home_strength_gamma"] / (
        out["home_strength_gamma"] + out["away_strength_gamma"]
    )

    # ------------------------------------------------------
    # 6) Logistic calibration (only if we have pnorm_home_win + enough data)
    # ------------------------------------------------------
    out["p_model_home_win"] = np.nan
    valid_for_logit = out.dropna(
        subset=["home_strength_gamma", "away_strength_gamma", "pnorm_home_win"]
    ).copy()

    if len(valid_for_logit) >= 5:
        valid_for_logit["strength_diff_gamma"] = (
            valid_for_logit["home_strength_gamma"] - valid_for_logit["away_strength_gamma"]
        )

        X = valid_for_logit["strength_diff_gamma"].to_numpy(dtype=float).reshape(-1, 1)
        y = valid_for_logit["pnorm_home_win"].to_numpy(dtype=float)

        eps = 1e-6
        y_clip = np.clip(y, eps, 1.0 - eps)
        logit_y = np.log(y_clip / (1.0 - y_clip))

        reg = LinearRegression()
        reg.fit(X, logit_y)

        logit_pred = reg.predict(X)
        p_cal = 1.0 / (1.0 + np.exp(-logit_pred))

        out.loc[valid_for_logit.index, "p_model_home_win"] = p_cal

    out["abs_error_naive"] = np.abs(out["p_model_naive"] - out["pnorm_home_win"])
    out["abs_error_cal"] = np.abs(out["p_model_home_win"] - out["pnorm_home_win"])

    # ------------------------------------------------------
    # 7) Sample = N matches (reproducible RNG)
    # ------------------------------------------------------
    rng = np.random.default_rng(int(seed))
    out = out.reset_index(drop=True)

    eligible = out.dropna(subset=["pnorm_home_win", "p_model_home_win"]).copy()

    if len(eligible) < sample_n_matches:
        raise ValueError(
            f"Not enough comparable matches to sample {sample_n_matches}. "
            f"Only {len(eligible)} available."
        )

    idx = rng.choice(len(eligible), size=sample_n_matches, replace=False)
    out_sample = eligible.iloc[idx].reset_index(drop=True)

    # Round for readability
    round_cols = [
        "home_strength",
        "away_strength",
        "strength_diff",
        "home_strength_gamma",
        "away_strength_gamma",
        "p_model_naive",
        "p_model_home_win",
        "pnorm_home_win",
        "abs_error_naive",
        "abs_error_cal",
    ]
    for col in round_cols:
        if col in out_sample.columns:
            out_sample[col] = pd.to_numeric(out_sample[col], errors="coerce").round(4)


    # ------------------------------------------------------
    # 8) Output paths
    # ------------------------------------------------------
    csv_path = predictions_dir / "h_team_strength_random_gw.csv"
    md_path = predictions_dir / "h_team_strength_random_gw.md"


    # ------------------------------------------------------
    # 9) Write outputs
    # ------------------------------------------------------
    out_sample.to_csv(csv_path, index=False)

    md_lines: list[str] = []
    md_lines.append(f"# Sample match team strength – {test_season}\n")
    md_lines.append(f"{len(out_sample)} sampled matches ...")

    md_cols = [
        "gameweek",
        "home_team",
        "away_team",
        "p_model_home_win",
        "pnorm_home_win",
        "abs_error_cal",
    ]
    md_cols_existing = [c for c in md_cols if c in out_sample.columns]

    md_sample = out_sample[md_cols_existing].copy()
    md_sample = md_sample.rename(columns={"p_model_home_win": "p_home_model", "pnorm_home_win": "p_home_b365"})

    md_lines.append(md_sample.to_markdown(index=False) + "\n")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return out_sample


def enrich_sample_match_summary(sample: pd.DataFrame) -> pd.DataFrame:
    """
    Add match-level interpretable columns on top of export_sample_match_team_strength output.
    Keeps this presentation logic out of main.py.
    """
    if sample is None or len(sample) == 0:
        return sample

    df = sample.copy()

    col_candidates = {
        "home_team": ["home_team", "team_home", "home"],
        "away_team": ["away_team", "team_away", "away"],
        "home_strength": ["home_strength", "strength_home", "team_strength_home"],
        "away_strength": ["away_strength", "strength_away", "team_strength_away"],
    }

    def _first_existing(names: list[str]) -> str | None:
        for n in names:
            if n in df.columns:
                return n
        return None

    c_home_team = _first_existing(col_candidates["home_team"])
    c_away_team = _first_existing(col_candidates["away_team"])
    c_home_str = _first_existing(col_candidates["home_strength"])
    c_away_str = _first_existing(col_candidates["away_strength"])

    missing = [k for k, c in {
        "home_team": c_home_team,
        "away_team": c_away_team,
        "home_strength": c_home_str,
        "away_strength": c_away_str,
    }.items() if c is None]

    if missing:
        raise KeyError(
            "Sample match summary cannot be computed because "
            f"required columns are missing: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    rename_map = {
        c_home_team: "home_team",
        c_away_team: "away_team",
        c_home_str: "home_strength",
        c_away_str: "away_strength",
    }
    rename_map = {k: v for k, v in rename_map.items() if k != v}
    if rename_map:
        df = df.rename(columns=rename_map)

    df["delta_strength"] = (df["home_strength"] - df["away_strength"]).abs()

    def _favorite(row) -> str:
        if row["home_strength"] > row["away_strength"]:
            return row["home_team"]
        if row["away_strength"] > row["home_strength"]:
            return row["away_team"]
        return "DRAW"

    df["favorite"] = df.apply(_favorite, axis=1)
    return df



# =============================================================================
# EXPORTS: GW RESULTS (METRICS + FIGURES + TABLES)
# =============================================================================


def export_gw_results(
    test_season: str = "2022/23",
    output_dir: str | Path = "results",
) -> pd.DataFrame:
    """
    Generate GW evaluation outputs (metrics + figures) and write them to results/.
    """
    from datetime import datetime
    import json

    output_dir = Path(output_dir)

    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures" 
    predictions_dir = output_dir / "predictions"
    tables_dir = output_dir / "tables"

    _ensure_dir(metrics_dir)
    _ensure_dir(figures_dir)
    _ensure_dir(predictions_dir)
    _ensure_dir(tables_dir)

    suites: Dict[str, Callable[[], Tuple[np.ndarray, np.ndarray]]] = {
        "linear_anytime_lag3": lambda: get_ytrue_ypred_anytime_linear_gw(3, test_season=test_season),
        "linear_anytime_lag5": lambda: get_ytrue_ypred_anytime_linear_gw(5, test_season=test_season),
        "linear_seasonal": lambda: get_ytrue_ypred_seasonal_linear_gw(test_season=test_season),
        "gbm_seasonal": lambda: get_ytrue_ypred_seasonal_gbm_gw(test_season=test_season),
    }

    rows = []
    for model_key, fn in suites.items():
        y_true, y_pred = fn()

        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        yt = y_true[mask]
        yp = y_pred[mask]

        resid = yp - yt
        abs_err = np.abs(resid)

        rows.append(
            {
                "model_key": model_key,
                "test_season": test_season,
                "n_obs": int(mask.sum()),
                "mae": _compute_mae(yt, yp),
                "rmse": _compute_rmse(yt, yp),
                "r2": _compute_r2(yt, yp),
                "spearman": _compute_spearman(yt, yp),
                "bias": float(np.mean(resid)) if len(resid) else np.nan,
                "medae": float(np.median(abs_err)) if len(abs_err) else np.nan,
                "p90_ae": float(np.quantile(abs_err, 0.90)) if len(abs_err) else np.nan,
            }
        )

        fig1 = _plot_pred_vs_actual(yt, yp, f"{model_key} – Pred vs Actual ({test_season})")
        _save_fig(fig1, figures_dir / f"{model_key}_pred_vs_actual.png")

        fig2 = _plot_residuals_hist(yt, yp, f"{model_key} – Residuals ({test_season})")
        _save_fig(fig2, figures_dir / f"{model_key}_residuals.png")

    df = pd.DataFrame(rows)
    df = df.sort_values(["mae", "rmse"], ascending=[True, True]).reset_index(drop=True)

    df["rank_mae"] = df["mae"].rank(method="min", ascending=True).astype(int)
    df["rank_rmse"] = df["rmse"].rank(method="min", ascending=True).astype(int)
    df["rank_r2"] = df["r2"].rank(method="min", ascending=False).astype(int)
    df["rank_spearman"] = df["spearman"].rank(method="min", ascending=False).astype(int)

    for col, nd in {
        "mae": 3,
        "rmse": 3,
        "r2": 3,
        "spearman": 3,
        "bias": 3,
        "medae": 3,
        "p90_ae": 3,
    }.items():
        if col in df.columns:
            df[col] = df[col].round(nd)

    df.to_csv(metrics_dir / "gw_metrics_detailed.csv", index=False)

    best = df.iloc[0]
    summary = (
        f"Hold-out season {test_season}: best GW model by MAE is {best['model_key']} "
        f"(MAE={best['mae']:.3f}, RMSE={best['rmse']:.3f}, "
        f"R2={best['r2']:.3f}, Spearman={best['spearman']:.3f}, "
        f"Bias={best['bias']:.3f}, MedAE={best['medae']:.3f}, P90_AE={best['p90_ae']:.3f}).\n"
    )

    md_lines = []
    md_lines.append(f"# GW metrics – {test_season}\n")
    md_lines.append(summary.strip() + "\n")

    md_cols = ["model_key", "mae", "rmse", "r2", "spearman", "bias", "medae", "p90_ae", "rank_mae"]
    md_cols_existing = [c for c in md_cols if c in df.columns]
    md_lines.append(df[md_cols_existing].to_markdown(index=False) + "\n")

    (metrics_dir / "gw_metrics_detailed.md").write_text("\n".join(md_lines), encoding="utf-8")


    # -----------------------------
    # Predictions random sample (10 players, 1 random GW each)
    # -----------------------------
    try:
        preds = get_test_predictions_seasonal_gbm_gw(test_season=test_season)

        if preds is None or len(preds) == 0:
            raise ValueError("get_test_predictions_seasonal_gbm_gw returned empty dataframe.")

        preds = preds.copy()

        col_map_candidates = {
            "player_id": ["player_id", "element", "id"],
            "name": ["name", "player_name", "web_name"],
            "team": ["team", "team_name"],
            "position": ["position", "pos"],
            "gameweek": ["gameweek", "gw", "round"],
            "predicted_points": ["predicted_points", "y_pred", "pred", "prediction"],
            "actual_points": ["actual_points", "y_true", "true", "target", "total_points"],
        }

        def _first_existing(cols: list[str]) -> str | None:
            for c in cols:
                if c in preds.columns:
                    return c
            return None

        c_player = _first_existing(col_map_candidates["player_id"])
        c_gw = _first_existing(col_map_candidates["gameweek"])
        c_pred = _first_existing(col_map_candidates["predicted_points"])

        if c_player is None or c_gw is None or c_pred is None:
            raise ValueError(
                "Predictions dataframe missing required columns. "
                f"Found columns: {list(preds.columns)}"
            )

        c_name = _first_existing(col_map_candidates["name"])
        c_team = _first_existing(col_map_candidates["team"])
        c_pos = _first_existing(col_map_candidates["position"])
        c_true = _first_existing(col_map_candidates["actual_points"])

        # ------------------------------------------------------
        # If "actual points" is missing from preds, merge it from player gameweeks
        # ------------------------------------------------------
        if c_true is None:
            try:
                from src.data_loader import load_player_gameweeks

                gw = load_player_gameweeks().copy()

                gw_col_map = {
                    "player_id": ["player_id", "element", "id"],
                    "season": ["season"],
                    "gameweek": ["gameweek", "gw", "round"],
                    "actual_points": ["total_points", "actual_points", "y_true", "target", "true", "points"],
                }

                def _first_existing_in(df: pd.DataFrame, cols: list[str]) -> str | None:
                    for c in cols:
                        if c in df.columns:
                            return c
                    return None

                gw_pid = _first_existing_in(gw, gw_col_map["player_id"])
                gw_season = _first_existing_in(gw, gw_col_map["season"])
                gw_gw = _first_existing_in(gw, gw_col_map["gameweek"])
                gw_true = _first_existing_in(gw, gw_col_map["actual_points"])

                if gw_pid and gw_season and gw_gw and gw_true:
                    preds["_pid"] = pd.to_numeric(preds[c_player], errors="coerce")
                    preds["_gw"] = pd.to_numeric(preds[c_gw], errors="coerce")
                    preds["_season"] = preds.get("season", test_season).astype(str)

                    gw["_pid"] = pd.to_numeric(gw[gw_pid], errors="coerce")
                    gw["_gw"] = pd.to_numeric(gw[gw_gw], errors="coerce")
                    gw["_season"] = gw[gw_season].astype(str)
                    gw["_actual_points"] = pd.to_numeric(gw[gw_true], errors="coerce")

                    gw = gw[gw["_season"] == str(test_season)].copy()

                    preds = preds.merge(
                        gw[["_pid", "_season", "_gw", "_actual_points"]],
                        on=["_pid", "_season", "_gw"],
                        how="left",
                    )

                    preds["actual_points"] = preds["_actual_points"]
                    c_true = "actual_points"
                else:
                    c_true = None

            except Exception:
                c_true = None

        # ------------------------------------------------------
        # Filter valid numeric predictions (and actual if available)
        # ------------------------------------------------------
        try:
            preds = preds[np.isfinite(pd.to_numeric(preds[c_pred], errors="coerce"))]
            if c_true is not None:
                preds = preds[np.isfinite(pd.to_numeric(preds[c_true], errors="coerce"))]

            rng = np.random.default_rng(42)

            unique_players = preds[c_player].dropna().unique()
            if len(unique_players) == 0:
                raise ValueError("No players available for sampling.")

            n_players = min(10, len(unique_players))
            sampled_players = rng.choice(unique_players, size=n_players, replace=False)

            sampled_rows = []
            for pid in sampled_players:
                sub = preds[preds[c_player] == pid]
                idx = int(rng.integers(0, len(sub)))
                sampled_rows.append(sub.iloc[idx])

            sample_df = pd.DataFrame(sampled_rows).reset_index(drop=True)

            # --------------------------------------------------
            # Reorder columns (if available)
            # --------------------------------------------------
            out_cols = []
            for c in [c_player, c_name, c_team, c_pos, c_gw, c_pred, c_true]:
                if c is not None and c in sample_df.columns and c not in out_cols:
                    out_cols.append(c)

            remaining = [c for c in sample_df.columns if c not in out_cols]
            sample_df = sample_df[out_cols + remaining]

            # --------------------------------------------------
            # Rename to stable public names
            # --------------------------------------------------
            rename = {
                c_player: "player_id",
                c_gw: "gameweek",
                c_pred: "predicted_points",
            }
            if c_name is not None:
                rename[c_name] = "name"
            if c_team is not None:
                rename[c_team] = "team"
            if c_pos is not None:
                rename[c_pos] = "position"
            if c_true is not None:
                rename[c_true] = "actual_points"

            sample_df = sample_df.rename(columns=rename)

            if "name" in sample_df.columns:
                sample_df = sample_df.sort_values(["name", "gameweek"]).reset_index(drop=True)

            # --------------------------------------------------
            # Write CSV (full columns)
            # --------------------------------------------------
            pred_csv = predictions_dir / "10_random_gbm_predictions.csv"
            sample_df.to_csv(pred_csv, index=False)

            # --------------------------------------------------
            # Write Markdown (compact, one-line friendly)
            # --------------------------------------------------
            pred_md = predictions_dir / "10_random_gbm_predictions.md"

            md_sample = sample_df.copy()
            md_sample = md_sample.rename(
                columns={
                    "position": "pos",
                    "gameweek": "GW",
                    "predicted_points": "pred",
                    "actual_points": "actual",
                }
            )

            for c in ["pred", "actual"]:
                if c in md_sample.columns:
                    md_sample[c] = pd.to_numeric(md_sample[c], errors="coerce").map(
                        lambda x: f"{x:.2f}" if pd.notna(x) else ""
                    )

            md_preview_cols = [
                c
                for c in ["player_id", "name", "team", "pos", "GW", "pred", "actual"]
                if c in md_sample.columns
            ]

            md_lines_pred = []
            md_lines_pred.append(f"# Sample predictions – {test_season} (GBM seasonal)\n")
            md_lines_pred.append(
                "10 random players, each evaluated on 1 randomly selected gameweek "
                "(reproducible RNG seed=42).\n"
            )
            md_lines_pred.append(md_sample[md_preview_cols].to_markdown(index=False) + "\n")

            pred_md.write_text("\n".join(md_lines_pred), encoding="utf-8")

        except Exception as e:
            (predictions_dir / "10_random_gbm_predictions.ERROR.txt").write_text(
                f"Failed to generate random sample predictions: {e}\n",
                encoding="utf-8",
            )

    except Exception:
        pass

    # ------------------------------------------------------
    # NEW: Export summary tables (players / teams)
    # Non-blocking: tables are informative only
    # ------------------------------------------------------
    try:
        preds_tables = get_test_predictions_seasonal_gbm_gw(test_season=test_season).copy()

        export_tables_results(
            preds_tables,
            output_dir=output_dir,
            test_season=test_season,
            points_col="predicted_points",
            top_players_n=10,
            top_teams_n=5,
            top_n_team_players=11,
        )

    except Exception as e:
        (tables_dir / "tables_export.ERROR.txt").write_text(
            f"Tables export failed: {e}\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------
    # Optional bookmaker comparison export (CSV + Markdown)
    # Non-blocking: informative only
    # ------------------------------------------------------
    try:
        # Requires: export_bookmaker_comparison(...) to exist in reporting.py
        export_bookmaker_comparison(
            test_season=test_season,
            output_dir=output_dir,
            model="gw_seasonal_gbm",
            verbose=False,
        )
    except Exception as e:
        (predictions_dir / "bookmakers_comp.ERROR.txt").write_text(
            f"Bookmaker comparison failed: {type(e).__name__}: {e}\n",
            encoding="utf-8",
        )

    return df



# =============================================================================
# TABLE EXPORT HELPERS
# =============================================================================


def _resolve_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name that exists in df among candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _resolve_points_col(df: pd.DataFrame) -> str:
    """
    Choose best available points column for 'performance'.
    Priority: actual -> total -> predicted -> generic.
    """
    for c in ["actual_points", "total_points", "points", "predicted_points", "y_true", "y_pred"]:
        if c in df.columns:
            return c
    raise KeyError(
        "Cannot infer points column. Expected one of: "
        "actual_points, total_points, points, predicted_points, y_true, y_pred."
    )


def export_tables_results(
    df: pd.DataFrame,
    *,
    output_dir: str | Path = "results",
    test_season: str | None = None,
    gameweek: int | None = None,
    top_players_n: int = 10,
    top_teams_n: int = 5,
    points_col: str | None = None,
    top_n_team_players: int = 11,  # NEW: top-11 per team strength
) -> Dict[str, pd.DataFrame]:
    """
    Export summary tables under results/tables.

    Tables:
      1) top_players__<season>__<gw>.csv/.md
      2) top_teams__<season>__<gw>.csv/.md

    Team strength definition (user choice):
      - team_strength = sum of predicted_points of the top N players (N=11 by default)
        for each team within the filtered slice (season/GW if provided).
    """
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    _ensure_dir(tables_dir)

    df0 = df.copy()
    top_teams_out: pd.DataFrame | None = None


    # Canonicalize ground-truth points column to "actual_points"
    if "actual_points" not in df0.columns:
        if "points" in df0.columns:
            df0 = df0.rename(columns={"points": "actual_points"})
        elif "total_points" in df0.columns:
            df0 = df0.rename(columns={"total_points": "actual_points"})

    # Resolve column aliases
    c_season = _resolve_first_existing(df0, ["season"])
    c_gw = _resolve_first_existing(df0, ["gameweek", "gw", "round"])
    c_team = _resolve_first_existing(df0, ["team", "team_name"])
    c_name = _resolve_first_existing(df0, ["name", "player_name", "web_name"])
    c_pos = _resolve_first_existing(df0, ["position", "pos"])
    c_pid = _resolve_first_existing(df0, ["player_id", "element", "id"])

    # Optional filtering
    if test_season is not None and c_season is not None:
        df0[c_season] = df0[c_season].astype(str)
        df0 = df0[df0[c_season] == str(test_season)].copy()

    if gameweek is not None and c_gw is not None:
        df0[c_gw] = pd.to_numeric(df0[c_gw], errors="coerce")
        df0 = df0[df0[c_gw] == int(gameweek)].copy()

    if df0.empty:
        raise ValueError("export_tables_results: dataframe empty after filtering.")

    # Points column for top players table
    pcol = points_col if (points_col is not None and points_col in df0.columns) else _resolve_points_col(df0)
    df0[pcol] = pd.to_numeric(df0[pcol], errors="coerce")

    # Tag for filenames
    season_tag = (str(test_season).replace("/", "_") if test_season else "all_seasons")
    gw_tag = (f"gw{int(gameweek)}" if gameweek is not None else "all_gw")

    # ------------------------------------------------------------------
    # 1a) Random gameweeks: best player (1 per GW, 10 random GWs)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)  # reproductible

    # Column to keep
    keep_cols = []
    for c in [c_gw, c_pid, c_name, c_team, "venue", "opponent", c_pos, "predicted_points", "actual_points"]:
        if c is not None and c in df0.columns and c not in keep_cols:
            keep_cols.append(c)

    # We explicitly choose the criterion of ‘best player’.
    if "predicted_points" not in df0.columns:
        raise KeyError("random_gw_best_player requires 'predicted_points' column.")

    if c_gw is None or c_gw not in df0.columns:
        raise KeyError("random_gw_best_player requires a gameweek column (gameweek/gw/round).")

    df1 = df0.copy()
    df1[c_gw] = pd.to_numeric(df1[c_gw], errors="coerce")
    df1["predicted_points"] = pd.to_numeric(df1["predicted_points"], errors="coerce")

    # Valid lines only
    df1 = df1.dropna(subset=[c_gw, "predicted_points"]).copy()
    df1[c_gw] = df1[c_gw].astype(int)

    # Draw 10 distinct GWs at random
    available_gws = np.array(sorted(df1[c_gw].unique()))
    if len(available_gws) == 0:
        raise ValueError("random_gw_best_player: no gameweeks available after filtering.")

    n_gws = int(min(10, len(available_gws)))
    sampled_gws = rng.choice(available_gws, size=n_gws, replace=False)

    # For each GW drawn, take the player with the best prediction.
    best_rows = []
    for gw in sampled_gws:
        sub = df1[df1[c_gw] == int(gw)].copy()
        sub = sub.sort_values(by="predicted_points", ascending=False)
        best_rows.append(sub.iloc[0])

    top_players = pd.DataFrame(best_rows).reset_index(drop=True)

    # --------------------------------------------------------------
    # Add opponent + venue columns from fixtures
    # --------------------------------------------------------------
    try:
        fixtures = load_fixtures().copy()

        fx_season = next((c for c in ["season", "Season", "season_str"] if c in fixtures.columns), None)
        fx_gw = next((c for c in ["gameweek", "gw", "round", "event"] if c in fixtures.columns), None)
        fx_home = next((c for c in ["home_team", "team_h", "home"] if c in fixtures.columns), None)
        fx_away = next((c for c in ["away_team", "team_a", "away"] if c in fixtures.columns), None)

        if fx_gw and fx_home and fx_away:
            fixtures[fx_gw] = pd.to_numeric(fixtures[fx_gw], errors="coerce")

            # filter season if possible
            if fx_season is not None:
                fixtures[fx_season] = fixtures[fx_season].astype(str)
                fixtures = fixtures[fixtures[fx_season] == str(test_season)].copy()

            # build mapping (gw, team) -> opponent + venue
            team_to_opp = {}
            team_to_venue = {}
            for _, r in fixtures.iterrows():
                gw = r[fx_gw]
                ht = r[fx_home]
                at = r[fx_away]
                if pd.notna(gw) and pd.notna(ht) and pd.notna(at):
                    gwi = int(gw)
                    team_to_opp[(gwi, ht)] = at
                    team_to_opp[(gwi, at)] = ht
                    team_to_venue[(gwi, ht)] = "H"
                    team_to_venue[(gwi, at)] = "A"

            if "team" in top_players.columns and "gameweek" in top_players.columns:
                top_players["opponent"] = top_players.apply(
                    lambda r: team_to_opp.get((int(r["gameweek"]), r["team"]), pd.NA),
                    axis=1,
                )
                top_players["venue"] = top_players.apply(
                    lambda r: team_to_venue.get((int(r["gameweek"]), r["team"]), ""),
                    axis=1,
                )
            else:
                top_players["opponent"] = pd.NA
                top_players["venue"] = ""
        else:
            top_players["opponent"] = pd.NA
            top_players["venue"] = ""

    except Exception:
        top_players["opponent"] = pd.NA
        top_players["venue"] = ""

    # Keep only useful columns
    top_players_out = top_players[keep_cols].copy() if keep_cols else top_players.copy()

    # Rename to stable names
    rename_players = {}
    if c_pid and c_pid in top_players_out.columns:
        rename_players[c_pid] = "player_id"
    if c_name and c_name in top_players_out.columns:
        rename_players[c_name] = "name"
    if c_team and c_team in top_players_out.columns:
        rename_players[c_team] = "team"
    if c_pos and c_pos in top_players_out.columns:
        rename_players[c_pos] = "position"
    if c_gw and c_gw in top_players_out.columns:
        rename_players[c_gw] = "gameweek"

    top_players_out = top_players_out.rename(columns=rename_players)

    # Rounding for readability
    for col in ["predicted_points", "actual_points"]:
        if col in top_players_out.columns:
            top_players_out[col] = pd.to_numeric(top_players_out[col], errors="coerce").round(2)

    _write_table_csv_md(top_players_out, tables_dir / "best_player_random_gw")


    # ------------------------------------------------------------------
    # 1b) Top 10 players to have for ONE demo gameweek 
    #     Output: CSV + Markdown (single table)
    # ------------------------------------------------------------------

    selected_gw = 21 #CHANGE HERE IF YOU WANT TO TEST ANOTHER GAMEWEEK

    if c_gw is None or c_gw not in df0.columns:
        raise KeyError("top10_players_demo_gw requires a gameweek column (gameweek/gw/round).")

    if "predicted_points" not in df0.columns:
        raise KeyError("top10_players_demo_gw requires 'predicted_points' column.")

    # --- Prepare base slice with numeric GW + predicted_points
    candidates = df0.copy()
    candidates[c_gw] = pd.to_numeric(candidates[c_gw], errors="coerce")
    candidates["predicted_points"] = pd.to_numeric(candidates["predicted_points"], errors="coerce")
    candidates = candidates.dropna(subset=[c_gw, "predicted_points"]).copy()
    candidates[c_gw] = candidates[c_gw].astype(int)

    available_gws = set(int(x) for x in candidates[c_gw].unique())
    if int(selected_gw) not in available_gws:
        raise ValueError(
            f"top10_players_demo_gw: requested GW {selected_gw} not found in data. "
            f"Available sample: {sorted(list(available_gws))[:10]}..."
        )

    # --- Top 10 for the selected GW
    top_gw = (
        candidates[candidates[c_gw] == int(selected_gw)]
        .sort_values(by="predicted_points", ascending=False)
        .head(10)
        .copy()
    )

    # --- Add opponent + venue from fixtures for this GW
    try:
        fixtures = load_fixtures().copy()

        fx_season = next((c for c in ["season", "Season", "season_str"] if c in fixtures.columns), None)
        fx_gw = next((c for c in ["gameweek", "gw", "round", "event"] if c in fixtures.columns), None)
        fx_home = next((c for c in ["home_team", "team_h", "home"] if c in fixtures.columns), None)
        fx_away = next((c for c in ["away_team", "team_a", "away"] if c in fixtures.columns), None)

        if fx_gw and fx_home and fx_away:
            fixtures[fx_gw] = pd.to_numeric(fixtures[fx_gw], errors="coerce")

            if fx_season is not None:
                fixtures[fx_season] = fixtures[fx_season].astype(str)
                fixtures = fixtures[fixtures[fx_season] == str(test_season)].copy()

            fixtures = fixtures[fixtures[fx_gw] == int(selected_gw)].copy()

            team_to_opp = {}
            team_to_venue = {}
            for _, r in fixtures.iterrows():
                gwv = r[fx_gw]
                ht = r[fx_home]
                at = r[fx_away]
                if pd.notna(gwv) and pd.notna(ht) and pd.notna(at):
                    gwi = int(gwv)
                    team_to_opp[(gwi, ht)] = at
                    team_to_opp[(gwi, at)] = ht
                    team_to_venue[(gwi, ht)] = "H"
                    team_to_venue[(gwi, at)] = "A"

            if c_team is not None and c_team in top_gw.columns:
                top_gw["opponent"] = top_gw.apply(
                    lambda r: team_to_opp.get((int(r[c_gw]), r[c_team]), pd.NA),
                    axis=1,
                )
                top_gw["venue"] = top_gw.apply(
                    lambda r: team_to_venue.get((int(r[c_gw]), r[c_team]), ""),
                    axis=1,
                )
            else:
                top_gw["opponent"] = pd.NA
                top_gw["venue"] = ""
        else:
            top_gw["opponent"] = pd.NA
            top_gw["venue"] = ""

    except Exception:
        top_gw["opponent"] = pd.NA
        top_gw["venue"] = ""

    # --- Canonicalize ground-truth points column to "actual_points" (if needed)
    if "actual_points" not in top_gw.columns:
        if "points" in top_gw.columns:
            top_gw = top_gw.rename(columns={"points": "actual_points"})
        elif "total_points" in top_gw.columns:
            top_gw = top_gw.rename(columns={"total_points": "actual_points"})

    # --- Keep & rename columns (ensure GW included)
    keep_cols_top: list[str] = []
    for c in [
        c_gw,
        c_pid,
        c_name,
        c_team,
        "venue",
        "opponent",
        "predicted_points",
        "actual_points",
    ]:
        if c is not None and c in top_gw.columns and c not in keep_cols_top:
            keep_cols_top.append(c)

    top_gw_out = top_gw[keep_cols_top].copy() if keep_cols_top else top_gw.copy()

    rename_top: dict[str, str] = {}
    if c_pid and c_pid in top_gw_out.columns:
        rename_top[c_pid] = "player_id"
    if c_name and c_name in top_gw_out.columns:
        rename_top[c_name] = "name"
    if c_team and c_team in top_gw_out.columns:
        rename_top[c_team] = "team"
    if c_gw and c_gw in top_gw_out.columns:
        rename_top[c_gw] = "gameweek"

    top_gw_out = top_gw_out.rename(columns=rename_top)

    # --- Numeric rounding for readability
    for col in ["predicted_points", "actual_points", "gameweek"]:
        if col in top_gw_out.columns:
            top_gw_out[col] = pd.to_numeric(top_gw_out[col], errors="coerce").round(2)

    # Sort for nicer display
    if "predicted_points" in top_gw_out.columns:
        # (GW is constant anyway, but keeps determinism)
        sort_cols = [c for c in ["gameweek", "predicted_points"] if c in top_gw_out.columns]
        asc = [True, False] if sort_cols == ["gameweek", "predicted_points"] else [False]
        top_gw_out = top_gw_out.sort_values(sort_cols, ascending=asc).reset_index(drop=True)

    # Light truncation for readability
    def _truncate_name(s: object, n: int = 18) -> str:
        if pd.isna(s):
            return ""
        s = str(s)
        return s if len(s) <= n else s[: n - 1] + "…"

    if "name" in top_gw_out.columns:
        top_gw_out["name"] = top_gw_out["name"].map(_truncate_name)

    # --- Write CSV + Markdown (single-table output)
    out_base = tables_dir / "top10_players_demo_gw"
    _write_table_csv_md(top_gw_out, out_base)

    # Overwrite .md with a single clean table (no GW-splitting)
    try:
        md_lines: list[str] = []
        md_lines.append(f"# Top 10 players – demo GW{selected_gw} ({test_season})\n")

        md_df = top_gw_out.copy().rename(
            columns={
                "player_id": "pid",
                "gameweek": "gw",
                "predicted_points": "pred_points",
                "actual_points": "act_points",
            }
        )

        for c in ["team", "opponent"]:
            if c in md_df.columns:
                repl = {
                    "Brighton & Hove Albion": "Brighton",
                    "Tottenham Hotspur": "Spurs",
                    "Manchester City": "Man City",
                    "Manchester United": "Man United",
                    "Newcastle United": "Newcastle",
                    "Wolverhampton Wanderers": "Wolves",
                    "Nottingham Forest": "Nott'm Forest",
                    "West Ham United": "West Ham",
                    "AFC Bournemouth": "Bournemouth",
                    "Sheffield United": "Sheff Utd",
                    "Crystal Palace": "Palace",
                }
                md_df[c] = md_df[c].map(lambda x: "" if pd.isna(x) else repl.get(str(x), str(x)))

        for c in ["pred_points", "act_points"]:
            if c in md_df.columns:
                md_df[c] = pd.to_numeric(md_df[c], errors="coerce").map(
                    lambda x: f"{x:.2f}" if pd.notna(x) else ""
                )

        md_lines.append(md_df.to_markdown(index=False, tablefmt="github") + "\n")
        out_base.with_suffix(".md").write_text("\n".join(md_lines), encoding="utf-8")
    except Exception:
        pass



    # ------------------------------------------------------------------
    # 2) Top 5 teams (sum of top-11 predicted points players)
    # ------------------------------------------------------------------
    if c_team is None:
        raise KeyError("export_tables_results: cannot compute team table without a 'team' column.")
    if "predicted_points" not in df0.columns:
        raise KeyError(
            "export_tables_results: team strength requires 'predicted_points' column "
            "(since you asked 'somme des points prédits des 11 meilleurs')."
        )

    df0["predicted_points"] = pd.to_numeric(df0["predicted_points"], errors="coerce")
    df_team = df0.dropna(subset=[c_team, "predicted_points"]).copy()

    # Sort players within each team by predicted points (desc)
    df_team = df_team.sort_values([c_team, "predicted_points"], ascending=[True, False])

    # Keep top N players per team (default N=11)
    topN = df_team.groupby(c_team, dropna=False, as_index=False).head(int(top_n_team_players))

    team_strength = (
        topN.groupby(c_team, dropna=False)["predicted_points"]
        .sum(min_count=1)
        .rename("top11_sum")
        .reset_index()
        .rename(columns={c_team: "team"})
    )

    team_strength["top11_avg"] = team_strength["top11_sum"] / float(top_n_team_players)

    team_counts = (
        df_team.groupby(c_team, dropna=False)
        .agg(
            rows=("predicted_points", "size"),
            n_players_used=(c_name, "nunique") if c_name is not None else ("predicted_points", "size"),
        )
        .reset_index()
        .rename(columns={c_team: "team"})
    )

    top_teams_out = team_strength.merge(team_counts, on="team", how="left")
    top_teams_out["top_n_players_used"] = int(top_n_team_players)

    top_teams_out = (
        top_teams_out.sort_values("top11_sum", ascending=False)
        .head(int(top_teams_n))
        .reset_index(drop=True)
    )

    # Rounding for readability
    for col in ["top11_sum", "top11_avg"]:
        if col in top_teams_out.columns:
            top_teams_out[col] = pd.to_numeric(top_teams_out[col], errors="coerce").round(2)

    _write_table_csv_md(top_teams_out, tables_dir / "top5_teams")


    return {"top_players": top_players_out, "top_teams": top_teams_out}


def export_bookmaker_comparison(
    *,
    test_season: str = "2022/23",
    output_dir: str | Path = "results",
    model: str = "gw_seasonal_gbm",
    verbose: bool = False,
) -> tuple[pd.DataFrame, float, float]:
    """
    Export bookmaker comparison under results/predictions:
      - bookmakers_comp__<season>.csv : full match-level table from evaluation.py
      - bookmakers_comp__<season>.md  : report-friendly summary (MAE/Corr/n + worst/best 5)

    Returns (comp, mae, corr).
    """
    output_dir = Path(output_dir)
    predictions_dir = output_dir / "predictions"
    _ensure_dir(predictions_dir)

    from src.evaluation import compare_model_vs_bookmakers
    # summarize_bookmaker_comparison is in this module (reporting.py) per your snippet
    # otherwise import it from where it lives.

    comp, mae, corr = compare_model_vs_bookmakers(
        model=model,
        test_season=test_season,
        verbose=verbose,
    )

    # Display-ready tables (already renamed + diff + rounded)
    worst5, best5 = summarize_bookmaker_comparison(comp)

    # ------------------------
    # CSV (full comp)
    # ------------------------
    season_tag = test_season.replace("/", "_")
    out_csv = predictions_dir / f"bookmakers_comparison.csv"
    comp.to_csv(out_csv, index=False)

    # ------------------------
    # Markdown (photo-ready)
    # ------------------------
    out_md = predictions_dir / f"bookmakers_comparison.md"

    md_lines: list[str] = []
    md_lines.append(f"# Bookmaker comparison (Bet365) — {test_season}\n")
    md_lines.append("## Statistics summary\n")
    md_lines.append(f"- **MAE**: {float(mae):.3f}")
    md_lines.append(f"- **Corr**: {float(corr):.3f}")
    md_lines.append(f"- **n**: {int(len(comp))}\n")

    md_lines.append("## Worst 5 predictions (largest |Bet365 − model|)\n")
    md_lines.append(worst5.to_markdown(index=False, tablefmt="github") + "\n")

    md_lines.append("## Best 5 predictions (smallest |Bet365 − model|)\n")
    md_lines.append(best5.to_markdown(index=False, tablefmt="github") + "\n")

    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    return comp, float(mae), float(corr)
