from __future__ import annotations

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


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save_fig(fig, path: Path) -> None:
    _ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def _write_table_csv_md(df: pd.DataFrame, out_base: Path) -> None:
    """Write df to CSV + compact Markdown with base path (no extension)."""
    _ensure_dir(out_base.parent)

    # 1) CSV full fidelity
    df.to_csv(out_base.with_suffix(".csv"), index=False, encoding="utf-8")

    # 2) Markdown compact view
    md_df = df.copy()

    # Short headers to prevent wrapping
    col_rename = {
        "position": "pos",
        "gameweek": "gw",
        "predicted_points": "pred_points",
        "actual_points": "act_points",
    }
    md_df = md_df.rename(columns={k: v for k, v in col_rename.items() if k in md_df.columns})

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
        x_line = np.array(
            [float(np.nanmin(y_true[mask])), float(np.nanmax(y_true[mask]))]
        )
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
        # same idea as evaluation.py: mean minutes / 90 clipped
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

    # Clean invalid rows similarly
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

    if len(valid_for_logit) >= 5:  # avoid fitting on tiny samples
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

        # write back into out (align by index)
        out.loc[valid_for_logit.index, "p_model_home_win"] = p_cal

    # Summary error columns if Bet365 exists
    out["abs_error_naive"] = np.abs(out["p_model_naive"] - out["pnorm_home_win"])
    out["abs_error_cal"] = np.abs(out["p_model_home_win"] - out["pnorm_home_win"])

    # ------------------------------------------------------
    # 7) sample = 10 matches
    # ------------------------------------------------------
    rng = np.random.default_rng(int(seed))
    out = out.reset_index(drop=True)

    # 1) Keep only matches where comparison makes sense
    eligible = out.dropna(subset=["pnorm_home_win", "p_model_home_win"]).copy()

    if len(eligible) < sample_n_matches:
        raise ValueError(
            f"Not enough comparable matches to sample {sample_n_matches}. "
            f"Only {len(eligible)} available."
        )

    # 2) Sample exactly N matches
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
    season_tag = str(test_season).replace("/", "_")
    csv_path = predictions_dir / (
        f"sample_match_team_strength__{season_tag}__gbm_seasonal_top{top_n_players}.csv"
    )
    md_path = predictions_dir / (
        f"sample_match_team_strength__{season_tag}__gbm_seasonal_top{top_n_players}.md"
    )

    # ------------------------------------------------------
    # 9) Write outputs
    # ------------------------------------------------------
    out_sample.to_csv(csv_path, index=False)

    md_lines: list[str] = []
    md_lines.append(f"# Sample match team strength – {test_season}\n")
    md_lines.append(
        f"{len(out_sample)} sampled matches ..."
    )

    md_cols = [
        "gameweek",
        "home_team",
        "away_team",
        "p_model_home_win",   # model implied home-win probability (calibrated)
        "pnorm_home_win",   # Bet365 normalized home-win probability
        "abs_error_cal",
    ]
    md_cols_existing = [c for c in md_cols if c in out_sample.columns]

    # --- Markdown-only view: rename columns to save space ---
    md_sample = out_sample[md_cols_existing].copy()
    md_sample = md_sample.rename(columns={
        "p_model_home_win": "p_home_model",
        "pnorm_home_win": "p_home_b365",
    })

    md_lines.append(md_sample.to_markdown(index=False) + "\n")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return out_sample


def export_gw_results(
    test_season: str = "2022/23",
    output_dir: str | Path = "results",
) -> pd.DataFrame:
    """
    Generate GW evaluation outputs (metrics + figures) and write them to results/.
    """
    from datetime import datetime
    import json

    # NEW: make sure output_dir is a Path (required for the new folders)
    output_dir = Path(output_dir)

    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures" / "gw"
    predictions_dir = output_dir / "predictions"
    tables_dir = output_dir / "tables"

    _ensure_dir(metrics_dir)
    _ensure_dir(figures_dir)
    _ensure_dir(predictions_dir)
    _ensure_dir(tables_dir)

    suites: Dict[str, Callable[[], Tuple[np.ndarray, np.ndarray]]] = {
        "linear_anytime_lag3": lambda: get_ytrue_ypred_anytime_linear_gw(
            3, test_season=test_season
        ),
        "linear_anytime_lag5": lambda: get_ytrue_ypred_anytime_linear_gw(
            5, test_season=test_season
        ),
        "linear_seasonal": lambda: get_ytrue_ypred_seasonal_linear_gw(
            test_season=test_season
        ),
        "gbm_seasonal": lambda: get_ytrue_ypred_seasonal_gbm_gw(
            test_season=test_season
        ),
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

        fig1 = _plot_pred_vs_actual(
            yt, yp, f"{model_key} – Pred vs Actual ({test_season})"
        )
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

    md_cols = [
        "model_key",
        "mae",
        "rmse",
        "r2",
        "spearman",
        "bias",
        "medae",
        "p90_ae",
        "rank_mae",
    ]
    md_cols_existing = [c for c in md_cols if c in df.columns]
    md_lines.append(df[md_cols_existing].to_markdown(index=False) + "\n")

    (metrics_dir / "gw_metrics_detailed.md").write_text("\n".join(md_lines), encoding="utf-8")

    leaderboard = {
        "type": "gw_holdout_leaderboard",
        "test_season": test_season,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "best_model": {
            "model_key": str(best["model_key"]),
            "mae": float(best["mae"]),
            "rmse": float(best["rmse"]),
            "r2": float(best["r2"]),
            "spearman": float(best["spearman"]),
            "bias": float(best["bias"]),
            "medae": float(best["medae"]),
            "p90_ae": float(best["p90_ae"]),
            "n_obs": int(best["n_obs"]),
        },
        "models": df.to_dict(orient="records"),
    }
    (metrics_dir / "gw_leaderboard.json").write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")

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
            pred_csv = predictions_dir / "sample10_random_players_random_gw__gbm_seasonal.csv"
            sample_df.to_csv(pred_csv, index=False)

            # --------------------------------------------------
            # Write Markdown (compact, one-line friendly)
            # --------------------------------------------------
            pred_md = predictions_dir / "sample10_random_players_random_gw__gbm_seasonal.md"

            md_sample = sample_df.copy()
            md_sample = md_sample.rename(columns={
                "position": "pos",
                "gameweek": "GW",
                "predicted_points": "pred",
                "actual_points": "actual",
            })

            for c in ["pred", "actual"]:
                if c in md_sample.columns:
                    md_sample[c] = pd.to_numeric(md_sample[c], errors="coerce").map(
                        lambda x: f"{x:.2f}" if pd.notna(x) else ""
                    )

            md_preview_cols = [
                c for c in ["player_id", "name", "team", "pos", "GW", "pred", "actual"]
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
            (predictions_dir / "sample10_random_players_random_gw__gbm_seasonal.ERROR.txt").write_text(
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
        preds_tables = get_test_predictions_seasonal_gbm_gw(
            test_season=test_season
        ).copy()

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
        
    return df


# ---------------------------------------------------------------------
# Helpers for tables export
# ---------------------------------------------------------------------

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
    # 1a) Random gameweek Top player
    # ------------------------------------------------------------------
    keep_cols = []
    for c in [c_gw, c_pid, c_name, c_team, c_pos, pcol, "predicted_points", "actual_points"]:
        if c is not None and c in df0.columns and c not in keep_cols:
            keep_cols.append(c)


    top_players = (
        df0.dropna(subset=[pcol])
        .sort_values(by=pcol, ascending=False)
        .head(int(top_players_n))
        .reset_index(drop=True)
    )
    top_players_out = top_players[keep_cols].copy() if keep_cols else top_players.copy()

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

    for col in [pcol, "predicted_points", "actual_points", "total_points"]:
        if col in top_players_out.columns:
            top_players_out[col] = pd.to_numeric(top_players_out[col], errors="coerce").round(3)

    _write_table_csv_md(
        top_players_out,
        tables_dir / "random_gw_best_player",
    )

    # ------------------------------------------------------------------
    # 1b) "Best XI" suggestion — Top 10 players for a given GW (GW10)
    # ------------------------------------------------------------------

    target_gw_for_squad = 10  # GW choisi (ici fixé à 10)
    squad_gw = pd.DataFrame()


    if c_gw is not None and c_gw in df0.columns:
        squad_candidates = df0.copy()
        squad_candidates[c_gw] = pd.to_numeric(squad_candidates[c_gw], errors="coerce")

        squad_gw = (
            squad_candidates[squad_candidates[c_gw] == target_gw_for_squad]
            .dropna(subset=[pcol])
            .sort_values(by=pcol, ascending=False)
            .head(10)
            .reset_index(drop=True)
        )

    if squad_gw.empty:
        (tables_dir / "gw10_top10_players.SKIPPED.txt").write_text(
            f"Skipped gw10_top10_players: no rows for GW={target_gw_for_squad} "
            f"(c_gw={c_gw!r}, df0_rows={len(df0)}).\n",
            encoding="utf-8",
        )
    else:
        # --------------------------------------------------------------
        # Build opponent from fixtures gw10
        # --------------------------------------------------------------
        try:
            from src.data_loader import load_fixtures

            fixtures = load_fixtures().copy()

            # Identify likely fixtures columns (best-effort)
            fx_season = next((c for c in ["season", "Season", "season_str"] if c in fixtures.columns), None)
            fx_gw = next((c for c in ["gameweek", "gw", "round", "event"] if c in fixtures.columns), None)
            fx_home = next((c for c in ["home_team", "team_h", "home"] if c in fixtures.columns), None)
            fx_away = next((c for c in ["away_team", "team_a", "away"] if c in fixtures.columns), None)

            if fx_gw and fx_home and fx_away:
                fixtures[fx_gw] = pd.to_numeric(fixtures[fx_gw], errors="coerce")
                fx = fixtures[fixtures[fx_gw] == target_gw_for_squad].copy()

                # If fixtures has a season col, try to filter to current season tag(s)
                if fx_season is not None:
                    season_candidates = []
                    try:
                        season_candidates.append(str(test_season))
                    except Exception:
                        pass
                    try:
                        season_candidates.append(str(season_tag))
                    except Exception:
                        pass
                    if season_candidates:
                        fx_season_str = fx[fx_season].astype(str)
                        fx = fx[fx_season_str.isin(season_candidates)].copy()

                # Build mapping both directions
                team_to_opp = {}
                for _, r in fx.iterrows():
                    ht = r[fx_home]
                    at = r[fx_away]
                    if pd.notna(ht) and pd.notna(at):
                        team_to_opp[ht] = at
                        team_to_opp[at] = ht

                # Apply mapping to squad_gw using c_team
                if c_team is not None and c_team in squad_gw.columns:
                    # Direct mapping first
                    squad_gw["opponent"] = squad_gw[c_team].map(team_to_opp)

                    # Fallback: if many NaNs, try a normalized string match (names vs names)
                    if squad_gw["opponent"].isna().mean() > 0.5:
                        def _norm(x):
                            return str(x).strip().lower()

                        team_to_opp_norm = {_norm(k): v for k, v in team_to_opp.items()}
                        squad_gw["opponent"] = squad_gw[c_team].map(lambda x: team_to_opp_norm.get(_norm(x), pd.NA))
                else:
                    squad_gw["opponent"] = pd.NA
            else:
                squad_gw["opponent"] = pd.NA

        except Exception:
            squad_gw["opponent"] = pd.NA

        # --------------------------------------------------------------
        # Output columns
        # --------------------------------------------------------------

        # Ensure ground-truth points column exists in the GW10 slice
        if "actual_points" not in squad_gw.columns:
            if "points" in squad_gw.columns:
                squad_gw = squad_gw.rename(columns={"points": "actual_points"})
            elif "total_points" in squad_gw.columns:
                squad_gw = squad_gw.rename(columns={"total_points": "actual_points"})

        # Build keep columns (only include columns that actually exist)
        keep_cols_squad: list[str] = []
        for c in [c_pid, c_name, c_team, "opponent", c_pos, pcol, "predicted_points", "actual_points"]:
            if c is not None and c in squad_gw.columns and c not in keep_cols_squad:
                keep_cols_squad.append(c)

        # Slice to output
        squad_gw_out = squad_gw[keep_cols_squad].copy() if keep_cols_squad else squad_gw.copy()

        # Renaming mapping (player_id/name/team/position)
        rename_squad: dict[str, str] = {}
        if c_pid and c_pid in squad_gw_out.columns:
            rename_squad[c_pid] = "player_id"
        if c_name and c_name in squad_gw_out.columns:
            rename_squad[c_name] = "name"
        if c_team and c_team in squad_gw_out.columns:
            rename_squad[c_team] = "team"
        if c_pos and c_pos in squad_gw_out.columns:
            rename_squad[c_pos] = "position"
        # "opponent" stays as-is
        # predicted_points / actual_points stay as-is (canonical names)

        squad_gw_out = squad_gw_out.rename(columns=rename_squad)

        # Numeric formatting
        num_cols = []
        for c in [pcol, "predicted_points", "actual_points"]:
            if c is not None and c not in num_cols:
                num_cols.append(c)

        for col in num_cols:
            if col in squad_gw_out.columns:
                squad_gw_out[col] = pd.to_numeric(squad_gw_out[col], errors="coerce").round(3)


        # compact team/opponent names to optimize markdown width
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
            if c in squad_gw_out.columns:
                squad_gw_out[c] = squad_gw_out[c].map(_compact_team_name)
    
        # --------------------------------------------------------------
        # Nbuild a dedicated compact "display" table 
        # --------------------------------------------------------------
        display_rename = {
            "position": "pos",
            "predicted_points": "pred_points",
            "actual_points": "act_points",
        }

        squad_gw_display = squad_gw_out.rename(
            columns={k: v for k, v in display_rename.items() if k in squad_gw_out.columns}
        )

        
        _write_table_csv_md(squad_gw_display, tables_dir / "gw10_top10_players")



        # ------------------------------------------------------------------
        # 2) Team top 5 (sum of top-11 predicted points players)
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

        # For each team: take top N predicted_points and sum
        df_team = df_team.sort_values([c_team, "predicted_points"], ascending=[True, False])
        topN = df_team.groupby(c_team, dropna=False, as_index=False).head(int(top_n_team_players))

        team_strength = (
        topN.groupby(c_team, dropna=False)["predicted_points"]
        .sum(min_count=1)
        .rename("top11_sum")  # RENAMED
        .reset_index()
        .rename(columns={c_team: "team"})
        )

        team_strength["top11_avg"] = (
        team_strength["top11_sum"] / float(top_n_team_players)
        )


        # Useful context columns
        team_counts = (
            df_team.groupby(c_team, dropna=False)
            .agg(rows=("predicted_points", "size"), n_players_used=(c_name, "nunique") if c_name is not None else ("predicted_points", "size"))
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


        top_teams_out["top11_sum"] = pd.to_numeric(
        top_teams_out["top11_sum"], errors="coerce").round(3)


        _write_table_csv_md(
            top_teams_out,
            tables_dir / "team_top5",
        )

    return {"top_players": top_players_out, "top_teams": top_teams_out}

