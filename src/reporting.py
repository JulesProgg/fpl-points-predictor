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
) -> pd.DataFrame:
    """
    Export a random sample of matches for a given season, with predicted team strength
    for both teams (home/away), where team strength is defined as the sum of predicted
    points of the top-N predicted players in the squad for that GW (proxy XI).

    Writes (REPLACES previous team strength exports):
      - results/predictions/sample_match_team_strength__<season>__gbm_seasonal_top<N>.csv
      - results/predictions/sample_match_team_strength__<season>__gbm_seasonal_top<N>.md

    Output has at most `sample_n_matches` rows (one row per match).
    """
    output_dir = Path(output_dir)
    predictions_dir = output_dir / "predictions"
    _ensure_dir(predictions_dir)

    # 1) Load player-level predictions for the season (GBM seasonal)
    preds = get_test_predictions_seasonal_gbm_gw(test_season=test_season).copy()

    required = {"team", "season", "gameweek", "predicted_points"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(
            f"get_test_predictions_seasonal_gbm_gw is missing required columns: {sorted(missing)}"
        )

    preds["gameweek"] = preds["gameweek"].astype(int)
    preds["predicted_points"] = pd.to_numeric(preds["predicted_points"], errors="coerce")
    preds = preds.dropna(subset=["team", "season", "gameweek", "predicted_points"])
    preds = preds[preds["season"].astype(str) == str(test_season)].copy()

    if preds.empty:
        raise ValueError(f"No predictions available for season {test_season!r}.")

    # 2) Compute team strength per (season, gameweek, team): sum of top-N predicted players
    preds = preds.sort_values(
        ["season", "gameweek", "team", "predicted_points"],
        ascending=[True, True, True, False],
    )
    preds["rank_within_team_gw"] = preds.groupby(["season", "gameweek", "team"]).cumcount() + 1
    top = preds[preds["rank_within_team_gw"] <= int(top_n_players)].copy()

    team_strength = (
        top.groupby(["season", "gameweek", "team"], as_index=False)["predicted_points"]
        .sum()
        .rename(columns={"predicted_points": "team_strength_pred_points"})
    )

    # 3) Load fixtures for the season
    fixtures = load_fixtures().copy()
    fixtures = fixtures[fixtures["season"].astype(str) == str(test_season)].copy()
    fixtures["gameweek"] = fixtures["gameweek"].astype(int)

    if fixtures.empty:
        raise ValueError(f"No fixtures available for season {test_season!r}.")

    # 4) Merge home and away strengths into fixtures
    out = fixtures.merge(
        team_strength.rename(
            columns={"team": "home_team", "team_strength_pred_points": "home_strength"}
        ),
        on=["season", "gameweek", "home_team"],
        how="left",
    ).merge(
        team_strength.rename(
            columns={"team": "away_team", "team_strength_pred_points": "away_strength"}
        ),
        on=["season", "gameweek", "away_team"],
        how="left",
    )

    out["top_n"] = int(top_n_players)
    out["strength_diff"] = pd.to_numeric(out["home_strength"], errors="coerce") - pd.to_numeric(
        out["away_strength"], errors="coerce"
    )

    # 5) Sample N matches
    rng = np.random.default_rng(int(seed))
    n = int(min(sample_n_matches, len(out)))
    idx = rng.choice(len(out), size=n, replace=False)
    out_sample = out.iloc[idx].reset_index(drop=True)

    # Improve Markdown readability: round numeric columns
    for col in ["home_strength", "away_strength", "strength_diff"]:
        if col in out_sample.columns:
            out_sample[col] = out_sample[col].round(2)


    # 6) Output paths
    season_tag = str(test_season).replace("/", "_")
    csv_path = predictions_dir / f"sample_match_team_strength__{season_tag}__gbm_seasonal_top{top_n_players}.csv"
    md_path = predictions_dir / f"sample_match_team_strength__{season_tag}__gbm_seasonal_top{top_n_players}.md"

    # 7) Write CSV (full columns) and MD (readable subset)
    out_sample.to_csv(csv_path, index=False)

    md_cols = ["gameweek", "home_team", "away_team", "home_strength", "away_strength"]
    md_cols_existing = [c for c in md_cols if c in out_sample.columns]

    md_lines = []
    md_lines.append(f"# Sample match team strength – {test_season}\n")
    md_lines.append(
        f"{n} random matches (seed={seed}), team strength = sum of top-{top_n_players} predicted players (GBM seasonal).\n"
    )
    md_lines.append(out_sample[md_cols_existing].to_markdown(index=False) + "\n")
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
    # NEW: Predictions random sample (10 players, 1 random GW each)
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

        preds = preds[np.isfinite(preds[c_pred].astype(float))]
        if c_true is not None:
            preds = preds[np.isfinite(preds[c_true].astype(float))]

        rng = np.random.default_rng(42)

        unique_players = preds[c_player].dropna().unique()
        if len(unique_players) == 0:
            raise ValueError("No players available for sampling.")

        n_players = int(min(10, len(unique_players)))
        sampled_players = rng.choice(unique_players, size=n_players, replace=False)

        sampled_rows = []
        for pid in sampled_players:
            sub = preds[preds[c_player] == pid]
            idx = int(rng.integers(0, len(sub)))
            sampled_rows.append(sub.iloc[idx])

        sample_df = pd.DataFrame(sampled_rows).reset_index(drop=True)

        out_cols = []
        for c in [c_player, c_name, c_team, c_pos, c_gw, c_true, c_pred]:
            if c is not None and c in sample_df.columns and c not in out_cols:
                out_cols.append(c)

        remaining = [c for c in sample_df.columns if c not in out_cols]
        sample_df = sample_df[out_cols + remaining]

        rename = {c_player: "player_id", c_gw: "gameweek", c_pred: "predicted_points"}
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

        pred_csv = predictions_dir / "sample10_random_players_random_gw__gbm_seasonal.csv"
        sample_df.to_csv(pred_csv, index=False)

        pred_md = predictions_dir / "sample10_random_players_random_gw__gbm_seasonal.md"
        md_preview_cols = [
            c
            for c in ["player_id", "name", "team", "position", "gameweek", "actual_points", "predicted_points"]
            if c in sample_df.columns
        ]
        md_lines_pred = []
        md_lines_pred.append(f"# Sample predictions – {test_season} (GBM seasonal)\n")
        md_lines_pred.append(
            "10 random players, each evaluated on 1 randomly selected gameweek (reproducible RNG seed=42).\n"
        )
        md_lines_pred.append(sample_df[md_preview_cols].to_markdown(index=False) + "\n")
        pred_md.write_text("\n".join(md_lines_pred), encoding="utf-8")

    except Exception as e:
        (predictions_dir / "sample10_random_players_random_gw__gbm_seasonal.ERROR.txt").write_text(
            f"Failed to generate random sample predictions: {e}\n",
            encoding="utf-8",
        )

    # -----------------------------
    # NEW: Sample match team strength (replaces previous team-strength exports)
    # -----------------------------
    try:
        export_sample_match_team_strength(
            test_season=test_season,
            output_dir=output_dir,
            top_n_players=11,
            sample_n_matches=10,
            seed=42,
            )
    except Exception as e:
        (predictions_dir / "sample_match_team_strength.ERROR.txt").write_text(
        f"Failed to export sample match team strength: {e}\n",
        encoding="utf-8",
        )


    return df
