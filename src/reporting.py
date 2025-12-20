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
    get_test_predictions_seasonal_gbm_gw
)


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
    ax.scatter(y_true, y_pred, alpha=0.25)
    ax.set_xlabel("Actual points")
    ax.set_ylabel("Predicted points")
    ax.set_title(title)

    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

    return fig


def _plot_residuals_hist(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    resid = y_pred - y_true
    fig = plt.figure()
    ax = plt.gca()
    ax.hist(resid, bins=40)
    ax.set_xlabel("Residual (pred - actual)")
    ax.set_ylabel("Count")
    ax.set_title(title)

    ax.axvline(0.0, linestyle="--", linewidth=1)

    return fig


def _plot_rank_vs_rank(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    df["rank_true"] = df["y_true"].rank(method="average")
    df["rank_pred"] = df["y_pred"].rank(method="average")

    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(df["rank_true"], df["rank_pred"], alpha=0.25)
    ax.set_xlabel("Actual rank")
    ax.set_ylabel("Predicted rank")
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


def export_gw_results(
    test_season: str = "2022/23",
    output_dir: str | Path = "results",
) -> pd.DataFrame:
    """
    Generate GW evaluation outputs (metrics + figures) and write them to results/.

    Writes:
      - results/metrics/gw_metrics_detailed.csv
      - results/metrics/gw_metrics_detailed.txt
      - results/metrics/gw_metrics_summary.txt
      - results/figures/gw/<model_key>_pred_vs_actual.png
      - results/figures/gw/<model_key>_residuals.png
      - results/figures/gw/<model_key>_rank_vs_rank.png
      - results/figures/gw/summary_mae_by_model.png
      - results/figures/gw/summary_spearman_by_model.png
    """
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures" / "gw"
    _ensure_dir(metrics_dir)
    _ensure_dir(figures_dir)

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

        mae = _compute_mae(y_true, y_pred)
        rmse = _compute_rmse(y_true, y_pred)
        r2 = _compute_r2(y_true, y_pred)
        spearman = _compute_spearman(y_true, y_pred)

        rows.append(
            {
                "model_key": model_key,
                "test_season": test_season,
                "n_obs": int(len(y_true)),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "spearman": spearman,
            }
        )

        fig1 = _plot_pred_vs_actual(
            y_true, y_pred, f"{model_key} – Pred vs Actual ({test_season})"
        )
        _save_fig(fig1, figures_dir / f"{model_key}_pred_vs_actual.png")

        fig2 = _plot_residuals_hist(
            y_true, y_pred, f"{model_key} – Residuals ({test_season})"
        )
        _save_fig(fig2, figures_dir / f"{model_key}_residuals.png")

        fig3 = _plot_rank_vs_rank(
            y_true, y_pred, f"{model_key} – Rank vs Rank ({test_season})"
        )
        _save_fig(fig3, figures_dir / f"{model_key}_rank_vs_rank.png")

    df = pd.DataFrame(rows).sort_values("mae", ascending=True).reset_index(drop=True)

    # Round for readability
    for col, nd in {"mae": 3, "rmse": 3, "r2": 3, "spearman": 3}.items():
        if col in df.columns:
            df[col] = df[col].round(nd)

    # CSV export
    df.to_csv(metrics_dir / "gw_metrics_detailed.csv", index=False)

    # Plain-text table export (no extra dependency)
    (metrics_dir / "gw_metrics_detailed.txt").write_text(
        df.to_string(index=False),
        encoding="utf-8",
    )

    # Short summary for report
    best = df.iloc[0]
    summary = (
        f"Hold-out season {test_season}: best GW model by MAE is {best['model_key']} "
        f"(MAE={best['mae']:.3f}, RMSE={best['rmse']:.3f}, "
        f"R2={best['r2']:.3f}, Spearman={best['spearman']:.3f}).\n"
    )
    (metrics_dir / "gw_metrics_summary.txt").write_text(summary, encoding="utf-8")

    # Summary figures
    fig_mae = _plot_metric_bar(df, "mae", f"MAE by model ({test_season})")
    _save_fig(fig_mae, figures_dir / "summary_mae_by_model.png")

    fig_spear = _plot_metric_bar(df, "spearman", f"Spearman by model ({test_season})")
    _save_fig(fig_spear, figures_dir / "summary_spearman_by_model.png")

    return df


def export_gbm_results_by_position(
    test_season: str = "2022/23",
    output_dir: str | Path = "results",
) -> pd.DataFrame:
    """
    Export GBM seasonal evaluation broken down by position.
    Writes:
      - results/metrics/gbm_seasonal_metrics_by_position.csv
      - results/figures/gw/gbm_seasonal_mae_by_position.png
      - results/figures/gw/gbm_seasonal_spearman_by_position.png
    """
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures" / "gw"
    _ensure_dir(metrics_dir)
    _ensure_dir(figures_dir)

    df = get_test_predictions_seasonal_gbm_gw(test_season=test_season)

    # Compute per-position metrics
    rows = []
    for pos, g in df.groupby("position"):
        y_true = g["points"].to_numpy(dtype=float)
        y_pred = g["predicted_points"].to_numpy(dtype=float)

        rows.append(
            {
                "model_key": "gbm_seasonal",
                "test_season": test_season,
                "position": pos,
                "n_obs": int(len(g)),
                "mae": _compute_mae(y_true, y_pred),
                "rmse": _compute_rmse(y_true, y_pred),
                "r2": _compute_r2(y_true, y_pred),
                "spearman": _compute_spearman(y_true, y_pred),
            }
        )

    out = pd.DataFrame(rows).sort_values("mae", ascending=True).reset_index(drop=True)

    for col, nd in {"mae": 3, "rmse": 3, "r2": 3, "spearman": 3}.items():
        if col in out.columns:
            out[col] = out[col].round(nd)

    out_path = metrics_dir / "gbm_seasonal_metrics_by_position.csv"
    out.to_csv(out_path, index=False)

    # Figures: MAE by position
    fig_mae = plt.figure()
    ax = plt.gca()
    ax.bar(out["position"], out["mae"])
    ax.set_xlabel("Position")
    ax.set_ylabel("MAE")
    ax.set_title(f"GBM seasonal – MAE by position ({test_season})")
    _save_fig(fig_mae, figures_dir / "gbm_seasonal_mae_by_position.png")

    # Figures: Spearman by position
    fig_sp = plt.figure()
    ax = plt.gca()
    ax.bar(out["position"], out["spearman"])
    ax.set_xlabel("Position")
    ax.set_ylabel("Spearman")
    ax.set_title(f"GBM seasonal – Spearman by position ({test_season})")
    _save_fig(fig_sp, figures_dir / "gbm_seasonal_spearman_by_position.png")

    return out


def export_gbm_error_tables(
    test_season: str = "2022/23",
    output_dir: str | Path = "results",
    n: int = 15,
) -> None:
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    _ensure_dir(tables_dir)

    df = get_test_predictions_seasonal_gbm_gw(test_season=test_season).copy()

    cols = ["season", "gameweek", "player_id", "name", "team", "position", "points", "predicted_points", "error", "abs_error"]

    over = df.sort_values("error", ascending=False).head(n)[cols]
    under = df.sort_values("error", ascending=True).head(n)[cols]

    over.to_csv(tables_dir / "gbm_seasonal_top_overpredicted.csv", index=False)
    under.to_csv(tables_dir / "gbm_seasonal_top_underpredicted.csv", index=False)
