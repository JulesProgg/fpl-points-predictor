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

    hb = ax.hexbin(
        y_true,
        y_pred,
        gridsize=35,
        mincnt=1
    )
        # -----------------------------
    # 1) Linear trend line (central tendency)
    # -----------------------------
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() >= 2:
        a, b = np.polyfit(y_true[mask], y_pred[mask], 1)
        x_line = np.array([float(np.nanmin(y_true[mask])), float(np.nanmax(y_true[mask]))])
        y_line = a * x_line + b
        ax.plot(x_line, y_line, linewidth=2)  # (color optional)

    # -----------------------------
    # 2) Make hexes look "round" (regular) instead of stretched
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

    # Identify highest bar
    max_idx = int(np.argmax(counts))
    bar = patches[max_idx]

    # Bin interval
    left = float(bins[max_idx])
    right = float(bins[max_idx + 1])
    label = f"[{left:.2f}; {right:.2f}]"

    # Bar geometry
    x = bar.get_x() + bar.get_width() / 2
    y = bar.get_height() / 2

    # Annotate INSIDE the bar, vertically, in white
    ax.text (
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


def export_gw_results(
    test_season: str = "2022/23",
    output_dir: str | Path = "results",
) -> pd.DataFrame:
    """
    Generate GW evaluation outputs (metrics + figures) and write them to results/.

    Writes (for backward compatibility):
      - results/metrics/gw_metrics_detailed.csv
      - results/metrics/gw_metrics_detailed.txt
      - results/metrics/gw_metrics_summary.txt
      - results/figures/gw/<model_key>_pred_vs_actual.png
      - results/figures/gw/<model_key>_residuals.png

    Additional :
      - results/metrics/gw_metrics_detailed.md
      - results/metrics/gw_leaderboard.json
    """
    from datetime import datetime
    import json

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

        # Defensive: ensure numpy arrays
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        yt = y_true[mask]
        yp = y_pred[mask]

        # Derived diagnostics (no dependency on src.evaluation)
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
                # new metrics
                "bias": float(np.mean(resid)) if len(resid) else np.nan,
                "medae": float(np.median(abs_err)) if len(abs_err) else np.nan,
                "p90_ae": float(np.quantile(abs_err, 0.90)) if len(abs_err) else np.nan,
            }
        )

        fig1 = _plot_pred_vs_actual(
            yt, yp, f"{model_key} – Pred vs Actual ({test_season})"
        )
        _save_fig(fig1, figures_dir / f"{model_key}_pred_vs_actual.png")

        fig2 = _plot_residuals_hist(
            yt, yp, f"{model_key} – Residuals ({test_season})"
        )
        _save_fig(fig2, figures_dir / f"{model_key}_residuals.png")

    df = pd.DataFrame(rows)

    # Sort by MAE (primary), RMSE (secondary) for stable leaderboard
    df = df.sort_values(["mae", "rmse"], ascending=[True, True]).reset_index(drop=True)

    # Add ranks (1 = best)
    df["rank_mae"] = df["mae"].rank(method="min", ascending=True).astype(int)
    df["rank_rmse"] = df["rmse"].rank(method="min", ascending=True).astype(int)
    df["rank_r2"] = df["r2"].rank(method="min", ascending=False).astype(int)
    df["rank_spearman"] = df["spearman"].rank(method="min", ascending=False).astype(int)

    # Round for human readability
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

    # -----------------------------
    # (CSV)
    # -----------------------------
    df.to_csv(metrics_dir / "gw_metrics_detailed.csv", index=False)
    

    best = df.iloc[0]
    summary = (
        f"Hold-out season {test_season}: best GW model by MAE is {best['model_key']} "
        f"(MAE={best['mae']:.3f}, RMSE={best['rmse']:.3f}, "
        f"R2={best['r2']:.3f}, Spearman={best['spearman']:.3f}, "
        f"Bias={best['bias']:.3f}, MedAE={best['medae']:.3f}, P90_AE={best['p90_ae']:.3f}).\n"
    )
    

    # -----------------------------
    # (Markdown + JSON leaderboard)
    # -----------------------------
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

    # Defensive: keep only cols that exist (robust to future edits)
    md_cols_existing = [c for c in md_cols if c in df.columns]
    md_lines.append(df[md_cols_existing].to_markdown(index=False) + "\n")

    (metrics_dir / "gw_metrics_detailed.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )

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
    (metrics_dir / "gw_leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2), encoding="utf-8"
    )

    return df
