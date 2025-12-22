from __future__ import annotations

from pathlib import Path
import os
import sys


def _hr(n: int = 70) -> str:
    return "=" * n


def _set_project_root() -> Path:
    """
    Ensure execution from the project root (where main.py lives),
    so that Path.cwd() used in src/data_loader.py resolves correctly.
    """
    root = Path(__file__).resolve().parent
    os.chdir(root)
    return root


def _status_line(label: str, status: str, details: str = "") -> str:
    """
    Standardize status lines: DONE / SKIPPED / WARNING / ERROR / INFO
    """
    tail = f" | {details}" if details else ""
    return f"   - {status}: {label}{tail}"


def _df_to_string_one_line(df, cols: list[str] | None = None, float_round: int = 2) -> str:
    """
    Render a dataframe in a single compact line (records separated by " | ").
    Intended for small tables (best-by-position, top3).
    """
    import pandas as pd

    if df is None or len(df) == 0:
        return ""

    tmp = df.copy()
    if cols is not None:
        cols = [c for c in cols if c in tmp.columns]
        tmp = tmp[cols]

    # Round numeric columns for compactness
    for c in tmp.columns:
        if pd.api.types.is_numeric_dtype(tmp[c]):
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce").round(float_round)

    # Convert each row to "col=value" chunks
    parts: list[str] = []
    for _, r in tmp.iterrows():
        row_parts = []
        for c in tmp.columns:
            v = r.get(c, "")
            row_parts.append(f"{c}={v}")
        parts.append(", ".join(row_parts))
    return " | ".join(parts)


def _print_df(df, cols: list[str], one_line: bool = False, float_round: int = 2) -> None:
    """
    Print dataframe either as standard table or compact one-liner.
    """
    import pandas as pd

    if df is None or len(df) == 0:
        print("   (empty)")
        return

    show_cols = [c for c in cols if c in df.columns]
    out = df.copy()

    # Apply rounding (only to columns that exist)
    for c in show_cols:
        if c in out.columns and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce").round(float_round)

    if one_line:
        s = _df_to_string_one_line(out, cols=show_cols, float_round=float_round)
        print(f"   {s}")
    else:
        print(out[show_cols].to_string(index=False))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="FPL Points Predictor — main entrypoint (end-to-end runner)"
    )
    parser.add_argument("--season", default="2022/23", help='Test season string (e.g. "2022/23")')
    parser.add_argument(
    "--gw",
    type=int,
    default=None,
    help="Override a single gameweek for the prediction demo (ignores --gws if set)",
    )
    parser.add_argument(
        "--gws",
        type=int,
        nargs="+",
        default=[5, 20, 35],
        help="Gameweeks used for the prediction demo (default: 5 20 35)",
    )

    parser.add_argument("--output-dir", default="results", help="Output directory (default: results/)")

    # Output formatting
    parser.add_argument(
        "--one-line",
        action="store_true",
        help="Print compact one-line tables for recommendations (reduces vertical space).",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Do not print the big header block.",
    )

    # Pipeline steps (enabled by default)
    parser.add_argument("--skip-export", action="store_true", help="Skip GW metrics/figures exports")
    parser.add_argument("--skip-predict", action="store_true", help="Skip GW prediction demo")
    parser.add_argument(
        "--skip-sample-matches",
        action="store_true",
        help="Skip sample match team-strength export",
    )

    # Optional bookmaker comparison
    parser.add_argument(
        "--run-bookmakers",
        action="store_true",
        help="Run model vs Bet365 comparison (optional)",
    )

    args = parser.parse_args()
    demo_gws = [int(args.gw)] if args.gw is not None else [int(x) for x in args.gws]


    root = _set_project_root()
    output_dir = Path(args.output_dir)

    if not args.no_header:
        print(_hr())
        print("FPL Points Predictor — main.py")
        print(f"Project root : {root}")
        print(f"Test season  : {args.season}")
        print(f"Demo GWs     : {demo_gws}")
        print(f"Output dir   : {output_dir.resolve()}")
        print(_hr())

    # ------------------------------------------------------------------
    # High-level imports (stable public API)
    # ------------------------------------------------------------------
    from src import predict_gw_all_players
    from src.reporting import export_gw_results, export_sample_match_team_strength
    from src.data_loader import (
        PLAYER_GW_FILE,
        EPL_FIXTURES_FILE,
        ODDS_FILE,
        RAW_GW_FILE,
        RAW_ODDS_FILE,
    )

    # ------------------------------------------------------------------
    # 0) Data availability check (informative only)
    # ------------------------------------------------------------------
    print("\n0) Data file checks (informative)")
    checks = [
        ("RAW_GW_FILE", RAW_GW_FILE),
        ("RAW_ODDS_FILE", RAW_ODDS_FILE),
        ("PLAYER_GW_FILE (processed)", PLAYER_GW_FILE),
        ("EPL_FIXTURES_FILE (processed)", EPL_FIXTURES_FILE),
        ("ODDS_FILE (processed)", ODDS_FILE),
    ]
    for label, p in checks:
        pth = Path(p)
        status = "OK" if pth.exists() else "MISSING"
        print(f"   - {label}: {p}  -> {status}")

    # ------------------------------------------------------------------
    # 1) Export GW results (metrics + figures + tables)
    # ------------------------------------------------------------------
    print("\n1) Exporting GW results (metrics, figures, tables)")
    if args.skip_export:
        print(_status_line("export_gw_results", "SKIPPED", "--skip-export"))
    else:
        try:
            df_metrics = export_gw_results(
                test_season=args.season,
                output_dir=output_dir,
            )
            shape = getattr(df_metrics, "shape", None)
            print(_status_line("export_gw_results", "DONE", f"metrics shape={shape}"))
            print(
                _status_line(
                    "Outputs",
                    "INFO",
                    f"{output_dir}/metrics, {output_dir}/figures, {output_dir}/tables, {output_dir}/predictions",
                )
            )
        except Exception as e:
            print(_status_line("export_gw_results", "ERROR", f"{type(e).__name__}: {e}"))
            raise

    # ------------------------------------------------------------------
    # 2) Prediction demo (best buy per position) — runs on multiple GWs
    # ------------------------------------------------------------------
    print("\n2) Prediction demo: predict_gw_all_players")

    if args.skip_predict:
        print(_status_line("predict_gw_all_players", "SKIPPED", "--skip-predict"))
    else:
        try:
            import pandas as pd
            import numpy as np

            SEP = "-" * 72  # separator line (same spirit as bookmaker output)
            TABLE_COL_SPACE = {
                "gameweek": 8,
                "position": 8,
                "name": 24,
                "team": 22,
                "predicted_points": 16,
            }

            def _print_table_indented(
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

                # Use provided widths (keep same variable name to avoid touching anything else)
                W = col_space or TABLE_COL_SPACE

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

                # Header (same widths as rows)
                header_parts: list[str] = []
                for c in cols_existing:
                    w = int(W.get(c, max(len(c), 10)))
                    if c == "predicted_points":
                        header_parts.append(c.rjust(w))
                    else:
                        header_parts.append(c.ljust(w))
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

            # demo_gws must be defined right after argparse:
            # demo_gws = [int(args.gw)] if args.gw is not None else [int(x) for x in args.gws]
            for gw in demo_gws:
                print(f"\n{SEP}")
                print(f"   GW {gw}")
                print(f"{SEP}")

                preds = predict_gw_all_players(
                    model="gw_seasonal_gbm",
                    test_season=args.season,
                    gameweek=gw,
                )

                n_preds = 0 if preds is None else len(preds)
                print(_status_line("predictions generated", "DONE", f"n={n_preds}"))
                print(_status_line("Context", "INFO", f"gameweek={gw}"))

                if preds is None or len(preds) == 0:
                    continue

                tmp = preds.copy()
                tmp["predicted_points"] = pd.to_numeric(tmp["predicted_points"], errors="coerce")
                tmp = tmp.dropna(subset=["predicted_points"])

                # Enforce FPL position order everywhere (GK, DEF, MID, FWD)
                POS_ORDER = ["GK", "DEF", "MID", "FWD"]
                if "position" not in tmp.columns:
                    print(_status_line("best-by-position", "WARNING", "'position' column missing"))
                    continue
                tmp["position"] = pd.Categorical(tmp["position"], categories=POS_ORDER, ordered=True)

                # ----------------------------------------------------------
                # Best buy per position (table)
                # ----------------------------------------------------------
                print("\n   Best buy per position:")

                best_by_pos = (
                    tmp.sort_values(["position", "predicted_points"], ascending=[True, False])
                    .groupby("position", as_index=False, observed=True)
                    .head(1)
                    .sort_values("position")
                    .reset_index(drop=True)
                )

                _print_table_indented(
                    best_by_pos,
                    cols=["gameweek", "position", "name", "team", "predicted_points"],
                    indent="   ",
                    float_round=2,
                    col_space=TABLE_COL_SPACE,
                )

                # ------------------------------------------------------
                # Top 3 options per position (table)
                # ------------------------------------------------------
                print("\n   Top 3 options per position:")

                top3_by_pos = (
                    tmp.sort_values(["position", "predicted_points"], ascending=[True, False])
                    .groupby("position", as_index=False, observed=True)
                    .head(3)
                    .sort_values(["position", "predicted_points"], ascending=[True, False])
                    .reset_index(drop=True)
                    )

                _print_table_indented(
                    top3_by_pos,
                    cols=["gameweek", "position", "name", "team", "predicted_points"],
                    indent="   ",
                    float_round=2,
                    col_space=TABLE_COL_SPACE,
                )

        except Exception as e:
            print(_status_line("prediction demo", "ERROR", f"{type(e).__name__}: {e}"))
            raise

    # ------------------------------------------------------------------
    # 3) Sample match team-strength export (optional, non-blocking)
    # ------------------------------------------------------------------
    print("\n3) Exporting sample match team strength")
    if args.skip_sample_matches:
        print(_status_line("export_sample_match_team_strength", "SKIPPED", "--skip-sample-matches"))
    else:
        try:
            sample = export_sample_match_team_strength(
                test_season=args.season,
                output_dir=output_dir,
                top_n_players=11,
                sample_n_matches=10,
                seed=42,
            )
            print(_status_line("sample match team strength exported", "DONE", f"n={len(sample)}"))
            print(_status_line("Output", "INFO", f"{output_dir}/predictions/"))

        except Exception as e:
            # Non-blocking: depends on fixtures/odds availability
            print(_status_line("sample match export", "WARNING", f"{type(e).__name__}: {e}"))


    # ------------------------------------------------------------------
    # 4) Optional bookmaker comparison (explicit opt-in)
    # ------------------------------------------------------------------
    print("\n4) Bookmaker comparison (Bet365) — optional")
    if not args.run_bookmakers:
        print(_status_line("compare_model_vs_bookmakers", "SKIPPED", "use --run-bookmakers to enable"))
    else:
        try:
            from src.evaluation import compare_model_vs_bookmakers

            comp, mae, corr = compare_model_vs_bookmakers(
                model="gw_seasonal_gbm",
                test_season=args.season,
                verbose=False,
                )

            print(_status_line("bookmaker comparison", "DONE"))
            
            # --------------------------------------------------------------
            # Readable statistics summary 
            # --------------------------------------------------------------
            print("-" * 80)
            print("Statistics summary")
            print("-" * 80)
            print(f"MAE  : {mae:.3f}")
            print(f"Corr : {corr:.3f}")
            print(f"n    : {len(comp)}")
            print("-" * 80)

            # --------------------------------------------------------------
            # Worst / Best predictions (tables, direct from comp)
            # --------------------------------------------------------------
            df = comp.copy()
            # --------------------------------------------------------------
            # Pretty column names (LOCAL: display + export only)
            # --------------------------------------------------------------
            pretty_rename = {
                "pnorm_home_win": "p_home_win_b365",
                "p_model_home_win": "p_home_win_model",
            }


            # Safety (should already be numeric, but keep robust)
            df["pnorm_home_win"] = pd.to_numeric(df["pnorm_home_win"], errors="coerce")
            df["p_model_home_win"] = pd.to_numeric(df["p_model_home_win"], errors="coerce")
            df["abs_error"] = pd.to_numeric(df["abs_error"], errors="coerce")

            df = df.dropna(subset=["pnorm_home_win", "p_model_home_win", "abs_error"])

            df["diff"] = df["p_model_home_win"] - df["pnorm_home_win"]

            display_cols = [
                "season",
                "gameweek",
                "home_team",
                "away_team",
                "pnorm_home_win",
                "p_model_home_win",
                "diff",
                "abs_error",
            ]

            df_display = df[display_cols].rename(columns=pretty_rename)

            worst5 = (
                df_display.sort_values("abs_error", ascending=False)
                        .head(5)
                        .round(3)
            )

            best5 = (
                df_display.sort_values("abs_error", ascending=True)
                        .head(5)
                        .round(3)
            )

            print("\nWorst 5 predictions (largest |Bet365 - model|)")
            print("-" * 80)
            print(worst5.to_string(index=False))

            print("\nBest 5 predictions (smallest |Bet365 - model|)")
            print("-" * 80)
            print(best5.to_string(index=False))





            out_path = (
                output_dir
                / "predictions"
                / f"bookmakers_comp__{args.season.replace('/', '_')}.csv"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            comp.to_csv(out_path, index=False)
            print(_status_line("CSV export", "DONE", f"{out_path}"))

        except Exception as e:
            print(_status_line("bookmaker comparison", "WARNING", f"{type(e).__name__}: {e}"))

    print("\n" + _hr())
    print("DONE: main.py completed without blocking errors.")
    print(_hr())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
    except Exception:
        raise

