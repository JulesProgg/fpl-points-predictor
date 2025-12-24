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
        default=[12],
        help="Gameweek used for the prediction demo (default: 27)",
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
    demo_gws = [21] # CHANGE HERE IF YOU WANT TO HAVE ANOTHER DEMO GAMEWEEK


    root = _set_project_root()
    output_dir = Path(args.output_dir)

    if not args.no_header:
        print(_hr())
        print("FPL Points Predictor — main.py")
        print(f"Project root : {root}")
        print(f"Test season  : {args.season}")
        print(f"Demo GW     : {demo_gws}")
        print(f"Output dir   : {output_dir.resolve()}")
        print(_hr())

    # ------------------------------------------------------------------
    # High-level imports (stable public API)
    # ------------------------------------------------------------------
    from src import predict_gw_all_players

    from src.reporting import (
        export_gw_results,
        export_sample_match_team_strength,
        enrich_sample_match_summary,
        console_print_prediction_demo,
        summarize_bookmaker_comparison,
        console_print_bookmaker_stats
    )

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
    print("\n1) Exporting GW results (figures, metrics, predictions, tables)")
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
                    f"{output_dir}/figures, {output_dir}/metrics, {output_dir}/predictions, {output_dir}/tables",
                )
            )
        except Exception as e:
            print(_status_line("export_gw_results", "ERROR", f"{type(e).__name__}: {e}"))
            raise

    # ------------------------------------------------------------------
    # 2) Prediction demo on a choosen gw
    # ------------------------------------------------------------------
    print("\n2) gameweek predictions demo: predict_gw_all_players")

    if args.skip_predict:
        print(_status_line("predict_gw_all_players", "SKIPPED", "--skip-predict"))
    else:
        try:
            SEP = "-" * 72

            for gw in demo_gws:
                print(f"\n{SEP}")
                print(f"   GW {gw}")
                print(f"{SEP}")

                from src.evaluation import get_gw_predictions_from_evaluation

                preds = get_gw_predictions_from_evaluation(
                    test_season=args.season,
                    gameweek=gw,
                )


                n_preds = 0 if preds is None else len(preds)
                print(_status_line("predictions generated", "DONE", f"n={n_preds}"))
                print(_status_line("Context", "INFO", f"gameweek={gw}"))

                if preds is None or len(preds) == 0:
                    continue

                # All presentation logic lives in src.reporting now
                console_print_prediction_demo(preds, one_line=args.one_line, indent="   ")

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

            sample = enrich_sample_match_summary(sample)

            print(_status_line("sample match team strength exported", "DONE", f"n={len(sample)}"))
            print(_status_line("CSV output", "INFO", f"{output_dir}/predictions/"))

            preview_cols = ["home_team", "away_team", "home_strength", "away_strength", "delta_strength", "favorite"]
            preview_cols = [c for c in preview_cols if c in sample.columns]
            print("\n   Match summary preview:")
            print(sample[preview_cols].to_string(index=False))

            out_path = Path(output_dir) / "predictions" / "sample_match_team_strength_summary.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sample.to_csv(out_path, index=False)
            print(_status_line("Summary CSV", "DONE", str(out_path)))

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

            console_print_bookmaker_stats(mae=mae, corr=corr, n=len(comp))
            
            worst5, best5 = summarize_bookmaker_comparison(comp)

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

