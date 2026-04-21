"""CLI entry point.

Supports::

    python -m pyfair [--model ...] [--data-source ...] [--resume N] [--force]
    python run.py    (thin wrapper that invokes this module)

See ``pipeline.run`` for the full semantics of each flag.
"""
from __future__ import annotations

import argparse

from .pipeline import is_pipeline as pipeline


_EPILOG = """\
Examples:
  # Default run: IS model on freshly-fetched FRED data
  python run.py

  # Reproduce Fair's 2013 published coefficients (hard parity check)
  python run.py --data-source fair_2013

  # Re-run only step 3 onward, ignoring step 3/4 caches
  python run.py --resume 3 --force
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyfair",
        description="JAX port of Ray C. Fair's US macroeconomic model.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["is", "us", "mc"],
        default="is",
        help="'is' = 2-equation tutorial model. "
             "'us' = standalone US model. "
             "'mc' = Multi-Country model (36 countries, estimate + solve).",
    )
    parser.add_argument(
        "--mc-country",
        default=None,
        help="For --model mc: single country to estimate (default: all).",
    )
    parser.add_argument(
        "--data-source",
        choices=["fred", "fair_2013"],
        default="fred",
        help="'fred' (default): live FRED CSV, rebuilt by "
             "build_is_dat_from_fred.py. "
             "'fair_2013': shipped 06_examples/IS.DAT, the frozen 2013 "
             "snapshot; use this to reproduce Fair's published coefficients.",
    )
    parser.add_argument(
        "--resume",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Start running from step N (earlier steps load from parquet cache).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore caches for the active steps and re-run them.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.model == "mc":
        _run_mc(country=args.mc_country)
        return
    pipeline.run(
        model=args.model,
        resume=args.resume,
        force=args.force,
        data_source=args.data_source,
    )


def _run_mc(country: str | None = None) -> None:
    """Run the MC pipeline (load + estimate + solve) via ``pipeline_mc``.

    If ``country`` is specified, only that country is included.
    """
    from .pipeline import mc_pipeline as pipeline_mc
    if country:
        countries = (country,)
    else:
        countries = None
    pipeline_mc.run_mc_pipeline(
        countries=countries, start_period="2010Q1", end_period="2015Q4",
    )


if __name__ == "__main__":
    main()
