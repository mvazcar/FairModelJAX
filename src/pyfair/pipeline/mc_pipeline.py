"""MC-model pipeline orchestrator with parquet caching.

Three stages, each parquet-cached so re-runs skip straight to the next:

  step01_mc — load MC data (YAW, YDATA, QUAR + SHRDDD.DAT)  →  wide frame
  step02_mc — estimate every MC equation                   →  coef dict
  step03_mc — run in-sample solve + optional forecast       →  solution path

Matches ``pipeline.run`` for the IS/US model. The ``--model mc`` CLI
entry point dispatches here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

from .. import config
from ..mc import model as mc_model, shr as mc_shr, solve as mc_solve

_LOG = logging.getLogger(__name__)


_OUTPUT_DIR = config.PYFAIR_ROOT / "output"
_STEP01_MC_FRAME = _OUTPUT_DIR / "step01_mc_frame.parquet"
_STEP02_MC_COEFS = _OUTPUT_DIR / "step02_mc_coefs.parquet"
_STEP02_SHR_COEFS = _OUTPUT_DIR / "step02_shr_coefs.parquet"
_STEP03_MC_SOLUTION = _OUTPUT_DIR / "step03_mc_solution.parquet"


@dataclass
class MCPipelineResult:
    """Container for MC pipeline outputs.

    Attributes:
      data: Wide frame from step01 (country + SHR data joined).
      country_coefs: {country_prefix: [EstimationResultMC]} from step02.
      shr_coefs: {eq_number: (c, β_lag, β_p)} from SHR estimator.
      solution: Per-country solved paths from step03. Keys are country
        prefixes; values are lists of ``{period, solved, ...}`` dicts.
    """
    data: pl.DataFrame
    country_coefs: dict[str, list]
    shr_coefs: dict[int, tuple[float, float, float]]
    solution: dict[str, list]


def step01_mc_load(
    countries: tuple[str, ...] | None = None,
    force: bool = False,
) -> pl.DataFrame:
    """Load MC data frame; cache to parquet for fast reuse.

    Args:
      countries: Country prefixes to materialize GENRs for. Defaults to
        every ROW country except the Euro-zone composite.
      force: Rebuild even if cache exists.

    Returns:
      Wide Polars frame (periods × all columns needed for MC solve).
    """
    if _STEP01_MC_FRAME.exists() and not force:
        return pl.read_parquet(_STEP01_MC_FRAME)

    if countries is None:
        from ..mc import countries as mcc
        countries = tuple(
            c.prefix for c in mcc.row_countries() if c.prefix != "EU"
        )

    # Build the full country frame and join SHR bilateral data.
    base = mc_model.build_frame_mc(countries=countries)
    shr_frame = mc_shr.build_shr_frame()
    combined = base.join(shr_frame, on="period", how="full",
                          coalesce=True).sort("period")

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(_STEP01_MC_FRAME)
    return combined


def step02_mc_estimate(
    frame: pl.DataFrame | None = None,
    countries: tuple[str, ...] | None = None,
    force: bool = False,
) -> tuple[dict[str, list], dict[int, tuple[float, float, float]]]:
    """Estimate every MC country equation + SHR equation.

    Country coefs are re-computed every run (fast, ~seconds). SHR coefs
    are cached separately because their estimation costs ~155s.

    Returns:
      ``(country_results_by_prefix, shr_coefs)``.
    """
    if frame is None:
        frame = step01_mc_load(countries)
    if countries is None:
        from ..mc import countries as mcc
        countries = tuple(
            c.prefix for c in mcc.row_countries() if c.prefix != "EU"
        )

    # Country estimation (fast).
    country_results: dict[str, list] = {}
    for prefix in countries:
        try:
            country_results[prefix] = mc_model.estimate_country(prefix)
        except Exception:
            _LOG.exception("[step02_mc] %s estimation failed", prefix)
            country_results[prefix] = []

    # SHR coefficients (cached via mc_shr.load_shr_coefs_cached).
    shr_coefs = mc_shr.load_shr_coefs_cached(
        path=_STEP02_SHR_COEFS, refresh=force,
    )
    return country_results, shr_coefs


def step03_mc_solve(
    frame: pl.DataFrame | None = None,
    country_coefs: dict | None = None,
    shr_coefs: dict | None = None,
    start_period: str = "2005Q1",
    end_period: str = "2015Q4",
    force: bool = False,
) -> dict[str, list]:
    """Multi-country in-sample solve over [start, end].

    Returns:
      ``{country_prefix: [{period, solved, iterations, residual_norm}, ...]}``.
    """
    if frame is None:
        frame = step01_mc_load()
    if country_coefs is None or shr_coefs is None:
        country_coefs, shr_coefs = step02_mc_estimate(frame)

    # Per-country in-sample simulate. (Full Gauss-Seidel is available via
    # simulate_mc_endogenous but expensive; this default is the faster
    # independent per-country path.)
    solution: dict[str, list] = {}
    for prefix in country_coefs:
        if not country_coefs[prefix]:
            continue
        try:
            solution[prefix] = mc_solve.simulate_country_path(
                prefix, frame, start_period, end_period,
            )
        except Exception:
            _LOG.exception("[step03_mc] %s simulation failed", prefix)
            solution[prefix] = []
    return solution


def run_mc_pipeline(
    countries: tuple[str, ...] | None = None,
    start_period: str = "2005Q1",
    end_period: str = "2015Q4",
    force: bool = False,
) -> MCPipelineResult:
    """End-to-end MC pipeline with parquet caching.

    Args:
      countries: Subset of MC country prefixes. Defaults to every ROW
        country except EU.
      start_period, end_period: In-sample solve window for step03.
      force: Force re-run every step (ignore caches).

    Returns:
      ``MCPipelineResult`` containing data frame, country coefs, SHR
      coefs, and the solved per-country path.
    """
    print("[mc_pipeline] step01: load")
    frame = step01_mc_load(countries=countries, force=force)
    print(f"[mc_pipeline]   frame shape: {frame.shape}")

    print("[mc_pipeline] step02: estimate")
    country_coefs, shr_coefs = step02_mc_estimate(
        frame=frame, countries=countries, force=force,
    )
    n_country_eqs = sum(len(v) for v in country_coefs.values())
    print(f"[mc_pipeline]   {n_country_eqs} country equations + "
          f"{len(shr_coefs)} SHR coefs")

    print(f"[mc_pipeline] step03: solve {start_period}–{end_period}")
    solution = step03_mc_solve(
        frame=frame, country_coefs=country_coefs, shr_coefs=shr_coefs,
        start_period=start_period, end_period=end_period, force=force,
    )
    n_periods = (max(len(v) for v in solution.values())
                 if solution else 0)
    print(f"[mc_pipeline]   solved {len(solution)} countries × {n_periods} periods")

    return MCPipelineResult(
        data=frame, country_coefs=country_coefs,
        shr_coefs=shr_coefs, solution=solution,
    )
