"""MC Trade-Share equations — Fair's SHR.INP stochastic bilateral shares.

Fair's MC model closes through two routes: the per-country demand blocks
(handled in ``mc_model`` / ``mc_solve``) and a bilateral trade-share
matrix ``A_ij`` that routes each country's imports across partners. The
shares themselves are **endogenous** — SHR.INP declares one stochastic
equation per (source, destination) pair::

    EQ N  LAA<SRC><DST>  C  LA<SRC><DST>(-1)  P<SRC><DST>

with ``LAA = log(AA + 0.00001)``, ``LA = log(A + 0.00001)``, and
``P = <SRC>PX$ / <DST>PMM`` (relative export price). Every equation is
plain OLS with a single lag plus the price ratio.

The 2018 vintage has **1,686 active equations** (of ~2,100 total; the
rest are commented out — discontinued corridors, former USSR countries,
etc.). Samples vary per corridor but most use 1966Q1–2016Q4.

This module provides:

* ``parse_shr_inp(path)`` — extract equation specs from SHR.INP.
* ``load_shrddd_full(path)`` — load every series from SHRDDD.DAT (AA,
  A, PMM aggregates) into a wide frame.
* ``build_shr_frame()`` — join SHRDDD data with country export prices
  from YDATA.DAT and compute every ``P<SRC><DST>`` relative-price GENR.
* ``EQUATIONS_SHR`` — list of ``MCEquation`` instances (one per active
  equation).
* ``estimate_all_shr(frame)`` — run OLS on every equation and compare
  against Fair's OUT.

For the MC solve driver, these equations let the trade-share matrix
evolve endogenously (needed for out-of-sample forecasts where
``SHRDDD.DAT`` doesn't cover the forecast horizon).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import jax.numpy as jnp
import polars as pl

from .. import config
from ..core import readers
from ..core.estimate import two_sls_with_se
from .model import MCEquation, REFERENCE_PARAMS_MC, parse_mc_out

_LOG = logging.getLogger(__name__)


# Fair's ``LOG(AA + .00001)`` / ``LOG(A + .00001)`` regularizer — the
# constant added before the log so bilateral shares at zero don't blow
# the log up. Matches every GENR in SHR.INP.
_LOG_EPSILON = 1e-5

# Classification tolerances for ``estimate_all_shr``:
_STRICT_TOL = 5e-5
_CLOSE_TOL = 5e-3
_LOOSE_TOL = 1.0


# EQ line pattern — matches a single active EQ directive on one line,
# with regressors terminated by ``;`` or end-of-line. The ``(?<!@)``
# prevents matching commented ``@EQ`` and ``@@EQ`` alternatives.
_EQ_LINE_RE = re.compile(
    r"(?<!@)^\s+EQ\s+(\d+)\s+(\S+)\s+([^;\n]+?);",
    re.MULTILINE,
)

# Active (uncommented) SMPL directive on its own line. Skips any
# ``@ SMPL ...`` commented-out variants between the EQ and the real SMPL.
_ACTIVE_SMPL_RE = re.compile(
    r"^\s+SMPL\s+(\d+\.\d+)\s+(\d+\.\d+)\s*;",
    re.MULTILINE,
)


def _period_from_fair(fair_period: str) -> str:
    """``"1966.1"`` → ``"1966Q1"``."""
    year, q = fair_period.split(".")
    return f"{int(year)}Q{int(q)}"


def parse_shr_inp(path: Path | str | None = None) -> list[dict]:
    """Extract every active SHR equation from SHR.INP.

    Returns a list of dicts with keys:
      * ``number``: integer equation number
      * ``dependent``: dep var name (``"LAA<SRC><DST>"``)
      * ``regressors``: list of 3 tokens (``["C", "LA<SRC><DST>(-1)", "P<SRC><DST>"]``)
      * ``source``, ``dest``: 2-letter country codes parsed from the dep var
      * ``sample_start``, ``sample_end``: Fair-format periods (``"1966Q1"``)

    Some blocks have a commented ``@ SMPL ...`` alternative between the
    EQ and the active SMPL; the parser walks forward until it finds the
    first non-commented SMPL to pick the right estimation window.
    """
    p = Path(path) if path is not None else (config.MC_MODEL_DIR / "SHR.INP")
    text = p.read_text()
    specs = []
    for eq_match in _EQ_LINE_RE.finditer(text):
        number = int(eq_match.group(1))
        dep = eq_match.group(2)
        regressors_raw = eq_match.group(3).split()
        if not dep.startswith("LAA") or len(dep) < 7:
            continue
        src, dst = dep[3:5], dep[5:7]
        # Find the next active SMPL after this EQ.
        smpl_match = _ACTIVE_SMPL_RE.search(text, eq_match.end())
        if not smpl_match:
            continue
        specs.append({
            "number": number,
            "dependent": dep,
            "regressors": regressors_raw,
            "source": src,
            "dest": dst,
            "sample_start": _period_from_fair(smpl_match.group(1)),
            "sample_end": _period_from_fair(smpl_match.group(2)),
        })
    return specs


def load_shrddd_full(path: Path | str | None = None,
                    sample_start: str = "1960.1") -> pl.DataFrame:
    """Load every series from SHRDDD.DAT into a wide frame.

    SHRDDD.DAT contains 6,787 ``LOAD`` blocks — ``AA_ij`` bilateral
    shares, ``A_ij`` lag variants, ``xxPMM`` aggregates, plus auxiliary
    series. Caller's responsible for downstream filtering.
    """
    p = Path(path) if path is not None else (config.MC_MODEL_DIR / "SHRDDD.DAT")
    text = f"SMPL {sample_start} 2099.4;\n" + p.read_text()
    tmp = p.parent / ".shrddd_full.dat"
    tmp.write_text(text)
    try:
        long = readers.parse_fair_data(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    return readers.pivot_to_wide(long).sort("period")


def build_shr_frame(
    shrddd_frame: pl.DataFrame | None = None,
    country_frame: pl.DataFrame | None = None,
    specs: list[dict] | None = None,
) -> pl.DataFrame:
    """Join SHRDDD data with country export prices and add SHR GENRs.

    For every parsed equation the caller needs three GENRs:

      * ``P<SRC><DST>`` = ``<SRC>PX$ / <DST>PMM``  (relative export price)
      * ``LAA<SRC><DST>`` = ``log(AA<SRC><DST> + 0.00001)``
      * ``LA<SRC><DST>`` = ``log(A<SRC><DST> + 0.00001)``

    Args:
      shrddd_frame: Output of :func:`load_shrddd_full`. Loaded if None.
      country_frame: Frame with country export prices (``xxPX$``) from
        YDATA.DAT. Loaded via :func:`mc_model._load_data_files` if None.
      specs: List of parsed equation dicts (from :func:`parse_shr_inp`).
        If None, parses all SHR.INP equations.

    Returns:
      Wide frame containing all raw SHRDDD series, country export
      prices, and the ``P/LAA/LA`` GENRs for every equation.
    """
    from . import model as mc_model
    if shrddd_frame is None:
        shrddd_frame = load_shrddd_full()
    if country_frame is None:
        country_frame = mc_model._load_data_files((
            config.MC_YAW, config.MC_YDATA, config.MC_QUAR,
        ))
    if specs is None:
        specs = parse_shr_inp()

    # Join country frame (which has xxPX$, xxPMP, etc.) with SHRDDD (AA_ij, xxPMM).
    # Column overlap (xxPMM appears in both if include_pmm was used) → outer-coalesce.
    frame = country_frame.join(shrddd_frame, on="period", how="full",
                                coalesce=True).sort("period")

    # Build GENR columns for each unique (src, dst).
    exprs: list[pl.Expr] = []
    for spec in specs:
        src, dst = spec["source"], spec["dest"]
        px_col = f"{src}PX$"
        pmm_col = f"{dst}PMM"
        p_name = f"P{src}{dst}"
        aa_name = f"AA{src}{dst}"
        a_name = f"A{src}{dst}"

        # P_ij = src's export price / dst's import-price aggregate.
        if px_col in frame.columns and pmm_col in frame.columns:
            exprs.append((pl.col(px_col) / pl.col(pmm_col)).alias(p_name))
        # LAA = log(AA + 0.00001)
        if aa_name in frame.columns:
            exprs.append((pl.col(aa_name) + _LOG_EPSILON).log().alias(f"LAA{src}{dst}"))
        # LA  = log(A  + 0.00001)
        if a_name in frame.columns:
            exprs.append((pl.col(a_name) + _LOG_EPSILON).log().alias(f"LA{src}{dst}"))

    if exprs:
        # Apply in one go (duplicate aliases across specs are harmless —
        # Polars keeps the last).
        deduped: dict[str, pl.Expr] = {}
        for e in exprs:
            deduped[e.meta.output_name()] = e
        frame = frame.with_columns(list(deduped.values()))
    return frame


# ---------------------------------------------------------------------------
# Equation registry + estimator
# ---------------------------------------------------------------------------

def _specs_to_equations(specs: list[dict]) -> list[MCEquation]:
    """Convert parser output into ``MCEquation`` instances (one per spec)."""
    equations: list[MCEquation] = []
    for spec in specs:
        src, dst = spec["source"], spec["dest"]
        equations.append(MCEquation(
            country="SHR",      # All SHR equations tagged "SHR" (not a real country).
            number=spec["number"],
            dependent=spec["dependent"],
            regressors=tuple(spec["regressors"]),
            instruments=tuple(spec["regressors"]),  # OLS → Z == X.
            has_ar1=False, has_ar2=False,
            sample_start=spec["sample_start"],
            sample_end=spec["sample_end"],
            lhs_transform=f"AA{src}{dst}=EXP(LAA{src}{dst})-.00001",
            notes=f"SHR trade-share equation {src}->{dst}.",
        ))
    return equations


EQUATIONS_SHR: list[MCEquation] = _specs_to_equations(parse_shr_inp())


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

def _parse_token(token: str) -> tuple[str, int]:
    if "(" in token:
        base, rest = token.split("(", 1)
        return base, abs(int(rest.rstrip(")")))
    return token, 0


def estimate_shr_equation(eq: MCEquation, frame: pl.DataFrame) -> dict[str, float]:
    """Plain OLS estimation for one SHR equation.

    Args:
      eq: ``MCEquation`` from ``EQUATIONS_SHR``.
      frame: Output of :func:`build_shr_frame`.

    Returns:
      Dict ``{token: coefficient}`` matching the structure of
      ``EstimationResultMC.coefficients``. ``"RHO(-1)"`` is absent
      (SHR equations are all OLS).
    """
    # Build the lag columns needed for this equation on-the-fly to avoid
    # materializing lag columns for all 1,686 equations up front.
    lag_cols_needed: list[tuple[str, int]] = []
    for tok in eq.regressors:
        base, lag = _parse_token(tok)
        if lag > 0:
            lag_cols_needed.append((base, lag))
    tmp_frame = frame
    for base, lag in lag_cols_needed:
        col_name = f"{base}_lag{lag}"
        if col_name not in tmp_frame.columns and base in tmp_frame.columns:
            tmp_frame = tmp_frame.with_columns(pl.col(base).shift(lag).alias(col_name))

    required = [eq.dependent] + [
        (f"{base}_lag{lag}" if lag > 0 else base)
        for tok in eq.regressors
        for base, lag in [_parse_token(tok)]
        if tok != "C"
    ]
    missing = [c for c in required if c not in tmp_frame.columns]
    if missing:
        raise KeyError(f"EQ {eq.number}: missing columns {missing}")

    est = tmp_frame.filter(
        (pl.col("period") >= pl.lit(eq.sample_start))
        & (pl.col("period") <= pl.lit(eq.sample_end))
    ).drop_nulls(subset=required).sort("period")

    if est.height < 4:
        raise ValueError(f"EQ {eq.number}: insufficient observations ({est.height})")

    y = jnp.asarray(est[eq.dependent].to_numpy(), dtype=jnp.float64)
    cols = []
    for tok in eq.regressors:
        if tok == "C":
            cols.append(jnp.ones(est.height, dtype=jnp.float64))
            continue
        base, lag = _parse_token(tok)
        col_name = f"{base}_lag{lag}" if lag > 0 else base
        cols.append(jnp.asarray(est[col_name].to_numpy(), dtype=jnp.float64))
    X = jnp.column_stack(cols)
    beta, _se = two_sls_with_se(y, X, X)

    return {tok: float(beta[i]) for i, tok in enumerate(eq.regressors)}


def _shr_cache_path() -> Path:
    """Default cache location for estimated SHR coefficients."""
    return config.PYFAIR_ROOT / "output" / "shr_coefs.parquet"


def save_shr_coefs(
    coefs: dict[int, tuple[float, float, float]],
    path: Path | str | None = None,
) -> Path:
    """Persist SHR coefficients as a parquet for fast reuse.

    Schema: one row per equation with columns ``number``, ``c``,
    ``b_lag``, ``b_p``. Caller pairs each number with the (c, β_lag, β_p)
    triple produced by ``estimate_all_shr_coefs``.
    """
    p = Path(path) if path is not None else _shr_cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"number": num, "c": c, "b_lag": b_lag, "b_p": b_p}
        for num, (c, b_lag, b_p) in sorted(coefs.items())
    ]
    pl.DataFrame(rows).write_parquet(p)
    return p


def load_shr_coefs_cached(
    path: Path | str | None = None,
    frame: pl.DataFrame | None = None,
    refresh: bool = False,
) -> dict[int, tuple[float, float, float]]:
    """Return cached SHR coefficients, re-estimating on cache miss.

    The estimator takes ~155s across all 1,686 equations; this wrapper
    runs it once and writes the result to parquet. Subsequent calls
    load the parquet in milliseconds. Pass ``refresh=True`` to force
    re-estimation (e.g. after a data update).

    Args:
      path: Parquet cache location (default ``output/shr_coefs.parquet``).
      frame: Override frame for estimation (for tests). Ignored on
        cache hit.
      refresh: When True, re-estimate even if the cache exists.

    Returns:
      ``{eq_number: (c, β_lag, β_p)}`` for every estimable equation.
    """
    p = Path(path) if path is not None else _shr_cache_path()
    if p.exists() and not refresh:
        df = pl.read_parquet(p)
        return {
            int(row["number"]): (
                float(row["c"]), float(row["b_lag"]), float(row["b_p"])
            )
            for row in df.iter_rows(named=True)
        }
    coefs = estimate_all_shr_coefs(frame)
    save_shr_coefs(coefs, p)
    return coefs


def estimate_all_shr_coefs(
    frame: pl.DataFrame | None = None,
) -> dict[int, tuple[float, float, float]]:
    """Return ``{eq_number: (c, β_lag, β_p)}`` for every SHR equation.

    Convenience for the forecast path — callers cache this once and
    reuse it when projecting AA_ij forward across periods. Entries
    where estimation fails (insufficient data) are omitted.
    """
    if frame is None:
        frame = build_shr_frame()
    coefs: dict[int, tuple[float, float, float]] = {}
    for eq in EQUATIONS_SHR:
        try:
            result = estimate_shr_equation(eq, frame)
            coefs[eq.number] = (
                result["C"],
                result[eq.regressors[1]],  # LA<src><dst>(-1)
                result[eq.regressors[2]],  # P<src><dst>
            )
        except Exception:
            _LOG.debug("SHR coef estimation failed for EQ %d", eq.number,
                        exc_info=True)
            continue
    return coefs


def project_shr_one_period(
    frame: pl.DataFrame,
    period: str,
    coefs: dict[int, tuple[float, float, float]],
    specs: list[dict] | None = None,
) -> pl.DataFrame:
    """Apply every SHR equation at ``period`` to populate AA_ij and LAA_ij.

    For each (src, dst) pair with a cached coefficient vector::

        LAA_ij[t] = c + β_lag · LA_ij[t-1] + β_p · P_ij[t]
        AA_ij[t]  = exp(LAA_ij[t]) − 0.00001

    Writes results back to the frame at ``period``. Requires:

      * ``LA<src><dst>_lag1`` or ``A<src><dst>(t-1)`` at period
      * ``P<src><dst>`` at period (derived from ``<src>PX$``/``<dst>PMM``)

    Args:
      frame: Frame from ``build_shr_frame`` (with lag columns for LA_ij).
      period: Quarter to project.
      coefs: From ``estimate_all_shr_coefs``. Missing equation numbers
        are silently skipped.
      specs: Cached list from ``parse_shr_inp()``. Parses if None.

    Returns:
      Frame with AA_ij and LAA_ij populated at ``period``.
    """
    if specs is None:
        specs = parse_shr_inp()

    # Need ``LA<src><dst>_lag1`` columns on the frame.
    needed_lags = [f"LA{s['source']}{s['dest']}" for s in specs]
    frame = frame.with_columns([
        pl.col(c).shift(1).alias(f"{c}_lag1")
        for c in needed_lags if c in frame.columns
        and f"{c}_lag1" not in frame.columns
    ])

    # Pull the row at period, compute new LAA + AA for every spec.
    row = frame.filter(pl.col("period") == pl.lit(period))
    if row.height != 1:
        raise ValueError(f"Period {period} not in frame")
    row_dict = row.row(0, named=True)

    updates_laa: dict[str, float] = {}
    updates_aa: dict[str, float] = {}
    for spec in specs:
        num = spec["number"]
        if num not in coefs:
            continue
        src, dst = spec["source"], spec["dest"]
        la_lag_key = f"LA{src}{dst}_lag1"
        p_key = f"P{src}{dst}"
        la_lag = row_dict.get(la_lag_key)
        p_val = row_dict.get(p_key)
        if la_lag is None or p_val is None:
            continue
        c, b_lag, b_p = coefs[num]
        laa_new = c + b_lag * float(la_lag) + b_p * float(p_val)
        aa_new = jnp.exp(jnp.asarray(laa_new)) - _LOG_EPSILON
        updates_laa[f"LAA{src}{dst}"] = laa_new
        updates_aa[f"AA{src}{dst}"] = float(aa_new)

    if not updates_laa:
        return frame

    # Apply updates at ``period`` row.
    exprs = []
    for col, val in {**updates_laa, **updates_aa}.items():
        if col not in frame.columns:
            continue
        exprs.append(
            pl.when(pl.col("period") == pl.lit(period))
              .then(pl.lit(val))
              .otherwise(pl.col(col))
              .alias(col)
        )
    if exprs:
        frame = frame.with_columns(exprs)
    return frame


def aggregate_pmm_one_period(
    frame: pl.DataFrame,
    period: str,
    dest_countries: tuple[str, ...],
    specs: list[dict] | None = None,
) -> pl.DataFrame:
    """Aggregate bilateral ``A_ij`` and export prices into ``xxPMM``.

    Fair's MC.INP formula (line ~5 of MCSHR1.INP)::

        xxPMM = Σ_j A_ji(-1) · yyPX$

    where the sum runs over all source countries ``j`` that export to
    ``xx``. Uses ``A_ij(-1)`` (lagged share) times ``yyPX$`` (current
    export price) — matches Fair's block ordering of per-country solve
    then trade-aggregate update.

    Args:
      frame: Frame with A_ij and xxPX$ columns.
      period: Quarter to aggregate at.
      dest_countries: Destinations to compute PMM for (e.g. ``("US", "CA")``).
      specs: SHR equation specs (used to enumerate the (src, dst)
        pairs). Parses if None.

    Returns:
      Frame with ``xxPMM`` populated at ``period`` for each destination.
    """
    # Ignore ``specs`` for source enumeration — use the frame directly so
    # the xxPMM aggregation picks up EVERY source (including the ~23
    # "exogenous" shares per destination that SHR.INP doesn't estimate
    # with stochastic equations). The AA_ij shares sum to 1.0 across
    # every source (estimable + exogenous), so skipping the exogenous
    # ones drops roughly 30% of the contribution.

    # Enumerate all 5-char A_ij columns (primary share variables).
    sources_by_dest: dict[str, list[str]] = {}
    for col in frame.columns:
        if len(col) != 5 or not col.startswith("A"):
            continue
        src, dst = col[1:3], col[3:5]
        if dst in dest_countries:
            sources_by_dest.setdefault(dst, []).append(src)

    # Materialize lag columns.
    lag_exprs = [
        pl.col(f"A{src}{dst}").shift(1).alias(f"A{src}{dst}_lag1")
        for dst in dest_countries
        for src in sources_by_dest.get(dst, [])
        if f"A{src}{dst}_lag1" not in frame.columns
    ]
    if lag_exprs:
        frame = frame.with_columns(lag_exprs)

    row = frame.filter(pl.col("period") == pl.lit(period)).row(0, named=True)
    new_pmm: dict[str, float] = {}
    for dst in dest_countries:
        total = 0.0
        for src in sources_by_dest.get(dst, []):
            a_val = row.get(f"A{src}{dst}_lag1")
            px_val = row.get(f"{src}PX$")
            if a_val is None or px_val is None:
                continue
            total += float(a_val) * float(px_val)
        if total > 0:
            new_pmm[f"{dst}PMM"] = total

    exprs = [
        pl.when(pl.col("period") == pl.lit(period))
          .then(pl.lit(val))
          .otherwise(pl.col(col))
          .alias(col)
        for col, val in new_pmm.items()
        if col in frame.columns
    ]
    if exprs:
        frame = frame.with_columns(exprs)
    return frame


def estimate_all_shr(frame: pl.DataFrame | None = None) -> list[dict]:
    """Estimate every SHR equation and compare to Fair's OUT.

    Returns a list of per-equation dicts with ``{number, dep, coefs,
    reference, max_err, n_obs, status}`` where ``status`` is one of
    ``"STRICT"`` (<5e-5), ``"CLOSE"`` (<5e-3), ``"LOOSE"`` (<1.0), or
    ``"FAILED"``.
    """
    from . import model as mc_model
    # Make sure OUT has been parsed for these equation numbers.
    if not any(eq.number in REFERENCE_PARAMS_MC for eq in EQUATIONS_SHR[:5]):
        parse_mc_out()  # idempotent

    if frame is None:
        frame = build_shr_frame()

    results: list[dict] = []
    for eq in EQUATIONS_SHR:
        try:
            coefs = estimate_shr_equation(eq, frame)
            fair = {
                tok: REFERENCE_PARAMS_MC.get(eq.number, {}).get(
                    f"{base}({-lag:+d})" if lag > 0
                    else (f"{base}(0)" if base != "C" else "C(0)")
                )
                for tok in eq.regressors
                for base, lag in [_parse_token(tok)]
            }
            errs = [abs(coefs[t] - fair[t])
                    for t in coefs if fair.get(t) is not None]
            max_err = max(errs) if errs else float("nan")
            if max_err < _STRICT_TOL:
                status = "STRICT"
            elif max_err < _CLOSE_TOL:
                status = "CLOSE"
            elif max_err < _LOOSE_TOL:
                status = "LOOSE"
            else:
                status = "FAILED"
            results.append({
                "number": eq.number, "dep": eq.dependent,
                "coefs": coefs, "reference": fair,
                "max_err": max_err, "status": status,
            })
        except Exception as exc:
            _LOG.debug("SHR EQ %d estimation failed", eq.number,
                        exc_info=True)
            results.append({
                "number": eq.number, "dep": eq.dependent,
                "status": "FAILED", "error": str(exc),
            })
    return results
