"""US Model EQ 1 — consumption of services (LCSZ).

Standalone proof-of-concept: estimates Fair's first structural equation from
``fminput.txt`` end-to-end, so we can validate the GENR / regime-dummy / big
instrument-list plumbing before scaling to the full US model.

Fair form (``03_us_model/fminput.txt`` line 155)::

    EQ 1 LCSZ  CNST2CS C AG1 AG2 AG3 LCSZ(-1) LYDZ RSA LAAZ(-1) RHO=1;
    LHS CS = EXP(LCSZ) * POP;

Plus 8 COVID-quarter dummies added via ``MODEQ 1 D20201 D20202 ...`` at line
536. Total 17 structural coefficients + 1 autoregressive parameter.

Estimation sample: 1954 Q1 — 2025 Q4 (288 obs).

Reference coefficients (from ``03_us_model/fmout.txt`` lines 2494–2512, Fair's
February 20 2026 vintage) are stored in :data:`REFERENCE_PARAMS`.

Because CS appears in identities that involve other endogenous variables
(X = CS + CN + CD + ... in the full model), we can't *simulate* this equation
in isolation — only estimate it. Full simulation waits for a critical mass of
equations; for now this module proves the plumbing.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import polars as pl

from .. import config
from ..core import readers
from ..core.estimate import two_sls_ar1

_LOG = logging.getLogger(__name__)


PRESAMPLE_PERIOD = "1953Q4"


# ---------------------------------------------------------------------------
# Specification (from fminput.txt)
# ---------------------------------------------------------------------------

SAMPLE_START = "1954Q1"
SAMPLE_END = "2025Q4"

# Constructed dummies for the CS equation. CNST2CS is a smooth regime ramp:
# 0 during D1 (1952Q1–1973Q4), linear rise from ~0 to 1 across D2 (1974Q1–1994Q4),
# and 1 during D3 (1995Q1 onward). See fminput.txt lines 482–496.
_REGIME_D1_END = "1973Q4"
_REGIME_D2_START, _REGIME_D2_END = "1974Q1", "1994Q4"
_REGIME_D3_START = "1995Q1"
_T1, _T2 = 88, 172   # Fair's quarterly-index anchors for the D2 ramp


# Structural regressors in Fair's order (line 155 + MODEQ line 536).
STRUCTURAL_REGRESSORS: list[str] = [
    "CNST2CS",
    "C",            # plain constant
    "AG1", "AG2", "AG3",
    "LCSZ_lag1",
    "LYDZ",
    "RSA",
    "LAAZ_lag1",
    "D20201", "D20202", "D20203", "D20204",
    "D20211", "D20212", "D20213", "D20214",
]

# First-stage regressors (instruments) from fminput.txt line 387 + MODEQ FSR line 537.
INSTRUMENTS: list[str] = [
    "CNST2CS",
    "C",
    "AG1", "AG2", "AG3",
    "LCSZ_lag1",
    "LAAZ_lag3",
    "RSA_lag1",
    "CNST2CS_lag1",
    "AG1_lag1", "AG2_lag1", "AG3_lag1",
    "LCSZ_lag2",
    "LCOGSZ_lag1", "LTRGSZ_lag1", "LEXZ_lag1",
    "LPOP", "LPOP_lag1",
    "D20201", "D20202", "D20203", "D20204",
    "D20211", "D20212", "D20213", "D20214",
    "D20214_lag1",
]


# Reference coefficients from fmout.txt — what Fair's FORTRAN program outputs
# estimating this exact equation on the exact same data.
REFERENCE_PARAMS: dict[str, float] = {
    "CNST2CS":   0.072237533,
    "C":        -0.165386129,
    "AG1":      -0.094778446,
    "AG2":      -0.212258166,
    "AG3":      -0.070319382,
    "LCSZ_lag1": 0.768748919,
    "LYDZ":      0.154041295,
    "RSA":      -0.001041322,
    "LAAZ_lag1": 0.038504368,
    "D20201":   -0.030509379,
    "D20202":   -0.150163414,
    "D20203":   0.023827985,
    "D20204":  -0.015888461,
    "D20211":  -0.036394166,
    "D20212":  -0.001908018,
    "D20213":  -0.002037284,
    "D20214":  -0.008186015,
    "RHO":      0.214173554,
}


# ---------------------------------------------------------------------------
# GENR preprocessing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenrSpec:
    """Specification for one derived variable built from raw fmdata columns.

    Attributes:
      name: Output column name (e.g. ``"LCSZ"``).
      formula: One-arg callable ``(polars DataFrame) -> polars.Expr`` that
        returns the expression to compute. Using a callable lets us use any
        polars operation without string parsing.
    """
    name: str
    formula: Callable[[pl.DataFrame], pl.Expr]


# Fair's GENR lines from fminput.txt §"GENR ..." block.
DERIVED_VARIABLES: list[GenrSpec] = [
    GenrSpec("CSZ",    lambda df: pl.col("CS") / pl.col("POP")),
    GenrSpec("LCSZ",   lambda df: (pl.col("CS") / pl.col("POP")).log()),
    GenrSpec("LYDZ",   lambda df: (pl.col("YD") / (pl.col("POP") * pl.col("PH"))).log()),
    GenrSpec("LAAZ",   lambda df: (pl.col("AA") / pl.col("POP")).log()),
    GenrSpec("LPOP",   lambda df: pl.col("POP").log()),
    GenrSpec("LCOGSZ", lambda df: ((pl.col("COG") + pl.col("COS")) / pl.col("POP")).log()),
    GenrSpec("LTRGSZ", lambda df: ((pl.col("TRGH") + pl.col("TRSH"))
                                   / (pl.col("POP") * pl.col("PH"))).log()),
    GenrSpec("LEXZ",   lambda df: (pl.col("EX") / pl.col("POP")).log()),
]


def apply_genr(df: pl.DataFrame, specs: list[GenrSpec]) -> pl.DataFrame:
    """Add each derived variable as a new column."""
    return df.with_columns([spec.formula(df).alias(spec.name) for spec in specs])


def add_time_trend(df: pl.DataFrame) -> pl.DataFrame:
    """Attach ``T`` (1-indexed quarter counter from 1952Q1).

    fmdata.txt ships a ``T`` column already, but Fair's own value starts at 0.
    Fair's ``CAPITAL ... BENCHPER=1952.1 BENCHVAL=1`` sets T=1 at 1952Q1, so
    we override with that convention for consistency with fminput.txt math.
    """
    return df.with_columns((pl.int_range(1, df.height + 1)).alias("T"))


def add_cnst2cs(df: pl.DataFrame) -> pl.DataFrame:
    """Attach ``CNST2CS`` — Fair's ramp dummy for the post-1994 CS regime.

    Formula (fminput.txt line 496)::

        CNST2CS = D3 + T1 * D2 / (T1 - T2) - T * D2 / (T1 - T2)

    with D1=1 for 1952Q1–1973Q4, D2=1 for 1974Q1–1994Q4, D3=1 from 1995Q1 on,
    T1=88, T2=172. Evaluates to 0 before 1974Q1, a smooth ramp over D2, and
    1 from 1995Q1 onward.
    """
    period = pl.col("period")
    d2 = pl.when((period >= pl.lit(_REGIME_D2_START))
                 & (period <= pl.lit(_REGIME_D2_END))).then(1.0).otherwise(0.0)
    d3 = pl.when(period >= pl.lit(_REGIME_D3_START)).then(1.0).otherwise(0.0)
    ramp = d3 + _T1 * d2 / (_T1 - _T2) - pl.col("T") * d2 / (_T1 - _T2)
    return df.with_columns(ramp.alias("CNST2CS"))


def add_constant(df: pl.DataFrame) -> pl.DataFrame:
    """Attach ``C`` = 1 (Fair's name for the plain intercept)."""
    return df.with_columns(pl.lit(1.0).alias("C"))


def add_lags(df: pl.DataFrame, variables: list[str], lag_list: list[int]) -> pl.DataFrame:
    """For each ``variable`` in ``variables`` add columns ``{var}_lag{k}``."""
    lagged = [
        pl.col(var).shift(k).alias(f"{var}_lag{k}")
        for var in variables
        for k in lag_list
    ]
    return df.with_columns(lagged)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

# Variables that need lags somewhere in our regressor/instrument lists.
_LAGGED_VARIABLES = [
    "CNST2CS", "AG1", "AG2", "AG3",
    "LCSZ", "LAAZ", "RSA",
    "LCOGSZ", "LTRGSZ", "LEXZ", "LPOP",
    "D20214",
]
_LAG_ORDERS = [1, 2, 3]


def build_regression_frame(fmdata_path=None) -> pl.DataFrame:
    """Parse ``fmdata.txt`` and assemble every column the CS regression needs.

    Includes one extra quarter (``PRESAMPLE_PERIOD`` = 1953Q4) before
    ``SAMPLE_START`` so the AR(1) lag is available for the first estimation
    observation. Call :func:`split_presample_and_estimation` to slice it.

    Args:
      fmdata_path: Override the default path (``config.US_FMDATA``). Mainly
        useful for tests.

    Returns:
      Wide polars frame containing raw series + derived variables + lags,
      restricted to ``PRESAMPLE_PERIOD`` to ``SAMPLE_END`` with no null rows.
    """
    path = fmdata_path or config.US_FMDATA
    long_frame = readers.parse_fair_data(path)
    wide = readers.pivot_to_wide(long_frame)

    wide = add_time_trend(wide)
    wide = add_constant(wide)
    wide = add_cnst2cs(wide)
    wide = apply_genr(wide, DERIVED_VARIABLES)
    wide = add_lags(wide, _LAGGED_VARIABLES, _LAG_ORDERS)

    in_window = wide.filter(
        (pl.col("period") >= pl.lit(PRESAMPLE_PERIOD))
        & (pl.col("period") <= pl.lit(SAMPLE_END))
    )
    required = set(STRUCTURAL_REGRESSORS) | set(INSTRUMENTS) | {"LCSZ"}
    required &= set(in_window.columns)
    return in_window.drop_nulls(subset=list(required))


def split_presample_and_estimation(
    frame: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Separate the pre-sample lag row from the estimation sample."""
    presample = frame.head(1)
    estimation = frame.tail(frame.height - 1)
    return presample, estimation


def _stack_columns(df: pl.DataFrame, columns: list[str]) -> jnp.ndarray:
    """Pack named polars columns into a ``(T, len(columns))`` jnp array."""
    return jnp.column_stack(
        [jnp.asarray(df[col].to_numpy(), dtype=jnp.float64) for col in columns]
    )


# ---------------------------------------------------------------------------
# Estimator entry point
# ---------------------------------------------------------------------------

def estimate(fmdata_path=None) -> dict[str, float]:
    """Estimate the CS equation and return ``{name: coefficient}``.

    The returned dict has one entry per structural regressor plus ``"RHO"``
    for the AR(1) coefficient, using the same names as :data:`REFERENCE_PARAMS`.
    """
    window = build_regression_frame(fmdata_path)
    presample, estimation = split_presample_and_estimation(window)

    y = jnp.asarray(estimation["LCSZ"].to_numpy(), dtype=jnp.float64)
    X = _stack_columns(estimation, STRUCTURAL_REGRESSORS)
    Z = _stack_columns(estimation, INSTRUMENTS)
    y_presample = jnp.asarray(float(presample["LCSZ"].item()))
    X_presample = jnp.array([
        float(presample[c].item()) for c in STRUCTURAL_REGRESSORS
    ])

    beta, rho, _se, iterations = two_sls_ar1(
        y, X, Z, y_presample, X_presample, max_iter=200,
    )

    params = {name: float(beta[i]) for i, name in enumerate(STRUCTURAL_REGRESSORS)}
    params["RHO"] = float(rho)
    _LOG.info(
        "n_obs=%d regressors=%d instruments=%d rho_iters=%d",
        estimation.height, X.shape[1], Z.shape[1], int(iterations),
    )
    return params


def report_drift(estimated: dict[str, float]) -> pl.DataFrame:
    """Side-by-side comparison vs Fair's fmout.txt reference."""
    rows = []
    for name, fair_value in REFERENCE_PARAMS.items():
        our_value = estimated[name]
        rows.append({
            "param":      name,
            "fair":       fair_value,
            "this_run":   our_value,
            "delta":      our_value - fair_value,
            "abs_err":    abs(our_value - fair_value),
        })
    drift = pl.DataFrame(rows)

    print()
    print(f"  {'param':12s} {'fair':>14s} {'this_run':>14s} {'delta':>12s}")
    for row in drift.iter_rows(named=True):
        print(f"  {row['param']:12s} {row['fair']:+14.6f} "
              f"{row['this_run']:+14.6f} {row['delta']:+12.4f}")
    print(f"\n[us_cs] max abs delta = {drift['abs_err'].max():.3e}")
    return drift


if __name__ == "__main__":
    params = estimate()
    report_drift(params)
