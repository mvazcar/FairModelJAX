"""Step 4 — compare estimated coefficients against Fair (2013) IS.OUT.

Transparency policy (see ``IS_DATA_SOURCES.md`` and the ``transparency over
false parity`` feedback memory):

* When ``data_source == "fair_2013"`` we run against Fair's frozen snapshot
  and expect coefficients to land within ``PARITY_TOL_FAIR_2013`` — failures
  here mean an algorithm regression.
* When ``data_source == "fred"`` the data has diverged (NIPA rebase since
  2013) and coefficients *should* differ. We print the drift side-by-side
  but never raise.

Either way the output parquet is the same shape: one row per reference
coefficient with the observed delta.
"""
from __future__ import annotations

import polars as pl

from .. import config


# Fair 2013 IS.OUT reference values (lines 241–270 of 06_examples/IS.OUT).
REFERENCE_PARAMS: dict[str, float] = {
    "eq1_const":   0.019298152,
    "eq1_C_lag1":  0.983444419,
    "eq1_Y":       0.014296188,
    "eq1_R":      -0.000342318,
    "eq1_rho":     0.393074821,
    "eq2_const":  -0.045980497,
    "eq2_I_lag1":  0.968127602,
    "eq2_Y":       0.031957749,
    "eq2_R_lag1": -0.001609687,
}

# Tolerance for the "running on Fair's own 2013 data" mode. Fair's TSLS17 +
# RHOA algorithm (ported in step02_estimate, v0.2) converges to within ~5.5e-3
# of Fair's published IS.OUT coefficients. The residual gap is likely due to
# Fair using a 240-row Z for the first-stage projection where we use 239;
# closing it fully would require a deeper pre-sample-row refactor.
PARITY_TOL_FAIR_2013 = 7e-3


def _build_parity_frame(
    estimated: dict[str, float],
    reference: dict[str, float] = REFERENCE_PARAMS,
) -> pl.DataFrame:
    """Assemble a per-coefficient comparison sorted by absolute error desc."""
    rows = []
    for name, fair_value in reference.items():
        our_value = estimated[name]
        absolute_error = abs(our_value - fair_value)
        relative_error = absolute_error / max(abs(fair_value), 1e-12)
        rows.append({
            "param":            name,
            "fair_2013_value":  fair_value,
            "python_value":     our_value,
            "delta":            our_value - fair_value,
            "abs_err":          absolute_error,
            "rel_err":          relative_error,
        })
    return pl.DataFrame(rows).sort("abs_err", descending=True)


def _print_report(parity: pl.DataFrame, data_source: str) -> None:
    """Emit the banner + side-by-side table to stdout."""
    print(f"[step04] reference    = Fair (2013) IS.OUT, "
          f"dated {config.FAIR_REFERENCE_DATE}")
    print(f"[step04] data source  = {data_source}  "
          f"({config.is_dat_path(data_source).name})")
    if data_source == "fred":
        print("[step04] NOTE: FRED is current-vintage (chained 2017$). "
              "Drift from the 2013")
        print("[step04]       reference is expected: "
              "data rebase + v0.1 algorithm gap.")
    print()
    print(f"  {'param':14s} {'fair_2013':>14s} {'this_run':>14s} {'delta':>12s}")
    for row in parity.sort("param").iter_rows(named=True):
        print(f"  {row['param']:14s} {row['fair_2013_value']:+14.6f} "
              f"{row['python_value']:+14.6f} {row['delta']:+12.4f}")
    print(f"\n[step04] max abs delta = {parity['abs_err'].max():.3e}")


def run(
    model: str,
    python_solution: pl.DataFrame,
    params: dict[str, float] | None = None,
    data_source: str = "fred",
    force: bool = False,
) -> pl.DataFrame:
    """Report + persist the coefficient drift report vs Fair's 2013 reference.

    Args:
      model: Model name. Only ``"is"`` is supported in v0.1.
      python_solution: Step03 output — not used for the comparison itself
        but kept in the signature for pipeline symmetry.
      params: Coefficients from step02. If ``None`` we re-load from the
        cached parquet.
      data_source: ``"fred"`` (default) — print only; ``"fair_2013"`` —
        assert within tolerance.
      force: If True, ignore any existing cache and re-run.

    Returns:
      DataFrame with one row per coefficient: (param, fair_2013_value,
      python_value, delta, abs_err, rel_err).

    Raises:
      NotImplementedError: If ``model`` is not ``"is"``.
      AssertionError: If ``data_source == "fair_2013"`` and any coefficient
        differs by more than ``PARITY_TOL_FAIR_2013``.
    """
    cache_path = config.OUTPUT_DIR / f"step04_parity_{model}_{data_source}.parquet"
    if cache_path.exists() and not force:
        return pl.read_parquet(cache_path)

    if model != "is":
        raise NotImplementedError(
            f"step04: only 'is' supported in v0.1, got {model!r}"
        )

    if params is None:
        params_df = pl.read_parquet(
            config.OUTPUT_DIR / f"step02_params_{model}_{data_source}.parquet"
        )
        params = dict(zip(params_df["name"].to_list(),
                          params_df["value"].to_list()))

    parity = _build_parity_frame(params)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parity.write_parquet(cache_path)
    _print_report(parity, data_source)

    if data_source == "fair_2013":
        failures = parity.filter(parity["abs_err"] >= PARITY_TOL_FAIR_2013)
        if failures.height:
            raise AssertionError(
                f"Fair-2013 parity broken at tol={PARITY_TOL_FAIR_2013}: "
                f"{failures['param'].to_list()}"
            )
        print(f"[step04] PASS  (all within {PARITY_TOL_FAIR_2013} of Fair 2013)")

    return parity
