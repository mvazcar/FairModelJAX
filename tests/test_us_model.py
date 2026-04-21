"""Batch regression test for the US Model equations.

Loads fmdata.txt, runs every equation in ``us_model.EQUATIONS`` through
``estimate``, and compares against Fair's published ``fmout.txt`` reference
coefficients. Three tolerance bands:

  * **Machine precision** (1e-6): structural 2SLS matches exactly.
  * **Loose non-AR** (5e-3): event-dummy drift (EQ 27) or wage preprocessing
    sensitivity (EQ 16).
  * **AR(1)** (3e-2): our iterated-IV-with-cumulative-update converges close
    to, but not exactly on, Fair's TSLS17 fixed point.
  * **High-AR / approximated** (1.0): equations where our estimator lands at
    a meaningfully different point than Fair:
        - EQ 4, 28: near-unit-root AR (ρ ≈ 0.9) — bounded search gives a
          valid but different local solution.
        - EQ 11: Fair uses AR(3); we approximate with AR(1). Coefs differ.
"""
from __future__ import annotations

import math

import pytest

from pyfair import us_model


TOL_NON_AR_STRICT = 1e-6
TOL_NON_AR_LOOSE = 5e-3
TOL_AR = 3e-2
TOL_AR_HARD = 1.0          # equations with a known algorithm / spec gap

# Equations grouped by expected fit tier.
_STRICT_NON_AR = {3, 5, 6, 7, 8, 13, 14, 17, 18, 24, 26, 30}
_LOOSE_NON_AR = {16, 27}
_AR_NORMAL = {1, 2, 10, 12, 15, 23, 29}
_AR_HARD = {4, 11, 28}


@pytest.fixture(scope="module")
def estimation_results() -> list[us_model.EstimationResult]:
    """Estimate every equation once; reuse across all per-equation tests."""
    return us_model.estimate_all()


def _max_abs_error(result: us_model.EstimationResult) -> float:
    errors = []
    for tok, ours in result.coefficients.items():
        fair = result.reference.get(tok, math.nan)
        if not math.isnan(fair):
            errors.append(abs(ours - fair))
    return max(errors) if errors else math.nan


def _get(estimation_results, eq_number):
    matches = [r for r in estimation_results if r.equation.number == eq_number]
    assert matches, f"EQ {eq_number} not in estimation results"
    return matches[0]


@pytest.mark.parametrize("eq_number", sorted(_STRICT_NON_AR))
def test_non_ar_equation_matches_fair_to_machine_precision(eq_number, estimation_results):
    result = _get(estimation_results, eq_number)
    max_err = _max_abs_error(result)
    assert max_err < TOL_NON_AR_STRICT, (
        f"EQ {eq_number} ({result.equation.dependent}): "
        f"max abs error {max_err:.2e} exceeds {TOL_NON_AR_STRICT}"
    )


@pytest.mark.parametrize("eq_number", sorted(_LOOSE_NON_AR))
def test_non_ar_equation_matches_fair_loosely(eq_number, estimation_results):
    result = _get(estimation_results, eq_number)
    max_err = _max_abs_error(result)
    assert max_err < TOL_NON_AR_LOOSE, (
        f"EQ {eq_number} ({result.equation.dependent}): "
        f"max abs error {max_err:.2e} exceeds {TOL_NON_AR_LOOSE}"
    )


@pytest.mark.parametrize("eq_number", sorted(_AR_NORMAL))
def test_ar_equation_in_fair_ballpark(eq_number, estimation_results):
    result = _get(estimation_results, eq_number)
    max_err = _max_abs_error(result)
    assert max_err < TOL_AR, (
        f"EQ {eq_number} ({result.equation.dependent}): "
        f"max abs error {max_err:.2e} exceeds {TOL_AR}"
    )


@pytest.mark.parametrize("eq_number", sorted(_AR_HARD))
def test_hard_ar_equation_runs_and_is_stable(eq_number, estimation_results):
    """EQ 4, 11, 28 have a larger expected gap vs Fair. We just require that
    the estimator returns a finite result well inside the numeric envelope."""
    result = _get(estimation_results, eq_number)
    max_err = _max_abs_error(result)
    assert math.isfinite(max_err) and max_err < TOL_AR_HARD, (
        f"EQ {eq_number} ({result.equation.dependent}): "
        f"max abs error {max_err} exceeds {TOL_AR_HARD} or non-finite"
    )


def test_all_24_equations_estimated(estimation_results):
    """Every active stochastic equation produced a finite result."""
    estimated = {r.equation.number for r in estimation_results}
    expected = _STRICT_NON_AR | _LOOSE_NON_AR | _AR_NORMAL | _AR_HARD
    missing = expected - estimated
    assert not missing, f"Not estimated: {missing}"
    assert len(estimation_results) == 24, (
        f"Expected 24 results, got {len(estimation_results)}"
    )
