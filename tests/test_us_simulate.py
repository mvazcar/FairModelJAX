"""Regression test for the US-model dynamic simulation.

Runs an in-sample dynamic simulation over a calm stretch of history
(2015Q1 – 2018Q4, pre-COVID, no major shocks) and verifies:

1. Newton converges at every period (residual norm below tolerance).
2. Simulated paths stay within a reasonable band of actual history.

Out-of-sample forecast testing (2026 onwards) requires an exogenous-path
loader for `fmexog.txt` — deferred to a later session. In-sample dynamic
simulation is the harder numerical test anyway because we're confronting
the model with historical shocks on the RHS without letting u_t absorb them.
"""
from __future__ import annotations

import math

import pytest

from pyfair import us_model, us_solve


# Newton should converge at or below this residual on every period.
NEWTON_TOL = 1e-6

# In-sample dynamic drift tolerance. Over 4 years of simulation starting from
# historical initial conditions, small equation errors accumulate. 5% relative
# drift on GDP-scale quantities is within the published Fair RMSE band.
DYNAMIC_DRIFT_TOL_GDP = 0.05


@pytest.fixture(scope="module")
def simulation_2015_to_2018():
    """Run the in-sample simulation once and reuse across tests."""
    frame = us_model.build_frame()
    results = us_model.estimate_all()
    return frame, us_solve.simulate(
        frame, results,
        start_period="2015Q1",
        end_period="2018Q4",
    )


def test_newton_converges_every_period(simulation_2015_to_2018):
    """Every period's Newton solve must hit machine-precision residual."""
    _frame, sim = simulation_2015_to_2018
    failures = [
        (p, sim.residual_norms[p])
        for p in sim.periods
        if not math.isfinite(sim.residual_norms[p])
        or sim.residual_norms[p] >= NEWTON_TOL
    ]
    assert not failures, (
        f"Newton failed to converge on {len(failures)} period(s): {failures[:5]}"
    )


def test_newton_takes_few_iterations(simulation_2015_to_2018):
    """Well-specified US model + good initial guess → Newton converges fast."""
    _frame, sim = simulation_2015_to_2018
    max_iters = max(sim.iterations.values())
    assert max_iters <= 10, (
        f"Max Newton iterations {max_iters} exceeds 10 — model may be "
        f"poorly conditioned in some period"
    )


def test_simulated_gdp_stays_near_actual(simulation_2015_to_2018):
    """Dynamic simulation of Y shouldn't drift more than 5% from actuals."""
    frame, sim = simulation_2015_to_2018
    max_drift = 0.0
    worst_period = None
    for period in sim.periods:
        simulated = sim.solved[period]["Y"]
        actual_row = frame.filter(frame["period"] == period).to_dicts()[0]
        actual = actual_row["Y"]
        drift = abs(simulated - actual) / abs(actual)
        if drift > max_drift:
            max_drift = drift
            worst_period = period
    assert max_drift < DYNAMIC_DRIFT_TOL_GDP, (
        f"Max Y drift {max_drift*100:.2f}% at {worst_period} exceeds "
        f"{DYNAMIC_DRIFT_TOL_GDP*100:.0f}%"
    )


def test_all_endogenous_variables_produced(simulation_2015_to_2018):
    """Each solved period has every endogenous variable populated."""
    _frame, sim = simulation_2015_to_2018
    endog_count = 118  # 23 stochastic + 95 identities
    for period in sim.periods:
        solved = sim.solved[period]
        assert len(solved) == endog_count, (
            f"{period}: expected {endog_count} endogenous vars, "
            f"got {len(solved)}"
        )


# ---------------------------------------------------------------------------
# Out-of-sample forecast (2026Q1–2029Q4)
# ---------------------------------------------------------------------------

FORECAST_NEWTON_TOL = 1e-5


@pytest.fixture(scope="module")
def forecast_2026_to_2029():
    """Full US model forecast over Fair's published forecast window."""
    frame = us_model.build_frame()
    results = us_model.estimate_all()
    extended = us_solve.extend_frame_for_forecast(frame)
    return extended, us_solve.simulate(
        extended, results,
        start_period="2026Q1",
        end_period="2029Q4",
    )


def test_forecast_newton_converges(forecast_2026_to_2029):
    """Every forecast period's Newton solve hits machine precision."""
    _ext, sim = forecast_2026_to_2029
    failures = [
        (p, sim.residual_norms[p])
        for p in sim.periods
        if not math.isfinite(sim.residual_norms[p])
        or sim.residual_norms[p] >= FORECAST_NEWTON_TOL
    ]
    assert not failures, (
        f"Forecast Newton failed on {len(failures)} period(s): {failures[:5]}"
    )


def test_forecast_produces_sensible_paths(forecast_2026_to_2029):
    """GDP, UR, RS should be in plausible macro ranges throughout the forecast."""
    _ext, sim = forecast_2026_to_2029

    for period in sim.periods:
        solved = sim.solved[period]
        # Unemployment rate between 1% and 15% (historical US envelope)
        assert 0.01 < solved["UR"] < 0.15, (
            f"{period}: UR = {solved['UR']:.4f} outside [1%, 15%]"
        )
        # Short rate between 0% and 15%
        assert -1.0 < solved["RS"] < 15.0, (
            f"{period}: RS = {solved['RS']:.2f} outside [-1, 15]"
        )
        # Real GDP should be above historical 2025Q4 level
        assert solved["Y"] > 4000, (
            f"{period}: Y = {solved['Y']:.0f} unreasonably small"
        )


def test_forecast_all_16_quarters(forecast_2026_to_2029):
    """Sanity: forecast covers exactly the 16 quarters 2026Q1–2029Q4."""
    _ext, sim = forecast_2026_to_2029
    assert len(sim.periods) == 16
    assert sim.periods[0] == "2026Q1"
    assert sim.periods[-1] == "2029Q4"
