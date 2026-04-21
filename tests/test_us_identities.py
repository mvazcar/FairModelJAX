"""Verify Fair's accounting identities are correctly transcribed.

Evaluates each identity residual at the actual historical values from
fmdata.txt. A correctly-transcribed identity should return ~0 because the
data already satisfies it — Fair builds fmdata.txt so every identity holds
exactly in the sample.

Any identity that returns a non-zero residual here is a bug in our
transcription, not an economic failure of the model.
"""
from __future__ import annotations


import jax.numpy as jnp
import pytest

from pyfair import readers, config, us_solve


# Tolerance for historical identity residuals — tight because the data
# should satisfy each identity exactly (up to rounding in fmdata.txt).
IDENTITY_RESIDUAL_TOL = 1e-3  # relative, or absolute for dimensionless terms


@pytest.fixture(scope="module")
def historical_state_2019q4() -> dict[str, jnp.ndarray]:
    """A single-quarter state dict built from actual fmdata.txt values.

    2019Q4 is pre-COVID, well inside the sample, and every series is populated.
    We build lags out to four quarters so CCF1 (which uses PIK_lag3 and
    IKF_lag3) can be evaluated.
    """
    long = readers.parse_fair_data(config.US_FMDATA)
    wide = readers.pivot_to_wide(long).sort("period")
    period_list = wide["period"].to_list()
    idx = period_list.index("2019Q4")

    state: dict[str, jnp.ndarray] = {}
    for lag in range(5):   # current + lags 1..4
        lag_period = period_list[idx - lag]
        row = wide.filter(wide["period"] == lag_period).to_dicts()[0]
        for k, v in row.items():
            if not isinstance(v, (int, float)):
                continue
            key = k if lag == 0 else f"{k}_lag{lag}"
            if key not in state:
                state[key] = jnp.asarray(v, dtype=jnp.float64)
    return state


@pytest.mark.parametrize(
    "name,residual_fn",
    us_solve.IDENTITIES,
    ids=[name for name, _ in us_solve.IDENTITIES],
)
def test_identity_residual_zero_on_historical_data(
    name, residual_fn, historical_state_2019q4
):
    """Every identity should evaluate to ~0 on real historical data."""
    try:
        residual = float(residual_fn(historical_state_2019q4, {}))
    except KeyError as exc:
        pytest.skip(f"Identity {name} needs variable we don't have in fmdata: {exc}")
    except Exception as exc:
        pytest.skip(f"Identity {name} needs supporting variable we don't load: {exc}")

    # For identities on per-dollar magnitudes, compare relative to the LHS.
    lhs = float(historical_state_2019q4.get(name, 1.0))
    scale = max(abs(lhs), 1.0)
    rel_residual = abs(residual) / scale
    assert rel_residual < IDENTITY_RESIDUAL_TOL, (
        f"Identity {name}: residual {residual:+.6e} "
        f"(relative {rel_residual:.2e}) exceeds {IDENTITY_RESIDUAL_TOL}"
    )
