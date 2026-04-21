"""Step 3 — dynamic simulation of the model.

The outer time loop is a ``jax.lax.scan`` so the whole forecast compiles to a
single fused kernel. Per-period Newton-Raphson lives inside the scan body
(see ``solver.make_scan_step``); autodiff, jit, and vmap all compose from here.

Scan protocol
-------------

Scan carry (state threaded from one quarter to the next):
    ``SimCarry = (C_prev, I_prev, eq1_u_lag1)``
    — the two endogenous variables Newton needs as lags, plus the AR(1)
    residual we propagate forward for equation 1.

Scan inputs (exogenous values read at each quarter):
    ``SimInputs = (G_t, R_t, R_lag1)``

Scan outputs (collected across quarters):
    ``SimOutputs = (C_t, I_t, Y_t, newton_iters, newton_rnorm)``
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from .. import config
from ..core import equations, solver


# ---------------------------------------------------------------------------
# Scan protocol types
# ---------------------------------------------------------------------------

class SimCarry(NamedTuple):
    """State threaded through the scan from one quarter to the next."""
    C_prev: jnp.ndarray
    I_prev: jnp.ndarray
    eq1_u_lag1: jnp.ndarray


class SimInputs(NamedTuple):
    """Exogenous values read at each quarter."""
    G: jnp.ndarray
    R: jnp.ndarray
    R_lag1: jnp.ndarray


class SimOutputs(NamedTuple):
    """Per-quarter simulation result; stacked along time by ``lax.scan``."""
    C: jnp.ndarray
    I: jnp.ndarray  # noqa: E741  (Fair's IS-model investment variable is canonically "I")
    Y: jnp.ndarray
    newton_iters: jnp.ndarray
    newton_residual: jnp.ndarray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# First period for which ``R_lag1`` is available. With IS.DAT/FRED starting at
# 1952Q1, the first lagged quarter is 1952Q2 (`_lag1 = 1952Q1`), but Fair's IS
# model also needs consumption/investment lags so the simulation can't really
# start from the very beginning. 1954Q1 matches his in-sample window.
_SIM_START_PERIOD = "1954Q1"


def _compute_historical_eq1_residual(
    df: pl.DataFrame, params: dict[str, float]
) -> np.ndarray:
    """Compute u_t = LOGC_t − fitted(LOGC_t, no-AR) for every historical row.

    We use the last historical value (at ``sim_start - 1``) to seed the AR(1)
    term at the first simulated period. Note we drop the ρ piece here — the
    residual *is* the thing ρ multiplies.

    Args:
      df: Wide frame indexed by period, containing at least C, Y, R and their
        relevant lags.
      params: Coefficient dict as returned by step02.

    Returns:
      (n,) numpy array of residuals, with NaN in positions where ``C_lag1``
      is unavailable (i.e., the first period).
    """
    log_C = np.log(df["C"].to_numpy())
    log_C_lag1 = np.log(df["C"].shift(1).to_numpy())
    log_Y = np.log(df["Y"].to_numpy())
    R = df["R"].to_numpy()
    fitted_without_rho = (
        params["eq1_const"]
        + params["eq1_C_lag1"] * log_C_lag1
        + params["eq1_Y"] * log_Y
        + params["eq1_R"] * R
    )
    return log_C - fitted_without_rho


def _build_scan_body(params: dict[str, jnp.ndarray]):
    """Return a ``(carry, inputs) -> (new_carry, outputs)`` scan body.

    Closes over ``params`` and the IS equation registry so the scan body has
    the signature ``lax.scan`` expects: all non-array arguments must be fixed
    at trace time.
    """
    solve_period = solver.make_scan_step(
        equations=equations.IS_EQUATIONS,
        endogenous=equations.IS_ENDOGENOUS,
        params=params,
    )

    def scan_body(carry: SimCarry, inputs: SimInputs
                  ) -> tuple[SimCarry, SimOutputs]:
        # Build the state dict the equation functions will read.
        state_other = {
            "G":          inputs.G,
            "R":          inputs.R,
            "R_lag1":     inputs.R_lag1,
            "C_lag1":     carry.C_prev,
            "I_lag1":     carry.I_prev,
            "eq1_u_lag1": carry.eq1_u_lag1,
        }
        # Warm-start Newton with the previous period's values (Y via identity).
        initial_guess = jnp.stack(
            [carry.C_prev, carry.I_prev, carry.C_prev + carry.I_prev + inputs.G]
        )
        newton = solve_period(state_other, initial_guess)
        C_t, I_t, Y_t = newton.x[0], newton.x[1], newton.x[2]

        # Advance the AR(1) state for the next period.
        fitted_without_rho = (
            params["eq1_const"]
            + params["eq1_C_lag1"] * jnp.log(carry.C_prev)
            + params["eq1_Y"] * jnp.log(Y_t)
            + params["eq1_R"] * inputs.R
        )
        u_t = jnp.log(C_t) - fitted_without_rho

        new_carry = SimCarry(C_prev=C_t, I_prev=I_t, eq1_u_lag1=u_t)
        outputs = SimOutputs(
            C=C_t, I=I_t, Y=Y_t,
            newton_iters=newton.iterations,
            newton_residual=newton.residual_norm,
        )
        return new_carry, outputs

    return scan_body


def _stitch_simulated_onto_historical(
    periods: list[str],
    historical: dict[str, np.ndarray],
    simulated: SimOutputs,
    sim_start_index: int,
) -> pl.DataFrame:
    """Assemble the step output frame.

    For ``t < sim_start_index`` rows keep their historical values; for ``t ≥
    sim_start_index`` the simulated path replaces them.
    """
    n_periods = len(periods)
    C_combined = historical["C"].copy()
    I_combined = historical["I"].copy()
    Y_combined = historical["Y"].copy()
    newton_iters = np.zeros(n_periods, dtype=np.int32)
    newton_residual = np.zeros(n_periods)

    C_combined[sim_start_index:] = np.asarray(simulated.C)
    I_combined[sim_start_index:] = np.asarray(simulated.I)
    Y_combined[sim_start_index:] = np.asarray(simulated.Y)
    newton_iters[sim_start_index:] = np.asarray(simulated.newton_iters,
                                                dtype=np.int32)
    newton_residual[sim_start_index:] = np.asarray(simulated.newton_residual)

    converged = np.zeros(n_periods, dtype=bool)
    converged[sim_start_index:] = newton_residual[sim_start_index:] < config.NEWTON_TOL

    return pl.DataFrame(
        {
            "period":           periods,
            "C_sim":            C_combined,
            "I_sim":            I_combined,
            "Y_sim":            Y_combined,
            "C_actual":         historical["C"],
            "I_actual":         historical["I"],
            "Y_actual":         historical["Y"],
            "newton_iters":     newton_iters,
            "newton_residual":  newton_residual,
            "newton_converged": converged,
        }
    )


# ---------------------------------------------------------------------------
# Step entry point
# ---------------------------------------------------------------------------

def run(
    model: str,
    data: pl.DataFrame,
    params: dict[str, float],
    data_source: str = "fred",
    force: bool = False,
) -> pl.DataFrame:
    """Simulate the model quarter by quarter and cache the result.

    Args:
      model: Model name. Only ``"is"`` is supported in v0.1.
      data: Output of step01 — wide frame with raw C, I, Y, G, R columns.
      params: Coefficients from step02.
      data_source: Label threaded through cache paths.
      force: If True, ignore any existing cache and re-solve.

    Returns:
      Wide DataFrame with actual + simulated C/I/Y and per-period Newton stats.

    Raises:
      NotImplementedError: if ``model`` is not ``"is"``.
    """
    cache_path = config.OUTPUT_DIR / f"step03_solved_{model}_{data_source}.parquet"
    if cache_path.exists() and not force:
        return pl.read_parquet(cache_path)

    if model != "is":
        raise NotImplementedError(
            f"step03: only 'is' supported in v0.1, got {model!r}"
        )

    df = data.sort("period")
    periods = df["period"].to_list()
    historical = {
        "C": df["C"].to_numpy(),
        "I": df["I"].to_numpy(),
        "Y": df["Y"].to_numpy(),
        "G": df["G"].to_numpy(),
        "R": df["R"].to_numpy(),
    }
    historical_eq1_residual = _compute_historical_eq1_residual(df, params)

    sim_start = periods.index(_SIM_START_PERIOD)
    n_sim = len(periods) - sim_start

    # Prepare scan inputs and initial carry.
    scan_inputs = SimInputs(
        G=jnp.asarray(historical["G"][sim_start:]),
        R=jnp.asarray(historical["R"][sim_start:]),
        R_lag1=jnp.asarray(historical["R"][sim_start - 1 : sim_start - 1 + n_sim]),
    )
    initial_carry = SimCarry(
        C_prev=jnp.asarray(historical["C"][sim_start - 1]),
        I_prev=jnp.asarray(historical["I"][sim_start - 1]),
        eq1_u_lag1=jnp.asarray(historical_eq1_residual[sim_start - 1]),
    )

    params_jnp = {k: jnp.asarray(v) for k, v in params.items()}
    scan_body = _build_scan_body(params_jnp)
    _, simulated = jax.lax.scan(scan_body, initial_carry, scan_inputs)

    result = _stitch_simulated_onto_historical(
        periods, historical, simulated, sim_start
    )
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result.write_parquet(cache_path)

    max_iter = int(result["newton_iters"][sim_start:].max())
    print(f"[step03] scan solved {n_sim} periods in one kernel; "
          f"max Newton iters = {max_iter}")
    return result
