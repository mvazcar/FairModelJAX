"""Model equations as pure JAX functions.

Every stochastic equation and every accounting identity is a pure function
with the same signature:

    def equation(state, params) -> residual

The solver drives the stacked residual vector to zero at each quarter to find
the simultaneous equilibrium.

``state`` is a dict mapping variable names (Fair's conventions — ``C``, ``I``,
``Y``, ``G``, ``R``) plus lagged names (``C_lag1``, ``I_lag1``, ``R_lag1``) and
the AR state ``eq1_u_lag1`` to scalar jnp values for **one period**.
``params`` is a dict of estimated coefficients.

This file currently implements Fair's IS tutorial model (2 stochastic
equations + 1 identity, per ``06_examples/IS.INP``). The full US model
(~30 stochastic equations + ~30 identities) will be translated from
``03_us_model/fminput.txt`` in v0.2.
"""
from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

# Type aliases for readability. At runtime these are plain dicts and arrays.
State = dict[str, jnp.ndarray]
Params = dict[str, jnp.ndarray]
Residual = jnp.ndarray
EquationFn = Callable[[State, Params], Residual]


# ---------------------------------------------------------------------------
# IS tutorial model — see 06_examples/IS.INP
# ---------------------------------------------------------------------------

def eq1_consumption(state: State, params: Params) -> Residual:
    """Consumption equation (AR(1) errors).

    Fair form::

        EQ 1 LOGC CNST LOGC(-1) LOGY R RHO=1;  LHS C=EXP(LOGC);

    Structural equation::

        log(C_t) = α + β · log(C_{t-1}) + γ · log(Y_t) + δ · R_t + ρ · u_{t-1}

    where u_{t-1} is the prior period's residual (carried via
    ``state["eq1_u_lag1"]`` from the scan carry). Residual = LHS − RHS;
    Newton drives it to zero.
    """
    fitted = (
        params["eq1_const"]
        + params["eq1_C_lag1"] * jnp.log(state["C_lag1"])
        + params["eq1_Y"]      * jnp.log(state["Y"])
        + params["eq1_R"]      * state["R"]
        + params["eq1_rho"]    * state["eq1_u_lag1"]
    )
    return jnp.log(state["C"]) - fitted


def eq2_investment(state: State, params: Params) -> Residual:
    """Investment equation (no AR term).

    Fair form::

        EQ 2 LOGI CNST LOGI(-1) LOGY R(-1);  LHS I=EXP(LOGI);

    Structural equation::

        log(I_t) = α + β · log(I_{t-1}) + γ · log(Y_t) + δ · R_{t-1}
    """
    fitted = (
        params["eq2_const"]
        + params["eq2_I_lag1"] * jnp.log(state["I_lag1"])
        + params["eq2_Y"]      * jnp.log(state["Y"])
        + params["eq2_R_lag1"] * state["R_lag1"]
    )
    return jnp.log(state["I"]) - fitted


def ident_Y(state: State, _: Params) -> Residual:
    """Output identity Y = C + I + G (closed-economy aggregation)."""
    return state["Y"] - (state["C"] + state["I"] + state["G"])


# ---------------------------------------------------------------------------
# Registry + variable lists
# ---------------------------------------------------------------------------

IS_EQUATIONS: list[tuple[str, EquationFn, str]] = [
    ("eq1_consumption", eq1_consumption, "stochastic"),
    ("eq2_investment",  eq2_investment,  "stochastic"),
    ("ident_Y",         ident_Y,         "identity"),
]
"""Ordered registry the solver iterates over. The order must match
``IS_ENDOGENOUS`` — equation *i* is responsible for endogenous variable *i*."""

IS_ENDOGENOUS: list[str] = ["C", "I", "Y"]
"""Variables the solver finds jointly at each quarter. len must equal
len(IS_EQUATIONS)."""

IS_EXOGENOUS: list[str] = ["G", "R"]
"""Variables taken as given at each quarter (forcing variables)."""

IS_PARAM_NAMES: dict[str, list[str]] = {
    "eq1_consumption": ["eq1_const", "eq1_C_lag1", "eq1_Y", "eq1_R", "eq1_rho"],
    "eq2_investment":  ["eq2_const", "eq2_I_lag1", "eq2_Y", "eq2_R_lag1"],
}
"""Per-equation parameter vectors. Step02 builds design matrices from these."""


# ---------------------------------------------------------------------------
# Full US model — TODO v0.3
# ---------------------------------------------------------------------------
# Translate each ``EQ n ...;`` block from fminput.txt into a function here.
# ~30 stochastic equations + ~30 identities. Keep Fair's numbering (EQ 1..30)
# for easy cross-reference with the workbook PDFs.
