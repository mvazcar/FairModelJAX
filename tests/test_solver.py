"""Sanity tests for the Newton solver.

Uses toy systems where the answer is known analytically. Catches regressions
in the Jacobian plumbing (``jax.jacfwd``) and the ``lax.while_loop``
integration before we trust the solver on Fair's nonlinear equations.
"""
from __future__ import annotations

import jax.numpy as jnp

from pyfair.solver import newton_solve


def _make_stochastic(name: str, fn):
    """Shorthand for a registry entry — kind doesn't matter for these tests."""
    return (name, fn, "stochastic")


def test_newton_solves_linear_system():
    """Solve  2x + y = 5 ,  x + 3y = 10.  Expected: x = 1, y = 3."""
    def eq_a(state, _):
        return 2 * state["x"] + state["y"] - 5.0

    def eq_b(state, _):
        return state["x"] + 3 * state["y"] - 10.0

    x, info = newton_solve(
        equations=[_make_stochastic("a", eq_a), _make_stochastic("b", eq_b)],
        endogenous=["x", "y"],
        state_other={},
        params={},
        initial_guess=jnp.array([0.0, 0.0]),
    )
    assert info["converged"], info
    assert jnp.allclose(x, jnp.array([1.0, 3.0]), atol=1e-8)


def test_newton_solves_nonlinear_system():
    """Solve  x² − 4 = 0 ,  y − x − 1 = 0.

    Two real roots for x (−2, 2); Newton converges to whichever the initial
    guess sits closest to. We start at (1.5, 1.5) so we expect x = 2, y = 3.
    """
    def eq_a(state, _):
        return state["x"] ** 2 - 4.0

    def eq_b(state, _):
        return state["y"] - state["x"] - 1.0

    x, info = newton_solve(
        equations=[_make_stochastic("a", eq_a), _make_stochastic("b", eq_b)],
        endogenous=["x", "y"],
        state_other={},
        params={},
        initial_guess=jnp.array([1.5, 1.5]),
    )
    assert info["converged"], info
    assert jnp.allclose(x, jnp.array([2.0, 3.0]), atol=1e-6)
