"""Per-period simultaneous-equation solver.

Given a list of equations (each a pure function ``state, params -> residual``)
and a set of endogenous variables, finds values of those endogenous variables
that make every equation residual zero at a single period.

The solver is pure JAX:

* Newton-Raphson with ``jax.jacfwd`` for the Jacobian.
* ``jax.lax.while_loop`` for the iteration — no Python control flow, so the
  whole Newton iteration compiles to a single kernel and is safe to call from
  inside ``jax.lax.scan`` (see ``step03_solve``).
* Forward-mode differentiation is preferred over reverse because our system
  has few endogenous variables (<10); forward is cheaper in that regime.

The algorithm at each iteration::

    r = F(x)                    # residual vector
    J = ∂F/∂x evaluated at x    # Jacobian
    x_new = x - J⁻¹ r           # Newton step

with convergence declared when ‖F(x)‖ < tol or iteration count exceeds
``max_iter``.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .. import config


# An equation is a triple of (name, function, kind). ``kind`` is one of
# "stochastic" (estimated coefficients, may have AR error) or "identity"
# (deterministic accounting). The solver treats them identically — both are
# driven to residual zero.
Equation = tuple[str, Callable, str]
EquationRegistry = list[Equation]


class NewtonState(NamedTuple):
    """Loop carry for the Newton iteration.

    Attributes:
      iteration: How many Newton steps have been taken so far.
      x: Current guess for the endogenous variables.
      residual_norm: Euclidean norm of ``F(x)`` — the scalar we drive to zero.
    """
    iteration: jnp.ndarray
    x: jnp.ndarray
    residual_norm: jnp.ndarray


class NewtonResult(NamedTuple):
    """Per-period solve summary returned by ``make_scan_step``.

    Attributes:
      x: The solved endogenous variables.
      iterations: Number of Newton steps actually used (≤ max_iter).
      residual_norm: Final ‖F(x)‖ — should be below ``tol`` if converged.
    """
    x: jnp.ndarray
    iterations: jnp.ndarray
    residual_norm: jnp.ndarray


def _build_residual_fn(
    equations: EquationRegistry,
    endogenous: list[str],
) -> Callable[[jnp.ndarray, dict, dict], jnp.ndarray]:
    """Build the vector-valued residual function ``F(x, state_other, params)``.

    The returned function packs the ``x`` vector (one scalar per endogenous
    variable, in the order given by ``endogenous``) into the ``state`` dict
    that equation functions consume, calls every equation, and stacks the
    residuals into a single array.

    Args:
      equations: Registry of (name, fn, kind) triples. Each ``fn(state, params)``
        must return a scalar residual.
      endogenous: Variable names solved for jointly, in order. ``x[i]`` will
        be placed at ``state[endogenous[i]]`` before each call.

    Returns:
      A function ``(x, state_other, params) -> residuals`` where
      ``residuals.shape == (len(equations),)``.
    """

    def residuals(
        x: jnp.ndarray,
        state_other: dict[str, jnp.ndarray],
        params: dict[str, jnp.ndarray],
    ) -> jnp.ndarray:
        state = dict(state_other)
        for name, value in zip(endogenous, x):
            state[name] = value
        return jnp.stack([equation_fn(state, params)
                          for _, equation_fn, _ in equations])

    return residuals


def _newton_iteration(
    F: Callable,
    state_other: dict,
    params: dict,
    tol: float,
    max_iter: int,
) -> Callable[[jnp.ndarray], NewtonState]:
    """Return a ``while_loop``-compatible Newton driver that finds ``F(x) = 0``.

    The returned function takes an initial guess and runs the while loop.
    We close over the fixed quantities (``F``, ``state_other``, ``params``,
    ``tol``, ``max_iter``) so the while-loop body has no non-array arguments.
    """
    jacobian = jax.jacfwd(F, argnums=0)

    def not_converged(loop_state: NewtonState) -> jnp.ndarray:
        below_max = loop_state.iteration < max_iter
        above_tol = loop_state.residual_norm >= tol
        return jnp.logical_and(below_max, above_tol)

    def one_newton_step(loop_state: NewtonState) -> NewtonState:
        r = F(loop_state.x, state_other, params)
        J = jacobian(loop_state.x, state_other, params)
        step = jnp.linalg.solve(J, r)
        x_new = loop_state.x - step
        r_new_norm = jnp.linalg.norm(F(x_new, state_other, params))
        return NewtonState(loop_state.iteration + 1, x_new, r_new_norm)

    def driver(initial_guess: jnp.ndarray) -> NewtonState:
        initial_norm = jnp.linalg.norm(F(initial_guess, state_other, params))
        initial = NewtonState(jnp.int32(0), initial_guess, initial_norm)
        return jax.lax.while_loop(not_converged, one_newton_step, initial)

    return driver


def newton_solve(
    equations: EquationRegistry,
    endogenous: list[str],
    state_other: dict[str, jnp.ndarray],
    params: dict[str, jnp.ndarray],
    initial_guess: jnp.ndarray,
    tol: float = config.NEWTON_TOL,
    max_iter: int = config.NEWTON_MAX_ITER,
) -> tuple[jnp.ndarray, dict]:
    """Solve the system of equations for one period.

    Stand-alone entry point used by tests and any non-``scan`` caller. For the
    production forecast path (``step03_solve``) use ``make_scan_step`` instead,
    which compiles the whole time loop as one kernel.

    Args:
      equations: Equation registry from ``equations.py``.
      endogenous: Variable names to solve for, in the order that matches
        ``initial_guess``.
      state_other: All non-endogenous state the equations read (lags, exogenous
        variables, AR residuals).
      params: Estimated coefficients keyed by name.
      initial_guess: Starting point for Newton.
      tol: Convergence threshold on ‖F(x)‖.
      max_iter: Safety cap on iterations.

    Returns:
      Tuple ``(x, info)`` where ``x`` is the solved vector and ``info`` is a
      dict with keys ``iterations``, ``residual_norm``, ``converged`` (all
      traced scalars — call ``float()`` / ``bool()`` to extract).
    """
    F = _build_residual_fn(equations, endogenous)
    driver = _newton_iteration(F, state_other, params, tol, max_iter)
    final = driver(initial_guess)
    return final.x, {
        "iterations": final.iteration,
        "residual_norm": final.residual_norm,
        "converged": final.residual_norm < tol,
    }


def make_scan_step(
    equations: EquationRegistry,
    endogenous: list[str],
    params: dict[str, jnp.ndarray],
    tol: float = config.NEWTON_TOL,
    max_iter: int = config.NEWTON_MAX_ITER,
) -> Callable[[dict, jnp.ndarray], NewtonResult]:
    """Return a solve-one-period function suited for ``jax.lax.scan``.

    The returned function closes over ``equations``, ``endogenous``, and
    ``params`` — this matters because ``lax.scan`` bodies can't accept
    non-array arguments. ``state_other`` and ``initial_guess`` vary each
    period and are passed in explicitly.

    Args:
      equations: Equation registry.
      endogenous: Endogenous variable order.
      params: Fixed coefficients for the scan run.
      tol: Convergence threshold on ‖F(x)‖.
      max_iter: Safety cap on Newton iterations.

    Returns:
      A function ``solve(state_other, initial_guess) -> NewtonResult`` that
      can be called from within a scan body.
    """
    F = _build_residual_fn(equations, endogenous)

    def solve(
        state_other: dict[str, jnp.ndarray],
        initial_guess: jnp.ndarray,
    ) -> NewtonResult:
        driver = _newton_iteration(F, state_other, params, tol, max_iter)
        final = driver(initial_guess)
        return NewtonResult(final.x, final.iteration, final.residual_norm)

    return solve
