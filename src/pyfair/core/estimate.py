"""Step 2 — estimate coefficients by iterated 2SLS with AR(1) errors.

All linear algebra is ``jax.numpy``. The outer ρ-iteration is a
``jax.lax.while_loop`` so the whole estimator compiles to one kernel and is
``jax.grad``-able end-to-end.

Algorithm — Fair's TSLS17 + RHOA (FP.FOR lines 11750-12890)
-----------------------------------------------------------

Model::

    y_t = X_t · β + u_t,        u_t = ρ · u_{t-1} + e_t

Fair's procedure has four features that distinguish it from a simple
Cochrane-Orcutt loop and collectively pin down the final β and ρ:

1. **Pre-sample row retained.** One row of data *before* the estimation
   window (``y_presample``, ``X_presample``) supplies the lag for the first
   ρ-differenced observation. The ρ-differenced fit uses all T observations
   of the estimation sample.

2. **Full-sample first-stage projection ρ-differenced.** ``X̂ = P_Z · X`` is
   computed once on the T estimation rows and then ρ-differenced. We do NOT
   re-project the ρ-transformed X each iteration.

3. **β formula** ``β̂ = (X̂*' X*)⁻¹ (X̂*' Y*)`` with ``*`` = ρ-difference.
   Subtly different from the textbook IV-2SLS formula.

4. **Cumulative ρ update with IV.** At each iteration::

         U_full[0]   = y_presample − X_presample · β       (pre-sample resid)
         U_full[j≥1] = y_j − X_j · β − ρ · (y_{j-1} − X_{j-1} · β)
         Q           = U_full[0 : T]                        (lag including pre-sample)
         U_dep       = U_full[1 : T+1]
         Δρ          = (Q' P_Z U_dep) / (Q' P_Z Q)
         ρ_new       = ρ_old + Δρ

   The update is *cumulative* (not replacement) — each Δρ is a correction
   relative to the current ρ, absorbing bias in β caused by ρ error. This
   matches Fair's iteration sequence (IS model: 0 → 0.363 → 0.388 → ...).

Calibration on fair_2013 (239 obs, IS model): converges to ρ=0.3946 vs
Fair's 0.3931, max β delta ~5e-3. On US CS (288 obs): converges to ρ=0.238
vs Fair's 0.214, max β delta ~2e-2. Not exact parity with Fair's FORTRAN
(probably due to pre-sample Z handling we haven't replicated), but an
order of magnitude tighter than a plain Cochrane-Orcutt loop.

Target values (Fair 2013 IS.OUT)::

    EQ 1 (LOGC, AR(1))
        CNST      0.019298152    LOGC(-1)  0.983444419    LOGY      0.014296188
        R        -0.000342318    RHO       0.393074821

    EQ 2 (LOGI, no AR)
        CNST     -0.045980497    LOGI(-1)  0.968127602    LOGY      0.031957749
        R(-1)    -0.001609687
"""
from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import polars as pl

from .. import config


# ---------------------------------------------------------------------------
# Pure-JAX estimators
# ---------------------------------------------------------------------------

@jax.jit
def two_sls(y: jnp.ndarray, X: jnp.ndarray, Z: jnp.ndarray) -> jnp.ndarray:
    """Standard 2SLS: β̂ = (X' P_Z X)⁻¹ X' P_Z y.

    Args:
      y: (T,) dependent variable.
      X: (T, k) regressors.
      Z: (T, m) instruments (m ≥ k).

    Returns:
      (k,) coefficient vector.
    """
    ZtZ_inv = jnp.linalg.inv(Z.T @ Z)
    projection_onto_Z = Z @ ZtZ_inv @ Z.T
    return jnp.linalg.solve(
        X.T @ projection_onto_Z @ X,
        X.T @ projection_onto_Z @ y,
    )


@jax.jit
def two_sls_with_se(
    y: jnp.ndarray, X: jnp.ndarray, Z: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """2SLS returning both coefficients and asymptotic SEs."""
    beta = two_sls(y, X, Z)
    residuals = y - X @ beta
    sigma_squared = (residuals @ residuals) / y.shape[0]
    ZtZ_inv = jnp.linalg.inv(Z.T @ Z)
    projection_onto_Z = Z @ ZtZ_inv @ Z.T
    var_beta = sigma_squared * jnp.linalg.inv(X.T @ projection_onto_Z @ X)
    return beta, jnp.sqrt(jnp.abs(jnp.diag(var_beta)))


class _Ar1LoopState(NamedTuple):
    """Loop carry for Fair's ρ-iteration.

    Attributes:
      iteration: Completed iterations so far.
      rho_previous: ρ at the start of the last iteration (for convergence check).
      rho_current: ρ at the end of the last iteration.
      beta: Coefficient vector matching ``rho_current``.
    """
    iteration: jnp.ndarray
    rho_previous: jnp.ndarray
    rho_current: jnp.ndarray
    beta: jnp.ndarray


@jax.jit
def _ar1_transformed_fit(
    y: jnp.ndarray,
    X: jnp.ndarray,
    y_presample: jnp.ndarray,
    X_presample: jnp.ndarray,
    PZ_X_ext: jnp.ndarray,
    rho: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Given ρ, return (beta, SSE) for Fair's ρ-differenced 2SLS.

    Used by the bounded-search estimator. Extracts the β fit and the residual
    sum of squares of the transformed system — the quantity the optimizer
    minimizes over ρ.
    """
    y_ext = jnp.concatenate([jnp.array([y_presample]), y])
    X_ext = jnp.vstack([X_presample[None, :], X])
    YS = y_ext[1:] - rho * y_ext[:-1]
    XS = X_ext[1:] - rho * X_ext[:-1]
    XSH = PZ_X_ext[1:] - rho * PZ_X_ext[:-1]
    beta = jnp.linalg.solve(XSH.T @ XS, XSH.T @ YS)
    residuals = YS - XS @ beta
    return beta, residuals @ residuals


def two_sls_ar1_bounded(
    y: jnp.ndarray,
    X: jnp.ndarray,
    Z: jnp.ndarray,
    y_presample: jnp.ndarray,
    X_presample: jnp.ndarray,
    rho_lo: float = -0.99,
    rho_hi: float = 0.999,
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """Find ρ by bounded 1D search over [rho_lo, rho_hi] (golden section).

    Unlike the fixed-point cumulative update, this always converges: it
    minimizes the concentrated SSE ``min_ρ SSE(ρ, β̂(ρ))`` on a bounded
    interval using golden section. Used for near-unit-root equations where
    the fixed-point iteration oscillates or diverges.

    Args:
      y, X, Z: Estimation-sample data.
      y_presample, X_presample: Pre-sample lag (one period before the window).
      rho_lo, rho_hi: Search bounds; usually just shy of (-1, 1).

    Returns:
      ``(beta, rho, iterations)``.
    """
    T = y.shape[0]
    ZtZ_inv = jnp.linalg.inv(Z.T @ Z)
    PZ_X_est = Z @ ZtZ_inv @ Z.T @ X
    PZ_X_ext = jnp.vstack([X_presample[None, :], PZ_X_est])

    def sse_at(rho: float) -> float:
        _beta, sse = _ar1_transformed_fit(
            y, X, y_presample, X_presample, PZ_X_ext, jnp.asarray(rho),
        )
        return float(sse)

    # Golden-section search.
    phi = (jnp.sqrt(5.0) - 1.0) / 2.0  # ~0.618
    a, b = float(rho_lo), float(rho_hi)
    x1 = b - float(phi) * (b - a)
    x2 = a + float(phi) * (b - a)
    f1, f2 = sse_at(x1), sse_at(x2)
    max_iter = 60  # golden section converges at rate phi ≈ 0.618; 60 iters -> ~1e-13
    iters = 0
    for iters in range(1, max_iter + 1):
        if b - a < 1e-10:
            break
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - float(phi) * (b - a)
            f1 = sse_at(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + float(phi) * (b - a)
            f2 = sse_at(x2)
    rho_hat = (a + b) / 2
    beta_hat, _ = _ar1_transformed_fit(
        y, X, y_presample, X_presample, PZ_X_ext, jnp.asarray(rho_hat),
    )
    return beta_hat, jnp.asarray(rho_hat), iters


def two_sls_ar2_bounded(
    y: jnp.ndarray,
    X: jnp.ndarray,
    Z: jnp.ndarray,
    y_presample: jnp.ndarray,
    X_presample: jnp.ndarray,
    rho1_bounds: tuple[float, float] = (-1.99, 1.99),
    rho2_bounds: tuple[float, float] = (-0.99, 0.99),
) -> tuple[jnp.ndarray, jnp.ndarray, int]:
    """2D Nelder–Mead search over (ρ1, ρ2) minimizing concentrated SSE.

    Used for AR(2) equations where instruments == regressors — the IV
    ρ-update in ``two_sls_ar2`` is degenerate there (the projection kills
    OLS residuals). Nelder–Mead converges robustly even when Fair's
    ρ estimate lies outside the classical AR(1) stationarity band
    (e.g. CA EQ 48 has ρ1 = 1.38).

    Args:
      y, X, Z: estimation-sample data.
      y_presample: (2,) y at the two quarters before the window.
      X_presample: (2, k) X at those same quarters.
      rho1_bounds: Search bounds on ρ1. Wider default than AR(1) because
        AR(2) stationarity allows ρ1 > 1 as long as ρ1 + ρ2 < 1.
      rho2_bounds: Search bounds on ρ2.

    Returns:
      ``(beta, rho, iterations)`` where ``rho`` is a (2,) vector ``[ρ1, ρ2]``.
    """
    # Import scipy lazily — only AR(2) bounded search needs it.
    from scipy.optimize import minimize

    T = y.shape[0]
    y_ext = jnp.concatenate([y_presample, y])
    X_ext = jnp.vstack([X_presample, X])
    ZtZ_inv = jnp.linalg.inv(Z.T @ Z)
    PZ_X_ext = jnp.vstack([X_presample, Z @ ZtZ_inv @ Z.T @ X])

    def fit_given_rhos(rho: jnp.ndarray):
        YS = y_ext[2:] - rho[0] * y_ext[1:-1] - rho[1] * y_ext[:-2]
        XS = X_ext[2:] - rho[0] * X_ext[1:-1] - rho[1] * X_ext[:-2]
        XSH = PZ_X_ext[2:] - rho[0] * PZ_X_ext[1:-1] - rho[1] * PZ_X_ext[:-2]
        beta = jnp.linalg.solve(XSH.T @ XS, XSH.T @ YS)
        residuals = YS - XS @ beta
        return beta, float(residuals @ residuals)

    def sse(rho_np):
        rho = jnp.asarray(rho_np, dtype=jnp.float64)
        # Penalize non-stationarity heavily.
        r1, r2 = rho_np
        if r2 <= -1 or r1 + r2 >= 1 or r2 - r1 >= 1:
            return 1e12
        _, s = fit_given_rhos(rho)
        return s

    result = minimize(
        sse, x0=[0.0, 0.0], method="Nelder-Mead",
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000},
    )
    rho_hat = jnp.asarray(result.x, dtype=jnp.float64)
    beta_hat, _ = fit_given_rhos(rho_hat)
    return beta_hat, rho_hat, int(result.nit)


def nlols_lpxa(
    lpxa: jnp.ndarray,
    lpxb: jnp.ndarray,
    lag_step: int = 1,
    initial: tuple[float, float, float] = (0.5, 1.1, -0.1),
) -> tuple[jnp.ndarray, int]:
    """Nonlinear OLS for Fair's ``xxLPXA`` export-price equations.

    Model (Fair MC.INP 2018, e.g. BE EQ 188, DE EQ 198)::

        LPXA[t] = α · (LPXB[t] − β · LPXB[t−4] − γ · LPXB[t−8])
                  + β · LPXA[t−4] + γ · LPXA[t−8]

    The ``(-4)`` / ``(-8)`` are literal 4-quarter and 8-quarter lags in
    Fair's DSL. Since every ``xxLPXA`` equation belongs to an annual-lag
    country (BE..PE), the underlying series are observed only at yearly
    ``Q1`` intervals — after ``drop_nulls`` the array has one entry per
    year and a lag of one year is ``lag_step=1`` positions back.
    (For a hypothetical quarterly application pass ``lag_step=4``.)

    Fair's FP.EXE runs DFP (Davidon-Fletcher-Powell); we use SciPy
    Levenberg-Marquardt via ``scipy.optimize.least_squares`` which
    reproduces Fair's OUT coefficients to within ~1e-6 and is
    dramatically simpler than rolling our own DFP.

    Args:
      lpxa: (T,) LPXA series after ``drop_nulls`` — earliest
        ``2*lag_step`` entries are used only as lags.
      lpxb: (T,) LPXB series aligned with ``lpxa``.
      lag_step: Positions between "year t" and "year t−1" in the
        dropped-null array. ``1`` for annual-only data (the common case);
        ``4`` if the array contains quarterly observations.
      initial: Starting values for ``(α, β, γ)``. Fair seeds
        ``(0.5, 1.1, -0.1)``.

    Returns:
      ``(coefs, n_fev)`` where ``coefs`` is ``jnp.array([α, β, γ])``.
    """
    from scipy.optimize import least_squares
    import numpy as _np

    lpxa_np = _np.asarray(lpxa, dtype=float)
    lpxb_np = _np.asarray(lpxb, dtype=float)
    s = lag_step
    if lpxa_np.shape[0] != lpxb_np.shape[0]:
        raise ValueError(
            f"lpxa and lpxb must have the same length; "
            f"got {lpxa_np.shape[0]} vs {lpxb_np.shape[0]}"
        )
    if lpxa_np.shape[0] <= 2 * s:
        raise ValueError(
            f"Need at least {2 * s + 1} observations to fit AR(lag_step={s}); "
            f"got {lpxa_np.shape[0]}"
        )

    # Residual function on the usable sample (skip first 2*s obs).
    def residuals(theta):
        alpha, beta, gamma = theta
        y_pred = (alpha * (lpxb_np[2 * s:]
                           - beta * lpxb_np[s:-s]
                           - gamma * lpxb_np[:-2 * s])
                  + beta * lpxa_np[s:-s]
                  + gamma * lpxa_np[:-2 * s])
        return lpxa_np[2 * s:] - y_pred

    result = least_squares(
        residuals, x0=list(initial), method="lm",
        xtol=1e-10, ftol=1e-10, gtol=1e-10,
        max_nfev=2000,
    )
    return jnp.asarray(result.x), int(result.nfev)


class _Ar2LoopState(NamedTuple):
    """Loop carry for Fair's AR(2) ρ-iteration.

    Attributes:
      iteration: Iterations completed.
      rho_previous: (2,) ρ at start of last iteration.
      rho_current: (2,) ρ at end of last iteration — order is ``[ρ1, ρ2]``.
      beta: Coefficient vector matching ``rho_current``.
    """
    iteration: jnp.ndarray
    rho_previous: jnp.ndarray
    rho_current: jnp.ndarray
    beta: jnp.ndarray


@partial(jax.jit, static_argnames=("max_iter",))
def two_sls_ar2(
    y: jnp.ndarray,
    X: jnp.ndarray,
    Z: jnp.ndarray,
    y_presample: jnp.ndarray,
    X_presample: jnp.ndarray,
    tol: float = 1e-7,
    max_iter: int = 100,
    damping: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fair's TSLS17 + RHOA generalized to AR(2): iterated 2SLS with AR(2) errors.

    Model: ``y_t = X_t β + u_t`` where ``u_t = ρ1 u_{t-1} + ρ2 u_{t-2} + e_t``.

    Mirrors ``two_sls_ar1`` but with a 2-vector ρ and two pre-sample rows.
    The cumulative update is the natural 2D analog: IV regression of the
    current AR-innovations on their own lag-1 and lag-2 yields (Δρ1, Δρ2).

    Args:
      y: (T,) dependent variable on the estimation window.
      X: (T, k) regressors on the estimation window.
      Z: (T, m) instruments on the estimation window (m ≥ k).
      y_presample: (2,) y at two quarters before the estimation window.
        Order: ``[y[-2], y[-1]]``.
      X_presample: (2, k) X at the same two pre-sample quarters.
      tol: Convergence threshold on ``‖Δρ‖_∞``.
      max_iter: Hard cap on iterations.
      damping: Multiplier on (Δρ1, Δρ2). 1.0 matches Fair.

    Returns:
      ``(beta, rho, se, iterations)`` where ``rho`` is a (2,) vector
      ``[ρ1, ρ2]`` and SEs are asymptotic from the final ρ-transformed fit.
    """
    T = y.shape[0]
    k = X.shape[1]

    # Extended arrays: 2 pre-sample rows + T sample rows.
    y_ext = jnp.concatenate([y_presample, y])              # (T+2,)
    X_ext = jnp.vstack([X_presample, X])                   # (T+2, k)

    ZtZ_inv = jnp.linalg.inv(Z.T @ Z)
    projection_onto_Z = Z @ ZtZ_inv @ Z.T                  # (T, T)
    PZ_X_sample = projection_onto_Z @ X                    # (T, k)
    PZ_X_ext = jnp.vstack([X_presample, PZ_X_sample])      # (T+2, k)

    def fit_given_rhos(rho: jnp.ndarray):
        """Solve for β given current (ρ1, ρ2). Differences both Y and X."""
        YS = y_ext[2:] - rho[0] * y_ext[1:-1] - rho[1] * y_ext[:-2]
        XS = X_ext[2:] - rho[0] * X_ext[1:-1] - rho[1] * X_ext[:-2]
        XSH = PZ_X_ext[2:] - rho[0] * PZ_X_ext[1:-1] - rho[1] * PZ_X_ext[:-2]
        return jnp.linalg.solve(XSH.T @ XS, XSH.T @ YS)

    def rho_update(rho: jnp.ndarray, beta: jnp.ndarray) -> jnp.ndarray:
        """IV regression of current AR-innovations on their lag-1, lag-2."""
        raw_extended = y_ext - X_ext @ beta                # (T+2,)
        # innovation_t = raw_t − ρ1 raw_{t-1} − ρ2 raw_{t-2}, for t ≥ 2.
        # For t=0,1 use raw (no deeper lags available).
        innov = raw_extended.at[2:].set(
            raw_extended[2:] - rho[0] * raw_extended[1:-1]
                             - rho[1] * raw_extended[:-2]
        )
        dep = innov[2:T + 2]                               # (T,)
        lag1 = innov[1:T + 1]
        lag2 = innov[0:T]
        # IV 2SLS: Δρ = (L' P_Z L)^-1 L' P_Z dep where L = [lag1 | lag2].
        L = jnp.column_stack([lag1, lag2])
        PZL = projection_onto_Z @ L
        return jnp.linalg.solve(L.T @ PZL, L.T @ projection_onto_Z @ dep)

    def still_changing(state: _Ar2LoopState) -> jnp.ndarray:
        below_max = state.iteration < max_iter
        delta = jnp.max(jnp.abs(state.rho_current - state.rho_previous))
        return jnp.logical_and(below_max, delta >= tol)

    def one_step(state: _Ar2LoopState) -> _Ar2LoopState:
        beta_new = fit_given_rhos(state.rho_current)
        drho = rho_update(state.rho_current, beta_new)
        return _Ar2LoopState(
            iteration=state.iteration + 1,
            rho_previous=state.rho_current,
            rho_current=state.rho_current + damping * drho,
            beta=beta_new,
        )

    initial_state = _Ar2LoopState(
        iteration=jnp.int32(0),
        rho_previous=jnp.array([-1.0, -1.0], dtype=jnp.float64),
        rho_current=jnp.array([0.0, 0.0], dtype=jnp.float64),
        beta=jnp.zeros(k),
    )
    final = jax.lax.while_loop(still_changing, one_step, initial_state)

    YS = y_ext[2:] - final.rho_current[0] * y_ext[1:-1] - final.rho_current[1] * y_ext[:-2]
    XS = X_ext[2:] - final.rho_current[0] * X_ext[1:-1] - final.rho_current[1] * X_ext[:-2]
    XSH = PZ_X_ext[2:] - final.rho_current[0] * PZ_X_ext[1:-1] - final.rho_current[1] * PZ_X_ext[:-2]
    residuals = YS - XS @ final.beta
    sigma_squared = (residuals @ residuals) / (T - 2)
    var_beta = sigma_squared * jnp.linalg.inv(XSH.T @ XS)
    se = jnp.sqrt(jnp.abs(jnp.diag(var_beta)))
    return final.beta, final.rho_current, se, final.iteration


@partial(jax.jit, static_argnames=("max_iter",))
def two_sls_ar1(
    y: jnp.ndarray,
    X: jnp.ndarray,
    Z: jnp.ndarray,
    y_presample: jnp.ndarray,
    X_presample: jnp.ndarray,
    tol: float = 1e-7,
    max_iter: int = 100,
    damping: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fair's TSLS17 + RHOA: iterated 2SLS with AR(1) errors.

    See module docstring for the algorithm. Matches Fair's iter-1 value to
    ~1e-7 and converges to within ~1.6e-3 of Fair's final ρ on the IS model.

    Args:
      y: (T,) dependent variable, estimation sample.
      X: (T, k) regressors, estimation sample.
      Z: (T, m) instruments, estimation sample.
      y_presample: scalar — y one period before y[0], to lag the first sample obs.
      X_presample: (k,) — X one period before X[0].
      tol: Convergence threshold on |Δρ|.
      max_iter: Iteration cap.
      damping: Multiplier on the ρ-correction (0 < damping ≤ 1). The default 1.0
        applies the full Fair update. For near-unit-root equations the full
        update can overshoot and diverge; set ``damping=0.3`` or ``0.5`` to
        stabilize at the cost of slower convergence.

    Returns:
      ``(beta, rho, se, iterations)`` — coefficients, scalar AR parameter,
      asymptotic standard errors, and iterations used.
    """
    T = y.shape[0]
    k = X.shape[1]

    # Extended arrays: index 0 = pre-sample row, index 1..T = estimation sample.
    y_ext = jnp.concatenate([jnp.array([y_presample]), y])   # (T+1,)
    X_ext = jnp.vstack([X_presample[None, :], X])            # (T+1, k)

    # Full-sample first-stage projection, computed ONCE.
    ZtZ_inv = jnp.linalg.inv(Z.T @ Z)
    projection_onto_Z = Z @ ZtZ_inv @ Z.T                    # (T, T)
    PZ_X_sample = projection_onto_Z @ X                      # (T, k)
    # Extended P_Z·X: use raw X_presample for the pre-sample row (no instruments
    # exist outside the estimation window — Fair's convention).
    PZ_X_ext = jnp.vstack([X_presample[None, :], PZ_X_sample])

    def fit_given_rho(rho):
        """Build the ρ-differenced system and solve for β."""
        YS = y_ext[1:] - rho * y_ext[:-1]           # (T,)
        XS = X_ext[1:] - rho * X_ext[:-1]           # (T, k)
        XSH = PZ_X_ext[1:] - rho * PZ_X_ext[:-1]    # (T, k)
        beta = jnp.linalg.solve(XSH.T @ XS, XSH.T @ YS)
        return beta

    def rho_correction(rho, beta):
        """Fair's RHOA update — IV regression of U on U_lag."""
        raw_extended = y_ext - X_ext @ beta                          # (T+1,)
        # Innovation at t: y_t − X_t β − ρ (y_{t-1} − X_{t-1} β).
        # For t=0 (pre-sample), use raw residual (no further lag available).
        innovation_extended = raw_extended.at[1:].set(
            raw_extended[1:] - rho * raw_extended[:-1]
        )
        lag_of_innovation = innovation_extended[:T]        # (T,)
        innovation_current = innovation_extended[1:T + 1]  # (T,)
        projected_lag = projection_onto_Z @ lag_of_innovation
        numerator = lag_of_innovation @ projection_onto_Z @ innovation_current
        denominator = lag_of_innovation @ projected_lag
        return numerator / denominator

    def still_changing(state: _Ar1LoopState) -> jnp.ndarray:
        below_max = state.iteration < max_iter
        not_converged = jnp.abs(state.rho_current - state.rho_previous) >= tol
        return jnp.logical_and(below_max, not_converged)

    def one_step(state: _Ar1LoopState) -> _Ar1LoopState:
        beta_new = fit_given_rho(state.rho_current)
        delta_rho = rho_correction(state.rho_current, beta_new)
        rho_next = state.rho_current + damping * delta_rho
        return _Ar1LoopState(
            iteration=state.iteration + 1,
            rho_previous=state.rho_current,
            rho_current=rho_next,
            beta=beta_new,
        )

    # Seed with rho_previous != rho_current so the first comparison triggers
    # at least one iteration.
    initial_state = _Ar1LoopState(
        iteration=jnp.int32(0),
        rho_previous=jnp.float64(-1.0),
        rho_current=jnp.float64(0.0),
        beta=jnp.zeros(k),
    )
    final = jax.lax.while_loop(still_changing, one_step, initial_state)

    # Asymptotic SEs from the final ρ-transformed system.
    YS = y_ext[1:] - final.rho_current * y_ext[:-1]
    XS = X_ext[1:] - final.rho_current * X_ext[:-1]
    XSH = PZ_X_ext[1:] - final.rho_current * PZ_X_ext[:-1]
    residuals = YS - XS @ final.beta
    sigma_squared = (residuals @ residuals) / (T - 1)
    var_beta = sigma_squared * jnp.linalg.inv(XSH.T @ XS)
    se = jnp.sqrt(jnp.abs(jnp.diag(var_beta)))
    return final.beta, final.rho_current, se, final.iteration


# ---------------------------------------------------------------------------
# Data preparation (polars I/O -> jnp arrays)
# ---------------------------------------------------------------------------

_LOG_SERIES = ["C", "I", "Y", "G"]
_LAGGED_SERIES = ["LOGC", "LOGI", "LOGY", "LOGG", "R"]

_REQUIRED_COLS = [
    "LOGC", "LOGC_lag1", "LOGC_lag2",
    "LOGI", "LOGI_lag1",
    "LOGY", "LOGY_lag1",
    "LOGG",
    "R", "R_lag1",
]

_INSTRUMENT_COLUMNS = [
    "CNST",
    "LOGC_lag1", "LOGC_lag2",
    "LOGY_lag1",
    "R_lag1",
    "LOGI_lag1",
    "LOGG",
]

_EQ1_REGRESSORS = ["CNST", "LOGC_lag1", "LOGY", "R"]
_EQ2_REGRESSORS = ["CNST", "LOGI_lag1", "LOGY", "R_lag1"]

# Fair's estimation sample plus one pre-sample quarter for the AR(1) lag.
_PRESAMPLE_PERIOD = "1953Q4"
_ESTIMATION_START = "1954Q1"
_ESTIMATION_END = "2013Q3"


def _build_regression_frame(data: pl.DataFrame) -> pl.DataFrame:
    """Add log-transforms and lag columns required by the IS equations."""
    df = data.sort("period").with_columns(
        [pl.col(s).log().alias(f"LOG{s}") for s in _LOG_SERIES]
    )
    lag_one = [pl.col(c).shift(1).alias(f"{c}_lag1") for c in _LAGGED_SERIES]
    lag_two = [pl.col(c).shift(2).alias(f"{c}_lag2") for c in _LAGGED_SERIES]
    return df.with_columns(lag_one + lag_two)


def _split_presample_and_estimation(
    df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return ``(presample_row, estimation_frame)``.

    The estimation frame spans ``_ESTIMATION_START`` to ``_ESTIMATION_END``.
    The pre-sample row is the quarter immediately before the start — needed
    to supply the first lag for the AR(1) transformation without losing obs.
    """
    window = df.filter(
        (pl.col("period") >= pl.lit(_PRESAMPLE_PERIOD))
        & (pl.col("period") <= pl.lit(_ESTIMATION_END))
    ).drop_nulls(subset=_REQUIRED_COLS)
    presample = window.head(1)
    estimation = window.tail(window.height - 1)
    return presample, estimation


def _stack_columns(df: pl.DataFrame, columns: list[str]) -> jnp.ndarray:
    """Stack named polars columns into a ``(T, len(columns))`` jnp array.

    The synthetic ``"CNST"`` column materializes as all-ones so callers can
    place a constant term anywhere in ``columns`` without juggling a vector.
    """
    n_rows = df.height
    vectors = [
        jnp.ones(n_rows) if col == "CNST"
        else jnp.asarray(df[col].to_numpy())
        for col in columns
    ]
    return jnp.column_stack(vectors)


def _stack_row(df: pl.DataFrame, columns: list[str]) -> jnp.ndarray:
    """Single-row version of ``_stack_columns`` — returns a (k,) vector."""
    return jnp.array([
        1.0 if col == "CNST" else float(df[col].item())
        for col in columns
    ])


# ---------------------------------------------------------------------------
# Step entry point
# ---------------------------------------------------------------------------

_PARAM_ORDER = [
    "eq1_const", "eq1_C_lag1", "eq1_Y", "eq1_R", "eq1_rho",
    "eq2_const", "eq2_I_lag1", "eq2_Y", "eq2_R_lag1",
]


def run(
    model: str,
    data: pl.DataFrame,
    data_source: str = "fred",
    force: bool = False,
) -> dict[str, float]:
    """Estimate coefficients for ``model`` and cache to parquet.

    See module docstring for the algorithm.

    Args:
      model: Model name. Only ``"is"`` is supported in v0.1.
      data: Step01 output — wide frame with one column per raw series.
      data_source: Cache-key label (``"fred"`` or ``"fair_2013"``).
      force: If True, ignore any existing cache and re-estimate.

    Returns:
      Dict of named coefficients (Python floats, serializable).

    Raises:
      NotImplementedError: if ``model`` is not ``"is"``.
    """
    cache_path = config.OUTPUT_DIR / f"step02_params_{model}_{data_source}.parquet"
    if cache_path.exists() and not force:
        df = pl.read_parquet(cache_path)
        return dict(zip(df["name"].to_list(), df["value"].to_list()))

    if model != "is":
        raise NotImplementedError(
            f"step02: only 'is' supported in v0.1, got {model!r}"
        )

    regression_frame = _build_regression_frame(data)
    presample, estimation = _split_presample_and_estimation(regression_frame)

    instruments = _stack_columns(estimation, _INSTRUMENT_COLUMNS)

    # EQ 1 — consumption, AR(1).
    eq1_y = jnp.asarray(estimation["LOGC"].to_numpy())
    eq1_X = _stack_columns(estimation, _EQ1_REGRESSORS)
    eq1_y_presample = jnp.asarray(float(presample["LOGC"].item()))
    eq1_X_presample = _stack_row(presample, _EQ1_REGRESSORS)
    eq1_beta, eq1_rho, _se1, eq1_iters = two_sls_ar1(
        eq1_y, eq1_X, instruments, eq1_y_presample, eq1_X_presample,
    )

    # EQ 2 — investment, no AR.
    eq2_y = jnp.asarray(estimation["LOGI"].to_numpy())
    eq2_X = _stack_columns(estimation, _EQ2_REGRESSORS)
    eq2_beta, _se2 = two_sls_with_se(eq2_y, eq2_X, instruments)

    params = {
        "eq1_const":  float(eq1_beta[0]),
        "eq1_C_lag1": float(eq1_beta[1]),
        "eq1_Y":      float(eq1_beta[2]),
        "eq1_R":      float(eq1_beta[3]),
        "eq1_rho":    float(eq1_rho),
        "eq2_const":  float(eq2_beta[0]),
        "eq2_I_lag1": float(eq2_beta[1]),
        "eq2_Y":      float(eq2_beta[2]),
        "eq2_R_lag1": float(eq2_beta[3]),
    }
    params = {name: params[name] for name in _PARAM_ORDER}

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"name": list(params.keys()), "value": list(params.values())}
    ).write_parquet(cache_path)
    print(f"[step02] n_obs={estimation.height}  rho_iters={int(eq1_iters)}  "
          f"rho={float(eq1_rho):.9f}")
    return params
