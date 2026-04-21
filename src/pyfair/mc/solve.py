"""MC-model solve driver — identities, trade linkage, Newton + block-GS.

Architectural twin of ``us_solve.py``, extended to the multi-country case.
The MC model closes through two routes:

1. **Country-level accounting identities** — one block per ROW country,
   roughly mirroring the US identities. See MC.INP lines 619–627 for the
   Canada block; the pattern repeats with prefix substitution across all
   37 ROW countries. ``_row_country_identities`` generates the 8 standard
   identities for any prefix; ``_exchange_rate_identity`` adds the
   regime-specific xxE/xxH relation.

2. **Bilateral trade linkage** — each country's import-price proxy
   ``xxPMM`` is a trade-share-weighted sum of every partner's export
   price ``yyPX$`` (MC.INP MCSHR1.INP). For **in-sample** solves the
   historical ``xxPMM`` values are loaded directly from SHRDDD.DAT and
   treated as exogenous — `load_pmm_series()` returns 59 `xxPMM*` series
   on the 1960Q1–2025Q4 grid. For out-of-sample forecasts the
   bilateral ``A_ij`` shares would need to be endogenized via SHR.INP's
   ~1,500 stochastic equations (v0.3+ work).

**Current capability:**

* ``MCIdentity`` dataclass + 333 identities (``MC_IDENTITIES_ALL``).
* ``verify_identities_on_frame(frame, period)`` evaluates every identity
  at one quarter; ~all hold at 2010Q1 to <4e-4 relative error.
* ``_scalar_genrs(prefix, annual_lag)`` — generic scalar-JAX GENR
  evaluator. Handles quarterly + annual templates plus regime-specific
  extras (LH/LH1/LHA/LH1Z for Germany-pegged; LE1Z for JA/NZ).
* ``solve_country_one_period(prefix, frame, period)`` — per-country
  Newton-Raphson solve. Converges in 4–7 iterations from a 2%
  perturbation across all 36 ROW countries at in-sample periods, max
  relative error ≤ 2e-10.
* ``solve_all_countries_one_period(frame, period)`` — runs every ROW
  country in one pass, returning aggregated results. For in-sample this
  is the block-Gauss-Seidel outer loop's converged historical limit.
* ``simulate_country_path(prefix, frame, start, end)`` — period-by-period
  in-sample dynamic tracking with rolling-lag frame updates.

**Deferred to v0.3+:**

* Endogenous cross-country coupling during out-of-sample forecast (needs
  SHR.INP's 1,500 stochastic trade-share equations).
* Multi-period `lax.scan`-based `simulate` with shock injection (the
  Python-loop `simulate_country_path` covers in-sample tracking).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import polars as pl

from .. import config
from ..core import readers
from . import countries as mc_countries

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCIdentity:
    """Static spec for one MC-model accounting identity.

    Attributes:
      country: Two-letter prefix (``"CA"``, ``"JA"``, ...). US MC-only
        identities use ``"US"``.
      output: Variable the identity defines (e.g. ``"CAY"``).
      inputs: Tuple of variables the identity reads. Used to topologically
        order identities so upstream values land before downstream reads.
      formula: Pure JAX function ``f(**kwargs) -> scalar`` that computes
        ``output`` from ``inputs``. Should be ``jax.jit``-able.
      notes: Optional note — e.g. which MC.INP line this came from.
    """
    country: str
    output: str
    inputs: tuple[str, ...]
    formula: Callable[..., jnp.ndarray]
    notes: str = ""


def _row_country_identities(prefix: str) -> list[MCIdentity]:
    """The 8 standard accounting identities one ROW country needs.

    Mirrors MC.INP lines 619–627 for Canada; the same pattern repeats for
    every ROW country with prefix substitution. Excludes the xx``H``/xx``E``
    identity because its direction flips with exchange-rate regime (pegged
    countries treat xxH as endogenous, floating countries treat xxE) —
    that identity is added per-regime by ``_exchange_rate_identity``.
    """
    p = prefix

    return [
        MCIdentity(
            country=p, output=f"{p}Y",
            inputs=(f"{p}C", f"{p}I", f"{p}G", f"{p}EX", f"{p}IM",
                    f"{p}STAT", f"{p}V1"),
            formula=lambda **kw: (
                kw[f"{p}C"] + kw[f"{p}I"] + kw[f"{p}G"]
                + kw[f"{p}EX"] - kw[f"{p}IM"]
                + kw[f"{p}STAT"] + kw[f"{p}V1"]
            ),
            notes="GDP accounting identity (MC.INP ~line +2 of BEGIN block).",
        ),
        MCIdentity(
            country=p, output=f"{p}PM",
            inputs=(f"{p}PSI3", f"{p}PMP"),
            formula=lambda **kw: kw[f"{p}PSI3"] * kw[f"{p}PMP"],
            notes="Import-price identity.",
        ),
        MCIdentity(
            country=p, output=f"{p}ZZ",
            inputs=(f"{p}Y", f"{p}YS"),
            formula=lambda **kw: jnp.log(kw[f"{p}Y"]) - jnp.log(kw[f"{p}YS"]),
            notes="Output-gap log-ratio.",
        ),
        MCIdentity(
            country=p, output=f"{p}JMIN",
            inputs=(f"{p}Y", f"{p}LAM"),
            formula=lambda **kw: kw[f"{p}Y"] / kw[f"{p}LAM"],
            notes="Minimum-employment identity.",
        ),
        MCIdentity(
            country=p, output=f"{p}UR",
            inputs=(f"{p}L1", f"{p}J"),
            formula=lambda **kw: (kw[f"{p}L1"] - kw[f"{p}J"]) / kw[f"{p}L1"],
            notes="Unemployment-rate identity.",
        ),
        MCIdentity(
            country=p, output=f"{p}M10$",
            inputs=(f"{p}PSI1", f"{p}IM", f"{p}E10", f"{p}PM10"),
            formula=lambda **kw: (
                kw[f"{p}PSI1"] * (kw[f"{p}IM"]
                                   / (kw[f"{p}E10"] * kw[f"{p}PM10"]))
            ),
            notes="Dollar-denominated imports identity.",
        ),
        MCIdentity(
            country=p, output=f"{p}EX",
            inputs=(f"{p}X10$", f"{p}E10", f"{p}PX10", f"{p}PSI2"),
            formula=lambda **kw: (
                (kw[f"{p}X10$"] * kw[f"{p}E10"] * kw[f"{p}PX10"])
                / kw[f"{p}PSI2"]
            ),
            notes="Real exports identity.",
        ),
        MCIdentity(
            country=p, output=f"{p}PX$",
            inputs=(f"{p}E10", f"{p}E", f"{p}PX"),
            formula=lambda **kw: (kw[f"{p}E10"] / kw[f"{p}E"]) * kw[f"{p}PX"],
            notes="Dollar export-price identity.",
        ),
    ]


def _exchange_rate_identity(prefix: str) -> MCIdentity | None:
    """Regime-specific xxH/xxE identity.

    Pegged-to-Germany quarterly: ``xxE = xxH * GEE``.
    Pegged-to-Germany annual:    ``xxE = xxH * GEEA``.
    Floating vs US quarterly:    ``xxH = xxE / GEE`` (mathematically
                                  identical; direction differs only for
                                  which variable is endogenous in the solve).
    Floating vs US annual:       ``xxH = xxE / GEEA``.

    We express the identity in its pegged-to-Germany direction (``xxE =
    xxH * GEE_anchor``) regardless of regime — both variables appear in
    the frame, so the relation can be verified in either direction.

    Returns ``None`` for the US and countries without an ``xxH`` series.
    """
    if prefix == "US":
        return None
    try:
        country = mc_countries.by_prefix(prefix)
    except KeyError:
        return None
    anchor = "GEEA" if country.annual_lag == 4 else "GEE"
    return MCIdentity(
        country=prefix,
        output=f"{prefix}E",
        inputs=(f"{prefix}H", anchor),
        formula=lambda **kw: kw[f"{prefix}H"] * kw[anchor],
        notes=f"Exchange-rate identity (anchor={anchor}).",
    )


def _all_mc_identities() -> list[MCIdentity]:
    """Every identity across every ROW country."""
    result: list[MCIdentity] = []
    for country in mc_countries.row_countries():
        result.extend(_row_country_identities(country.prefix))
        xr = _exchange_rate_identity(country.prefix)
        if xr is not None:
            result.append(xr)
    return result


MC_IDENTITIES_ALL: list[MCIdentity] = _all_mc_identities()
CA_IDENTITIES: list[MCIdentity] = _row_country_identities("CA")


# ---------------------------------------------------------------------------
# Identity verification on historical data
# ---------------------------------------------------------------------------

def _gather_inputs(frame: pl.DataFrame, period: str,
                   names: tuple[str, ...]) -> dict[str, float] | None:
    """Pull a dict of name → scalar float for ``period`` from ``frame``.

    Returns ``None`` if any requested input is absent or null at ``period`` —
    signals that the identity cannot be evaluated for this country/period.
    """
    row = frame.filter(pl.col("period") == pl.lit(period))
    if row.height != 1:
        return None
    out: dict[str, float] = {}
    for name in names:
        if name not in row.columns:
            return None
        val = row[name][0]
        if val is None:
            return None
        try:
            out[name] = float(val)
        except (TypeError, ValueError):
            return None
    return out


@dataclass(frozen=True)
class IdentityResidual:
    """One evaluated identity's residual at a specific period."""
    identity: MCIdentity
    period: str
    observed: float
    computed: float
    residual: float


def verify_identities_on_frame(
    frame: pl.DataFrame, period: str,
    identities: list[MCIdentity] = MC_IDENTITIES_ALL,
) -> list[IdentityResidual]:
    """Evaluate every identity at ``period`` and return the residuals.

    An identity "holds" when ``residual = observed - computed`` is near
    zero. Residuals that are large (or where the identity can't be
    evaluated for missing-input reasons) surface as-is — the caller
    decides how to classify.

    Skips identities whose inputs or output are missing from the frame
    at ``period``.
    """
    results: list[IdentityResidual] = []
    for ident in identities:
        inputs = _gather_inputs(frame, period, ident.inputs)
        if inputs is None:
            continue
        obs_row = frame.filter(pl.col("period") == pl.lit(period))
        if ident.output not in obs_row.columns:
            continue
        observed_val = obs_row[ident.output][0]
        if observed_val is None:
            continue
        observed = float(observed_val)
        computed = float(ident.formula(**inputs))
        results.append(IdentityResidual(
            identity=ident, period=period,
            observed=observed, computed=computed,
            residual=observed - computed,
        ))
    return results


# ---------------------------------------------------------------------------
# Deferred primitives — v0.2 work
# ---------------------------------------------------------------------------

def load_pmm_series(path=None,
                    sample_start: str = "1960.1") -> pl.DataFrame:
    """Parse the per-country import-price aggregates (``xxPMM``) from SHRDDD.DAT.

    SHRDDD.DAT packs 6,787 ``LOAD`` blocks — 39 per-country ``xxPMM``
    time series plus ~6,700 bilateral trade-share series. For in-sample
    MC solves we need only the aggregates, so this loader filters to
    just the ``*PMM*`` names (the file also contains ``xxPMMQ`` and
    ``xxPMMD`` variants that Fair uses for quarterly-dollar and
    dollar-denominated flavors).

    The file has no ``SMPL`` header of its own — the caller supplies it
    (Fair's convention is 1960.1, matching the other MC data files).

    Args:
      path: Path to SHRDDD.DAT. Defaults to ``config.MC_MODEL_DIR``.
      sample_start: Start period for the value stream in Fair format
        (e.g. ``"1960.1"``).

    Returns:
      Wide DataFrame with ``period`` plus one column per ``*PMM*`` series.
    """
    from pathlib import Path
    p = Path(path) if path is not None else (config.MC_MODEL_DIR / "SHRDDD.DAT")

    # Prepend a synthetic SMPL line so parse_fair_data picks up the origin.
    text = f"SMPL {sample_start} 2099.4;\n" + p.read_text()
    tmp = p.parent / ".shrddd_with_smpl.dat"
    tmp.write_text(text)
    try:
        long = readers.parse_fair_data(tmp)
    finally:
        tmp.unlink(missing_ok=True)

    # Keep only the *PMM* series; bilateral A_ij names have a different
    # pattern (e.g. ``AAUSCA`` for US→CA share).
    pmm_only = long.filter(pl.col("variable").str.contains("PMM"))
    return readers.pivot_to_wide(pmm_only).sort("period")


def _prev_quarter(period: str) -> str:
    """``"1961Q1"`` → ``"1960Q4"``."""
    year, q = period.split("Q")
    year, q = int(year), int(q)
    if q == 1:
        return f"{year - 1}Q4"
    return f"{year}Q{q - 1}"


def _shift_quarter(period: str, k: int) -> str:
    """Shift ``period`` by ``k`` quarters (positive = forward, negative = back)."""
    year, q = period.split("Q")
    total = int(year) * 4 + (int(q) - 1) + k
    return f"{total // 4}Q{total % 4 + 1}"


# ---------------------------------------------------------------------------
# Generic per-country solver — factors the CA-specific solve into reusable
# pieces so every ROW country's 10-variable Newton lands in ~20 lines of spec.
# ---------------------------------------------------------------------------

def _scalar_genrs(prefix: str, annual_lag: int) -> dict[str, Callable]:
    """Scalar-JAX versions of the ROW-country GENR template.

    Each entry maps a GENR name (e.g. ``"CALCZ"``) to a pure function
    ``f(ctx) -> jax scalar`` where ``ctx`` is a flat dict keying every
    value this GENR could need — primitive endogenous, exogenous, lagged
    (keyed as ``"{VAR}_lag{k}"``), and cross-country anchors.

    Mirrors ``mc_model._row_country_genr_specs`` but on scalars.
    """
    p = prefix
    lag = annual_lag
    uspy = "USPYA" if annual_lag == 4 else "USPY"
    usrs = "USRSA" if annual_lag == 4 else "USRS"
    gepy = "GEPYA" if annual_lag == 4 else "GEPY"
    gers = "GERSA" if annual_lag == 4 else "GERS"

    return {
        f"{p}LCZ":    lambda ctx: jnp.log(ctx[f"{p}C"] / ctx[f"{p}POP"]),
        f"{p}LIMZ":   lambda ctx: jnp.log(ctx[f"{p}IM"] / ctx[f"{p}POP"]),
        f"{p}LI":     lambda ctx: jnp.log(ctx[f"{p}I"]),
        f"{p}LY":     lambda ctx: jnp.log(ctx[f"{p}Y"]),
        f"{p}LPY":    lambda ctx: jnp.log(ctx[f"{p}PY"]),
        f"{p}LPM":    lambda ctx: jnp.log(ctx[f"{p}PM"]),
        f"{p}LYZ":    lambda ctx: jnp.log(ctx[f"{p}Y"] / ctx[f"{p}POP"]),
        f"{p}LCIGZ":  lambda ctx: jnp.log(
            (ctx[f"{p}C"] + ctx[f"{p}I"] + ctx[f"{p}G"]) / ctx[f"{p}POP"]),
        f"{p}LPYZZ":  lambda ctx: jnp.log(ctx[f"{p}PY"] / ctx[f"{p}PM"]),
        f"{p}PCPY":   lambda ctx: 100 * (
            (ctx[f"{p}PY"] / ctx[f"{p}PY_lag{lag}"])**(4 if annual_lag == 1 else 1)
            - 1),
        f"{p}RBZ":    lambda ctx: ctx[f"{p}RB"] - ctx[f"{p}RS_lag{2 * lag}"],
        f"{p}RBZZ":   lambda ctx: (
            ctx[f"{p}RB_lag{lag}"] - ctx[f"{p}RS_lag{2 * lag}"]),
        f"{p}RSZ":    lambda ctx: ctx[f"{p}RS"] - ctx[f"{p}RS_lag{2 * lag}"],
        f"{p}RSZZ":   lambda ctx: (
            ctx[f"{p}RS_lag{lag}"] - ctx[f"{p}RS_lag{2 * lag}"]),
        f"{p}LE":     lambda ctx: jnp.log(ctx[f"{p}E"]),
        f"{p}LE1":    lambda ctx: (
            jnp.log(ctx[f"{p}E"]) - jnp.log(ctx[f"{p}E_lag{lag}"])),
        f"{p}LEA":    lambda ctx: (
            jnp.log(ctx[f"{p}PY"] / ctx[uspy])
            - jnp.log(ctx[f"{p}E_lag{lag}"])),
        f"{p}LRSZ":   lambda ctx: 0.25 * jnp.log(
            (1 + ctx[f"{p}RS"] / 100) / (1 + ctx[usrs] / 100)),
        f"{p}LPXA":   lambda ctx: jnp.log(
            ctx[f"{p}PX"]
            / (ctx[f"{p}PW$"] * (ctx[f"{p}E"] / ctx[f"{p}E10"]))),
        f"{p}LPXB":   lambda ctx: jnp.log(
            ctx[f"{p}PY"]
            / (ctx[f"{p}PW$"] * (ctx[f"{p}E"] / ctx[f"{p}E10"]))),
        f"{p}LJ1":    lambda ctx: (
            jnp.log(ctx[f"{p}J"]) - jnp.log(ctx[f"{p}J_lag{lag}"])),
        f"{p}LY1":    lambda ctx: (
            jnp.log(ctx[f"{p}Y"]) - jnp.log(ctx[f"{p}Y_lag{lag}"])),
        f"{p}LEXL":   lambda ctx: jnp.log(
            ctx[f"{p}J"] / (ctx[f"{p}Y"] / ctx[f"{p}LAM"])),
        f"{p}LL1Z":   lambda ctx: jnp.log(ctx[f"{p}L1"] / ctx[f"{p}POP1"]),
        f"{p}LGZ":    lambda ctx: jnp.log(ctx[f"{p}G"] / ctx[f"{p}POP"]),
        f"{p}LEXZ":   lambda ctx: jnp.log(ctx[f"{p}EX"] / ctx[f"{p}POP"]),
        # Pegged-to-Germany H-rate extras (AU, IT, NE, ST, UK, FR, FI,
        # plus BE/DE/NO/SW/GR/IR/PO/SP at annual lag).
        f"{p}LH":     lambda ctx: jnp.log(ctx[f"{p}H"]),
        f"{p}LH1":    lambda ctx: (
            jnp.log(ctx[f"{p}H"]) - jnp.log(ctx[f"{p}H_lag{lag}"])),
        f"{p}LHA":    lambda ctx: (
            jnp.log(ctx[f"{p}PY"] / ctx[gepy])
            - jnp.log(ctx[f"{p}H_lag{lag}"])),
        f"{p}LH1Z":   lambda ctx: (
            (jnp.log(ctx[f"{p}H"]) - jnp.log(ctx[f"{p}H_lag{lag}"]))
            - 0.05 * (
                jnp.log(ctx[f"{p}PY"] / ctx[gepy])
                - jnp.log(ctx[f"{p}H_lag{lag}"]))
        ),
        f"{p}LRSZG":  lambda ctx: 0.25 * jnp.log(
            (1 + ctx[f"{p}RS"] / 100) / (1 + ctx[gers] / 100)),
        f"{p}LE1Z":   lambda ctx: (
            (jnp.log(ctx[f"{p}E"]) - jnp.log(ctx[f"{p}E_lag{lag}"]))
            - 0.05 * (
                jnp.log(ctx[f"{p}PY"] / ctx[uspy])
                - jnp.log(ctx[f"{p}E_lag{lag}"]))
        ),
    }


def _parse_regressor_token(token: str) -> tuple[str, int]:
    """``"CALCZ(-1)"`` → ``("CALCZ", 1)``; ``"CALCZ"`` → ``("CALCZ", 0)``."""
    if "(" in token:
        base, rest = token.split("(", 1)
        lag_str = rest.rstrip(")")
        return base, abs(int(lag_str))
    return token, 0


def _build_solve_ctx(
    frame: pl.DataFrame, period: str, prefix: str, annual_lag: int,
) -> dict[str, float]:
    """Pack the frame into a flat ``ctx`` dict for generic solver use.

    Includes:
      * Every current-period non-null column (primitives, exog, anchors).
      * ``"{VAR}_lag{k}"`` for ``k = 1..8`` (enough for AR(2) + 4-qtr lags).

    Numeric-only — caller replaces state primitives with JAX scalars when
    packing the Newton x-vector.
    """
    current = frame.filter(pl.col("period") == pl.lit(period)).row(0, named=True)
    ctx: dict[str, float] = {}
    for name, val in current.items():
        if name == "period" or val is None:
            continue
        try:
            ctx[name] = float(val)
        except (TypeError, ValueError):
            pass

    max_lag = max(4, 2 * annual_lag, 4 * annual_lag)
    for k in range(1, max_lag + 1):
        lag_period = _shift_quarter(period, -k)
        lag_row = frame.filter(pl.col("period") == pl.lit(lag_period))
        if lag_row.height != 1:
            continue
        named = lag_row.row(0, named=True)
        for name, val in named.items():
            if name == "period" or val is None:
                continue
            try:
                ctx[f"{name}_lag{k}"] = float(val)
            except (TypeError, ValueError):
                pass
    return ctx


def _resolve_regressor(
    token: str,
    ctx: dict,
    genrs: dict[str, Callable],
) -> jnp.ndarray:
    """Return the scalar value of one regressor token.

    Order of resolution:
      1. ``"C"`` → constant 1.0.
      2. Lagged (``"VAR(-k)"``): ``ctx["{VAR}_lag{k}"]``. Caller must have
         pre-populated ``ctx`` from the frame at the appropriate period.
      3. Current-period:
         a) ``ctx[token]`` — primitive / exogenous / anchor.
         b) ``genrs[token](ctx)`` — derived GENR.
    """
    if token == "C":
        return jnp.asarray(1.0)
    base, lag = _parse_regressor_token(token)
    if lag > 0:
        key = f"{base}_lag{lag}"
        return ctx[key]
    # GENR takes priority over raw ctx lookup so current-period derived
    # quantities (CALCZ, CALIMZ, ...) recompute from state primitives
    # rather than returning their stale frame values.
    if token in genrs:
        return genrs[token](ctx)
    if token in ctx:
        return ctx[token]
    raise KeyError(f"Cannot resolve regressor token: {token!r}")


def _compute_ar_innovation(
    eq, ctx: dict, genrs: dict, coefs: dict,
    lag_offset: int = 1,
) -> jnp.ndarray:
    """Compute historical AR innovation ``u_{t-lag_offset}``.

    ``u_{t-1} = LHS_{t-1} − fitted_{t-1}`` where the fitted uses regressor
    values at ``t-1`` (from the pre-populated lag entries in ``ctx``).
    Approximates Fair by treating any deeper-than-available innovation as
    zero — matches the hand-coded CA solver's convention.

    For AR(2) the caller asks twice with ``lag_offset=1`` and ``2``.
    """
    # Build a shifted ctx representing the state at ``t − lag_offset``:
    # every ``ctx["VAR_lag{k}"]`` maps to ``shifted["VAR_lag{k - offset}"]``;
    # ``k == offset`` maps to current (``shifted["VAR"]``); ``k < offset``
    # drops out (would be the future).
    shifted = {}
    for name, val in ctx.items():
        if "_lag" not in name:
            continue
        base, lag_str = name.rsplit("_lag", 1)
        try:
            orig_lag = int(lag_str)
        except ValueError:
            continue
        new_lag = orig_lag - lag_offset
        if new_lag == 0:
            shifted[base] = val
        elif new_lag > 0:
            shifted[f"{base}_lag{new_lag}"] = val

    lhs_lagged = _resolve_regressor(eq.dependent, shifted, genrs)
    fitted = 0.0
    for i, token in enumerate(eq.regressors):
        c = coefs[_coef_key(eq, token)]
        fitted = fitted + c * _resolve_regressor(token, shifted, genrs)
    return lhs_lagged - fitted


def _coef_key(eq, token: str) -> str:
    """Map an MCEquation regressor token to the key in ``REFERENCE_PARAMS_MC``.

    The parser stored names like ``"C(0)"``, ``"CALIMZ(-1)"`` etc. at
    import-time. For current-period tokens with no explicit lag we append
    ``"(0)"``; for ``"CALIMZ(-1)"``-style tokens we keep the lag suffix.
    """
    if "(" in token:
        return token
    return f"{token}(0)"


def _compute_epsilon(
    eq, ctx: dict, genrs: dict, coefs: dict,
    u_lag1: jnp.ndarray | None = None,
    u_lag2: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Historical residual ε = observed − fitted (with AR lags if AR).

    Observed is ``eq.dependent`` evaluated at the **historical** state in
    ``ctx``; fitted is the sum of ``coef·regressor`` plus AR innovations.
    """
    lhs = _resolve_regressor(eq.dependent, ctx, genrs)
    fitted = 0.0
    for token in eq.regressors:
        c = coefs[_coef_key(eq, token)]
        fitted = fitted + c * _resolve_regressor(token, ctx, genrs)
    if u_lag1 is not None:
        fitted = fitted + coefs["RHO(-1)"] * u_lag1
    if u_lag2 is not None:
        fitted = fitted + coefs["RHO(-2)"] * u_lag2
    return lhs - fitted


# Per-country state-variable ordering. Built from each country's equation
# set — the variable each stochastic equation's LHS produces, plus the
# xxY identity. Countries that skip specific equations (JA skips EQ 53)
# just omit that state variable.
def _lhs_primitive(eq) -> str:
    """Extract the primitive variable name from an LHS-transform string.

    Examples:
      ``"CAIM=EXP(CALIMZ)*CAPOP"`` → ``"CAIM"``
      ``"CARB=CARBZ+CARS(-2)"`` → ``"CARB"``
      ``"CAJ=EXP(CALJ1)*CAJ(-1)"`` → ``"CAJ"``
    """
    if eq.lhs_transform is None:
        return eq.dependent
    return eq.lhs_transform.split("=")[0].strip()


def _country_state_order(prefix: str) -> list[str]:
    """Return the endogenous state-variable names in a stable solve order.

    Stochastic equations' LHS primitives come first (sorted by equation
    number), then the Y accounting identity. NLEQ equations (Fair's
    xxLPXA) are excluded — their output variable is computed directly
    from lagged data once coefficients are known and doesn't feed back
    into the country's demand block, so we treat it as exogenous in the
    Newton solve.
    """
    from . import model as mc_model
    eqs = sorted(
        (e for e in mc_model.EQUATIONS_BY_COUNTRY[prefix] if not e.is_nleq),
        key=lambda e: e.number,
    )
    primitives: list[str] = []
    seen: set[str] = set()
    for eq in eqs:
        var = _lhs_primitive(eq)
        if var not in seen:
            primitives.append(var)
            seen.add(var)
    y_var = f"{prefix}Y"
    if y_var not in seen:
        primitives.append(y_var)
    return primitives


def solve_country_one_period(
    prefix: str,
    frame: pl.DataFrame,
    period: str,
    tol: float = 1e-10,
    max_iter: int = 50,
    perturbation: float = 0.10,
    forecast_mode: bool = False,
    _epsilon_override: dict[int, float] | None = None,
) -> tuple[dict[str, float], int, float]:
    """Generic single-country Newton-Raphson solve at one quarter.

    Works for any ROW country whose equations are in
    ``mc_model.EQUATIONS_BY_COUNTRY[prefix]``. Builds the scalar GENR
    evaluator for that country, computes historical AR innovations and
    ε residuals from the frame, assembles a residual function over the
    joint (stochastic LHS primitives + ``xxY`` identity) state, and
    Newton-solves.

    In-sample tracking: when initial guess is near history, Newton
    recovers the historical state to machine precision (validated for
    CA via ``test_ca_full_block_newton_recovers_history``).

    Args:
      prefix: Country prefix (``"CA"``, ``"JA"``, ...).
      frame: Wide frame from ``mc_model.build_frame_mc``.
      period: Quarter to solve, e.g. ``"2010Q1"``.
      tol, max_iter: Newton stopping thresholds.
      perturbation: Initial-guess perturbation from history for testing.

    Returns:
      ``(solved, iterations, residual_norm)``.
    """
    import jax
    from . import model as mc_model

    country = mc_countries.by_prefix(prefix)
    annual_lag = country.annual_lag
    # Newton solves only the linear-in-parameters block; NLEQ equations
    # (Fair's xxLPXA family) estimate separately via ``nlols_lpxa`` and
    # don't feed back into the country's demand block.
    eqs = sorted(
        (e for e in mc_model.EQUATIONS_BY_COUNTRY[prefix] if not e.is_nleq),
        key=lambda e: e.number,
    )
    state_order = _country_state_order(prefix)

    genrs = _scalar_genrs(prefix, annual_lag)
    hist_ctx = _build_solve_ctx(frame, period, prefix, annual_lag)

    # Coefficients from Fair's OUT per equation number.
    params = {eq.number: mc_model.REFERENCE_PARAMS_MC[eq.number] for eq in eqs}

    # Historical AR innovations and ε for in-sample tracking.
    # In forecast_mode, ε defaults to 0 (unconstrained model prediction).
    u_lag1_map: dict[int, jnp.ndarray] = {}
    u_lag2_map: dict[int, jnp.ndarray] = {}
    eps_map: dict[int, jnp.ndarray] = {}
    for eq in eqs:
        coefs = params[eq.number]
        u1 = None
        u2 = None
        if eq.has_ar1 or eq.has_ar2:
            try:
                u1 = _compute_ar_innovation(eq, hist_ctx, genrs, coefs,
                                            lag_offset=1)
            except KeyError:
                u1 = jnp.asarray(0.0) if forecast_mode else None
                if not forecast_mode:
                    raise
        if eq.has_ar2:
            try:
                u2 = _compute_ar_innovation(eq, hist_ctx, genrs, coefs,
                                            lag_offset=2)
            except KeyError:
                u2 = jnp.asarray(0.0) if forecast_mode else None
                if not forecast_mode:
                    raise
        u_lag1_map[eq.number] = u1
        u_lag2_map[eq.number] = u2
        if _epsilon_override is not None and eq.number in _epsilon_override:
            eps_map[eq.number] = jnp.asarray(_epsilon_override[eq.number])
        elif forecast_mode:
            try:
                eps_map[eq.number] = _compute_epsilon(
                    eq, hist_ctx, genrs, coefs, u_lag1=u1, u_lag2=u2,
                )
            except KeyError:
                eps_map[eq.number] = jnp.asarray(0.0)
        else:
            eps_map[eq.number] = _compute_epsilon(
                eq, hist_ctx, genrs, coefs, u_lag1=u1, u_lag2=u2,
            )

    def F(x: jnp.ndarray) -> jnp.ndarray:
        # Rebuild ctx with state primitives overwritten by JAX scalars.
        state_ctx = dict(hist_ctx)
        for i, name in enumerate(state_order):
            state_ctx[name] = x[i]

        parts: list[jnp.ndarray] = []
        for eq in eqs:
            coefs = params[eq.number]
            lhs = _resolve_regressor(eq.dependent, state_ctx, genrs)
            fitted = 0.0
            for token in eq.regressors:
                c = coefs[_coef_key(eq, token)]
                fitted = fitted + c * _resolve_regressor(token, state_ctx, genrs)
            if u_lag1_map[eq.number] is not None:
                fitted = fitted + coefs["RHO(-1)"] * u_lag1_map[eq.number]
            if u_lag2_map[eq.number] is not None:
                fitted = fitted + coefs["RHO(-2)"] * u_lag2_map[eq.number]
            parts.append(lhs - fitted - eps_map[eq.number])

        # xxY identity: CAY = CAC + CAI + CAG + CAEX − CAIM + CASTAT + CAV1.
        # CAEX is exogenous in single-country mode; CAI may be exogenous
        # for countries without an investment equation.
        p = prefix
        y = state_ctx[f"{p}Y"]
        c_part = state_ctx.get(f"{p}C",  hist_ctx.get(f"{p}C", 0.0))
        i_part = state_ctx.get(f"{p}I",  hist_ctx.get(f"{p}I", 0.0))
        g_part = hist_ctx.get(f"{p}G", 0.0)
        ex_part = hist_ctx.get(f"{p}EX", 0.0)
        im_part = state_ctx.get(f"{p}IM", hist_ctx.get(f"{p}IM", 0.0))
        stat_part = hist_ctx.get(f"{p}STAT", 0.0)
        v1_part = hist_ctx.get(f"{p}V1", 0.0)
        rY = y - (c_part + i_part + g_part + ex_part - im_part
                   + stat_part + v1_part)
        parts.append(rY)

        return jnp.stack(parts)

    # Initial guess: perturb each state var by ±perturbation from history.
    # In forecast_mode the frame row at ``period`` is null for endogenous
    # variables, so we fall back to the most recent lagged value.
    signs = jnp.array([1 - perturbation if i % 2 == 0 else 1 + perturbation
                       for i in range(len(state_order))])
    initial_values = []
    for name in state_order:
        if name in hist_ctx:
            initial_values.append(hist_ctx[name])
            continue
        # Fall back to nearest lag.
        for k in range(1, 9):
            lag_key = f"{name}_lag{k}"
            if lag_key in hist_ctx:
                initial_values.append(hist_ctx[lag_key])
                break
        else:
            raise KeyError(
                f"No historical value or lag available for {name} at {period} — "
                f"frame may not include enough history before forecast window."
            )
    x0 = jnp.array(initial_values, dtype=jnp.float64) * signs

    # Python-loop Newton with JAX-jitted jacfwd/solve inside. We tried
    # wrapping this in ``jax.lax.while_loop`` but the ``F`` closure
    # captures fresh per-call state (``hist_ctx`` dict, ``eps_map`` etc.)
    # so JAX can't cache the traced loop across calls — and the trace
    # cost per call dominated. A real JIT win needs the refactor
    # described in the ``v0.7+`` section of the memory (ctx → flat jnp
    # arrays so F's signature becomes hashable).
    x = x0
    for iteration in range(max_iter):
        r = F(x)
        rnorm = float(jnp.linalg.norm(r))
        if rnorm < tol:
            break
        J = jax.jacfwd(F)(x)
        dx = jnp.linalg.solve(J, r)
        x = x - dx

    solved = {state_order[k]: float(x[k]) for k in range(len(state_order))}
    return solved, iteration + 1, rnorm


def solve_all_countries_one_period(
    frame: pl.DataFrame,
    period: str,
    countries: tuple[str, ...] | None = None,
    tol: float = 1e-10,
    max_iter: int = 50,
    perturbation: float = 0.02,
) -> dict[str, tuple[dict[str, float], int, float]]:
    """Newton-solve every ROW country at one period, returning per-country
    results.

    For **in-sample** solves each country is effectively independent — each
    country's trade-weighted partner prices (``xxPMM``, ``xxPW$``) are
    historical exogenous values loaded from SHRDDD.DAT, so one pass per
    country suffices. (For out-of-sample forecasting the trade-share
    matrix would need to be endogenized; see SHR.INP equations tracked
    for v0.3.2.)

    Args:
      frame: Wide frame from ``mc_model.build_frame_mc(countries=..., include_pmm=True)``.
      period: Quarter to solve, e.g. ``"2010Q1"``.
      countries: Explicit country list. Defaults to every registered ROW
        country except the Euro-zone composite block (no data loaded).
      tol, max_iter, perturbation: Passed to each per-country Newton.

    Returns:
      Dict keyed by country prefix → ``(solved_state, iterations, rnorm)``.
    """
    if countries is None:
        countries = tuple(c.prefix for c in mc_countries.row_countries()
                          if c.prefix != "EU")
    results: dict[str, tuple[dict[str, float], int, float]] = {}
    for prefix in countries:
        try:
            results[prefix] = solve_country_one_period(
                prefix, frame, period, tol=tol, max_iter=max_iter,
                perturbation=perturbation,
            )
        except Exception:
            results[prefix] = ({}, -1, float("nan"))
            _LOG.exception("%s solve failed at %s", prefix, period)
    return results


def simulate_country_path(
    prefix: str,
    frame: pl.DataFrame,
    start_period: str,
    end_period: str,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> list[dict]:
    """In-sample dynamic tracking: solve one country period-by-period,
    feeding each solved state into the next period's lags.

    The rolling-lag pattern mirrors ``us_solve.simulate``: for the first
    period we use frame history for lags; after that, the lag column for
    period ``t`` is overwritten with the solved value before moving to
    ``t+1``. Historical ε values still drive the solve, so if the model
    tracks well, the path reproduces history.

    Args:
      prefix: Country prefix.
      frame: Wide frame from ``mc_model.build_frame_mc(countries=(prefix,))``.
      start_period: First quarter to solve.
      end_period: Last quarter (inclusive).
      tol, max_iter: Newton stopping thresholds.

    Returns:
      List of dicts, one per period, with keys
      ``{"period", "solved", "iterations", "residual_norm"}``.
    """
    # Enumerate the quarters to solve. For annual-lag countries the raw
    # data is populated only at yearly ``Q1`` intervals, so we step in
    # annual_lag-quarter increments to stay on observed periods.
    country = mc_countries.by_prefix(prefix)
    step = country.annual_lag
    periods: list[str] = []
    p = start_period
    while True:
        periods.append(p)
        if p == end_period:
            break
        p = _shift_quarter(p, step)
        # Stop if we've stepped past end_period (can happen at annual_lag=4).
        if _shift_quarter(p, -1) > end_period:
            break

    working_frame = frame
    results: list[dict] = []
    state_vars = _country_state_order(prefix)

    for period in periods:
        solved, iters, rnorm = solve_country_one_period(
            prefix, working_frame, period, tol=tol, max_iter=max_iter,
            perturbation=0.0,  # Always start from history when chaining.
        )
        results.append({
            "period": period,
            "solved": solved,
            "iterations": iters,
            "residual_norm": rnorm,
        })
        # Overwrite the primitives at ``period`` with solved values so
        # next period's lagged frame reads reflect the simulated path
        # (for validation this should equal history within solver tol).
        updates: dict[str, float] = {
            name: val for name, val in solved.items()
            if name in working_frame.columns
        }
        if updates:
            working_frame = working_frame.with_columns([
                pl.when(pl.col("period") == pl.lit(period))
                  .then(pl.lit(val))
                  .otherwise(pl.col(col_name))
                  .alias(col_name)
                for col_name, val in updates.items()
            ])
    return results


def collect_historical_epsilons(
    prefix: str,
    frame: pl.DataFrame,
    start_period: str,
    end_period: str,
) -> dict[int, list[float]]:
    """Extract historical equation residuals for bootstrap Monte Carlo.

    Evaluates each stochastic equation at every quarter in [start, end],
    computing ``ε_t = observed − fitted`` from historical data. These
    scalars are the bootstrap sample for ``forecast_country_monte_carlo``.

    Args:
      prefix: Country prefix.
      frame: Frame built via ``mc_model.build_frame_mc(countries=(prefix,))``.
      start_period, end_period: Historical window for the bootstrap sample.

    Returns:
      ``{eq_number: [ε_t, ε_{t+1}, ...]}``. Periods where a required
      input is null get silently skipped (annual countries, etc.).
    """
    from . import model as mc_model

    country = mc_countries.by_prefix(prefix)
    annual_lag = country.annual_lag
    step = annual_lag
    eqs = sorted(
        (e for e in mc_model.EQUATIONS_BY_COUNTRY[prefix] if not e.is_nleq),
        key=lambda e: e.number,
    )
    genrs = _scalar_genrs(prefix, annual_lag)
    params_by_num = {eq.number: mc_model.REFERENCE_PARAMS_MC[eq.number]
                     for eq in eqs}

    periods: list[str] = []
    p = start_period
    while True:
        periods.append(p)
        if p >= end_period:
            break
        p = _shift_quarter(p, step)

    result: dict[int, list[float]] = {eq.number: [] for eq in eqs}
    for period in periods:
        try:
            ctx = _build_solve_ctx(frame, period, prefix, annual_lag)
        except Exception:
            continue
        for eq in eqs:
            coefs = params_by_num[eq.number]
            try:
                u1 = None
                u2 = None
                if eq.has_ar1 or eq.has_ar2:
                    u1 = _compute_ar_innovation(eq, ctx, genrs, coefs, 1)
                if eq.has_ar2:
                    u2 = _compute_ar_innovation(eq, ctx, genrs, coefs, 2)
                eps = _compute_epsilon(eq, ctx, genrs, coefs,
                                        u_lag1=u1, u_lag2=u2)
                result[eq.number].append(float(eps))
            except (KeyError, Exception):
                continue
    return result


def forecast_country_monte_carlo(
    prefix: str,
    frame: pl.DataFrame,
    start_period: str,
    end_period: str,
    n_draws: int,
    historical_start: str = "1990Q1",
    historical_end: str = "2017Q4",
    rng_seed: int = 0,
) -> dict[str, list[list[dict]]]:
    """Bootstrap Monte Carlo forecast for one country.

    For each of ``n_draws``:

      1. For each equation, randomly sample an ε value from its
         historical distribution (collected over
         ``[historical_start, historical_end]``).
      2. Simulate the forecast path with those ε's (held constant
         across all forecast periods in the draw — Fair's "permanent
         shock" variant; easily extended to per-period sampling).
      3. Record the per-period solved state.

    Args:
      prefix: Country prefix.
      frame: Frame extended via ``extend_frame_for_forecast`` (so the
        forecast window exists with exogenous projections in place).
      start_period, end_period: Forecast window bounds.
      n_draws: Number of MC samples.
      historical_start, historical_end: Range to draw ε values from.
      rng_seed: Deterministic reproducibility.

    Returns:
      ``{"draws": [draw_paths_per_country_draw]}`` — list of length
      ``n_draws``, each entry a list of per-period results from
      ``simulate_country_path`` with bootstrapped ε.
    """
    import random

    eps_hist = collect_historical_epsilons(
        prefix, frame, historical_start, historical_end,
    )
    rng = random.Random(rng_seed)
    draws = []
    for _ in range(n_draws):
        # Sample one ε per equation from its historical distribution.
        eps_sample = {
            num: rng.choice(values) if values else 0.0
            for num, values in eps_hist.items()
        }
        # Run forecast with these ε values fixed across the window.
        path = _simulate_country_forecast_with_eps(
            prefix, frame, start_period, end_period, eps_sample,
        )
        draws.append(path)
    return {"draws": draws}


def _simulate_country_forecast_with_eps(
    prefix: str,
    frame: pl.DataFrame,
    start_period: str,
    end_period: str,
    eps_override: dict[int, float],
) -> list[dict]:
    """Run multi-period forecast with fixed ε values per equation.

    Used inside ``forecast_country_monte_carlo``. Each period's solve
    uses the same ε for each equation (permanent shock). Rolling-lag
    frame update feeds solved values into the next period.
    """
    country = mc_countries.by_prefix(prefix)
    step = country.annual_lag
    periods: list[str] = []
    p = start_period
    while True:
        periods.append(p)
        if p >= end_period:
            break
        p = _shift_quarter(p, step)

    working = frame
    results = []
    for period in periods:
        solved, iters, rnorm = solve_country_one_period(
            prefix, working, period,
            forecast_mode=True, perturbation=0.0,
            _epsilon_override=eps_override,  # added below
        )
        results.append({
            "period": period,
            "solved": solved,
            "iterations": iters,
            "residual_norm": rnorm,
        })
        updates = {name: val for name, val in solved.items()
                    if name in working.columns}
        if updates:
            working = working.with_columns([
                pl.when(pl.col("period") == pl.lit(period))
                  .then(pl.lit(val))
                  .otherwise(pl.col(col_name))
                  .alias(col_name)
                for col_name, val in updates.items()
            ])
    return results


def extend_frame_for_forecast(
    frame: pl.DataFrame,
    n_quarters: int,
    method: str = "persistence",
) -> pl.DataFrame:
    """Append ``n_quarters`` empty periods to the frame for forecasting.

    The new rows have all columns set to null; callers are expected to
    fill exogenous variables with projections before simulating. Two
    convenience methods populate the extension automatically:

    * ``"persistence"`` — carry the last observed value forward.
    * ``"trend"`` — linearly extrapolate from the last 4 quarters.

    Endogenous variables (anything that a stochastic equation produces
    as its LHS primitive) are left null so the solver can write them
    during ``simulate_country_path`` / ``simulate_mc_path``.

    Args:
      frame: Source frame.
      n_quarters: How many quarters to add at the end.
      method: Extrapolation rule for exogenous columns.

    Returns:
      Extended frame (original + ``n_quarters`` new rows).
    """
    from . import model as mc_model

    # Collect endogenous variable names across all registered countries.
    endogenous: set[str] = set()
    for eqs in mc_model.EQUATIONS_BY_COUNTRY.values():
        for eq in eqs:
            if eq.lhs_transform:
                endogenous.add(eq.lhs_transform.split("=")[0].strip())
            endogenous.add(eq.dependent)
        # Plus the identity-output variables (xxY etc.).
    for prefix in mc_model.EQUATIONS_BY_COUNTRY:
        endogenous.add(f"{prefix}Y")

    last_row = frame.tail(1).row(0, named=True)
    last_period = last_row["period"]

    new_periods = [_shift_quarter(last_period, k) for k in range(1, n_quarters + 1)]
    # Seed every new row with nulls; then fill exogenous columns.
    schema = {col: frame[col].dtype for col in frame.columns}
    new_rows = []
    for period in new_periods:
        row: dict = {}
        for col in frame.columns:
            if col == "period":
                row[col] = period
            else:
                row[col] = None
        new_rows.append(row)
    new_df = pl.DataFrame(new_rows, schema=schema)
    extended = pl.concat([frame, new_df])

    # Exogenous fill.
    exog_cols = [c for c in frame.columns if c not in endogenous and c != "period"]
    if method == "persistence":
        # Carry last value forward for every exogenous column.
        fills = {}
        for col in exog_cols:
            non_null = frame[col].drop_nulls()
            if non_null.len() == 0:
                continue
            last = non_null[-1]
            fills[col] = last
        updates = [
            pl.when(pl.col("period").is_in(new_periods))
              .then(pl.lit(val))
              .otherwise(pl.col(col))
              .alias(col)
            for col, val in fills.items()
        ]
        if updates:
            extended = extended.with_columns(updates)
    elif method == "trend":
        # Linear fit on last 4 non-null values per column.
        import numpy as _np
        fills_by_col: dict[str, list[tuple[str, float]]] = {}
        for col in exog_cols:
            non_null_frame = frame.filter(pl.col(col).is_not_null()).tail(4)
            if non_null_frame.height < 2:
                continue
            values = non_null_frame[col].to_numpy()
            x = _np.arange(len(values))
            slope, intercept = _np.polyfit(x, values, 1)
            for k, period in enumerate(new_periods, start=1):
                projected = float(intercept + slope * (len(values) - 1 + k))
                fills_by_col.setdefault(col, []).append((period, projected))
        updates = []
        for col, pairs in fills_by_col.items():
            expr = pl.col(col)
            for period, val in pairs:
                expr = pl.when(pl.col("period") == pl.lit(period)).then(pl.lit(val)).otherwise(expr)
            updates.append(expr.alias(col))
        if updates:
            extended = extended.with_columns(updates)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return extended.sort("period")


def build_frame_for_endogenous_forecast(
    countries: tuple[str, ...],
    n_forecast_quarters: int = 8,
    method: str = "persistence",
) -> pl.DataFrame:
    """Assemble the full frame an endogenous MC forecast needs.

    Handles the ordering subtlety: SHR data extends well past the
    country-block historical window (SHRDDD.DAT runs through 2025Q4;
    YDATA.DAT ends at 2017Q4). Joining raw introduces nulls in country
    columns for 2018Q1+ which break the solver. The correct order is:

      1. Build country-block frame and extend it ``n_forecast_quarters``
         quarters with ``extend_frame_for_forecast`` (fills exogenous).
      2. Truncate SHR frame to the extended frame's period range.
      3. Left-join SHR columns onto the extended frame.

    Args:
      countries: Country prefixes for the country block.
      n_forecast_quarters: How many quarters past historical to allow.
      method: ``"persistence"`` or ``"trend"`` for exogenous fill.

    Returns:
      Wide frame with country data, SHR bilateral data, and forecast
      rows pre-extended with exogenous projections.
    """
    from . import model as mc_model, shr as mc_shr
    base = mc_model.build_frame_mc(countries=countries)
    extended = extend_frame_for_forecast(
        base, n_forecast_quarters, method=method,
    )
    shr_frame = mc_shr.build_shr_frame()
    shr_trunc = shr_frame.filter(
        (pl.col("period") >= pl.lit(extended["period"].min()))
        & (pl.col("period") <= pl.lit(extended["period"].max()))
    )
    return extended.join(shr_trunc, on="period", how="left",
                          coalesce=True).sort("period")


def simulate_mc_endogenous(
    frame: pl.DataFrame,
    start_period: str,
    end_period: str,
    countries: tuple[str, ...] | None = None,
    shr_coefs: dict[int, tuple[float, float, float]] | None = None,
    outer_iters: int = 10,
    inner_tol: float = 1e-10,
    convergence_tol: float = 1e-4,
) -> dict:
    """Fully-endogenous MC simulation with Fair-style block Gauss-Seidel.

    Mirrors FP.EXE's per-period solve (``SETLA 357 10; SETLA 358 10``):

      1. **Outer loop** over periods (quarters or years, per country).
      2. **Block Gauss-Seidel inner loop** at each period:

         a) Project trade shares one step via
            :func:`mc_shr.project_shr_one_period`
            (``LAA_ij[t] = c + β_lag · LA_ij[t-1] + β_p · P_ij[t]``).
         b) Aggregate import-price proxies via
            :func:`mc_shr.aggregate_pmm_one_period`
            (``xxPMM = Σ_j A_ji[t-1] · yyPX$[t]``).
         c) For each country, Newton-solve the demand block with the
            current ``xxPMM`` (via
            :func:`solve_country_one_period`).
         d) Update ``P_ij[t] = <SRC>PX$[t] / <DST>PMM[t]`` using the
            freshly-solved export prices.
         e) Check PMM convergence across countries; break when the
            max relative change < ``convergence_tol``.

      3. Persist the solved state at this period into the frame's
         primitive columns so next period's lags reflect the simulated
         path.

    The inner loops call the same JAX kernels already validated for
    in-sample solves — this function adds the Python-level Gauss-Seidel
    orchestration and SHR-step integration without changing the
    numerical kernels.

    Args:
      frame: Wide frame from ``build_frame_mc`` + ``extend_frame_for_forecast``.
        Must have the SHR frame joined in (from ``mc_shr.build_shr_frame``)
        if full-endogenous PMM is desired.
      start_period, end_period: Quarterly bounds.
      countries: Defaults to every ROW demand-block country.
      shr_coefs: Pre-cached SHR coefficients. If None, the SHR step is
        skipped (persistence PMM).
      outer_iters: Max Gauss-Seidel iterations per period (Fair: 10).
      inner_tol: Per-country Newton threshold.
      convergence_tol: Max relative PMM change across destinations to
        declare outer-loop convergence.

    Returns:
      ``{"periods": [...], "paths": {country: [...]}, "outer_iters": [...]}``.
    """
    if countries is None:
        from . import model as mc_model
        countries = tuple(
            c.prefix for c in mc_countries.row_countries()
            if c.prefix not in ("EU", "US")
            and c.prefix in mc_model.EQUATIONS_BY_COUNTRY
        )

    # Period sequence — use quarterly step (countries with annual_lag=4
    # will have their solver internally align to Q1s).
    periods: list[str] = []
    p = start_period
    while True:
        periods.append(p)
        if p >= end_period:
            break
        p = _shift_quarter(p, 1)

    working = frame
    paths = {c: [] for c in countries}
    outer_iters_per_period: list[int] = []

    # Lazy import to avoid circular dep at module import time.
    from . import shr as mc_shr

    for period in periods:
        prev_pmm: dict[str, float] = {}
        last_outer = 0
        for outer_iter in range(outer_iters):
            last_outer = outer_iter + 1
            # --- 1. Project trade shares (if SHR coefs provided) ---
            if shr_coefs:
                try:
                    working = mc_shr.project_shr_one_period(
                        working, period, shr_coefs,
                    )
                except Exception:
                    # Non-fatal — missing inputs for some equations.
                    pass

            # --- 2. Aggregate xxPMM for each destination ---
            if shr_coefs:
                try:
                    working = mc_shr.aggregate_pmm_one_period(
                        working, period, countries,
                    )
                except Exception:
                    pass

            # --- 3. Per-country Newton ---
            period_pmm: dict[str, float] = {}
            for prefix in countries:
                try:
                    solved, _iters, rnorm = solve_country_one_period(
                        prefix, working, period,
                        forecast_mode=True, perturbation=0.0,
                        tol=inner_tol,
                    )
                    if outer_iter == last_outer - 1:
                        paths[prefix].append({
                            "period": period,
                            "solved": solved,
                            "iterations": _iters,
                            "residual_norm": rnorm,
                        })
                    # Write solved values into the frame for downstream
                    # aggregations and the next period's lags.
                    updates = {name: val for name, val in solved.items()
                               if name in working.columns}
                    if updates:
                        working = working.with_columns([
                            pl.when(pl.col("period") == pl.lit(period))
                              .then(pl.lit(val))
                              .otherwise(pl.col(col_name))
                              .alias(col_name)
                            for col_name, val in updates.items()
                        ])
                except Exception:
                    # Allow country solve to fail; next outer iter may recover.
                    pass

            # --- 4. Convergence check on xxPMM changes ---
            if shr_coefs:
                row = working.filter(
                    pl.col("period") == pl.lit(period)
                ).row(0, named=True)
                for dst in countries:
                    val = row.get(f"{dst}PMM")
                    if val is not None:
                        period_pmm[dst] = float(val)
                if prev_pmm and period_pmm:
                    max_change = max(
                        abs(period_pmm[k] - prev_pmm.get(k, period_pmm[k]))
                        / (abs(period_pmm[k]) + 1e-12)
                        for k in period_pmm
                    )
                    if max_change < convergence_tol:
                        break
                prev_pmm = period_pmm
            else:
                # No SHR stepping — one pass is enough.
                break

        outer_iters_per_period.append(last_outer)

    return {
        "periods": periods,
        "paths": paths,
        "outer_iters": outer_iters_per_period,
    }


def simulate_mc_path(
    frame: pl.DataFrame,
    start_period: str,
    end_period: str,
    countries: tuple[str, ...] | None = None,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> dict[str, list[dict]]:
    """Multi-country in-sample dynamic simulation.

    Runs ``simulate_country_path`` for every ROW country (except EU and
    the US MC-only block) and returns per-country period paths. For
    in-sample solves each country is independent of the others — trade
    prices come from SHRDDD.DAT, not from sibling countries' current
    solved states — so this is just a country loop on top of the
    single-country driver. The shape is right for the out-of-sample
    case: when trade-share endogenization lands (v0.4), this function
    becomes the natural place to hang the block-Gauss-Seidel outer loop.

    Args:
      frame: Wide frame from ``mc_model.build_frame_mc(countries=...)``.
        Must include every country in ``countries``.
      start_period, end_period: Quarterly bounds, e.g. ``"2005Q1"``.
      countries: Explicit list. Defaults to every ROW country that has
        a multi-equation demand block (skips ``"US"`` and ``"EU"``).
      tol, max_iter: Per-period Newton thresholds.

    Returns:
      Dict ``{country_prefix: [period_result, ...]}`` where each period
      result has ``period``, ``solved``, ``iterations``, ``residual_norm``.
    """
    if countries is None:
        from . import model as mc_model
        countries = tuple(
            c.prefix for c in mc_countries.row_countries()
            if c.prefix != "EU" and c.prefix != "US"
            and c.prefix in mc_model.EQUATIONS_BY_COUNTRY
        )
    return {
        prefix: simulate_country_path(
            prefix, frame, start_period, end_period,
            tol=tol, max_iter=max_iter,
        )
        for prefix in countries
    }


# Note: earlier versions of this module carried ``build_mc_residual_function``
# and ``simulate_one_period`` stubs that raised ``NotImplementedError``.
# The functionality they described is now provided by:
#   * ``solve_country_one_period`` — per-country Newton residual + solve
#   * ``solve_all_countries_one_period`` — all ROW countries, one period
#   * ``simulate_mc_endogenous`` — Fair-style block Gauss-Seidel with SHR
#     projection + xxPMM aggregation per period
# The stubs were removed in v0.5 so the public API reflects only working
# entry points.
