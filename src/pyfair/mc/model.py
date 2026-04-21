"""MC Model — JAX port of Fair's Multi-Country specification.

Architectural twin of ``us_model.py``: ``MCEquation`` mirrors ``UsEquation``,
``build_frame_mc`` is the MC analog of ``build_frame``, and ``estimate``
reuses the same ``step02_estimate`` 2SLS-with-AR(1) kernels.

Scope of v0.1 (this session):
  * Country registry for all 38 MC countries (see ``mc_countries``).
  * GENR template for Canada, plus hand-written specs for CA EQ 41–49.
  * Parser for Fair's ``OUT`` file → ``REFERENCE_PARAMS_MC``.
  * ``estimate`` dispatch that matches the US pipeline's contract.

Deliberately out of scope until the architecture is settled:
  * Full 37-country GENR rollout (we start with CA, generalize once the
    template is validated against Fair's reference coefficients).
  * Trade-share matrix linkage (SHR.INP / SHRDDD.DAT).
  * Forecast driver and Newton-per-period solver.
  * EU / SA / JO blocks that live implicitly inside other country sections.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import polars as pl

from .. import config
from ..core import readers
from ..core.estimate import (
    nlols_lpxa,
    two_sls_ar1, two_sls_ar1_bounded,
    two_sls_ar2, two_sls_ar2_bounded,
    two_sls_with_se,
)
from ..us.model import _parse_token, add_lags, add_time_trend_and_constant
from . import countries as mc_countries


_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Equation specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MCEquation:
    """Static spec for one Fair MC-model stochastic equation.

    Parallel to ``UsEquation`` but with an explicit ``country`` field so the
    dispatcher can look up country-specific samples, lag conventions, and
    reference coefficients without inferring from the variable name.

    Attributes:
      country: Two-letter prefix (``"CA"``, ``"JA"``, ...). The US block
        also appears in MC.INP; we label its MC-only additions with
        ``"US"`` and defer to ``us_model`` for 1–30.
      number: Fair's equation number (e.g. ``41`` for CA imports).
      dependent: Dependent-variable column name (e.g. ``"CALIMZ"``).
      regressors: RHS variable tokens in order. ``"C"`` is the constant.
        ``"CALIMZ(-1)"`` denotes lag-1 of CALIMZ.
      instruments: First-stage regressors (instruments) for 2SLS. When
        empty or equal to ``regressors``, 2SLS collapses to OLS —
        Fair's ``ALT2SLS`` option defaults to this for equations without
        an explicit ``FSR`` block.
      has_ar1: True if Fair declares ``RHO=1``.
      has_ar2: True if Fair declares ``RHO=2``. Uses ``two_sls_ar2`` and
        needs two pre-sample rows instead of one. Mutually exclusive with
        ``has_ar1``.
      sample_start: First period of Fair's estimation window (e.g.
        ``"1961Q1"`` — ROW countries start later than the US).
      sample_end: Last period (usually ``"2017Q4"`` in the 2018 vintage).
      lhs_transform: Optional DSL directive mapping the estimated
        dependent variable back to the raw level (e.g.
        ``"CAIM=EXP(CALIMZ)*CAPOP"``). Not used by the estimator but
        recorded for the solve driver.
      damping, use_bounded_search, notes: Same semantics as ``UsEquation``.
    """
    country: str
    number: int
    dependent: str
    regressors: tuple[str, ...]
    instruments: tuple[str, ...]
    has_ar1: bool
    sample_start: str
    sample_end: str
    has_ar2: bool = False
    is_nleq: bool = False  # Fair's nonlinear-in-parameters xxLPXA equations.
    lhs_transform: str | None = None
    damping: float = 1.0
    use_bounded_search: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Reference coefficients from Fair's ``OUT`` file
# ---------------------------------------------------------------------------

# Keyed by equation number → token → coefficient. MC equation numbers span
# the full range (1..30 for US, 41..422 for ROW); the same dict serves all.
REFERENCE_PARAMS_MC: dict[int, dict[str, float]] = {}


def parse_mc_out(path: Path = config.MC_OUT) -> None:
    """Populate ``REFERENCE_PARAMS_MC`` from Fair's MC estimation output.

    The OUT format is identical to ``fmout.txt`` modulo the three-digit
    equation numbers (``411``, ``412``, ...). We keep the first occurrence
    of each equation number — Fair sometimes re-runs equations under
    ``MCCRIS`` scenario sections later in the file.
    """
    text = path.read_text()
    block_starts = [m.start() for m in
                    re.finditer(r"^Equation number =\s+\d+", text, re.M)]
    block_starts.append(len(text))
    coef_re = re.compile(
        r"^\s*\d+\s+(\S+)\s*\(\s*(-?\d+)\)\s+([-+\d\.E]+)\s+[-+\d\.E]+\s+[-+\d\.]+",
        re.M,
    )
    # NLEQ blocks have rows like "  1  initial  coef_est  SE  T-stat" with
    # no variable name. Their keys become ``COEF(+1)``, ``COEF(+2)``, ...
    nleq_coef_re = re.compile(
        r"^\s*(\d+)\s+[-+\d\.E]+\s+([-+\d\.E]+)\s+[-+\d\.E]+\s+[-+\d\.]+",
        re.M,
    )
    for i in range(len(block_starts) - 1):
        block = text[block_starts[i]: block_starts[i + 1]]
        num_match = re.match(r"Equation number =\s+(\d+)", block)
        if not num_match:
            continue
        eq_num = int(num_match.group(1))
        if eq_num in REFERENCE_PARAMS_MC:
            continue
        se_pos = block.find("SE of equation")
        if se_pos < 0:
            continue
        table = block[:se_pos]
        coefs = {}
        named_matches = coef_re.findall(table)
        if named_matches:
            for name, lag, val in named_matches:
                key = f"{name}({int(lag):+d})" if lag != "0" else f"{name}(0)"
                coefs[key] = float(val)
        else:
            # Fall through to NLEQ row format.
            for idx, val in nleq_coef_re.findall(table):
                coefs[f"COEF({int(idx):+d})"] = float(val)
        REFERENCE_PARAMS_MC[eq_num] = coefs


if config.MC_OUT.exists():
    parse_mc_out()


# ---------------------------------------------------------------------------
# GENR template — per-country derived variables
# ---------------------------------------------------------------------------

def _c(name: str) -> pl.Expr:
    """Shorthand for ``pl.col``."""
    return pl.col(name)


def _mc_global_genr_specs() -> list[tuple[str, Callable[[pl.DataFrame], pl.Expr]]]:
    """GENRs that MC.INP computes once before any country block runs.

    Mirrors MC.INP lines 70–78: 4-quarter moving averages and US/Euro-zone
    rate/price aliases that ROW country GENRs reference. Annual-data
    (annual_lag=4) countries use the ``*A`` averages in place of the
    current-quarter values.
    """
    def forward_avg(c: pl.Expr) -> pl.Expr:
        """Forward 4-quarter average (current + 3 leads).

        Fair's MC.INP uses ``GEE(1)+GEE(2)+GEE(3)`` in these moving-average
        GENRs, where positive ``(1)`` (etc.) means a *lead* — confirmed by
        reproducing BE's ``IDENT BEE=BEH*GEEA`` at 2010Q1 to 1e-16.
        """
        return (c + c.shift(-1) + c.shift(-2) + c.shift(-3)) / 4

    return [
        # Variable-rename aliases noted in MC.INP lines 35-37: Fair renamed
        # a few data columns between the 2018 YDATA.DAT and the equation
        # spec. The data file stores ``COGG``/``THGG``/``DELHH`` but the
        # equations (and our GENR template) reference ``COG``/``THG``/``DELH``.
        ("COG",    lambda df: _c("COGG")),
        ("THG",    lambda df: _c("THGG")),
        ("DELH",   lambda df: _c("DELHH")),
        # US aliases — US model uses unprefixed RS, GDPD; MC wires them to
        # the "US"-prefixed names that ROW countries reference.
        ("USRS",   lambda df: _c("RS")),
        ("USPY",   lambda df: _c("GDPD")),
        # 4-quarter moving averages of US, German, and Euro-zone anchors
        # used by annual-lag country blocks (BE onwards).
        ("USRSA",  lambda df: forward_avg(_c("RS"))),
        ("USPYA",  lambda df: forward_avg(_c("GDPD"))),
        # US MC-only derived: export-price and domestic-deflator log ratios
        # vs trade-weighted partner prices. Feed Fair's MC-only EQ 31
        # (``LPXA ~ LPXB + ρ1·u(-1) + ρ2·u(-2)``, RHO=2).
        ("LPXA",   lambda df: _c("USPX$").log() - _c("USPW$").log()),
        ("LPXB",   lambda df: _c("GDPD").log() - _c("USPW$").log()),
        ("GERSA",  lambda df: forward_avg(_c("GERS"))),
        ("GEPYA",  lambda df: forward_avg(_c("GEPY"))),
        ("GEEA",   lambda df: forward_avg(_c("GEE"))),
        ("EURSA",  lambda df: forward_avg(_c("EURS"))),
        ("EURBA",  lambda df: forward_avg(_c("EURB"))),
    ]


def _apply_global_genrs(df: pl.DataFrame) -> pl.DataFrame:
    """Apply the MC-wide pre-country GENR block, skipping missing inputs."""
    for name, formula in _mc_global_genr_specs():
        try:
            df = df.with_columns(formula(df).alias(name))
        except pl.exceptions.ColumnNotFoundError:
            pass
    return df


def _row_country_genr_specs_annual(p: str) -> list[tuple[str, Callable[[pl.DataFrame], pl.Expr]]]:
    """Annual-lag variant of the ROW country GENR template.

    Fair's "small-country" blocks (BE onwards) use annual-interpolated data
    where real observations land at ``Q1`` of each year — so Fair writes
    ``X(-4)`` everywhere the quarterly template wrote ``X(-1)``, and
    ``PCPY`` becomes year-over-year (exponent=1) instead of annualized-quarterly.

    This factory mirrors ``_row_country_genr_specs`` with shift(4) in place
    of shift(1), and with the rate-anchor of RB/RS using ``(-4)`` in place of
    ``(-2)``. Extensions for GEPYA / GEEA / USRSA-anchored quantities are
    picked up from ``_mc_global_genr_specs``.
    """
    def col(suffix: str) -> pl.Expr:
        return _c(f"{p}{suffix}")

    return [
        (f"{p}LIMZ",   lambda df: (col("IM") / col("POP")).log()),
        (f"{p}LCZ",    lambda df: (col("C") / col("POP")).log()),
        (f"{p}LCIGZ",  lambda df: ((col("C") + col("I") + col("G"))
                                   / col("POP")).log()),
        (f"{p}LPYZZ",  lambda df: (col("PY") / col("PM")).log()),
        (f"{p}LYZ",    lambda df: (col("Y") / col("POP")).log()),
        (f"{p}LI",     lambda df: col("I").log()),
        (f"{p}LY",     lambda df: col("Y").log()),
        (f"{p}LPY",    lambda df: col("PY").log()),
        (f"{p}LPM",    lambda df: col("PM").log()),
        # YoY inflation — exponent 1 rather than quarterly-annualized 4.
        (f"{p}PCPY",   lambda df: 100 * ((col("PY") / col("PY").shift(4))**1 - 1)),
        # Rate spreads use 4-quarter anchor (== 1 year back).
        (f"{p}RBZ",    lambda df: col("RB") - col("RS").shift(4)),
        (f"{p}RBZZ",   lambda df: col("RB").shift(4) - col("RS").shift(4)),
        (f"{p}RSZ",    lambda df: col("RS") - col("RS").shift(4)),
        (f"{p}RSZZ",   lambda df: col("RS").shift(4) - col("RS").shift(4)),
        # Exchange-rate quantities.
        (f"{p}LE",     lambda df: col("E").log()),
        (f"{p}LE1",    lambda df: col("E").log() - col("E").shift(4).log()),
        # Annual-lag countries anchor to the 4-quarter trailing averages
        # ``USPYA`` and ``USRSA`` (matches Fair's ``GENR xxLEA=LOG(xxPY/USPYA)-...``).
        (f"{p}LEA",    lambda df: (col("PY") / _c("USPYA")).log() - col("E").shift(4).log()),
        (f"{p}LRSZ",   lambda df: 0.25 * ((1 + col("RS") / 100)
                                          / (1 + _c("USRSA") / 100)).log()),
        (f"{p}LPXA",   lambda df: (col("PX") / (_c(f"{p}PW$") * (col("E") / col("E10")))).log()),
        (f"{p}LPXB",   lambda df: (col("PY") / (_c(f"{p}PW$") * (col("E") / col("E10")))).log()),
        # Employment & output growth — both at annual horizon.
        (f"{p}LJ1",    lambda df: col("J").log() - col("J").shift(4).log()),
        (f"{p}LY1",    lambda df: col("Y").log() - col("Y").shift(4).log()),
        (f"{p}LEXL",   lambda df: (col("J") / (col("Y") / col("LAM"))).log()),
        (f"{p}LL1Z",   lambda df: (col("L1") / col("POP1")).log()),
        (f"{p}LGZ",    lambda df: (col("G") / col("POP")).log()),
        (f"{p}LEXZ",   lambda df: (col("EX") / col("POP")).log()),
    ]


def _row_country_genr_specs(p: str) -> list[tuple[str, Callable[[pl.DataFrame], pl.Expr]]]:
    """GENR specs for one ROW country, parameterized by its 2-letter prefix.

    Mirrors MC.INP lines 628–654 (the CA block). Other ROW countries share
    the identical template — only the prefix changes — so this factory can
    be reused across CA, JA, AU, FR, GE, IT, NE, ST, UK, FI, AS by calling
    it with the appropriate prefix. Countries with ``annual_lag=4`` need
    a variant that replaces ``shift(1)`` with ``shift(4)``.

    Args:
      p: Two-letter country prefix (e.g. ``"CA"``).

    Returns:
      List of ``(output_column_name, formula_fn)`` pairs suitable for
      sequential ``with_columns`` application.
    """
    def col(suffix: str) -> pl.Expr:
        return _c(f"{p}{suffix}")

    return [
        # Imports per capita, log — dependent for EQ x1.
        (f"{p}LIMZ",   lambda df: (col("IM") / col("POP")).log()),
        # Consumption per capita, log — dependent for EQ x2.
        (f"{p}LCZ",    lambda df: (col("C") / col("POP")).log()),
        # (C + I + G) per capita, log — RHS for EQ x1.
        (f"{p}LCIGZ",  lambda df: ((col("C") + col("I") + col("G"))
                                   / col("POP")).log()),
        # Price ratio log(PY/PM) — RHS for EQ x1.
        (f"{p}LPYZZ",  lambda df: (col("PY") / col("PM")).log()),
        # Output per capita, log — RHS for EQ x2, x3.
        (f"{p}LYZ",    lambda df: (col("Y") / col("POP")).log()),
        # Investment log — dependent for EQ x3.
        (f"{p}LI",     lambda df: col("I").log()),
        # Output log — RHS for EQ x3.
        (f"{p}LY",     lambda df: col("Y").log()),
        # Domestic price log — dependent for EQ x4.
        (f"{p}LPY",    lambda df: col("PY").log()),
        # Import-price log — RHS for EQ x4.
        (f"{p}LPM",    lambda df: col("PM").log()),
        # Annualized inflation — RHS for EQ x5 (e.g. JAPCPY for Japan).
        (f"{p}PCPY",   lambda df: 100 * ((col("PY") / col("PY").shift(1))**4 - 1)),
        # Long-rate / short-rate spread measures — EQ x6.
        (f"{p}RBZ",    lambda df: col("RB") - col("RS").shift(2)),
        (f"{p}RBZZ",   lambda df: col("RB").shift(1) - col("RS").shift(2)),
        (f"{p}RSZ",    lambda df: col("RS") - col("RS").shift(2)),
        (f"{p}RSZZ",   lambda df: col("RS").shift(1) - col("RS").shift(2)),
        # Exchange-rate quantities — EQ x7.
        (f"{p}LE",     lambda df: col("E").log()),
        (f"{p}LE1",    lambda df: col("E").log() - col("E").shift(1).log()),
        # Relative price level vs US — EQ x7 RHS. Cross-country link.
        (f"{p}LEA",    lambda df: (col("PY") / _c("USPY")).log() - col("E").shift(1).log()),
        # Relative real short-rate vs US — FSR term. Cross-country link.
        (f"{p}LRSZ",   lambda df: 0.25 * ((1 + col("RS")/100) / (1 + _c("USRS")/100)).log()),
        # Export-price LHS / RHS — EQ x8.
        (f"{p}LPXA",   lambda df: (col("PX") / (_c(f"{p}PW$") * (col("E") / col("E10")))).log()),
        (f"{p}LPXB",   lambda df: (col("PY") / (_c(f"{p}PW$") * (col("E") / col("E10")))).log()),
        # Employment growth — EQ x9.
        (f"{p}LJ1",    lambda df: col("J").log() - col("J").shift(1).log()),
        # Output growth — RHS for EQ x9.
        (f"{p}LY1",    lambda df: col("Y").log() - col("Y").shift(1).log()),
        # Excess-labor ratio — EQ x9 RHS. Depends on CAJMIN = CAY/CALAM
        # (IDENT from MC.INP line 626) which we compute inline.
        (f"{p}LEXL",   lambda df: (col("J") / (col("Y") / col("LAM"))).log()),
        # Labor force per capita — EQ x10 (not all countries).
        (f"{p}LL1Z",   lambda df: (col("L1") / col("POP1")).log()),
        # FSR-only helpers: per-capita government spending, per-capita exports.
        (f"{p}LGZ",    lambda df: (col("G") / col("POP")).log()),
        (f"{p}LEXZ",   lambda df: (col("EX") / col("POP")).log()),
    ]


# ---------------------------------------------------------------------------
# Identities needed before GENRs — CAY = CAC+CAI+CAG+... etc.
# ---------------------------------------------------------------------------

def _row_country_identities(p: str) -> list[tuple[str, Callable[[pl.DataFrame], pl.Expr]]]:
    """Accounting identities Fair declares via ``IDENT`` for each ROW country.

    Only the identities whose outputs feed the GENR template above are
    included — the full IDENT set (trade-share formulas, PX$, PM, etc.)
    is deferred to the solve driver.

    Mirrors MC.INP lines 619–627 for Canada; other ROW countries duplicate
    the same seven identities with prefix-swapping.
    """
    def col(suffix: str) -> pl.Expr:
        return _c(f"{p}{suffix}")

    return [
        # CAY = CAC + CAI + CAG + CAEX - CAIM + CASTAT + CAV1. Computed
        # here so GENRs can reference CAY; real data file may already
        # provide this column in which case we recompute to the identity.
        (f"{p}Y",    lambda df: (col("C") + col("I") + col("G")
                                 + col("EX") - col("IM")
                                 + col("STAT") + col("V1"))),
    ]


# Per-country catalog of which extra GENRs to materialize. Names map to
# closure factories below; the factory looks up ``annual_lag`` so the same
# key ("LH1") produces shift(1) for quarterly countries and shift(4) for
# annual-lag countries, and swaps the German anchor from ``GEPY`` to
# ``GEPYA`` (4-quarter average) for annual countries.
_EXTRAS_BY_COUNTRY: dict[str, tuple[str, ...]] = {
    # JA (floating vs US) and NZ (floating vs US, annual) lock 0.05 on LEA.
    "JA": ("LE1Z",),
    "NZ": ("LE1Z",),
    # Pegged-to-Germany countries. ``LH1Z`` is for countries whose EQ x7
    # regresses ``LH1Z = LH1 − 0.05·LHA``. ``LRSZG`` is a rate-spread-vs-
    # Germany helper used by a handful of FSR blocks.
    "AU": ("LH", "LH1", "LHA", "LH1Z"),
    "FR": ("LH", "LH1", "LHA"),
    "IT": ("LH", "LH1", "LHA", "LH1Z"),
    "NE": ("LH", "LH1", "LHA", "LH1Z", "LRSZG"),
    "ST": ("LH", "LH1", "LHA", "LH1Z"),
    "UK": ("LH", "LH1", "LHA", "LH1Z", "LRSZG"),
    "FI": ("LH", "LH1", "LHA", "LRSZG"),
    # Annual pegged-to-Germany countries. BE/SW/GR/IR/PO/SP estimate
    # LH1 directly; DE/NO use the LH1Z pre-transform.
    "BE": ("LH", "LH1", "LHA"),
    "DE": ("LH", "LH1", "LHA", "LH1Z"),
    "NO": ("LH", "LH1", "LHA", "LH1Z"),
    "SW": ("LH", "LH1", "LHA"),
    "GR": ("LH", "LH1", "LHA"),
    "IR": ("LH", "LH1", "LHA"),
    "PO": ("LH", "LH1", "LHA"),
    "SP": ("LH", "LH1", "LHA"),
}


def _country_extra_genrs(prefix: str) -> list[tuple[str, Callable[[pl.DataFrame], pl.Expr]]]:
    """Country-specific GENRs that sit outside the shared ROW template.

    Adapts to the country's ``annual_lag``: quarterly countries shift by
    1 quarter and anchor to Fair's current ``GEPY``/``GERS``; annual
    countries shift by 4 quarters and anchor to the 4-quarter trailing
    averages ``GEPYA``/``GERSA`` defined in ``_mc_global_genr_specs``.
    """
    try:
        annual_lag = mc_countries.by_prefix(prefix).annual_lag
    except KeyError:
        annual_lag = 1
    lag = annual_lag
    gepy = "GEPYA" if annual_lag == 4 else "GEPY"
    gers = "GERSA" if annual_lag == 4 else "GERS"

    def col(suffix: str, _p=prefix) -> pl.Expr:
        return _c(f"{_p}{suffix}")

    factories: dict[str, tuple[str, Callable[[pl.DataFrame], pl.Expr]]] = {
        "LE1Z":  (f"{prefix}LE1Z",
                  lambda df: col("LE1") - 0.05 * col("LEA")),
        "LH":    (f"{prefix}LH",
                  lambda df: col("H").log()),
        "LH1":   (f"{prefix}LH1",
                  lambda df: col("H").log() - col("H").shift(lag).log()),
        "LHA":   (f"{prefix}LHA",
                  lambda df: (col("PY") / _c(gepy)).log()
                             - col("H").shift(lag).log()),
        "LH1Z":  (f"{prefix}LH1Z",
                  lambda df: (col("H").log() - col("H").shift(lag).log())
                             - 0.05 * ((col("PY") / _c(gepy)).log()
                                       - col("H").shift(lag).log())),
        "LRSZG": (f"{prefix}LRSZG",
                  lambda df: 0.25 * ((1 + col("RS") / 100)
                                     / (1 + _c(gers) / 100)).log()),
    }
    return [factories[k] for k in _EXTRAS_BY_COUNTRY.get(prefix, ())
            if k in factories]


def _apply_country_block(
    df: pl.DataFrame, prefix: str,
) -> pl.DataFrame:
    """Apply IDENTs then GENRs then country-specific extras for one prefix.

    Looks up the country's ``annual_lag`` from ``mc_countries``; countries
    with ``annual_lag=4`` use the annual-lag GENR template (shift(4) in
    place of shift(1)). Missing-input GENRs are silently skipped — the
    equation that depends on one surfaces a clear ``KeyError`` downstream.
    """
    try:
        country = mc_countries.by_prefix(prefix)
        annual_lag = country.annual_lag
    except KeyError:
        annual_lag = 1

    genr_fn = (_row_country_genr_specs_annual if annual_lag == 4
               else _row_country_genr_specs)

    for name, formula in _row_country_identities(prefix):
        try:
            df = df.with_columns(formula(df).alias(name))
        except pl.exceptions.ColumnNotFoundError:
            pass
    for name, formula in genr_fn(prefix):
        try:
            df = df.with_columns(formula(df).alias(name))
        except pl.exceptions.ColumnNotFoundError:
            pass
    for name, formula in _country_extra_genrs(prefix):
        try:
            df = df.with_columns(formula(df).alias(name))
        except pl.exceptions.ColumnNotFoundError:
            pass
    return df


# ---------------------------------------------------------------------------
# Canada equation registry (first concrete translation target)
# ---------------------------------------------------------------------------

# MC.INP lines 656–690. Samples from EST commands lines 691–709.
# No inline FSR for most CA equations → 2SLS with ALT2SLS default collapses
# to OLS-with-AR1. For the few with inline FSR (43, 45, 46, 47, 49), we
# list the instruments explicitly.
EQUATIONS_CA: list[MCEquation] = [
    MCEquation(
        country="CA", number=41, dependent="CALIMZ",
        regressors=("C", "CALIMZ(-1)", "CALPYZZ", "CALCIGZ"),
        instruments=("C", "CALIMZ(-1)", "CALPYZZ", "CALCIGZ"),
        has_ar1=True,
        # No inline FSR → Fair's ALT2SLS collapses to OLS-with-AR(1). The IV
        # ρ-update in ``two_sls_ar1`` is degenerate when X=Z (residuals are
        # orthogonal to X by construction), so we minimize SSE over ρ directly.
        use_bounded_search=True,
        sample_start="1961Q1", sample_end="2017Q4",
        lhs_transform="CAIM=EXP(CALIMZ)*CAPOP",
    ),
    MCEquation(
        country="CA", number=42, dependent="CALCZ",
        regressors=("C", "CALCZ(-1)", "CARB(-1)", "CALYZ"),
        instruments=("C", "CALCZ(-1)", "CARB(-1)", "CALYZ"),
        has_ar1=False,
        sample_start="1961Q1", sample_end="2017Q2",
        lhs_transform="CAC=EXP(CALCZ)*CAPOP",
    ),
    MCEquation(
        country="CA", number=43, dependent="CALI",
        regressors=("C", "CALI(-1)", "CALY", "CARB(-1)"),
        instruments=("C", "CALGZ", "CALEXZ(-1)", "CALPY(-1)", "CALYZ(-1)",
                     "CARB(-1)", "CALI(-1)", "CALY(-1)"),
        has_ar1=False,
        sample_start="1961Q1", sample_end="2017Q2",
        lhs_transform="CAI=EXP(CALI)",
    ),
    MCEquation(
        country="CA", number=44, dependent="CALPY",
        regressors=("CALPY(-1)", "C", "T", "CALPM", "CAZZ(-1)"),
        instruments=("CALPY(-1)", "C", "T", "CALPM", "CAZZ(-1)"),
        has_ar1=True,
        use_bounded_search=True,  # Same OLS-with-AR(1) reason as EQ 41.
        sample_start="1970Q1", sample_end="2017Q4",
        lhs_transform="CAPY=EXP(CALPY)",
    ),
    MCEquation(
        country="CA", number=45, dependent="CARS",
        regressors=("C", "CARS(-1)", "CAZZ", "USRS"),
        instruments=("C", "CALGZ", "CALEXZ(-1)", "CALPY(-1)", "CALYZ(-1)",
                     "CARB(-1)", "CARS(-1)", "CAZZ(-1)", "USRS(-1)",
                     "CALPY(-2)"),
        has_ar1=False,
        sample_start="1972Q2", sample_end="2017Q1",
        lhs_transform="CARS=(ABS(CARS-.0)+CARS-.0)/2.+.0",
    ),
    MCEquation(
        country="CA", number=46, dependent="CARBZ",
        regressors=("C", "CARBZZ", "CARSZ", "CARSZZ"),
        instruments=("C", "CALGZ", "CALEXZ(-1)", "CALPY(-1)", "CALYZ(-1)",
                     "CARB(-1)", "CARB(-2)", "CARS(-1)", "CARS(-2)"),
        has_ar1=False,
        sample_start="1961Q1", sample_end="2017Q1",
        lhs_transform="CARB=CARBZ+CARS(-2)",
    ),
    MCEquation(
        country="CA", number=47, dependent="CALE1",
        regressors=("C", "CALRSZ", "CALEA"),
        instruments=("C", "CALGZ", "CALEXZ(-1)", "CALPY(-1)", "CALYZ(-1)",
                     "CARB(-1)", "CALE1(-1)", "CALEA(-1)", "CALRSZ(-1)"),
        has_ar1=True,
        sample_start="1972Q2", sample_end="2017Q1",
        lhs_transform="CAE=EXP(CALE1)*CAE(-1)",
    ),
    MCEquation(
        country="CA", number=48, dependent="CALPXA",
        regressors=("CALPXB",),
        instruments=("CALPXB",),
        has_ar1=False, has_ar2=True, use_bounded_search=True,
        sample_start="1961Q1", sample_end="2016Q4",
        lhs_transform="CAPX=EXP(CALPXA)*(CAPW$*(CAE/CAE10))",
    ),
    MCEquation(
        country="CA", number=49, dependent="CALJ1",
        regressors=("C", "T", "CALEXL(-1)", "CALY1", "CALY1(-1)"),
        instruments=("C", "CALGZ", "CALEXZ(-1)", "CALPY(-1)", "CALYZ(-1)",
                     "CARB(-1)", "CALJ1(-1)", "CALEXL(-1)", "CALY1(-1)", "T"),
        has_ar1=False,
        sample_start="1961Q1", sample_end="2016Q3",
        lhs_transform="CAJ=EXP(CALJ1)*CAJ(-1)",
    ),
]


# ---------------------------------------------------------------------------
# Japan equation registry
# ---------------------------------------------------------------------------

# MC.INP lines 751–804. Japan skips EQ 53 (investment is exogenous) and
# transforms the exchange-rate equation: EQ 57 regresses ``JALE1Z =
# JALE1 − 0.05*JALEA`` on just ``C`` and ``JALRSZ`` (see ``_country_extra_genrs``).
EQUATIONS_JA: list[MCEquation] = [
    MCEquation(
        country="JA", number=51, dependent="JALIMZ",
        regressors=("C", "JALIMZ(-1)", "JALCIGZ", "JALPYZZ"),
        instruments=("C", "JALIMZ(-1)", "JALCIGZ", "JALPYZZ"),
        has_ar1=True,
        use_bounded_search=True,  # No FSR → X=Z → bounded search.
        sample_start="1961Q1", sample_end="2017Q4",
        lhs_transform="JAIM=EXP(JALIMZ)*JAPOP",
    ),
    MCEquation(
        country="JA", number=52, dependent="JALCZ",
        regressors=("C", "JALCZ(-1)", "JARB", "JALYZ"),
        instruments=("C", "JALGZ", "JALEXZ(-1)", "JALPY(-1)", "JALYZ(-1)",
                     "JARB(-1)", "JALCZ(-1)", "JALCZ(-2)"),
        has_ar1=True,
        sample_start="1966Q1", sample_end="2016Q3",
        lhs_transform="JAC=EXP(JALCZ)*JAPOP",
    ),
    # EQ 53 (investment) is not specified for Japan — Fair treats JAI as
    # exogenous and drives it via the solve identity alone.
    MCEquation(
        country="JA", number=54, dependent="JALPY",
        regressors=("JALPY(-1)", "C", "T", "JALPM", "JAZZ(-1)"),
        instruments=("JALPY(-1)", "C", "T", "JALPM", "JAZZ(-1)"),
        has_ar1=True,
        use_bounded_search=True,
        sample_start="1970Q1", sample_end="2017Q4",
        lhs_transform="JAPY=EXP(JALPY)",
    ),
    MCEquation(
        country="JA", number=55, dependent="JARS",
        regressors=("C", "JARS(-1)", "JAPCPY", "USRS"),
        instruments=("C", "JALGZ", "JALEXZ(-1)", "JALPY(-1)", "JALYZ(-1)",
                     "JARB(-1)", "JARS(-1)", "JARS(-2)", "USRS(-1)",
                     "JALPY(-2)", "JAZZ(-1)"),
        has_ar1=True,
        sample_start="1972Q2", sample_end="2016Q4",
        lhs_transform="JARS=(ABS(JARS-.0)+JARS-.0)/2.+.0",
    ),
    MCEquation(
        country="JA", number=56, dependent="JARBZ",
        regressors=("C", "JARBZZ", "JARSZ", "JARSZZ"),
        instruments=("C", "JALGZ", "JALEXZ(-1)", "JALPY(-1)", "JALYZ(-1)",
                     "JARB(-1)", "JARB(-2)", "JARS(-1)", "JARS(-2)"),
        has_ar1=False,
        sample_start="1966Q1", sample_end="2016Q3",
        lhs_transform="JARB=JARBZ+JARS(-2)",
    ),
    MCEquation(
        country="JA", number=57, dependent="JALE1Z",
        regressors=("C", "JALRSZ"),
        instruments=("C", "JALGZ", "JALEXZ(-1)", "JALPY(-1)", "JALYZ(-1)",
                     "JARB(-1)", "JALE1(-1)", "JALEA(-1)", "JALRSZ(-1)"),
        has_ar1=True,
        sample_start="1972Q2", sample_end="2016Q4",
        lhs_transform="JAE=EXP(JALE1Z+.050*JALEA)*JAE(-1)",
        notes="Fair pre-transforms JALE1Z = JALE1 - 0.05*JALEA; see _country_extra_genrs.",
    ),
    MCEquation(
        country="JA", number=58, dependent="JALPXA",
        regressors=("JALPXB",),
        instruments=("JALPXB",),
        has_ar1=False, has_ar2=True, use_bounded_search=True,
        sample_start="1961Q1", sample_end="2016Q4",
        lhs_transform="JAPX=EXP(JALPXA)*(JAPW$*(JAE/JAE10))",
    ),
    MCEquation(
        country="JA", number=59, dependent="JALJ1",
        regressors=("C", "T", "JALEXL(-1)", "JALY1"),
        instruments=("C", "T", "JALEXL(-1)", "JALY1"),
        has_ar1=False,
        sample_start="1961Q1", sample_end="2016Q3",
        lhs_transform="JAJ=EXP(JALJ1)*JAJ(-1)",
    ),
    MCEquation(
        country="JA", number=60, dependent="JALL1Z",
        regressors=("C", "T", "JALL1Z(-1)", "JAUR"),
        instruments=("C", "JALGZ", "JALEXZ(-1)", "JALPY(-1)", "JALYZ(-1)",
                     "JARB(-1)", "JALL1Z(-1)", "T", "JAUR(-1)"),
        has_ar1=False,
        sample_start="1965Q4", sample_end="2016Q3",
        lhs_transform="JAL1=EXP(JALL1Z)*JAPOP1",
    ),
]


# ---------------------------------------------------------------------------
# Per-country equation registry — extend as countries come online.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Austria equation registry
# ---------------------------------------------------------------------------

# MC.INP lines 845–894. Austria is pegged-to-Germany (EQ 67 uses AULH1Z,
# LHS absorbed into EU1 dummy from 1999 onward). EQ 61 has no FSR/AR →
# plain OLS. EQ 64/66/67 use AR(1) with FSR. EQ 68 is RHO=2.
EQUATIONS_AU: list[MCEquation] = [
    MCEquation(
        country="AU", number=61, dependent="AULIMZ",
        regressors=("C", "AULIMZ(-1)", "AULCIGZ", "AULPYZZ"),
        instruments=("C", "AULIMZ(-1)", "AULCIGZ", "AULPYZZ"),
        has_ar1=False,
        sample_start="1961Q1", sample_end="2017Q4",
        lhs_transform="AUIM=EXP(AULIMZ)*AUPOP",
    ),
    MCEquation(
        country="AU", number=62, dependent="AULCZ",
        regressors=("C", "AULCZ(-1)", "AULYZ", "AURS"),
        instruments=("C", "AULGZ", "AULEXZ(-1)", "AULPY(-1)", "AULYZ(-1)",
                     "AURB(-1)", "AULCZ(-1)", "AURS(-1)"),
        has_ar1=False,
        sample_start="1971Q3", sample_end="2017Q1",
        lhs_transform="AUC=EXP(AULCZ)*AUPOP",
    ),
    MCEquation(
        country="AU", number=63, dependent="AULI",
        regressors=("C", "AULI(-1)", "AULY", "AURB"),
        instruments=("C", "AULGZ", "AULEXZ(-1)", "AULPY(-1)", "AULYZ(-1)",
                     "AURB(-1)", "AULI(-1)", "AULY(-1)"),
        has_ar1=False,
        sample_start="1971Q3", sample_end="2017Q1",
        lhs_transform="AUI=EXP(AULI)",
    ),
    MCEquation(
        country="AU", number=64, dependent="AULPY",
        regressors=("AULPY(-1)", "C", "T", "AULPM", "AUZZ(-1)"),
        instruments=("C", "AULGZ", "AULEXZ(-1)", "AULPY(-1)", "AULYZ(-1)",
                     "AURB(-1)", "AULPY(-2)", "T", "AULPM(-1)",
                     "AUZZ(-1)", "AUZZ(-2)"),
        has_ar1=True,
        sample_start="1971Q3", sample_end="2017Q2",
        lhs_transform="AUPY=EXP(AULPY)",
    ),
    MCEquation(
        country="AU", number=65, dependent="AURS",
        regressors=("C", "AURS(-1)", "AUZZ", "USRS"),
        instruments=("C", "AULGZ", "AULEXZ(-1)", "AULPY(-1)", "AULYZ(-1)",
                     "AURB(-1)", "AURS(-1)", "GERS(-1)", "USRS(-1)",
                     "AUZZ(-1)"),
        has_ar1=False,
        sample_start="1972Q2", sample_end="1998Q4",
        lhs_transform="AURS=(1-EU1)*AURS+EU1*EURS",
    ),
    MCEquation(
        country="AU", number=66, dependent="AURBZ",
        regressors=("C", "AURBZZ", "AURSZ", "AURSZZ"),
        instruments=("C", "AULGZ", "AULEXZ(-1)", "AULPY(-1)", "AULYZ(-1)",
                     "AURB(-1)", "AURB(-2)", "AURS(-1)", "AURS(-2)",
                     "AURS(-3)"),
        has_ar1=True,
        sample_start="1971Q3", sample_end="1998Q4",
        lhs_transform="AURB=(1-EU1)*(AURBZ+AURS(-2))+EU1*EURB",
    ),
    MCEquation(
        country="AU", number=67, dependent="AULH1Z",
        regressors=("C",),
        instruments=("C", "AULGZ", "AULEXZ(-1)", "AULPY(-1)", "AULYZ(-1)",
                     "AURB(-1)", "AULH1(-1)", "AULHA(-1)"),
        has_ar1=True,
        sample_start="1972Q2", sample_end="1998Q4",
        lhs_transform="AUH=(1-EU1)*(EXP(AULH1Z+.050*AULHA)*AUH(-1))+EU1*1",
        notes="Fair pre-transforms AULH1Z = AULH1 - 0.05*AULHA (see _country_extra_genrs).",
    ),
    MCEquation(
        country="AU", number=68, dependent="AULPXA",
        regressors=("AULPXB",),
        instruments=("AULPXB",),
        has_ar1=False, has_ar2=True, use_bounded_search=True,
        sample_start="1961Q1", sample_end="2016Q4",
        lhs_transform="AUPX=EXP(AULPXA)*(AUPW$*(AUE/AUE10))",
    ),
]


# ---------------------------------------------------------------------------
# Factory + data-driven rollout for remaining quarterly countries
# ---------------------------------------------------------------------------

def _eq_from_spec(country: str, s: dict) -> MCEquation:
    """Build an ``MCEquation`` from a compact spec dict.

    Agents extract Fair's per-equation metadata into dicts; this factory
    turns them into typed equations without boilerplate. The
    ``use_bounded_search`` flag is inferred automatically: true when the
    equation has AR(1)/AR(2) and no extra instruments (the IV ρ-update
    is degenerate there, as documented on ``step02_estimate.two_sls_ar1``).
    """
    regressors = tuple(s["regressors"])
    instruments = tuple(s["instruments"])
    is_nleq = s.get("is_nleq", False)
    use_bounded = (
        (s["has_ar1"] or s["has_ar2"])
        and instruments == regressors
        and not is_nleq
    )
    return MCEquation(
        country=country,
        number=s["number"],
        dependent=s["dependent"],
        regressors=regressors,
        instruments=instruments,
        has_ar1=s["has_ar1"],
        has_ar2=s["has_ar2"],
        is_nleq=is_nleq,
        sample_start=s["sample_start"],
        sample_end=s["sample_end"],
        lhs_transform=s.get("lhs_transform") or None,
        use_bounded_search=use_bounded,
        notes=s.get("notes", ""),
    )


# France — MC.INP lines 897–1004. EQ 74 has ``@ RHO=1`` *commented out*,
# so Fair fits it as plain 2SLS (no AR).
_FR_SPECS: list[dict] = [
    {"number": 71, "dependent": "FRLIMZ",
     "regressors": ["C", "FRLIMZ(-1)", "FRLPYZZ", "FRLCIGZ"],
     "instruments": ["C", "FRLIMZ(-1)", "FRLPYZZ", "FRLCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1965Q1", "sample_end": "2016Q4",
     "lhs_transform": "FRIM=EXP(FRLIMZ)*FRPOP"},
    {"number": 72, "dependent": "FRLCZ",
     "regressors": ["C", "FRLCZ(-1)", "FRLYZ", "FRRS"],
     "instruments": ["C", "FRLCZ(-1)", "FRLYZ", "FRRS"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1965Q1", "sample_end": "2017Q1",
     "lhs_transform": "FRC=EXP(FRLCZ)*FRPOP"},
    {"number": 73, "dependent": "FRLI",
     "regressors": ["C", "FRLI(-1)", "FRLY", "FRRB(-1)"],
     "instruments": ["C", "FRLGZ", "FRLEXZ(-1)", "FRLPY(-1)", "FRLYZ(-1)",
                     "FRRB(-1)", "FRLI(-1)", "FRLY(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1970Q1", "sample_end": "2017Q1",
     "lhs_transform": "FRI=EXP(FRLI)"},
    {"number": 74, "dependent": "FRLPY",
     "regressors": ["FRLPY(-1)", "C", "T", "FRLPM", "FRZZ(-1)"],
     "instruments": ["C", "FRLGZ", "FRLEXZ(-1)", "FRLPY(-1)", "FRLYZ(-1)",
                     "FRRS(-1)", "FRLPY(-2)", "T", "FRZZ(-1)", "FRLPM(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1965Q3", "sample_end": "2016Q4",
     "lhs_transform": "FRPY=EXP(FRLPY)"},
    {"number": 75, "dependent": "FRRS",
     "regressors": ["C", "FRRS(-1)", "USRS", "GERS", "FRPCPY", "FRZZ"],
     "instruments": ["C", "FRLGZ", "FRLEXZ(-1)", "FRLPY(-1)", "FRLYZ(-1)",
                     "FRRS(-1)", "USRS(-1)", "GERS(-1)", "FRZZ(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q2", "sample_end": "1998Q4",
     "lhs_transform": "FRRS=(1-EU1)*FRRS+EU1*EURS"},
    {"number": 76, "dependent": "FRRBZ",
     "regressors": ["C", "FRRBZZ", "FRRSZ", "FRRSZZ"],
     "instruments": ["C", "FRLGZ", "FRLEXZ(-1)", "FRLPY(-1)", "FRLYZ(-1)",
                     "FRRB(-1)", "FRRB(-2)", "FRRS(-1)", "FRRS(-2)", "FRRS(-3)"],
     "has_ar1": True, "has_ar2": False,
     "sample_start": "1970Q1", "sample_end": "1998Q4",
     "lhs_transform": "FRRB=(1-EU1)*(FRRBZ+FRRS(-2))+EU1*EURB"},
    {"number": 77, "dependent": "FRLH1",
     "regressors": ["C", "FRLHA"],
     "instruments": ["C", "FRLGZ", "FRLEXZ(-1)", "FRLPY(-1)", "FRLYZ(-1)",
                     "FRRS(-1)", "FRLH1(-1)", "FRLHA(-1)"],
     "has_ar1": True, "has_ar2": False,
     "sample_start": "1972Q2", "sample_end": "1998Q4",
     "lhs_transform": "FRH=(1-EU1)*(EXP(FRLH1)*FRH(-1))+EU1*1"},
    {"number": 78, "dependent": "FRLPXA",
     "regressors": ["FRLPXB"],
     "instruments": ["FRLPXB"],
     "has_ar1": False, "has_ar2": True,
     "sample_start": "1961Q1", "sample_end": "2016Q4",
     "lhs_transform": "FRPX=EXP(FRLPXA)*(FRPW$*(FRE/FRE10))"},
]
EQUATIONS_FR: list[MCEquation] = [_eq_from_spec("FR", s) for s in _FR_SPECS]


# Germany — MC.INP lines 1006–1103. Floating vs US; EQ 87 is the exchange-rate
# equation (GELE1), EQ 88 is export price RHO=2.
_GE_SPECS: list[dict] = [
    {"number": 81, "dependent": "GELIMZ",
     "regressors": ["C", "GELIMZ(-1)", "GELCIGZ", "GELPYZZ"],
     "instruments": ["C", "GELIMZ(-1)", "GELCIGZ", "GELPYZZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1961Q1", "sample_end": "2017Q4",
     "lhs_transform": "GEIM=EXP(GELIMZ)*GEPOP"},
    {"number": 82, "dependent": "GELCZ",
     "regressors": ["C", "GELCZ(-1)", "GELYZ"],
     "instruments": ["C", "GELCZ(-1)", "GELYZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1961Q1", "sample_end": "2017Q1",
     "lhs_transform": "GEC=EXP(GELCZ)*GEPOP"},
    {"number": 83, "dependent": "GELI",
     "regressors": ["C", "GELI(-1)", "GELY", "GERB(-1)"],
     "instruments": ["C", "GELI(-1)", "GELY", "GERB(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1961Q1", "sample_end": "2017Q1",
     "lhs_transform": "GEI=EXP(GELI)"},
    {"number": 84, "dependent": "GELPY",
     "regressors": ["GELPY(-1)", "C", "T", "GELPM", "GEZZ(-1)"],
     "instruments": ["C", "GELGZ", "GELEXZ(-1)", "GELPY(-1)", "GELYZ(-1)",
                     "GERB(-1)", "T", "GELPM(-1)", "GEZZ(-1)", "GEZZ(-2)",
                     "GELPY(-2)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q2", "sample_end": "2017Q2",
     "lhs_transform": "GEPY=EXP(GELPY)"},
    {"number": 85, "dependent": "GERS",
     "regressors": ["C", "GERS(-1)", "GEZZ", "USRS", "GEPCPY(-1)"],
     "instruments": ["C", "GELGZ", "GELEXZ(-1)", "GELPY(-1)", "GELYZ(-1)",
                     "GERB(-1)", "GERS(-1)", "GEZZ(-1)", "USRS(-1)", "GELPY(-2)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q2", "sample_end": "1998Q4",
     "lhs_transform": "GERS=(1-EU1)*GERS+EU1*EURS"},
    {"number": 86, "dependent": "GERBZ",
     "regressors": ["C", "GERBZZ", "GERSZ", "GERSZZ"],
     "instruments": ["C", "GELGZ", "GELEXZ(-1)", "GELPY(-1)", "GELYZ(-1)",
                     "GERB(-1)", "GERB(-2)", "GERS(-1)", "GERS(-2)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1961Q1", "sample_end": "1998Q4",
     "lhs_transform": "GERB=(1-EU1)*(GERBZ+GERS(-2))+EU1*EURB"},
    {"number": 87, "dependent": "GELE1",
     "regressors": ["C", "GELEA", "GELRSZ"],
     "instruments": ["C", "GELGZ", "GELEXZ(-1)", "GELPY(-1)", "GELYZ(-1)",
                     "GERB(-1)", "GELE1(-1)", "GELEA(-1)", "GELRSZ(-1)"],
     "has_ar1": True, "has_ar2": False,
     "sample_start": "1972Q2", "sample_end": "1998Q4",
     "lhs_transform": "GEE=(1-EU1)*(EXP(GELE1)*GEE(-1))+EU1*EUE"},
    {"number": 88, "dependent": "GELPXA",
     "regressors": ["GELPXB"],
     "instruments": ["GELPXB"],
     "has_ar1": False, "has_ar2": True,
     "sample_start": "1961Q1", "sample_end": "2016Q4",
     "lhs_transform": "GEPX=EXP(GELPXA)*(GEPW$*(GEE/GEE10))"},
    {"number": 89, "dependent": "GELJ1",
     "regressors": ["C", "T", "GELEXL(-1)", "GELY1"],
     "instruments": ["C", "GELGZ", "GELEXZ(-1)", "GELPY(-1)", "GELYZ(-1)",
                     "GERB(-1)", "GELJ1(-1)", "T", "GELEXL(-1)", "GELY1(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1963Q1", "sample_end": "2017Q2",
     "lhs_transform": "GEJ=EXP(GELJ1)*GEJ(-1)"},
]
EQUATIONS_GE: list[MCEquation] = [_eq_from_spec("GE", s) for s in _GE_SPECS]


# Italy — MC.INP 1103–1201. Pegged-to-Germany. EQ 97 uses ITLH1Z pre-transform.
_IT_SPECS: list[dict] = [
    {"number": 91, "dependent": "ITLIMZ",
     "regressors": ["C", "ITLIMZ(-1)", "ITLPYZZ", "ITLCIGZ"],
     "instruments": ["C", "ITLIMZ(-1)", "ITLPYZZ", "ITLCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1971Q2", "sample_end": "2017Q2",
     "lhs_transform": "ITIM=EXP(ITLIMZ)*ITPOP"},
    {"number": 92, "dependent": "ITLCZ",
     "regressors": ["C", "ITLCZ(-1)", "ITLYZ"],
     "instruments": ["C", "ITLCZ(-1)", "ITLYZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1961Q1", "sample_end": "2017Q4",
     "lhs_transform": "ITC=EXP(ITLCZ)*ITPOP"},
    {"number": 93, "dependent": "ITLI",
     "regressors": ["C", "ITLI(-1)", "ITLY"],
     "instruments": ["C", "ITLI(-1)", "ITLY"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1961Q1", "sample_end": "2017Q4",
     "lhs_transform": "ITI=EXP(ITLI)"},
    {"number": 94, "dependent": "ITLPY",
     "regressors": ["ITLPY(-1)", "C", "T", "ITLPM", "ITZZ(-1)"],
     "instruments": ["C", "ITLGZ", "ITLEXZ(-1)", "ITLPY(-1)", "ITLYZ(-1)",
                     "ITRS(-1)", "T", "ITLPM(-1)", "ITZZ(-1)", "ITZZ(-2)",
                     "ITLPY(-2)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q2", "sample_end": "2017Q2",
     "lhs_transform": "ITPY=EXP(ITLPY)"},
    {"number": 95, "dependent": "ITRS",
     "regressors": ["C", "ITRS(-1)", "ITPCPY", "ITZZ"],
     "instruments": ["C", "ITLGZ", "ITLEXZ(-1)", "ITLPY(-1)", "ITLYZ(-1)",
                     "ITRS(-1)", "ITRS(-2)", "ITZZ(-1)", "ITZZ(-2)", "ITLPY(-2)"],
     "has_ar1": True, "has_ar2": False,
     "sample_start": "1972Q2", "sample_end": "1998Q4",
     "lhs_transform": "ITRS=(1-EU1)*ITRS+EU1*EURS"},
    {"number": 97, "dependent": "ITLH1Z",
     "regressors": ["C"],
     "instruments": ["C", "ITLGZ", "ITLEXZ(-1)", "ITLPY(-1)", "ITLYZ(-1)",
                     "ITRS(-1)", "ITLH1(-1)", "ITLHA(-1)"],
     "has_ar1": True, "has_ar2": False,
     "sample_start": "1972Q2", "sample_end": "1998Q4",
     "lhs_transform": "ITH=(1-EU1)*(EXP(ITLH1Z+.050*ITLHA)*ITH(-1))+EU1*1"},
    {"number": 98, "dependent": "ITLPXA",
     "regressors": ["ITLPXB"],
     "instruments": ["ITLPXB"],
     "has_ar1": False, "has_ar2": True,
     "sample_start": "1961Q1", "sample_end": "2016Q4",
     "lhs_transform": "ITPX=EXP(ITLPXA)*(ITPW$*(ITE/ITE10))"},
    {"number": 99, "dependent": "ITLJ1",
     "regressors": ["C", "T", "ITLEXL(-1)", "ITLY1"],
     "instruments": ["C", "T", "ITLEXL(-1)", "ITLY1"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1963Q2", "sample_end": "2017Q4",
     "lhs_transform": "ITJ=EXP(ITLJ1)*ITJ(-1)"},
]
EQUATIONS_IT: list[MCEquation] = [_eq_from_spec("IT", s) for s in _IT_SPECS]


# Netherlands — MC.INP 1202–1314. Pegged-to-Germany.
_NE_SPECS: list[dict] = [
    {"number": 101, "dependent": "NELIMZ",
     "regressors": ["C", "NELIMZ(-1)", "NELPYZZ", "NELCIGZ"],
     "instruments": ["C", "NELIMZ(-1)", "NELPYZZ", "NELCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1971Q1", "sample_end": "2017Q4",
     "lhs_transform": "NEIM=EXP(NELIMZ)*NEPOP"},
    {"number": 102, "dependent": "NELCZ",
     "regressors": ["C", "NELCZ(-1)", "NELYZ", "NERB"],
     "instruments": ["C", "NELCZ(-1)", "NELYZ", "NERB"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1974Q1", "sample_end": "2017Q1",
     "lhs_transform": "NEC=EXP(NELCZ)*NEPOP"},
    {"number": 103, "dependent": "NELI",
     "regressors": ["C", "NELI(-1)", "NELY", "NERB(-1)"],
     "instruments": ["C", "NELGZ", "NELEXZ(-1)", "NELPY(-1)", "NELYZ(-1)",
                     "NERB(-1)", "NELI(-1)", "NELY(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1974Q1", "sample_end": "2017Q1",
     "lhs_transform": "NEI=EXP(NELI)"},
    {"number": 104, "dependent": "NELPY",
     "regressors": ["NELPY(-1)", "C", "T", "NELPM", "NEZZ(-1)"],
     "instruments": ["NELPY(-1)", "C", "T", "NELPM", "NEZZ(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1971Q1", "sample_end": "2017Q4",
     "lhs_transform": "NEPY=EXP(NELPY)"},
    {"number": 105, "dependent": "NERS",
     "regressors": ["C", "NERS(-1)", "NEZZ", "GERS", "USRS"],
     "instruments": ["C", "NELGZ", "NELEXZ(-1)", "NELPY(-1)", "NELYZ(-1)",
                     "NERB(-1)", "NERS(-1)", "NEZZ(-1)", "GERS(-1)", "USRS(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1973Q2", "sample_end": "1998Q4",
     "lhs_transform": "NERS=(1-EU1)*NERS+EU1*EURS"},
    {"number": 106, "dependent": "NERBZ",
     "regressors": ["C", "NERBZZ", "NERSZ", "NERSZZ"],
     "instruments": ["C", "NELGZ", "NELEXZ(-1)", "NELPY(-1)", "NELYZ(-1)",
                     "NERB(-1)", "NERB(-2)", "NERS(-1)", "NERS(-2)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1973Q3", "sample_end": "1998Q4",
     "lhs_transform": "NERB=(1-EU1)*(NERBZ+NERS(-2))+EU1*EURB"},
    {"number": 107, "dependent": "NELH1Z",
     "regressors": ["C", "NELRSZG"],
     "instruments": ["C", "NELGZ", "NELEXZ(-1)", "NELPY(-1)", "NELYZ(-1)",
                     "NERB(-1)", "NELH1(-1)", "NELHA(-1)", "NELRSZG(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1973Q2", "sample_end": "1998Q4",
     "lhs_transform": "NEH=(1-EU1)*(EXP(NELH1Z+.050*NELHA)*NEH(-1))+EU1*1"},
    {"number": 108, "dependent": "NELPXA",
     "regressors": ["NELPXB"],
     "instruments": ["NELPXB"],
     "has_ar1": False, "has_ar2": True,
     "sample_start": "1961Q1", "sample_end": "2016Q4",
     "lhs_transform": "NEPX=EXP(NELPXA)*(NEPW$*(NEE/NEE10))"},
    {"number": 109, "dependent": "NELJ1",
     "regressors": ["C", "T", "NELEXL(-1)", "NELY1"],
     "instruments": ["C", "T", "NELEXL(-1)", "NELY1"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1998Q2", "sample_end": "2017Q4",
     "lhs_transform": "NEJ=EXP(NELJ1)*NEJ(-1)"},
]
EQUATIONS_NE: list[MCEquation] = [_eq_from_spec("NE", s) for s in _NE_SPECS]


# Switzerland — MC.INP 1315–1413. Pegged-to-Germany. Has 10 equations
# including EQ 120 labor force. No Euro switch (ST is non-Euro).
_ST_SPECS: list[dict] = [
    {"number": 111, "dependent": "STLIMZ",
     "regressors": ["C", "STLIMZ(-1)", "STLCIGZ", "STLPYZZ"],
     "instruments": ["C", "STLIMZ(-1)", "STLCIGZ", "STLPYZZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1963Q1", "sample_end": "2017Q4",
     "lhs_transform": "STIM=EXP(STLIMZ)*STPOP"},
    {"number": 112, "dependent": "STLCZ",
     "regressors": ["C", "STLCZ(-1)", "STRB(-1)", "STLYZ"],
     "instruments": ["C", "STLCZ(-1)", "STRB(-1)", "STLYZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1961Q1", "sample_end": "2017Q2",
     "lhs_transform": "STC=EXP(STLCZ)*STPOP"},
    {"number": 113, "dependent": "STLI",
     "regressors": ["C", "STLI(-1)", "STLY", "STRB(-1)"],
     "instruments": ["C", "STLI(-1)", "STLY", "STRB(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1961Q1", "sample_end": "2017Q2",
     "lhs_transform": "STI=EXP(STLI)"},
    {"number": 114, "dependent": "STLPY",
     "regressors": ["STLPY(-1)", "C", "T", "STLPM", "STZZ(-1)"],
     "instruments": ["C", "STLGZ", "STLEXZ(-1)", "STLPY(-1)", "STLYZ(-1)",
                     "STRB(-1)", "T", "STZZ(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1963Q2", "sample_end": "2017Q3",
     "lhs_transform": "STPY=EXP(STLPY)"},
    {"number": 115, "dependent": "STRS",
     "regressors": ["C", "STRS(-1)", "STPCPY"],
     "instruments": ["C", "STLGZ", "STLEXZ(-1)", "STLPY(-1)", "STLYZ(-1)",
                     "STRB(-1)", "STRS(-1)", "STRS(-2)", "STZZ(-1)",
                     "STLPY(-2)"],
     "has_ar1": True, "has_ar2": False,
     "sample_start": "1972Q2", "sample_end": "2017Q2",
     "lhs_transform": "STRS=(ABS(STRS-.0)+STRS-.0)/2.+.0"},
    {"number": 116, "dependent": "STRBZ",
     "regressors": ["C", "STRBZZ", "STRSZ", "STRSZZ"],
     "instruments": ["C", "STLGZ", "STLEXZ(-1)", "STLPY(-1)", "STLYZ(-1)",
                     "STRB(-1)", "STRB(-2)", "STRS(-1)", "STRS(-2)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q2", "sample_end": "2017Q2",
     "lhs_transform": "STRB=STRBZ+STRS(-2)"},
    {"number": 117, "dependent": "STLH1Z",
     "regressors": ["C"],
     "instruments": ["C", "STLGZ", "STLEXZ(-1)", "STLPY(-1)", "STLYZ(-1)",
                     "STRB(-1)", "STLH1(-1)", "STLHA(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q2", "sample_end": "2017Q3",
     "lhs_transform": "STH=EXP(STLH1Z+.050*STLHA)*STH(-1)"},
    {"number": 118, "dependent": "STLPXA",
     "regressors": ["STLPXB"],
     "instruments": ["STLPXB"],
     "has_ar1": False, "has_ar2": True,
     "sample_start": "1961Q1", "sample_end": "2016Q4",
     "lhs_transform": "STPX=EXP(STLPXA)*(STPW$*(STE/STE10))"},
    {"number": 119, "dependent": "STLJ1",
     "regressors": ["C", "T", "STLEXL(-1)", "STLY1"],
     "instruments": ["C", "T", "STLEXL(-1)", "STLY1"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1991Q2", "sample_end": "2017Q4",
     "lhs_transform": "STJ=EXP(STLJ1)*STJ(-1)"},
    {"number": 120, "dependent": "STLL1Z",
     "regressors": ["C", "T", "STLL1Z(-1)", "STUR"],
     "instruments": ["C", "STLGZ", "STLEXZ(-1)", "STLPY(-1)", "STLYZ(-1)",
                     "STRB(-1)", "STLL1Z(-1)", "T", "STUR(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "2005Q2", "sample_end": "2017Q3",
     "lhs_transform": "STL1=EXP(STLL1Z)*STPOP1"},
]
EQUATIONS_ST: list[MCEquation] = [_eq_from_spec("ST", s) for s in _ST_SPECS]


# UK — MC.INP 1414–1536. Pegged-to-Germany. 10 equations.
_UK_SPECS: list[dict] = [
    {"number": 121, "dependent": "UKLIMZ",
     "regressors": ["C", "UKLIMZ(-1)", "UKLPYZZ", "UKLCIGZ"],
     "instruments": ["C", "UKLGZ", "UKLEXZ(-1)", "UKLPY(-1)", "UKLYZ(-1)",
                     "UKRB(-1)", "UKLIMZ(-1)", "UKLPYZZ(-1)", "UKLCIGZ(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1963Q2", "sample_end": "2017Q4",
     "lhs_transform": "UKIM=EXP(UKLIMZ)*UKPOP"},
    {"number": 122, "dependent": "UKLCZ",
     "regressors": ["C", "UKLCZ(-1)", "UKRB", "UKLYZ"],
     "instruments": ["C", "UKLCZ(-1)", "UKRB", "UKLYZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1961Q1", "sample_end": "2017Q4",
     "lhs_transform": "UKC=EXP(UKLCZ)*UKPOP"},
    {"number": 123, "dependent": "UKLI",
     "regressors": ["C", "UKLI(-1)", "UKLY", "UKRB(-1)"],
     "instruments": ["C", "UKLI(-1)", "UKLY", "UKRB(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1961Q1", "sample_end": "2017Q4",
     "lhs_transform": "UKI=EXP(UKLI)"},
    {"number": 124, "dependent": "UKLPY",
     "regressors": ["UKLPY(-1)", "C", "T", "UKLPM(-1)", "UKZZ(-1)"],
     "instruments": ["C", "UKLGZ", "UKLEXZ(-1)", "UKLPY(-1)", "UKLYZ(-1)",
                     "UKRB(-1)", "T", "UKLPM(-1)", "UKZZ(-1)", "UKZZ(-2)",
                     "UKLPY(-2)", "UKLPM(-2)"],
     "has_ar1": True, "has_ar2": False,
     "sample_start": "1963Q4", "sample_end": "2017Q4",
     "lhs_transform": "UKPY=EXP(UKLPY)"},
    {"number": 125, "dependent": "UKRS",
     "regressors": ["C", "UKRS(-1)", "UKPCPY", "UKZZ", "USRS"],
     "instruments": ["C", "UKLGZ", "UKLEXZ(-1)", "UKLPY(-1)", "UKLYZ(-1)",
                     "UKRB(-1)", "UKRS(-1)", "UKZZ(-1)", "USRS(-1)",
                     "UKPCPY(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q2", "sample_end": "2016Q3",
     "lhs_transform": "UKRS=(ABS(UKRS-.0)+UKRS-.0)/2.+.0"},
    {"number": 126, "dependent": "UKRBZ",
     "regressors": ["C", "UKRBZZ", "UKRSZ", "UKRSZZ"],
     "instruments": ["C", "UKLGZ", "UKLEXZ(-1)", "UKLPY(-1)", "UKLYZ(-1)",
                     "UKRB(-1)", "UKRB(-2)", "UKRS(-1)", "UKRS(-2)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q2", "sample_end": "2016Q3",
     "lhs_transform": "UKRB=UKRBZ+UKRS(-2)"},
    {"number": 127, "dependent": "UKLH1Z",
     "regressors": ["C", "UKLRSZG"],
     "instruments": ["C", "UKLGZ", "UKLEXZ(-1)", "UKLPY(-1)", "UKLYZ(-1)",
                     "UKRB(-1)", "UKLH1(-1)", "UKLHA(-1)", "UKLRSZG(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q2", "sample_end": "2016Q3",
     "lhs_transform": "UKH=EXP(UKLH1Z+.050*UKLHA)*UKH(-1)"},
    {"number": 128, "dependent": "UKLPXA",
     "regressors": ["UKLPXB"],
     "instruments": ["UKLPXB"],
     "has_ar1": False, "has_ar2": True,
     "sample_start": "1961Q1", "sample_end": "2016Q4",
     "lhs_transform": "UKPX=EXP(UKLPXA)*(UKPW$*(UKE/UKE10))"},
    {"number": 129, "dependent": "UKLJ1",
     "regressors": ["C", "T", "UKLEXL(-1)", "UKLY1"],
     "instruments": ["C", "T", "UKLEXL(-1)", "UKLY1"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1978Q3", "sample_end": "2017Q4",
     "lhs_transform": "UKJ=EXP(UKLJ1)*UKJ(-1)"},
    {"number": 130, "dependent": "UKLL1Z",
     "regressors": ["C", "T", "UKLL1Z(-1)", "UKUR"],
     "instruments": ["C", "T", "UKLL1Z(-1)", "UKUR"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1999Q3", "sample_end": "2017Q3",
     "lhs_transform": "UKL1=EXP(UKLL1Z)*UKPOP1"},
]
EQUATIONS_UK: list[MCEquation] = [_eq_from_spec("UK", s) for s in _UK_SPECS]


# Finland — MC.INP 1537–1627. Pegged-to-Germany, no LH1Z pre-transform.
_FI_SPECS: list[dict] = [
    {"number": 131, "dependent": "FILIMZ",
     "regressors": ["C", "FILIMZ(-1)", "FILCIGZ", "FILPYZZ"],
     "instruments": ["C", "FILGZ", "FILEXZ(-1)", "FILPY(-1)", "FILYZ(-1)",
                     "FILIMZ(-1)", "FILPYZZ(-1)", "FILCIGZ(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1961Q1", "sample_end": "2017Q4",
     "lhs_transform": "FIIM=EXP(FILIMZ)*FIPOP"},
    {"number": 132, "dependent": "FILCZ",
     "regressors": ["C", "FILCZ(-1)", "FIRS", "FILYZ"],
     "instruments": ["C", "FILGZ", "FILEXZ(-1)", "FILPY(-1)", "FILYZ(-1)",
                     "FILCZ(-1)", "FIRS(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1978Q2", "sample_end": "2017Q1",
     "lhs_transform": "FIC=EXP(FILCZ)*FIPOP"},
    {"number": 134, "dependent": "FILPY",
     "regressors": ["FILPY(-1)", "C", "T", "FILPM", "FIZZ"],
     "instruments": ["C", "FILGZ", "FILEXZ(-1)", "FILPY(-1)", "FILYZ(-1)",
                     "FIRS(-1)", "FILPM(-1)", "T", "FIZZ(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1978Q2", "sample_end": "2017Q2",
     "lhs_transform": "FIPY=EXP(FILPY)"},
    {"number": 135, "dependent": "FIRS",
     "regressors": ["C", "FIRS(-1)", "FIZZ"],
     "instruments": ["C", "FILGZ", "FILEXZ(-1)", "FILPY(-1)", "FILYZ(-1)",
                     "FIRS(-1)", "FIZZ(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1979Q1", "sample_end": "1998Q4",
     "lhs_transform": "FIRS=(1-EU1)*FIRS+EU1*EURS"},
    {"number": 137, "dependent": "FILH1",
     "regressors": ["C", "FILHA", "FILRSZG"],
     "instruments": ["C", "FILGZ", "FILEXZ(-1)", "FILPY(-1)", "FILYZ(-1)",
                     "FIRS(-1)", "FILH1(-1)", "FILHA(-1)", "FILRSZG(-1)"],
     "has_ar1": True, "has_ar2": False,
     "sample_start": "1978Q3", "sample_end": "1998Q4",
     "lhs_transform": "FIH=(1-EU1)*(EXP(FILH1)*FIH(-1))+EU1*1"},
    {"number": 138, "dependent": "FILPXA",
     "regressors": ["FILPXB"],
     "instruments": ["FILPXB"],
     "has_ar1": False, "has_ar2": True,
     "sample_start": "1961Q1", "sample_end": "2016Q4",
     "lhs_transform": "FIPX=EXP(FILPXA)*(FIPW$*(FIE/FIE10))"},
    {"number": 139, "dependent": "FILJ1",
     "regressors": ["C", "T", "FILEXL(-1)", "FILY1"],
     "instruments": ["C", "FILGZ", "FILEXZ(-1)", "FILPY(-1)", "FILYZ(-1)",
                     "FIRS(-1)", "FILJ1(-1)", "T", "FILEXL(-1)", "FILEXL(-2)",
                     "FILY1(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1978Q2", "sample_end": "2017Q2",
     "lhs_transform": "FIJ=EXP(FILJ1)*FIJ(-1)"},
    {"number": 140, "dependent": "FILL1Z",
     "regressors": ["C", "T", "FILL1Z(-1)", "FIUR"],
     "instruments": ["C", "FILGZ", "FILEXZ(-1)", "FILPY(-1)", "FILYZ(-1)",
                     "FIRS(-1)", "FILL1Z(-1)", "T", "FIUR(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "2000Q2", "sample_end": "2017Q2",
     "lhs_transform": "FIL1=EXP(FILL1Z)*FIPOP1"},
]
EQUATIONS_FI: list[MCEquation] = [_eq_from_spec("FI", s) for s in _FI_SPECS]


# Australia — MC.INP 1628–1727. Floating vs US, with LE1 exchange rate.
_AS_SPECS: list[dict] = [
    {"number": 141, "dependent": "ASLIMZ",
     "regressors": ["C", "ASLIMZ(-1)", "ASLPYZZ", "ASLCIGZ"],
     "instruments": ["C", "ASLGZ", "ASLEXZ(-1)", "ASLPY(-1)", "ASLYZ(-1)",
                     "ASRB(-1)", "ASLIMZ(-1)", "ASLPYZZ(-1)", "ASLCIGZ(-1)",
                     "ASLIMZ(-2)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1969Q4", "sample_end": "2017Q4",
     "lhs_transform": "ASIM=EXP(ASLIMZ)*ASPOP"},
    {"number": 142, "dependent": "ASLCZ",
     "regressors": ["C", "ASLCZ(-1)", "ASLYZ"],
     "instruments": ["C", "ASLGZ", "ASLEXZ(-1)", "ASLPY(-1)", "ASLYZ(-1)",
                     "ASRB(-1)", "ASLCZ(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1969Q4", "sample_end": "2017Q4",
     "lhs_transform": "ASC=EXP(ASLCZ)*ASPOP"},
    {"number": 143, "dependent": "ASLI",
     "regressors": ["C", "ASLI(-1)", "ASLY", "ASRB(-1)"],
     "instruments": ["C", "ASLGZ", "ASLEXZ(-1)", "ASLPY(-1)", "ASLYZ(-1)",
                     "ASRB(-1)", "ASLI(-1)", "ASLY(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1969Q4", "sample_end": "2017Q4",
     "lhs_transform": "ASI=EXP(ASLI)"},
    {"number": 144, "dependent": "ASLPY",
     "regressors": ["ASLPY(-1)", "C", "T", "ASZZ(-1)"],
     "instruments": ["ASLPY(-1)", "C", "T", "ASZZ(-1)"],
     "has_ar1": True, "has_ar2": False,
     "sample_start": "1970Q1", "sample_end": "2017Q4",
     "lhs_transform": "ASPY=EXP(ASLPY)"},
    {"number": 145, "dependent": "ASRS",
     "regressors": ["C", "ASRS(-1)", "ASPCPY", "ASZZ", "USRS"],
     "instruments": ["C", "ASLGZ", "ASLEXZ(-1)", "ASLPY(-1)", "ASLYZ(-1)",
                     "ASRB(-1)", "ASRS(-1)", "ASZZ(-1)", "USRS(-1)",
                     "ASLPY(-2)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1976Q4", "sample_end": "2017Q4",
     "lhs_transform": "ASRS=(ABS(ASRS-.0)+ASRS-.0)/2.+.0"},
    {"number": 146, "dependent": "ASRBZ",
     "regressors": ["C", "ASRBZZ", "ASRSZ", "ASRSZZ"],
     "instruments": ["C", "ASLGZ", "ASLEXZ(-1)", "ASLPY(-1)", "ASLYZ(-1)",
                     "ASRB(-1)", "ASRB(-2)", "ASRS(-1)", "ASRS(-2)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1977Q1", "sample_end": "2017Q4",
     "lhs_transform": "ASRB=ASRBZ+ASRS(-2)"},
    {"number": 147, "dependent": "ASLE1",
     "regressors": ["C", "ASLEA"],
     "instruments": ["C", "ASLGZ", "ASLEXZ(-1)", "ASLPY(-1)", "ASLYZ(-1)",
                     "ASRB(-1)", "ASLE1(-1)", "ASLEA(-1)"],
     "has_ar1": True, "has_ar2": False,
     "sample_start": "1972Q2", "sample_end": "2017Q4",
     "lhs_transform": "ASE=EXP(ASLE1)*ASE(-1)"},
    {"number": 148, "dependent": "ASLPXA",
     "regressors": ["ASLPXB"],
     "instruments": ["ASLPXB"],
     "has_ar1": False, "has_ar2": True,
     "sample_start": "1961Q1", "sample_end": "2016Q4",
     "lhs_transform": "ASPX=EXP(ASLPXA)*(ASPW$*(ASE/ASE10))"},
    {"number": 149, "dependent": "ASLJ1",
     "regressors": ["C", "T", "ASLEXL(-1)", "ASLY1"],
     "instruments": ["C", "T", "ASLEXL(-1)", "ASLY1"],
     "has_ar1": True, "has_ar2": False,
     "sample_start": "1967Q4", "sample_end": "2017Q4",
     "lhs_transform": "ASJ=EXP(ASLJ1)*ASJ(-1)"},
    {"number": 150, "dependent": "ASLL1Z",
     "regressors": ["C", "T", "ASLL1Z(-1)", "ASUR"],
     "instruments": ["C", "ASLGZ", "ASLEXZ(-1)", "ASLPY(-1)", "ASLYZ(-1)",
                     "ASRB(-1)", "ASLL1Z(-1)", "T", "ASUR(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1978Q2", "sample_end": "2017Q4",
     "lhs_transform": "ASL1=EXP(ASLL1Z)*ASPOP1"},
]
EQUATIONS_AS: list[MCEquation] = [_eq_from_spec("AS", s) for s in _AS_SPECS]


# South Africa (SO) — MC.INP 1728–1793. Floating vs US; 5 equations only.
_SO_SPECS: list[dict] = [
    {"number": 151, "dependent": "SOLIMZ",
     "regressors": ["C", "SOLIMZ(-1)", "SOLCIGZ"],
     "instruments": ["C", "SOLIMZ(-1)", "SOLCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1961Q1", "sample_end": "2016Q4",
     "lhs_transform": "SOIM=EXP(SOLIMZ)*SOPOP"},
    {"number": 152, "dependent": "SOLCZ",
     "regressors": ["C", "SOLCZ(-1)", "SORS(-1)", "SOLYZ"],
     "instruments": ["C", "SOLCZ(-1)", "SORS(-1)", "SOLYZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1961Q1", "sample_end": "2016Q4",
     "lhs_transform": "SOC=EXP(SOLCZ)*SOPOP"},
    {"number": 153, "dependent": "SOLI",
     "regressors": ["C", "SOLI(-1)", "SOLY", "SORB(-1)"],
     "instruments": ["C", "SOLGZ", "SOLEXZ(-1)", "SOLPY(-1)", "SOLYZ(-1)",
                     "SORB(-1)", "SOLI(-1)", "SOLY(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1961Q1", "sample_end": "2016Q4",
     "lhs_transform": "SOI=EXP(SOLI)"},
    {"number": 155, "dependent": "SORS",
     "regressors": ["C", "SORS(-1)", "USRS"],
     "instruments": ["C", "SOLGZ", "SOLEXZ(-1)", "SOLPY(-1)", "SOLYZ(-1)",
                     "SORB(-1)", "SORS(-1)", "SORS(-2)", "USRS(-1)"],
     "has_ar1": True, "has_ar2": False,
     "sample_start": "1972Q2", "sample_end": "2016Q4",
     "lhs_transform": "SORS=(ABS(SORS-.0)+SORS-.0)/2.+.0"},
    {"number": 156, "dependent": "SORBZ",
     "regressors": ["C", "SORBZZ", "SORSZ", "SORSZZ"],
     "instruments": ["C", "SOLGZ", "SOLEXZ(-1)", "SOLPY(-1)", "SOLYZ(-1)",
                     "SORB(-1)", "SORB(-2)", "SORS(-1)", "SORS(-2)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1961Q1", "sample_end": "2016Q4",
     "lhs_transform": "SORB=SORBZ+SORS(-2)"},
]
EQUATIONS_SO: list[MCEquation] = [_eq_from_spec("SO", s) for s in _SO_SPECS]


# South Korea (KO) — MC.INP 1794–1919. Floating vs US; 7 equations.
_KO_SPECS: list[dict] = [
    {"number": 161, "dependent": "KOLIMZ",
     "regressors": ["C", "KOLIMZ(-1)", "KOLCIGZ"],
     "instruments": ["C", "KOLIMZ(-1)", "KOLCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1961Q1", "sample_end": "2017Q2",
     "lhs_transform": "KOIM=EXP(KOLIMZ)*KOPOP"},
    {"number": 162, "dependent": "KOLCZ",
     "regressors": ["C", "KOLCZ(-1)", "KOLYZ", "KORB(-1)"],
     "instruments": ["C", "KOLGZ", "KOLEXZ(-1)", "KOLPY(-1)", "KOLYZ(-1)",
                     "KORB(-1)", "KOLCZ(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1973Q4", "sample_end": "2017Q2",
     "lhs_transform": "KOC=EXP(KOLCZ)*KOPOP"},
    {"number": 163, "dependent": "KOLI",
     "regressors": ["C", "KOLI(-1)", "KOLY", "KORB(-1)"],
     "instruments": ["C", "KOLI(-1)", "KOLY", "KORB(-1)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1973Q4", "sample_end": "2017Q2",
     "lhs_transform": "KOI=EXP(KOLI)"},
    {"number": 164, "dependent": "KOLPY",
     "regressors": ["KOLPY(-1)", "C", "T", "KOZZ(-1)"],
     "instruments": ["C", "KOLGZ", "KOLEXZ(-1)", "KOLPY(-1)", "KOLYZ(-1)",
                     "KORB(-1)", "KOLPY(-2)", "T", "KOLPM(-1)", "KOZZ(-1)",
                     "KOZZ(-2)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1973Q4", "sample_end": "2017Q2",
     "lhs_transform": "KOPY=EXP(KOLPY)"},
    {"number": 166, "dependent": "KORBZ",
     "regressors": ["C", "KORBZZ", "KORSZ", "KORSZZ"],
     "instruments": ["C", "KOLGZ", "KOLEXZ(-1)", "KOLPY(-1)", "KOLYZ(-1)",
                     "KORB(-1)", "KORB(-2)", "KORS(-1)", "KORS(-2)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1977Q2", "sample_end": "2017Q2",
     "lhs_transform": "KORB=KORBZ+KORS(-2)"},
    {"number": 167, "dependent": "KOLE1",
     "regressors": ["C", "KOLEA"],
     "instruments": ["C", "KOLGZ", "KOLEXZ(-1)", "KOLPY(-1)", "KOLYZ(-1)",
                     "KORB(-1)", "KOLE1(-1)", "KOLEA(-1)"],
     "has_ar1": True, "has_ar2": False,
     "sample_start": "1974Q1", "sample_end": "2017Q2",
     "lhs_transform": "KOE=EXP(KOLE1)*KOE(-1)"},
    {"number": 168, "dependent": "KOLPXA",
     "regressors": ["KOLPXB"],
     "instruments": ["KOLPXB"],
     "has_ar1": False, "has_ar2": True,
     "sample_start": "1961Q1", "sample_end": "2016Q4",
     "lhs_transform": "KOPX=EXP(KOLPXA)*(KOPW$*(KOE/KOE10))"},
]
EQUATIONS_KO: list[MCEquation] = [_eq_from_spec("KO", s) for s in _KO_SPECS]


# ---------------------------------------------------------------------------
# Annual-lag countries (BE onwards) — 23 countries, ~110 equations
# ---------------------------------------------------------------------------
#
# These blocks use ``(-4)`` lags throughout and reference trailing 4-quarter
# averages (``GEPYA``, ``GERSA``, ``USPYA``, ``USRSA``, ``EURSA``, ``EURBA``)
# in place of the current-quarter values. Fair's xxLPXA export-price
# equations (x8 suffix) are NLEQ in these blocks — not estimated — so
# they're omitted from the registry.

# Compact spec dicts are grouped by country. Each entry has the same
# schema as the quarterly factory input (``_eq_from_spec``).
_BE_SPECS: list[dict] = [
    {"number": 181, "dependent": "BELIMZ",
     "regressors": ["C", "BELIMZ(-4)", "BELPYZZ", "BELCIGZ"],
     "instruments": ["C", "BELGZ", "BELEXZ(-4)", "BELPY(-4)", "BELYZ(-4)",
                     "BERB(-4)", "BELIMZ(-4)", "BELPYZZ(-4)", "BELCIGZ(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "BEIM=EXP(BELIMZ)*BEPOP"},
    {"number": 182, "dependent": "BELCZ",
     "regressors": ["C", "BELCZ(-4)", "BELYZ"],
     "instruments": ["C", "BELGZ", "BELEXZ(-4)", "BELPY(-4)", "BELYZ(-4)",
                     "BERB(-4)", "BELCZ(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "BEC=EXP(BELCZ)*BEPOP"},
    {"number": 183, "dependent": "BELI",
     "regressors": ["C", "BELI(-4)", "BELY", "BERB"],
     "instruments": ["C", "BELGZ", "BELEXZ(-4)", "BELPY(-4)", "BELYZ(-4)",
                     "BERB(-4)", "BELI(-4)", "BELY(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "BEI=EXP(BELI)"},
    {"number": 184, "dependent": "BELPY",
     "regressors": ["BELPY(-4)", "C", "BET", "BELPM", "BEZZ(-4)"],
     "instruments": ["BELPY(-4)", "C", "BET", "BELPM", "BEZZ(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "BEPY=EXP(BELPY)"},
    {"number": 185, "dependent": "BERS",
     "regressors": ["C", "BERS(-4)", "GERSA", "BEPCPY", "BEZZ"],
     "instruments": ["C", "BERS(-4)", "GERSA", "BEPCPY", "BEZZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "1998Q4",
     "lhs_transform": "BERS=(1-EU1)*BERS+EU1*EURSA"},
    {"number": 186, "dependent": "BERBZ",
     "regressors": ["C", "BERBZZ", "BERSZ"],
     "instruments": ["C", "BERBZZ", "BERSZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "1998Q4",
     "lhs_transform": "BERB=(1-EU1)*(BERBZ+BERS(-4))+EU1*EURBA"},
    {"number": 187, "dependent": "BELH1",
     "regressors": ["C", "BELHA"],
     "instruments": ["C", "BELHA"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "1998Q4",
     "lhs_transform": "BEH=(1-EU1)*(EXP(BELH1)*BEH(-4))+EU1*1"},
    {"number": 189, "dependent": "BELJ1",
     "regressors": ["C", "BET", "BELEXL(-4)", "BELY1"],
     "instruments": ["C", "BET", "BELEXL(-4)", "BELY1"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1985Q1", "sample_end": "2016Q4",
     "lhs_transform": "BEJ=EXP(BELJ1)*BEJ(-4)"},
    {"number": 190, "dependent": "BELL1Z",
     "regressors": ["C", "BET", "BELL1Z(-4)", "BEUR(-4)"],
     "instruments": ["C", "BET", "BELL1Z(-4)", "BEUR(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1985Q1", "sample_end": "2016Q4",
     "lhs_transform": "BEL1=EXP(BELL1Z)*BEPOP1"},
]
EQUATIONS_BE: list[MCEquation] = [_eq_from_spec("BE", s) for s in _BE_SPECS]

_DE_SPECS: list[dict] = [
    {"number": 191, "dependent": "DELIMZ",
     "regressors": ["C", "DELIMZ(-4)", "DELCIGZ", "DELPYZZ"],
     "instruments": ["C", "DELGZ", "DELEXZ(-4)", "DELPY(-4)", "DELYZ(-4)",
                     "DERS(-4)", "DELIMZ(-4)", "DELPYZZ(-4)", "DELCIGZ(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1973Q1", "sample_end": "2017Q4",
     "lhs_transform": "DEIM=EXP(DELIMZ)*DEPOP"},
    {"number": 192, "dependent": "DELCZ",
     "regressors": ["C", "DELCZ(-4)", "DELYZ", "DERS"],
     "instruments": ["C", "DELGZ", "DELEXZ(-4)", "DELPY(-4)", "DELYZ(-4)",
                     "DERS(-4)", "DELCZ(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1973Q1", "sample_end": "2017Q4",
     "lhs_transform": "DEC=EXP(DELCZ)*DEPOP"},
    {"number": 193, "dependent": "DELI",
     "regressors": ["C", "DELI(-4)", "DELY", "DERS"],
     "instruments": ["C", "DELGZ", "DELEXZ(-4)", "DELPY(-4)", "DELYZ(-4)",
                     "DERS(-4)", "DELI(-4)", "DELY(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1973Q1", "sample_end": "2017Q4",
     "lhs_transform": "DEI=EXP(DELI)"},
    {"number": 194, "dependent": "DELPY",
     "regressors": ["DELPY(-4)", "C", "BET", "DELPM"],
     "instruments": ["DELPY(-4)", "C", "BET", "DELPM"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1973Q1", "sample_end": "2017Q4",
     "lhs_transform": "DEPY=EXP(DELPY)"},
    {"number": 195, "dependent": "DERS",
     "regressors": ["C", "DERS(-4)", "GERSA", "DEZZ", "DEPCPY"],
     "instruments": ["C", "DERS(-4)", "GERSA", "DEZZ", "DEPCPY"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1973Q1", "sample_end": "2017Q4",
     "lhs_transform": "DERS=(ABS(DERS-.0)+DERS-.0)/2.+.0"},
    {"number": 196, "dependent": "DERBZ",
     "regressors": ["C", "DERBZZ", "DERSZ"],
     "instruments": ["C", "DERBZZ", "DERSZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1985Q1", "sample_end": "2017Q4",
     "lhs_transform": "DERB=DERBZ+DERS(-4)"},
    {"number": 197, "dependent": "DELH1Z",
     "regressors": ["C"],
     "instruments": ["C"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1973Q1", "sample_end": "2017Q4",
     "lhs_transform": "DEH=EXP(DELH1Z+.050*DELHA)*DEH(-4)"},
    {"number": 199, "dependent": "DELJ1",
     "regressors": ["C", "BET", "DELEXL(-4)", "DELY1"],
     "instruments": ["C", "BET", "DELEXL(-4)", "DELY1"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1985Q1", "sample_end": "2016Q4",
     "lhs_transform": "DEJ=EXP(DELJ1)*DEJ(-4)"},
]
EQUATIONS_DE: list[MCEquation] = [_eq_from_spec("DE", s) for s in _DE_SPECS]

_NO_SPECS: list[dict] = [
    {"number": 201, "dependent": "NOLIMZ",
     "regressors": ["C", "NOLIMZ(-4)", "NOLPYZZ", "NOLCIGZ"],
     "instruments": ["C", "NOLGZ", "NOLEXZ(-4)", "NOLPY(-4)", "NOLYZ(-4)",
                     "NORB(-4)", "NOLIMZ(-4)", "NOLPYZZ(-4)", "NOLCIGZ(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "NOIM=EXP(NOLIMZ)*NOPOP"},
    {"number": 202, "dependent": "NOLCZ",
     "regressors": ["C", "NOLCZ(-4)", "NOLYZ", "NORS"],
     "instruments": ["C", "NOLGZ", "NOLEXZ(-4)", "NOLPY(-4)", "NOLYZ(-4)",
                     "NORB(-4)", "NOLCZ(-4)", "NORS(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "NOC=EXP(NOLCZ)*NOPOP"},
    {"number": 203, "dependent": "NOLI",
     "regressors": ["C", "NOLI(-4)", "NOLY", "NORS"],
     "instruments": ["C", "NOLGZ", "NOLEXZ(-4)", "NOLPY(-4)", "NOLYZ(-4)",
                     "NORB(-4)", "NOLI(-4)", "NORS(-4)", "NOLY(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "NOI=EXP(NOLI)"},
    {"number": 204, "dependent": "NOLPY",
     "regressors": ["NOLPY(-4)", "C", "BET", "NOLPM", "NOZZ(-4)"],
     "instruments": ["NOLPY(-4)", "C", "BET", "NOLPM", "NOZZ(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "NOPY=EXP(NOLPY)"},
    {"number": 205, "dependent": "NORS",
     "regressors": ["C", "NORS(-4)", "GERSA", "NOZZ"],
     "instruments": ["C", "NORS(-4)", "GERSA", "NOZZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "NORS=(ABS(NORS-.0)+NORS-.0)/2.+.0"},
    {"number": 206, "dependent": "NORBZ",
     "regressors": ["C", "NORBZZ", "NORSZ"],
     "instruments": ["C", "NORBZZ", "NORSZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "NORB=NORBZ+NORS(-4)"},
    {"number": 207, "dependent": "NOLH1Z",
     "regressors": ["C"],
     "instruments": ["C"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "NOH=EXP(NOLH1Z+.050*NOLHA)*NOH(-4)"},
    {"number": 209, "dependent": "NOLJ1",
     "regressors": ["C", "BET", "NOLEXL(-4)", "NOLY1"],
     "instruments": ["C", "BET", "NOLEXL(-4)", "NOLY1"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1974Q1", "sample_end": "2016Q4",
     "lhs_transform": "NOJ=EXP(NOLJ1)*NOJ(-4)"},
    {"number": 210, "dependent": "NOLL1Z",
     "regressors": ["C", "BET", "NOLL1Z(-4)", "NOUR"],
     "instruments": ["C", "BET", "NOLL1Z(-4)", "NOUR"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1974Q1", "sample_end": "2016Q4",
     "lhs_transform": "NOL1=EXP(NOLL1Z)*NOPOP1"},
]
EQUATIONS_NO: list[MCEquation] = [_eq_from_spec("NO", s) for s in _NO_SPECS]

_SW_SPECS: list[dict] = [
    {"number": 211, "dependent": "SWLIMZ",
     "regressors": ["SWLIMZ(-4)", "SWLCIGZ", "C"],
     "instruments": ["C", "SWLGZ", "SWLEXZ(-4)", "SWLPY(-4)", "SWLYZ(-4)",
                     "SWRS(-4)", "SWLIMZ(-4)", "SWLPYZZ(-4)", "SWLCIGZ(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "SWIM=EXP(SWLIMZ)*SWPOP"},
    {"number": 212, "dependent": "SWLCZ",
     "regressors": ["C", "SWLCZ(-4)", "SWLYZ", "SWRS"],
     "instruments": ["C", "SWLGZ", "SWLEXZ(-4)", "SWLPY(-4)", "SWLYZ(-4)",
                     "SWRS(-4)", "SWLCZ(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "SWC=EXP(SWLCZ)*SWPOP"},
    {"number": 213, "dependent": "SWLI",
     "regressors": ["C", "SWLI(-4)", "SWLY", "SWRS"],
     "instruments": ["C", "SWLGZ", "SWLEXZ(-4)", "SWLPY(-4)", "SWLYZ(-4)",
                     "SWRS(-4)", "SWLI(-4)", "SWLY(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "SWI=EXP(SWLI)"},
    {"number": 214, "dependent": "SWLPY",
     "regressors": ["SWLPY(-4)", "C", "BET", "SWLPM", "SWZZ(-4)"],
     "instruments": ["C", "SWLGZ", "SWLEXZ(-4)", "SWLPY(-4)", "SWLYZ(-4)",
                     "SWRS(-4)", "BET", "SWLPM(-4)", "SWZZ(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "SWPY=EXP(SWLPY)"},
    {"number": 215, "dependent": "SWRS",
     "regressors": ["C", "SWRS(-4)", "USRSA", "SWPCPY", "SWZZ"],
     "instruments": ["C", "SWRS(-4)", "USRSA", "SWPCPY", "SWZZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "SWRS=(ABS(SWRS-.0)+SWRS-.0)/2.+.0"},
    {"number": 217, "dependent": "SWLH1",
     "regressors": ["C", "SWLHA"],
     "instruments": ["C", "SWLHA"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "SWH=EXP(SWLH1)*SWH(-4)"},
    {"number": 219, "dependent": "SWLJ1",
     "regressors": ["C", "BET", "SWLEXL(-4)", "SWLY1"],
     "instruments": ["C", "BET", "SWLEXL(-4)", "SWLY1"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "SWJ=EXP(SWLJ1)*SWJ(-4)"},
    {"number": 220, "dependent": "SWLL1Z",
     "regressors": ["C", "BET", "SWLL1Z(-4)", "SWUR"],
     "instruments": ["C", "BET", "SWLL1Z(-4)", "SWUR"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "SWL1=EXP(SWLL1Z)*SWPOP1"},
]
EQUATIONS_SW: list[MCEquation] = [_eq_from_spec("SW", s) for s in _SW_SPECS]

_GR_SPECS: list[dict] = [
    {"number": 221, "dependent": "GRLIMZ",
     "regressors": ["C", "GRLIMZ(-4)", "GRLPYZZ", "GRLCIGZ"],
     "instruments": ["C", "GRLIMZ(-4)", "GRLPYZZ", "GRLCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "GRIM=EXP(GRLIMZ)*GRPOP"},
    {"number": 222, "dependent": "GRLCZ",
     "regressors": ["C", "GRLCZ(-4)", "GRLYZ"],
     "instruments": ["C", "GRLCZ(-4)", "GRLYZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "GRC=EXP(GRLCZ)*GRPOP"},
    {"number": 223, "dependent": "GRLI",
     "regressors": ["C", "GRLI(-4)", "GRLY"],
     "instruments": ["C", "GRLI(-4)", "GRLY"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "GRI=EXP(GRLI)"},
    {"number": 227, "dependent": "GRLH1",
     "regressors": ["C", "GRLHA"],
     "instruments": ["C", "GRLHA"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2000Q4",
     "lhs_transform": "GRH=(1-EU2)*(EXP(GRLH1)*GRH(-4))+EU2*1"},
]
EQUATIONS_GR: list[MCEquation] = [_eq_from_spec("GR", s) for s in _GR_SPECS]

_IR_SPECS: list[dict] = [
    {"number": 231, "dependent": "IRLIMZ",
     "regressors": ["C", "IRLIMZ(-4)", "IRLPYZZ", "IRLCIGZ"],
     "instruments": ["C", "IRLIMZ(-4)", "IRLPYZZ", "IRLCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "IRIM=EXP(IRLIMZ)*IRPOP"},
    {"number": 232, "dependent": "IRLCZ",
     "regressors": ["C", "IRLCZ(-4)", "IRLYZ", "IRRS"],
     "instruments": ["C", "IRLGZ", "IRLEXZ(-4)", "IRLPY(-4)", "IRLYZ(-4)",
                     "IRRS(-4)", "IRLCZ(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1974Q1", "sample_end": "2016Q4",
     "lhs_transform": "IRC=EXP(IRLCZ)*IRPOP"},
    {"number": 233, "dependent": "IRLI",
     "regressors": ["C", "IRLI(-4)", "IRLY"],
     "instruments": ["C", "IRLI(-4)", "IRLY"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1973Q1", "sample_end": "2016Q4",
     "lhs_transform": "IRI=EXP(IRLI)"},
    {"number": 234, "dependent": "IRLPY",
     "regressors": ["IRLPY(-4)", "C", "BET", "IRLPM(-4)", "IRZZ(-4)"],
     "instruments": ["IRLPY(-4)", "C", "BET", "IRLPM(-4)", "IRZZ(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1974Q1", "sample_end": "2017Q4",
     "lhs_transform": "IRPY=EXP(IRLPY)"},
    {"number": 235, "dependent": "IRRS",
     "regressors": ["C", "IRRS(-4)", "IRPCPY", "GERSA", "USRSA"],
     "instruments": ["C", "IRRS(-4)", "IRPCPY", "GERSA", "USRSA"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1974Q1", "sample_end": "1998Q4",
     "lhs_transform": "IRRS=(1-EU1)*IRRS+EU1*EURSA"},
    {"number": 237, "dependent": "IRLH1",
     "regressors": ["C", "IRLHA"],
     "instruments": ["C", "IRLHA"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "1998Q4",
     "lhs_transform": "IRH=(1-EU1)*(EXP(IRLH1)*IRH(-4))+EU1*1"},
    {"number": 239, "dependent": "IRLJ1",
     "regressors": ["C", "BET", "IRLEXL(-4)", "IRLY1"],
     "instruments": ["C", "BET", "IRLEXL(-4)", "IRLY1"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1985Q1", "sample_end": "2016Q4",
     "lhs_transform": "IRJ=EXP(IRLJ1)*IRJ(-4)"},
    {"number": 240, "dependent": "IRLL1Z",
     "regressors": ["C", "BET", "IRLL1Z(-4)", "IRUR"],
     "instruments": ["C", "BET", "IRLL1Z(-4)", "IRUR"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1985Q1", "sample_end": "2016Q4",
     "lhs_transform": "IRL1=EXP(IRLL1Z)*IRPOP1"},
]
EQUATIONS_IR: list[MCEquation] = [_eq_from_spec("IR", s) for s in _IR_SPECS]

_PO_SPECS: list[dict] = [
    {"number": 241, "dependent": "POLIMZ",
     "regressors": ["C", "POLIMZ(-4)", "POLPYZZ", "POLCIGZ"],
     "instruments": ["C", "POLGZ", "POLEXZ(-4)", "POLPY(-4)", "PORB(-4)",
                     "POLIMZ(-4)", "POLPYZZ(-4)", "POLCIGZ(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "POIM=EXP(POLIMZ)*POPOP"},
    {"number": 242, "dependent": "POLCZ",
     "regressors": ["C", "POLCZ(-4)", "POLYZ", "PORB"],
     "instruments": ["C", "POLGZ", "POLEXZ(-4)", "POLPY(-4)", "POLYZ(-4)",
                     "PORB(-4)", "POLCZ(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "POC=EXP(POLCZ)*POPOP"},
    {"number": 243, "dependent": "POLI",
     "regressors": ["C", "POLI(-4)", "POLY", "PORB"],
     "instruments": ["C", "POLI(-4)", "POLY", "PORB"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "POI=EXP(POLI)"},
    {"number": 244, "dependent": "POLPY",
     "regressors": ["POLPY(-4)", "C", "BET", "POLPM", "POZZ(-4)"],
     "instruments": ["C", "POLGZ", "POLEXZ(-4)", "POLPY(-4)", "POLYZ(-4)",
                     "PORB(-4)", "BET", "POLPM(-4)", "POZZ(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "POPY=EXP(POLPY)"},
    {"number": 245, "dependent": "PORS",
     "regressors": ["C", "PORS(-4)", "POPCPY", "POZZ"],
     "instruments": ["C", "PORS(-4)", "POPCPY", "POZZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1984Q1", "sample_end": "1998Q4",
     "lhs_transform": "PORS=(1-EU1)*PORS+EU1*EURSA"},
    {"number": 246, "dependent": "PORBZ",
     "regressors": ["C", "PORBZZ", "PORSZ"],
     "instruments": ["C", "PORBZZ", "PORSZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1984Q1", "sample_end": "1998Q4",
     "lhs_transform": "PORB=(1-EU1)*(PORBZ+PORS(-4))+EU1*EURBA"},
    {"number": 247, "dependent": "POLH1",
     "regressors": ["C", "POLHA"],
     "instruments": ["C", "POLHA"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "1998Q4",
     "lhs_transform": "POH=(1-EU1)*(EXP(POLH1)*POH(-4))+EU1*1"},
]
EQUATIONS_PO: list[MCEquation] = [_eq_from_spec("PO", s) for s in _PO_SPECS]

_SP_SPECS: list[dict] = [
    {"number": 251, "dependent": "SPLIMZ",
     "regressors": ["C", "SPLIMZ(-4)", "SPLPYZZ", "SPLCIGZ"],
     "instruments": ["C", "SPLIMZ(-4)", "SPLPYZZ", "SPLCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "SPIM=EXP(SPLIMZ)*SPPOP"},
    {"number": 252, "dependent": "SPLCZ",
     "regressors": ["C", "SPLCZ(-4)", "SPLYZ"],
     "instruments": ["C", "SPLGZ", "SPLEXZ(-4)", "SPLPY(-4)", "SPLYZ(-4)",
                     "SPLCZ(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "SPC=EXP(SPLCZ)*SPPOP"},
    {"number": 253, "dependent": "SPLI",
     "regressors": ["C", "SPLI(-4)", "SPLY"],
     "instruments": ["C", "SPLI(-4)", "SPLY"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "SPI=EXP(SPLI)"},
    {"number": 254, "dependent": "SPLPY",
     "regressors": ["SPLPY(-4)", "C", "BET", "SPLPM(-4)", "SPZZ(-4)"],
     "instruments": ["SPLPY(-4)", "C", "BET", "SPLPM(-4)", "SPZZ(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2017Q4",
     "lhs_transform": "SPPY=EXP(SPLPY)"},
    {"number": 255, "dependent": "SPRS",
     "regressors": ["C", "SPRS(-4)", "USRSA", "SPPCPY"],
     "instruments": ["C", "SPRS(-4)", "USRSA", "SPPCPY"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1989Q1", "sample_end": "1998Q4",
     "lhs_transform": "SPRS=(1-EU1)*SPRS+EU1*EURSA"},
    {"number": 256, "dependent": "SPRBZ",
     "regressors": ["C", "SPRBZZ", "SPRSZ"],
     "instruments": ["C", "SPRBZZ", "SPRSZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1989Q1", "sample_end": "1998Q4",
     "lhs_transform": "SPRB=(1-EU1)*(SPRBZ+SPRS(-4))+EU1*EURBA"},
    {"number": 257, "dependent": "SPLH1",
     "regressors": ["C", "SPLHA"],
     "instruments": ["C", "SPLHA"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "1998Q4",
     "lhs_transform": "SPH=(1-EU1)*(EXP(SPLH1)*SPH(-4))+EU1*1"},
]
EQUATIONS_SP: list[MCEquation] = [_eq_from_spec("SP", s) for s in _SP_SPECS]

_NZ_SPECS: list[dict] = [
    {"number": 261, "dependent": "NZLIMZ",
     "regressors": ["C", "NZLIMZ(-4)", "NZLPYZZ", "NZLCIGZ"],
     "instruments": ["C", "NZLGZ", "NZLEXZ(-4)", "NZLPY(-4)", "NZLYZ(-4)",
                     "NZRB(-4)", "NZLIMZ(-4)", "NZLPYZZ(-4)", "NZLCIGZ(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1975Q1", "sample_end": "2016Q4",
     "lhs_transform": "NZIM=EXP(NZLIMZ)*NZPOP"},
    {"number": 262, "dependent": "NZLCZ",
     "regressors": ["C", "NZLCZ(-4)", "NZLYZ", "NZRS"],
     "instruments": ["C", "NZLGZ", "NZLEXZ(-4)", "NZLPY(-4)", "NZLYZ(-4)",
                     "NZRB(-4)", "NZLCZ(-4)", "NZRS(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1975Q1", "sample_end": "2016Q4",
     "lhs_transform": "NZC=EXP(NZLCZ)*NZPOP"},
    {"number": 263, "dependent": "NZLI",
     "regressors": ["C", "NZLI(-4)", "NZLY"],
     "instruments": ["C", "NZLGZ", "NZLEXZ(-4)", "NZLPY(-4)", "NZLYZ(-4)",
                     "NZRB(-4)", "NZLI(-4)", "NZLY(-4)", "NZRS(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1975Q1", "sample_end": "2016Q4",
     "lhs_transform": "NZI=EXP(NZLI)"},
    {"number": 264, "dependent": "NZLPY",
     "regressors": ["NZLPY(-4)", "C", "BET", "NZLPM", "NZZZ(-4)"],
     "instruments": ["C", "NZLGZ", "NZLEXZ(-4)", "NZLPY(-4)", "NZLYZ(-4)",
                     "NZRB(-4)", "BET", "NZLPM(-4)", "NZZZ(-4)", "NZRS(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1975Q1", "sample_end": "2016Q4",
     "lhs_transform": "NZPY=EXP(NZLPY)"},
    {"number": 265, "dependent": "NZRS",
     "regressors": ["C", "NZRS(-4)", "NZPCPY", "USRSA", "NZZZ"],
     "instruments": ["C", "NZRS(-4)", "NZPCPY", "USRSA", "NZZZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1975Q1", "sample_end": "2016Q4",
     "lhs_transform": "NZRS=(ABS(NZRS-.0)+NZRS-.0)/2.+.0"},
    {"number": 266, "dependent": "NZRBZ",
     "regressors": ["C", "NZRBZZ", "NZRSZ"],
     "instruments": ["C", "NZRBZZ", "NZRSZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1975Q1", "sample_end": "2017Q4",
     "lhs_transform": "NZRB=NZRBZ+NZRS(-4)"},
    {"number": 267, "dependent": "NZLE1Z",
     "regressors": ["C", "NZLRSZ"],
     "instruments": ["C", "NZLRSZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1974Q1", "sample_end": "2016Q4",
     "lhs_transform": "NZE=EXP(NZLE1Z+.050*NZLEA)*NZE(-4)"},
]
EQUATIONS_NZ: list[MCEquation] = [_eq_from_spec("NZ", s) for s in _NZ_SPECS]

# Countries with only 2–4 equations each.
_SA_SPECS: list[dict] = [
    {"number": 271, "dependent": "SALIMZ",
     "regressors": ["C", "SALIMZ(-4)", "SALCIGZ"],
     "instruments": ["C", "SALIMZ(-4)", "SALCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1982Q1", "sample_end": "2016Q4",
     "lhs_transform": "SAIM=EXP(SALIMZ)*SAPOP"},
    {"number": 272, "dependent": "SALCZ",
     "regressors": ["C", "SALCZ(-4)", "SALYZ"],
     "instruments": ["C", "SALCZ(-4)", "SALYZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1982Q1", "sample_end": "2016Q4",
     "lhs_transform": "SAC=EXP(SALCZ)*SAPOP"},
]
EQUATIONS_SA: list[MCEquation] = [_eq_from_spec("SA", s) for s in _SA_SPECS]

_CO_SPECS: list[dict] = [
    {"number": 291, "dependent": "COLIMZ",
     "regressors": ["C", "COLIMZ(-4)", "COLCIGZ"],
     "instruments": ["C", "COLIMZ(-4)", "COLCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "COIM=EXP(COLIMZ)*COPOP"},
    {"number": 292, "dependent": "COLCZ",
     "regressors": ["C", "COLCZ(-4)", "COLYZ"],
     "instruments": ["C", "COLCZ(-4)", "COLYZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "COC=EXP(COLCZ)*COPOP"},
]
EQUATIONS_CO: list[MCEquation] = [_eq_from_spec("CO", s) for s in _CO_SPECS]

_JO_SPECS: list[dict] = [
    {"number": 301, "dependent": "JOLIMZ",
     "regressors": ["C", "JOLIMZ(-4)", "JOLCIGZ"],
     "instruments": ["C", "JOLIMZ(-4)", "JOLCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "JOIM=EXP(JOLIMZ)*JOPOP"},
    {"number": 302, "dependent": "JOLCZ",
     "regressors": ["C", "JOLCZ(-4)", "JOLYZ"],
     "instruments": ["C", "JOLCZ(-4)", "JOLYZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "JOC=EXP(JOLCZ)*JOPOP"},
    {"number": 304, "dependent": "JOLPY",
     "regressors": ["JOLPY(-4)", "C", "BET", "JOLPM"],
     "instruments": ["JOLPY(-4)", "C", "BET", "JOLPM"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "JOPY=EXP(JOLPY)"},
]
EQUATIONS_JO: list[MCEquation] = [_eq_from_spec("JO", s) for s in _JO_SPECS]

_ID_SPECS: list[dict] = [
    {"number": 321, "dependent": "IDLIMZ",
     "regressors": ["C", "IDLIMZ(-4)", "IDLCIGZ"],
     "instruments": ["C", "IDLIMZ(-4)", "IDLCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1978Q1", "sample_end": "2016Q4",
     "lhs_transform": "IDIM=EXP(IDLIMZ)*IDPOP"},
    {"number": 322, "dependent": "IDLCZ",
     "regressors": ["C", "IDLCZ(-4)", "IDLYZ"],
     "instruments": ["C", "IDLCZ(-4)", "IDLYZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1978Q1", "sample_end": "2016Q4",
     "lhs_transform": "IDC=EXP(IDLCZ)*IDPOP"},
    {"number": 323, "dependent": "IDLI",
     "regressors": ["C", "IDLI(-4)", "IDLY"],
     "instruments": ["C", "IDLI(-4)", "IDLY"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1978Q1", "sample_end": "2016Q4",
     "lhs_transform": "IDI=EXP(IDLI)"},
]
EQUATIONS_ID: list[MCEquation] = [_eq_from_spec("ID", s) for s in _ID_SPECS]

_MA_SPECS: list[dict] = [
    {"number": 331, "dependent": "MALIMZ",
     "regressors": ["C", "MALIMZ(-4)", "MALCIGZ"],
     "instruments": ["C", "MALIMZ(-4)", "MALCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "MAIM=EXP(MALIMZ)*MAPOP"},
    {"number": 332, "dependent": "MALCZ",
     "regressors": ["C", "MALCZ(-4)", "MALYZ"],
     "instruments": ["C", "MALCZ(-4)", "MALYZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "MAC=EXP(MALCZ)*MAPOP"},
    {"number": 334, "dependent": "MALPY",
     "regressors": ["MALPY(-4)", "C", "BET", "MAZZ"],
     "instruments": ["MALPY(-4)", "C", "BET", "MAZZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "MAPY=EXP(MALPY)"},
]
EQUATIONS_MA: list[MCEquation] = [_eq_from_spec("MA", s) for s in _MA_SPECS]

_PA_SPECS: list[dict] = [
    {"number": 341, "dependent": "PALIMZ",
     "regressors": ["C", "PALIMZ(-4)", "PALCIGZ"],
     "instruments": ["C", "PALIMZ(-4)", "PALCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "PAIM=EXP(PALIMZ)*PAPOP"},
    {"number": 342, "dependent": "PALCZ",
     "regressors": ["C", "PALCZ(-4)", "PALYZ"],
     "instruments": ["C", "PALCZ(-4)", "PALYZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "PAC=EXP(PALCZ)*PAPOP"},
    {"number": 343, "dependent": "PALI",
     "regressors": ["C", "PALI(-4)", "PALY"],
     "instruments": ["C", "PALI(-4)", "PALY"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "PAI=EXP(PALI)"},
    {"number": 344, "dependent": "PALPY",
     "regressors": ["PALPY(-4)", "C", "BET", "PALPM", "PAZZ"],
     "instruments": ["PALPY(-4)", "C", "BET", "PALPM", "PAZZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "PAPY=EXP(PALPY)"},
]
EQUATIONS_PA: list[MCEquation] = [_eq_from_spec("PA", s) for s in _PA_SPECS]

_PH_SPECS: list[dict] = [
    {"number": 351, "dependent": "PHLIMZ",
     "regressors": ["C", "PHLIMZ(-4)", "PHLCIGZ"],
     "instruments": ["C", "PHLIMZ(-4)", "PHLCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1962Q1", "sample_end": "2016Q4",
     "lhs_transform": "PHIM=EXP(PHLIMZ)*PHPOP"},
    {"number": 352, "dependent": "PHLCZ",
     "regressors": ["C", "PHLCZ(-4)", "PHLYZ", "PHRS"],
     "instruments": ["C", "PHLCZ(-4)", "PHLYZ", "PHRS"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1979Q1", "sample_end": "2016Q4",
     "lhs_transform": "PHC=EXP(PHLCZ)*PHPOP"},
    {"number": 355, "dependent": "PHRS",
     "regressors": ["C", "PHRS(-4)", "PHPCPY", "USRSA"],
     "instruments": ["C", "PHRS(-4)", "PHPCPY", "USRSA"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1979Q1", "sample_end": "2016Q4",
     "lhs_transform": "PHRS=(ABS(PHRS-.0)+PHRS-.0)/2.+.0"},
    {"number": 357, "dependent": "PHLE1",
     "regressors": ["C", "PHLEA"],
     "instruments": ["C", "PHLEA"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "PHE=EXP(PHLE1)*PHE(-4)"},
]
EQUATIONS_PH: list[MCEquation] = [_eq_from_spec("PH", s) for s in _PH_SPECS]

_TH_SPECS: list[dict] = [
    {"number": 361, "dependent": "THLIMZ",
     "regressors": ["C", "THLIMZ(-4)", "THLCIGZ"],
     "instruments": ["C", "THLIMZ(-4)", "THLCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "THIM=EXP(THLIMZ)*THPOP"},
    {"number": 362, "dependent": "THLCZ",
     "regressors": ["C", "THLCZ(-4)", "THLYZ"],
     "instruments": ["C", "THLCZ(-4)", "THLYZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "THC=EXP(THLCZ)*THPOP"},
    {"number": 364, "dependent": "THLPY",
     "regressors": ["THLPY(-4)", "C", "BET", "THLPM", "THZZ(-4)"],
     "instruments": ["THLPY(-4)", "C", "BET", "THLPM", "THZZ(-4)"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "THPY=EXP(THLPY)"},
]
EQUATIONS_TH: list[MCEquation] = [_eq_from_spec("TH", s) for s in _TH_SPECS]

_CH_SPECS: list[dict] = [
    {"number": 371, "dependent": "CHLIMZ",
     "regressors": ["C", "CHLIMZ(-4)", "CHLCIGZ"],
     "instruments": ["C", "CHLIMZ(-4)", "CHLCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1982Q1", "sample_end": "2016Q4",
     "lhs_transform": "CHIM=EXP(CHLIMZ)*CHPOP"},
    {"number": 372, "dependent": "CHLCZ",
     "regressors": ["C", "CHLCZ(-4)", "CHLYZ"],
     "instruments": ["C", "CHLCZ(-4)", "CHLYZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1982Q1", "sample_end": "2016Q4",
     "lhs_transform": "CHC=EXP(CHLCZ)*CHPOP"},
    {"number": 373, "dependent": "CHLI",
     "regressors": ["C", "CHLI(-4)", "CHLY"],
     "instruments": ["C", "CHLI(-4)", "CHLY"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1982Q1", "sample_end": "2016Q4",
     "lhs_transform": "CHI=EXP(CHLI)"},
    {"number": 374, "dependent": "CHLPY",
     "regressors": ["CHLPY(-4)", "C", "BET", "CHZZ"],
     "instruments": ["CHLPY(-4)", "C", "BET", "CHZZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1982Q1", "sample_end": "2016Q4",
     "lhs_transform": "CHPY=EXP(CHLPY)"},
]
EQUATIONS_CH: list[MCEquation] = [_eq_from_spec("CH", s) for s in _CH_SPECS]

_AR_SPECS: list[dict] = [
    {"number": 381, "dependent": "ARLIMZ",
     "regressors": ["C", "ARLIMZ(-4)", "ARLCIGZ"],
     "instruments": ["C", "ARLIMZ(-4)", "ARLCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1992Q1", "sample_end": "2016Q4",
     "lhs_transform": "ARIM=EXP(ARLIMZ)*ARPOP"},
    {"number": 382, "dependent": "ARLCZ",
     "regressors": ["C", "ARLCZ(-4)", "ARLYZ"],
     "instruments": ["C", "ARLCZ(-4)", "ARLYZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1992Q1", "sample_end": "2016Q4",
     "lhs_transform": "ARC=EXP(ARLCZ)*ARPOP"},
]
EQUATIONS_AR: list[MCEquation] = [_eq_from_spec("AR", s) for s in _AR_SPECS]

_BR_SPECS: list[dict] = [
    {"number": 391, "dependent": "BRLIMZ",
     "regressors": ["C", "BRLIMZ(-4)", "BRLCIGZ"],
     "instruments": ["C", "BRLIMZ(-4)", "BRLCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1997Q1", "sample_end": "2016Q4",
     "lhs_transform": "BRIM=EXP(BRLIMZ)*BRPOP"},
    {"number": 392, "dependent": "BRLCZ",
     "regressors": ["C", "BRLCZ(-4)", "BRLYZ"],
     "instruments": ["C", "BRLCZ(-4)", "BRLYZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1997Q1", "sample_end": "2016Q4",
     "lhs_transform": "BRC=EXP(BRLCZ)*BRPOP"},
]
EQUATIONS_BR: list[MCEquation] = [_eq_from_spec("BR", s) for s in _BR_SPECS]

_CE_SPECS: list[dict] = [
    {"number": 401, "dependent": "CELIMZ",
     "regressors": ["C", "CELIMZ(-4)", "CELCIGZ"],
     "instruments": ["C", "CELIMZ(-4)", "CELCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1983Q1", "sample_end": "2016Q4",
     "lhs_transform": "CEIM=EXP(CELIMZ)*CEPOP"},
    {"number": 402, "dependent": "CELCZ",
     "regressors": ["C", "CELCZ(-4)", "CELYZ"],
     "instruments": ["C", "CELCZ(-4)", "CELYZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1983Q1", "sample_end": "2016Q4",
     "lhs_transform": "CEC=EXP(CELCZ)*CEPOP"},
]
EQUATIONS_CE: list[MCEquation] = [_eq_from_spec("CE", s) for s in _CE_SPECS]

_ME_SPECS: list[dict] = [
    {"number": 411, "dependent": "MELIMZ",
     "regressors": ["C", "MELIMZ(-4)", "MELPYZZ", "MELCIGZ"],
     "instruments": ["C", "MELIMZ(-4)", "MELPYZZ", "MELCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "MEIM=EXP(MELIMZ)*MEPOP"},
    {"number": 412, "dependent": "MELCZ",
     "regressors": ["C", "MELCZ(-4)", "MELYZ"],
     "instruments": ["C", "MELCZ(-4)", "MELYZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1972Q1", "sample_end": "2016Q4",
     "lhs_transform": "MEC=EXP(MELCZ)*MEPOP"},
]
EQUATIONS_ME: list[MCEquation] = [_eq_from_spec("ME", s) for s in _ME_SPECS]

_PE_SPECS: list[dict] = [
    {"number": 421, "dependent": "PELIMZ",
     "regressors": ["C", "PELIMZ(-4)", "PELCIGZ"],
     "instruments": ["C", "PELIMZ(-4)", "PELCIGZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1981Q1", "sample_end": "2016Q4",
     "lhs_transform": "PEIM=EXP(PELIMZ)*PEPOP"},
    {"number": 422, "dependent": "PELCZ",
     "regressors": ["C", "PELCZ(-4)", "PELYZ"],
     "instruments": ["C", "PELCZ(-4)", "PELYZ"],
     "has_ar1": False, "has_ar2": False,
     "sample_start": "1981Q1", "sample_end": "2016Q4",
     "lhs_transform": "PEC=EXP(PELCZ)*PEPOP"},
]
EQUATIONS_PE: list[MCEquation] = [_eq_from_spec("PE", s) for s in _PE_SPECS]


# NLEQ equations: Fair's xxLPXA export-price family (nonlinear in params).
# ``NLEQ LPXA = α(LPXB − βLPXB(-4) − γLPXB(-8)) + βLPXA(-4) + γLPXA(-8)``.
# Sample is annual 1972Q1–2016Q4 for every country; ``nlols_lpxa`` handles
# estimation via Levenberg-Marquardt on the dropped-null annual series.
_NLEQ_SPECS: list[tuple[str, int]] = [
    ("BE", 188), ("DE", 198), ("SW", 218), ("IR", 238), ("SP", 258),
    ("NZ", 268), ("ID", 328), ("PA", 348), ("TH", 368), ("ME", 418),
]
_NLEQ_EQUATIONS: dict[str, MCEquation] = {}
for _country, _num in _NLEQ_SPECS:
    _dep = f"{_country}LPXA"
    _rhs = f"{_country}LPXB"
    _NLEQ_EQUATIONS[_country] = MCEquation(
        country=_country, number=_num, dependent=_dep,
        regressors=(_rhs,), instruments=(_rhs,),
        has_ar1=False, has_ar2=False, is_nleq=True,
        sample_start="1972Q1", sample_end="2016Q4",
        lhs_transform=f"{_country}PX=EXP({_dep})*({_country}PW$*({_country}E/{_country}E10))",
        notes="Fair NLOLS (DFP); we use Levenberg-Marquardt. Non-convex — "
              "some countries land at a lower-SSE local minimum than Fair's DFP.",
    )


# US MC-only additions — EQ 31 extends the standalone US block with an
# export-price AR(2) equation used only in the multi-country context.
EQUATIONS_US_MC: list[MCEquation] = [
    MCEquation(
        country="US", number=31, dependent="LPXA",
        regressors=("LPXB",), instruments=("LPXB",),
        has_ar1=False, has_ar2=True, use_bounded_search=True,
        sample_start="1961Q1", sample_end="2016Q4",
        lhs_transform="USPX$=EXP(LPXA)*USPW$",
        notes="MC.INP line 610 — US export-price AR(2). Not part of the "
              "standalone US pyfair model (us_model.EQUATIONS).",
    ),
]


EQUATIONS_BY_COUNTRY: dict[str, list[MCEquation]] = {
    # US MC-only additions (EQ 31). Standalone US EQs 1–30 live in us_model.
    "US": EQUATIONS_US_MC,
    # Quarterly countries.
    "CA": EQUATIONS_CA, "JA": EQUATIONS_JA, "AU": EQUATIONS_AU,
    "FR": EQUATIONS_FR, "GE": EQUATIONS_GE, "IT": EQUATIONS_IT,
    "NE": EQUATIONS_NE, "ST": EQUATIONS_ST, "UK": EQUATIONS_UK,
    "FI": EQUATIONS_FI, "AS": EQUATIONS_AS, "SO": EQUATIONS_SO,
    "KO": EQUATIONS_KO,
    # Annual-lag countries (shift(4) GENRs, use *A averages).
    "BE": EQUATIONS_BE + [_NLEQ_EQUATIONS["BE"]],
    "DE": EQUATIONS_DE + [_NLEQ_EQUATIONS["DE"]],
    "NO": EQUATIONS_NO,
    "SW": EQUATIONS_SW + [_NLEQ_EQUATIONS["SW"]],
    "GR": EQUATIONS_GR,
    "IR": EQUATIONS_IR + [_NLEQ_EQUATIONS["IR"]],
    "PO": EQUATIONS_PO,
    "SP": EQUATIONS_SP + [_NLEQ_EQUATIONS["SP"]],
    "NZ": EQUATIONS_NZ + [_NLEQ_EQUATIONS["NZ"]],
    "SA": EQUATIONS_SA, "CO": EQUATIONS_CO, "JO": EQUATIONS_JO,
    "ID": EQUATIONS_ID + [_NLEQ_EQUATIONS["ID"]],
    "MA": EQUATIONS_MA,
    "PA": EQUATIONS_PA + [_NLEQ_EQUATIONS["PA"]],
    "PH": EQUATIONS_PH,
    "TH": EQUATIONS_TH + [_NLEQ_EQUATIONS["TH"]],
    "CH": EQUATIONS_CH,
    "AR": EQUATIONS_AR, "BR": EQUATIONS_BR, "CE": EQUATIONS_CE,
    "ME": EQUATIONS_ME + [_NLEQ_EQUATIONS["ME"]],
    "PE": EQUATIONS_PE,
}


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _load_data_files(paths: tuple[Path, ...]) -> pl.DataFrame:
    """Parse and merge Fair MC data files into a single wide frame.

    ``YAW.DAT`` uses the classic Fair format (one variable per ``LOAD``,
    values on following lines). ``YDATA.DAT`` and ``QUAR.DAT`` use the
    ``LOAD XID`` format (many variables per block, period-prefixed rows).
    We detect which by the presence of ``LOAD XID`` and dispatch.

    Merging is by ``period`` with outer join, preserving observations
    present in any file.
    """
    frames = []
    for p in paths:
        text = p.read_text()
        if "LOAD XID" in text:
            long = readers.parse_fair_xid_data(p)
        else:
            long = readers.parse_fair_data(p)
        wide = readers.pivot_to_wide(long)
        frames.append(wide)
    merged = frames[0]
    for other in frames[1:]:
        merged = merged.join(other, on="period", how="full", coalesce=True)
    return merged.sort("period")


def _all_required_lags(equations: list[MCEquation]) -> set[tuple[str, int]]:
    """Gather ``(base, lag)`` pairs referenced across the equation list."""
    lags: set[tuple[str, int]] = set()
    for eq in equations:
        for tok in (*eq.regressors, *eq.instruments, eq.dependent):
            _col, base, lag = _parse_token(tok)
            if lag > 0:
                lags.add((base, lag))
    return lags


def build_frame_mc(
    countries: tuple[str, ...] = ("CA",),
    data_paths: tuple[Path, ...] | None = None,
    include_pmm: bool = False,
) -> pl.DataFrame:
    """Load MC data and apply GENRs for the requested country prefixes.

    Args:
      countries: Tuple of 2-letter prefixes to materialize GENRs for.
        Defaults to just ``("CA",)``; pass more as needed.
      data_paths: Override for the (YAW.DAT, YDATA.DAT, QUAR.DAT)
        default from ``config``. Useful in tests.
      include_pmm: If True, also load the per-country import-price
        aggregates (``xxPMM`` series) from SHRDDD.DAT. Required for
        the MC solve driver; skip for estimation-only work.

    Returns:
      A wide Polars frame with ``period`` plus all raw columns from the
      data files and all GENR-derived columns for the requested countries.
    """
    paths = data_paths or (config.MC_YAW, config.MC_YDATA, config.MC_QUAR)
    frame = _load_data_files(paths)
    if include_pmm:
        from .solve import load_pmm_series
        pmm_frame = load_pmm_series()
        frame = frame.join(pmm_frame, on="period", how="left")
    frame = add_time_trend_and_constant(frame)
    frame = _apply_global_genrs(frame)
    for prefix in countries:
        frame = _apply_country_block(frame, prefix)
    equations = []
    for prefix in countries:
        equations.extend(EQUATIONS_BY_COUNTRY.get(prefix, []))
    frame = add_lags(frame, _all_required_lags(equations))
    return frame


# ---------------------------------------------------------------------------
# Estimation dispatch — mirrors us_model.estimate()
# ---------------------------------------------------------------------------

@dataclass
class EstimationResultMC:
    """Per-equation estimation output and comparison vs Fair's OUT."""
    equation: MCEquation
    coefficients: dict[str, float]
    reference: dict[str, float]
    n_obs: int
    rho_iterations: int | None


def _period_before(period: str) -> str:
    """Return the quarter one step before ``period``."""
    year, q = period.split("Q")
    year, q = int(year), int(q)
    if q == 1:
        return f"{year - 1}Q4"
    return f"{year}Q{q - 1}"


def _stack(df: pl.DataFrame, tokens: tuple[str, ...]) -> jnp.ndarray:
    cols = [_parse_token(t)[0] for t in tokens]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}")
    return jnp.column_stack(
        [jnp.asarray(df[c].to_numpy(), dtype=jnp.float64) for c in cols]
    )


def _first_row(df: pl.DataFrame, tokens: tuple[str, ...]) -> jnp.ndarray:
    cols = [_parse_token(t)[0] for t in tokens]
    return jnp.array([float(df[c].item()) for c in cols])


def estimate(equation: MCEquation, frame: pl.DataFrame) -> EstimationResultMC:
    """Estimate one MC equation end-to-end via iterated 2SLS-with-AR(1).

    For AR(1) equations we include one pre-sample quarter (the estimator
    needs a lag for the first ρ-difference). Non-AR equations skip the
    pre-sample row so ``n_obs`` matches Fair's sample-size report verbatim.
    """
    start = equation.sample_start
    end = equation.sample_end
    required_cols = {_parse_token(t)[0] for t in
                     (*equation.regressors, *equation.instruments,
                      equation.dependent)}

    if equation.is_nleq:
        # Fair's ``xxLPXA`` family. LM on 3-coef nonlinear model using the
        # LPXA/LPXB series over the estimation window (drop_nulls picks
        # the annual observations Fair actually uses).
        dep = equation.dependent
        rhs = equation.regressors[0] if equation.regressors else None
        if rhs is None:
            raise ValueError(f"NLEQ EQ {equation.number}: no regressor LPXB")
        window = frame.filter(
            (pl.col("period") >= pl.lit(start))
            & (pl.col("period") <= pl.lit(end))
        ).drop_nulls(subset=[dep, rhs]).sort("period")
        lpxa_arr = window[dep].to_numpy()
        lpxb_arr = window[rhs].to_numpy()
        coefs_vec, _ = nlols_lpxa(lpxa_arr, lpxb_arr, lag_step=1)
        coefs = {
            f"COEF(1,{equation.number})": float(coefs_vec[0]),
            f"COEF(2,{equation.number})": float(coefs_vec[1]),
            f"COEF(3,{equation.number})": float(coefs_vec[2]),
        }
        ref_raw = REFERENCE_PARAMS_MC.get(equation.number, {})
        reference = {
            f"COEF(1,{equation.number})": ref_raw.get("COEF(+1)", float("nan")),
            f"COEF(2,{equation.number})": ref_raw.get("COEF(+2)", float("nan")),
            f"COEF(3,{equation.number})": ref_raw.get("COEF(+3)", float("nan")),
        }
        return EstimationResultMC(
            equation=equation, coefficients=coefs, reference=reference,
            n_obs=lpxa_arr.shape[0], rho_iterations=None,
        )

    n_presample = 2 if equation.has_ar2 else (1 if equation.has_ar1 else 0)
    if n_presample > 0:
        presample_start = start
        for _ in range(n_presample):
            presample_start = _period_before(presample_start)
        window = frame.filter(
            (pl.col("period") >= pl.lit(presample_start))
            & (pl.col("period") <= pl.lit(end))
        ).drop_nulls(subset=list(required_cols & set(frame.columns)))
        presample = window.head(n_presample)
        est = window.tail(window.height - n_presample)
    else:
        est = frame.filter(
            (pl.col("period") >= pl.lit(start))
            & (pl.col("period") <= pl.lit(end))
        ).drop_nulls(subset=list(required_cols & set(frame.columns)))
        presample = None

    y = jnp.asarray(est[equation.dependent].to_numpy(), dtype=jnp.float64)
    X = _stack(est, equation.regressors)
    Z = _stack(est, equation.instruments)

    if equation.has_ar2:
        y_ps = jnp.asarray(presample[equation.dependent].to_numpy(),
                           dtype=jnp.float64)
        X_ps = _stack(presample, equation.regressors)
        if equation.use_bounded_search:
            beta, rho, iters = two_sls_ar2_bounded(y, X, Z, y_ps, X_ps)
        else:
            beta, rho, _se, iters = two_sls_ar2(
                y, X, Z, y_ps, X_ps, max_iter=300, damping=equation.damping,
            )
        coefs = {tok: float(beta[i]) for i, tok in enumerate(equation.regressors)}
        coefs["RHO(-1)"] = float(rho[0])
        coefs["RHO(-2)"] = float(rho[1])
        iter_count = int(iters)
    elif equation.has_ar1:
        y_ps = jnp.asarray(float(presample[equation.dependent].item()))
        X_ps = _first_row(presample, equation.regressors)
        if equation.use_bounded_search:
            beta, rho, iters = two_sls_ar1_bounded(y, X, Z, y_ps, X_ps)
        else:
            beta, rho, _se, iters = two_sls_ar1(
                y, X, Z, y_ps, X_ps, max_iter=300, damping=equation.damping,
            )
        coefs = {tok: float(beta[i]) for i, tok in enumerate(equation.regressors)}
        coefs["RHO(-1)"] = float(rho)
        iter_count = int(iters)
    else:
        beta, _se = two_sls_with_se(y, X, Z)
        coefs = {tok: float(beta[i]) for i, tok in enumerate(equation.regressors)}
        iter_count = None

    ref_raw = REFERENCE_PARAMS_MC.get(equation.number, {})
    reference = {}
    for tok in equation.regressors:
        _col, base, lag = _parse_token(tok)
        key = f"{base}(0)" if lag == 0 else f"{base}({-lag:+d})"
        reference[tok] = ref_raw.get(key, float("nan"))
    if equation.has_ar1 or equation.has_ar2:
        reference["RHO(-1)"] = ref_raw.get("RHO(-1)", float("nan"))
    if equation.has_ar2:
        reference["RHO(-2)"] = ref_raw.get("RHO(-2)", float("nan"))

    return EstimationResultMC(
        equation=equation,
        coefficients=coefs,
        reference=reference,
        n_obs=est.height,
        rho_iterations=iter_count,
    )


def estimate_country(prefix: str) -> list[EstimationResultMC]:
    """Estimate every equation for one country."""
    frame = build_frame_mc(countries=(prefix,))
    equations = EQUATIONS_BY_COUNTRY.get(prefix, [])
    results = []
    for eq in equations:
        try:
            results.append(estimate(eq, frame))
        except Exception:
            _LOG.exception(
                "%s EQ %d (%s) estimation failed",
                prefix, eq.number, eq.dependent,
            )
    return results


if __name__ == "__main__":
    results = estimate_country("CA")
    print(f"\n=== Estimated {len(results)} / {len(EQUATIONS_CA)} CA equations ===\n")
    for r in results:
        eq = r.equation
        deltas = [abs(our - r.reference.get(tok, float("nan")))
                  for tok, our in r.coefficients.items()
                  if r.reference.get(tok, float("nan")) == r.reference.get(tok, float("nan"))]
        max_err = max(deltas) if deltas else float("nan")
        iters = f" iters={r.rho_iterations}" if r.rho_iterations is not None else ""
        print(f"  EQ {eq.number:3d} {eq.dependent:12s} n_obs={r.n_obs:3d}"
              f"  max_abs_err={max_err:.2e}{iters}")
