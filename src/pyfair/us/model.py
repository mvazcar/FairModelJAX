"""US Model — batch port of Fair's equations from ``fminput.txt``.

Parses Fair's DSL spec (``EQ`` lines, ``FSR`` lines, ``MODEQ`` COVID additions,
regime-dummy construction) at import time, computes every GENR-derived
variable, and exposes a generic ``estimate(equation, frame)`` that runs
``two_sls_ar1`` or ``two_sls_with_se`` depending on whether the equation
has an AR(1) term.

Skipped equations:
  * EQ 9, 19, 20, 21, 22, 25  — NONE placeholders (dropped by Fair, no estimation)
  * EQ 11                     — AR(3). Our estimator supports AR(1) only; v0.4.
  * EQ 16                     — Uses a "create BETA1, BETA2, DELTA1" preprocessing
                                step that requires EQ 10's coefficients first; v0.4.
  * EQ 30                     — Uses a piecewise LHS (ABS function for ZLB);
                                treated as linear here with caveat.

All remaining ~18 equations are handled by the generic path.
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
from ..core.estimate import two_sls_ar1, two_sls_ar1_bounded, two_sls_with_se

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Equation specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UsEquation:
    """Static spec for one Fair US-model stochastic equation.

    Attributes:
      number: Fair's equation number (1..30).
      dependent: Dependent variable name (e.g. ``"LCSZ"``).
      regressors: RHS variables in order. ``"C"`` denotes the plain constant;
        entries like ``"LCSZ_lag1"`` denote lag-1 of LCSZ.
      instruments: First-stage regressors (instruments) for 2SLS. Empty
        tuple means OLS (used for pure-exogenous regressor lists like EQ 15).
      has_ar1: True if Fair's spec declares ``RHO=1``.
      sample_start: First period of the estimation window (e.g. ``"1954Q1"``).
      sample_end: Last period (e.g. ``"2025Q4"``).
      damping: ρ-update multiplier for the iterated AR(1) estimator. Default
        1.0 (full Fair step); set lower (0.3-0.5) for near-unit-root equations
        where the full step overshoots.
      notes: Optional comment for known issues.
    """
    number: int
    dependent: str
    regressors: tuple[str, ...]
    instruments: tuple[str, ...]
    has_ar1: bool
    sample_start: str
    sample_end: str
    damping: float = 1.0
    use_bounded_search: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# GENR library — derived variables computed from raw fmdata columns
# ---------------------------------------------------------------------------

def _c(name: str) -> pl.Expr:
    """Shorthand for pl.col."""
    return pl.col(name)


# Each entry is (output_name, formula(df) -> pl.Expr). Ordering matters only
# when one derived variable depends on another — we list dependencies first.
# Lines 35–151 of fminput.txt, in equivalent Python form.
_GENR_SPECS: list[tuple[str, Callable[[pl.DataFrame], pl.Expr]]] = [
    # Government-spending normalizer (EQ 12 regressor).
    ("CGZ3",   lambda df: (_c("CG") + _c("CG").shift(1) + _c("CG").shift(2))
                          / (_c("PX") * _c("YS")
                             + _c("PX").shift(1) * _c("YS").shift(1)
                             + _c("PX").shift(2) * _c("YS").shift(2))),
    # Per-capita ratios (real quantities divided by POP).
    ("CSZ",    lambda df: _c("CS") / _c("POP")),
    ("CNZ",    lambda df: _c("CN") / _c("POP")),
    ("CDZ",    lambda df: _c("CD") / _c("POP")),
    ("IHHZ",   lambda df: _c("IHH") / _c("POP")),
    ("L1Z",    lambda df: _c("L1") / _c("POP1")),
    ("L2Z",    lambda df: _c("L2") / _c("POP2")),
    ("L3Z",    lambda df: _c("L3") / _c("POP3")),
    ("LMZ",    lambda df: _c("LM") / _c("POP")),
    ("IMZ",    lambda df: _c("IM") / _c("POP")),
    ("KHZ",    lambda df: _c("KH") / _c("POP")),
    ("KDZ",    lambda df: _c("KD") / _c("POP")),
    ("AA1Z",   lambda df: _c("AA1") / _c("POP")),
    ("AA2Z",   lambda df: _c("AA2") / _c("POP")),
    ("AAZ",    lambda df: _c("AA") / _c("POP")),
    ("YDZ",    lambda df: _c("YD") / (_c("POP") * _c("PH"))),
    ("XMFAZ",  lambda df: (_c("X") - _c("FA")) / _c("POP")),
    # Sums
    ("COGS",   lambda df: _c("COG") + _c("COS")),
    ("TRGSZ",  lambda df: (_c("TRGH") + _c("TRSH")) / (_c("POP") * _c("PH"))),
    ("TFGS",   lambda df: _c("TFG") + _c("TFS")),
    # Logs of raw variables.
    ("LPF",    lambda df: _c("PF").log()),
    ("LHO",    lambda df: _c("HO").log()),
    ("LWF",    lambda df: _c("WF").log()),
    ("LHF",    lambda df: _c("HF").log()),
    ("LPIM",   lambda df: _c("PIM").log()),
    ("LEX",    lambda df: _c("EX").log()),
    ("LY",     lambda df: _c("Y").log()),
    ("LX",     lambda df: _c("X").log()),
    ("LV",     lambda df: _c("V").log()),
    ("LPH",    lambda df: _c("PH").log()),
    ("LDF",    lambda df: _c("DF").log()),
    ("LKK",    lambda df: _c("KK").log()),
    ("LUB",    lambda df: _c("UB").log()),
    ("LXMFA",  lambda df: (_c("X") - _c("FA")).log()),
    # Log of ratios.
    ("LL1Z",   lambda df: _c("L1Z").log()),
    ("LL2Z",   lambda df: _c("L2Z").log()),
    ("LL3Z",   lambda df: _c("L3Z").log()),
    ("LLMZ",   lambda df: _c("LMZ").log()),
    ("LCSZ",   lambda df: _c("CSZ").log()),
    ("LCNZ",   lambda df: _c("CNZ").log()),
    ("LCDZ",   lambda df: _c("CDZ").log()),
    ("LIHHZ",  lambda df: _c("IHHZ").log()),
    ("LIMZ",   lambda df: _c("IMZ").log()),
    ("LKHZ",   lambda df: _c("KHZ").log()),
    ("LKDZ",   lambda df: _c("KDZ").log()),
    ("LAAZ",   lambda df: _c("AAZ").log()),
    ("LYZ",    lambda df: (_c("Y") / _c("POP")).log()),
    ("LYDZ",   lambda df: (_c("YD") / (_c("POP") * _c("PH"))).log()),
    ("LXMFAZ", lambda df: ((_c("X") - _c("FA")) / _c("POP")).log()),
    ("LCOGSZ", lambda df: (_c("COGS") / _c("POP")).log()),
    ("LTRGSZ", lambda df: _c("TRGSZ").log()),
    ("LEXZ",   lambda df: (_c("EX") / _c("POP")).log()),
    ("LPOP",   lambda df: _c("POP").log()),
    ("LMFZ",   lambda df: (_c("MF") / _c("PF")).log()),
    ("LCURZ",  lambda df: (_c("CUR") / (_c("POP") * _c("PF"))).log()),
    ("LCURL1Q", lambda df: (_c("CUR").shift(1) / (_c("POP").shift(1) * _c("PF"))).log()),
    ("LMFL1Q", lambda df: (_c("MF").shift(1) / _c("PF")).log()),
    # Log-growth-rate style variables used as dependent vars for some equations.
    # For these we rely on fmdata.txt containing lag columns we shift to build.
    # Filled in below after primary GENRs.
    # Price/wage transforms.
    ("LWFQ",   lambda df: _c("WF").log() - _c("LAM").log()),
    ("LPIMZ",  lambda df: (_c("PIM") / _c("PF")).log()),
    ("LPFZPIM", lambda df: (_c("PF") / _c("PIM")).log()),
    ("LWFZPF", lambda df: _c("WF").log() - _c("PF").log()),
    ("LWFQZPF", lambda df: (_c("WF").log() - _c("LAM").log()) - _c("PF").log()),
    ("LWFD5",  lambda df: (_c("WF") * (1 + _c("D5G"))).log() - _c("LAM").log()),
    ("WAZPH",  lambda df: _c("WA") / _c("PH")),
    ("LWAZPH", lambda df: (_c("WA") / _c("PH")).log()),
    # Interest-rate transforms.
    ("RSB",    lambda df: _c("RS") * (1 - _c("D2G") - _c("D2S"))),
    ("RBA",    lambda df: _c("RB") * (1 - _c("D2G") - _c("D2S"))),
    # Slack / gap / unemployment / time trend helpers.
    ("GAP",    lambda df: (_c("YS") - _c("Y")) / _c("YS")),
    ("UMM",    lambda df: ((_c("U") - 1.0).abs() + (_c("U") - 1.0)) / 2 + 1.0),
    ("LU",     lambda df: (((_c("U") - 1.0).abs() + (_c("U") - 1.0)) / 2 + 1.0).log()),
    ("UR1",    lambda df: _c("UR") - _c("UR").shift(1)),
    ("ONEZUR", lambda df: 1.0 / _c("UR")),
    ("ONEZGAP", lambda df: 1.0 / (((_c("YS") - _c("Y")) / _c("YS")) + 0.07)),
    # Inflation measures — quarterly annualized and 4-quarter.
    ("PCPD",   lambda df: 100 * ((_c("PD") / _c("PD").shift(1))**4 - 1)),
    ("PCPD4",  lambda df: 100 * (_c("PD") / _c("PD").shift(4) - 1)),
    # Capital-excess.
    ("EXKK",   lambda df: _c("KK") - _c("KKMIN")),
    ("LEXKK",  lambda df: (_c("KK") / _c("KKMIN")).log()),
    # Excess labor.
    ("LEXL",   lambda df: (_c("JF") / (_c("JHMIN") / _c("HFS"))).log()),
    # Assorted derived variables used across equations.
    ("Y1",     lambda df: _c("Y") - _c("Y").shift(1)),
    ("LY1",    lambda df: (_c("Y") / _c("Y").shift(1)).log()),
    ("LJF1",   lambda df: (_c("JF") / _c("JF").shift(1)).log()),
    ("LHF1",   lambda df: (_c("HF") / _c("HF").shift(1)).log()),
    ("LHFL1A", lambda df: (_c("HF").shift(1) / _c("HFS").shift(1)).log()),
    ("LDF1",   lambda df: (_c("DF") / _c("DF").shift(1)).log()),
    ("LKK1",   lambda df: _c("KK").log() - _c("KK").shift(1).log()),
    ("RB1",    lambda df: _c("RB") - _c("RB").shift(1)),
    ("RS1",    lambda df: _c("RS") - _c("RS").shift(1)),
    # Rate-gap transforms (EQs 23, 24).
    ("RBMRSL2", lambda df: _c("RB") - _c("RS").shift(2)),
    ("RSMRSL2", lambda df: _c("RS") - _c("RS").shift(2)),
    ("RSLMRSL2", lambda df: _c("RS").shift(1) - _c("RS").shift(2)),
    ("RMMRSL2", lambda df: _c("RM") - _c("RS").shift(2)),
    ("RBLMRSL2", lambda df: _c("RB").shift(1) - _c("RS").shift(2)),
    ("RMLMRSL2", lambda df: _c("RM").shift(1) - _c("RS").shift(2)),
    # Dividend/profit path (EQ 18).
    ("PIEFA",  lambda df: _c("PIEF") - _c("TFG") - _c("TFS") - _c("TFR")),
    ("PIEFAMM", lambda df: (((_c("PIEF") - _c("TFG") - _c("TFS") - _c("TFR")) - 4.0).abs()
                            + ((_c("PIEF") - _c("TFG") - _c("TFS") - _c("TFR")) - 4.0)) / 2 + 4.0),
    # Fed rule (EQ 30) / D794823, D20083 dummies used in PCM1L1A / PCM1L1B —
    # these are raw dummies from fmdata.txt.
    ("PCM1L1A", lambda df: _c("D794823") * _c("PCM1").shift(1)),
    ("PCM1L1B", lambda df: _c("D20083") * _c("PCM1").shift(1)),
    # Government interest flow (EQ 29).
    ("AAG",    lambda df: -_c("AG")),
    ("INTGZ",  lambda df: _c("INTG") / (-_c("AG"))),
    ("RQG",    lambda df: (0.4 * (_c("RS") / 400)
                           + 0.75 * 0.6 * (1/8) * (1/400) * (_c("RB")
                           + _c("RB").shift(1) + _c("RB").shift(2) + _c("RB").shift(3)
                           + _c("RB").shift(4) + _c("RB").shift(5) + _c("RB").shift(6)
                           + _c("RB").shift(7)))),
    # Custom-duties / misc.
    ("LCUSTZ", lambda df: (_c("CUST") / (_c("PIM") * _c("IM"))).log()),
]


# LPIEFAZ = log(PIEFAMM / DF(-1)) where PIEFAMM = max(PIEFA - 4, 0) + 4.
# The max() is written as (|x| + x)/2 which equals x when x>=0 and 0 when x<0.
def _lpiefaz(df: pl.DataFrame) -> pl.Expr:
    piefa_minus_4 = _c("PIEF") - _c("TFG") - _c("TFS") - _c("TFR") - 4.0
    piefamm = (piefa_minus_4.abs() + piefa_minus_4) / 2 + 4.0
    return (piefamm / _c("DF").shift(1)).log()


_GENR_SPECS.append(("LPIEFAZ", _lpiefaz))


def apply_genr(df: pl.DataFrame) -> pl.DataFrame:
    """Run every GENR spec in order, skipping specs whose inputs are missing."""
    for name, formula in _GENR_SPECS:
        try:
            df = df.with_columns(formula(df).alias(name))
        except pl.exceptions.ColumnNotFoundError:
            # Input series not in fmdata — skip silently, the equation that
            # depends on this derived variable will also be skipped.
            pass
    return df


# ---------------------------------------------------------------------------
# Regime dummies — CNST2CS, CNST2L2, CNST2KK
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _RegimeDummy:
    name: str
    d2_start: str
    d2_end: str
    d3_start: str
    t1: int
    t2: int


_REGIME_DUMMIES = [
    # CNST2CS: D2 = 1974Q1–1994Q4, D3 from 1995Q1; T1=88, T2=172.
    _RegimeDummy("CNST2CS", "1974Q1", "1994Q4", "1995Q1", 88, 172),
    # CNST2L2: D2 = 1972Q1–1989Q4, D3 from 1990Q1; T1=80, T2=152.
    _RegimeDummy("CNST2L2", "1972Q1", "1989Q4", "1990Q1", 80, 152),
    # CNST2KK: D2 = 1979Q1–1987Q4, D3 from 1988Q1; T1=108, T2=144.
    _RegimeDummy("CNST2KK", "1979Q1", "1987Q4", "1988Q1", 108, 144),
]


def apply_regime_dummies(df: pl.DataFrame) -> pl.DataFrame:
    """Attach all three Fair regime-ramp dummies.

    Each evaluates to 0 before the D2 window, a linear ramp from ~0 to 1
    across D2, and 1 from the D3 start onwards. See fminput.txt §480–534.
    """
    period = pl.col("period")
    for spec in _REGIME_DUMMIES:
        d2 = pl.when((period >= pl.lit(spec.d2_start))
                     & (period <= pl.lit(spec.d2_end))).then(1.0).otherwise(0.0)
        d3 = pl.when(period >= pl.lit(spec.d3_start)).then(1.0).otherwise(0.0)
        ramp = d3 + spec.t1 * d2 / (spec.t1 - spec.t2) - _c("T") * d2 / (spec.t1 - spec.t2)
        df = df.with_columns(ramp.alias(spec.name))
    # TBL2 = T * CNST2L2 (fminput.txt line 516).
    df = df.with_columns((_c("T") * _c("CNST2L2")).alias("TBL2"))
    return df


def add_time_trend_and_constant(df: pl.DataFrame) -> pl.DataFrame:
    """Attach ``T`` (1-indexed quarter counter) and ``C`` (constant = 1)."""
    return df.with_columns(
        pl.int_range(1, df.height + 1).cast(pl.Float64).alias("T"),
        pl.lit(1.0).alias("C"),
    )


def add_lags(df: pl.DataFrame, lag_specs: set[tuple[str, int]]) -> pl.DataFrame:
    """For each ``(variable, k)`` in ``lag_specs`` add column ``{var}_lag{k}``."""
    new_cols = []
    for var, k in lag_specs:
        if var in df.columns and f"{var}_lag{k}" not in df.columns:
            new_cols.append(pl.col(var).shift(k).alias(f"{var}_lag{k}"))
    return df.with_columns(new_cols) if new_cols else df


# ---------------------------------------------------------------------------
# Token parsing: "LCSZ(-1)" -> ("LCSZ_lag1", ("LCSZ", 1))
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"^([A-Z0-9$]+)(?:\(\s*(-?\d+)\s*\))?$")


def _parse_token(token: str) -> tuple[str, str, int]:
    """Parse ``"LCSZ(-1)"`` into ``(column_name, base_var, lag)``.

    Lag 0 (or no lag) returns the base variable name unchanged. Lag -k
    returns ``"{base}_lag{k}"``.
    """
    m = _TOKEN_RE.match(token)
    if not m:
        raise ValueError(f"Unparseable token: {token!r}")
    base, lag_str = m.group(1), m.group(2)
    lag = 0 if lag_str is None else -int(lag_str)
    if lag == 0:
        return base, base, 0
    return f"{base}_lag{lag}", base, lag


def _required_lags(tokens: list[str]) -> set[tuple[str, int]]:
    """Collect the ``(base, lag)`` pairs we need lag columns for."""
    lags: set[tuple[str, int]] = set()
    for token in tokens:
        _col, base, lag = _parse_token(token)
        if lag > 0:
            lags.add((base, lag))
    return lags


# ---------------------------------------------------------------------------
# Equation registry (populated below)
# ---------------------------------------------------------------------------

# Reference coefficients parsed once from fmout.txt at module import. Keyed by
# equation number and token string (e.g. "LCSZ(-1)" or "RHO(-1)").
REFERENCE_PARAMS: dict[int, dict[str, float]] = {}

# Iteration sequence reference (first iteration rho for AR(1) equations).
REFERENCE_RHO_ITER1: dict[int, float] = {}


def _parse_fmout(path: Path = config.US_FMOUT) -> None:
    """Populate ``REFERENCE_PARAMS`` from Fair's estimation output file."""
    text = path.read_text()
    block_starts = [m.start() for m in
                    re.finditer(r"^Equation number =\s+\d+", text, re.M)]
    block_starts.append(len(text))
    coef_re = re.compile(
        r"^\s*\d+\s+(\S+)\s*\(\s*(-?\d+)\)\s+([-+\d\.E]+)\s+[-+\d\.E]+\s+[-+\d\.]+",
        re.M,
    )
    for i in range(len(block_starts) - 1):
        block = text[block_starts[i]: block_starts[i + 1]]
        num_match = re.match(r"Equation number =\s+(\d+)", block)
        if not num_match:
            continue
        eq_num = int(num_match.group(1))
        # Use the first occurrence of this equation number only.
        if eq_num in REFERENCE_PARAMS:
            continue
        # Coefficient block ends at "SE of equation".
        se_pos = block.find("SE of equation")
        if se_pos < 0:
            continue
        table = block[:se_pos]
        coefs = {}
        for name, lag, val in coef_re.findall(table):
            # Convert "VAR(-k)" style key to match our token convention.
            key = f"{name}({int(lag):+d})" if lag != "0" else f"{name}(0)"
            coefs[key] = float(val)
        REFERENCE_PARAMS[eq_num] = coefs


_parse_fmout()


# ---------------------------------------------------------------------------
# Hand-curated equation specs
# ---------------------------------------------------------------------------

_COVID_DUMMIES = ("D20201", "D20202", "D20203", "D20204",
                  "D20211", "D20212", "D20213", "D20214")
_COVID_INSTRUMENTS = _COVID_DUMMIES + ("D20214(-1)",)
_COVID_INSTRUMENTS_NO_LAG = _COVID_DUMMIES


def _eq(
    number: int,
    dependent: str,
    regressors: str,
    instruments: str,
    has_ar1: bool,
    *,
    sample_start: str = "1954Q1",
    sample_end: str = "2025Q4",
    covid_lag_in_fsr: bool = True,
    has_covid: bool = True,
) -> UsEquation:
    """Factory that splits whitespace-separated regressor/instrument strings.

    Appends the MODEQ COVID dummies per Fair's fminput.txt when
    ``has_covid`` is True. Set ``has_covid=False`` for equations whose sample
    ends before 2020 (EQ 28) or otherwise lack a ``MODEQ`` line.
    """
    reg = tuple(regressors.split())
    inst = tuple(instruments.split())
    if has_covid:
        reg += _COVID_DUMMIES
        inst += _COVID_INSTRUMENTS if covid_lag_in_fsr else _COVID_INSTRUMENTS_NO_LAG
    return UsEquation(number, dependent, reg, inst, has_ar1, sample_start, sample_end)


# fminput.txt lines 155-208 for structural equations + lines 387-478 for FSR lists.
# MODEQ line 536-575 adds COVID dummies to both regressors and instruments.
EQUATIONS: list[UsEquation] = [
    _eq(1,  "LCSZ",
        "CNST2CS C AG1 AG2 AG3 LCSZ(-1) LYDZ RSA LAAZ(-1)",
        "CNST2CS C AG1 AG2 AG3 LCSZ(-1) LAAZ(-3) RSA(-1) "
        "CNST2CS(-1) AG1(-1) AG2(-1) AG3(-1) LCSZ(-2) "
        "LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1) LPOP LPOP(-1)",
        has_ar1=True),
    _eq(2,  "LCNZ",
        "C AG1 AG2 AG3 LCNZ(-1) LAAZ(-1) LYDZ RMA",
        "C AG1 AG2 AG3 LCNZ(-1) LAAZ(-3) LYDZ(-1) RMA(-1) "
        "LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1) AG1(-1) AG2(-1) AG3(-1) LCNZ(-2)",
        has_ar1=True),
    _eq(3,  "LCDZ",
        "C AG1 AG2 AG3 LCDZ(-1) LYDZ RMA",
        "C AG1 AG2 AG3 LAAZ(-3) LYDZ(-1) RMA(-1) LCDZ(-1) "
        "LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1) T",
        has_ar1=False, covid_lag_in_fsr=False),
    # Near-unit-root AR (Fair ρ=0.916) — needs heavy damping.
    UsEquation(4, "LIHHZ",
        regressors=tuple("C AG1 AG2 AG3 LIHHZ(-1) LYDZ RMA(-1)".split()) + _COVID_DUMMIES,
        instruments=tuple(("C RMA(-1) LYDZ(-1) AG1 AG2 AG3 LIHHZ(-1) "
                           "LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1) T "
                           "AG1(-1) AG2(-1) AG3(-1) LIHHZ(-2) RMA(-2)").split())
                    + _COVID_INSTRUMENTS,
        has_ar1=True, sample_start="1954Q1", sample_end="2025Q4",
        use_bounded_search=True),
    _eq(5,  "LL1Z",
        "C LL1Z(-1) LAAZ(-1) UR",
        "C LL1Z(-1) LAAZ(-3) UR(-1) LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1)",
        has_ar1=False, covid_lag_in_fsr=False),
    _eq(6,  "LL2Z",
        "CNST2L2 C TBL2 T LL2Z(-1) LAAZ(-1) UR",
        "C CNST2L2 T TBL2 LL2Z(-1) LAAZ(-3) UR(-1) "
        "LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1)",
        has_ar1=False, covid_lag_in_fsr=False),
    _eq(7,  "LL3Z",
        "C LL3Z(-1) LWAZPH LAAZ(-1) UR",
        "C LL3Z(-1) LAAZ(-3) LWAZPH(-1) UR(-1) LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1)",
        has_ar1=False, covid_lag_in_fsr=False),
    _eq(8,  "LLMZ",
        "C LLMZ(-1) UR",
        "C LLMZ(-1) UR(-1) LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1)",
        has_ar1=False, covid_lag_in_fsr=False),
    _eq(10, "LPF",
        "LPF(-1) LWFD5 C T LPIM ONEZUR LCUSTZ",
        "LPF(-1) LWFD5(-1) C T LPIM(-1) ONEZUR(-1) UR(-1) "
        "LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1) "
        "LPF(-2) LCUSTZ(-1) LCUSTZ(-2)",
        has_ar1=True),
    _eq(13, "LJF1",
        "C LEXL(-1) LJF1(-1) LY1 D593",
        "C LEXL(-1) LJF1(-1) LY1(-1) D593 LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1)",
        has_ar1=False, covid_lag_in_fsr=False),
    _eq(14, "LHF1",
        "C LHFL1A LEXL(-1) LY1 T",
        "C LHFL1A LEXL(-1) LY1(-1) T LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1)",
        has_ar1=False, covid_lag_in_fsr=False),
    _eq(17, "LMFZ",
        "C LMFZ(-1) LXMFA RSB",
        "C LMFZ(-1) LXMFA(-1) RSB(-1) LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1) LMFL1Q(-1)",
        has_ar1=False, covid_lag_in_fsr=False),
    _eq(23, "RBMRSL2",
        "C RBLMRSL2 RSMRSL2 RSLMRSL2",
        "C RB(-1) RB(-2) RS(-1) RS(-2) RS(-3) "
        "LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1) PCPD(-1) UR(-1) T LPIMZ(-1)",
        has_ar1=True),
    _eq(24, "RMMRSL2",
        "C RMLMRSL2 RSMRSL2 RSLMRSL2",
        "C RM(-1) RS(-1) RS(-2) LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1) "
        "PCPD(-1) UR(-1) T LPIMZ(-1)",
        has_ar1=False, covid_lag_in_fsr=False),
    _eq(26, "LCURZ",
        "C LCURZ(-1) LXMFAZ RSA",
        "C LCURZ(-1) LXMFAZ(-1) RSA(-1) LCURL1Q(-1) "
        "LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1)",
        has_ar1=False, covid_lag_in_fsr=False),
    _eq(18, "LDF1",
        "LPIEFAZ",
        "LPIEFAZ(-1) C LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1)",
        has_ar1=False, covid_lag_in_fsr=False),
    # EQ 11 — firm output. Fair uses AR(3); we approximate with AR(1) via
    # bounded search. Coefficients will differ slightly from Fair's RHO=3 fit.
    _eq(11, "LY",
        "C LY(-1) LX LV(-1) D593 D594 D601",
        "C LY(-1) LV(-1) LY(-2) LY(-3) LY(-4) LV(-2) LV(-3) LV(-4) "
        "D593 D594 D601 D601(-1) D601(-2) D601(-3) "
        "LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1)",
        has_ar1=True),
    # EQ 12 — capital stock growth. AR(1).
    _eq(12, "LKK1",
        "CNST2KK C LEXKK(-1) LKK1(-1) LY1 LY1(-1) LY1(-2) LY1(-3) LY1(-4) CGZ3(-2)",
        "CNST2KK C LKK(-1) LKK(-2) LY(-1) LY(-2) LY(-3) LY(-4) LY(-5) "
        "CGZ3(-2) LEXKK(-1) LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1) "
        "CNST2KK(-1) LEXKK(-2) LKK1(-2) LY1(-5) CGZ3(-3)",
        has_ar1=True),
    # EQ 15 — overtime hours. Near-unit-root AR (ρ≈0.97); uses bounded search.
    # Fair uses a later sample start (1956Q1). No FSR → OLS via identity instruments.
    UsEquation(15, "LHO",
        regressors=tuple("C HFF HFF(-1)".split()) + _COVID_DUMMIES,
        # No explicit FSR in fminput — Fair runs pure OLS. Pass the regressors
        # themselves as "instruments" so 2SLS collapses to OLS (X = X).
        instruments=tuple("C HFF HFF(-1)".split()) + _COVID_DUMMIES,
        has_ar1=True, sample_start="1956Q1", sample_end="2025Q4",
        use_bounded_search=True),
    # EQ 30 — Fed policy rule. Fair clips RS at zero (ZLB) via an ABS LHS;
    # we estimate linearly, ignoring the clipping. Sample ends at 2008Q3
    # (Fair avoids the ZLB era in estimation).
    _eq(30, "RS",
        "C RS(-1) PCPD UR UR1 PCM1L1B PCM1L1A RS1(-1) RS1(-2)",
        "C RS(-1) PCPD(-1) UR(-1) UR1(-1) PCM1L1B PCM1L1A RS1(-1) RS1(-2) "
        "LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1)",
        has_ar1=False, sample_end="2008Q3", has_covid=False),
    _eq(27, "LIMZ",
        "C AG1 AG2 AG3 LIMZ(-1) LYZ LPFZPIM T D691 D692 D714 D721",
        "C LIMZ(-1) LYZ(-1) LPFZPIM(-1) D691 D692 D714 D721 "
        "AG1 AG2 AG3 LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1) T "
        "LPOP LPOP(-1) LPIM(-1) LIMZ(-2)",
        has_ar1=False),
    # Near-unit-root AR (Fair ρ=0.896) — needs heavy damping.
    UsEquation(28, "LUB",
        regressors=tuple("C LUB(-1) LU LWF".split()),
        instruments=tuple(("C LUB(-1) LU(-1) LWF(-1) LUB(-2) "
                           "LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1) LPIMZ(-1) PCPD(-1) T").split()),
        has_ar1=True, sample_start="1954Q1", sample_end="2000Q4",
        use_bounded_search=True),
    # Thin instrument set (3 for 3 regressors); mild damping stabilizes.
    UsEquation(29, "INTGZ",
        regressors=tuple("C INTGZ(-1) RQG".split()) + _COVID_DUMMIES,
        instruments=tuple("C INTGZ(-1) RQG LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1)".split())
                    + _COVID_INSTRUMENTS_NO_LAG,
        has_ar1=True, sample_start="1954Q1", sample_end="2025Q4",
        use_bounded_search=True),
]

# ---------------------------------------------------------------------------
# Runtime data preparation + per-equation estimation
# ---------------------------------------------------------------------------

# Some tokens in regressor/instrument lists reference derived variables that
# need lag columns; gather them all once.
def _all_required_lags() -> set[tuple[str, int]]:
    lags: set[tuple[str, int]] = set()
    for eq in EQUATIONS:
        lags |= _required_lags(list(eq.regressors))
        lags |= _required_lags(list(eq.instruments))
    return lags


def build_frame(fmdata_path: Path | None = None) -> pl.DataFrame:
    """Parse fmdata.txt, compute everything needed for all active equations."""
    path = fmdata_path or config.US_FMDATA
    wide = readers.pivot_to_wide(readers.parse_fair_data(path)).sort("period")
    wide = add_time_trend_and_constant(wide)
    wide = apply_genr(wide)
    wide = apply_regime_dummies(wide)
    wide = add_lags(wide, _all_required_lags())
    return wide


def _period_before(period: str) -> str:
    """Return the quarter one step before ``period`` (e.g. ``"1954Q1"`` -> ``"1953Q4"``)."""
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


def _row(df: pl.DataFrame, tokens: tuple[str, ...]) -> jnp.ndarray:
    cols = [_parse_token(t)[0] for t in tokens]
    return jnp.array([float(df[c].item()) for c in cols])


@dataclass
class EstimationResult:
    """Per-equation estimation output and comparison vs Fair's reference."""
    equation: UsEquation
    coefficients: dict[str, float]      # token -> our estimate
    reference: dict[str, float]         # token -> Fair's fmout.txt value
    n_obs: int
    rho_iterations: int | None


def estimate(equation: UsEquation, frame: pl.DataFrame) -> EstimationResult:
    """Estimate one Fair equation end-to-end."""
    start = equation.sample_start
    end = equation.sample_end
    presample_period = _period_before(start)

    required_cols = {_parse_token(t)[0] for t in
                     (*equation.regressors, *equation.instruments, equation.dependent)}
    window = frame.filter(
        (pl.col("period") >= pl.lit(presample_period))
        & (pl.col("period") <= pl.lit(end))
    ).drop_nulls(subset=list(required_cols & set(frame.columns)))
    presample_row = window.head(1)
    est = window.tail(window.height - 1)

    y = jnp.asarray(est[equation.dependent].to_numpy(), dtype=jnp.float64)
    X = _stack(est, equation.regressors)
    Z = _stack(est, equation.instruments)

    if equation.has_ar1:
        y_ps = jnp.asarray(float(presample_row[equation.dependent].item()))
        X_ps = _row(presample_row, equation.regressors)
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

    # Build reference dict keyed by the same token form (``VAR(0)`` or ``VAR(-k)``).
    ref_raw = REFERENCE_PARAMS.get(equation.number, {})
    reference = {}
    for tok in equation.regressors:
        _col, base, lag = _parse_token(tok)
        key = f"{base}(0)" if lag == 0 else f"{base}({-lag:+d})"
        reference[tok] = ref_raw.get(key, float("nan"))
    if equation.has_ar1:
        reference["RHO(-1)"] = ref_raw.get("RHO(-1)", float("nan"))

    return EstimationResult(
        equation=equation,
        coefficients=coefs,
        reference=reference,
        n_obs=est.height,
        rho_iterations=iter_count,
    )


def _estimate_eq16_wage(
    eq10_result: EstimationResult, frame: pl.DataFrame,
) -> EstimationResult:
    """Estimate the wage equation (EQ 16) using EQ 10's coefficients.

    Fair's two-step recipe (fminput.txt lines 581-598):
      1. Estimate EQ 10 (price, LPF) to get β(LPF(-1)) = BETA1 and
         β(LWFD5) = BETA2.
      2. Compute DELTA1 = BETA1 / (1 − BETA2).
      3. Generate LWFQZ = LWFQ − DELTA1 · LPF(-1) and
         LPFZ = LPF − DELTA1 · LPF(-1).
      4. Estimate EQ 16: LWFQZ ~ LWFQZPF(-1) + LPFZ + C (+ COVID).
    """
    beta1 = eq10_result.coefficients["LPF(-1)"]
    beta2 = eq10_result.coefficients["LWFD5"]
    delta1 = beta1 / (1.0 - beta2)

    frame_with_wage = frame.with_columns(
        (_c("LWFQ") - delta1 * _c("LPF").shift(1)).alias("LWFQZ"),
        (_c("LPF") - delta1 * _c("LPF").shift(1)).alias("LPFZ"),
        pl.col("LWFQZPF").shift(1).alias("LWFQZPF_lag1"),
        pl.col("LPF").shift(2).alias("LPF_lag2"),
    )

    eq16 = UsEquation(
        number=16, dependent="LWFQZ",
        regressors=tuple("LWFQZPF(-1) LPFZ C".split()) + _COVID_DUMMIES,
        instruments=tuple(("LWFQZPF(-1) LPF(-1) LPF(-2) C T "
                           "LCOGSZ(-1) LTRGSZ(-1) LEXZ(-1) "
                           "LPIM(-1) ONEZUR(-1) UR(-1)").split())
                    + _COVID_INSTRUMENTS_NO_LAG,
        has_ar1=False,
        sample_start="1954Q1", sample_end="2025Q4",
    )
    return estimate(eq16, frame_with_wage)


def estimate_all(
    fmdata_path: Path | None = None,
) -> list[EstimationResult]:
    """Estimate every active equation and return the list of results."""
    frame = build_frame(fmdata_path)
    results = []
    eq10_result = None
    for eq in EQUATIONS:
        try:
            result = estimate(eq, frame)
            results.append(result)
            if eq.number == 10:
                eq10_result = result
        except Exception:
            _LOG.exception(
                "EQ %d (%s) estimation failed", eq.number, eq.dependent,
            )

    # Two-step: EQ 16 depends on EQ 10's coefficients.
    if eq10_result is not None:
        try:
            results.append(_estimate_eq16_wage(eq10_result, frame))
        except Exception:
            _LOG.exception("EQ 16 (LWFQZ) estimation failed")

    return results


def drift_report(results: list[EstimationResult]) -> pl.DataFrame:
    """Flatten ``estimate_all`` output into a (eq, token, fair, ours, delta) frame."""
    rows = []
    for r in results:
        for tok, our in r.coefficients.items():
            fair = r.reference.get(tok, float("nan"))
            rows.append({
                "eq": r.equation.number,
                "dependent": r.equation.dependent,
                "token": tok,
                "fair": fair,
                "ours": our,
                "delta": our - fair if fair == fair else float("nan"),
                "abs_err": abs(our - fair) if fair == fair else float("nan"),
            })
    return pl.DataFrame(rows)


if __name__ == "__main__":
    results = estimate_all()
    print(f"\n=== Estimated {len(results)} / {len(EQUATIONS)} equations ===\n")
    for r in results:
        eq = r.equation
        deltas = [abs(our - r.reference.get(tok, float("nan")))
                  for tok, our in r.coefficients.items()
                  if r.reference.get(tok, float("nan")) == r.reference.get(tok, float("nan"))]
        max_err = max(deltas) if deltas else float("nan")
        iters = f" iters={r.rho_iterations}" if r.rho_iterations is not None else ""
        print(f"  EQ {eq.number:2d} {eq.dependent:10s} n_obs={r.n_obs:3d}"
              f"  max_abs_err={max_err:.2e}{iters}")
