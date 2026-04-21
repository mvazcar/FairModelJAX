"""US Model — dynamic simulation.

Combines the 24 estimated stochastic equations with Fair's accounting
identities into one simultaneous system and solves it quarter-by-quarter.

Structure mirrors IS: `jax.lax.scan` over the forecast window, Newton-Raphson
per period via `solver.make_scan_step`, all pure JAX. The model is larger
(~40+ endogenous variables joint per period), which slows the Jacobian solve
but doesn't change the algorithm.

Identity source: ``03_us_model/fminput.txt`` lines 212–305.
Stochastic equation source: :mod:`pyfair.us_model` ``EQUATIONS`` list.

The solve-able "core" of the US model (what this module covers):

* Goods identities: X, V, KD, KH, HN
* Price identities: PX, PEX, PD, PH, PCS, PCN, PCD, PIH, PIK, PG, PS, PIV, PKH
* Wage identities: WH, WG, WM, WS
* Tax identities: THG, THS, TFG, TFS, SIHG, SIFG
* Financial simplifications: INTGR, USROW

Deferred (treated as exogenous for v0.3):

* Government and financial stock/flow accounts (YT, SH, AH, PIEF, SF, AF, MB,
  SB, AB, SR, AR, SG, AG, SS, AS, MG, MS, MB, MH, MR, etc.)
* CCF1 (capital consumption with lags).

These are not essential for solving the structural core; they propagate
policy/fiscal effects through the model but don't feed back into the core
demand/price/labor block we estimate.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    # Type-only imports — keep them out of the runtime import graph. The
    # functions below also do `import polars as pl` and
    # `from . import model as us_model` locally, so runtime behaviour is
    # unchanged; this block only helps static tools resolve the
    # forward-reference annotations.
    import polars as pl
    from . import model as us_model

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Identity registry
# ---------------------------------------------------------------------------

def _c(state: dict, name: str) -> jnp.ndarray:
    """Shorthand: state[name]."""
    return state[name]


# Each identity is a function (state, params) -> residual.
# Residual = LHS − RHS; Newton drives to zero to recover the LHS variable.
# state is a dict mapping variable names to scalar jnp values for one period.


def ident_X(state, _params):
    """X = CS + CN + CD + IHH + IKF + EX − IM + COG + COS + IKH + IKB + IKG + IHF."""
    return state["X"] - (
        state["CS"] + state["CN"] + state["CD"] + state["IHH"] + state["IKF"]
        + state["EX"] - state["IM"] + state["COG"] + state["COS"]
        + state["IKH"] + state["IKB"] + state["IKG"] + state["IHF"]
    )


def ident_V(state, _params):
    """V = V(-1) + Y − X (inventory stock)."""
    return state["V"] - (state["V_lag1"] + state["Y"] - state["X"])


def ident_KD(state, _params):
    """KD = (1 − DELD) · KD(-1) + CD (durables stock)."""
    return state["KD"] - ((1.0 - state["DELD"]) * state["KD_lag1"] + state["CD"])


def ident_KH(state, _params):
    """KH = (1 − DELH) · KH(-1) + IHH (housing stock)."""
    return state["KH"] - ((1.0 - state["DELH"]) * state["KH_lag1"] + state["IHH"])


def ident_HN(state, _params):
    """HN = HF − HO (non-overtime hours)."""
    return state["HN"] - (state["HF"] - state["HO"])


def ident_PX(state, _params):
    """PX = (PF · (X − FA) + PFA · FA) / X  (output deflator)."""
    return state["PX"] - (
        (state["PF"] * (state["X"] - state["FA"]) + state["PFA"] * state["FA"])
        / state["X"]
    )


def ident_PEX(state, _params):
    """PEX = PSI1 · PX (export deflator, PSI1 exogenous)."""
    return state["PEX"] - state["PSI1"] * state["PX"]


def ident_PD(state, _params):
    """PD = (PX·X − PEX·EX + PIM·IM) / (X − EX + IM) (domestic-sales deflator)."""
    return state["PD"] - (
        (state["PX"] * state["X"] - state["PEX"] * state["EX"] + state["PIM"] * state["IM"])
        / (state["X"] - state["EX"] + state["IM"])
    )


def ident_PH(state, _params):
    """PH = (PCS·CS + PCN·CN + PCD·CD + PIH·IHH) / (CS+CN+CD+IHH)."""
    numer = (state["PCS"] * state["CS"] + state["PCN"] * state["CN"]
             + state["PCD"] * state["CD"] + state["PIH"] * state["IHH"])
    denom = state["CS"] + state["CN"] + state["CD"] + state["IHH"]
    return state["PH"] - numer / denom


def ident_PCS(state, _params):
    return state["PCS"] - state["PSI2"] * state["PD"]


def ident_PCN(state, _params):
    return state["PCN"] - state["PSI3"] * state["PD"]


def ident_PCD(state, _params):
    return state["PCD"] - state["PSI4"] * state["PD"]


def ident_PIH(state, _params):
    return state["PIH"] - state["PSI5"] * state["PD"]


def ident_PIK(state, _params):
    return state["PIK"] - state["PSI6"] * state["PD"]


def ident_PG(state, _params):
    return state["PG"] - state["PSI7"] * state["PD"]


def ident_PS(state, _params):
    return state["PS"] - state["PSI8"] * state["PD"]


def ident_PIV(state, _params):
    return state["PIV"] - state["PSI9"] * state["PD"]


def ident_PKH(state, _params):
    return state["PKH"] - state["PSI14"] * state["PD"]


def ident_WG(state, _params):
    return state["WG"] - state["PSI10"] * state["WF"]


def ident_WM(state, _params):
    return state["WM"] - state["PSI11"] * state["WF"]


def ident_WS(state, _params):
    return state["WS"] - state["PSI12"] * state["WF"]


def ident_WH(state, _params):
    """Weighted average wage."""
    numer = (state["WF"] * state["JF"] * (state["HN"] + 1.5 * state["HO"])
             + state["WG"] * state["JG"] * state["HG"]
             + state["WM"] * state["JM"] * state["HM"]
             + state["WS"] * state["JS"] * state["HS"])
    denom = (state["JF"] * (state["HN"] + 1.5 * state["HO"])
             + state["JG"] * state["HG"]
             + state["JM"] * state["HM"]
             + state["JS"] * state["HS"])
    return state["WH"] - 100.0 * numer / denom


def ident_THG(state, _params):
    return state["THG"] - state["D1G"] * state["YT"]


def ident_THS(state, _params):
    return state["THS"] - state["D1S"] * state["YT"]


def ident_TFG(state, _params):
    return state["TFG"] - state["D2G"] * (state["PIEF"] - state["TFS"])


def ident_TFS(state, _params):
    return state["TFS"] - state["D2S"] * state["PIEF"]


def ident_SIHG(state, _params):
    return state["SIHG"] - state["D4G"] * (
        state["WF"] * state["JF"] * (state["HN"] + 1.5 * state["HO"])
    )


def ident_SIFG(state, _params):
    return state["SIFG"] - state["D5G"] * (
        state["WF"] * state["JF"] * (state["HN"] + 1.5 * state["HO"])
    )


def ident_INTGR(state, _params):
    return state["INTGR"] - state["PSI15"] * state["INTG"]


# Labour and slack ---------------------------------------------------------

def ident_E(state, _params):
    """Employment: E = JF + JG + JM + JS − LM."""
    return state["E"] - (state["JF"] + state["JG"] + state["JM"]
                         + state["JS"] - state["LM"])


def ident_U(state, _params):
    """Unemployed: U = L1 + L2 + L3 − E."""
    return state["U"] - (state["L1"] + state["L2"] + state["L3"] - state["E"])


def ident_UR(state, _params):
    """Unemployment rate: UR = U / (L1 + L2 + L3 − AFT)."""
    return state["UR"] - state["U"] / (
        state["L1"] + state["L2"] + state["L3"] - state["AFT"]
    )


def ident_POP(state, _params):
    return state["POP"] - (state["POP1"] + state["POP2"] + state["POP3"])


# Assets & wealth ----------------------------------------------------------

def ident_AA1(state, _params):
    return state["AA1"] - (state["AH"] + state["MH"]) / state["PH"]


def ident_AA2(state, _params):
    return state["AA2"] - state["KH"] * state["PKH"] / state["PH"]


def ident_AA(state, _params):
    return state["AA"] - (state["AA1"] + state["AA2"])


# Production side ----------------------------------------------------------

def ident_IKF(state, _params):
    return state["IKF"] - (state["KK"] - (1.0 - state["DELK"]) * state["KK_lag1"])


def ident_KKMIN(state, _params):
    return state["KKMIN"] - state["Y"] / state["MUH"]


def ident_JHMIN(state, _params):
    return state["JHMIN"] - state["Y"] / state["LAM"]


def ident_IVF(state, _params):
    return state["IVF"] - (state["V"] - state["V_lag1"])


def ident_PROD(state, _params):
    return state["PROD"] - state["Y"] / (state["JF"] * state["HF"])


def ident_WR(state, _params):
    return state["WR"] - state["WF"] / state["PF"]


def ident_HFF(state, _params):
    return state["HFF"] - (state["HF"] - state["HFS"])


# NIPA aggregates ----------------------------------------------------------

def ident_XX(state, _params):
    """Nominal GDP-style aggregate (without inventory investment)."""
    return state["XX"] - (
        state["PCS"] * state["CS"] + state["PCN"] * state["CN"]
        + state["PCD"] * state["CD"]
        + state["PIH"] * (state["IHF"] + state["IHH"])
        + state["PIK"] * (state["IKH"] + state["IKG"] + state["IKB"] + state["IKF"])
        + state["PEX"] * state["EX"] - state["PIM"] * state["IM"]
        + state["PG"] * state["COG"] + state["PS"] * state["COS"]
    )


def ident_GDP(state, _params):
    return state["GDP"] - (
        state["XX"] + state["PIV"] * (state["V"] - state["V_lag1"])
        + state["WG"] * state["JG"] * state["HG"]
        + state["WM"] * state["JM"] * state["HM"]
        + state["WS"] * state["JS"] * state["HS"]
    )


def ident_GDPR(state, _params):
    return state["GDPR"] - (
        state["Y"]
        + state["PSI13"] * (state["JG"] * state["HG"]
                            + state["JM"] * state["HM"]
                            + state["JS"] * state["HS"])
        + state["STATP"]
    )


def ident_GDPD(state, _params):
    return state["GDPD"] - state["GDP"] / state["GDPR"]


def ident_GNP(state, _params):
    return state["GNP"] - (state["GDP"] + state["USROW"])


def ident_GNPR(state, _params):
    return state["GNPR"] - (state["GDPR"] + state["USROW"] / state["GDPD"])


def ident_GNPD(state, _params):
    return state["GNPD"] - state["GNP"] / state["GNPR"]


# Government accounting ---------------------------------------------------

def ident_CDH(state, _params):
    return state["CDH"] - state["THETA2"] * state["PCD"] * state["CD"]


def ident_NICD(state, _params):
    return state["NICD"] - state["THETA3"] * state["PCD"] * state["CD"]


def ident_TF1(state, _params):
    return state["TF1"] - (state["TFG"] + state["TFS"] + state["TFR"])


def ident_TCG(state, _params):
    return state["TCG"] - (state["TFG"] + state["TBG"])


def ident_SIG(state, _params):
    return state["SIG"] - (state["SIHG"] + state["SIFG"] + state["SIGG"])


def ident_PUG(state, _params):
    return state["PUG"] - (
        state["PG"] * state["COG"]
        + state["WG"] * state["JG"] * state["HG"]
        + state["WM"] * state["JM"] * state["HM"]
    )


def ident_RECG(state, _params):
    return state["RECG"] - (
        state["THG"] + state["TCG"] + state["IBTG"] + state["CUST"]
        + state["SIG"] + state["TRFG"] - state["DG"]
    )


def ident_EXPG(state, _params):
    return state["EXPG"] - (
        state["PUG"] + state["TRGH"] + state["TRGR"] + state["TRGS"]
        + state["INTG"] + state["SUBG"] - state["IGZ"] + state["UB"]
    )


def ident_SGP(state, _params):
    return state["SGP"] - (state["RECG"] - state["EXPG"])


def ident_TCS(state, _params):
    return state["TCS"] - (state["TFS"] + state["TBS"])


def ident_SIS(state, _params):
    return state["SIS"] - (state["SIHS"] + state["SIFS"] + state["SISS"])


def ident_PUS(state, _params):
    return state["PUS"] - (
        state["PS"] * state["COS"]
        + state["WS"] * state["JS"] * state["HS"]
    )


def ident_RECS(state, _params):
    return state["RECS"] - (
        state["THS"] + state["TCS"] + state["IBTS"] + state["SIS"]
        + state["TRGS"] + state["TRFS"] - state["DS"]
    )


def ident_EXPS(state, _params):
    return state["EXPS"] - (
        state["PUS"] + state["TRSH"] + state["INTS"] + state["SUBS"]
        - state["ISZ"]
    )


def ident_SSP(state, _params):
    return state["SSP"] - (state["RECS"] - state["EXPS"])


def ident_PFA(state, _params):
    return state["PFA"] - state["THETA1"] * state["GDPD"]


# Income, saving, stocks ---------------------------------------------------

def ident_YT(state, _params):
    """Total income subject to tax."""
    labor_income = (state["WF"] * state["JF"] * (state["HN"] + 1.5 * state["HO"])
                    + state["WG"] * state["JG"] * state["HG"]
                    + state["WM"] * state["JM"] * state["HM"]
                    + state["WS"] * state["JS"] * state["HS"])
    other = (state["RNT"] + state["INTZ"] + state["INTF"] + state["INTG"]
             - state["INTGR"] + state["INTS"] + state["DF"] + state["DB"]
             + state["DR"] + state["DG"] + state["DS"] + state["TRFH"]
             - state["TRHR"] - state["SIGG"] - state["SISS"])
    return state["YT"] - (labor_income + other)


def ident_YD(state, _params):
    """Disposable income."""
    labor_income = (state["WF"] * state["JF"] * (state["HN"] + 1.5 * state["HO"])
                    + state["WG"] * state["JG"] * state["HG"]
                    + state["WM"] * state["JM"] * state["HM"]
                    + state["WS"] * state["JS"] * state["HS"])
    other = (state["RNT"] + state["INTZ"] + state["INTF"] + state["INTG"]
             - state["INTGR"] + state["INTS"] + state["DF"] + state["DB"]
             + state["DR"] + state["DG"] + state["DS"])
    transfers = (state["TRFH"] + state["TRGH"] + state["TRSH"] + state["UB"]
                 - state["SIHG"] - state["SIHS"] - state["THG"] - state["THS"]
                 - state["TRHR"] - state["SIGG"] - state["SISS"])
    return state["YD"] - (labor_income + other + transfers)


def ident_SRZ(state, _params):
    return state["SRZ"] - (
        (state["YD"] - state["PCS"] * state["CS"] - state["PCN"] * state["CN"]
         - state["PCD"] * state["CD"]) / state["YD"]
    )


def ident_SH(state, _params):
    """Household saving."""
    return state["SH"] - (
        state["YT"] - state["SIHG"] - state["SIHS"] - state["THG"] - state["THS"]
        - state["PCS"] * state["CS"] - state["PCN"] * state["CN"]
        - state["PCD"] * state["CD"] + state["TRGH"] + state["TRSH"] + state["UB"]
        + state["INS"] + state["NICD"] + state["CCH"] - state["CTH"]
        - state["PIH"] * state["IHH"] - state["CDH"] - state["PIK"] * state["IKH"]
        - state["NNH"]
    )


def ident_AH(state, _params):
    return state["AH"] - (state["AH_lag1"] + state["SH"] - state["MH"]
                          + state["MH_lag1"] + state["CG"] - state["DISH"])


def ident_CCF1(state, _params):
    return state["CCF1"] - state["D6G"] * (
        state["PIK"] * state["IKF"]
        + state["PIK_lag1"] * state["IKF_lag1"]
        + state["PIK_lag2"] * state["IKF_lag2"]
        + state["PIK_lag3"] * state["IKF_lag3"]
    ) / 4.0


def ident_PIEF(state, _params):
    """Firm profit (before tax)."""
    return state["PIEF"] - (
        state["XX"] + state["PIV"] * state["IVF"]
        + state["SUBS"] + state["SUBG"] + state["USOTHER"]
        - state["WF"] * state["JF"] * (state["HN"] + 1.5 * state["HO"])
        - state["RNT"] - state["INTZ"] - state["INTF"]
        - state["TRFH"] - state["NICD"] - state["CCH"] + state["CDH"]
        - state["TBS"] - state["TRFS"] - state["CCS"] - state["TRFR"]
        - state["DB"] - state["GSB"] - state["CTGB"]
        - state["GSMA"] - state["GSCA"] - state["TBG"] - state["TRFG"] - state["CCG"]
        - state["SIFG"] - state["SIFS"]
        - state["GSNN"] - state["IVA"] - state["CCF1"] - state["STAT"]
        + state["TTRRF"]
        - state["IBTG"] - state["CUST"] - state["IBTS"]
    )


def ident_PIEFRET(state, _params):
    return state["PIEFRET"] - state["THETA4"] * state["PIEF"]


def ident_USROW(state, _params):
    return state["USROW"] - (
        -state["INTGR"] + state["DR"] + state["PIEFRET"] + state["USOTHER"]
    )


def ident_SF(state, _params):
    """Firm saving."""
    return state["SF"] - (
        state["XX"] + state["SUBS"] + state["SUBG"] + state["USOTHER"]
        + state["PIEFRET"]
        - state["WF"] * state["JF"] * (state["HN"] + 1.5 * state["HO"])
        - state["RNT"] - state["INTZ"] - state["INTF"]
        - state["TRFH"] - state["NICD"] - state["CCH"] + state["CDH"]
        - state["TBS"] - state["TRFS"] - state["CCS"]
        - state["TRFR"]
        - state["DB"] - state["GSB"] - state["CTGB"]
        - state["GSMA"] - state["GSCA"] - state["TBG"] - state["TRFG"] - state["CCG"]
        - state["SIFG"] - state["SIFS"]
        - state["STAT"]
        - state["DF"] - state["TF1"]
        - state["PIK"] * state["IKF"] - state["PIH"] * state["IHF"]
        - state["NNF"] - state["CTF1"] - state["CTNN"]
        + state["TTRRF"]
        - state["IBTG"] - state["CUST"] - state["IBTS"]
    )


def ident_AF(state, _params):
    return state["AF"] - (state["AF_lag1"] + state["SF"] - state["MF"]
                          + state["MF_lag1"] - state["DISF"])


def ident_MB(state, _params):
    """Bank currency reserves."""
    return state["MB"] - (
        state["MB_lag1"]
        - state["MH"] + state["MH_lag1"]
        - state["MF"] + state["MF_lag1"]
        - state["MR"] + state["MR_lag1"]
        - state["MG"] + state["MG_lag1"]
        - state["MS"] + state["MS_lag1"]
        + state["CUR"] - state["CUR_lag1"]
    )


def ident_SB(state, _params):
    return state["SB"] - (state["GSB"] - state["CTB"] - state["PIK"] * state["IKB"])


def ident_AB(state, _params):
    return state["AB"] - (
        state["AB_lag1"] + state["SB"] - state["MB"] + state["MB_lag1"]
        - (state["BR"] - state["BO"])
        + (state["BR_lag1"] - state["BO_lag1"])
        - state["DISB"]
    )


def ident_SR(state, _params):
    return state["SR"] - (
        -state["PEX"] * state["EX"] - state["USROW"]
        + state["PIM"] * state["IM"] + state["TFR"] + state["TRFR"]
        + state["TRHR"] + state["TRGR"] - state["TRRG2"]
        - state["CTR"] - state["NNR"] - state["TRRS"] - state["TTRRF"]
    )


def ident_AR(state, _params):
    return state["AR"] - (
        state["AR_lag1"] + state["SR"] - state["MR"] + state["MR_lag1"]
        + state["Q"] - state["Q_lag1"] - state["DISR"]
    )


def ident_SG(state, _params):
    return state["SG"] - (
        state["GSMA"] + state["GSCA"] + state["THG"] + state["IBTG"]
        + state["CUST"] + state["TBG"] + state["TFG"] + state["SIHG"]
        + state["SIFG"] - state["DG"] + state["TRFG"]
        - state["PG"] * state["COG"]
        - state["WG"] * state["JG"] * state["HG"]
        - state["WM"] * state["JM"] * state["HM"]
        - state["TRGH"] - state["UB"] - state["TRGR"] - state["TRGS"]
        - state["INTG"] - state["SUBG"]
        + state["CCG"] - state["INS"] - state["CTGMB"] - state["NNG"]
        - state["PIK"] * state["IKG"] + state["SIGG"]
        + state["CTGB"]
    )


def ident_AG(state, _params):
    return state["AG"] - (
        state["AG_lag1"] + state["SG"] - state["MG"] + state["MG_lag1"]
        + state["CUR"] - state["CUR_lag1"]
        + (state["BR"] - state["BO"])
        - (state["BR_lag1"] - state["BO_lag1"])
        - state["Q"] + state["Q_lag1"] - state["DISG"]
    )


def ident_SS(state, _params):
    return state["SS"] - (
        state["THS"] + state["IBTS"] + state["TBS"] + state["TFS"]
        + state["SIHS"] + state["SIFS"] - state["DS"] + state["TRGS"]
        + state["TRFS"]
        - state["PS"] * state["COS"]
        - state["WS"] * state["JS"] * state["HS"]
        - state["TRSH"] - state["INTS"] - state["SUBS"]
        + state["CCS"] - state["CTS"] - state["NNS"]
        + state["SISS"] + state["TRRS"]
    )


def ident_AS_stock(state, _params):
    """State saving (called AS to avoid Python keyword)."""
    return state["AS"] - (state["AS_lag1"] + state["SS"] - state["MS"]
                          + state["MS_lag1"] - state["DISS"])


def ident_TESTZERO(state, _params):
    """Sanity identity: all sector savings sum to zero."""
    return state["TESTZERO"] - (
        state["SH"] + state["SF"] + state["SB"] + state["SR"]
        + state["SG"] + state["SS"] + state["STAT"] + state["TRRG2"]
    )


def ident_M1(state, _params):
    return state["M1"] - (
        state["M1_lag1"]
        + state["MH"] - state["MH_lag1"]
        + state["MF"] - state["MF_lag1"]
        + state["MR"] - state["MR_lag1"]
        + state["MS"] - state["MS_lag1"]
        + state["MDIF"]
    )


# Rate & growth transforms ------------------------------------------------

def ident_RSA(state, _params):
    return state["RSA"] - state["RS"] * (1.0 - state["D1G"] - state["D1S"])


def ident_RMA(state, _params):
    return state["RMA"] - state["RM"] * (1.0 - state["D1G"] - state["D1S"])


def ident_SHRPIE(state, _params):
    return state["SHRPIE"] - (
        (1.0 - state["D2G"] - state["D2S"]) * state["PIEF"]
        / (state["WF"] * state["JF"] * (state["HN"] + 1.5 * state["HO"]))
    )


def ident_PCGDPR(state, _params):
    return state["PCGDPR"] - 100.0 * ((state["GDPR"] / state["GDPR_lag1"]) ** 4 - 1.0)


def ident_PCGDPD(state, _params):
    return state["PCGDPD"] - 100.0 * ((state["GDPD"] / state["GDPD_lag1"]) ** 4 - 1.0)


def ident_PCM1(state, _params):
    return state["PCM1"] - 100.0 * ((state["M1"] / state["M1_lag1"]) ** 4 - 1.0)


def ident_WA(state, _params):
    """Average wage for benefit calculations."""
    numer = ((1.0 - state["D1G"] - state["D1S"] - state["D4G"])
             * state["WF"] * state["JF"] * (state["HN"] + 1.5 * state["HO"])
             + (1.0 - state["D1G"] - state["D1S"]) * (
                 state["WG"] * state["JG"] * state["HG"]
                 + state["WM"] * state["JM"] * state["HM"]
                 + state["WS"] * state["JS"] * state["HS"]
                 - state["SIGG"] - state["SISS"]))
    denom = (state["JF"] * (state["HN"] + 1.5 * state["HO"])
             + state["JG"] * state["HG"]
             + state["JM"] * state["HM"]
             + state["JS"] * state["HS"])
    return state["WA"] - 100.0 * numer / denom


# ---------------------------------------------------------------------------
# Identity registry — core identities for the solve
# ---------------------------------------------------------------------------

IDENTITIES: list[tuple[str, Callable]] = [
    # ---- Structural-core identities (always needed for a solve) ----------
    # Goods side
    ("X",   ident_X),
    ("V",   ident_V),
    ("KD",  ident_KD),
    ("KH",  ident_KH),
    ("HN",  ident_HN),
    ("IVF", ident_IVF),
    ("IKF", ident_IKF),
    ("HFF", ident_HFF),
    ("KKMIN", ident_KKMIN),
    ("JHMIN", ident_JHMIN),
    # Prices
    ("PX",  ident_PX),
    ("PEX", ident_PEX),
    ("PD",  ident_PD),
    ("PH",  ident_PH),
    ("PCS", ident_PCS),
    ("PCN", ident_PCN),
    ("PCD", ident_PCD),
    ("PIH", ident_PIH),
    ("PIK", ident_PIK),
    ("PG",  ident_PG),
    ("PS",  ident_PS),
    ("PIV", ident_PIV),
    ("PKH", ident_PKH),
    ("PFA", ident_PFA),
    # Wages
    ("WG",  ident_WG),
    ("WM",  ident_WM),
    ("WS",  ident_WS),
    ("WH",  ident_WH),
    ("WA",  ident_WA),
    ("WR",  ident_WR),
    # Taxes & contributions
    ("THG", ident_THG),
    ("THS", ident_THS),
    ("TFG", ident_TFG),
    ("TFS", ident_TFS),
    ("TF1", ident_TF1),
    ("TCG", ident_TCG),
    ("TCS", ident_TCS),
    ("SIHG", ident_SIHG),
    ("SIFG", ident_SIFG),
    ("SIG",  ident_SIG),
    ("SIS",  ident_SIS),
    ("CDH",  ident_CDH),
    ("NICD", ident_NICD),
    # Labour aggregates
    ("E",   ident_E),
    ("U",   ident_U),
    ("UR",  ident_UR),
    ("POP", ident_POP),
    # Wealth / assets
    ("AA1", ident_AA1),
    ("AA2", ident_AA2),
    ("AA",  ident_AA),
    # Production transforms
    ("PROD", ident_PROD),
    # NIPA aggregates
    ("XX",   ident_XX),
    ("GDP",  ident_GDP),
    ("GDPR", ident_GDPR),
    ("GDPD", ident_GDPD),
    ("GNP",  ident_GNP),
    ("GNPR", ident_GNPR),
    ("GNPD", ident_GNPD),
    # Interest-rate transforms
    ("RSA",  ident_RSA),
    ("RMA",  ident_RMA),
    # ---- Accounting-shell identities (government & financial flows) ------
    # Income
    ("YT",   ident_YT),
    ("YD",   ident_YD),
    ("SRZ",  ident_SRZ),
    # Profits
    ("PIEF",    ident_PIEF),
    ("PIEFRET", ident_PIEFRET),
    ("CCF1",    ident_CCF1),
    ("SHRPIE",  ident_SHRPIE),
    # Sectoral saving & asset stocks
    ("SH",  ident_SH),  ("AH", ident_AH),
    ("SF",  ident_SF),  ("AF", ident_AF),
    ("SB",  ident_SB),  ("AB", ident_AB),
    ("SR",  ident_SR),  ("AR", ident_AR),
    ("SG",  ident_SG),  ("AG", ident_AG),
    ("SS",  ident_SS),  ("AS", ident_AS_stock),
    # Cross-sector
    ("MB",  ident_MB),
    ("M1",  ident_M1),
    ("USROW", ident_USROW),
    ("TESTZERO", ident_TESTZERO),
    # Government budget
    ("PUG",  ident_PUG),
    ("RECG", ident_RECG),
    ("EXPG", ident_EXPG),
    ("SGP",  ident_SGP),
    ("PUS",  ident_PUS),
    ("RECS", ident_RECS),
    ("EXPS", ident_EXPS),
    ("SSP",  ident_SSP),
    # Growth transforms
    ("PCGDPR", ident_PCGDPR),
    ("PCGDPD", ident_PCGDPD),
    ("PCM1",   ident_PCM1),
    # Financial / link
    ("INTGR", ident_INTGR),
]
"""All 80 identities from fminput.txt lines 212-347, transcribed to JAX."""


def list_identity_variables() -> list[str]:
    """Return the list of endogenous variables defined by identities."""
    return [name for name, _ in IDENTITIES]


# ---------------------------------------------------------------------------
# Stochastic-equation residual functions — auto-generated from us_model specs
# ---------------------------------------------------------------------------

# Each estimated equation produces a residual of the form:
#
#    residual = LHS_derived(raw_state) − α − β·regressor_1 − ... − ρ·u_lag1
#
# where LHS_derived is the per-equation transformation (e.g. log(CS/POP) for
# EQ 1). The solver drives residuals to zero to find the endogenous values.
#
# The LHS transforms (derived-variable formulas) match the GENR lines in
# us_model._GENR_SPECS. Endogenous variables live on the raw scale (CS, CN,
# Y, …) so identities that reference them directly just work.

# Map from EQ number to the transformation: given raw state + lag state,
# compute the variable that appears on the LHS of Fair's estimated equation.
_LHS_TRANSFORMS: dict[int, Callable[[dict], jnp.ndarray]] = {
    1:  lambda s: jnp.log(s["CS"]  / s["POP"]),                    # LCSZ
    2:  lambda s: jnp.log(s["CN"]  / s["POP"]),                    # LCNZ
    3:  lambda s: jnp.log(s["CD"]  / s["POP"]),                    # LCDZ
    4:  lambda s: jnp.log(s["IHH"] / s["POP"]),                    # LIHHZ
    5:  lambda s: jnp.log(s["L1"]  / s["POP1"]),                   # LL1Z
    6:  lambda s: jnp.log(s["L2"]  / s["POP2"]),                   # LL2Z
    7:  lambda s: jnp.log(s["L3"]  / s["POP3"]),                   # LL3Z
    8:  lambda s: jnp.log(s["LM"]  / s["POP"]),                    # LLMZ
    10: lambda s: jnp.log(s["PF"]),                                # LPF
    11: lambda s: jnp.log(s["Y"]),                                 # LY
    12: lambda s: jnp.log(s["KK"]) - jnp.log(s["KK_lag1"]),        # LKK1
    13: lambda s: jnp.log(s["JF"] / s["JF_lag1"]),                 # LJF1
    14: lambda s: jnp.log(s["HF"] / s["HF_lag1"]),                 # LHF1
    15: lambda s: jnp.log(s["HO"]),                                # LHO
    17: lambda s: jnp.log(s["MF"] / s["PF"]),                      # LMFZ
    18: lambda s: jnp.log(s["DF"] / s["DF_lag1"]),                 # LDF1
    23: lambda s: s["RB"] - s["RS_lag2"],                          # RBMRSL2
    24: lambda s: s["RM"] - s["RS_lag2"],                          # RMMRSL2
    26: lambda s: jnp.log(s["CUR"] / (s["POP"] * s["PF"])),        # LCURZ
    27: lambda s: jnp.log(s["IM"] / s["POP"]),                     # LIMZ
    28: lambda s: jnp.log(s["UB"]),                                # LUB
    29: lambda s: s["INTG"] / (-s["AG"]),                          # INTGZ
    30: lambda s: s["RS"],                                         # RS
    # EQ 16 LWFQZ depends on DELTA1 from EQ 10; handled via a post-hoc
    # closure in the simulate() entry point.
}


# Token-name mapping for regressors: converts "LCSZ(-1)" etc. into a getter
# from the state dict. For derived-variable regressors (like LYDZ, LCSZ_lag1),
# we compute them inline from raw state.
#
# This is the stateful bridge from Fair's variable conventions to solver state.

def _regressor_value(token: str, state: dict) -> jnp.ndarray:
    """Resolve a regressor token (e.g. ``"LCSZ(-1)"``, ``"CNST2CS"``) to its
    value in the current state.

    * ``"C"`` or a constant column: returns 1.0 or state[name].
    * Raw variables with no lag (e.g. ``"AG1"``, ``"RSA"``): return state[name].
    * Lagged variables (``"LCSZ(-1)"``, ``"LAAZ(-3)"``): return the derived
      value evaluated against the appropriate lag state.
    * Derived-variable tokens like ``"LYDZ"`` or ``"RSB"``: compute inline
      from the raw state using the same formula as in ``us_model._GENR_SPECS``.
    """
    # The full token parser from us_model — but implemented inline to avoid
    # circular imports and to handle derived variables.
    if token == "C":
        return jnp.asarray(1.0)

    # Detect lag suffix.
    if "(" in token:
        base, lag_part = token.split("(")
        lag = int(lag_part.rstrip(")"))  # negative for lag
        lag_suffix = f"_lag{-lag}" if lag < 0 else ""
    else:
        base, lag_suffix = token, ""

    # Derived variables — map to their definitions in terms of raw state.
    derived_map = {
        "LCSZ":    lambda s: jnp.log(s["CS"]  / s["POP"]),
        "LCNZ":    lambda s: jnp.log(s["CN"]  / s["POP"]),
        "LCDZ":    lambda s: jnp.log(s["CD"]  / s["POP"]),
        "LIHHZ":   lambda s: jnp.log(s["IHH"] / s["POP"]),
        "LL1Z":    lambda s: jnp.log(s["L1"]  / s["POP1"]),
        "LL2Z":    lambda s: jnp.log(s["L2"]  / s["POP2"]),
        "LL3Z":    lambda s: jnp.log(s["L3"]  / s["POP3"]),
        "LLMZ":    lambda s: jnp.log(s["LM"]  / s["POP"]),
        "LIMZ":    lambda s: jnp.log(s["IM"]  / s["POP"]),
        "LAAZ":    lambda s: jnp.log(s["AA"]  / s["POP"]),
        "LKHZ":    lambda s: jnp.log(s["KH"]  / s["POP"]),
        "LKDZ":    lambda s: jnp.log(s["KD"]  / s["POP"]),
        "LYDZ":    lambda s: jnp.log(s["YD"] / (s["POP"] * s["PH"])),
        "LYZ":     lambda s: jnp.log(s["Y"]  / s["POP"]),
        "LPOP":    lambda s: jnp.log(s["POP"]),
        "LXMFAZ":  lambda s: jnp.log((s["X"] - s["FA"]) / s["POP"]),
        "LXMFA":   lambda s: jnp.log(s["X"] - s["FA"]),
        "LPF":     lambda s: jnp.log(s["PF"]),
        "LPIM":    lambda s: jnp.log(s["PIM"]),
        "LPIMZ":   lambda s: jnp.log(s["PIM"] / s["PF"]),
        "LPFZPIM": lambda s: jnp.log(s["PF"]  / s["PIM"]),
        "LWF":     lambda s: jnp.log(s["WF"]),
        "LWFQ":    lambda s: jnp.log(s["WF"]) - jnp.log(s["LAM"]),
        "LWFZPF":  lambda s: jnp.log(s["WF"]) - jnp.log(s["PF"]),
        "LWFD5":   lambda s: jnp.log(s["WF"] * (1.0 + s["D5G"])) - jnp.log(s["LAM"]),
        "LMFZ":    lambda s: jnp.log(s["MF"]  / s["PF"]),
        "LCURZ":   lambda s: jnp.log(s["CUR"] / (s["POP"] * s["PF"])),
        "RSB":     lambda s: s["RS"] * (1.0 - s["D2G"] - s["D2S"]),
        "RSA":     lambda s: s["RSA"],
        "RMA":     lambda s: s["RMA"],
        "LEXKK":   lambda s: jnp.log(s["KK"] / s["KKMIN"]),
        "LEXL":    lambda s: jnp.log(s["JF"] / (s["JHMIN"] / s["HFS"])),
        "LY":      lambda s: jnp.log(s["Y"]),
        "LKK":     lambda s: jnp.log(s["KK"]),
        "LV":      lambda s: jnp.log(s["V"]),
        "LEX":     lambda s: jnp.log(s["EX"]),
        "LEXZ":    lambda s: jnp.log(s["EX"] / s["POP"]),
        "LCOGSZ":  lambda s: jnp.log((s["COG"] + s["COS"]) / s["POP"]),
        "LTRGSZ":  lambda s: jnp.log((s["TRGH"] + s["TRSH"]) / (s["POP"] * s["PH"])),
        "INTGZ":   lambda s: s["INTG"] / (-s["AG"]),
        "LU":      lambda s: jnp.log(
            (jnp.abs(s["U"] - 1.0) + (s["U"] - 1.0)) / 2 + 1.0),
        "LHO":     lambda s: jnp.log(s["HO"]),
        "LY1":     lambda s: jnp.log(s["Y"]) - jnp.log(s["Y_lag1"]),
        "LKK1":    lambda s: jnp.log(s["KK"]) - jnp.log(s["KK_lag1"]),
        "LHF1":    lambda s: jnp.log(s["HF"]) - jnp.log(s["HF_lag1"]),
        "LJF1":    lambda s: jnp.log(s["JF"]) - jnp.log(s["JF_lag1"]),
        "LDF1":    lambda s: jnp.log(s["DF"]) - jnp.log(s["DF_lag1"]),
        "LHFL1A":  lambda s: jnp.log(s["HF_lag1"] / s["HFS_lag1"]),
        "LUB":     lambda s: jnp.log(s["UB"]),
        "LMFL1Q":  lambda s: jnp.log(s["MF_lag1"] / s["PF"]),
        "LCURL1Q": lambda s: jnp.log(s["CUR_lag1"] / (s["POP_lag1"] * s["PF"])),
        "LWAZPH":  lambda s: jnp.log(s["WA"] / s["PH"]),
        "UR1":     lambda s: s["UR"] - s["UR_lag1"],
        "RBMRSL2": lambda s: s["RB"] - s["RS_lag2"],
        "RMMRSL2": lambda s: s["RM"] - s["RS_lag2"],
        "RSMRSL2": lambda s: s["RS"] - s["RS_lag2"],
        "RSLMRSL2": lambda s: s["RS_lag1"] - s["RS_lag2"],
        "RBLMRSL2": lambda s: s["RB_lag1"] - s["RS_lag2"],
        "RMLMRSL2": lambda s: s["RM_lag1"] - s["RS_lag2"],
        "RBA":     lambda s: s["RB"] * (1.0 - s["D2G"] - s["D2S"]),
        "RS1":     lambda s: s["RS"] - s["RS_lag1"],
        "ONEZUR":  lambda s: 1.0 / s["UR"],
        "GAP":     lambda s: (s["YS"] - s["Y"]) / s["YS"],
        "LCUSTZ":  lambda s: jnp.log(s["CUST"] / (s["PIM"] * s["IM"])),
        "PCPD":    lambda s: 100.0 * ((s["PD"] / s["PD_lag1"]) ** 4 - 1.0),
        "PCM1L1A": lambda s: s["D794823"] * s["PCM1_lag1"],
        "PCM1L1B": lambda s: s["D20083"] * s["PCM1_lag1"],
        "RQG":     lambda s: (0.4 * (s["RS"] / 400)
                              + 0.75 * 0.6 * (1.0 / 8) * (1.0 / 400) *
                              (s["RB"] + s["RB_lag1"] + s["RB_lag2"] + s["RB_lag3"]
                               + s["RB_lag4"] + s["RB_lag5"] + s["RB_lag6"] + s["RB_lag7"])),
    }

    # 1) Try direct lookup first — the frame may already contain a
    #    pre-computed lag column (e.g. "LKK1_lag1") that's exactly what we need.
    key = f"{base}{lag_suffix}" if lag_suffix else base
    if key in state:
        return state[key]

    # 2) Current-period derived variable: compute from current state.
    if not lag_suffix and base in derived_map:
        return derived_map[base](state)

    # 3) Lagged derived variable: compute from a lag-shifted state. The shift
    #    maps every "_lag(k+n)" key down to "_lag(k)" so the derived formula
    #    (which references the base name and its own "_lag1") can be applied.
    if lag_suffix and base in derived_map:
        import re as _re
        n = int(lag_suffix.removeprefix("_lag"))
        pattern = _re.compile(r"_lag(\d+)$")
        shifted = {}
        for state_key, val in state.items():
            m = pattern.search(state_key)
            if m:
                k = int(m.group(1))
                stem = state_key[: m.start()]
                if k == n:
                    shifted[stem] = val
                elif k > n:
                    shifted[f"{stem}_lag{k - n}"] = val
            # else: key with no lag suffix — represents current period; not
            # available after an n-step backshift.
        return derived_map[base](shifted)

    raise KeyError(f"No mapping for regressor token {token!r} (key {key!r})")


__all__ = ["IDENTITIES", "list_identity_variables",
           "_LHS_TRANSFORMS", "_regressor_value",
           "STOCHASTIC_RAW_ENDOGENOUS", "build_residual_function",
           "simulate_one_period", "simulate", "build_state_at_period",
           "SimulationResult", "parse_fmexog", "extend_frame_for_forecast"]


# ---------------------------------------------------------------------------
# Forecast-period exogenous extrapolation (fmexog.txt)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExogRule:
    """One exogenous-variable extrapolation spec from ``fmexog.txt``.

    Attributes:
      variable: Name of the exogenous variable (e.g. ``"COG"``, ``"POP1"``).
      rule: One of ``"CHGSAMEPCT"`` (grow by pct), ``"CHGSAMEABS"`` (add abs),
        ``"SAMEVALUE"`` (hold constant), or ``"EXPLICIT"`` (per-quarter list).
      values: For scalar rules a single float; for ``"EXPLICIT"`` one value
        per forecast quarter.
    """
    variable: str
    rule: str
    values: tuple[float, ...]


# Rules Fair's engine supports in CHANGEVAR blocks. Any variable with a bare
# numeric follow-up (no rule keyword) is treated as CHGSAMEPCT per Fair's
# default.
_EXOG_RULES = {"CHGSAMEPCT", "CHGSAMEABS", "SAMEVALUE"}


def parse_fmexog(path) -> list[ExogRule]:
    """Parse Fair's ``fmexog.txt`` into a list of ``ExogRule``.

    The file format (see pyfair/SOURCES.md or fminput.txt):

        SMPL 2026.1 2029.4;
        CHANGEVAR;
        VAR1 CHGSAMEPCT
        0.0025
        VAR2 SAMEVALUE
        0.0
        VAR3 ;
        v_2026Q1
        v_2026Q2
        ...
        RETURN;

    Args:
      path: Path to ``fmexog.txt``.

    Returns:
      List of ``ExogRule`` entries in source order.
    """
    from pathlib import Path

    text = Path(path).read_text()
    rules: list[ExogRule] = []

    current_var: str | None = None
    current_rule: str | None = None
    current_values: list[float] = []

    def flush_current():
        if current_var is None:
            return
        if current_rule == "EXPLICIT":
            rules.append(ExogRule(current_var, "EXPLICIT", tuple(current_values)))
        elif current_rule in _EXOG_RULES:
            rules.append(ExogRule(current_var, current_rule, tuple(current_values)))
        elif current_rule is None and current_values:
            # Default to CHGSAMEPCT when the rule keyword is omitted.
            rules.append(ExogRule(current_var, "CHGSAMEPCT", tuple(current_values)))

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("@"):
            continue
        upper = line.upper().rstrip(";").strip()

        if upper.startswith("SMPL") or upper == "CHANGEVAR" or upper == "RETURN":
            flush_current()
            current_var, current_rule, current_values = None, None, []
            continue

        # Pure numeric line → accumulate under the current variable.
        try:
            current_values.append(float(line))
            continue
        except ValueError:
            pass

        # Otherwise: a variable declaration, possibly with a rule keyword.
        flush_current()
        parts = line.rstrip(";").split()
        if not parts:
            current_var, current_rule, current_values = None, None, []
            continue
        current_var = parts[0]
        current_values = []
        if len(parts) >= 2 and parts[1].upper() in _EXOG_RULES:
            current_rule = parts[1].upper()
        else:
            # Explicit-list form: "VAR ;" followed by one value per quarter.
            current_rule = "EXPLICIT"

    flush_current()
    return rules


def _apply_exog_rule(
    last_value: float, rule: ExogRule, quarters_ahead: int,
) -> float:
    """Compute the extrapolated value at ``quarters_ahead`` periods into the future."""
    if rule.rule == "CHGSAMEPCT":
        pct = rule.values[0]
        return last_value * (1.0 + pct) ** quarters_ahead
    if rule.rule == "CHGSAMEABS":
        step = rule.values[0]
        return last_value + step * quarters_ahead
    if rule.rule == "SAMEVALUE":
        return rule.values[0]
    if rule.rule == "EXPLICIT":
        return rule.values[quarters_ahead - 1]
    raise ValueError(f"Unknown exog rule {rule.rule!r}")


def extend_frame_for_forecast(
    frame: "pl.DataFrame",
    fmexog_path=None,
    fmage_path=None,
    forecast_start: str = "2026Q1",
    forecast_end: str = "2029Q4",
) -> "pl.DataFrame":
    """Append forecast-period rows to a historical frame.

    Extrapolation sources:

    * ``fmexog.txt`` — explicit rules (CHGSAMEPCT/ABS/SAMEVALUE/list) for the
      variables Fair projects.
    * ``fmage.txt`` — age-share demographic variables AG1/AG2/AG3, which Fair
      projects separately.
    * Everything else — held at the last historical value (``2025Q4``).

    After the raw rows are filled, regime dummies (CNST2CS etc.), the time
    trend ``T``, GENR-derived variables, and the needed lag columns are all
    recomputed.

    Args:
      frame: Historical wide frame from ``us_model.build_frame``.
      fmexog_path: Defaults to ``config.US_FMEXOG``.
      fmage_path: Defaults to ``config.US_FMAGE``.
      forecast_start: First forecast quarter.
      forecast_end: Last forecast quarter.

    Returns:
      Wide frame with rows extending through ``forecast_end``.
    """
    import polars as pl
    from .. import config
    from ..core import readers
    from . import model as us_model

    fmexog_path = fmexog_path or config.US_FMEXOG
    fmage_path = fmage_path or config.US_FMAGE

    # Generate the forecast period labels (1952Q1-style).
    def _periods_between(start: str, end: str) -> list[str]:
        y, q = start.split("Q")
        y, q = int(y), int(q)
        ye, qe = end.split("Q")
        ye, qe = int(ye), int(qe)
        out = []
        while (y, q) <= (ye, qe):
            out.append(f"{y}Q{q}")
            q += 1
            if q > 4:
                q = 1
                y += 1
        return out

    forecast_periods = _periods_between(forecast_start, forecast_end)
    n_fc = len(forecast_periods)

    # Start from the raw historical frame: keep everything except columns we
    # are going to recompute below (GENR outputs, regime dummies, trend,
    # and explicit lag columns). The alternative WIP filter we tried (based
    # on column-name prefixes) was too blunt; this dict-based filter is
    # exact.
    computed_later_columns: set[str] = {name for name, _fn in us_model._GENR_SPECS}
    computed_later_columns |= {"T", "C", "CNST2CS", "CNST2L2", "CNST2KK", "TBL2"}

    keep_cols = [
        c for c in frame.columns
        if c == "period"
        or (c not in computed_later_columns and "_lag" not in c)
    ]
    raw_frame = frame.select(keep_cols)

    # Build forecast rows: start all columns at their 2025Q4 values, then
    # override via fmexog.txt / fmage.txt.
    last_hist_period = "2025Q4"
    last_row = raw_frame.filter(pl.col("period") == last_hist_period).to_dicts()[0]

    forecast_rows: list[dict] = []
    for fc_period in forecast_periods:
        row = dict(last_row)
        row["period"] = fc_period
        forecast_rows.append(row)

    # Apply fmexog rules.
    exog_rules = parse_fmexog(fmexog_path)
    for rule in exog_rules:
        if rule.variable not in last_row:
            continue   # Variable not in our frame (e.g. "ISZQ" not loaded)
        baseline = last_row[rule.variable]
        if baseline is None:
            continue
        baseline = float(baseline)
        for k in range(1, n_fc + 1):
            try:
                val = _apply_exog_rule(baseline, rule, k)
            except (IndexError, ValueError):
                continue
            forecast_rows[k - 1][rule.variable] = val

    # Load and apply fmage.txt for AG1/AG2/AG3 and other demographic series.
    fmage_long = readers.parse_fair_data(fmage_path)
    fmage_wide = readers.pivot_to_wide(fmage_long)
    for fc_period in forecast_periods:
        age_matches = fmage_wide.filter(pl.col("period") == fc_period)
        if age_matches.height == 0:
            continue
        age_row = age_matches.to_dicts()[0]
        for k, v in age_row.items():
            if k != "period" and isinstance(v, (int, float)):
                forecast_rows[forecast_periods.index(fc_period)][k] = v

    # Zero any COVID-era dummies in the forecast window (they end at 2021Q4).
    covid_vars = {f"D202{q}{s}" for q in "01" for s in "1234"}
    for row in forecast_rows:
        for cv in covid_vars:
            if cv in row:
                row[cv] = 0.0

    # Append forecast rows to the historical raw frame.
    fc_frame = pl.DataFrame(forecast_rows, schema=raw_frame.schema)
    extended = pl.concat([raw_frame, fc_frame]).sort("period")

    # Recompute everything downstream.
    extended = us_model.add_time_trend_and_constant(extended)
    extended = us_model.apply_genr(extended)
    extended = us_model.apply_regime_dummies(extended)
    extended = us_model.add_lags(extended, us_model._all_required_lags())
    return extended


# ---------------------------------------------------------------------------
# Multi-period dynamic simulation
# ---------------------------------------------------------------------------

def build_state_at_period(
    frame: "pl.DataFrame", period: str, max_lag: int = 8,
) -> dict[str, jnp.ndarray]:
    """Assemble a per-period state dict from the historical frame.

    The frame already has derived variables and some pre-computed lags from
    ``us_model.build_frame``. We augment with raw-variable lags by walking
    back through prior periods — the solver needs ``KK_lag1``, ``V_lag1``,
    ``RS_lag2``, etc. which aren't stored as columns.

    Args:
      frame: Output of ``us_model.build_frame``.
      period: The period to build state for (e.g. ``"2019Q4"``).
      max_lag: How many quarters back to gather raw-variable lag values.

    Returns:
      Dict mapping variable name (and ``name_lag1``, ``name_lag2``, …) to
      scalar jnp values.
    """
    import polars as pl

    periods = frame["period"].to_list()
    idx = periods.index(period)
    row = frame.filter(pl.col("period") == period).to_dicts()[0]
    state = {k: float(v) for k, v in row.items() if isinstance(v, (int, float))}
    for lag in range(1, max_lag + 1):
        if idx - lag < 0:
            continue
        lag_period = periods[idx - lag]
        lag_row = frame.filter(pl.col("period") == lag_period).to_dicts()[0]
        for k, v in lag_row.items():
            if isinstance(v, (int, float)):
                key = f"{k}_lag{lag}"
                if key not in state:
                    state[key] = float(v)
    return {k: jnp.asarray(v, dtype=jnp.float64) for k, v in state.items()}


def _compute_historical_ar_residuals(
    estimation_results: "list[us_model.EstimationResult]",
    state_lag1: dict[str, jnp.ndarray],
    params: dict[str, float],
) -> dict[str, jnp.ndarray]:
    """For each AR(1) equation, compute u_{t-1} = LHS − fitted at the given
    prior-period state. Returns ``{eqN_u_lag1: value}`` dict."""
    residuals: dict[str, jnp.ndarray] = {}
    for result in estimation_results:
        eq = result.equation
        if not eq.has_ar1 or eq.number == 16:
            continue
        try:
            lhs = _LHS_TRANSFORMS[eq.number](state_lag1)
            fitted = jnp.asarray(0.0)
            for i, token in enumerate(eq.regressors):
                coef = params[f"eq{eq.number}_{i}"]
                fitted = fitted + coef * _regressor_value(token, state_lag1)
            residuals[f"eq{eq.number}_u_lag1"] = lhs - fitted
        except Exception:
            # Missing data for this eq's regressors — carry zero residual.
            _LOG.debug(
                "AR innovation skipped for EQ %d (missing input)",
                eq.number, exc_info=True,
            )
            residuals[f"eq{eq.number}_u_lag1"] = jnp.asarray(0.0)
    return residuals


@dataclass
class SimulationResult:
    """Output of :func:`simulate`.

    Attributes:
      periods: List of simulated period strings (e.g. ``["2020Q1", "2020Q2", …]``).
      solved: Per-period dict ``{period: {var: value}}`` of solved endogenous values.
      iterations: Per-period Newton iteration count.
      residual_norms: Per-period ‖F(x*)‖.
    """
    periods: list[str]
    solved: dict[str, dict[str, float]]
    iterations: dict[str, int]
    residual_norms: dict[str, float]


def simulate(
    frame: "pl.DataFrame",
    estimation_results: "list[us_model.EstimationResult]",
    start_period: str,
    end_period: str,
    tol: float = 1e-8,
    max_newton_iter: int = 30,
) -> SimulationResult:
    """Solve the US model quarter by quarter over a forecast window.

    For each period:
      1. Assemble state: historical exogenous at t + solved lagged values from t-1.
      2. Update AR(1) residuals from the prior period's solve.
      3. Newton-solve the joint system of 24 stochastic + 95 identity residuals.
      4. Record solved values; propagate to next period.

    Args:
      frame: Output of ``us_model.build_frame()``.
      estimation_results: Output of ``us_model.estimate_all()``.
      start_period: First forecast quarter (inclusive).
      end_period: Last forecast quarter (inclusive).
      tol: Newton convergence tolerance.
      max_newton_iter: Newton iteration cap per period.

    Returns:
      ``SimulationResult`` with per-period solved values and diagnostics.
    """

    params = _flatten_params_for_solve(estimation_results)
    eqs_for_solve = [r.equation for r in estimation_results
                     if r.equation.number != 16]
    endog_order = endogenous_variable_order(eqs_for_solve)
    residual_fn = build_residual_function(eqs_for_solve)

    all_periods = frame["period"].to_list()
    start_idx = all_periods.index(start_period)
    end_idx = all_periods.index(end_period)

    solved_by_period: dict[str, dict[str, float]] = {}
    iter_by_period: dict[str, int] = {}
    rnorm_by_period: dict[str, float] = {}

    for idx in range(start_idx, end_idx + 1):
        period = all_periods[idx]
        lag_period = all_periods[idx - 1]

        # Assemble state: historical exogenous at t + solved endog at t-1.
        state = build_state_at_period(frame, period)
        # Override endogenous-at-t-1 with the previous period's solved values
        # (if we've simulated it). For the first simulated period, this is
        # already the historical frame value.
        if lag_period in solved_by_period:
            prev = solved_by_period[lag_period]
            for name, value in prev.items():
                state[f"{name}_lag1"] = jnp.asarray(value)

        # AR residuals from the prior period.
        state_lag1 = build_state_at_period(frame, lag_period)
        if lag_period in solved_by_period:
            for name, value in solved_by_period[lag_period].items():
                if name in state_lag1:
                    state_lag1[name] = jnp.asarray(value)
        ar_residuals = _compute_historical_ar_residuals(
            estimation_results, state_lag1, params,
        )
        state.update(ar_residuals)

        initial_guess = {name: state[name] for name in endog_order
                         if name in state}
        for name in endog_order:
            if name not in initial_guess:
                initial_guess[name] = jnp.asarray(1.0)

        solved, iters, rnorm = simulate_one_period(
            state_other=state,
            initial_guess=initial_guess,
            residual_fn=residual_fn,
            params=params,
            endogenous_order=endog_order,
            tol=tol,
            max_iter=max_newton_iter,
        )

        solved_by_period[period] = solved
        iter_by_period[period] = iters
        rnorm_by_period[period] = rnorm

    return SimulationResult(
        periods=all_periods[start_idx: end_idx + 1],
        solved=solved_by_period,
        iterations=iter_by_period,
        residual_norms=rnorm_by_period,
    )


# ---------------------------------------------------------------------------
# Endogenous variable list + residual function assembly
# ---------------------------------------------------------------------------

# Map Fair equation number → the raw-level variable it pins down.
# Residual = LHS_TRANSFORMS[eq_num](state) − fitted. Newton solves for the
# raw-level endogenous by driving the residual to zero.
STOCHASTIC_RAW_ENDOGENOUS: dict[int, str] = {
    1:  "CS",    2:  "CN",   3:  "CD",   4:  "IHH",
    5:  "L1",    6:  "L2",   7:  "L3",   8:  "LM",
    10: "PF",    11: "Y",    12: "KK",   13: "JF",
    14: "HF",    15: "HO",   16: "WF",   17: "MF",
    18: "DF",    23: "RB",   24: "RM",   26: "CUR",
    27: "IM",    28: "UB",   29: "INTG", 30: "RS",
}


def _stochastic_residual(
    equation: "us_model.UsEquation",
    params: dict[str, float],
    state: dict,
) -> jnp.ndarray:
    """Evaluate residual = LHS_derived − fitted for one stochastic equation."""
    if equation.number == 16:
        # EQ 16 is built on top of EQ 10's DELTA1 — handle specially at the
        # caller level so we have DELTA1 available.
        raise NotImplementedError("EQ 16 requires DELTA1 preprocessing")

    lhs_value = _LHS_TRANSFORMS[equation.number](state)

    fitted = 0.0
    for i, token in enumerate(equation.regressors):
        coef_key = f"eq{equation.number}_{i}"
        fitted = fitted + params[coef_key] * _regressor_value(token, state)

    if equation.has_ar1:
        u_key = f"eq{equation.number}_u_lag1"
        rho_key = f"eq{equation.number}_rho"
        fitted = fitted + params[rho_key] * state[u_key]

    return lhs_value - fitted


def _flatten_params_for_solve(
    estimation_results: "list[us_model.EstimationResult]",
) -> dict[str, float]:
    """Convert the estimate_all output into a flat {key: coef} dict used by
    the residual function.

    Keys are ``eq{N}_{i}`` for the i-th regressor of equation N, and
    ``eq{N}_rho`` for AR(1) equations. The flat layout makes the dict
    pytree-leaf-friendly for JAX.
    """
    flat = {}
    for result in estimation_results:
        eq = result.equation
        for i, token in enumerate(eq.regressors):
            flat[f"eq{eq.number}_{i}"] = result.coefficients[token]
        if eq.has_ar1:
            flat[f"eq{eq.number}_rho"] = result.coefficients["RHO(-1)"]
    return flat


def build_residual_function(
    stochastic_equations: "list[us_model.UsEquation]",
    identities: list[tuple[str, Callable]] = None,
) -> Callable:
    """Return a function that, given a full state dict, returns the residual
    vector (one entry per endogenous variable in a stable order).

    The order is: stochastic residuals first (by equation number), then
    identity residuals. Callers must match this ordering when packing the
    Newton x-vector.
    """
    if identities is None:
        identities = IDENTITIES

    # Stable ordering of stochastic equations (sorted by eq number, excluding
    # the EQ 16 wage equation which is handled via the wrapper below).
    stoch_eqs = sorted(
        (eq for eq in stochastic_equations if eq.number != 16),
        key=lambda e: e.number,
    )

    def residuals(state: dict, params: dict) -> jnp.ndarray:
        parts = []
        for eq in stoch_eqs:
            parts.append(_stochastic_residual(eq, params, state))
        for _name, ident_fn in identities:
            parts.append(ident_fn(state, params))
        return jnp.stack(parts)

    return residuals


def endogenous_variable_order(
    stochastic_equations: "list[us_model.UsEquation]",
    identities: list[tuple[str, Callable]] = None,
) -> list[str]:
    """Return the endogenous-variable names, in the same order as the
    residual function output. Use this to pack/unpack Newton's x-vector.
    """
    if identities is None:
        identities = IDENTITIES

    stoch_eqs = sorted(
        (eq for eq in stochastic_equations if eq.number != 16),
        key=lambda e: e.number,
    )
    names = [STOCHASTIC_RAW_ENDOGENOUS[eq.number] for eq in stoch_eqs]
    names.extend(name for name, _ in identities)
    return names


# ---------------------------------------------------------------------------
# Single-period Newton solve
# ---------------------------------------------------------------------------

def simulate_one_period(
    state_other: dict[str, jnp.ndarray],
    initial_guess: dict[str, jnp.ndarray],
    residual_fn: Callable,
    params: dict,
    endogenous_order: list[str],
    tol: float = 1e-8,
    max_iter: int = 100,
) -> tuple[dict[str, jnp.ndarray], int, float]:
    """Newton-Raphson on the joint stochastic+identity residual.

    Args:
      state_other: Dict of everything the residual function needs that is
        NOT endogenous (lags, exogenous variables, AR residuals).
      initial_guess: Dict of starting values for the endogenous variables
        (keys must match ``endogenous_order``).
      residual_fn: From :func:`build_residual_function`.
      params: Flat coefficient dict.
      endogenous_order: Variable-name order that matches ``residual_fn`` output.
      tol: Convergence threshold on ‖F(x)‖.
      max_iter: Safety cap.

    Returns:
      ``(solved_dict, iterations, residual_norm)``.
    """
    x0 = jnp.array([initial_guess[name] for name in endogenous_order],
                   dtype=jnp.float64)

    def F(x):
        state = dict(state_other)
        for name, value in zip(endogenous_order, x):
            state[name] = value
        return residual_fn(state, params)

    x = x0
    for iteration in range(max_iter):
        r = F(x)
        rnorm = float(jnp.linalg.norm(r))
        if rnorm < tol:
            break
        J = jax.jacfwd(F)(x)
        try:
            dx = jnp.linalg.solve(J, r)
        except Exception as exc:
            raise RuntimeError(
                f"Newton step failed at iter {iteration}"
            ) from exc
        x = x - dx

    solved = {name: float(x[i]) for i, name in enumerate(endogenous_order)}
    return solved, iteration + 1, rnorm
