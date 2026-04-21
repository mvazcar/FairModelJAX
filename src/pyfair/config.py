"""Paths, sample ranges, and numerical defaults for the pipeline.

This module is the single source of truth for anything a step needs to know
about *where* files live and *what* numeric defaults apply. Equation structure
lives in ``equations.py``; algorithm-specific tolerances that only matter to
one estimator live with that estimator.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Repo root is the directory that contains ``src/`` and ``raw_source/``. That
# is ``.../FairModelJAX`` in the published layout, or ``.../pyfair`` during
# local development.
PYFAIR_ROOT = Path(__file__).resolve().parents[2]

# Fair's original artifacts. The published layout bundles them under
# ``raw_source/`` inside the repo; the legacy development layout keeps them
# at the parent directory alongside ``pyfair/``. We prefer the bundled
# location when present so a fresh ``git clone`` is self-contained.
_BUNDLED_RAW = PYFAIR_ROOT / "raw_source"
_LEGACY_RAW = PYFAIR_ROOT.parent  # parent of repo root
RAW_SOURCE_ROOT = _BUNDLED_RAW if _BUNDLED_RAW.exists() else _LEGACY_RAW

# Upstream Fair model assets (see raw_source/SOURCES.md or the project-level
# SOURCES.md in the legacy layout).
FP_EXE = RAW_SOURCE_ROOT / "02_executable" / "FP.EXE"

US_MODEL_DIR = RAW_SOURCE_ROOT / "03_us_model"
US_FMINPUT = US_MODEL_DIR / "fminput.txt"
US_FMDATA = US_MODEL_DIR / "fmdata.txt"
US_FMAGE = US_MODEL_DIR / "fmage.txt"
US_FMEXOG = US_MODEL_DIR / "fmexog.txt"
US_FMOUT = US_MODEL_DIR / "fmout.txt"

MC_MODEL_DIR = RAW_SOURCE_ROOT / "04_mc_model" / "mcj_extracted"
MC_MCINP = MC_MODEL_DIR / "MC.INP"
MC_OUT = MC_MODEL_DIR / "OUT"
MC_YAW = MC_MODEL_DIR / "YAW.DAT"
MC_YDATA = MC_MODEL_DIR / "YDATA.DAT"
MC_QUAR = MC_MODEL_DIR / "QUAR.DAT"
MC_SHR = MC_MODEL_DIR / "SHR.INP"

EXAMPLES_DIR = RAW_SOURCE_ROOT / "06_examples"
IS_INP = EXAMPLES_DIR / "IS.INP"
IS_OUT = EXAMPLES_DIR / "IS.OUT"

# Kept as an alias for downstream code that referenced the old name.
PROJECT_ROOT = RAW_SOURCE_ROOT

# Working directories inside the repo (gitignored).
OUTPUT_DIR = PYFAIR_ROOT / "output"   # parquet caches per pipeline step
TEMP_DIR = PYFAIR_ROOT / "temp"       # scratch space
RAW_DIR = PYFAIR_ROOT / "raw"         # generated IS.DAT (FRED-built)


# ---------------------------------------------------------------------------
# IS model data sources
# ---------------------------------------------------------------------------
#
# Two choices, see pyfair/IS_DATA_SOURCES.md for the rationale:
#
#   "fred" (default)
#       Live FRED vintage, rebuilt on demand via
#       build_is_dat_from_fred.py. Always current, will diverge from Fair's
#       2013 coefficients as the NIPA chain base moves.
#
#   "fair_2013"
#       The shipped 06_examples/IS.DAT, frozen 2013-11-11. The only dataset
#       that reproduces Fair's published IS.OUT coefficients.

IS_DAT_FRED = RAW_DIR / "IS.DAT"
IS_DAT_FAIR_2013 = EXAMPLES_DIR / "IS.DAT"
FAIR_REFERENCE_DATE = "2013-11-11"

# The default. Override by passing ``data_source="fair_2013"`` through the
# pipeline or the ``--data-source`` CLI flag.
IS_DAT = IS_DAT_FRED


def is_dat_path(data_source: str) -> Path:
    """Resolve the IS.DAT path for a given data-source label.

    Args:
      data_source: Either ``"fred"`` or ``"fair_2013"``.

    Returns:
      Absolute path to the corresponding IS.DAT file.

    Raises:
      ValueError: If ``data_source`` isn't one of the two known labels.
    """
    if data_source == "fred":
        return IS_DAT_FRED
    if data_source == "fair_2013":
        return IS_DAT_FAIR_2013
    raise ValueError(
        f"Unknown data_source {data_source!r}; expected 'fred' or 'fair_2013'"
    )


# ---------------------------------------------------------------------------
# Sample ranges (Fair period strings, quarterly)
# ---------------------------------------------------------------------------

# Full US-model sample window; individual equations narrow this in fminput.txt.
US_FIRSTPER = "1952Q1"
US_LASTPER = "2025Q4"

# Estimation start: 2 quarters of leading lags needed for the regressors.
US_EST_START = "1954Q1"

# Forecast window used by fmexog.txt / extrapolate commands.
US_FORECAST_START = "2026Q1"
US_FORECAST_END = "2029Q4"

IS_FIRSTPER = "1952Q1"
IS_LASTPER = "2013Q3"   # last quarter of the shipped IS.DAT
IS_EST_START = "1954Q1"


# ---------------------------------------------------------------------------
# Solver numeric defaults
# ---------------------------------------------------------------------------

# Newton-Raphson convergence threshold on ‖F(x)‖. 1e-10 is well below the
# coefficient precision we care about while still leaving headroom for
# float64 round-off in the Jacobian solve.
NEWTON_TOL = 1e-10
NEWTON_MAX_ITER = 50

# Gauss-Seidel is carried here only for reference / future comparisons; the
# production solve uses Newton. The MINITERS default matches
# ``SETUPSOLVE MINITERS=3`` in Fair's shipped input files.
GAUSS_SEIDEL_MIN_ITER = 3
GAUSS_SEIDEL_MAX_ITER = 100
GAUSS_SEIDEL_DAMP = 1.0
