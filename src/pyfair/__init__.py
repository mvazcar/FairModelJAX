"""pyfair — JAX port of Ray C. Fair's US and Multi-Country macroeconomic models.

The package is organized into focused subpackages:

* ``pyfair.us``       — US-model (24 equations + 95 identities)
* ``pyfair.mc``       — Multi-Country model (36 countries, SHR trade shares)
* ``pyfair.core``     — Shared primitives: Newton solver, readers, estimators
* ``pyfair.pipeline`` — Orchestrators for IS / US / MC workflows
* ``pyfair.tools``    — Helper tools (FP.EXE shell-out, etc.)

For convenience and backward compatibility, the most-used submodules are also
exposed at the top level under their historical names (``pyfair.mc_model``,
``pyfair.us_model``, etc.). New code should prefer the canonical subpackage
paths: ``from pyfair.mc import model as mc_model``.
"""

# Fair's model needs float64 precision; enable before any jnp array is created.
import jax

jax.config.update("jax_enable_x64", True)

__version__ = "0.1.0.dev0"


# ---------------------------------------------------------------------------
# Back-compat top-level reexports.
#
# Prior to the 0.1.0 reorganization, every module lived at the top level
# (``pyfair.mc_model``, ``pyfair.us_solve``, etc.). We keep those names
# alive by importing the relocated module and binding it under its historic
# attribute name here. We also register the same modules in ``sys.modules``
# under their old dotted names so ``from pyfair.solver import X`` continues
# to resolve.
# ---------------------------------------------------------------------------
import sys as _sys

from . import config  # always at top level

# Canonical subpackage imports (fresh names — non-clashing).
from .core import equations as _equations
from .core import estimate as _estimate
from .core import readers as _readers
from .core import solver as _solver
from .mc import countries as _mc_countries
from .mc import model as _mc_model
from .mc import plot as _mc_plot
from .mc import shr as _mc_shr
from .mc import solve as _mc_solve
from .pipeline import is_pipeline as _is_pipeline
from .pipeline import mc_pipeline as _mc_pipeline
from .pipeline import step01_load as _step01_load
from .pipeline import step03_solve as _step03_solve
from .pipeline import step04_validate as _step04_validate
from .tools import fpexe as _fpexe
from .us import cs as _us_cs
from .us import model as _us_model
from .us import solve as _us_solve


# ``historic_name -> relocated module``. Everything below the underscore is
# one-step away from the historic flat layout.
_BACK_COMPAT = {
    "equations":        _equations,
    "estimate":         _estimate,
    "fpexe":            _fpexe,
    "mc_countries":     _mc_countries,
    "mc_model":         _mc_model,
    "mc_plot":          _mc_plot,
    "mc_shr":           _mc_shr,
    "mc_solve":         _mc_solve,
    # NOTE: ``pyfair.pipeline`` stays bound to the real subpackage to avoid
    # shadowing. Historic ``from pyfair import pipeline`` still works because
    # ``is_pipeline.run`` is reexported on the subpackage itself (see below).
    "pipeline_mc":      _mc_pipeline,
    "readers":          _readers,
    "solver":           _solver,
    "step01_load":      _step01_load,
    "step02_estimate":  _estimate,  # old name for the estimator module
    "step03_solve":     _step03_solve,
    "step04_validate":  _step04_validate,
    "us_cs":            _us_cs,
    "us_model":         _us_model,
    "us_solve":         _us_solve,
}
for _alias, _mod in _BACK_COMPAT.items():
    globals()[_alias] = _mod
    _sys.modules[f"{__name__}.{_alias}"] = _mod
del _alias, _mod


# ``from pyfair import pipeline`` used to yield the IS pipeline driver. Expose
# its public surface on the ``pipeline`` subpackage so the import still works.
from .pipeline import is_pipeline as pipeline  # noqa: E402  (must come after sys.modules wiring)


__all__ = [
    "__version__",
    "config", "pipeline",
    "equations", "estimate", "fpexe", "mc_countries", "mc_model", "mc_plot",
    "mc_shr", "mc_solve", "pipeline_mc", "readers", "solver",
    "step01_load", "step02_estimate", "step03_solve", "step04_validate",
    "us_cs", "us_model", "us_solve",
]
