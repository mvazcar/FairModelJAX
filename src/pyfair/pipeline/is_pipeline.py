"""Pipeline orchestrator.

The IS-model workflow is a four-step linear pipeline::

    step01_load       raw data file  ->  wide polars frame
    step02_estimate   wide frame     ->  coefficient dict
    step03_solve      frame + coefs  ->  simulated-vs-actual frame
    step04_validate   coefs          ->  drift report vs Fair (2013)

Every step caches its output to ``output/step0N_<kind>_<model>_<data_source>.parquet``,
so re-runs are cheap and ``--resume N`` skips steps 1..N-1 by reading from cache.
Mirrors the MCVL pipeline convention.

The ``data_source`` argument threads through all steps so that cached outputs
are keyed on it — ``fred`` and ``fair_2013`` runs don't collide.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from . import step01_load, step03_solve, step04_validate
from ..core import estimate as step02_estimate


@dataclass
class PipelineResult:
    """Everything the pipeline produces, in one object.

    Attributes:
      data: Wide frame of raw input series (step01 output).
      params: Estimated coefficient dict (step02 output).
      solution: Actual + simulated series with Newton diagnostics (step03 output).
      parity: Per-coefficient drift report vs Fair (2013) reference (step04 output).
    """
    data: pl.DataFrame
    params: dict[str, float]
    solution: pl.DataFrame
    parity: pl.DataFrame


def _step_is_active(step_number: int, resume_from: int) -> bool:
    """True if this step is actually running this invocation.

    Steps before ``resume_from`` are read from cache; they still return the
    right data (needed as inputs to later steps) but don't run their full
    compute and don't print their banner.
    """
    return step_number >= resume_from


def _announce(step_number: int, resume_from: int, description: str) -> None:
    """Print a banner for the step only if it's actually running."""
    if _step_is_active(step_number, resume_from):
        print(f"[step0{step_number}] {description}")


def _force_active(force: bool, step_number: int, resume_from: int) -> bool:
    """Return the effective ``force`` flag for one step.

    ``--force`` ignores the cache, but it only applies to steps we're actually
    running — earlier steps (being skipped via ``--resume``) still load from
    cache. This prevents a cache rebuild cascade we didn't ask for.
    """
    return force and _step_is_active(step_number, resume_from)


def run(
    model: str,
    resume: int = 1,
    force: bool = False,
    data_source: str = "fred",
) -> PipelineResult:
    """Run the four-step pipeline end-to-end.

    Args:
      model: Model name. ``"is"`` is the only one wired up in v0.1.
      resume: Start running from this step number. Earlier steps load from
        their parquet cache. Defaults to 1 (full pipeline).
      force: If True, ignore caches for the active steps and recompute.
      data_source: ``"fred"`` (live, default) or ``"fair_2013"`` (shipped
        2013 snapshot, the only dataset that reproduces Fair's published
        IS.OUT coefficients).

    Returns:
      ``PipelineResult`` bundling all four step outputs.
    """
    print(f"[pipeline] model={model}  data_source={data_source}  "
          f"resume={resume}  force={force}")

    _announce(1, resume, "load data")
    data = step01_load.run(
        model,
        data_source=data_source,
        force=_force_active(force, 1, resume),
    )

    _announce(2, resume, "estimate coefficients")
    params = step02_estimate.run(
        model, data,
        data_source=data_source,
        force=_force_active(force, 2, resume),
    )

    _announce(3, resume, "solve dynamically")
    solution = step03_solve.run(
        model, data, params,
        data_source=data_source,
        force=_force_active(force, 3, resume),
    )

    _announce(4, resume, "validate vs Fair (2013) reference")
    parity = step04_validate.run(
        model, solution,
        params=params,
        data_source=data_source,
        force=_force_active(force, 4, resume),
    )

    return PipelineResult(
        data=data, params=params, solution=solution, parity=parity
    )
