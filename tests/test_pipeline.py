"""End-to-end pipeline regression test on the IS tutorial model.

The 2013-vintage IS.DAT is the only dataset where we *can* verify exact
agreement with Fair's published IS.OUT coefficients. Any drift beyond
``PARITY_TOL_FAIR_2013`` indicates an algorithm regression in step02 or
step03 — not a data-source issue. This test is our guardrail against such
regressions.

We do not have an analogous end-to-end test against FRED data: the answer
is not a fixed target there (FRED publishes revisions).
"""
from __future__ import annotations

from pyfair import pipeline
from pyfair.step04_validate import PARITY_TOL_FAIR_2013, REFERENCE_PARAMS


def test_is_model_against_fair_2013_snapshot():
    """Full pipeline on shipped IS.DAT reproduces Fair's 2013 coefficients."""
    result = pipeline.run(
        model="is",
        resume=1,
        force=True,
        data_source="fair_2013",
    )

    # Structural checks — every step produced the expected shape.
    assert result.data.height > 200
    assert set(result.params.keys()) == set(REFERENCE_PARAMS.keys())
    assert result.solution.height == result.data.height
    assert result.parity.height == len(REFERENCE_PARAMS)

    # Newton converged on every simulated period (239 post-1954Q1 quarters).
    last_239_converged = result.solution["newton_converged"].tail(239)
    assert bool(last_239_converged.all()), "Newton failed to converge somewhere"

    # Hard parity assertion. step04_validate.run() itself raises on failure
    # when data_source="fair_2013", so reaching this line already means we
    # passed; the explicit check here is belt + suspenders.
    assert result.parity["abs_err"].max() < PARITY_TOL_FAIR_2013
