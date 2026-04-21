"""Regression test for US Model EQ 1 (consumption of services).

Loads fmdata.txt, builds GENR-derived variables (per-capita logs, wealth
ratios), constructs Fair's CNST2CS regime ramp, attaches lag columns, runs
Fair's TSLS17+RHOA iterated-2SLS-AR(1), and compares to Fair's published
fmout.txt values.

Observed gap (v0.2 algorithm)::

    β coefficients: within ~0.022 of Fair
    ρ:              within ~0.025 of Fair

The residual gap from exact parity is likely the same pre-sample Z handling
that leaves ~5e-3 on the IS model; we haven't fully replicated it yet.
"""
from __future__ import annotations

from pyfair import us_cs


# Absolute-delta tolerances. Calibrated from a clean run at 1.2-1.3x the
# observed max delta — tight enough to catch real regressions (e.g. a GENR
# typo or dummy-construction error) while leaving headroom for numerical noise.
TOL_BETA = 0.03   # all structural β (observed max: 0.022)
TOL_RHO = 0.03    # AR coefficient     (observed:     0.024)


def test_us_cs_coefficients_in_fair_ballpark():
    params = us_cs.estimate()

    # Every coefficient present.
    assert set(params.keys()) == set(us_cs.REFERENCE_PARAMS.keys()), (
        f"param names mismatch: {set(params) ^ set(us_cs.REFERENCE_PARAMS)}"
    )

    # Every coefficient has the same sign as Fair's (sign flip = real bug).
    for name, fair_value in us_cs.REFERENCE_PARAMS.items():
        our_value = params[name]
        if abs(fair_value) < 1e-4:   # near-zero skip-check
            continue
        assert (fair_value > 0) == (our_value > 0), (
            f"{name}: sign mismatch (fair={fair_value}, ours={our_value})"
        )

    # Magnitude tolerance — β vs ρ get different bands.
    for name, fair_value in us_cs.REFERENCE_PARAMS.items():
        tol = TOL_RHO if name == "RHO" else TOL_BETA
        delta = abs(params[name] - fair_value)
        assert delta < tol, (
            f"{name}: |{params[name]:.6f} - {fair_value:.6f}| = "
            f"{delta:.4f} exceeds tol {tol}"
        )
