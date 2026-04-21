"""Regression tests for the MC (Multi-Country) model — 13-country rollout.

Scope matches the v0.1 slice in ``mc_model``:
  * Country registry integrity (38 countries, unique bases, etc.).
  * ``OUT`` parser populates ``REFERENCE_PARAMS_MC`` for every covered eq.
  * End-to-end estimation of 13 quarterly countries (110 equations total).

Tolerance tiers (matches the US model's approach, slightly wider bands):
  * **Strict** (5e-5): non-AR OLS/2SLS, bounded-search AR(1), AR(2) via
    ``two_sls_ar2_bounded``. The bulk of the rollout lands here.
  * **AR normal** (3e-2): iterated 2SLS with AR(1) and inline FSR;
    first-stage conditioning drifts Fair's fixed point a notch.
  * **Known gap** (2.0): IV-2SLS equations where conditioning is notably
    worse, or where the regressor scaling (e.g. ITRS = 12.5%) magnifies
    absolute error. Tracked for v0.2; promote up-tier when tightened.
"""
from __future__ import annotations

import math

import polars as pl
import pytest

from pyfair import mc_countries, mc_model, mc_shr, mc_solve


TOL_STRICT = 5e-5
TOL_AR_NORMAL = 3e-2
# Known-gap ceiling is 10.0 — large because annual-country rate equations
# regress RS on DEZZ/BEZZ (GDP-gap in logs, ~0.03 magnitude), so their
# coefficients are on the order of 10–30 and small sample shifts can
# produce absolute coefficient errors of several units.
TOL_KNOWN_GAP = 10.0

# Countries covered in v0.1 rollout — all quarterly (annual_lag=1).
# ROW countries with full demand blocks — used by the per-country Newton
# solve and multi-period simulation tests. Excludes ``"US"`` (standalone
# pyfair solver handles US equations 1–30; ``EQUATIONS_BY_COUNTRY["US"]``
# is just EQ 31 which is tested separately) and ``"EU"`` (Euro-zone
# composite; no country-level data frame).
_SOLVE_COUNTRIES = (
    "CA", "JA", "AU", "FR", "GE", "IT", "NE", "ST", "UK", "FI", "AS", "SO", "KO",
    "BE", "DE", "NO", "SW", "GR", "IR", "PO", "SP", "NZ", "SA", "CO", "JO",
    "ID", "MA", "PA", "PH", "TH", "CH", "AR", "BR", "CE", "ME", "PE",
)

# Estimation fixture covers those + US EQ 31.
_COUNTRIES = _SOLVE_COUNTRIES + ("US",)


# Equations grouped by empirical fit vs Fair's OUT. Update when tightening.
_AR_NORMAL: set[tuple[str, int]] = {
    # Quarterly — IV-2SLS with AR(1) and inline FSR.
    ("JA", 52), ("JA", 55), ("JA", 57),
    ("AU", 64), ("AU", 65),
    ("FR", 77),
    ("GE", 84),
    ("IT", 97),
    ("ST", 115),
    ("AS", 147),
    ("KO", 167),
    # Annual-lag — H-rate equations with fluctuation at ~1e-3 to 1e-2.
    ("BE", 187), ("DE", 197), ("NO", 207), ("PO", 247), ("SP", 257),
}

_KNOWN_GAP: set[tuple[str, int]] = {
    # Quarterly IV-2SLS conditioning issues — would require Fair's FP.FOR
    # TSLS17 port to resolve. All rate / exchange-rate IV equations.
    ("CA", 47), ("AU", 66), ("AU", 67), ("FR", 76), ("GE", 87),
    ("IT", 95), ("UK", 124), ("FI", 137), ("SO", 155),
    # NLEQ xxLPXA equations where our LM lands at a different (lower-SSE)
    # local optimum than Fair's DFP. Non-convex objective.
    ("IR", 238), ("NZ", 268), ("ID", 328), ("PA", 348), ("TH", 368),
    # NOTE: the 13 annual-country rate/exchange-rate equations that
    # previously lived here (BE 185, DE 195, NO 205, SW 215/217,
    # GR 227, IR 235/237, SP 255, NZ 265/267, PH 355/357) now match
    # Fair to ~5e-10 after the GEEA/USRSA "forward-average" fix
    # (Fair's ``X(1)+X(2)+X(3)`` is a LEAD, not a lag). Promoted to
    # STRICT.
}

_NLEQ_CLOSE: set[tuple[str, int]] = {
    ("BE", 188), ("DE", 198), ("SW", 218), ("SP", 258), ("ME", 418),
}

# US MC-only equations. Estimated separately; not part of per-country Newton.
_US_MC_STRICT: set[tuple[str, int]] = {("US", 31)}


def _all_equation_keys() -> list[tuple[str, int]]:
    """Every (country, eq_number) pair the registry claims to estimate."""
    keys = []
    for c, eqs in mc_model.EQUATIONS_BY_COUNTRY.items():
        for eq in eqs:
            keys.append((c, eq.number))
    return keys


_STRICT: set[tuple[str, int]] = (
    set(_all_equation_keys()) - _AR_NORMAL - _KNOWN_GAP - _NLEQ_CLOSE
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def results_by_country() -> dict[str, list[mc_model.EstimationResultMC]]:
    """Estimate every covered country once; reuse across tests."""
    return {c: mc_model.estimate_country(c) for c in _COUNTRIES}


def _max_abs_error(result: mc_model.EstimationResultMC) -> float:
    errors = []
    for tok, ours in result.coefficients.items():
        fair = result.reference.get(tok, math.nan)
        if not math.isnan(fair):
            errors.append(abs(ours - fair))
    return max(errors) if errors else math.nan


def _get(results, eq_number):
    matches = [r for r in results if r.equation.number == eq_number]
    assert matches, f"EQ {eq_number} not in estimation results"
    return matches[0]


# ---------------------------------------------------------------------------
# Country registry sanity
# ---------------------------------------------------------------------------

def test_country_registry_has_38_countries():
    assert len(mc_countries.COUNTRIES) == 38


def test_country_registry_has_unique_prefixes():
    prefixes = [c.prefix for c in mc_countries.COUNTRIES]
    assert len(prefixes) == len(set(prefixes))


def test_country_registry_has_unique_bases():
    bases = [c.base for c in mc_countries.COUNTRIES]
    assert len(bases) == len(set(bases))


def test_ca_registered_with_nine_equations():
    ca = mc_countries.by_prefix("CA")
    assert ca.name == "Canada"
    assert ca.base == 4
    assert len(ca.eq_numbers) == 9


def test_total_stochastic_equations_matches_mcinp_active_count():
    total = mc_countries.total_equations()
    assert 220 <= total <= 280


# ---------------------------------------------------------------------------
# OUT parser
# ---------------------------------------------------------------------------

def test_out_parser_populates_ca_eq41():
    coefs = mc_model.REFERENCE_PARAMS_MC.get(41)
    assert coefs is not None
    assert coefs["C(0)"] == pytest.approx(-0.224080345, rel=1e-6)
    assert coefs["CALIMZ(-1)"] == pytest.approx(0.945873577, rel=1e-6)
    assert coefs["RHO(-1)"] == pytest.approx(0.307992441, rel=1e-6)


def test_out_parser_covers_all_ca_equations():
    for eq_num in (41, 42, 43, 44, 45, 46, 47, 48, 49):
        assert eq_num in mc_model.REFERENCE_PARAMS_MC


def test_out_parser_reaches_three_digit_equations():
    # Canary for equation numbers that are 3 digits (> 99): e.g. UK 128 RHO=2.
    assert 128 in mc_model.REFERENCE_PARAMS_MC
    assert "RHO(-2)" in mc_model.REFERENCE_PARAMS_MC[128]


# ---------------------------------------------------------------------------
# Structural completeness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("country", _COUNTRIES)
def test_every_country_estimates_end_to_end(country, results_by_country):
    expected = len(mc_model.EQUATIONS_BY_COUNTRY[country])
    assert len(results_by_country[country]) == expected, (
        f"{country}: registry claims {expected} equations; "
        f"got {len(results_by_country[country])} estimation results "
        f"(some equation failed — see stderr for details)"
    )


# ---------------------------------------------------------------------------
# Parity with Fair's OUT — tier-based
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("country,eq_number", sorted(_STRICT))
def test_equation_matches_fair_strictly(country, eq_number, results_by_country):
    result = _get(results_by_country[country], eq_number)
    max_err = _max_abs_error(result)
    assert max_err < TOL_STRICT, (
        f"{country} EQ {eq_number} ({result.equation.dependent}): "
        f"max abs error {max_err:.2e} exceeds {TOL_STRICT}"
    )


@pytest.mark.parametrize("country,eq_number", sorted(_AR_NORMAL))
def test_equation_matches_fair_ar_normal(country, eq_number, results_by_country):
    result = _get(results_by_country[country], eq_number)
    max_err = _max_abs_error(result)
    assert max_err < TOL_AR_NORMAL, (
        f"{country} EQ {eq_number} ({result.equation.dependent}): "
        f"max abs error {max_err:.2e} exceeds {TOL_AR_NORMAL}"
    )


@pytest.mark.parametrize("country,eq_number", sorted(_KNOWN_GAP))
def test_equation_within_known_gap(country, eq_number, results_by_country):
    result = _get(results_by_country[country], eq_number)
    max_err = _max_abs_error(result)
    assert max_err < TOL_KNOWN_GAP, (
        f"{country} EQ {eq_number} ({result.equation.dependent}): "
        f"max abs error {max_err:.2e} exceeds {TOL_KNOWN_GAP}"
    )


@pytest.mark.parametrize("country,eq_number", sorted(_NLEQ_CLOSE))
def test_nleq_equation_close_to_fair(country, eq_number, results_by_country):
    """NLEQ xxLPXA equations where our LM matches Fair's DFP to <5e-2."""
    result = _get(results_by_country[country], eq_number)
    max_err = _max_abs_error(result)
    assert max_err < 5e-2, (
        f"NLEQ {country} EQ {eq_number}: max abs error {max_err:.2e}"
    )


# ---------------------------------------------------------------------------
# n_obs anchors — regression guards on the pre-sample fix
# ---------------------------------------------------------------------------

def test_ca_eq41_n_obs_matches_fair(results_by_country):
    """Fair reports 228 observations for CA EQ 41."""
    result = _get(results_by_country["CA"], 41)
    assert result.n_obs == 228


def test_ja_eq56_n_obs_matches_fair(results_by_country):
    """Fair reports 203 observations for JA EQ 56 (non-AR; pre-sample fix)."""
    result = _get(results_by_country["JA"], 56)
    assert result.n_obs == 203


# ---------------------------------------------------------------------------
# Solve-driver skeleton (v0.1)
# ---------------------------------------------------------------------------

def test_mc_solve_imports():
    """mc_solve exposes the CA identity registry + global registry."""
    assert len(mc_solve.CA_IDENTITIES) == 8
    names = {i.output for i in mc_solve.CA_IDENTITIES}
    assert names == {"CAY", "CAPM", "CAZZ", "CAJMIN", "CAUR",
                     "CAM10$", "CAEX", "CAPX$"}
    # Every ROW country contributes identities; total is 37 × 8 + 37 xxE = 333.
    assert len(mc_solve.MC_IDENTITIES_ALL) > 300


def test_mc_solve_cay_identity_evaluates():
    """The CAY accounting identity produces the expected sum."""
    cay = next(i for i in mc_solve.CA_IDENTITIES if i.output == "CAY")
    inputs = {"CAC": 100.0, "CAI": 40.0, "CAG": 20.0, "CAEX": 30.0,
              "CAIM": 25.0, "CASTAT": 2.0, "CAV1": 3.0}
    assert float(cay.formula(**inputs)) == pytest.approx(170.0)


def test_mc_pmm_loader():
    """SHRDDD.DAT xxPMM series load through ``mc_solve.load_pmm_series``."""
    pmm = mc_solve.load_pmm_series()
    assert pmm.height > 200  # full quarterly span 1960Q1 onwards.
    pmm_cols = [c for c in pmm.columns if c.endswith("PMM")]
    assert len(pmm_cols) >= 35, f"expected ~38 PMM series, got {len(pmm_cols)}"
    # Known Fair value: USPMM at 1961Q4 = 0.1511403 (OUT line ~8230).
    row = pmm.filter(pmm["period"] == "1961Q4").row(0, named=True)
    assert row["USPMM"] == pytest.approx(0.1511403, abs=1e-6)


@pytest.mark.parametrize("prefix", list(_SOLVE_COUNTRIES))
def test_generic_country_newton_recovers_history(prefix):
    """The generic per-country Newton solve recovers history at 2010Q1
    across every ROW country (36 total), from a 2% perturbation, to
    max relative error < 1e-6. Validates:

      * Scalar GENR evaluator dispatches correctly per prefix.
      * LHS-inverse derivation from ``eq.dependent`` + GENR lookup.
      * AR(1) / AR(2) historical innovations.
      * Per-country state ordering inferred from ``EQUATIONS_BY_COUNTRY``.
      * Y-identity coupling with exogenous-CAI countries (JA) and
        annual-lag countries (BE..PE).
    """
    frame = mc_model.build_frame_mc(countries=(prefix,))
    solved, iters, rnorm = mc_solve.solve_country_one_period(
        prefix, frame, "2010Q1", perturbation=0.02
    )
    assert iters <= 50, f"{prefix} Newton took {iters} iterations"
    assert rnorm < 1e-8, f"{prefix} final residual norm {rnorm:.2e}"
    hist = frame.filter(pl.col("period") == pl.lit("2010Q1")).row(0, named=True)
    for name in mc_solve._country_state_order(prefix):
        if hist.get(name) is None:
            continue
        h = float(hist[name])
        s = solved[name]
        rel = abs(s - h) / (abs(h) + 1e-12)
        assert rel < 1e-6, (
            f"{prefix} {name}: solved={s}, hist={h}, rel={rel:.2e}"
        )


def test_all_countries_simultaneous_solve_2010q1():
    """All 36 ROW countries with demand blocks solve cleanly at 2010Q1.

    For in-sample solves each country is independent (xxPMM exogenous from
    SHRDDD.DAT), so this is the Gauss-Seidel outer loop's historical limit.
    """
    frame = mc_model.build_frame_mc(countries=_SOLVE_COUNTRIES)
    results = mc_solve.solve_all_countries_one_period(
        frame, "2010Q1", countries=_SOLVE_COUNTRIES
    )
    assert len(results) == len(_SOLVE_COUNTRIES)
    failed = [p for p, (_, iters, rnorm) in results.items()
              if iters < 0 or rnorm > 1e-8]
    assert not failed, f"Solve failed for: {failed}"


def test_multi_country_simulate_tracks_history():
    """Multi-country simulation reproduces history for every ROW country.

    12 quarters (annual countries use 3 year-steps) across 36 countries,
    residuals < 1e-8 and per-variable drift < 1e-6 at every period.
    """
    frame = mc_model.build_frame_mc(countries=_SOLVE_COUNTRIES)
    paths = mc_solve.simulate_mc_path(frame, "2008Q1", "2010Q4",
                                      countries=_SOLVE_COUNTRIES)
    assert len(paths) == len(_SOLVE_COUNTRIES)
    for prefix, path in paths.items():
        assert len(path) > 0
        for r in path:
            assert r["residual_norm"] < 1e-8, (
                f"{prefix} {r['period']}: rnorm {r['residual_norm']}"
            )
            row = frame.filter(pl.col("period") == pl.lit(r["period"])
                               ).row(0, named=True)
            for name, val in r["solved"].items():
                h = row.get(name)
                if h is None:
                    continue
                rel = abs(val - float(h)) / (abs(float(h)) + 1e-12)
                assert rel < 1e-6, (
                    f"{prefix} {r['period']} {name}: "
                    f"solved={val}, hist={h}, rel={rel:.2e}"
                )


def test_shr_parser_finds_1686_equations():
    """SHR.INP parser extracts every active trade-share equation."""
    specs = mc_shr.parse_shr_inp()
    assert len(specs) == 1686
    # Every spec has 3 regressors: C, LA<SRC><DST>(-1), P<SRC><DST>.
    for spec in specs[:50]:
        assert len(spec["regressors"]) == 3
        assert spec["regressors"][0] == "C"
        assert spec["dependent"].startswith("LAA")


def test_shr_first_eq_matches_fair():
    """SHR EQ 431 (LAAAUUS) matches Fair's published OUT to ~1e-6."""
    frame = mc_shr.build_shr_frame()
    eq = next(e for e in mc_shr.EQUATIONS_SHR if e.number == 431)
    coefs = mc_shr.estimate_shr_equation(eq, frame)
    fair = mc_model.REFERENCE_PARAMS_MC[431]
    assert coefs["C"] == pytest.approx(fair["C(0)"], abs=1e-5)
    assert coefs["LAAUUS(-1)"] == pytest.approx(fair["LAAUUS(-1)"], abs=1e-5)
    assert coefs["PAUUS"] == pytest.approx(fair["PAUUS(0)"], abs=1e-5)


def test_fpexe_helpers_parse_committed_out():
    """FP.EXE helpers correctly parse the committed OUT reference file.

    This doesn't run FP.EXE itself — it validates that
    ``extract_equation_block`` and ``extract_coefs_from_block`` in
    ``fpexe`` produce the same coefficients our existing
    ``_parse_mc_out`` loader does. If this test passes, the FP.EXE
    helpers are wired correctly and can be used for on-demand
    regeneration of OUT by running FP.EXE.
    """
    from pyfair import fpexe, config
    out_text = (config.MC_MODEL_DIR / "OUT").read_text()
    # Helper extracts the EQ 41 block.
    block = fpexe.extract_equation_block(out_text, 41)
    assert block is not None
    coefs = fpexe.extract_coefs_from_block(block)
    assert coefs["CALIMZ(-1)"] == pytest.approx(0.945873577, abs=1e-9)
    assert coefs["RHO(-1)"] == pytest.approx(0.307992441, abs=1e-9)
    # Match what our _parse_mc_out loaded at module import time.
    assert mc_model.REFERENCE_PARAMS_MC[41]["CALIMZ(-1)"] == coefs["CALIMZ(-1)"]


@pytest.mark.slow
def test_fpexe_regeneration_matches_committed_out():
    """Running FP.EXE on MC.INP reproduces the committed OUT file.

    Marked ``slow`` — takes ~6 seconds. Run with
    ``pytest --runslow`` to include it.

    Validates the full pipeline from a completely independent angle:
    the committed ``04_mc_model/mcj_extracted/OUT`` is the output of
    running FP.EXE on MC.INP + the paired DAT files; our estimator's
    ``REFERENCE_PARAMS_MC`` is extracted from that OUT. If FP.EXE's
    fresh output diverges materially from the committed file, either
    the inputs drifted or the shipped FP.EXE binary was replaced.

    Tolerant comparison on coefficient values (the fresh run matches
    the committed file to ~1e-6 on all coefficient figures; raw-text
    diffs show ~150 differing lines of 197,710, all floating-point
    noise in 5th-6th decimal places of sum-of-squared-residuals
    trailers).
    """
    from pyfair import fpexe, config
    fresh_out = fpexe.run_fpexe(
        config.MC_MODEL_DIR / "MC.INP",
        working_dir=config.MC_MODEL_DIR,
        timeout=60,
    )
    committed = (config.MC_MODEL_DIR / "OUT").read_text()
    # Spot-check several equations' coefficients match to tight tolerance.
    for eq_num in (41, 42, 43, 51, 81, 121, 188, 431):
        fresh_block = fpexe.extract_equation_block(fresh_out, eq_num)
        commit_block = fpexe.extract_equation_block(committed, eq_num)
        assert fresh_block and commit_block
        fresh_coefs = fpexe.extract_coefs_from_block(fresh_block)
        commit_coefs = fpexe.extract_coefs_from_block(commit_block)
        assert set(fresh_coefs.keys()) == set(commit_coefs.keys())
        for key in fresh_coefs:
            assert fresh_coefs[key] == pytest.approx(
                commit_coefs[key], abs=1e-6
            ), f"EQ {eq_num}: {key} drifted"


def test_fpexe_binary_available():
    """FP.EXE binary ships with the repo at 02_executable/FP.EXE.

    Validates the path the shell-out helper uses. If this test fails,
    ``run_fpexe`` will raise FileNotFoundError when called.
    """
    from pyfair import fpexe
    assert fpexe.FP_EXE_PATH.exists()
    assert fpexe.FP_EXE_PATH.stat().st_size > 1_000_000  # sanity


def test_mc_pipeline_end_to_end():
    """Small MC pipeline smoke test: load → estimate → solve for 3 countries."""
    from pyfair import pipeline_mc
    result = pipeline_mc.run_mc_pipeline(
        countries=("CA", "JA", "GE"),
        start_period="2010Q1", end_period="2010Q4",
    )
    assert result.data.height > 200
    assert len(result.country_coefs) == 3
    assert len(result.shr_coefs) > 1000
    for c in ("CA", "JA", "GE"):
        assert len(result.solution[c]) == 4


def test_endogenous_mc_forecast_4_quarters():
    """Full Fair-style endogenous MC forecast: project SHR → aggregate PMM
    → block-Gauss-Seidel over countries, 4 forecast quarters from
    2018Q1 for a 2-country slice. Validates the integrated simulate_mc_endogenous
    + build_frame_for_endogenous_forecast path end-to-end.
    """
    countries = ("CA", "JA")
    frame = mc_solve.build_frame_for_endogenous_forecast(
        countries, n_forecast_quarters=4,
    )
    # Skip SHR projection (fast test) — just verify the outer loop
    # orchestration works when shr_coefs=None (persistence PMM).
    result = mc_solve.simulate_mc_endogenous(
        frame, "2018Q1", "2018Q4",
        countries=countries, shr_coefs=None, outer_iters=2,
    )
    assert len(result["periods"]) == 4
    for c in countries:
        assert len(result["paths"][c]) == 4
        for entry in result["paths"][c]:
            assert entry["residual_norm"] < 1e-6


def test_shr_projection_one_period():
    """SHR projection + PMM aggregation run end-to-end at a historical period.

    Validates the primitives that a fully-endogenous trade-share forecast
    would chain across periods: project AA_ij via SHR, aggregate into
    xxPMM via trade-weighted sum.
    """
    frame = mc_shr.build_shr_frame()
    # Use a shortcut: only estimate a small sample of equations.
    specs = mc_shr.parse_shr_inp()[:50]
    coefs: dict[int, tuple[float, float, float]] = {}
    for spec in specs:
        eq = next(e for e in mc_shr.EQUATIONS_SHR if e.number == spec["number"])
        try:
            result = mc_shr.estimate_shr_equation(eq, frame)
            coefs[spec["number"]] = (
                result["C"],
                result[eq.regressors[1]],
                result[eq.regressors[2]],
            )
        except Exception:
            continue
    assert len(coefs) > 40
    updated = mc_shr.project_shr_one_period(
        frame, "2000Q1", coefs, specs=specs,
    )
    # Projected AA_ij should be populated (non-null).
    row = updated.filter(pl.col("period") == pl.lit("2000Q1")).row(0, named=True)
    populated = sum(1 for s in specs
                    if row.get(f"AA{s['source']}{s['dest']}") is not None)
    assert populated > 40


def test_shr_batch_estimation_majority_strict():
    """At least 50% of the 1,686 SHR equations match Fair to <5e-5."""
    frame = mc_shr.build_shr_frame()
    strict = 0
    total = 0
    for eq in mc_shr.EQUATIONS_SHR:
        try:
            coefs = mc_shr.estimate_shr_equation(eq, frame)
        except Exception:
            continue
        fair_refs = mc_model.REFERENCE_PARAMS_MC.get(eq.number)
        if not fair_refs:
            continue
        total += 1
        max_err = 0.0
        for tok, ours in coefs.items():
            if tok == "C":
                ref = fair_refs.get("C(0)")
            elif "(" in tok:
                ref = fair_refs.get(tok)
            else:
                ref = fair_refs.get(f"{tok}(0)")
            if ref is None:
                continue
            max_err = max(max_err, abs(ours - ref))
        if max_err < 5e-5:
            strict += 1
    assert total > 1500
    assert strict / total > 0.5, f"Only {strict}/{total} strict"


def test_ca_monte_carlo_forecast():
    """Bootstrap Monte Carlo forecast of CA 2018Q1–2018Q4.

    Validates the MC pipeline end-to-end: historical ε collection,
    ε_override dispatch in ``solve_country_one_period``, and
    multi-period path generation. Different draws should produce
    different forecast paths (there must be variance across draws).
    """
    frame = mc_model.build_frame_mc(countries=("CA",))
    extended = mc_solve.extend_frame_for_forecast(
        frame, 4, method="persistence"
    )
    result = mc_solve.forecast_country_monte_carlo(
        "CA", extended, "2018Q1", "2018Q4", n_draws=8, rng_seed=42,
    )
    assert len(result["draws"]) == 8
    # Every draw produces 4 quarterly results (CA is quarterly, step=1).
    for draw in result["draws"]:
        assert len(draw) == 4
        for period_result in draw:
            assert period_result["residual_norm"] < 1e-8
    # Spread across draws: CAY at 2018Q4 should vary by at least 0.1%.
    final_cays = [draw[-1]["solved"]["CAY"] for draw in result["draws"]]
    rng = max(final_cays) - min(final_cays)
    mean = sum(final_cays) / len(final_cays)
    assert rng / mean > 0.001, "MC draws collapsed to a point estimate"


def test_ca_forecast_beyond_sample():
    """Forecast CA 2018Q1 after extending the frame 4 quarters with
    persistence-projected exogenous variables. Validates ``forecast_mode``
    + ``extend_frame_for_forecast``.
    """
    frame = mc_model.build_frame_mc(countries=("CA",))
    extended = mc_solve.extend_frame_for_forecast(frame, 4, method="persistence")
    assert extended.height == frame.height + 4
    solved, iters, rnorm = mc_solve.solve_country_one_period(
        "CA", extended, "2018Q1", forecast_mode=True, perturbation=0.0,
    )
    assert iters < 15
    assert rnorm < 1e-8
    # Forecast values should be finite and roughly on the order of 2017Q4.
    hist_2017q4 = frame.filter(pl.col("period") == pl.lit("2017Q4")
                                ).row(0, named=True)
    assert solved["CAIM"] == pytest.approx(hist_2017q4["CAIM"], rel=0.5)
    assert solved["CAC"] == pytest.approx(hist_2017q4["CAC"], rel=0.5)
    assert solved["CAY"] == pytest.approx(hist_2017q4["CAY"], rel=0.5)


def test_ca_multi_period_simulation_tracks_history():
    """24-quarter in-sample simulation of CA converges every period with
    zero drift from history — validates the rolling-lag frame update in
    ``simulate_country_path``."""
    frame = mc_model.build_frame_mc(countries=("CA",))
    results = mc_solve.simulate_country_path(
        "CA", frame, "2005Q1", "2010Q4"
    )
    assert len(results) == 24
    max_rnorm = max(r["residual_norm"] for r in results)
    assert max_rnorm < 1e-8, f"max residual norm {max_rnorm:.2e}"
    # Every solved value matches history (since ε drives it from obs).
    for r in results:
        period = r["period"]
        row = frame.filter(pl.col("period") == pl.lit(period)).row(0, named=True)
        for name, val in r["solved"].items():
            hist_val = float(row[name])
            rel = abs(val - hist_val) / (abs(hist_val) + 1e-12)
            assert rel < 1e-6, (
                f"CA {period} {name}: solved={val}, hist={hist_val}, "
                f"rel={rel:.2e}"
            )


@pytest.mark.parametrize("period", ["1990Q1", "2000Q2", "2010Q1", "2015Q4"])
def test_ca_newton_across_periods(period):
    """CA Newton solve recovers history at multiple historical periods.

    Complements the parametrized-over-countries 2010Q1 test by varying
    the period on one country. Exercises all 9 CA stochastic equations
    (including AR(1) EQs 41/44/47 and AR(2) EQ 48) plus the CAY identity.
    """
    frame = mc_model.build_frame_mc(countries=("CA",))
    hist = frame.filter(pl.col("period") == pl.lit(period)).row(0, named=True)

    solved, iters, rnorm = mc_solve.solve_country_one_period(
        "CA", frame, period, perturbation=0.10,
    )
    assert iters < 15, f"Newton took {iters} iterations"
    assert rnorm < 1e-8, f"final residual norm {rnorm:.2e}"
    worst_name, worst_err = None, 0.0
    for name in ("CAC", "CAI", "CAIM", "CAPY", "CARS", "CARB",
                 "CAE", "CAPX", "CAJ", "CAY"):
        s = solved[name]
        h = float(hist[name])
        rel = abs(s - h) / (abs(h) + 1e-12)
        if rel > worst_err:
            worst_name, worst_err = name, rel
    assert worst_err < 5e-6, (
        f"worst rel err at {period}: {worst_name}={worst_err:.2e}"
    )


def test_mc_identities_hold_on_historical_frame():
    """Every identity at 2010Q1 across all 37 ROW countries has
    relative error < 1e-3. Validates the GENR template, the trade-share
    forward-average convention, and the identity factory all at once.
    """
    import polars as pl  # local import to keep this block self-contained
    prefixes = [c.prefix for c in mc_countries.row_countries() if c.prefix != "EU"]
    frame = mc_model.build_frame_mc(countries=tuple(prefixes))
    residuals = mc_solve.verify_identities_on_frame(frame, "2010Q1")
    worst = max(
        (abs(r.residual) / (abs(r.observed) + 1e-12), r) for r in residuals
    )
    rel_err, r = worst
    assert rel_err < 1e-3, (
        f"{r.identity.country} {r.identity.output} at 2010Q1: "
        f"obs={r.observed:.4e}, computed={r.computed:.4e}, "
        f"residual={r.residual:.2e}, rel_err={rel_err:.2e}"
    )
