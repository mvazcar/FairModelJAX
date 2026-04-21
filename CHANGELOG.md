# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Subpackage layout.** Source now lives under focused subpackages:
  ``pyfair.us``, ``pyfair.mc``, ``pyfair.core``, ``pyfair.pipeline``,
  ``pyfair.tools``. Historic flat names (``pyfair.mc_model``,
  ``pyfair.us_solve``, …) still resolve via back-compat aliases in
  ``pyfair/__init__.py`` and ``sys.modules``.
- **``raw_source/``.** Fair's original FORTRAN, FP.EXE, US model (``03_us_model``),
  MC model (``04_mc_model``), IS examples (``06_examples``), and PDF documentation
  (``07_docs``) are now bundled inside the repo so a fresh ``git clone`` is
  self-contained. ``config.py`` prefers the bundled copy and falls back to
  the legacy adjacent layout for local development.

### Changed

- ``config.PROJECT_ROOT`` retained as an alias of the new
  ``config.RAW_SOURCE_ROOT`` for callers that imported it.
- ``pipeline`` top-level import now resolves to ``pyfair.pipeline.is_pipeline``
  (unchanged behaviour for end-users).

## [0.1.0] — in progress (0.1.0.dev0)

### US model

- Full JAX port of Fair's US model: 24 stochastic equations + 95 accounting
  identities + 4-year forecast window.
- Estimator: iterated 2SLS with AR(1) (``two_sls_ar1``) matching Fair's
  ``TSLS17 + RHOA`` loop; bounded golden-section variant for equations whose
  OLS Jacobian is singular in ρ.
- Solver: Newton–Raphson with ``jax.jacfwd`` Jacobian, 3–5 iterations per
  quarter on the in-sample window.
- 131 tests including FP.EXE golden-file parity on reference coefficients.

### MC model

- 36-country port: all stochastic equations (CA, JA, AU, …, PE) including the
  13 quarterly and 23 annual-lag country variants.
- Estimators:
    * AR(1) + AR(2) via iterated 2SLS (Nelder-Mead fallback for non-convex ρ).
    * NLEQ (``nlols_lpxa``) via SciPy Levenberg–Marquardt — lower SSE than
      Fair's DFP on 5 of 10 NLEQ equations.
- SHR (1,686 trade-share equations): batch estimator + PMM aggregation; 896 /
  1,686 match Fair within 5e-5. Parquet cache because ~155 s cold.
- Solver: per-country Newton, multi-country block Gauss-Seidel, forecast
  extension, bootstrap Monte Carlo.

### Tooling

- Pipelines: ``pipeline.is_pipeline`` (IS tutorial), ``pipeline.mc_pipeline``
  (MC end-to-end), with parquet caching at every step.
- CLI: ``python -m pyfair`` / ``run.py`` with ``--model {is,us,mc}``,
  ``--data-source {fred,fair_2013}``, ``--resume N``, ``--force``.
- ``tools.fpexe`` wraps Fair's FP.EXE for golden-file regression testing.

### Quality

- Google Python Style Guide pass across every MC + US file (logging →
  ``_LOG``, Callable type hints, named constants, trimmed dead code).
- 461 passing tests + 1 slow gated (``--runslow``).
