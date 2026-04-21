# FairModelJAX — a JAX port of Ray C. Fair's macroeconomic models

JAX + Python port of Ray C. Fair's US and Multi-Country (MCJ) macroeconomic
models, originally written in FORTRAN 77 + the Fair-Parke DSL.
Upstream reference: <https://fairmodel.econ.yale.edu/>.

**All of the underlying economics and every original line of code are the
work of Prof. Ray C. Fair**, who has been specifying, estimating, testing,
and re-estimating this system across more than five decades of a brilliant
career at Princeton, MIT, and Yale — from his 1971 D.C. Heath monograph to
the 2024 M.I.T. Press *Cowles Commission* volume. This repository is only a
translation of that work into JAX. See [Citing Fair](#citing-fair) for the
full set of book-length references that document each iteration of the
model.

The goal is a reproducible, modern, research-friendly base for macro modeling
at the scale Fair works at (~30 + ~230 stochastic equations) — with autodiff,
JIT, and vmap built in.

## What's here

| Layer | Module | What it does |
|---|---|---|
| **US model** | `pyfair.us.model`, `pyfair.us.solve` | 24 stochastic equations + 95 accounting identities + 4-year forecast window. |
| **MC model** | `pyfair.mc.model`, `pyfair.mc.solve`, `pyfair.mc.shr` | 36-country system (ROW + EU composite). 1,686 bilateral trade-share equations. Block Gauss-Seidel across countries. |
| **Shared core** | `pyfair.core.estimate`, `pyfair.core.solver`, `pyfair.core.readers` | Iterated 2SLS + AR(1)/AR(2), NLEQ via SciPy LM, Newton via `jax.jacfwd`, Fair-Parke file readers. |
| **Pipelines** | `pyfair.pipeline.is_pipeline`, `pyfair.pipeline.mc_pipeline` | Resumable step-wise orchestrators with parquet caching. |
| **Tools** | `pyfair.tools.fpexe` | Shell-out + parser for Fair's `FP.EXE` — golden-file regression harness. |
| **Raw source** | `raw_source/` | Fair's original FORTRAN (`FP.FOR`), Windows executable (`FP.EXE`), US + MC model specs, data, and `07_docs/` PDFs. Bundled so `git clone` is self-contained. |

## Why JAX

Fair's engine (`FP.FOR`, 209 subroutines) solves a simultaneous equation
system by Gauss–Seidel iteration. JAX lets us replace that with:

- **Newton–Raphson via `jax.jacfwd`** — autodiff gives the Jacobian for free;
  fewer iterations and better convergence on stiff periods (2008, COVID).
- **`jax.lax.scan` over quarters** — the forecast loop JIT-compiles to a
  single fused kernel.
- **`jax.vmap` for stochastic simulation** — thousands of Monte Carlo draws in
  parallel on CPU, trivial on GPU.
- **Gradient-based optimal control** — replaces Fair's DFP policy optimization
  with `jax.grad` / `optax`.

Each equation is a pure function `f(state, params) -> residual`. The solver
doesn't care which model it's solving.

## Install

```bash
git clone https://github.com/mvazcar/FairModelJAX.git
cd FairModelJAX
pip install -e .[dev]
```

Python 3.13+. JAX is CPU by default; for GPU, install the appropriate
`jaxlib` wheel per [JAX's install matrix](https://github.com/google/jax#installation).

## Quick start

```bash
# IS tutorial model (2 stochastic eqs + 1 identity) on live FRED data
python run.py --model is

# Reproduce Fair's 2013 published IS.OUT coefficients exactly
python run.py --model is --data-source fair_2013

# Full US model pipeline
python run.py --model us

# Full MC pipeline (36 countries, in-sample solve 2005Q1–2015Q4)
python run.py --model mc

# One MC country only (faster for development)
python run.py --model mc --mc-country CA
```

Every step caches to `output/` as parquet; re-running is cheap and
`--resume N` skips steps 1..N-1 by reading the cache.

## Data sources — important

The IS model has two data sources. See **[IS_DATA_SOURCES.md](IS_DATA_SOURCES.md)**
for the rationale.

| Label | What it is | When to use |
|---|---|---|
| `fred` (default) | Current FRED vintage, rebuilt by `build_is_dat_from_fred.py`. Chained 2017$, extends through the latest published quarter. | Normal use. |
| `fair_2013` | Fair's shipped 2013-11-11 `IS.DAT` (chained 2009$, last obs 2013Q3). | The only dataset that reproduces Fair's published `IS.OUT` coefficients. |

When `--data-source fair_2013` the pipeline enforces strict coefficient parity
(`PARITY_TOL_FAIR_2013 = 7e-3`). With `fred` it prints the drift but doesn't fail,
because NIPA rebasing since 2013 means divergence is expected.

## Updating the data

Three things can go out of date: the IS-tutorial data (pulled live from FRED),
Fair's US-model data file (published from his website), and Fair's MC-model
zip. All three refreshes are explicit commands — nothing happens implicitly.

### 1. IS model (FRED, the default `fred` source)

```bash
# Rebuild raw/IS.DAT from the latest FRED CSVs (no API key required).
# Writes raw/IS.DAT, overwriting any previous copy.
python build_is_dat_from_fred.py

# Then re-run the pipeline with --force so it ignores cached parquet:
python run.py --model is --force
```

The builder resolves five FRED series (`PCECC96`, `GPDIC1`, `GCEC1`,
`GDPC1`, `FEDFUNDS`) and writes them out in Fair's DSL format. Series
mapping and rationale are in
[IS_DATA_SOURCES.md](IS_DATA_SOURCES.md). Run it any time you want more
recent quarters — FRED typically lags the current quarter by 1–2 months.

### 2. US model (Fair's `03_us_model/`)

Fair re-estimates the US model every ~90 days and posts a fresh workbook.
To pick up a new vintage:

```bash
# 1. Download the latest archive from Fair's site:
#    https://fairmodel.econ.yale.edu/fp/FMFP.ZIP   (US model files)
#    Unzip into raw_source/03_us_model/, overwriting fminput.txt /
#    fmdata.txt / fmage.txt / fmexog.txt / fmout.txt.
#
# 2. Re-run. --force on step 1 forces a fresh read of fmdata.txt:
python run.py --model us --force
```

See `raw_source/SOURCES.md` for the permanent URLs. Fair's current
"Solve Previous Version" page (<https://fairmodel.econ.yale.edu/main3bne.htm>)
also maintains ~80 historical vintages if you want to pin the model to a
specific quarter.

### 3. MC model (Fair's `04_mc_model/`)

MC updates are less frequent (~yearly). Same pattern:

```bash
# Download https://fairmodel.econ.yale.edu/mcj/down/mcj.zip
# Unzip into raw_source/04_mc_model/mcj_extracted/, replacing MC.INP,
# YAW.DAT, YDATA.DAT, QUAR.DAT, SHR.INP, SHRDDD.DAT, OUT.

python run.py --model mc --force
```

The SHR coefficients are expensive (~155 s cold) and cached separately
in `output/step02_shr_coefs.parquet`. `--force` rebuilds them. For just
a subset of countries, add `--mc-country CA` (or any single MC prefix).

### 4. Adding a new FRED series (IS-tutorial extension)

Edit `build_is_dat_from_fred.py` — see the `_REAL_AGGREGATE_SERIES` and
`_INTEREST_RATE_SERIES` constants — then re-run the builder. No other
code changes needed for variables that already appear in `IS.INP`.

## Layout

```
FairModelJAX/
├── LICENSE                  MIT for the Python code
├── CHANGELOG.md
├── README.md
├── IS_DATA_SOURCES.md       rationale for fred vs fair_2013
├── pyproject.toml
├── run.py                   CLI entry point (thin wrapper for pyfair.__main__)
├── conftest.py              pytest: --runslow flag + sys.path shim
├── build_is_dat_from_fred.py    FRED → IS.DAT builder
├── raw_source/              Fair's originals (see raw_source/SOURCES.md)
│   ├── 01_fortran_source/       FP.FOR (1.02 MB, 209 subroutines)
│   ├── 02_executable/           FP.EXE (Windows binary)
│   ├── 03_us_model/             current US Model (PAB → Feb 20 2026)
│   ├── 04_mc_model/             current MC model (MCJ)
│   ├── 06_examples/             tutorial IS model from FP User's Guide
│   └── 07_docs/                 Fair's papers + mm2018.pdf, etc.
├── src/pyfair/
│   ├── config.py            paths, sample ranges, numerical defaults
│   ├── __main__.py          CLI parser
│   ├── core/                shared: estimate.py, solver.py, readers.py, equations.py
│   ├── us/                  US model: model.py, solve.py, cs.py
│   ├── mc/                  MC model: model.py, solve.py, shr.py, countries.py, plot.py
│   ├── pipeline/            is_pipeline.py, mc_pipeline.py, step01/03/04
│   └── tools/               fpexe.py (FP.EXE shell-out)
├── tests/                   461 tests (+1 slow, gated by --runslow)
├── output/                  parquet caches (gitignored)
├── temp/                    scratch (gitignored)
└── raw/                     FRED-built IS.DAT (gitignored)
```

## Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/ --runslow   # includes full FP.EXE regeneration (~5 min)
```

- `test_us_model.py`, `test_us_solve.py`, `test_us_identities.py`, `test_us_simulate.py`, `test_us_cs.py` — US model coverage (coefficients, identities, simulation).
- `test_mc_model.py` — MC equations, SHR trade shares, per-country solve.
- `test_pipeline.py` — IS pipeline end-to-end + Fair-2013 parity.
- `test_solver.py` — Newton on toy linear / nonlinear systems.

## Results

### US model

All 24 stochastic equations estimated to within published Fair tolerances;
131 tests pass (including FP.EXE golden-file parity on key coefficients).
Full 4-year forecast window compiles to a single fused `lax.scan` kernel.

### MC model

- 36 countries × up to 12 equations each, all solving via per-country Newton.
- SHR: 896 / 1,686 bilateral equations match Fair within 5e-5 on estimated
  coefficients. PMM aggregation error < 2% after fixing the source-set bug.
- Full multi-country Gauss-Seidel available (`simulate_mc_endogenous`) in
  addition to the faster independent per-country path.

### IS tutorial (reproduction of Fair, 2013)

```
$ python run.py --model is --data-source fair_2013
[step01] load data
[step02] estimate coefficients
[step03] solve dynamically
[step04] validate vs Fair (2013) reference
[step04] PASS  (all within 7e-3 of Fair 2013)
```

## Citing Fair

Each book below represents a milestone version of the model — Fair has
updated the specification, the estimation methodology, and the underlying
data across his career. If you use this port, please cite Fair directly;
this repo contributes nothing to the economics.

Chronologically, the model's published iterations are:

- Ray C. Fair, *A Short-Run Forecasting Model of the United States Economy*,
  D.C. Heath, **1971**. (Ph.D. work at MIT; the original single-equation
  quarterly forecasting system that the later model grew out of.)
- Ray C. Fair, *A Model of Macroeconomic Activity, Volume I: The Theoretical
  Model*, Ballinger, **1974**.
- Ray C. Fair, *A Model of Macroeconomic Activity, Volume II: The Empirical
  Model*, Ballinger, **1976**. (First full US macroeconometric model with
  simultaneous-equation estimation.)
- Ray C. Fair, *Specification, Estimation, and Analysis of Macroeconometric
  Models*, Harvard University Press, **1984**. (Introduces the Fair-Parke
  Program — the FORTRAN engine that this repo ports — and the
  tutorial IS model shipped in `raw_source/06_examples/`.)
- Ray C. Fair, *Testing Macroeconometric Models*, Harvard University Press,
  **1994**. (Bundled as `raw_source/07_docs/TestingMacroeconometricModels_1994.pdf`.)
- Ray C. Fair, *Estimating How the Macroeconomy Works*, Harvard University
  Press, **2004**. (Companion papers also on the website.)
- Ray C. Fair, *Macroeconomic Modeling: The Cowles Commission Approach*,
  M.I.T. Press, **2024** (draft in `raw_source/07_docs/Cowles_mmtext4.pdf`).
  The most recent book-length treatment, situating the model within the
  Cowles tradition.

For the Multi-Country model specifically, the authoritative reference is
the current MCJ Workbook (`raw_source/07_docs/MCJ_Workbook_current.pdf`),
with the rest-of-world chapters in `MCJ_AppendixB.pdf`. Forecast
performance records are at <https://fairmodel.econ.yale.edu/record/record.pdf>.

## Further reading — Fair, Cowles, and DSGE

For context on *where* Fair's approach sits in the macroeconomics
methodology landscape — and why the Cowles Commission "simultaneous
equations" tradition diverged from the Lucas-critique / DSGE research
program in the 1970s — the canonical review is:

- Jesús Fernández-Villaverde, *Horizons of Understanding: A Review of Ray
  Fair's Estimating How the Macroeconomy Works*, **Journal of Economic
  Literature**, Vol. 46, No. 3 (September 2008), pp. 685–703.
  [doi:10.1257/jel.46.3.685](https://doi.org/10.1257/jel.46.3.685) —
  PDF at <https://www.sas.upenn.edu/~jesusfv/Horizons_Understanding.pdf>.

Fernández-Villaverde places Fair's model in the historical arc from
Tinbergen (1937) and Klein–Goldberger (1955) through the 1970s
macroeconometric crisis (Nelson 1972; Lucas 1976) and the rise of DSGE,
and discusses the intellectual trade-offs between the two approaches.
Strongly recommended before reading any of Fair's books — it is the
clearest short statement of what the Cowles Commission approach is
actually trying to accomplish.

## Development notes

Entirely written with [Claude Code](https://claude.ai/code) using Anthropic's
**Claude Opus 4.7 Max** across two sessions — effectively two working days,
**April 20 and April 21, 2026** (there was a small kickoff on April 18
worth ~5% of the output). No human wrote a line of Python — every file
under `src/pyfair/` and `tests/` was produced by the model following
high-level direction (port Fair's US model; then port MC; then estimate
SHR; then reorganize for GitHub). The human contribution was the research
agenda, architectural choices (JAX-first, subpackage layout, raw_source
bundling), and review of the numeric results against Fair's FP.EXE golden
outputs.

**Resource usage**, counted from Claude Code's session transcripts:

| Metric | Session 1 | Session 2 | Total |
|---|---:|---:|---:|
| Assistant messages | 693 | 1,301 | **1,994** |
| Output tokens (model-written) | 3.04 M | 1.54 M | **4.57 M** |
| Distinct input context (live + cache writes) | 8.95 M | 8.73 M | **17.68 M** |
| Cache re-reads | 252.2 M | 712.0 M | **964.2 M** |
| **Total tokens processed** | **264.2 M** | **722.3 M** | **~986 M** |

The "total tokens processed" figure sounds huge but the cache-read component
is the conversation being re-ingested on every tool call — cheap per token
because of prompt caching, but it compounds fast once there are a couple of
thousand tool calls. The more honest "how much material did the model
write/read" number is the 4.57 M output + 17.68 M distinct input ≈ 22.3 M
tokens of real content.

## License

MIT for the Python code under `src/pyfair/` and `tests/`.
Fair's bundled originals under `raw_source/` retain his copyright; see the
note at the bottom of the [LICENSE](LICENSE) file.
