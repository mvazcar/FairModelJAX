# FairModelJAX

Ray C. Fair's US and Multi-Country macroeconomic models, ported from
FORTRAN 77 + the Fair-Parke DSL to Python + JAX. Fair is great; the
goal here is to play with his models in a modern stack — autodiff,
JIT, vmap — instead of feeding `FP.EXE` a deck.

Upstream: <https://fairmodel.econ.yale.edu/>.

## Install

```bash
git clone https://github.com/mvazcar/FairModelJAX.git
cd FairModelJAX
pip install -e .[dev]
```

Python 3.13+. CPU JAX by default; for GPU, pick the right `jaxlib` from
[JAX's install matrix](https://github.com/google/jax#installation).

## Run

```bash
python run.py --model is                       # 2-equation tutorial
python run.py --model is --data-source fair_2013   # reproduces Fair (2013) coefficients
python run.py --model us                       # full US model
python run.py --model mc                       # 36-country MC model
python run.py --model mc --mc-country CA       # one country only
```

Every step caches to `output/` as parquet. `--resume N` skips to step N
by reading from cache; `--force` ignores cache.

## Update the data

```bash
# IS model (FRED, default source) — refresh from the FRED CSVs:
python build_is_dat_from_fred.py && python run.py --model is --force

# US model — drop a new FMFP.ZIP into raw_source/03_us_model/, then:
python run.py --model us --force

# MC model — drop a new mcj.zip extract into raw_source/04_mc_model/mcj_extracted/, then:
python run.py --model mc --force
```

Fair re-estimates the US model every ~90 days; MC ~yearly.
`raw_source/SOURCES.md` has the permanent download URLs. See
`IS_DATA_SOURCES.md` for why the FRED vintage differs from Fair (2013).

## Layout

```
src/pyfair/
  us/         model.py, solve.py            — 24 stochastic eqs + 95 identities + forecast
  mc/         model.py, solve.py, shr.py    — 36 countries + 1,686 trade-share eqs
  core/       estimate.py, solver.py, readers.py
  pipeline/   is_pipeline.py, mc_pipeline.py
  tools/      fpexe.py                      — FP.EXE shell-out for golden-file tests
raw_source/   Fair's originals, bundled
tests/        461 tests (+1 slow, gated by --runslow)
```

## Test

```bash
python -m pytest
python -m pytest --runslow   # adds the FP.EXE golden-file regeneration (~5 min)
```

## References

**The model** — every line of the original economics is Ray C. Fair's. The
book-length versions that document each iteration:

- Fair, *A Short-Run Forecasting Model of the United States Economy*, D.C. Heath, 1971.
- Fair, *A Model of Macroeconomic Activity*, Vol. I (1974) and Vol. II (1976), Ballinger.
- Fair, *Specification, Estimation, and Analysis of Macroeconometric Models*, Harvard UP, 1984. (Introduces the Fair-Parke Program — the FORTRAN engine this repo ports.)
- Fair, *Testing Macroeconometric Models*, Harvard UP, 1994.
- Fair, *Estimating How the Macroeconomy Works*, Harvard UP, 2004.
- Fair, *Macroeconomic Modeling: The Cowles Commission Approach*, M.I.T. Press, 2024.

Primary references bundled as PDF in `raw_source/07_docs/`.

**The methodology** — if you want the history of macroeconometric models and
why the Cowles tradition and DSGE diverged:

- Fernández-Villaverde, *Horizons of Understanding: A Review of Ray Fair's
  Estimating How the Macroeconomy Works*, *Journal of Economic Literature*,
  46(3), September 2008, 685–703.
  [doi:10.1257/jel.46.3.685](https://doi.org/10.1257/jel.46.3.685) ·
  [PDF](https://www.sas.upenn.edu/~jesusfv/Horizons_Understanding.pdf).

  Good story, short, reads the lineage from Tinbergen (1937) → Klein–Goldberger
  (1955) → the 1970s macroeconometric crisis (Nelson 1972; Lucas 1976) → Fair
  and DSGE as the two surviving branches.

## Development

Written with [Claude Code](https://claude.ai/code) (Claude Opus 4.7 Max).
Humans picked the agenda and architecture; the model wrote the Python.

## License

MIT for the code under `src/` and `tests/`. Fair's bundled originals under
`raw_source/` keep his copyright — see [LICENSE](LICENSE).
