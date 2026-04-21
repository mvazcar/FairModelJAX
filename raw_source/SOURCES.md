# `raw_source/` — Fair's originals

Everything under this directory is produced and distributed by
Prof. Ray C. Fair (Yale / Cowles Foundation) at
<https://fairmodel.econ.yale.edu/>. The files are bundled here unmodified
so this repo is self-contained for a fresh `git clone`.

Upstream cite: Ray C. Fair, Yale Economics / Cowles Foundation. Primary
references:

- *Macroeconometric Modeling: 2018*, `07_docs/mm2018.pdf` (437 pp).
- *Macroeconomic Modeling: The Cowles Commission Approach*, M.I.T. Press,
  2024 (draft in `07_docs/Cowles_mmtext4.pdf`).

---

## 01_fortran_source/ — the engine

The Fair-Parke (FP) program. One monolithic FORTRAN 77 file, 209
subroutines/functions.

| File | Size | Dated | Source URL |
|---|---|---|---|
| FP.FOR | 1.02 MB | 2013-11-11 (header), 2018-11-25 (file) | <https://fairmodel.econ.yale.edu/fp/FPFOR.ZIP> |

## 02_executable/ — compiled Windows binary

| File | Size | Dated | Source URL |
|---|---|---|---|
| FP.EXE | 1.91 MB | 2018-11-25 | <https://fairmodel.econ.yale.edu/fp/FPEXE.ZIP> |

## 03_us_model/ — current US Model

Model specification, data, forecast assumptions, and the FP.EXE reference
output (vintage PAB, Feb 20 2026 refresh).

| File | Size | Role | Source |
|---|---|---|---|
| fminput.txt | 20 KB | DSL: equation spec, estimation/solve commands | <https://fairmodel.econ.yale.edu/fp/FMFP.ZIP> |
| fmdata.txt  | 1.83 MB | Quarterly data 1952.1–2025.4 (LOADDATA-formatted) | same |
| fmage.txt   | 19 KB | AG1/AG2/AG3 age-demographic variables through 2029.4 | same |
| fmexog.txt  | 1.6 KB | Forecast exogenous assumptions 2026.1–2029.4 | same |
| fmout.txt   | 979 KB | Reference `FP.EXE` solve output | same |

Latest forecast memo: <https://fairmodel.econ.yale.edu/memo/fm.htm>.

## 04_mc_model/ — current Multi-Country model (MCJ)

| File | Size | Role | Source |
|---|---|---|---|
| MC.INP | 186 KB | MC model spec (equations, SMPLs, GENRs) | inside <https://fairmodel.econ.yale.edu/mcj/down/mcj.zip> |
| YAW.DAT / YDATA.DAT / QUAR.DAT | 10.0 MB total | Country data | same |
| SHR.INP | 540 KB | Trade-share equation spec | same |
| SHRDDD.DAT | 33 MB | Bilateral trade-share data | same |
| OUT | 9.8 MB | Reference `FP.EXE` MC output | same |
| MCEXGSHR.INP / MCEXGSTR.INP / MCSHR1.INP / MCSHR2.INP / MCXX10.INP / PRINT.VAR | (supporting) | same |

Workbook: <https://fairmodel.econ.yale.edu/mcj/docum/mcjwrk.pdf>
(bundled as `07_docs/MCJ_Workbook_current.pdf`).

## 06_examples/ — tutorial IS model

From Fair (1984, Appendix C of the FP User's Guide).

| Files | Role |
|---|---|
| IS.DAT / IS.INP / IS.OUT | Simple IS model (2-equation) — canonical FP tutorial |
| ISRE.INP / ISRE.OUT | IS model with rational expectations |
| SAR.DAT / SAR.INP / SAR.OUT | Stochastic AR example |

Source: <https://fairmodel.econ.yale.edu/fp/FPEG.ZIP>.

## 07_docs/ — documentation and papers

| File | Year | What it is |
|---|---|---|
| FP_UserGuide.pdf | — | Fair-Parke Program User's Guide — definitive reference for the DSL |
| mm2018.pdf | 2018 | **Macroeconometric Modeling** — the bible, 437 pp |
| mm2013.pdf | 2013 | Earlier edition of the same |
| Cowles_mmtext4.pdf | 2024 | Draft of **Macroeconomic Modeling: The Cowles Commission Approach** (M.I.T. Press) |
| TestingMacroeconometricModels_1994.pdf | 1994 | *Testing Macroeconometric Models* (Harvard U. Press) |
| Reflections_2013a.pdf | 2013 | 20-page overview — start here for a quick read |
| MiniVersion_2015a.pdf | 2015 | Simplified pedagogical version of the US Model |
| MCJ_Workbook_current.pdf | — | Current MCJ workbook |
| MCJ_AppendixB.pdf | — | MCJ Appendix B (rest-of-world) |
| USmodel_ChangeLog_2018_2023.pdf | 2023 | Changes to the US Model 2018-01-28 → 2023-07-27 |

---

## Licensing

The files under this directory are Prof. Fair's work and retain his
copyright. They are redistributed here under the terms of his public
distribution (free academic use). The MIT license at the repo root covers
the Python port under `src/` and `tests/` only — see the
[LICENSE](../LICENSE) file for the explicit note.

## Wayback fallback

If any upstream URL above goes dark, the Internet Archive has snapshots
from 1997 onward. Prefix with `https://web.archive.org/web/2024/`.
