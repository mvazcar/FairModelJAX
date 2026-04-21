# IS model data sources

The shipped `06_examples/IS.DAT` is a **frozen snapshot dated November 11, 2013**
(Fair has not updated it since). To use the IS model with fresh data we need to
rebuild it from primary sources.

## What's in IS.DAT

Five series, 247 quarterly observations covering 1952 Q1 – 2013 Q3:

| Name | Units | What it is |
|---|---|---|
| `C` | billion chained dollars, quarterly rate | Real personal consumption expenditures |
| `I` | billion chained dollars, quarterly rate | Real gross private domestic investment |
| `G` | billion chained dollars, quarterly rate | Real government consumption + gross investment |
| `Y` | billion chained dollars, quarterly rate | **Constructed** as `C + I + G` (not real GDP — note net exports are absent) |
| `R` | percent per annum | 3-month Treasury bill rate, quarterly average |

The identity `Y = C + I + G` in `IS.INP` enforces the closed-economy aggregation.
This is *not* the NIPA real GDP series.

## How IS.DAT values compare to current FRED

Spot-checked at four periods:

| Period | IS.DAT C | FRED PCECC96 / 4 | ratio | IS.DAT I | FRED GPDIC1 / 4 | ratio | IS.DAT G | FRED GCEC1 / 4 | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1980 Q1 | 1034.53 | 1133.59 | 0.913 |  243.47 |  243.06 | 1.002 | 351.48 | 466.00 | 0.754 |
| 2000 Q1 | 2027.72 | 2259.94 | 0.897 |  561.27 |  587.81 | 0.955 | 502.27 | 707.98 | 0.709 |
| 2008 Q3 | 2498.38 | 2815.03 | 0.888 |  605.92 |  647.31 | 0.936 | 619.45 | 858.77 | 0.721 |
| 2013 Q3 | 2687.25 | 2974.06 | 0.904 |  620.77 |  749.75 | 0.828 | 639.50 | 818.21 | 0.782 |

**The ratios are not constant**, so we can't rescale FRED to match IS.DAT. The
drift is entirely explained by NIPA chained-dollar rebasing:

- 2013-era FRED real-$ series were in chained **2009$**.
- Post-2023 NIPA comprehensive revision moved the base to chained **2017$**.
- Chain-weighted series *are not additive across base years* — even the
  historical 1980 Q1 value changes when the base year moves.

`R` (the 3-month T-bill) is an exception: IS.DAT's R matches a quarterly average
of FRED `TB3MS` within rounding. Interest rates don't rebase.

## Update strategy

**Accept that "updated" means a fresh dataset, not a splice.** Fair's IS model
is a pedagogical object — the coefficients will shift slightly under new data,
but the model structure is invariant. Since all LHS are logs (`LOGC`, `LOGI`,
`LOGY`), a multiplicative rebase collapses to an intercept shift.

### Option A (recommended): FRED, current vintage

```
C  ← PCECC96  (Real Personal Consumption Expenditures, billions chained 2017$, SAAR)  ÷ 4
I  ← GPDIC1   (Real Gross Private Domestic Investment,  billions chained 2017$, SAAR)  ÷ 4
G  ← GCEC1    (Real Government Consumption + Gross Investment, billions chained 2017$, SAAR)  ÷ 4
Y  ← C + I + G   (construct; do NOT pull GDPC1 — Fair's identity drops net exports)
R  ← TB3MS    (3-Month Treasury Bill, percent, monthly) → quarterly arithmetic mean
```

Pros: single public source, no credentials needed, full history.
Cons: won't exactly reproduce IS.DAT's 2013 coefficients; occasional sub-series
definition changes as BEA refines methodology.

Helper: `python build_is_dat_from_fred.py --output raw/IS.DAT`

### Option B (advanced): derive from Fair's `fmdata.txt`

Fair's US-model data file bundles finer-grained series on the same quarterly
grid:

```
C  = CS + CN + CD         (consumption: services + nondurable + durable)
I  = IHH + IKF + IKH + IKB + IHF + IKG
G  = COG + COS            (gov consumption: federal + state-local)
Y  = C + I + G
R  = RS                   (Fair's 3-month T-bill series)
```

Pros: tracks Fair's own methodological choices (deflators, seasonal
adjustment). Refreshes automatically when `03_us_model/fmdata.txt` is
updated.
Cons: needs quarterly download of `FMFP.ZIP`; only US; even Fair's own current
data doesn't match the 2013 IS.DAT values (same chain-rebase issue).

Not implemented yet — v0.2.

## FRED series reference

Raw CSVs (no API key required):

- <https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCECC96>
- <https://fred.stlouisfed.org/graph/fredgraph.csv?id=GPDIC1>
- <https://fred.stlouisfed.org/graph/fredgraph.csv?id=GCEC1>
- <https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS>
- <https://fred.stlouisfed.org/graph/fredgraph.csv?id=GDPC1>  *(for cross-check, not used)*

## Vintage access (if you really need the 2013 numbers back)

FRED's sibling ALFRED stores vintage-by-vintage data. To pull 2013-vintage
series, append `&vintage_dates=2013-11-22` to the graph CSV URL. Example:

```
https://alfred.stlouisfed.org/graph/fredgraph.csv?id=PCECC96&vintage_dates=2013-11-22
```

This returns the series *as it was published* on 2013-11-22, in whatever base
year was current at that vintage (chained 2009$). That's as close as you get
to Fair's original IS.DAT numbers from public sources.
