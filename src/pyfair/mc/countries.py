"""Country registry for Fair's Multi-Country (MC) model.

Fair's ``MC.INP`` packages 37 rest-of-world countries plus the US and a
Euro-zone composite block. Each ROW country gets a 2-letter prefix (CA, JA,
AU, ...) and a numeric ``base`` such that its equations are numbered
``base * 10 + k`` for ``k = 1..10`` (most countries have 9 equations;
a handful have fewer because Fair drops specific equations like labor force
or exchange rate for the small/fixed-rate countries).

The registry here is the source of truth for iterating over ROW countries.
Per-equation specs live in ``mc_model.py`` alongside the GENR template,
mirroring how ``us_model.py`` keeps ``UsEquation`` colocated with GENRs.

Enumeration follows ``MC.INP`` (2018 vintage) ``@ XX BEGIN`` markers plus
the three blocks Fair embeds without markers: EU (Euro-zone composite,
equations 175–177), SA (South Africa, 271–272), JO (Jordan, 301–304).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class McCountry:
    """Static metadata for one MC-model country block.

    Attributes:
      prefix: Two-letter country code used as variable-name prefix
        (e.g. ``"CA"`` for Canada, so ``CAIM`` is Canadian imports).
      name: Human-readable country name, for reporting only.
      base: Numeric prefix for equation numbers. Equations are numbered
        ``base * 10 + k``. For example CA has base=4, equations 41..49;
        ME has base=41, equations 411..418.
      eq_numbers: Tuple of equation numbers Fair actually specifies for
        this country. Gaps (e.g. JA skips EQ 53) are explicit.
      annual_lag: ``1`` if quarterly data is used with ``(-1)`` lags;
        ``4`` for the smaller blocks (BE onwards) that use ``(-4)`` lags
        because the underlying series are annual interpolated to quarterly.
    """
    prefix: str
    name: str
    base: int
    eq_numbers: tuple[int, ...]
    annual_lag: int = 1


# Base indices from MC.INP inspection — each ROW country's equations start
# at ``base * 10 + 1``. ``eq_numbers`` enumerates Fair's actual specs.
# ``annual_lag=4`` marks countries where Fair uses ``(-4)`` instead of
# ``(-1)`` because the raw data were annual (see BE BEGIN, line 1920+).
COUNTRIES: tuple[McCountry, ...] = (
    McCountry("US", "United States", 0,
              tuple(range(1, 31)) + (31,)),      # 1..30 + MC-only export price 31
    McCountry("CA", "Canada", 4,
              (41, 42, 43, 44, 45, 46, 47, 48, 49)),
    McCountry("JA", "Japan", 5,
              (51, 52, 54, 55, 56, 57, 58, 59, 60)),       # no 53
    McCountry("AU", "Austria", 6,
              (61, 62, 63, 64, 65, 66, 67, 68)),
    McCountry("FR", "France", 7,
              (71, 72, 73, 74, 75, 76, 77, 78)),
    McCountry("GE", "Germany", 8,
              (81, 82, 83, 84, 85, 86, 87, 88, 89)),
    McCountry("IT", "Italy", 9,
              (91, 92, 93, 94, 95, 97, 98, 99)),           # no 96
    McCountry("NE", "Netherlands", 10,
              (101, 102, 103, 104, 105, 106, 107, 108, 109)),
    McCountry("ST", "Switzerland", 11,
              (111, 112, 113, 114, 115, 116, 117, 118, 119, 120)),
    McCountry("UK", "United Kingdom", 12,
              (121, 122, 123, 124, 125, 126, 127, 128, 129, 130)),
    McCountry("FI", "Finland", 13,
              (131, 132, 134, 135, 137, 138, 139, 140)),    # no 133, 136
    McCountry("AS", "Australia", 14,
              (141, 142, 143, 144, 145, 146, 147, 148, 149, 150)),
    McCountry("SO", "South Africa (orig)", 15,
              (151, 152, 153, 155, 156)),                   # no 154
    McCountry("KO", "South Korea", 16,
              (161, 162, 163, 164, 166, 167, 168)),         # no 165
    McCountry("EU", "Euro Zone", 17,
              (175, 176, 177)),
    McCountry("BE", "Belgium", 18,
              (181, 182, 183, 184, 185, 186, 187, 188, 189, 190),
              annual_lag=4),
    McCountry("DE", "Denmark", 19,
              (191, 192, 193, 194, 195, 196, 197, 198, 199),
              annual_lag=4),
    McCountry("NO", "Norway", 20,
              (201, 202, 203, 204, 205, 206, 207, 209, 210),  # no 208
              annual_lag=4),
    McCountry("SW", "Sweden", 21,
              (211, 212, 213, 214, 215, 217, 218, 219, 220),  # no 216
              annual_lag=4),
    McCountry("GR", "Greece", 22,
              (221, 222, 223, 227),
              annual_lag=4),
    McCountry("IR", "Ireland", 23,
              (231, 232, 233, 234, 235, 237, 238, 239, 240),  # no 236
              annual_lag=4),
    McCountry("PO", "Portugal", 24,
              (241, 242, 243, 244, 245, 246, 247),
              annual_lag=4),
    McCountry("SP", "Spain", 25,
              (251, 252, 253, 254, 255, 256, 257, 258),
              annual_lag=4),
    McCountry("NZ", "New Zealand", 26,
              (261, 262, 263, 264, 265, 266, 267, 268),
              annual_lag=4),
    McCountry("SA", "Saudi Arabia", 27,
              (271, 272),
              annual_lag=4),
    McCountry("CO", "Colombia", 29,
              (291, 292),
              annual_lag=4),
    McCountry("JO", "Jordan", 30,
              (301, 302, 304),
              annual_lag=4),
    McCountry("ID", "India", 32,
              (321, 322, 323, 328),
              annual_lag=4),
    McCountry("MA", "Malaysia", 33,
              (331, 332, 334),
              annual_lag=4),
    McCountry("PA", "Pakistan", 34,
              (341, 342, 343, 344, 348),
              annual_lag=4),
    McCountry("PH", "Philippines", 35,
              (351, 352, 355, 357),
              annual_lag=4),
    McCountry("TH", "Thailand", 36,
              (361, 362, 364, 368),
              annual_lag=4),
    McCountry("CH", "China", 37,
              (371, 372, 373, 374),
              annual_lag=4),
    McCountry("AR", "Argentina", 38,
              (381, 382),
              annual_lag=4),
    McCountry("BR", "Brazil", 39,
              (391, 392),
              annual_lag=4),
    McCountry("CE", "Chile", 40,
              (401, 402),
              annual_lag=4),
    McCountry("ME", "Mexico", 41,
              (411, 412, 418),
              annual_lag=4),
    McCountry("PE", "Peru", 42,
              (421, 422),
              annual_lag=4),
)


def by_prefix(prefix: str) -> McCountry:
    """Look up a country by its 2-letter prefix.

    Raises:
      KeyError: If ``prefix`` is not a registered MC country.
    """
    for c in COUNTRIES:
        if c.prefix == prefix:
            return c
    raise KeyError(f"Unknown MC country prefix: {prefix!r}")


def row_countries() -> tuple[McCountry, ...]:
    """Rest-of-world countries — everything except the US block."""
    return tuple(c for c in COUNTRIES if c.prefix != "US")


def total_equations() -> int:
    """Total number of stochastic equations across all countries."""
    return sum(len(c.eq_numbers) for c in COUNTRIES)
