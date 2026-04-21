"""Build a fresh IS.DAT from FRED (St. Louis Fed).

Pulls the 5 series behind Fair's IS tutorial model from FRED CSVs (no API key
required), aggregates the monthly interest rate to quarterly averages,
constructs Y = C + I + G, and writes the result in Fair's DSL format
(``SMPL ... LOAD <var> ; <numbers>`` blocks).

Usage::

    python build_is_dat_from_fred.py
    python build_is_dat_from_fred.py --output raw/IS_updated.DAT --start 1952Q1

See ``IS_DATA_SOURCES.md`` for the series mapping and rationale. In short: the
resulting values will NOT match the shipped 2013-vintage ``06_examples/IS.DAT``
because the NIPA chained-dollar base year has moved; the model structure is
invariant, so coefficients shift by roughly a constant in the log-specification
equations.
"""
from __future__ import annotations

import argparse
import io
import urllib.request
from csv import DictReader
from datetime import date
from pathlib import Path
from typing import NamedTuple

import polars as pl


# ---------------------------------------------------------------------------
# FRED series mapping
# ---------------------------------------------------------------------------
#
# Variables are named after Fair's IS model conventions (C, I, G, Y, R). The
# mapping rationale lives in IS_DATA_SOURCES.md §Option A.

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


class FredSeries(NamedTuple):
    """One FRED series we pull for the IS model.

    Attributes:
      model_name: The variable name in Fair's IS model (e.g. ``"C"``).
      fred_id: The FRED series id to download.
      description: Human-readable description for documentation.
      frequency: ``"quarterly"`` (one obs per quarter, annualized for $)
        or ``"monthly"`` (aggregate to quarterly mean).
    """
    model_name: str
    fred_id: str
    description: str
    frequency: str


# Real quantities come out of FRED annualized; Fair's IS.DAT convention is
# quarterly rate, so we divide by 4.
_REAL_AGGREGATE_SERIES: list[FredSeries] = [
    FredSeries("C", "PCECC96",
               "Real Personal Consumption Expenditures (chained 2017$, SAAR)",
               "quarterly"),
    FredSeries("I", "GPDIC1",
               "Real Gross Private Domestic Investment (chained 2017$, SAAR)",
               "quarterly"),
    FredSeries("G", "GCEC1",
               "Real Gov Consumption + Gross Investment (chained 2017$, SAAR)",
               "quarterly"),
]

_INTEREST_RATE_SERIES: list[FredSeries] = [
    FredSeries("R", "TB3MS",
               "3-Month Treasury Bill Secondary Market Rate (monthly %)",
               "monthly"),
]


# ---------------------------------------------------------------------------
# FRED fetching + time-aggregation
# ---------------------------------------------------------------------------

def fetch_fred_csv(fred_id: str, timeout: float = 30.0) -> pl.DataFrame:
    """Download one FRED series as a (date, value) polars DataFrame.

    Uses the no-auth CSV endpoint. Missing observations (``.``) are skipped.

    Args:
      fred_id: FRED series id, e.g. ``"PCECC96"``.
      timeout: HTTP timeout in seconds.

    Returns:
      DataFrame with columns ``date`` (date) and ``value`` (float64).
    """
    url = FRED_CSV_URL.format(series=fred_id)
    print(f"  fetching {fred_id} ...", end=" ", flush=True)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        raw = response.read().decode()

    reader = DictReader(io.StringIO(raw))
    date_col, value_col = reader.fieldnames[0], reader.fieldnames[1]
    rows: list[dict] = []
    for row in reader:
        value_str = row[value_col]
        if value_str in ("", "."):
            continue
        rows.append({
            "date": date.fromisoformat(row[date_col]),
            "value": float(value_str),
        })

    df = pl.DataFrame(rows)
    print(f"{df.height} obs ({df['date'].min()} to {df['date'].max()})")
    return df


def quarterly_from_quarterly_series(df: pl.DataFrame) -> pl.DataFrame:
    """Extract (year, quarter, value) from a FRED quarterly series.

    FRED quarterly observations are dated on the first day of the quarter.
    Real-$ series are reported as Seasonally Adjusted Annual Rates (SAAR),
    which we divide by 4 to match Fair's quarterly-rate convention.
    """
    return df.with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.quarter().alias("quarter"),
        (pl.col("value") / 4.0).alias("value"),
    ).select(["year", "quarter", "value"])


def quarterly_from_monthly_series(df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate monthly observations to quarterly arithmetic means."""
    return (
        df.with_columns(
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.quarter().alias("quarter"),
        )
        .group_by(["year", "quarter"])
        .agg(pl.col("value").mean())
        .sort(["year", "quarter"])
    )


def aggregate_to_quarterly(
    df: pl.DataFrame, frequency: str
) -> pl.DataFrame:
    """Dispatch on frequency to the right aggregator."""
    if frequency == "quarterly":
        return quarterly_from_quarterly_series(df)
    if frequency == "monthly":
        return quarterly_from_monthly_series(df)
    raise ValueError(f"Unknown frequency {frequency!r}")


# ---------------------------------------------------------------------------
# Assemble the IS dataset
# ---------------------------------------------------------------------------

def fetch_all_is_series() -> dict[str, pl.DataFrame]:
    """Fetch every FRED series needed for IS and return them by model name.

    Returns:
      ``{"C": <frame>, "I": <frame>, "G": <frame>, "R": <frame>}`` where each
      frame has columns ``(year, quarter, value)``.
    """
    total = len(_REAL_AGGREGATE_SERIES) + len(_INTEREST_RATE_SERIES)
    print(f"Fetching {total} FRED series...")
    result: dict[str, pl.DataFrame] = {}
    for spec in _REAL_AGGREGATE_SERIES + _INTEREST_RATE_SERIES:
        raw = fetch_fred_csv(spec.fred_id)
        result[spec.model_name] = aggregate_to_quarterly(raw, spec.frequency)
    return result


def join_on_quarter_grid(series_by_name: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Inner-join every series on (year, quarter) and derive Y = C + I + G.

    Returns a wide DataFrame sorted chronologically with columns
    (year, quarter, C, I, Y, G, R).
    """
    joined: pl.DataFrame | None = None
    for name, df in series_by_name.items():
        df_renamed = df.rename({"value": name})
        joined = df_renamed if joined is None else joined.join(
            df_renamed, on=["year", "quarter"], how="inner"
        )
    assert joined is not None

    joined = joined.sort(["year", "quarter"]).with_columns(
        (pl.col("C") + pl.col("I") + pl.col("G")).alias("Y")
    )
    return joined.select(["year", "quarter", "C", "I", "Y", "G", "R"])


# ---------------------------------------------------------------------------
# Period filtering
# ---------------------------------------------------------------------------

def _parse_fair_period(period_str: str) -> tuple[int, int]:
    """Parse a period like ``"1952Q1"`` into ``(year, quarter)``."""
    year_part, quarter_part = period_str.upper().split("Q")
    return int(year_part), int(quarter_part)


def filter_to_sample(
    df: pl.DataFrame, start: str, end: str | None
) -> pl.DataFrame:
    """Restrict to a Fair-style period range (inclusive on both ends)."""
    start_year, start_quarter = _parse_fair_period(start)
    after_start = (
        (pl.col("year") > start_year)
        | ((pl.col("year") == start_year) & (pl.col("quarter") >= start_quarter))
    )
    result = df.filter(after_start)

    if end is not None:
        end_year, end_quarter = _parse_fair_period(end)
        before_end = (
            (pl.col("year") < end_year)
            | ((pl.col("year") == end_year) & (pl.col("quarter") <= end_quarter))
        )
        result = result.filter(before_end)
    return result


# ---------------------------------------------------------------------------
# Fair-format writer
# ---------------------------------------------------------------------------

_VARIABLES_IN_ORDER = ["C", "I", "Y", "G", "R"]
_VALUES_PER_LINE = 4


def write_fair_dat(df: pl.DataFrame, path: Path) -> None:
    """Serialize a wide DataFrame into Fair's IS.DAT text format.

    Emits one ``SMPL`` header followed by a ``LOAD`` block per variable,
    with values in scientific notation, four per line.
    """
    first_period = f"{df['year'][0]}.{df['quarter'][0]}"
    last_period = f"{df['year'][-1]}.{df['quarter'][-1]}"

    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write(f" SMPL    {first_period}   {last_period} ;\n")
        for variable in _VARIABLES_IN_ORDER:
            f.write(f" LOAD {variable:<8s} ;\n")
            values = df[variable].to_numpy()
            for start in range(0, len(values), _VALUES_PER_LINE):
                chunk = values[start : start + _VALUES_PER_LINE]
                formatted = "  ".join(f"{v:+.11E}" for v in chunk)
                f.write(f"  {formatted}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a fresh IS.DAT from FRED CSVs."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("raw/IS.DAT"),
        help="Output path (default: raw/IS.DAT).",
    )
    parser.add_argument(
        "--start", default="1952Q1",
        help="First quarter (default: 1952Q1).",
    )
    parser.add_argument(
        "--end", default=None,
        help="Last quarter (default: latest available).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    series_by_name = fetch_all_is_series()
    wide = join_on_quarter_grid(series_by_name)
    wide = filter_to_sample(wide, args.start, args.end)
    print(f"\nSample: "
          f"{wide['year'][0]}Q{wide['quarter'][0]} - "
          f"{wide['year'][-1]}Q{wide['quarter'][-1]}")
    print(f"Rows:   {wide.height}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_fair_dat(wide, args.output)
    print(f"\nWrote {args.output} ({args.output.stat().st_size} bytes)")

    print("\nLast 3 quarters:")
    for row in wide.tail(3).iter_rows(named=True):
        print(f"  {row['year']}Q{row['quarter']}: "
              f"C={row['C']:.2f}  I={row['I']:.2f}  G={row['G']:.2f}  "
              f"Y={row['Y']:.2f}  R={row['R']:.4f}")


if __name__ == "__main__":
    main()
