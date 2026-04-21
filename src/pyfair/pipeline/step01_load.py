"""Step 1 — load a Fair-format data file into a wide polars frame.

Parses ``IS.DAT`` or ``fmdata.txt`` via ``readers.parse_fair_data``, pivots
to wide layout (one row per quarter, one column per variable), and caches
to ``output/step01_data_<model>_<data_source>.parquet``.

The step is idempotent: subsequent runs return the cached frame unless
``force=True``.
"""
from __future__ import annotations

import polars as pl

from .. import config
from ..core import readers


_FRED_MISSING_HINT = (
    "FRED-built IS.DAT is missing. Build it first:\n"
    "  cd pyfair && python build_is_dat_from_fred.py\n"
    "Or pass --data-source fair_2013 to use the shipped 2013 snapshot."
)


def _resolve_input_path(model: str, data_source: str):
    """Pick the right raw data file for a (model, data_source) combo."""
    if model == "is":
        return config.is_dat_path(data_source)
    if model == "us":
        return config.US_FMDATA
    raise ValueError(f"Unknown model {model!r}")


def run(
    model: str,
    data_source: str = "fred",
    force: bool = False,
) -> pl.DataFrame:
    """Load the model's input data into a wide polars DataFrame.

    Args:
      model: ``"is"`` (tutorial 2-equation model) or ``"us"`` (full model —
        not fully wired in v0.1).
      data_source: ``"fred"`` or ``"fair_2013"``. Ignored when ``model != "is"``.
      force: If True, ignore any cached parquet and re-parse from source.

    Returns:
      Wide DataFrame: one row per period, one column per raw series.

    Raises:
      FileNotFoundError: If the selected input file doesn't exist on disk.
        For ``data_source="fred"`` the error includes instructions for
        rebuilding.
    """
    cache_path = config.OUTPUT_DIR / f"step01_data_{model}_{data_source}.parquet"
    if cache_path.exists() and not force:
        return pl.read_parquet(cache_path)

    input_path = _resolve_input_path(model, data_source)
    if not input_path.exists():
        if model == "is" and data_source == "fred":
            raise FileNotFoundError(f"{input_path}\n\n{_FRED_MISSING_HINT}")
        raise FileNotFoundError(f"{input_path} not found.")

    long_frame = readers.parse_fair_data(input_path)
    wide_frame = readers.pivot_to_wide(long_frame)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wide_frame.write_parquet(cache_path)
    return wide_frame
