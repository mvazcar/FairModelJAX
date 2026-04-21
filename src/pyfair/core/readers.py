"""Parsers for Fair's DSL data-file format.

The format looks like::

    SMPL 1952.1 2013.3;
    LOAD C;
    0.35775E+03  0.36437E+03  0.36867E+03 ...
    LOAD I;
    0.74650E+02 ...
    ...

``SMPL`` declares a quarterly sample range; each ``LOAD`` block names one
series whose values follow on subsequent lines, four per line, space-separated.
``@`` introduces a comment.

This module reads such a file into a long polars DataFrame
(``period``, ``variable``, ``value``), then optionally pivots to wide and
extracts ``jnp`` arrays. Polars is used only at the I/O boundary; the model
kernel downstream is pure JAX.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import polars as pl


def parse_fair_data(path: Path) -> pl.DataFrame:
    """Parse a Fair-format data file into a long DataFrame.

    Args:
      path: Path to a file like ``IS.DAT`` or ``fmdata.txt``.

    Returns:
      Long DataFrame with columns ``period`` (``"1952Q1"`` string),
      ``variable`` (e.g. ``"C"``), ``value`` (float). One row per observation.
    """
    text = path.read_text()
    rows: list[dict] = []
    current_sample_start: str | None = None
    current_variable: str | None = None
    current_values: list[float] = []

    def flush_current_block() -> None:
        """Emit accumulated values for the current LOAD block."""
        if current_variable is None or not current_values:
            return
        _emit_rows(
            rows=rows,
            variable=current_variable,
            values=current_values,
            sample_start=current_sample_start,
        )

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("@"):
            continue
        keyword = line.split(None, 1)[0].upper()

        if keyword == "SMPL":
            flush_current_block()
            current_variable, current_values = None, []
            # Example: "SMPL 1952.1 2013.3;"
            tokens = line.rstrip(";").split()
            current_sample_start = tokens[1]

        elif keyword == "LOAD":
            flush_current_block()
            # Example: "LOAD C ;"
            current_variable = line.rstrip(";").split()[1]
            current_values = []

        else:
            # Numeric payload line for the current LOAD block.
            for token in line.rstrip(";").split():
                try:
                    current_values.append(float(token))
                except ValueError:
                    # Silently skip unknown directives; matches Fair's behavior.
                    pass

    flush_current_block()

    return pl.DataFrame(
        rows,
        schema={"period": pl.Utf8, "variable": pl.Utf8, "value": pl.Float64},
    )


def _emit_rows(
    rows: list[dict],
    variable: str,
    values: list[float],
    sample_start: str | None,
) -> None:
    """Expand a LOAD block into one row per quarter and append to ``rows``.

    ``sample_start`` is a Fair-style period string like ``"1952.1"``; we count
    quarters forward from there.
    """
    if sample_start is None:
        return
    year, quarter = sample_start.split(".")
    year, quarter = int(year), int(quarter)
    for value in values:
        rows.append({
            "period": f"{year}Q{quarter}",
            "variable": variable,
            "value": value,
        })
        quarter += 1
        if quarter > 4:
            quarter = 1
            year += 1


def parse_fair_xid_data(path: Path) -> pl.DataFrame:
    """Parse Fair's MC-data ``LOAD XID`` format into a long DataFrame.

    This format — used in ``YDATA.DAT`` and ``QUAR.DAT`` — packs multiple
    variables into a single ``LOAD`` block, with the period label
    prefixing each data row::

        SMPL 1960.1 2017.4;
        LOAD XID CAC CAE CAEX CAE10;
         1960.1    v_CAC_1   v_CAE_1   v_CAEX_1   v_CAE10_1
         1960.2    v_CAC_2   v_CAE_2   v_CAEX_2   v_CAE10_2
         ...

    Fair's missing-value sentinel (``-99``) is converted to null.

    Args:
      path: Path to the data file.

    Returns:
      Long DataFrame with ``period``, ``variable``, ``value`` columns —
      schema identical to ``parse_fair_data`` so downstream pivots work.
    """
    text = path.read_text()
    rows: list[dict] = []
    current_variables: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("@"):
            continue
        stripped = line.rstrip(";")
        tokens = stripped.split()
        keyword = tokens[0].upper() if tokens else ""

        if keyword == "SMPL":
            current_variables = []
            continue

        if keyword == "LOAD":
            # Format: LOAD XID var1 var2 ... varN;
            if len(tokens) >= 2 and tokens[1].upper() == "XID":
                current_variables = tokens[2:]
            else:
                current_variables = []
            continue

        # Numeric data row: "1960.1  v1  v2  ... vN"
        if not current_variables:
            continue
        if "." not in tokens[0]:
            continue

        period_tok = tokens[0]
        year, quarter = period_tok.split(".")
        period = f"{int(year)}Q{int(quarter)}"
        values = tokens[1: 1 + len(current_variables)]
        for var, v in zip(current_variables, values):
            try:
                f = float(v)
            except ValueError:
                continue
            # Fair uses -99 as a missing-value sentinel.
            if f == -99.0:
                continue
            rows.append({"period": period, "variable": var, "value": f})

    return pl.DataFrame(
        rows,
        schema={"period": pl.Utf8, "variable": pl.Utf8, "value": pl.Float64},
    )


def pivot_to_wide(long_frame: pl.DataFrame) -> pl.DataFrame:
    """Reshape ``(period, variable, value)`` -> one row per period, one col per var.

    If the same variable appears in multiple LOAD blocks (fmdata.txt does this
    for a couple of series where Fair reloads over a different SMPL range),
    the last block wins — matching Fair's own sequential processing.

    Rows are sorted chronologically by ``period``.
    """
    deduped = long_frame.unique(subset=["period", "variable"], keep="last")
    return deduped.pivot(
        index="period", on="variable", values="value"
    ).sort("period")


def to_state_dict(
    wide_frame: pl.DataFrame, variables: list[str]
) -> dict[str, jnp.ndarray]:
    """Extract selected columns from the wide frame into a dict of jnp arrays.

    Args:
      wide_frame: Output of ``pivot_to_wide``.
      variables: Columns to extract.

    Returns:
      ``{variable_name: jnp.array}`` ready to feed the JAX kernel.

    Raises:
      KeyError: If a requested variable is missing from the frame. This is
        deliberate — forgetting to declare a series in the equation registry
        should surface as a clear error, not a silent zero.
    """
    result: dict[str, jnp.ndarray] = {}
    for variable in variables:
        if variable not in wide_frame.columns:
            raise KeyError(
                f"Variable {variable!r} not in data. Available: {wide_frame.columns}"
            )
        result[variable] = jnp.asarray(
            wide_frame[variable].to_numpy(), dtype=jnp.float64
        )
    return result
