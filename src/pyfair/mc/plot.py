"""Plotting utilities for MC-model forecasts and Monte Carlo paths.

Thin matplotlib wrappers for common visualizations:

* ``plot_country_path(history_frame, path, variable, ax=None)`` — overlay a
  single deterministic forecast on historical data for one country-level
  variable.
* ``plot_monte_carlo_fan(history_frame, mc_result, variable, ...)`` — fan
  chart showing 10/25/50/75/90th percentiles of Monte Carlo draws.
* ``plot_multi_country(history_frame, endogenous_result, variable, ...)`` —
  small-multiples panel comparing variable across countries.

The module has matplotlib as an optional dependency; import fails with
a clear error if matplotlib is absent. Core model functionality doesn't
depend on plotting.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    # Only evaluated by type checkers — avoids requiring matplotlib/numpy
    # at import time for users who don't plot.
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


# Fan-chart alpha is ``FAN_BAND_ALPHA_BASE + FAN_BAND_ALPHA_PER_PCT
# * (hi − lo) / 100`` capped at ``FAN_BAND_ALPHA_MAX`` — widens the
# colour fade as the percentile range grows so inner bands are
# visually tighter than outer bands.
_FAN_BAND_ALPHA_BASE = 0.15
_FAN_BAND_ALPHA_PER_PCT = 0.15
_FAN_BAND_ALPHA_MAX = 0.35


def _require_matplotlib() -> None:
    """Import matplotlib lazily; raise with install instructions on miss."""
    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pyfair.mc_plot requires matplotlib. Install via "
            "`pip install matplotlib` (already a dev-extras in pyproject)."
        ) from exc


def plot_country_path(
    history_frame: pl.DataFrame,
    path: list[dict[str, Any]],
    variable: str,
    ax: "Axes | None" = None,
    history_color: str = "C0",
    forecast_color: str = "C1",
    title: str | None = None,
) -> "Axes":
    """Overlay a forecast path on historical data for one variable.

    Args:
      history_frame: Wide frame containing ``period`` plus the variable.
      path: Result from ``simulate_country_path`` — list of
        ``{"period", "solved", ...}`` dicts.
      variable: Column to plot (e.g. ``"CAY"``).
      ax: Optional matplotlib axis; creates one if None.
      history_color, forecast_color: matplotlib colors.
      title: Title; defaults to the variable name.

    Returns:
      The matplotlib axis with history + forecast plotted.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    # Historical series.
    hist = history_frame.filter(pl.col(variable).is_not_null()).select(
        ["period", variable]
    ).sort("period")
    hist_periods = hist["period"].to_list()
    hist_values = hist[variable].to_list()
    ax.plot(hist_periods, hist_values, color=history_color,
            label=f"{variable} (history)", linewidth=1.2)

    # Forecast path.
    fcst_periods = [r["period"] for r in path]
    fcst_values = [r["solved"].get(variable) for r in path]
    fcst_values = [v for v in fcst_values if v is not None]
    if fcst_values:
        # Continuous line connecting history's last point to the forecast start.
        if hist_values:
            ax.plot(
                [hist_periods[-1], fcst_periods[0]],
                [hist_values[-1], fcst_values[0]],
                color=forecast_color, linestyle="--", linewidth=0.8,
            )
        ax.plot(fcst_periods[:len(fcst_values)], fcst_values,
                color=forecast_color,
                label=f"{variable} (forecast)", linewidth=1.8, marker="o",
                markersize=3)

    ax.set_title(title or variable)
    ax.set_xlabel("period")
    ax.set_ylabel(variable)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    return ax


def plot_monte_carlo_fan(
    history_frame: pl.DataFrame,
    mc_result: dict[str, list[list[dict[str, Any]]]],
    variable: str,
    ax: "Axes | None" = None,
    history_color: str = "C0",
    fan_color: str = "C1",
    percentiles: tuple[int, ...] = (10, 25, 50, 75, 90),
    title: str | None = None,
) -> "Axes":
    """Fan chart of Monte Carlo forecast paths.

    Args:
      history_frame: Historical data (wide frame with ``period`` column).
      mc_result: Output of ``forecast_country_monte_carlo``. Expects
        ``mc_result["draws"] = [[{period, solved, ...}, ...], ...]``.
      variable: Variable to visualize.
      ax: Optional axis.
      percentiles: Bands to shade. Defaults to (10, 25, 50, 75, 90).
      title: Chart title.

    Returns:
      The matplotlib axis with history + percentile fan plotted.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    # Historical overlay.
    hist = history_frame.filter(pl.col(variable).is_not_null()).select(
        ["period", variable]
    ).sort("period")
    ax.plot(hist["period"].to_list(), hist[variable].to_list(),
            color=history_color, label=f"{variable} (history)", linewidth=1.2)

    # Collect draw values per period.
    draws = mc_result["draws"]
    if not draws:
        return ax
    periods = [r["period"] for r in draws[0]]
    # matrix: n_draws x n_periods
    mat = np.array([
        [r["solved"].get(variable, np.nan) for r in draw]
        for draw in draws
    ])
    # Percentile bands.
    pctiles = np.nanpercentile(mat, percentiles, axis=0)

    # Pair up bands symmetrically around the median for shading.
    mid_idx = len(percentiles) // 2
    bands = []
    for i in range(mid_idx):
        bands.append((percentiles[i], percentiles[-1 - i],
                       pctiles[i], pctiles[-1 - i]))
    for lo, hi, lo_vals, hi_vals in bands:
        alpha = _FAN_BAND_ALPHA_BASE + _FAN_BAND_ALPHA_PER_PCT * (hi - lo) / 100
        ax.fill_between(periods, lo_vals, hi_vals,
                         color=fan_color, alpha=min(alpha, _FAN_BAND_ALPHA_MAX),
                         label=f"{lo}-{hi}%")
    # Median line.
    ax.plot(periods, pctiles[mid_idx], color=fan_color,
            linewidth=1.8, label="median")

    ax.set_title(title or f"{variable} — Monte Carlo fan")
    ax.set_xlabel("period")
    ax.set_ylabel(variable)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    return ax


def plot_multi_country(
    history_frame: pl.DataFrame,
    endogenous_result: dict[str, Any],
    variable_suffix: str,
    countries: tuple[str, ...] | None = None,
    ncols: int = 3,
) -> "Figure":
    """Small-multiples panel: the same variable across countries.

    Plots ``<country><variable_suffix>`` for each country, e.g.
    ``"Y"`` → CAY, JAY, GEY, ...

    Args:
      history_frame: Wide frame with historical data.
      endogenous_result: Output of ``simulate_mc_endogenous``. Expects
        ``result["paths"] = {country: [{period, solved, ...}, ...]}``.
      variable_suffix: The per-country suffix (``"Y"``, ``"IM"``, ``"C"``).
      countries: Explicit country list. Defaults to every key in
        ``endogenous_result["paths"]``.
      ncols: Columns in the small-multiples grid.

    Returns:
      Matplotlib figure with the grid of subplots.
    """
    _require_matplotlib()
    import matplotlib.pyplot as plt

    if countries is None:
        countries = tuple(endogenous_result["paths"].keys())
    n = len(countries)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                              squeeze=False)
    for i, prefix in enumerate(countries):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        variable = f"{prefix}{variable_suffix}"
        path = endogenous_result["paths"].get(prefix, [])
        plot_country_path(history_frame, path, variable, ax=ax,
                           title=f"{prefix}: {variable}")
    # Hide unused subplots.
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")
    fig.tight_layout()
    return fig
