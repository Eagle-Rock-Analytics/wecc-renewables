"""
Helper functions for plotting and visualizing renewable energy resource data.

This module provides utility functions for creating wind rose plots, handling geographic
coordinates, and visualizing meteorological data from WRF model outputs.
"""

import datetime
import tempfile
from typing import Any

import matplotlib.cm as cm
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib import rc_context

bucket = "wecc-renewables"
wrf_bucket = "wrf-cmip6-noversioning"
fs = s3fs.S3FileSystem()


# ------------------------------------------------------------------------------------------------
def get_ij_solano(domain: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Get i,j indices for Solano County area within WRF domain.

    Parameters
    ----------
    domain : str
        WRF domain identifier.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        Arrays of i and j indices for Solano County area.
    """

    # Define base state file name from wrf s3 bucket
    base_state = (
        f"s3://{wrf_bucket}/downscaled_products/wrf_coordinates/wrfinput_{domain}"
    )

    # Create a temp file
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=True) as tmp0:
        # Download from S3 into the temp file
        fs.get(base_state, tmp0.name)
        base_ds = xr.open_dataset(tmp0.name)[["XLONG", "XLAT"]].load()

    solano_center_x = -121.94
    solano_center_y = 38.27

    lat_min, lat_max = solano_center_y - 1, solano_center_y + 1
    lon_min, lon_max = solano_center_x - 1, solano_center_x + 1

    lat = base_ds["XLAT"].squeeze()
    lon = base_ds["XLONG"].squeeze()

    mask = (lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)

    # Find indices of area
    i_idx, j_idx = np.where(mask)

    # Find bounding box of indices
    _i_min, _i_max = i_idx.min(), i_idx.max()
    _j_min, _j_max = j_idx.min(), j_idx.max()

    return i_idx, j_idx


# ------------------------------------------------------------------------------------------------
def fix_zorder(ax: Any) -> None:
    """
    Fix z-order of rectangle patches in matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axis object.

    Returns
    -------
    None
    """
    from matplotlib.patches import Rectangle

    # Get all rectangle patches
    rectangle_patches = [c for c in ax.get_children() if isinstance(c, Rectangle)]

    # If no rectangles, nothing to fix
    if not rectangle_patches:
        return None

    # get the minimum zorder of the patches
    minzorder = min(c.get_zorder() for c in rectangle_patches)

    # loop through rectangle patches and set zorder manually
    # (keeping relative order of each rectangle and adding 3)
    for child in rectangle_patches:
        zorder = child.get_zorder()
        child.set_zorder(3 + (zorder - minzorder))
    return None


# ------------------------------------------------------------------------------------------------
def set_theta_grid(ax: Any) -> None:
    """
    Set custom theta grid for wind rose plots using 32-point compass.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Polar axis object.

    Returns
    -------
    None
    """
    # Set custom labels for 32-point compass
    labels = [
        "E",
        "ENE",
        "NE",
        "NNE",
        "N",
        "NNW",
        "NW",
        "WNW",
        "W",
        "WSW",
        "SW",
        "SSW",
        "S",
        "SSE",
        "SE",
        "ESE",
    ]

    # Set theta grid locations every 11.25 degrees
    angles = np.arange(0, 360, 45 / 2)

    ax.set_thetagrids(angles, labels, fontsize=8)

    return None


def scatter_ws_wd(data: Any, labels: list[str]) -> tuple[plt.Figure, Any]:
    """
    Create scatter plot of wind speed vs wind direction.

    Parameters
    ----------
    data : list of lists or pandas.DataFrame
        Wind data either as nested lists [[ws1, wd1], [ws2, wd2]] or DataFrame.
    labels : list of str
        Labels for the data series.

    Returns
    -------
    tuple[matplotlib.pyplot.Figure, Any]
        Figure and axes objects.
    """

    # Ensure we have exactly 2 labels for the comparison plot
    if len(labels) < 2:
        labels = labels + labels  # Duplicate if only one label

    # Determine data length based on input type
    if isinstance(data, pd.DataFrame):
        data_length = len(data)
        # Convert DataFrame to expected nested structure
        ws_cols = [f"windspeed_{labels[0].lower()}", f"windspeed_{labels[1].lower()}"]
        wd_cols = [
            f"winddirection_{labels[0].lower()}",
            f"winddirection_{labels[1].lower()}",
        ]

        plot_data = [
            [data[ws_cols[0]], data[ws_cols[1]]],  # Wind speed comparison
            [data[wd_cols[0]], data[wd_cols[1]]],  # Wind direction comparison
        ]
    else:
        # Assume nested list structure [[ws1, wd1], [ws2, wd2]]
        data_length = len(data[0][0]) if len(data) > 0 and len(data[0]) > 0 else 100
        plot_data = data

    # Create a list of datetime objects for mid-month
    times = np.arange(0, data_length, 1, dtype="int")
    mid_month_dates = [
        datetime.datetime(1980, month, 15, 12)
        for month in [9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8]
    ]  # adjust year if needed

    # Convert datetimes to hour indices since start of year
    tick_locs = np.concatenate(
        (np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]) * 24).cumsum()
        + np.array([[15] * 12]) * 24
    )  # In hours

    # Create plot labels
    plot_labels = [
        [f"WS {labels[0]} (m/s)", f"WS {labels[1]} (m/s)"],
        [f"WD {labels[0]} (deg)", f"WD {labels[1]} (deg)"],
    ]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    ax0, ax1 = axs

    titles = ["speed", "direction"]

    # Use a colormap to map the variable to RGBA colors
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=times.min(), vmax=times.max())
    cmap(norm(times))

    for i, ax in enumerate(axs):

        sc = ax.scatter(
            plot_data[i][0],
            plot_data[i][1],
            # edgecolors=edge_colors,
            c=times,
            s=10,
            marker="o",
            alpha=0.80,
            # facecolors='none',
            # facecolors=edge_colors,
            linewidths=0.75,
        )
        # Add identity line y = x
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        lim_min = min(xmin, ymin)
        lim_max = max(xmax, ymax)

        ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle=":", color="black")
        # Example DataFrame
        example_data = pd.DataFrame({"x": plot_data[i][0], "y": plot_data[i][1]})
        corr = example_data["x"].corr(example_data["y"])
        # Add correlation as text in top-left of the axes
        ax.text(
            0.05,
            0.95,
            f"r = {corr:.2f}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
        )

        # Set custom ticks and labels
        if i == 1:
            cb = fig.colorbar(sc, ax=ax, shrink=0.80)
            if len(tick_locs) <= len(times):  # Only set ticks if they fit
                cb.set_ticks(tick_locs.tolist())
                cb.set_ticklabels([dt.strftime("%b") for dt in mid_month_dates])
        ax.set_title(f"{titles[i]} EC-Earth3 1980")
        ax.set_aspect("equal")
        ax.set_xlabel(plot_labels[i][0])
        ax.set_ylabel(plot_labels[i][1])

    # fig.tight_layout()
    return fig, axs


def plot_3_windroses(ws_data, wd_data, rmax=None, bins=None, titles=None):

    if titles is None:
        titles = ["", "", ""]
    rcParams = {
        "axes.facecolor": "lightgrey",  # Light background for axes
        "figure.facecolor": "white",  # Overall figure background
        "axes.edgecolor": "white",  # For legend or border clarity
        "grid.color": "white",  # For grid color
        "savefig.dpi": 300,  # For high-quality output
    }

    with rc_context(rcParams):
        # Try windrose projection, fall back to polar if not available
        try:
            fig, axs = plt.subplots(
                1, 3, figsize=(16, 6), subplot_kw={"projection": "windrose"}
            )
        except ValueError:
            # Windrose projection not available, use polar instead
            fig, axs = plt.subplots(
                1, 3, figsize=(16, 6), subplot_kw={"projection": "polar"}
            )

        ax0, ax1, ax2 = axs

        cmap = cm.plasma  # Choose colormap

        if bins is not None:
            # Handle bins as either list or array
            bins_min = min(bins) if hasattr(bins, "__iter__") else bins
            # Convert data to numpy arrays if they aren't already
            ws_arrays = []
            for v in ws_data:
                if hasattr(v, "to_numpy"):
                    ws_arrays.append(v.to_numpy().ravel())
                else:
                    ws_arrays.append(np.array(v).ravel())

            if bins_min > np.min(np.concatenate(ws_arrays)):
                calm = [bins_min, bins_min, bins_min]
            else:
                calm = [None, None, None]
        else:
            calm = [None, None, None]
        bins = [bins, bins, bins]

        for i, ax in enumerate(axs):
            # Convert to numpy arrays if needed
            ws_array = (
                ws_data[i].to_numpy().ravel()
                if hasattr(ws_data[i], "to_numpy")
                else np.array(ws_data[i]).ravel()
            )
            wd_array = (
                wd_data[i].to_numpy().ravel()
                if hasattr(wd_data[i], "to_numpy")
                else np.array(wd_data[i]).ravel()
            )

            # Try windrose-specific method, fall back to scatter plot
            if hasattr(ax, "bar") and "windrose" in str(type(ax)):
                ax.bar(
                    wd_array,
                    ws_array,
                    bins=bins[i],
                    cmap=cmap,
                    normed=True,
                    opening=0.8,
                    edgecolor="none",
                    alpha=1,
                    calm_limit=calm[i],
                )
            else:
                # Fallback to polar scatter plot
                ax.scatter(np.radians(wd_array), ws_array, alpha=0.6)

        for i, ax in enumerate(axs):
            fix_zorder(ax)
            if hasattr(ax, "set_thetagrids"):  # Only for polar plots
                set_theta_grid(ax)
            ax.set_title(titles[i])
            if hasattr(ax, "set_legend"):
                ax.set_legend(loc=4)

        if rmax is not None:
            rs0 = np.floor(rmax / 4)
            rs1 = np.ceil(rmax / 4)
            rs = rs1 if rs1 - rmax >= rs0 - rmax else rs0

            for ax in axs:
                # Set consistent magnitude scale
                if hasattr(ax, "set_rmax"):
                    ax.set_rmax(rmax)
                    ticks = np.arange(rs, rmax + rs, rs, dtype="int")
                    ax.set_yticks(ticks)
                    ax.set_yticklabels([f"{p}%" for p in ticks])

        plt.show()

    return fig, axs
