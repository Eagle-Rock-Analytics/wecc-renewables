"""
Module for Converting Hourly Renewable Energy Data to Daily Aggregates.

This module contains the core function, 'hr_to_day', which performs time-series
aggregation on hourly capacity factor ('cf') and generation ('gen') data stored
in Zarr format. The process ensures accurate spatial masking and time handling
before saving the results as daily Zarr files.

Key Functions and Steps:
1.  **Time Localization and Correction:** The 'localize_and_fix_time' utility adjusts
    the time dimension from UTC to Pacific Standard Time (PST, UTC-8) and corrects
    for common time indexing issues, specifically on leap days.
2.  **Daily Aggregation:** The main function resamples the hourly xarray Dataset
    ('cf') to daily means or ('gen') to daily sums based on the 'var' parameter.
3.  **Spatial Mask Preservation:** The 'mk_exclusions_mask' utility is used to
    create a static mask from the first time step. This mask is applied to the
    aggregated daily data to reintroduce spatial NaN values (representing invalid
    sites due to land use/cover or physical constraints) that may have been
    inadvertently filled with zeros by the aggregation functions.
4.  **Metadata Attachment:** CF-compliant attributes for variables and coordinates
    are updated to reflect the new daily frequency and PST time zone.
5.  **Data Persistence:** The final daily-aggregated and masked Dataset is rechunked
    for efficient access and saved to a new, structured S3 Zarr store, organized
    by technology, module, simulation, scenario, and variable type.

The module is primarily executed via a command-line interface, accepting parameters
to define the specific dataset to process (domain, simulation ID, module, scenario,
variable, and technology).
"""

import sys

import numpy as np
import pandas as pd
import xarray as xr


def hr_to_day(
    domain: str, sim_id: str, module: str, scen: str, var: str, tech: str
) -> None:
    """
    Performs daily aggregations of hourly capacity factor and generation.

    Capacity factor ('cf') is aggregated using a daily mean. Generation ('gen') is summed daily.
    The resulting daily aggregates are saved as zarr to a destination.

    Parameters
    ----------
    domain : str
        WRF domain identifier ('d03' or 'd02').
    sim_id : str
        WRF simulation name (e.g., 'ERA5', 'MIROC6').
    module : str
        Installation module ('utility', 'distributed', 'onshore', or 'offshore').
    scen : str
        Emissions scenario ('historical' or 'ssp370').
    var : str
        Variable to aggregate ('gen' or 'cf').
    tech : str
        Renewables technology ('pv' or 'windpower').

    Returns
    -------
    None
        The function saves the resulting daily aggregates to a zarr store.
    """

    zarr_path = f"s3://wfclimres/era/{tech}_{module}/{sim_id.lower()}/{scen}/1hr/{var}/{domain}/"
    ds = xr.open_zarr(zarr_path)

    # convert from UTC to Pacific Standard Time
    ds = localize_and_fix_time(ds, var)

    # aggregate to daily mean or total depending on variable
    if var == "cf":
        ds_day = ds.resample(time="1D").mean().astype(np.float32)
    elif var == "gen":
        ds_day = ds.resample(time="1D").sum().astype(np.float32)

    # the sum procedure gives 0s where there used to be nans, even when skipna=True.
    # make an exclusions mask to reintroduce the land use/cover nans.
    excl_mask = mk_exclusions_mask(ds)
    ds_day = xr.where(np.isnan(excl_mask), x=np.nan, y=ds_day)

    # attach attributes
    ds_day[var].attrs = ds[var].attrs
    ds_day[var].attrs["frequency"] = "day"
    ds_day.attrs = ds.attrs
    for coord in ds.coords:
        ds_day[coord].attrs = ds[coord].attrs

    # some attrs seem to disappear
    ds_day["x"].attrs = {"standard_name": "projection_x_coordinate", "units": "m"}
    ds_day["y"].attrs = {"standard_name": "projection_y_coordinate", "units": "m"}
    ds_day["time"].attrs["time_zone"] = "UTC -8:00"
    ds_day["time"].attrs["standard_name"] = "time"

    # rechunk to ~128 MB chunks
    ds_day = ds_day.chunk({"time": 9143})

    # back to s3
    out_path = f"s3://wfclimres/era/{tech}_{module}/{sim_id.lower()}/{scen}/day/{var}/{domain}/"
    ds_day.to_zarr(out_path, mode="w")


def mk_exclusions_mask(ds: xr.Dataset, var: str) -> xr.Dataset:
    """
    Create a 2D mask based on NaN values in the first time step of a variable `var`.

    Parameters
    ----------
    ds : xarray.Dataset
        Input Dataset containing the variable to process.
    var: str
        The variable name.

    Returns
    -------
    xarray.DataArray
        A 2D DataArray (time dimension removed) where non-NaN values
        in the initial time step are preserved and NaN values remain NaN.
    """
    ds = ds[var]
    ds = ds.isel(time=0).squeeze()
    ds = xr.where(np.isnan(ds), x=np.nan, y=ds)
    return ds


def localize_and_fix_time(ds: xr.Dataset) -> xr.Dataset:
    """
    Localize the 'time' dimension by shifting it and correcting for leap day issues.

    The time is shifted by 8 hours backward (assuming conversion to PST) and
    leap day times (Feb 29) are shifted back to Feb 28.

    Parameters
    ----------
    ds : xarray.Dataset
        Input Dataset with a 'time' coordinate.

    Returns
    -------
    xarray.Dataset
        Dataset with the 'time' coordinate adjusted for localization and leap day.
    """
    # convert to Pacific Standard Time
    ds["time"] = ds["time"] - pd.Timedelta(hours=8)

    # remove the 8 hours of leap day data which the above correction introduces
    # + roll them back into 02/28
    ds_leap = ds.copy(deep=True)
    ds_leap["time"] = pd.to_datetime(ds_leap["time"])  # set time to datetime
    ds_leap["time"] = np.where(
        (ds_leap.time.dt.month == 2) & (ds_leap.time.dt.day == 29),
        pd.to_datetime(ds_leap["time"]) - pd.DateOffset(days=1),
        ds_leap.time,
    )
    return ds_leap


if __name__ == "__main__":
    domain = str(sys.argv[1])
    sim_id = str(sys.argv[2])
    module = str(sys.argv[3])
    scen = str(sys.argv[4])
    var = str(sys.argv[5])
    tech = str(sys.argv[6])

    hr_to_day(domain, sim_id, module, scen, var, tech)
