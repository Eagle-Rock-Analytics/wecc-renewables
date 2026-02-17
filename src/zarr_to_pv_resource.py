"""
Convert zarr solar resource data for photovoltaic analysis.

This module processes solar irradiance and meteorological data from zarr format,
calculating derived variables like wind speed, surface albedo, and atmospheric
parameters needed for PV resource assessment.
"""

import numpy as np
import xarray as xr
from xclim.indices import relative_humidity, uas_vas_2_sfcwind


def preprocess_pv_wrapper(ds: xr.Dataset) -> xr.Dataset:
    """
    Preprocess solar resource dataset for PV analysis.

    Performs variable renaming, unit conversions, and calculates derived
    variables such as surface albedo and wind speed.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing solar and meteorological variables.

    Returns
    -------
    xarray.Dataset
        Preprocessed dataset ready for PV analysis.
    """
    # ds = ds.copy()
    my_atts = ds.attrs
    rename_dict = {
        "swdnb": "ghi",
        "swddni": "dni",
        "swddif": "dhi",
        "t2": "air_temperature",
        "snownc": "snow_depth",
    }
    ds = ds.rename(rename_dict)
    ds["air_temperature"] = ds["air_temperature"] - 273.15
    ds["surface_albedo"] = _derive_surface_albedo(ds)
    ds["wind_speed"] = _wind_10m_speed(ds)
    ds["snow_depth"] = ds["snow_depth"] * 0.1
    ds.attrs = my_atts
    ds = ds.transpose("time", "y", "x")
    to_drop = ["u10", "v10", "swupb"]
    ds = ds.drop(to_drop)
    return ds


def _compute_dewpointtemp(ds: xr.Dataset) -> xr.DataArray:
    """
    Compute dew point temperature from air temperature, humidity, and pressure.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing air_temperature, q2, and surface_pressure variables.

    Returns
    -------
    xarray.DataArray
        Dew point temperature in degrees Celsius.
    """
    ds["rel_hum"] = relative_humidity(
        tas=ds["air_temperature"], huss=ds["q2"], ps=ds["surface_pressure"]
    )
    es = 0.611 * np.exp(
        5423 * ((1 / 273) - (1 / ds.air_temperature))
    )  # calculates saturation vapor pressure
    e_vap = (es * ds.rel_hum) / 100.0  # calculates vapor pressure
    tdps = (
        (1 / 273) - 0.0001844 * np.log(e_vap / 0.611)
    ) ** -1  # calculates dew point temperature, units = K
    tdps.name = "dew_point"
    tdps = tdps - 273.15
    tdps.attrs["units"] = "C"
    return tdps


def _derive_surface_albedo(ds: xr.Dataset) -> xr.DataArray:
    """
    Calculate surface albedo from upward and downward shortwave radiation.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing swupb and ghi variables.

    Returns
    -------
    xarray.DataArray
        Surface albedo (unitless).
    """
    albedo = ds["swupb"] / ds["ghi"]
    albedo = albedo.fillna(0)
    return albedo


def _wind_10m_speed(ds: xr.Dataset) -> xr.DataArray:
    """
    Calculate wind speed at 10m height from u and v components.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing u10 and v10 wind components.

    Returns
    -------
    xarray.DataArray
        Wind speed at 10m height in m/s.
    """
    calm_wind_thresh = "0.5 m/s"
    speed_da, _ = uas_vas_2_sfcwind(
        uas=ds.u10, vas=ds.v10, calm_wind_thresh=calm_wind_thresh
    )
    return speed_da
