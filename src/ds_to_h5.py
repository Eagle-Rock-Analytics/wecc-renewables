"""
Convert xarray datasets to HDF5 format for renewable resource analysis.

This module provides functions to create exclusion masks, write metadata, and append resource
data from xarray datasets to HDF5 files, supporting wind and solar resource workflows.
"""

from datetime import datetime
from typing import Any

import numpy as np
import pyproj
import rioxarray as rio
import tables as tb
import xarray as xr

from src.constants import UNSET

rio.set_options(export_grid_mapping=False)
import geopandas as gpd  # noqa: E402


def make_exclusions_mask(
    ds: xr.Dataset, exclusions_dict: dict[str, str], outfile: Any
) -> xr.Dataset:
    """
    Create exclusion masks for siting restrictions and merge into a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with coordinates.
    exclusions_dict : dict of str
        Dictionary mapping exclusion reason to file path.
    outfile : file-like
        Output file handle for logging.

    Returns
    -------
    xarray.Dataset
        Dataset containing exclusion masks for each reason.
    """
    # outpath="/shared/renewable-profiles/cli/pcluster/"
    f = outfile
    # WRF CRS
    crs = pyproj.CRS(
        """+proj=lcc +lat_0=38. +lon_0=-70. +lat_1=30.
        +lat_2=60. +R=6370000. +units=m +no_defs"""
    )
    exclusion_files = list(exclusions_dict.values())
    exclusion_reasons = list(exclusions_dict.keys())
    mask_list = []
    for i in range(len(exclusions_dict)):
        reason = exclusion_reasons[i]
        print(reason, end="\n", flush=True, file=f)
        fname = exclusion_files[i]
        # gdf = dask_geopandas.read_file(fname, npartitions=10)
        gdf = gpd.read_file(fname)
        print("gdf read in", end="\n", flush=True, file=f)
        # gdf_reproj = gdf.to_crs(crs).compute()
        gdf_reproj = gdf.to_crs(crs)
        print("reprojected", end="\n", flush=True, file=f)
        # gdf_reproj = gdf_reproj.compute()

        # make integer mask for a given siting restriction
        list(ds.data_vars)
        var_shape = (len(ds.y), len(ds.x))
        syn_data = np.zeros(var_shape)
        da = xr.Dataset(
            {
                reason: (["y", "x"], syn_data),
            },
            coords=ds.coords,
        )
        da = da.rio.write_crs("EPSG:4326")
        mask_da = da.rio.clip(
            geometries=gdf_reproj["geometry"],
            crs="epsg:4326",
            drop=True,
            invert=True,
            all_touched=False,
            from_disk=True,
        )
        print("ds clipped", end="\n", flush=True, file=f)
        mask_da = mask_da.fillna(1)
        mask_da = mask_da[reason].astype(int)
        mask_da.attrs["description"] = (
            "1 for land use restriction," + " 0 for no restriction."
        )
        mask_list.append(mask_da)
        print("da appended", end="\n", flush=True, file=f)
        gdf = UNSET
    mask_ds = xr.merge(mask_list)
    return mask_ds


def meta_h5(ds: xr.Dataset, meta_file: str = "meta_node.h5") -> None:
    """
    Write metadata to HDF5 file.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing metadata variables.
    meta_file : str, optional
        Path to HDF5 file.

    Returns
    -------
    None
    """

    class Particle(tb.IsDescription):
        """
        PyTables description for metadata table.

        Attributes
        ----------
        latitude : float
            Latitude value.
        longitude : float
            Longitude value.
        elevation : float
            Elevation value.
        timezone : int
            Timezone value.
        """

        latitude = tb.Float64Col(shape=(), dflt=0.0, pos=1)
        longitude = tb.Float64Col(shape=(), dflt=0.0, pos=2)
        elevation = tb.Float64Col(shape=(), dflt=0.0, pos=3)
        timezone = tb.Int64Col(shape=(), dflt=0, pos=4)
        # landmask = tb.Int64Col(shape=(), dflt=0, pos=5)
        # fed_and_state_protected = tb.Int64Col(shape=(), dflt=0, pos=6)
        # wind_medium_high_urban = tb.Int64Col(shape=(), dflt=0, pos=7)
        # pv_high_urban = tb.Int64Col(shape=(), dflt=0, pos=8)
        # wind_slope_over_20p = tb.Int64Col(shape=(), dflt=0, pos=9)
        # pv_slope_over_5p = tb.Int64Col(shape=(), dflt=0, pos=10)

    ds = ds.reset_coords(["lat", "lon"])
    ds = ds.rename(
        {
            "lat": "latitude",
            "lon": "longitude",
        }
    )
    len_x = len(ds.x)
    len_y = len(ds.y)
    timezone = np.full((len_y, len_x), 0)
    ds["timezone"] = (["y", "x"], timezone)
    ds = ds.stack(location=("y", "x"))
    n_rows = len(ds.location.to_numpy())
    with tb.open_file(meta_file, mode="a", root="/", driver="H5FD_CORE") as h5file:

        table = h5file.create_table(
            h5file.root,
            "meta",
            Particle,
            "meta",
            expectedrows=n_rows,
            byteorder="little",
        )
        data = [
            ds["latitude"].to_numpy(),
            ds["longitude"].to_numpy(),
            ds["elevation"].to_numpy(),
            ds["timezone"].to_numpy(),
            #  ds["landmask"].to_numpy(),
            #  ds["fed_state_protected"].to_numpy(),
            # ds["wind_land_cover_urban_medium_high"].to_numpy(),
            #  ds["pv_land_cover_urban_high"].to_numpy(),
            #  ds["wind_slope_over_20p"].to_numpy(),
            #  ds["pv_slope_over_5p"].to_numpy()
        ]
        table.append(data)
        table.flush()
        h5file.close()


def append_to_h5_wrapper(
    ds: xr.Dataset,
    res_file: str = UNSET,
    var_list: list[str] = UNSET,
    wrf: bool = False,
) -> None:
    """
    Append resource data and time index to HDF5 file.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing resource data.
    res_file : str, optional
        Path to HDF5 file.
    var_list : list of str, optional
        List of variable names to append.
    wrf : bool, optional
        If True, parse WRF time format.

    Returns
    -------
    None
    """
    ds = ds.transpose("time", "y", "x")
    ds = ds.stack(location=("y", "x"))
    if wrf:
        ds["time"] = [
            datetime.strptime(s.decode(), "%Y-%m-%d_%H:%M:%S")
            for s in ds.time.to_numpy()
        ]
    ds["time_index"] = ds["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").astype("|S30")
    _append_time(ds.time_index.to_numpy(), res_file=res_file)
    # xr.ds.apply() expects to output a dataset
    # so we spoof one here
    _ = ds.apply(_append_data_vars, res_file=res_file, var_list=var_list)
    return print("done appending resource data")


def _append_time(time_index: Any, res_file: str = UNSET) -> None:
    """
    Append time index to HDF5 file.

    Parameters
    ----------
    time_index : array-like
        Array of time index strings.
    res_file : str, optional
        Path to HDF5 file.

    Returns
    -------
    None
    """

    class Particle(tb.IsDescription):
        """
        PyTables description for time index table.

        Attributes
        ----------
        time_index : str
            Time index string.
        """

        time_index = tb.StringCol(itemsize=30, shape=(), dflt=b"")

    with tb.open_file(res_file, mode="r+", root="/", driver="H5FD_CORE") as h5file:

        table = h5file.create_table(
            h5file.root,
            "time_index",
            Particle,
            "time_index",
            expectedrows=8760,
            byteorder="little",
        )
        table.append(time_index)
        table.flush()
        h5file.close()


def _append_data_vars(
    dat: xr.DataArray, var_list: list[str] = UNSET, res_file: str = UNSET
) -> None:
    """
    Append data variables to HDF5 file.

    Parameters
    ----------
    dat : xarray.DataArray
        Data variable to append.
    var_list : list of str, optional
        List of variable names to append.
    res_file : str, optional
        Path to HDF5 file.

    Returns
    -------
    None
    """
    with tb.open_file(res_file, mode="r+", root="/", driver="H5FD_CORE") as h5file:
        varname = dat.name
        if varname in var_list:
            vardata = h5file.create_array(h5file.root, varname, dat.to_numpy(), varname)
            vardata.flush()
            h5file.close()
