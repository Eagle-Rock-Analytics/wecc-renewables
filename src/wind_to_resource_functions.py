"""
Utility functions for converting WRF output to wind resource datasets.

This module centralizes common preprocessing helpers used by the wind pipelines:
file bookkeeping, WRF variable transformations, and output dataset cleanup.

Notes
-----
These functions are designed to operate on WRF NetCDF inputs (local or S3) and
produce xarray objects ready for zarr serialization. The utilities are
intentionally lightweight and avoid project-specific orchestration logic.

"""

import os
import re
import tempfile

import numpy as np
import pandas as pd
import s3fs
import wrf
import xarray as xr
from netCDF4 import Dataset

# Ensure local scratch directories exist for temporary outputs.
dirs = ["./out", "./output_data", "./tmp"]
for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# Initialize filesystem object to interact with S3 storage.
fs = s3fs.S3FileSystem()


def count_all_paths(folder: str) -> int:
    """
    Count total child paths (files + directories) beneath a folder.

    Parameters
    ----------
    folder : str
        Root directory to walk.

    Returns
    -------
    int
        Count of all child files and directories.

    Notes
    -----
    Hidden files are included; symlinks are counted if present in the walk.
    """
    count = 0
    for _, dirnames, filenames in os.walk(folder):
        count += len(dirnames) + len(filenames)
    return count


def directory_size_bytes(path: str) -> int:
    """
    Compute total size of a directory tree in bytes.

    Parameters
    ----------
    path : str
        Directory to measure.

    Returns
    -------
    int
        Total size in bytes for all files found under ``path``.

    Notes
    -----
    Files that disappear or cannot be accessed are skipped.
    """
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            file_path = os.path.join(root, name)
            try:
                total += os.path.getsize(file_path)
            except OSError:
                # Skip files that are removed or inaccessible during the walk.
                continue
    return total


def list_year_zarr_files(directory: str, year: str) -> list[str]:
    """
    List zarr files for a specific year in directory.

    Parameters
    ----------
    directory : str
        Directory path to search.
    year : str
        Year to filter files by.

    Returns
    -------
    list[str]
        Zarr directory paths matching the year.

    Notes
    -----
    This scans only the immediate directory level.
    """
    pattern = re.compile(rf"^auxhist_d01_{year}-.*\.zarr$")
    with os.scandir(directory) as entries:
        return [
            entry.path
            for entry in entries
            if entry.is_dir() and pattern.match(entry.name)
        ]


def load_using_nc4(
    path: str, variables: list[str] | None = None, attributes: list[str] | None = None
) -> Dataset:
    """
    Load specified variables and attributes from a NetCDF file into a netCDF4.Dataset object.

    Parameters
    ----------
    path : str
        File path to a local NetCDF file.
    variables : list of str
        Names of variables to extract from dataset.
    attributes : list of str
        Names of dataset attributes to retain.

    Returns
    -------
    netCDF4.Dataset
        Dataset containing requested variables and attributes.

    Notes
    -----
    This helper expects both ``variables`` and ``attributes`` to be provided.
    The returned dataset is backed by a temporary file and should be closed by
    the caller when finished.
    """
    # Select variables from the dataset and load them into memory.
    if variables and attributes:
        tmp_ds = xr.open_dataset(path)[variables].load()
        attrs = tmp_ds.attrs
        tmp_ds.attrs = {}
        tmp_ds.attrs = {k: v for k, v in attrs.items() if k in attributes}
        # Download dataset from S3 URL and load specified variables and attributes using a temporary file.
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=True) as tmp1:
            # save to temp file and load using netCDF4
            tmp_ds.to_netcdf(tmp1.name)
            new_ds = Dataset(tmp1.name, mode="r+")

    return new_ds


def download_read_using_nc4(
    url: str, variables: list[str] | None = None, attributes: list[str] | None = None
) -> Dataset:
    """
    Download a NetCDF file from S3 and load specified variables and attributes.

    Parameters
    ----------
    url : str
        URL path to an S3-hosted NetCDF file.
    variables : list of str
        Names of variables to extract from dataset.
    attributes : list of str
        Names of dataset attributes to retain.

    Returns
    -------
    netCDF4.Dataset
        Dataset downloaded and loaded from provided URL.

    Notes
    -----
    This helper expects both ``variables`` and ``attributes`` to be provided.
    The returned dataset is backed by a temporary file and should be closed by
    the caller when finished.
    """
    # Loop over variables and attributes
    if variables and attributes:
        # Download dataset from S3 URL and load specified variables and attributes using a temporary file.
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=True) as tmp0:
            fs.get(url, tmp0.name)
            # Use xarray to load and select data
            tmp_ds = xr.open_dataset(tmp0.name)[variables].load()
            attrs = tmp_ds.attrs
            tmp_ds.attrs = {}
            tmp_ds.attrs = {k: v for k, v in attrs.items() if k in attributes}
            # save to temp file and load using netCDF4
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=True) as tmp1:
                tmp_ds.to_netcdf(tmp1.name)
                new_ds = Dataset(tmp1.name, mode="r+")

    return new_ds


def create_height_variable(ds: Dataset) -> Dataset:
    """
    Create and initialize a HEIGHT variable in a NetCDF dataset based on existing temperature dimensions.

    Parameters
    ----------
    ds : netCDF4.Dataset
        Input dataset containing atmospheric model data.

    Returns
    -------
    netCDF4.Dataset
        Initialized HEIGHT variable within the dataset.

    Notes
    -----
    The variable is created in-place on ``ds`` and filled with NaNs.
    """
    # Extract temperature to use as a template
    t_var = ds.variables["T"]
    # Extract dimensions
    dims = t_var.dimensions
    shape = tuple(ds.dimensions[dim].size for dim in dims)
    # Create a HEIGHT variable in the raw dataset.
    height_var = ds.createVariable("HEIGHT", "f4", dimensions=dims, fill_value=np.nan)
    # Assign attributes
    height_var.description = "Height in meters from geopotential and terrain elevation"
    height_var.units = "m"
    height_var.long_name = "Height in meters"
    # Fill variable with nans
    height_var[:] = np.full(shape, np.nan, dtype="f4")

    return height_var


def concat_base_perturbations(raw_ds: Dataset, base_ds: Dataset) -> Dataset:
    """
    Concatenate base state perturbations into a raw WRF dataset.

    Parameters
    ----------
    raw_ds : netCDF4.Dataset
        Raw dataset from WRF output containing atmospheric variables.
    base_ds : netCDF4.Dataset
        Base state dataset used to calculate perturbations.

    Returns
    -------
    netCDF4.Dataset
        Updated dataset including base + perturbations.

    Notes
    -----
    The input ``raw_ds`` is modified in-place and returned for convenience.
    """
    # Physics constants
    g = 9.81

    # Update values in raw_ds using base + perts
    raw_ds.variables["U"][:] = raw_ds.variables["U"][:]
    raw_ds.variables["V"][:] = raw_ds.variables["V"][:]
    raw_ds.variables["P"][:] = base_ds.variables["PB"][:] + raw_ds.variables["P"][:]
    raw_ds.variables["PH"][:] = base_ds.variables["PHB"][:] + raw_ds.variables["PH"][:]
    raw_ds.variables["T"][:] = base_ds.variables["T00"][:] + wrf.tk(
        raw_ds.variables["P"][:], raw_ds["T"], units="K"
    )
    # Create height variable and fill it with actual height values
    raw_ds.variables["HEIGHT"] = create_height_variable(raw_ds)
    raw_ds.variables["HEIGHT"][:] = wrf.destagger(
        raw_ds["PH"][:] / g - base_ds.variables["HGT"], -3
    )
    # Save lat,lon from base to raw
    raw_ds.variables["XLAT"] = base_ds.variables["XLAT"]
    raw_ds.variables["XLONG"] = base_ds.variables["XLONG"]

    return raw_ds


def create_height(raw_ds: Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute staggered HEIGHT arrays for U and V wind components from a WRF dataset.

    Parameters
    ----------
    raw_ds : netCDF4.Dataset
        Raw dataset from WRF output containing atmospheric variables.

    Returns
    -------
    tuple of np.ndarray
        Interpolated staggered HEIGHT arrays for U and V components.

    Notes
    -----
    A temporary NetCDF file is used to preserve WRF metadata for interpolation.
    """
    # Use a temporary NetCDF to preserve WRF metadata for interpolation
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=True) as tmp:
        dst = Dataset(tmp.name, "w", format="NETCDF4")
        for name in raw_ds.ncattrs():
            dst.setncattr(name, raw_ds.getncattr(name))
        for name, dimension in raw_ds.dimensions.items():
            dst.createDimension(
                name, len(dimension) if not dimension.isunlimited() else None
            )
        # Save necessary variables for the height interpolation
        names = ["HEIGHT", "U", "V"]
        for name in names:
            variable = raw_ds.variables[name]
            x = dst.createVariable(name, variable.datatype, variable.dimensions)
            x.setncatts({k: variable.getncattr(k) for k in variable.ncattrs()})
            x[:] = variable[:]
        dst.close()
        da = xr.open_dataset(tmp.name)[names].load()
    # Compute staggered height in u and v coordinates using xarray
    height_u = da["HEIGHT"].interp(west_east=da["U"].west_east_stag).to_numpy()
    height_v = da["HEIGHT"].interp(south_north=da["V"].south_north_stag).to_numpy()

    return (height_u, height_v)


def interp_uv_to_height(raw_ds: Dataset) -> tuple[Dataset, np.ndarray]:
    """
    Interpolate U and V wind variables to specified vertical height levels.

    Parameters
    ----------
    raw_ds : netCDF4.Dataset
        Raw dataset from WRF output containing atmospheric variables.

    Returns
    -------
    tuple
        Dataset with updated raw dataset and the vertical level values in meters.

    Notes
    -----
    The input ``raw_ds`` is modified in-place (U/V overwritten with interpolated values).
    """
    # Create a height levels array (meters) for interpolation targets
    levels = np.arange(30, 420, 10)
    # Calculate staggered heights for U and V
    height_u, height_v = create_height(raw_ds)
    # Interpolate U and V to height levels
    U_z = wrf.interplevel(raw_ds["U"][:], height_u[:], levels, squeeze=False)
    V_z = wrf.interplevel(raw_ds["V"][:], height_v[:], levels, squeeze=False)
    # Overwrite U/V to keep downstream wind rotation logic consistent
    raw_ds.variables["U"][:] = U_z.to_numpy()
    raw_ds.variables["V"][:] = V_z.to_numpy()

    return (raw_ds, levels)


def get_uvmet_wspd_wdir(raw_ds: Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Compute earth-rotated wind speed and direction from U and V components.

    Parameters
    ----------
    raw_ds : netCDF4.Dataset
        Raw dataset from WRF output containing atmospheric variables.

    Returns
    -------
    tuple of xarray.DataArray
        Wind speed and direction DataArrays.

    Notes
    -----
    Uses ``wrf.g_uvmet.get_uvmet_wspd_wdir`` to rotate winds to earth coordinates.
    """
    # Use wrf.g_uvmet.get_uvmet_wspd_wdir to calculate wind speed and direction
    raw_wdws = wrf.g_uvmet.get_uvmet_wspd_wdir(raw_ds)
    # The wind speed and wind direction for the wind rotated to earth coordinates,
    # whose leftmost dimensions is 2 (0=WSPD_EARTH, 1=WDIR_EARTH)
    raw_ws = raw_wdws.isel(wspd_wdir=0)
    raw_wd = raw_wdws.isel(wspd_wdir=1)

    return (raw_ws, raw_wd)


def create_wind_resources(
    raw_ws: xr.DataArray, raw_wd: xr.DataArray, zlev: int, level: int
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Create wind speed and direction DataArrays at a specified height level.

    Parameters
    ----------
    raw_ws : xarray.DataArray
        Wind speed.
    raw_wd : xarray.DataArray
        Wind direction.
    zlev : int
        Index representing vertical level in dataset.
    level : int
        Height in meters at which variables are interpolated or extracted.

    Returns
    -------
    tuple of xarray.DataArray
        Wind speed and direction at specified height.

    Notes
    -----
    ``zlev`` should index the height level within ``raw_ws``/``raw_wd``.
    """
    # Select vertical level data
    out_ws = (
        raw_ws.astype(np.float32)
        .isel(bottom_top=zlev)
        .drop_vars(["wspd_wdir", "Time"])
        .squeeze()
    )
    out_wd = (
        raw_wd.astype(np.float32)
        .isel(bottom_top=zlev)
        .drop_vars(["wspd_wdir", "Time"])
        .squeeze()
    )
    # Fill attributes
    out_ws.name = f"windspeed_{level}m"
    out_ws.attrs["description"] = f"earth rotated wspd at {level} m"
    out_ws.attrs["units"] = "m s-1"
    out_wd.name = f"winddirection_{level}m"
    out_wd.attrs["description"] = f"earth rotated wdir at {level} m"
    out_wd.attrs["units"] = "deg"

    return (out_ws, out_wd)


def create_temperature_resources(
    raw_ds: Dataset, ws: xr.DataArray, zlev: int, level: int
) -> xr.DataArray:
    """
    Create air temperature DataArray at a specified height level from WRF dataset.

    Parameters
    ----------
    raw_ds : netCDF4.Dataset
        Raw dataset from WRF output containing atmospheric variables.
    ws : xarray.DataArray
        Wind speed DataArray.
    zlev : int
        Index representing vertical level in dataset.
    level : int
        Height in meters at which variables are interpolated or extracted.

    Returns
    -------
    xarray.DataArray
        Temperature DataArray at specified height.

    Notes
    -----
    Temperatures are converted from Kelvin to Celsius.
    """
    # Define reference slice for dataset subsetting
    subset = (slice(None), zlev, slice(None), slice(None))
    # Extract sliced dataset and convert from K to C
    output_t = raw_ds.variables["T"][subset].squeeze() - 273.15
    # Create new xarray.DataArray
    out_t = xr.ones_like(ws.astype(np.float32))
    # Fill with extracted height temperature
    out_t.values = output_t.astype(np.float32)
    out_t = out_t.squeeze()
    # Add attributes
    out_t.name = f"temperature_{level}m"
    out_t.attrs["description"] = f"Air temperature at {level} m"
    out_t.attrs["units"] = "C"

    return out_t


def create_pressure_resources(
    raw_ds: Dataset, ws: xr.DataArray, zlev: int, level: int
) -> xr.DataArray:
    """
    Create air pressure DataArray at a specified height level from WRF dataset.

    Parameters
    ----------
    raw_ds : netCDF4.Dataset
        Raw dataset from WRF output containing atmospheric variables.
    ws : xarray.DataArray
        Wind speed DataArray.
    zlev : int
        Index representing vertical level in dataset.
    level : int
        Height in meters at which variables are interpolated or extracted.

    Returns
    -------
    xarray.DataArray
        Pressure DataArray at specified height.

    Notes
    -----
    Pressures are converted from Pascals to atmospheres.
    """
    # Pressure conversion from Pa to Atm
    Pa_in_atm = 101325
    # Define reference slice for dataset subsetting; handle surface pressure
    if zlev == 0:
        var_name = "PSFC"
        subset = (slice(None), slice(None), slice(None))
    else:
        var_name = "P"
        subset = (slice(None), zlev, slice(None), slice(None))

    # Extract sliced pressure data and convert to Atm
    output_p = raw_ds.variables[var_name][subset].squeeze() / Pa_in_atm
    # Create new xarray.DataArray
    out_p = xr.ones_like(ws.astype(np.float32))
    # Fill data with height interp pressure
    out_p.values = output_p.astype(np.float32)
    out_p = out_p.squeeze()
    # Add attributes
    out_p.name = f"pressure_{level}m"
    out_p.attrs["description"] = f"Air pressure at {level} m"
    out_p.attrs["units"] = "atm"

    return out_p


def process_output_zarr(ds: xr.Dataset, file_name: str) -> xr.Dataset:
    """
    Process and prepare a dataset for output, adding coordinates and cleaning attributes.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing computed resources
    file_name : str
        Original WRF filename or path string containing date/time information.

    Returns
    -------
    xarray.Dataset
        Processed dataset ready for output.

    Notes
    -----
    Expects the file name to include timestamp tokens separated by underscores.
    """
    # Reassign lat,lon coordinates
    ds = ds.assign_coords(lat=ds.XLAT, lon=ds.XLONG)
    # Rename dimensions
    ds = ds.rename_dims({"south_north": "y", "west_east": "x"})
    # Drop original lat,lon vars
    ds = ds.drop_vars(["XLONG", "XLAT"])
    # Add time zone variable
    len_x = len(ds.x)
    len_y = len(ds.y)
    timezone = np.full((len_x, len_y), 0)
    ds["timezone"] = (["x", "y"], timezone)
    # Extract time from model file name
    time_str = file_name.split("/")[-1]
    time_dt = pd.to_datetime(time_str.split("_")[-2] + " " + time_str.split("_")[-1])
    ref_time = np.datetime64(pd.Timestamp("1980-01-01 00:00:00"))
    time_val = (time_dt - ref_time) / pd.Timedelta("1D")
    # Expand dimensions to include time
    ds = ds.expand_dims("time")
    ds["time"] = ("time", [time_val])
    # Assign attrs to time dimension to allow for concatenation down the line in pipeline
    ds["time"] = ds["time"].assign_attrs({"units": "days since 1980-01-01 00:00:00"})
    # Overwrite projection attribute to allow for serialization when saving file
    str(ds.attrs["projection"])
    attrs_to_delete = ["projection", "stagger", "FieldType", "MemoryOrder"]
    for attr in attrs_to_delete:
        if attr in ds.attrs:
            del ds.attrs[attr]
        for var in ds.data_vars:
            if attr in ds[var].attrs:
                del ds[var].attrs[attr]

    return ds
