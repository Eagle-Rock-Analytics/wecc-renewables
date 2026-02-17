"""
Module for processing renewable energy generation data from HDF5 files to Zarr format.

This module contains the primary function, 'gen_h5_to_zarr', which handles the
conversion of hourly reV (Renewable Energy Potential Model) output HDF5 generation
files into ungridded Zarr stores. The resulting Zarr data is saved to a
temporary S3 bucket location for subsequent processing stages.

Key Functionality:
- Reads HDF5 generation files using xarray/h5netcdf.
- Cleans data by dropping metadata variables.
- Dynamically constructs S3 Zarr path based on domain, simulation, module, and technology.
- Saves the processed xarray Dataset to a Zarr store using an external utility
  ('ds_to_zarr').
- Includes command-line interface for execution with necessary parameters
  (domain, simulation ID, module, technology, and year).
"""

import sys

import xarray as xr

from src.utils import ds_to_zarr


def gen_h5_to_zarr(domain: str, sim_id: str, module: str, tech: str, year: str) -> None:
    """
    Saves reV output h5 generation files as ungridded zarrs in temporary bucket location.

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

    Returns
    -------
    None
        The function saves the resulting dataset to a zarr store.
    """

    bucket = "wfclimres"
    tstr = f"{year}"

    tech_dict = {"PVWattsv8": "pv", "Windpower": "wind"}
    tech_sname = tech_dict[tech]

    fname = f"/data/{sim_id}/{module}_{sim_id}/{module}_{sim_id}_generation_{tstr}.h5"

    ds = xr.open_dataset(fname, engine="h5netcdf", phony_dims="sort")
    ds = ds.drop(["meta"])

    if "redo_" in module:
        module = module.replace("redo_", "")
    zarr_path = f"era/tmp/ungridded/{tech}/{domain}/{module}/{sim_id}/"
    save_name = f"{tech_sname}_{module}_{sim_id}_{tstr}"

    ds_to_zarr(ds, zarr_path, save_name, bucket=bucket)

    filepath_zarr = f"s3://{bucket}/{zarr_path}{save_name}.zarr/"
    print(filepath_zarr)
    ds = xr.open_dataset(
        filepath_zarr,
        engine="zarr",
        consolidated=False,
    )


if __name__ == "__main__":
    domain = str(sys.argv[2])
    sim_id = str(sys.argv[3])
    module = str(sys.argv[4])
    tech = str(sys.argv[5])
    year = int(sys.argv[6])
    gen_h5_to_zarr(domain, sim_id, module, tech, year)
