#!/usr/bin/python3
"""
Create metadata node files for spatial datasets.

This module creates HDF5 metadata files containing spatial information
(coordinates, elevation, timezone) for renewable energy resource analysis.
It uses intake-esm to query datasets and extract spatial metadata.
"""

import os
import sys
from typing import Union

import intake
import s3fs
import xarray as xr

from src.ds_to_h5 import (
    meta_h5,
)
from src.utils import ds_to_zarr, upload_file


def spatial_h5_make(domain: str) -> None:
    """
    Create spatial metadata HDF5 file for a given domain.

    Parameters
    ----------
    domain : str
        WRF domain identifier.

    Returns
    -------
    None
        Saves HDF5 of spatial metadata for input to ReV.
    """
    # Open catalog of available data sets using intake-esm package
    # we only need one sample dataset, one time step for this

    # Define a preprocess function to select one time step
    def get_single_timestep(
        ds: Union[xr.Dataset, xr.DataArray],
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Selects and returns the first time step of an xarray Dataset or DataArray.

        The time dimension is removed via squeezing if it has size one.

        Parameters
        ----------
        ds : xarray.Dataset or xarray.DataArray
            Input Dataset or DataArray containing a 'time' dimension.

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            A copy of the input with only the first time step selected and
            singleton dimensions removed.
        """
        ds = ds.copy()
        return ds.isel(time=0).squeeze()

    cat = intake.open_esm_datastore(
        "https://cadcat.s3.amazonaws.com/cae-collection.json"
    )
    # form query dictionary
    query = {
        "activity_id": "WRF",
        "source_id": "CESM2",
        "experiment_id": "historical",
        "variable_id": "psfc",
        "table_id": "mon",
        "grid_label": domain,
    }
    # subset catalog and get some metrics grouped by 'source_id'
    cat_subset = cat.search(require_all_on=["source_id"], **query)
    dsets = cat_subset.to_dataset_dict(
        xarray_open_kwargs={"consolidated": True},
        storage_options={"anon": True},
        progressbar=False,
        preprocess=get_single_timestep,
    )
    ds = xr.merge(list(dsets.values()))

    # now add elevation, an essential meta variable
    fs = s3fs.S3FileSystem(anon=True)
    s3path = (
        "wrf-cmip6-noversioning/downscaled_products/"
        + f"wrf_coordinates/wrfinput_{domain}"
    )
    hgt_file = fs.open(s3path)
    hgt_ds = xr.open_dataset(
        hgt_file,
        engine="h5netcdf",
    )["HGT"]
    hgt_ds = hgt_ds.squeeze().rename({"south_north": "y", "west_east": "x"})
    hgt_ds = hgt_ds.drop(["XLAT", "XLONG"])
    ds["elevation"] = hgt_ds
    ds = ds.drop("psfc")

    # save the template zarr
    destination_path = f"inputs/{domain}/static_files/"
    ds_to_zarr(ds, destination_path, f"coord_ds_{domain}")
    # make the meta node
    meta_file = f"/data/meta_node_{domain}_base.h5"
    if os.path.exists(meta_file):
        os.remove(meta_file)
    meta_h5(ds, meta_file=meta_file)

    object_name = f"{domain}_meta_base.h5"
    upload_file(meta_file, destination_path, object_name=object_name)


if __name__ == "__main__":
    domain = str(sys.argv[1])
    spatial_h5_make(domain)
