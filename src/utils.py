"""
Utility functions for cloud storage and data format conversion.

This module provides functions for uploading files to S3 and converting
datasets to zarr format for efficient storage and access.
"""

import logging
import os
from typing import Union

import boto3
import xarray as xr
from botocore.exceptions import ClientError

# Set AWS stuff
s3 = boto3.resource("s3")
s3_cl = boto3.client("s3")  # for lower-level processes
bucket = "wfclimres"


def upload_file(
    file_name: str,
    destination_path: str,
    bucket: str = bucket,
    object_name: Union[str, None] = None,
) -> None:
    """
    Upload a file to an S3 bucket.

    Parameters
    ----------
    file_name : str
        File to upload.
    destination_path : str
        Bucket subpath.
    bucket : str, optional
        Bucket to upload to.
    object_name : str, optional
        S3 object name minus destination_path.
        If not specified then file_name is used.

    Returns
    -------
    None
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client("s3")
    object_name = f"{destination_path}{object_name}"
    try:
        s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return print("error uploading")


def ds_to_zarr(
    ds: xr.Dataset, destination_path: str, save_name: str, bucket: str = bucket
) -> None:
    """
    Convert netCDF dataset to zarr and send to S3 bucket.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to convert.
    destination_path : str
        S3 destination path.
    save_name : str
        Name for saved zarr file.
    bucket : str, optional
        S3 bucket name.

    Returns
    -------
    None
    """
    filepath_zarr = f"s3://{bucket}/{destination_path}{save_name}.zarr"
    ds = ds.chunk(chunks="auto")
    ds.to_zarr(store=filepath_zarr, mode="w")


def ds_to_pretty_zarr(
    ds: xr.Dataset,
    destination_path: str,
    save_name: str,
    bucket_name: str,
) -> None:
    """
    Convert netCDF dataset to zarr format with optimized chunking and encoding.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to convert.
    destination_path : str
        S3 destination path.
    save_name : str
        Name for saved zarr file.
    bucket_name : str
        S3 bucket name.

    Returns
    -------
    None
    """
    coord_to_encode = [
        coord
        for coord in ds.coords
        if coord
        not in [
            "x",
            "y",
            "time",
        ]
    ]

    filepath_zarr = f"s3://{bucket_name}/{destination_path}{save_name}"
    for var in ds:
        if "chunks" in ds[var].encoding:
            del ds[var].encoding["chunks"]
    for coord in coord_to_encode:
        if "chunks" in ds[coord].encoding:
            del ds[coord].encoding["chunks"]

    ds.time.attrs = {
        "standard_name": "time",
        "time_zone": "UTC",
    }
    ds = ds.transpose("time", "y", "x")
    ds = ds.chunk(chunks={"time": 8760, "y": 87, "x": 42})
    ds.to_zarr(
        store=filepath_zarr,
        mode="w",
    )


def preprocess_ungridded_zarrs(ds: xr.Dataset):
    """
    Renames phony_dim_0 and phony_dim_1 to the appropriate spatial
    or temporal nomenclature.
    """
    if len(ds["phony_dim_0"]) == 8760:
        rename_dim_dict = {"phony_dim_1": "gid", "phony_dim_0": "time"}
    else:
        rename_dim_dict = {"phony_dim_1": "time", "phony_dim_0": "gid"}
        ds = ds.rename_dims(rename_dim_dict)
        return ds
