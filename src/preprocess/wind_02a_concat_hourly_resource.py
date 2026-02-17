"""
Concatenate hourly wind resource data into annual datasets.

This module processes hourly wind resource data stored in zarr format,
concatenating them into annual datasets and uploading the results to S3
for efficient access and analysis.
"""

import argparse
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar


def hourly_wind_resource_to_annual(year: int, domain: str, sim_id: str) -> None:
    """
    Makes annual wind resource zarrs from the locally stored hourly wind resource zarrs
    then ships off the new annual zarrs to S3.

    Parameters
    ----------
    year : int
        A single year from the range 1981-2098 (inclusive).
    domain : str
        Spatial domain identifier (e.g., 'd02', 'd03').
    sim_id : str
        Simulation identifier (e.g., 'ERA5', 'MIROC6').

    Returns
    -------
    None
        The processed wind resource data is saved to zarr in S3 bucket.
    """
    datpath = "/data/"
    datpath + f"{domain}_meta_base.h5"

    # if sim_id == "ERA5":
    #    scen = "reanalysis"
    # else:
    #    if year < 2014:
    #        scen = "historical"
    #    else:
    #        scen = "ssp370"

    print("Getting started...", flush=True)

    # open and concatenate here
    # hourly_regex = f"/data/wind_resources/hourly/{domain}/{sim_id}/{scen}/auxhist_d01_{year}-*.zarr"
    hourly_regex = f"/data/wind_resources/hourly/{domain}/{sim_id}/{year}/auxhist_d01_{year}-*.zarr"
    hourly_paths = np.sort(glob.glob(hourly_regex))

    print(
        f"Concatenating {len(hourly_paths)} files for {sim_id} : {domain} : {year}",
        flush=True,
    )

    # Chunks depending on the domain
    chunks = {
        "d02": {"time": 1, "y": 340, "x": 270},
        # "d03": {"time": 1, "y": 41, "x": 27},
        "d03": {"time": 1, "y": 492, "x": 243},
    }
    chunks_int = {
        "d02": (1, 340, 270),
        # "d03": (1, 41, 27),
        "d03": (1, 492, 243),
    }

    with ProgressBar():
        ds = xr.open_mfdataset(
            hourly_paths,
            engine="zarr",
            # combine="by_coords",
            # consolidated=True,
            combine="nested",
            concat_dim="time",
            consolidated=True,
            # consolidated=False,
            decode_cf=False,
            parallel=True,
            chunks=chunks[domain],
            drop_variables=["lat", "lon"],
        )

    ds["timezone"] = ds["timezone"].transpose("time", "y", "x")

    for var in ds.data_vars:
        ds[var].encoding["chunks"] = chunks_int[domain]
        ds[var].encoding["preferred_chunks"] = chunks[domain]

    ds = xr.decode_cf(ds)
    ds = ds.sel(time=~((ds.time.dt.month == 2) & (ds.time.dt.day == 29)))
    ds = ds.convert_calendar("noleap", use_cftime=True)
    ds = ds.astype(np.float32, casting="same_kind")

    # Rechunk
    ds = ds.chunk(chunks[domain])

    print("files loaded! Checking length...", flush=True)

    # check for missing time steps
    target_time = pd.date_range(f"{year}-01-01", end=f"{year}-12-31 23:00:00", freq="h")
    target_time = target_time[~((target_time.month == 2) & (target_time.day == 29))]

    if len(ds.time.to_numpy() != len(target_time)):
        err_f = f"missing_times/{sim_id}_{year}_missing_times.txt"
        with open(err_f, "w") as f:
            f.write("Missing time steps:\n")
            ds_time_formatted = ds.time.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
            target_time_formatted = target_time.strftime("%Y-%m-%d %H:%M:%S").to_numpy()
            set1 = set(target_time_formatted)
            set2 = set(ds_time_formatted)
            missing = sorted(set1 - set2)
            for val in missing:
                # set to be consistent with file naming conventions
                fmt_val = datetime.strptime(val, "%Y-%m-%d %H:%M:%S").strftime(
                    "%Y-%m-%d_%H:%M:%S"
                )
                f.write(fmt_val + "\n")

    print("Uploading to s3", flush=True)

    # and zarrify
    destination_path = f"era/resource_data/{domain}/wind/{sim_id}/zarrs"
    zarr_name = f"{sim_id}_{domain}_wind_resource_{year}.zarr/"
    bucket = "wfclimres"

    try:
        filepath_zarr = f"s3://{bucket}/{destination_path}/{zarr_name}"
        with ProgressBar():
            ds.to_zarr(filepath_zarr, mode="w", consolidated=True)
        print(f"{filepath_zarr} zarr sent to s3.", flush=True)
    except Exception as _:
        filepath_zarr = f"/data/wind_resources/yearly/{domain}/{sim_id}/{zarr_name}"
        with ProgressBar():
            ds.to_zarr(filepath_zarr, mode="w", consolidated=True)
        print(f"{filepath_zarr} zarr written localy.", flush=True)

    # for file in hourly_paths:
    #    shutil.rmtree(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="", description="", epilog="")
    parser.add_argument("-m", "--model", default="EC-Earth3", help="", type=str)
    parser.add_argument("-d", "--domain", default="d02", help="", type=str)
    parser.add_argument("-r", "--rank", default="1", help="", type=int)
    parser.add_argument("-y", "--year", default="1981", help="", type=int)

    # Parse arguments
    args = parser.parse_args()
    domain = args.domain
    year = args.year
    sim_id = args.model

    hourly_wind_resource_to_annual(year, domain, sim_id)
