"""
Convert wind resource data to HDF5 format for reV analysis.

This module processes wind resource data from zarr format and converts it
to HDF5 format suitable for renewable energy analysis using the reV toolkit,
including proper metadata handling and file organization.
"""

import json
import os
import shutil
import sys

import boto3
import numpy as np
import xarray as xr
from pandas import HDFStore
from rex import Outputs

sys.path.append(os.path.expanduser("/shared/renewable-profiles"))
datpath = sys.path[-1] + "/data/"

bucket = "wfclimres"
s3 = boto3.client("s3")


def wind_resource_to_h5(rank: int, domain: str, sim_id: str) -> None:
    """
    Converts wind power resource data to HDF5 format for a specified rank, domain, and simulation.

    The function selects a single year of data based on the 'rank' (from a hardcoded
    range of 1980-2098) and converts the time coordinate to a 'noleap' calendar
    before processing. The resulting data is saved to disk.

    Parameters
    ----------
    rank : int
        Index used to select a single year from the range 1980-2098 (inclusive).
    domain : str
        Spatial domain identifier (e.g., 'd02', 'd03').
    sim_id : str
        Simulation identifier (e.g., 'ERA5', 'MIROC6').

    Returns
    -------
    None
        The processed wind resource data is saved to a file in HDF5 format.
    """
    datpath = "/data/"
    meta_file = datpath + f"{domain}_meta_base.h5"

    def year_rank(rank=rank):
        """
        Retrieves a year as a string based on the 'rank' index.

        The available years span from 1980 to 2098 inclusive.

        Parameters
        ----------
        rank : int, optional
            Index to select the year from the 'rank' index.

        Returns
        -------
        str
            The year corresponding to the given rank.
        """
        years = [str(y) for y in np.arange(1980, 2099)]
        return years[rank]

    my_year = year_rank(rank)
    tstr = str(my_year)

    # open annual zarr from s3
    s3_path = f"s3://{bucket}/era/resource_data/{domain}/wind/{sim_id}"
    zarr_name = f"zarrs/{sim_id}_{domain}_wind_resource_{tstr}.zarr"
    zarr_path = f"{s3_path}/{zarr_name}"
    print(zarr_path)
    ds = xr.open_zarr(zarr_path)

    meta_vars = [
        "latitude",
        "longitude",
        "elevation",
        "landmask",
        "x",
        "y",
        "timezone",
        "time_index",
    ]
    var_list = [v for v in ds.data_vars if v not in meta_vars]

    if rank < 35:  # 1981-2014
        analysis_years = [int(n + 1980) for n in np.arange(1, 34)]
        edit_flag = rank == 1
    elif 35 <= rank <= 64:  # 2015-2044
        analysis_years = [int(n + 1980) for n in np.arange(35, 65)]
        edit_flag = rank == 35
    elif 65 <= rank <= 94:  # 2045-2074
        analysis_years = [int(n + 1980) for n in np.arange(65, 95)]
        edit_flag = rank == 65
    elif rank >= 95:  # 2072-2099
        analysis_years = [int(n + 1980) for n in np.arange(95, 119)]
        edit_flag = rank == 95

    outpath = f"/data/{sim_id}"

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    res_file = f"{outpath}/{sim_id}_{domain}_wind_resource_{tstr}.h5"
    print(
        f"{res_file} made!",
        # flush=True, file=f
    )

    ds["time_index"] = ds["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").astype("|S30")

    if os.path.exists(res_file):
        os.remove(res_file)
    # copy meta node file and attach resource data
    hdf = HDFStore(meta_file, mode="r", root="/")
    meta = hdf.meta
    with Outputs(res_file, "w") as h5:
        h5["meta"] = meta
        h5["time_index"] = ds["time_index"].to_numpy()

    ds = ds.stack(location=("y", "x"))
    for var in var_list:
        var_data = ds[var].to_numpy()

        # Check for nans a priori
        if np.isnan(var_data).any():
            with open(f"out/nans_{sim_id}_{domain}_{rank+1980}.txt", "w") as f:
                f.write(
                    f"NaNs found on rank: {rank}\tyear: {rank + 1980}\tdomain: {domain}\t sim_id: {sim_id}"
                )

        Outputs.add_dataset(
            h5_file=res_file,
            dset_name=var,
            dset_data=ds[var].to_numpy(),
            dtype=ds[var].dtype,
        )

    # make new config files for the windpower runs
    if edit_flag:
        for module in ["onshore", "offshore"]:
            modpath = outpath + f"/{module}_{sim_id}"
            if not os.path.exists(modpath):
                os.makedirs(modpath)
            gen_file = sys.path[-1] + f"/rev_configs/wind_{module}_config_gen.json"
            collect_file = (
                sys.path[-1] + f"/rev_configs/wind_{module}_config_collect.json"
            )
            pipeline_file = (
                sys.path[-1] + f"/rev_configs/wind_{module}_config_pipeline.json"
            )
            pp_file = (
                sys.path[-1] + f"/rev_configs/wind_{domain}_{module}_projectpoints.csv"
            )
            sam_file = (
                sys.path[-1] + f"/rev_configs/{module}_windturbines_{domain}.json"
            )
            run_gen = f"{modpath}/wind_{module}_config_gen.json"
            shutil.copy(gen_file, run_gen)
            with open(gen_file) as fg:
                dat = json.load(fg)
            dat["analysis_years"] = analysis_years
            dat["resource_file"] = (
                f"{outpath}/{sim_id}_{domain}_wind_resource_" + "{}.h5"
            )
            dat["project_points"] = f"{pp_file}"
            dat["sam_files"] = {"def": sam_file}
            with open(run_gen, "w") as fp:
                json.dump(dat, fp, indent=4)
            dat = None
            run_collect = f"{modpath}/wind_{module}_config_collect.json"
            shutil.copy(collect_file, run_collect)
            with open(collect_file) as fc:
                dat = json.load(fc)
            dat["project_points"] = pp_file
            with open(run_collect, "w") as fp:
                json.dump(dat, fp, indent=4)
            dat = None
            run_pipeline = f"{modpath}/wind_{module}_config_pipeline.json"
            shutil.copy(pipeline_file, run_pipeline)


if __name__ == "__main__":
    rank = int(sys.argv[1])
    domain = str(sys.argv[2])
    sim_id = str(sys.argv[3])
    wind_resource_to_h5(rank, domain, sim_id)
