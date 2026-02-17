"""
Convert PV resource data to HDF5 format.

This module processes solar radiation and meteorological data from intake-esm catalogs,
converts them to HDF5 format suitable for PV resource analysis, and handles various
scenarios and time periods.
"""

import json
import os
import shutil
import sys

import boto3
import intake
import numpy as np
from pandas import HDFStore
from rex import Outputs

from src.utils import ds_to_zarr
from src.zarr_to_pv_resource import (
    preprocess_pv_wrapper,
)

s3 = boto3.client("s3")


def pv_resource_to_h5(rank: int, domain: str, sim_id: str) -> None:
    """
    Converts Photovoltaic (PV) resource data to HDF5 format for a specified rank, domain, and simulation.

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
        The processed PV resource data is saved to a file in HDF5 format.
    """
    datpath = "/data/"  # path to data
    meta_file = datpath + f"{domain}_meta_base.h5"

    # preprocess functions to map job array rank to individual years
    def preprocess_esm(ds):
        """
        Selects a single year of data based on the `rank` and
        converts the time coordinate to a 'noleap' calendar.

        Parameters
        ----------
        ds : xarray.Dataset
            Input Dataset with a 'time' coordinate to be subset and modified.

        Returns
        -------
        xarray.Dataset
            The Dataset subsetted to a single year, with 'noleap' calendar,
            and singleton dimensions squeezed.
        """
        year = _year_rank()
        ds = ds.copy()
        ds = ds.sel(time=year)
        ds = ds.convert_calendar("noleap", use_cftime=True)
        return ds.squeeze()

    def _year_rank(rank=rank):
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

    # Done with preprocess functions

    # Open catalog of available data sets using intake-esm package
    cat = intake.open_esm_datastore(
        "https://cadcat.s3.amazonaws.com/cae-collection.json"
    )
    scen = "reanalysis" if sim_id == "ERA5" else "historical" if rank < 34 else "ssp370"
    var_list = ["t2", "u10", "v10", "swdnb", "swupb", "swddif", "swddni", "snownc"]
    # form query dictionary
    query = {
        "activity_id": "WRF",
        "source_id": sim_id,
        "institution_id": "UCLA",
        "experiment_id": scen,
        "variable_id": var_list,
        "table_id": "1hr",
        "grid_label": domain,
    }
    # subset catalog and get some metrics grouped by 'source_id'
    cat_subset = cat.search(require_all_on=["source_id"], **query)
    dsets = cat_subset.to_dataset_dict(
        xarray_open_kwargs={"consolidated": True},
        storage_options={"anon": True},
        progressbar=False,
        preprocess=preprocess_esm,
    )
    ds = list(dsets.values())[0].squeeze()
    tstr = str(ds.time.dt.year.to_numpy()[0])
    ds = preprocess_pv_wrapper(ds).compute()
    ds["time_index"] = ds["time"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").astype("|S30")

    # and send to zarr
    zarr_path = f"era/resource_data/{domain}/pv/{sim_id}/"
    zarr_name = f"{sim_id}_{domain}_pv_resource_{tstr}"
    bucket = "wfclimres"
    ds_to_zarr(ds, zarr_path, zarr_name, bucket=bucket)

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

    # we stick with 30-ish year time batches
    # for performance and progress tracking reasons
    if rank < 35:  # 1981-2013
        analysis_years = [int(n + 1980) for n in np.arange(1, 35)]
        edit_flag = rank == 1
    elif 35 < rank <= 64:  # 2015-2044
        analysis_years = [int(n + 1980) for n in np.arange(35, 65)]
        edit_flag = rank == 41
    elif 65 <= rank <= 94:  # 2045-2074
        analysis_years = [int(n + 1980) for n in np.arange(65, 95)]
        edit_flag = rank == 65
    elif rank >= 95:  # 2075-2099
        analysis_years = [int(n + 1980) for n in np.arange(96, 120)]
        edit_flag = rank == 95

    outpath = f"/data/{sim_id}"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    res_file = f"{outpath}/{sim_id}_{domain}_pv_resource_{tstr}.h5"
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
        Outputs.add_dataset(
            h5_file=res_file,
            dset_name=var,
            dset_data=ds[var].to_numpy(),
            dtype=ds[var].dtype,
        )

    # make new config files for the PVWatts runs
    if edit_flag:
        for module in ["distributed", "utility"]:
            modpath = outpath + f"/{module}_{sim_id}"
            if not os.path.exists(modpath):
                os.makedirs(modpath)
            gen_file = sys.path[-1] + f"/rev_configs/pv_{module}_config_gen.json"
            collect_file = (
                sys.path[-1] + f"/rev_configs/pv_{module}_config_collect.json"
            )
            pipeline_file = (
                sys.path[-1] + f"/rev_configs/pv_{module}_config_pipeline.json"
            )
            pp_file = (
                sys.path[-1] + f"/rev_configs/pv_{domain}_{module}_projectpoints.csv"
            )
            sam_file = sys.path[-1] + f"/rev_configs/{module}_pvwattsv8.json"
            run_gen = f"{modpath}/pv_{module}_config_gen.json"
            shutil.copy(gen_file, run_gen)
            with open(gen_file) as fg:
                dat = json.load(fg)
            dat["analysis_years"] = analysis_years
            dat["resource_file"] = f"{outpath}/{sim_id}_{domain}_pv_resource_" + "{}.h5"
            dat["project_points"] = pp_file
            dat["sam_files"] = {"def": sam_file}
            with open(run_gen, "w") as fp:
                json.dump(dat, fp, indent=4)
            dat = None
            run_collect = f"{modpath}/pv_{module}_config_collect.json"
            shutil.copy(collect_file, run_collect)
            with open(collect_file) as fc:
                dat = json.load(fc)
            dat["project_points"] = pp_file
            with open(run_collect, "w") as fp:
                json.dump(dat, fp, indent=4)
            dat = None
            run_pipeline = f"{modpath}/pv_{module}_config_pipeline.json"
            shutil.copy(pipeline_file, run_pipeline)


if __name__ == "__main__":
    rank = int(sys.argv[1])
    domain = str(sys.argv[2])
    sim_id = str(sys.argv[3])
    pv_resource_to_h5(rank, domain, sim_id)
