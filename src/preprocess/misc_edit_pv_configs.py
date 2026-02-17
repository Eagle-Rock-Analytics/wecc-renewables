"""
Utility functions for editing PV configuration files.

This module provides functions to create and modify JSON configuration files
for PV resource analysis workflows, including reV configuration management
for different time periods and domains.
"""

import json
import os
import shutil
import sys

import boto3
import numpy as np

sys.path.append(os.path.expanduser("/shared/renewable-profiles"))
bucket = "wecc-renewables"
s3 = boto3.client("s3")


def make_pv_config(trange: str, domain: str, sim_id: str) -> None:
    """
    Create PV configuration files for specified time range, domain, and simulation.

    Parameters
    ----------
    trange : str
        Time range identifier (t0, t1, t2, t3).
    domain : str
        Spatial domain identifier.
    sim_id : str
        Simulation identifier.

    Returns
    -------
    None
    """

    years_dict = {
        "t0": np.arange(0, 35),
        "t1": np.arange(35, 65),
        "t2": np.arange(65, 95),
        "t3": np.arange(95, 119),
    }
    analysis_years = [int(n + 1980) for n in years_dict[trange]]

    outpath = f"/data/{sim_id}"

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # make new config files for the PVWatts runs
    for module in ["distributed", "utility"]:
        modpath = outpath + f"/{module}_{sim_id}"
        if not os.path.exists(modpath):
            os.makedirs(modpath)
        gen_file = sys.path[-1] + f"/rev_configs/pv_{module}_config_gen.json"
        collect_file = sys.path[-1] + f"/rev_configs/pv_{module}_config_collect.json"
        pipeline_file = sys.path[-1] + f"/rev_configs/pv_{module}_config_pipeline.json"
        pp_file = sys.path[-1] + f"/rev_configs/pv_{domain}_{module}_projectpoints.csv"
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
    trange = str(sys.argv[1])
    domain = str(sys.argv[2])
    sim_id = str(sys.argv[3])
    make_pv_config(trange, domain, sim_id)
