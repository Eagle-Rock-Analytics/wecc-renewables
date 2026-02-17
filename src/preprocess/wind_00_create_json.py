#!/shared/miniconda3/envs/renew/bin/python
"""
Create JSON configuration files for wind resource processing.

This module generates JSON configuration files for wind resource data processing,
organizing model outputs by domain, simulation ID, and time periods for parallel
processing workflows.
"""

import argparse
import json
from typing import Any

import boto3
import numpy as np

bucket = "wfclimres"
wrf_bucket = "wrf-cmip6-noversioning"
s3 = boto3.client("s3")
models = ["MIROC6", "EC-Earth3", "MPI-ESM1-2-HR", "TaiESM1"]


# ------------------------------------------------------------------------------------------------
def find_wrf_files_on_s3(bucket: str, prefix: str) -> list[str]:
    """
    Find WRF files on S3 bucket with given prefix.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    prefix : str
        S3 object prefix to search for.

    Returns
    -------
    List[str]
        List of S3 object keys matching the prefix.
    """

    s3 = boto3.client("s3")

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    files = []

    for page in pages:
        for obj in page.get("Contents", []):
            files.append(obj["Key"])

    return files


# ------------------------------------------------------------------------------------------------
def make_wrf_jsons_domain_model(domain: str, sim_id: str) -> dict[str, Any]:
    """
    Create JSON configuration files for WRF wind processing by domain and model.

    Parameters
    ----------
    domain : str
        WRF domain identifier.
    sim_id : str
        Simulation/model identifier.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing WRF file configuration data.
    """

    years = np.arange(1980, 2100, dtype=int)
    scens = {y: "historical" for y in years if y < 2014} | {
        y: "ssp370" for y in years if y >= 2014
    }

    years = years[1:-1]

    # print(years)
    # scenario = "historical"
    # Dict for JSON file names (in renewables s3 buckets)
    sim_arns = {
        "EC-Earth3": "ec-earth3_r1i1p1f1_2_{scenario}_bc",
        "MPI-ESM1-2-HR": "mpi-esm1-2-hr_r3i1p1f1_{scenario}_bc",
        "MIROC6": "miroc6_r1i1p1f1_{scenario}_bc",
        "TaiESM1": "taiesm1_r1i1p1f1_{scenario}_bc",
    }

    prefix = "downscaled_products/gcm/{sim_arn}/hourly/{year}/{domain}/"

    data = {}

    for year in years:
        sub_prefix0 = prefix.format(
            sim_arn=sim_arns[sim_id].format(scenario=scens[year]),
            year=year,
            domain=domain,
        )
        sub_prefix1 = prefix.format(
            sim_arn=sim_arns[sim_id].format(scenario=scens[year - 1]),
            year=year - 1,
            domain=domain,
        )
        files0 = find_wrf_files_on_s3(bucket=wrf_bucket, prefix=sub_prefix0)
        files0 = [f for f in files0 if f"{year+1}-09-01_00:00:00" not in f]
        files1 = find_wrf_files_on_s3(bucket=wrf_bucket, prefix=sub_prefix1)
        files1 = [f for f in files1 if f"{year+1}-09-01_00:00:00" not in f]

        files = list(np.sort(np.concatenate([files0, files1])))
        files = [f for f in files if "-02_29_" not in f]
        files = [f for f in files if f"auxhist_d01_{year}" in f]

        data[str(year)] = files

    return data


# --------------------------------------------------------------------------------------------------------
def create_and_save_json(domain):

    s3_key = "wrf_jsons"
    bucket = "wfclimres"

    data = {}

    for m in models:
        print(f"Processing model: {m}, domain {domain}", flush=True)
        sub_data = make_wrf_jsons_domain_model(domain, m)
        data[m] = sub_data
        fName = f"{m}_{domain}.json"

        with open(fName, "w") as json_file:
            json.dump(sub_data, json_file)
        s3.upload_file(Filename=fName, Bucket=bucket, Key=f"{s3_key}/{fName}")

    # fName = f"{domain}.json"
    # with open(fName, "w") as json_file:
    #    json.dump(data, json_file)
    # s3.upload_file(Filename=fName, Bucket=bucket, Key=f"{s3_key}/{fName}")


# --------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Create parser
    parser = argparse.ArgumentParser(
        prog="create_jsons",
        description="""
                    """,
        epilog="""
                  Possible domains: ["d01","d02","d03","d04"]
               """,
    )

    # Define arguments for the program
    parser.add_argument(
        "-d", "--domain", default="d02", help="Domain (Defaults to 'd02')", type=str
    )

    args = parser.parse_args()
    domain = args.domain

    create_and_save_json(domain)
