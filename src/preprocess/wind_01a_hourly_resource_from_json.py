"""
Convert WRF output files to hourly wind resource format using MPI.

This module orchestrates parallel preprocessing of WRF outputs into wind resource
datasets, using MPI for file distribution and xarray/zarr for output storage. The
core physics and transformation steps are delegated to helper functions in
``src.wind_to_resource_functions``.

Notes
-----
Inputs are expected to be listed in a JSON file (per model/domain/year) and may
reside on S3. Outputs are written as zarr stores organized by domain, model, and
year under ``/data/wind_resources/hourly``.
"""

import argparse
import json
import os
import time

import numpy as np
import s3fs
import xarray as xr
from mpi4py import MPI
from netCDF4 import Dataset
from simplempi import simpleMPI

from src.wind_to_resource_functions import (
    concat_base_perturbations,
    count_all_paths,
    create_pressure_resources,
    create_temperature_resources,
    create_wind_resources,
    directory_size_bytes,
    download_read_using_nc4,
    get_uvmet_wspd_wdir,
    interp_uv_to_height,
    process_output_zarr,
)


def hourly_wind_json_to_nc(domain: str, sim_id: str, year: int, fix_year: int) -> None:
    """
    Preprocess hourly WRF data to wind resources for reV
    using netCDF4 and wrf.g_uvmet.get_uvmet_wspd_wdir

    Parameters
    ----------
    domain : str
        Spatial domain identifier (e.g., 'd02', 'd03').
    sim_id : str
        Simulation identifier (e.g., 'ERA5', 'MIROC6').
    year : int
        Simulation year to process.
    fix_year : int
        When non-zero, only reprocess incomplete/corrupted outputs.

    Returns
    -------
    None
        Writes NetCDF files from JSON inputs, no direct return.

    Notes
    -----
    This routine uses MPI for file distribution and will overwrite existing
    zarr outputs for files deemed incomplete.

    This process is very specific tailored to the WRF CMIP6 data structure and the AWS parallel cluster
    setup used for processing. Be aware before adapting to other contexts.
    """
    # Initialize simplempi/mpi4py process
    smpi = simpleMPI()
    # Renewables and WRF files buckets
    wrf_bucket = "wrf-cmip6-noversioning"
    # Json path
    with open(f"/data/static/{sim_id}_{domain}.json") as f:
        data = json.load(f)

    # Extract a single year of json data
    data = data[str(year)]

    # Broadcast scenario, domain, and sim_id to all processors/tasks
    domain = smpi.comm.bcast(domain, root=0)
    sim_id = smpi.comm.bcast(sim_id, root=0)

    # Create list of files to process
    files = [f"s3://{wrf_bucket}/{item}" for item in data]
    files_scatter = smpi.scatterList(files)

    if domain == "d03":
        min_size = 3400000
    elif domain == "d02":
        min_size = 2500000

    bad_files = []
    if fix_year:
        for f in files_scatter:
            # File basename
            ftail = f.split("/")[-1]
            out_year = ftail.split("_")[2].split("-")[0]
            out = (
                f"/data/wind_resources/hourly/{domain}/{sim_id}/{out_year}/{ftail}.zarr"
            )
            if os.path.exists(out):
                size = directory_size_bytes(out)
                if count_all_paths(out) < 55 or size < min_size:
                    bad_files.append(f)
                else:
                    continue
            else:
                bad_files.append(f)

        gathered_bad_files = MPI.COMM_WORLD.allgather(bad_files)

        files = list(np.sort(np.concatenate(gathered_bad_files)))
        if smpi.rank == 0:
            smpi.pprint(f"{len(files)} bad_files to process", flush=True)

    del files_scatter
    files_scatter = smpi.scatterList(files)

    if smpi.rank == 0:
        smpi.pprint(f"{len(files_scatter)} scattered files per task", flush=True)

    # Loop over files (using mpi4py)
    for file in files_scatter:

        try:
            # File basename
            file_name_tail = file.split("/")[-1]
            out_year = file_name_tail.split("_")[2].split("-")[0]
            # Where to save the file
            directory = f"/data/wind_resources/hourly/{domain}/{sim_id}/{out_year}"
            # Full file name
            file_name = f"{directory}/{file_name_tail}.zarr"

            # Redefine buckets and base state paths in each process
            wrf_bucket = "wrf-cmip6-noversioning"
            # Define which variables to extract from WRF base state and perturbations
            variables = ["U", "V", "T", "P", "PH", "PSFC"]
            base_vars = ["U", "V", "T00", "PB", "PHB", "HGT"]
            base_vars.extend(["XLAT", "XLONG"])
            base_attrs = ["MAP_PROJ", "CEN_LON", "STAND_LON", "TRUELAT1", "TRUELAT2"]

            # Initialize filesystem object to interact with S3 storage on each process
            s3fs.S3FileSystem()
            # Download, read, and extract base variables
            base_ds = Dataset(f"/data/static/wrfinput_{domain}.nc", "r")

            # Time process and print which file is being processed
            t0 = time.time()
            smpi.pprint(f"\nProcessing {file}", flush=True)

            # Download, read and extract variables from WRF perturbation file
            raw_ds = download_read_using_nc4(
                file, variables=variables, attributes=base_attrs
            )
            # Concatenate and calculate absolute state from WRF base + perturbations
            raw_ds = concat_base_perturbations(raw_ds, base_ds)
            # Interpolate U,V to heights
            raw_ds, levels = interp_uv_to_height(raw_ds)
            # Calculate wind speed and direction
            raw_ws, raw_wd = get_uvmet_wspd_wdir(raw_ds)

            # Calculate output variables at each desired level
            output_vars = []
            output_levels = [100, 120]
            for level in output_levels:
                # Find index of desired level
                zlev = np.where(levels == level)[0]
                # Calculate ws and wd for specific level
                out_ws, out_wd = create_wind_resources(raw_ws, raw_wd, zlev, level)
                # Calculate temperature and pressure for specific level
                out_t = create_temperature_resources(raw_ds, out_ws, zlev, level)
                out_p = create_pressure_resources(raw_ds, out_ws, zlev, level)
                # Append variables from each level
                output_vars.append(out_ws)
                output_vars.append(out_wd)
                output_vars.append(out_t)
                output_vars.append(out_p)

            # Calculate surface pressure (at 0 m) outside the loop
            level = 0
            zlev = 0
            out_p = create_pressure_resources(raw_ds, out_ws, zlev, level)
            output_vars.append(out_p)

            # Merge all output variables into an xarray.Dataset
            ds_out = xr.merge(output_vars)
            # Process output xarray.Dataset with attributes, time coordinate, etc
            ds_out = process_output_zarr(ds_out, file)
            # Encode time coordinate to be able to use NCO down the line

            # If the output directory doesn't exist, create it
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            # Save processed wind resource file to disk
            ds_out = ds_out.chunk(chunks="auto")
            ds_out.to_zarr(file_name, mode="w")

            # Compute processing time and print to screen
            elapsed = time.time() - t0
            smpi.pprint(
                f"File saved to {file_name}: Elapsed time {elapsed:.1f}",
                flush=True,
            )
            # Save memory
            ds_out.close()
            raw_ds.close()
            base_ds.close()
            del (ds_out, raw_ds, base_ds)

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            # Log error, skip, and continue
            continue

    if not fix_year:
        time.sleep(10)
        hourly_wind_json_to_nc(domain, sim_id, year, fix_year=True)

    return None


if __name__ == "__main__":
    # Define parsing function
    parser = argparse.ArgumentParser(
        prog="wrf_to_resources_nc4",
        description="\n                    ",
        epilog='Possible models: ["EC-Earth3","MPI-ESM1-2-HR","MIROC6","TaiESM1"]\n                  Possible domains: ["d01","d02","d03","d04"]\n               ',
    )
    # Add model argument to parser
    parser.add_argument(
        "-m",
        "--model",
        default="EC-Earth3",
        help="Simulation name (Defaults to 'EC-Earth3')",
        type=str,
    )
    # Add year argument to parser
    parser.add_argument(
        "-y",
        "--year",
        default=1980,
        help="Simulation year",
        type=int,
    )
    # Add domain argument to parser
    parser.add_argument(
        "-d", "--domain", default="d01", help="Domain (Defaults to 'd01')", type=str
    )
    # Add domain argument to parser
    parser.add_argument(
        "-f",
        "--fix_year",
        default=False,
        help="If fixing an incomplete year serially set to True",
        type=bool,
    )

    # Parse arguments
    args = parser.parse_args()
    domain = args.domain
    year = args.year
    sim_id = args.model
    fix_year = args.fix_year

    # Run main preprocessing function
    hourly_wind_json_to_nc(domain, sim_id, year, fix_year)
