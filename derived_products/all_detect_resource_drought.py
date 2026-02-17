"""
## Module: Resource Drought Detection

This module calculates daily renewable energy resource droughts for PV and wind power.
It identifies drought events by comparing daily generation against a day-of-year mean threshold
derived from reference global warming levels (GWL) or reanalysis periods. It is hardcoded to
detect resource droughts as any day with < 50% of the mean production for that day of year over
the reference period.

**Note:**
This module is dependent on day-of-year production means derived in all_resource_drought_reference.py.
"""

import sys

import numpy as np
import pandas as pd
import xarray as xr

from postprocess.all_hourly_to_daily import mk_exclusions_mask


# Define specific write to zarr function
def write_to_zarr(ds: xr.Dataset, out_path: str) -> None:
    """
    Write dataset to Zarr format with standardized encoding.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to be saved.
    out_path : str
        S3 or local path for the Zarr store.

    Returns
    -------
    None
    """
    coord_to_encode = [coord for coord in ds.coords if coord not in ["x", "y", "time"]]
    for var in ds:
        if "chunks" in ds[var].encoding:
            del ds[var].encoding["chunks"]
    for coord in coord_to_encode:
        if "chunks" in ds[coord].encoding:
            del ds[coord].encoding["chunks"]
    ds = ds.transpose("time", "y", "x")
    ds = ds.chunk(chunks={"time": 1095, "y": 246, "x": 122})
    ds.to_zarr(out_path, mode="w")


# Primary workflow
def daily_binary_rsrc_drought(
    domain: str, sim_id: str, module: str, gwl: str, var: str, tech: str
) -> None:
    """
    Calculate and save daily resource drought masks and percentages.

    Parameters
    ----------
    domain : str
        WRF domain identifier ('d03' or 'd02').
    sim_id : str
        Simulation name (e.g., 'ERA5', 'EC-Earth3').
    module : str
        Installation type (e.g., 'utility', 'onshore').
    gwl : str
        Global warming level (e.g., '0.8', '1.5', '2.0').
    var : str
        Variable name ('gen' or 'cf').
    tech : str
        Technology type ('pv' or 'windpower').

    Returns
    -------
    None
    """

    member_dict = {
        "TaiESM1": "_r1i1p1f1",
        "MIROC6": "_r1i1p1f1",
        "EC-Earth3": "_r1i1p1f1",
        "MPI-ESM1-2-HR": "_r3i1p1f1",
        "ERA5": "",
    }
    member_id = member_dict[sim_id]
    sim = f"{sim_id}{member_id}"

    if sim_id == "ERA5":
        # no global warming level, needs to be treated differently
        scen = "reanalysis"
        exp_id = scen
        zarr_path = f"s3://wfclimres/era/{tech}_{module}/{sim_id.lower()}/{scen}/day/{var}/{domain}/"
        y0, y1 = 1981, 2011
        ds = xr.open_zarr(zarr_path).sel(time=slice("1981", "2010"))
        thresh_ds_path = f"s3://wfclimres/era/resource_drought/{tech}/{tech}_{module}/{sim_id.lower()}/reanalysis/threshold/{domain}/"
        # output directories
        out_path_dict = {
            "percent_mean": f"s3://wfclimres/era/resource_drought/{tech}/{tech}_{module}/{sim_id.lower()}/reanalysis/day/percent_mean/{domain}/",
            "drought_mask": f"s3://wfclimres/era/resource_drought/{tech}/{tech}_{module}/{sim_id.lower()}/reanalysis/day/drought_mask/{domain}/",
        }

    else:
        # load in the global warming reference data
        gwl_df = pd.read_csv("../data/renewables_gwl_reference.csv")
        gwl_df = gwl_df.set_index("simulation")

        # define the global warming level window
        t = pd.to_datetime(gwl_df.loc[sim][gwl])
        y = int(t.date().year)
        y0 = y - 15
        y1 = y + 15

        # put gwl window into a list
        my_years = list(np.arange(y0, y1))
        # divide years by scenario: historical or ssp
        hist_years = [str(int(y)) for y in my_years if y < 2014]
        ssp_years = [str(int(y)) for y in my_years if y > 2014]

        # use 0.8C global warming level as reference
        thresh_ds_path = f"s3://wfclimres/era/resource_drought/{tech}/{tech}_{module}/{sim_id.lower()}/plus08c/threshold/{domain}/"
        # output directories
        out_path_dict = {
            "percent_mean": f"s3://wfclimres/era/resource_drought/{tech}/{tech}_{module}/{sim_id.lower()}/plus{gwl.replace('.','')}c/day/percent_mean/{domain}/",
            "drought_mask": f"s3://wfclimres/era/resource_drought/{tech}/{tech}_{module}/{sim_id.lower()}/plus{gwl.replace('.','')}c/day/drought_mask/{domain}/",
        }

        if hist_years and ssp_years:
            scen0 = "historical"
            scen1 = "ssp370"
            zarr_path0 = f"s3://wfclimres/era/{tech}_{module}/{sim_id.lower()}/{scen0}/day/{var}/{domain}/"
            ds0 = xr.open_zarr(zarr_path0).sel(
                time=slice(hist_years[0], hist_years[-1])
            )
            zarr_path1 = f"s3://wfclimres/era/{tech}_{module}/{sim_id.lower()}/{scen1}/day/{var}/{domain}/"
            ds1 = xr.open_zarr(zarr_path1).sel(time=slice(ssp_years[0], ssp_years[-1]))
            ds = xr.concat([ds0, ds1], dim="time")
            exp_id = f"{scen0} + {scen1}"
        else:
            if hist_years:
                scen = "historical"
                gwl_years = hist_years
                exp_id = scen
            else:
                scen = "ssp370"
                gwl_years = ssp_years
                exp_id = scen
            zarr_path = f"s3://wfclimres/era/{tech}_{module}/{sim_id.lower()}/{scen}/day/{var}/{domain}/"
            ds = xr.open_zarr(zarr_path).sel(time=slice(gwl_years[0], gwl_years[-1]))

    thresh_ds = xr.open_zarr(thresh_ds_path)

    # make an exclusions mask to reintroduce the land use/cover nans.
    excl_mask = mk_exclusions_mask(ds)

    # ensure no leap days pop up during the dayofyear operation
    ds = ds.convert_calendar("noleap")
    ds["dayofyear"] = ds.time.dt.dayofyear
    ds["year"] = ds.time.dt.year
    ds = ds.assign_coords({"dayofyear": ds.time.dt.dayofyear, "year": ds.time.dt.year})
    # reshape time dimension
    ds = ds.drop_vars("time").set_index(time=["dayofyear", "year"]).unstack()
    ds = ds.chunk({"dayofyear": 365})
    # detect resource drought days
    drought_ds = ds.copy(deep=True)
    drought_ds["percent_mean"] = drought_ds["gen"] / thresh_ds["gen"]
    drought_ds["drought_mask"] = xr.where(drought_ds["percent_mean"] < 0.5, x=1, y=0)
    drought_ds["drought_mask"] = drought_ds["drought_mask"].astype(np.float32)

    # reverse the earlier reshape of the time dimension
    # make the target time vector
    tvec = pd.date_range(start=f"{y0}-01-01", end=f"{y1-1}-12-31", freq="D")
    # remove leap days - again
    tvec = tvec[~((tvec.month == 2) & (tvec.day == 29))]
    # remove 2014 - sometimes a GWL spans historical & SSP time frames
    tvec = tvec[~(tvec.year == 2014)]
    # reset the year, dayofyear index and replace with the time vector
    drought_ds = drought_ds.stack(time=["year", "dayofyear"])
    drought_ds = drought_ds.reset_index("time").assign_coords(time=tvec)

    # attach attributes
    drought_ds["drought_mask"].attrs = ds[var].attrs
    drought_ds["drought_mask"].attrs[
        "extended_description"
    ] = "Resource drought mask (1 for resource drought, 0 for non-drought)."
    drought_ds["drought_mask"].attrs["long_name"] = "Resource drought binary mask"
    drought_ds["drought_mask"].attrs["units"] = "1"

    drought_ds["percent_mean"].attrs = ds[var].attrs
    drought_ds["percent_mean"].attrs[
        "extended_description"
    ] = "Ratio of daily generation to day-of-year average taken over the reference timeframe"
    drought_ds["percent_mean"].attrs[
        "long_name"
    ] = "Ratio of daily generation to day-of-year average"
    drought_ds["percent_mean"].attrs["units"] = "1"
    drought_ds.attrs["experiment_id"] = exp_id

    # some attrs seem to disappear
    drought_ds.x.attrs = {"standard_name": "projection_x_coordinate", "units": "m"}
    drought_ds.y.attrs = {"standard_name": "projection_y_coordinate", "units": "m"}
    drought_ds["time"].attrs["time_zone"] = "UTC -8:00"
    drought_ds["time"].attrs["standard_name"] = "time"

    # apply land use/cover mask
    drought_ds = xr.where(np.isnan(excl_mask), x=np.nan, y=drought_ds)

    # ship off the two vars to their new homes
    for v in ["drought_mask", "percent_mean"]:
        out_da = drought_ds[v].to_dataset()
        out_da.attrs = ds.attrs
        out_path = out_path_dict[v]
        write_to_zarr(out_da, out_path)


if __name__ == "__main__":
    domain = str(sys.argv[1])
    sim_id = str(sys.argv[2])
    module = str(sys.argv[3])
    gwl = str(sys.argv[4])
    var = str(sys.argv[5])
    tech = str(sys.argv[6])

    daily_binary_rsrc_drought(domain, sim_id, module, gwl, var, tech)
