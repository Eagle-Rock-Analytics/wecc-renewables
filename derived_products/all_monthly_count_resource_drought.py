"""
## Module: Monthly total resource droughts

This module counts and saves monthly totals of the daily renewable energy resource droughts
calculated in all_detect_resource_drought.py.
"""

import sys

import xarray as xr


def monthly_rsrc_drought_count(
    domain: str, sim_id: str, module: str, gwl: str, tech: str
) -> None:
    """Counts the number of resource drought days per month.

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
    tech : str
        Technology type ('pv' or 'windpower').

    Returns
    -------
    None
    """

    if sim_id == "ERA5":
        # no global warming level, needs to be treated differently
        zarr_path = f"s3://wfclimres/era/resource_drought/{tech}/{tech}_{module}/{sim_id.lower()}/reanalysis/day/percent_mean/{domain}/"
        ds = xr.open_zarr(zarr_path)
        # output directory
        out_path = f"s3://wfclimres/era/resource_drought/{tech}/{tech}_{module}/{sim_id.lower()}/reanalysis/mon/drought_count/{domain}/"

    else:
        zarr_path = f"s3://wfclimres/era/resource_drought/{tech}/{tech}_{module}/{sim_id.lower()}/plus{gwl.replace('.','')}c/day/drought_mask/{domain}/"
        ds = xr.open_zarr(zarr_path)
        # output directory
        out_path = f"s3://wfclimres/era/resource_drought/{tech}/{tech}_{module}/{sim_id.lower()}/plus{gwl.replace('.','')}c/mon/drought_count/{domain}/"

    mon_ds = ds["drought_mask"].resample(time="1ME").sum().to_dataset()
    mon_ds["month"] = mon_ds.time.dt.month
    mon_ds["year"] = mon_ds.time.dt.year
    mon_ds = mon_ds.assign_coords(
        {"month": mon_ds.time.dt.month, "year": mon_ds.time.dt.year}
    )
    mon_ds = mon_ds.set_index(time=["month", "year"]).unstack()
    mon_ds = mon_ds.rename({"drought_mask": "drought_count"})
    mon_ds["drought_count"].attrs["frequency"] = "month"
    mon_ds["drought_count"].attrs["long_name"] = "Count of resource drought days"
    mon_ds["drought_count"].attrs[
        "extended_description"
    ] = "Count of resource drought days, defined as a day with total generation of less than 50% of the day-of-year average taken over the 30-year reference timeframe"
    coord_to_encode = [
        coord for coord in mon_ds.coords if coord not in ["x", "y", "month", "year"]
    ]
    for v in mon_ds:
        if "chunks" in mon_ds[v].encoding:
            del mon_ds[v].encoding["chunks"]
    for coord in coord_to_encode:
        if "chunks" in mon_ds[coord].encoding:
            del mon_ds[coord].encoding["chunks"]
    mon_ds = mon_ds.transpose("month", "year", "y", "x")
    mon_ds = mon_ds.chunk(chunks={"month": 12, "year": 30, "y": 300, "x": 150})

    mon_ds.attrs = ds.attrs
    mon_ds.to_zarr(out_path, mode="w")


if __name__ == "__main__":
    domain = str(sys.argv[1])
    sim_id = str(sys.argv[2])
    module = str(sys.argv[3])
    gwl = str(sys.argv[4])
    var = str(sys.argv[5])
    tech = str(sys.argv[6])

    monthly_rsrc_drought_count(domain, sim_id, module, gwl, var, tech)
