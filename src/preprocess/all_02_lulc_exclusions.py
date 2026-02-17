#!/usr/bin/python3
"""
Create land use and land cover exclusion masks for renewable energy siting.

This module processes land use and land cover data to create exclusion masks
that identify areas unsuitable for renewable energy development based on
environmental and regulatory constraints.
"""

import sys
import tempfile

import intake
import xarray as xr

from src.ds_to_h5 import (
    make_exclusions_mask,
)
from src.preprocess.all_01_meta_node_create import get_single_timestep
from src.utils import ds_to_zarr


def add_landuse_landcover_exclusions(domain: str, exclusion: str) -> None:
    """
    Add land use/land cover exclusions for renewable energy siting analysis.

    Parameters
    ----------
    domain : str
        Spatial domain identifier.
    exclusion : str
        Exclusion type identifier.

    Returns
    -------
    None
    """
    # Open catalog of available data sets using intake-esm package
    # we only need one sample dataset for this

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
        # grid resolution: d01 = 45km, d02 = 9km, d03 = 3km
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

    # define specified land use exclusion
    datpath = "/data/"
    shp = datpath + exclusion + ".gpkg"
    exclusions_dict = {exclusion: shp}

    # make the binary mask
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        mask_ds = make_exclusions_mask(ds, exclusions_dict, outfile=f)
    ds = xr.merge([mask_ds, ds])
    ds = ds.drop("psfc")

    # save the dataset
    destination_path = f"era/resource_data/{domain}/static_files/"
    ds_to_zarr(ds, destination_path, f"coord_ds_{domain}_{exclusion}")


if __name__ == "__main__":
    DOMAIN = str(sys.argv[1])
    EXCLUSION = str(sys.argv[2])
    add_landuse_landcover_exclusions(DOMAIN, EXCLUSION)
