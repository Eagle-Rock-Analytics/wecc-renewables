# PV and Wind Power Profile Creation

This README outlines our workflow for generating PV and wind power resource files, running the relevant ReV software, and S3 management. 

> [!NOTE]
> For details on the installation parameters (eg, offshore wind, utility-scale PV), please read the [data catalog memo (PDF download link)](https://wfclimres.s3.amazonaws.com/era/data-guide_pv-wind.pdf).

> [!WARNING]
> This README is based on the particular computing setup we used to generate and export the data. Users will have to change things to accommodate their own setups, including paths embedded in the various scripts (Python scripts in `src` as well as the batch scripts here) and how scripts are called on by job scheduling software, etc. In other words, we do not expect this to work without modifications.
> 
> Additionally, please be aware that the resource and generation files will be large and require a large storage space. Plan accordingly in terms of resource and/or cost. 

---

# Step 0: Make the ReV-compliant spatial datasets

ReV requires spatial information, including geographic location, elevation, and time zone, which make up the "meta node" portion of the resource files. It also requires "project points" CSVs which define the locations over which to simulate generation potential.

## Populate the meta node

* **Python Preprocessing:** `src/preprocess/all_01_meta_node_create.py`
* **SLURM Batch Script:** `scripts/all-01_meta_node_create.sbatch`

Running the above will pull Zarrs generated from WRF output via `intake-esm`, extract spatial information, and create a ReV compliant HDF5 file. Please change destination paths as necessary for your setup. 

## Workflow to make the project points

### Incorporating land use/land cover restrictions

The following table summarizes the relevant land use exclusions from PVWattsv8 or Windpower, respectively, under a given installation type. In other words, if a grid cell meets any of these criteria, it was excluded from the indicated simulations. Note that [PV and wind resource Zarrs in the S3 bucket](https://wfclimres.s3.amazonaws.com/index.html#era/resource_data/) do not have any exclusions, so there exists the potential for interested parties to perform simulations at excluded grid cells should they so desire.

| Restriction | Name in exclusion datasets | Utility PV | Distributed PV | Onshore wind | Offshore wind |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Federal and state protected lands | fed_state_protected | X | X | X | X |
| Slope > 20% | wind_slope_over_20p | X | | X | X |
| High intensity urban | pv_land_cover_urban_high | X | | X | X |
| Medium intensity urban | wind_land_cover_urban_medium_high | | | X | X |
| Open water | landmask | X | X | X | |

* _High intensity urban_ refers to row houses, apartment buildings, and commercial buildings.
* _Medium intensity urban_ refers to single family homes.

#### Original geospatial files

Geospatial files used to mask out grid points for exclusions were sourced from the following:

* Fed/State protected lands [Protected Areas Database of the United States (PAD-US) 3.0 (ver. 2.0, March 2023)](https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-download)
* Slope >20% [LandFire, Version 2.2.0, Slope Percent Rise](https://www.landfire.gov/version_download.php#)
* [High intensity urban NLCD](https://www.mrlc.gov/data?f%5B0%5D=category%3ALand%20Cover&f%5B1%5D=region%3Aconus&f%5B2%5D=year%3A2021)
* Open water was masked with the WRF domain’s land-sea mask

#### Derived geospatial datasets

* The geospatial files (.gpkg format) used to create the land use/cover exclusion mask are [archived in the S3 bucket](https://wfclimres.s3.amazonaws.com/index.html#era/resource_data/land_use_restrictions/)
* The gridded datasets with the binary land use/cover restriction masks are also available in the relevant [resource data folders](https://wfclimres.s3.amazonaws.com/index.html#era/resource_data/).

## Create land use/land cover exclusion masks for renewable energy siting

* **Python Preprocessing:** `src/preprocess/all_02_lulc_exclusions.py`
* **SLURM Batch Script:** `scripts/all-02_lulc_exclusions.sbatch`
* **Exclusion List:** `scripts/list-exclusions.dat`

Running the above will pull the .gpkg files from the S3 bucket which correspond to the listed exclusions and convert them to gridded Zarrs which match the coordinates of the WRF datasets. 

## Create the Project Points CSVs

* **Notebook:** `notebooks/all_03_mk_rev_project_points.ipynb` 

Running the above notebook will:
1. Merge the Zarrs containing individual restrictions into a single Zarr.
2. Use the resulting merged Zarr to create ReV-ready Project Points CSVs which define the valid grid cells for a given installation, as described in the above table. 

---

# PV Workflow

## Step 1: Resource File Generation

### Background Scripts
* **Python Preprocessing:** `src/preprocess/pv_resource_to_h5.py`
* **SLURM Batch Script:** `scripts/pv_resource_to_h5.sbatch`

### Procedure
1.  **Configure SLURM Array:**
    * Open `pv_resource_to_h5.sbatch`.
    * Modify **LINE 4**: `#SBATCH --array=1-[N years]` where `N years` is the total count of years to process (e.g., set to 24 for the period 2075-2098).

2.  **Generate Input Lists:**
    * Navigate to `scripts/`.
    * Edit `mk-batch.bash` with the following parameters:
        * **LINE 8 (SIMS):** GCM name (`EC-Earth3`, `MPI-ESM1-2-HR`, `MIROC6`, or `TaiESM1`).
        * **LINE 9 (MODS):** Set to `("utility")` or `("distributed")`.
        * **LINE 12/13 (START/END):** Year indices (Where an index = Desired Year - 1980).
    * Execute: `bash mk-batch.bash`.
    * Verify `input-list.dat` and `output-list.dat` match the job array size.

3.  **Submit Resource Job:**
    * Start the cluster.
    * Run: `sbatch pv_resource_to_h5.sbatch`.
    * Verify output files your specified path (must be defined manually in `src/preprocess/pv_resource_to_h5.py`).

4.  **Configure reV Files:**
    * Navigate to `src/preprocess/`.
    * `mamba activate renew`
    * Run (see note): `python3 misc_edit_pv_configs.py [trange] [DOMAIN] [SIM_ID]`
    * *Example:* `python3 misc_edit_pv_configs.py t3 d02 MPI-ESM1-2-HR`

- **`trange` note:** We did this workflow in four temporal batches, which are specified in the above argument as `trange`. The following `trange` options are:  
   - `t0`- for simulation years 1981-2013 (historical period)
   - `t1` - for simulation years 2015-2044
   - `t2` - for simulation years 2045-2074
   - `t3` - for simulation years 2075-2098

You may have to change `misc_edit_pv_configs.py` to suit your case.

---

## Step 2: Run ReV PVWatts

Note: This is written for “utility” installations, but you can substitute for “distributed” installations. 

### 2a: Generation
1.  Navigate to the directory containing the reV config files, eg: 
`cd /data/[SIM_ID]/[MODULE]_[SIM_ID]`.
2.  Open a `screen` session.
3.  `conda activate rev`
4.  Run: `reV pipeline`.
5.  Monitor progress. If the session disconnects, use `screen -r` to reattach.

### 2b: Collection and Redos
1.  **Stitch Files:**
    * Re-run `reV pipeline`. If previous generation jobs failed, it will restart them. If complete, it will begin "collect" jobs.
2.  **Handle Missing Years:**
    * **Option A (Reset):** `reV reset-status`. Delete the last simulated year (likely corrupt). Edit JSON to missing years and run `reV pipeline`.
    * **Option B (Redo Directory):**
        * Create `redo_[MODULE]_[SIM_ID]` and copy `.json` configs.
        * Edit `pv_[MODULE]_config_gen.json` `analysis_years` to only missing years.
        * Run `reV pipeline`.

---

## Step 3: Export to S3 (Temporary Zarrs)

> [!NOTE]
> This is how we sent temporary Zarrs to the public S3 bucket - you will have to store your Zarrs somewhere else, so be sure to change destination paths in `src/postprocess/all_generation_output_to_zarr.py`

If redo folders were used, complete this step for both the initial set and the `redo_[MODULE]` folder.

1.  **Prepare Batch:**
    * In `scripts/`, edit `mk-batch.bash`.
    * Set `MODULE`. 
    * Set `START` and `END` indices.
    * Run `bash mk-batch.bash`.

2.  **Submit Conversion Job:**
    * Edit `all-generation_output_to_zarr.sbatch`.
    * Update `--array` to match the number of lines in `input-list.dat`.
    * Run: `sbatch all-generation_output_to_zarr.sbatch`.

3.  **Verification:**
    * Check the destination path you specified for your Zarrs.


---

## Step 4: Final Packaging 

> [!NOTE]
> This is how we sent Zarrs to the public S3 bucket - you will have to store your Zarrs somewhere else, so be sure to change destination paths in `src/postprocess/pv_profiles_to_catalog.py`

1.  Navigate to `scripts/`.
2.  Edit `mk-package-list.bash` with correct domain (`d02` or `d03`), simulation, and module (`utility` or `distributed`).
3.  Run: `bash mk-package-list.bash`.
4.  Verify `package-list.dat` contains lines for `historical` and/or `ssp370`.
5.  Edit `pv-profiles_to_catalog.sbatch` and set `--array` option appropriately.
6.  Run: `sbatch pv-profiles_to_catalog.sbatch`.

---

## Troubleshooting Tips
* **Log Files:** Check `logs/stdout` or files ending in `.e` for errors.
* **JSON Configs:** Verify `analysis_years` in `pv_utility_config_gen.json` matches intended years.
* **Missing GIDs:** If "gids ... are missing" appears, spatial points are incomplete; rerun generation step for those years.

# Wind Power Workflow

---

## Step 1: Resource File Generation

### 1a: Multi-Zarr JSON Creation
* **SLURM:** `scripts/slurm/wind-00_create_multizarr_json.sh`
* **Python:** `scripts/preprocess/wind-00_create_multizarr_json.py`
* **Status:** Standard JSONs are pre-uploaded to the S3 bucket.
   - JSON structure:
     - One JSON file for each model/domain
     - Property names (keys): simulation years
     - Each year key has a list of the S3 bucket path to an hourly WRF file

### 1b: Hourly Resource Generation
1.  **Generate Batch List:**
    * Edit `mk-batch.sh` to set `DOMAIN` and `SIM_ID`.
    * Run `bash mk-batch.sh`.
    * Verify `input-list.dat` contains 118 lines (1981–2098).
2.  **Submit Hourly Job:**
    * Run `sbatch wind-01a_hourly_resource_from_json.sh`:
       - This runs `src/preprocess/wind_01a_hourly_resource_from_json.py.`
       - Helper functions for this py file are under `src/wind_to_resource_functions.py`.
3.  **Validate/Fix Hourly Files:**
    * **Count Files:** `python3 wind-01b_count_hourly_files.py --model [GCM] --year [YYYY] --domain [DXX]`
    * **Auto-Resubmit:** Add `--resubmit True` to the count script to fix incomplete files automatically.
    * **Manual Fix:** `sbatch wind-01b_fix_year_hourly_resource_from_json.sh [DOMAIN] [SIM_ID] [YEAR]`

### 1c: Concatenate and Upload
1.  **Concatenate:** `bash wind-02a_concat_hourly_resource.sh [DOMAIN] [SIM_ID]`
2.  **Verify S3 Size:** `bash wind-02b_check_zarr_s3_sizes.sh [DOMAIN] [SIM_ID]`
    * Check `zarr_files_size_$[SIM_ID]_$[DOMAIN].dat`. Rerun concatenation if sizes appear truncated.
    * Note: You will have to tune the `bash` script to your destination output path.

### 1d: Cleanup Hourly Files
1.  **Delete:** `sbatch wind-03a_delete_hourly.sh [DOMAIN] [SIM_ID]`
    * Use the `exclude_year` variable in the script to protect specific years from deletion.
    * Note: You will have to tune this script to your destination output path.

### 1e: Zarr to H5 Conversion
1.  **Configure `mk-batch.bash`:**
    * **LINE 8 (SIMS):** GCM choice (`EC-Earth3`, `MPI-ESM1-2-HR`, etc.).
    * **LINE 9 (MODS):** Set to `"onshore"` or `"offshore"`.
    * **LINE 12/13:** Set `START`/`END` (Year - 1980).
2.  **Submit:**
    * Set `#SBATCH --array=1-[N]` in `wind-03b_resource_to_h5.sbatch`.
    * Run `sbatch wind_03b_resource_to_h5.sbatch`.
3.  **Manual Config (Optional):**
    * If configs fail to auto-generate: `python3 misc_edit_wind_configs.py [trange] [DOMAIN] [SIM_ID]`

- **`trange` note:** We did this workflow in four temporal batches, which are specified in the above argument as `trange`. The following `trange` options are:  
   - `t0`- for simulation years 1981-2013 (historical period)
   - `t1` - for simulation years 2015-2044
   - `t2` - for simulation years 2045-2074
   - `t3` - for simulation years 2075-2098

You may have to change `python3 misc_edit_wind_configs.py` to suit your case.

---

## Step 2: Run reV Windpower

### 2a: Generation
1.  Navigate to the directory containing the reV config files, eg:
    `cd /data/[SIM_ID]/[MODULE]_[SIM_ID]`.
3.  Open `screen`, then `conda activate rev`.
4.  Run `reV pipeline`.
    * *Note:* Distributed over 48 CPUs/year. 30 years takes ~1-2 days.

### 2b: Collection and Redos
1.  **Collect:** Run `reV pipeline` again once generation is complete.
2.  **Handle Missing Years:**
    * **Option A (Reset):** `reV reset-status`. Delete the last simulated year (likely corrupt). Edit JSON to missing years and run `reV pipeline`.
    * **Option B (Redo Directory):**
        * Create `redo_[MODULE]_[SIM_ID]` and copy `.json` configs.
        * Edit `wind_[MODULE]_config_gen.json` `analysis_years` to only missing years.
        * Run `reV pipeline`.

---

## Step 3: Export to S3 (Temporary Zarrs)

> [!NOTE]
> This is how we sent temporary Zarrs to the public S3 bucket - you will have to store your Zarrs somewhere else, so be sure to change destination paths in `src/postprocess/all_generation_output_to_zarr.py`

1.  Navigate to `scripts/slurm/`.
2.  Update `mk-batch.bash` (eg set `MODULE = "onshore"`, set year indices).
3.  Run `bash mk-batch.bash`.
4.  Edit `all-generation_output_to_zarr.sbatch` (`--array` size).
5.  Submit: `sbatch all-generation_output_to_zarr.sbatch`.

---

## Step 4: Final Packaging 

> [!NOTE]
> This is how we sent Zarrs to the public S3 bucket - you will have to store your Zarrs somewhere else, so be sure to change destination paths in `src/postprocess/wind_profiles_to_catalog.py`

1.  Edit `mk-package-list.bash` (Domain, Simulation, Module).
2.  Run `bash mk-package-list.bash`.
3.  Verify `package-list.dat` (should contain `historical` and `ssp370`).
4.  Edit `wind-profiles_to_catalog.sbatch` (`--array=1-2`).
5.  Submit: `sbatch wind-profiles_to_catalog.sbatch`.
6.  Verify output.

