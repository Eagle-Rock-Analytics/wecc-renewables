# Renewable Profiles

A project for generating photovoltaic (PV) and wind power generation potential profiles in fulfillment of the California Energy Commission's grant EPC-21-037.

## Project description

The Projections Team (CEC grant EPC-20-006) identified a subset of global climate models (GCMs) from the Coupled Model Intercomparison Project version 6 (CMIP6) which suitably captured California’s climate. GCM output was bias adjusted with respect to reanalysis (ERA5) in order to mitigate impacts related to model biases in temperature and temperature-dependent variables. This a-priori bias-adjusted GCM output was dynamically downscaled via the Weather Research and Forecasting (WRF) model to capture two relevant domains at high spatial resolution: (1) California-wide at 3 km resolution and (2) WECC-wide (excluding Alberta and British Columbia) at 9 km.

National Laboratory of the Rockies’ (NLR) Renewable Energy Potential (ReV) software package provides the overall computing architecture for this work. We use the System Advisory Model (PySAM) PVWattsv8 for the photovoltaic energy generation profiles, and the PySAM WindPower module for the wind energy generation profiles. We output both the generation potential (MWh AC) as well as the capacity factor at hourly time steps. From these we also derived daily aggregations.

This code repository fulfills efforts towards transparency and reproducibility by sharing our workflow from preprocessing the WRF output into ReV-compliant PV and wind resource files, to running the ReV pipeline, to post-processing the results and archiving them as Zarrs.

### See also

* [The accompanying hydropower profiles](https://github.com/jszinai/WWSM-WEAP-SWITCH) made by our partners

* [Our browsable S3 bucket](https://wfclimres.s3.amazonaws.com/index.html#era/), made possible by the Amazon Sustainable Data Initiative

* [Our partners on the Projections Team](https://dept.atmos.ucla.edu/alexhall/downscaling-cmip6)

* [A companion repository](https://github.com/Eagle-Rock-Analytics/renewable-analysis) featuring example analysis workflows for this data

* [NLR's ReV GitHub](https://github.com/NatLabRockies/reV)

> [!NOTE]
> You will have to set up a separate ReV conda environment to run the ReV pipeline. Please follow their instructions.

## Setup instructions

Clone this repository, then use the Makefile commands to get your environment setup:

```bash
make setup
mamba activate renew
make all
```

> [!WARNING]
> This is not a software package and we do not expect that users will be run it "out of the box"; we provide this code to fulfill goals and obligations related to transparency and data reproducibility. All the code here is dependent on the particular computing setup we used to generate and export the data. Users will have to change things to accommodate their own setups, including paths embedded in the various scripts and how scripts are called on by job scheduling software, etc. In other words, we do not expect this to work without modifications.

> [!CAUTION]
> We can make no guarantees on the environment solving on all systems. For instance, there does not appear to be a wrf-python version which supports OSX-arm64 systems, and it is unclear if one is coming. We do not anticipate that the mamba environment will be able to solve on these systems.

If your make setup fails you can follow the commands in the make file one by one to set your environment up correctly:

## Pre-commit Workflow

For this repository `pre-commit` is set up to track changes to to the `src/` folder
where all the core functionality of the project lives. All notebooks will be formatted
using `isort`, `black`, and `ruff`.

> [!TIP]
> You can run `make all` before you commit your code every time to make sure that your
> code passes `pre-commit`, tests, and security checks before you run `git commit`

A workflow usually looks like this:
  1) develop your code
  2) run `make all` to format your notebooks and scripts, run tests, check security, etc.
  3) add your code `git add <changed files here>` (fewer is better, until you get the hang of how to write code that adhere's to PEP8 style guides)
  4) attempt to commit `git commit -m "my message"`
    4a) this will trigger `pre-commit` to run on your code and make a bunch of changes
    4b) after it runs, add your files again, fix anything it says you need to fix, then try again
    4c) repeat until it's clean
  5) done! Now you can push `git push`

> [!WARNING]
> Detect Secrets can confuse images left in notebooks as secrets due to how they are formatted as strings in the `.json` version of the notebook (which is what the parser reads). Please be sure to clear all notebook contents for repeatable notebooks in the `notebooks/` folder.

## Generating PV and Wind Power Energy Potential Profiles

To use the scripts in this repo and follow along with our process, please see the [documentation](https://github.com/Eagle-Rock-Analytics/renewable-profiles/blob/reorganize/scripts/README.md).
