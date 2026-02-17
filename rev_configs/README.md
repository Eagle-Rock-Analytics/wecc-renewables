# ReV configurations

This folder contains: 

1. The ReV configuration JSONs which define installation and software parameters for a given installation (eg, utility PV) 
2. The `_projectpoints` CSVs which define the grid cells suitable for a given installation for the PV and wind power simulations (ie, if the grid cell is not in that CSV, then ReV did not run our output any information at that grid cell). `d03` refers to the CA-wide (3 km) domain and `d02` refers to the WECC-wide (9 km) domain. 

The project points CSVs are made by running `notebooks/all_03_mk_rev_project_points.ipynb`. The land use/land cover restrictions defined there fulfilled our obligations related to installation siting work and might not reflect the restrictions you want to implement. Please read our [data guide](https://wfclimres.s3.amazonaws.com/era/data-guide_pv-wind.pdf) for more information on the land use/cover restrictions we employed.