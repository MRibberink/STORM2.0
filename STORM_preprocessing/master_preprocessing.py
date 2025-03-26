# -*- coding: utf-8 -*-
"""
This module is part of the STORM model

For more information, please see 
Bloemendaal, N., Haigh, I.D., de Moel, H. et al. 
Generation of a global synthetic tropical cyclone hazard dataset using STORM. 
Sci Data 7, 40 (2020). https://doi.org/10.1038/s41597-020-0381-2

This is the master program for the data pre-processing. 
This script is made for the data pre-processing of the IBTrACS dataset. For other input datasets, 
please change the syntax accordingly. 
This script will generate multiple output files in the working directory. 

The script is split up in multiple cells, with each cell running a specific part of the data
preprocessing. We advise you to read what is done per cell, and to run per cell rather than
the whole script at once. To keep the script as clean as possible, most of the code has been placed in 
seperate functions in the "preprocessing"-module. 
 
Copyright (C) 2020 Nadia Bloemendaal. All versions released under GNU General Public License v3.0
"""
import xarray as xr
import preprocessing
import coefficients
import environmental
import genesis_matrix_python3
import Make_land_ocean_mask 
import os
import sys
dir_path=os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

#%%
"""
Open the IBTrACS dataset. The IBTrACS dataset can be downloaded via https://www.ncdc.noaa.gov/ibtracs/
or ftp://eclipse.ncdc.noaa.gov/pub/ibtracs/v04r00/provisional/netcdf/ 
Here, we use the global dataset (version 4) from 1980-2017. End year is the last year of TC data you
want to include in the dataset
"""
end_year=2024
data=xr.open_dataset(os.path.join(__location__,'../IBTrACS.since1980.v04r01.nc'),decode_times=False)
preprocessing.extract_data(data,end_year)
data.close()

#%%
"""
Extract the important parameters necessary for the fitting of the regression formulas (and other parts
of the storm model
"""
preprocessing.TC_variables()
 
#%%
"""
Calculate the coefficients for the track and pressure regression formulas
"""
coefficients.track_coefficients()

#%%
"""
Calculate the monthly mean SST and MSLP fields
ECMWF has monthly mean MSLP fields available via the CDS (cds.climate.copernicus.eu)
These should be downloaded and stored as "Monthly_mean_MSLP.nc" and "Monthly_mean_SST.nc". 
Warning, this section takes a long while to run. 
"""
    
data=xr.open_dataset(os.path.join(__location__,'../Monthly_mean_MSLP.nc'))
environmental.monthly_mean_pressure(data)
data.close()

data=xr.open_dataset(os.path.join(__location__,'../Monthly_mean_SST.nc'))
environmental.monthly_mean_sst(data)
data.close()

print("Loaded")

environmental.wind_pressure_relationship()

print("WPR")
#%%
environmental.calculate_MPI_fields()

print("MPI")

environmental.pressure_coefficients()

print("Coefficients")

#%%
"""
Generate the genesis matrices and land ocean masks
Genesis matrices are used to ensure that the storms form in the correct places (i.e. not over land) and
with the correct frequency for the area of the ocean, first in 5 degree boxes, then interpolated to 1 degree boxes.
The land-ocean masks tell STORM whether a forming storm is over land or not.
"""

genesis_matrix_python3.Change_genesis_locations()

Make_land_ocean_mask.run_landmask()
