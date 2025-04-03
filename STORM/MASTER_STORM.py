# -*- coding: utf-8 -*-
"""
@authors: Nadia Bloemendaal, nadia.bloemendaal@vu.nl, Marjolein Ribberink, m.r.s.ribberink@vu.nl

For more information, please see 
Bloemendaal, N., Haigh, I.D., de Moel, H. et al. 
Generation of a global synthetic tropical cyclone hazard dataset using STORM. 
Sci Data 7, 40 (2020). https://doi.org/10.1038/s41597-020-0381-2

This is the STORM model master program

Copyright (C) 2020 Nadia Bloemendaal. All versions released under the GNU General Public License v3.0.
"""

#Master storm running - one at a time

import numpy as np
import pandas as pd

#Custom made modules
from SELECT_BASIN import Gen_basin

from SAMPLE_STARTING_POINT import Startingpoint
from SAMPLE_TC_MOVEMENT import TC_movement
from SAMPLE_TC_PRESSURE import TC_pressure

import os
import sys
dir_path=os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

import time
start_time=time.time()

#Preparing own folder
storm_data_path = os.path.join(dir_path,'STORM_data')
if os.path.isdir(storm_data_path)==False:
    os.mkdir(storm_data_path) 
#==============================================================================
# Step 1: Define basin and number of years to run
#==============================================================================
#please set basin (EP,NA,NI,SI,SP,WP)
basin='NA'
loop=0 #ranges between 0 and 9 to simulate slices of 1000 years

total_years=100 #set the total number of years you'd like to simulate

TC_data = pd.DataFrame(
    columns=[
        "year",
        "month",
        "time",
        "track_id",
        "timeStep",
        "basinID",
        "lat",
        "lon",
        "minP",
        "Vmax",
        "Rmax",
        "cat",
        "landfall",
        "dist_land",
    ]
)

times=np.empty(total_years+1)
times[0]=start_time

for year in range(0,total_years):
    storms_per_year,genesis_month,genesis_day,genesis_hour,idx=Gen_basin(basin) 
    
    if storms_per_year>0:
            #==============================================================================
            # Step 3: Generate (list of) genesis locations
            #=============================================================================
            lon_genesis_list,lat_genesis_list=Startingpoint(storms_per_year,genesis_month,genesis_day,genesis_hour,idx) 
            
            #==============================================================================
            # Step 4: Generate initial conditions    
            #==============================================================================
            
            latlist,lonlist,landfalllist,monthlist=TC_movement(lon_genesis_list,lat_genesis_list,genesis_month,genesis_day,genesis_hour,idx)
            
            TC_data=TC_pressure(idx,latlist,lonlist,landfalllist,year,storms_per_year,monthlist,genesis_day,genesis_hour,TC_data)
            print("done year "+str(year))
            times[year+1]=time.time()
            print(storms_per_year, times[year+1]-times[year])

TC_data.to_csv(
    os.path.join(storm_data_path, f'STORM_DATA_{basin}_{total_years}_YEARS_{loop}.csv'),
)

end_time=time.time()
print(end_time-start_time)


