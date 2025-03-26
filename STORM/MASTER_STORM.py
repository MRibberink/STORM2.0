# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 13:11:11 2025

@author: wul339
"""

#Master storm running - one at a time

import numpy as np

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
basin='SI'
loop=0 #ranges between 0 and 9 to simulate slices of 1000 years

total_years=20 #set the total number of years you'd like to simulate

TC_data=[] 

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
            
         
TC_data=np.array(TC_data)

    
np.savetxt(os.path.join(storm_data_path,'STORM_DATA_'+str(basin)+'_'+str(total_years)+'_YEARS_'+str(loop)+'.txt'),TC_data,fmt='%5s',delimiter=',')

end_time=time.time()
print(end_time-start_time)


