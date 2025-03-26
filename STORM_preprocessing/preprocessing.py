# -*- coding: utf-8 -*-
"""
@authors: Nadia Bloemendaal, nadia.bloemendaal@vu.nl, Marjolein Ribberink, m.r.s.ribberink@vu.nl

This module is part of the STORM model

For more information, please see 
Bloemendaal, N., Haigh, I.D., de Moel, H. et al. 
Generation of a global synthetic tropical cyclone hazard dataset using STORM. 
Sci Data 7, 40 (2020). https://doi.org/10.1038/s41597-020-0381-2

Functions described here are part of the data pre-processing. 

Copyright (C) 2020 Nadia Bloemendaal. All versions released under GNU General Public License v3.0
"""

import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import os
import sys
dir_path=os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

#Basin indices: 
# 0 = EP = Eastern Pacific
# 1 = NA = North Atlantic
# 2 = NI = North Indian
# 3 = SI = South Indian
# 4 = SP = South Pacific
# 5 = WP = Western Pacific

def BOUNDARIES_BASINS(idx):
    if idx==0: #Eastern Pacific
        lat0,lat1,lon0,lon1=5,90,180,285
    if idx==1: #North Atlantic
        lat0,lat1,lon0,lon1=5,90,255,390
    if idx==2: #North Indian
        lat0,lat1,lon0,lon1=5,90,30,100
    if idx==3: #South Indian
        lat0,lat1,lon0,lon1=-90,-5,10,105
    if idx==4: #South Pacific
        lat0,lat1,lon0,lon1=-90,-5,105,240
    if idx==5: #Western Pacific
        lat0,lat1,lon0,lon1=5,90,100,180
    
    return lat0,lat1,lon0,lon1

def find_date_time(t):
    """
    Find the datetime corresponding to the timestep in the IBTrACS dataset
    input:
        t: timestep (in days since 17-11-1858)
    output:
        mo: month (1-12)
        da: day (1-31)
        hr: hour (0-23)
    """
    
    start=datetime(1858,11,17,0)
    days=t
    delta=timedelta(days)
    offset=delta+start
    mo=offset.month
    da=offset.day
    hr=offset.hour
    
    return mo,da,hr

def find_basin(name):
    """
    Give a basin index to the basin name
    Input: 
        name: name of basin, EP=Eastern Pacific, NA=North Atlantic, NI=North Indian, SI=South Indian, SP=South Pacific, WP= Western Pacific
    Output:
        index: value between 0 and 5 corresponding to the basin names
    """
    for basinname,index in zip(['EP','NA','NI','SI','SP','WP'],range(0,6)):
        if name==basinname:
           return index
       
def interpolate(dataset):
    """
    Interpolate the data to 3-hourly values
    Input: 
        dataset: the respective dataset
    Output:
        dataset: the interpolated dataset
    
    """
    
    if np.any(np.isnan(dataset))==True and len([x for x,v in enumerate(dataset) if np.isnan(v)==False])>1:  
        #if there are nans in the dataset and there is more than 1 real value
        
        ind=[x for x,v in enumerate(dataset) if np.isnan(v)==False] #indices with values
        
        ind1=[x+ind[0] for x,v in enumerate(dataset[ind[0]:ind[-1]]) if np.isnan(v)==True] #indices with no values
        
        
        val=[v for v in dataset if np.isnan(v)==False] #real values
        
        if len(ind1)>0:
        
            interlist=np.interp(ind1,ind,val) #interpolate between the known values of dataset
            
            for ii,jj in zip(ind1,range(len(ind1))):
                dataset[ii]=interlist[jj]  
                
        else:
            dataset=dataset
    
    return dataset
        
def check_timelist(tlist):
    """
    Check whether the consecutive time steps are 3 hours apart
    Input:
        tlist: list of time steps
    Output:
        print tlist if the consecutive time steps are not 3 hours apart
    
    """

    for ii in range(1,len(tlist)):
        if tlist[ii]-tlist[ii-1]!=0.125:
            print(tlist)
            
def convert_wind_speed(wind,agency):
    """
    Convert IBTrACS wind speed to 10-min sustained wind speed. 
    From the official IBTrACS documentation:
        Maximum sustained wind speed from the WMO agency for the current location.
        NO adjustment is made for differences in wind speed averaging periods.
        hurdat/atcf = North Atlantic - U.S. Miami (NOAA NHC) - 1-minute winds
        tokyo = RSMC Tokyo (JMA) - 10-minute
        newdelhi = RSMC New Delhi (IMD) - 3-minute
        reunion = RSMC La Reunion (MFLR) - 10 minute
        bom = Australian TCWCs (TCWC Perth, Darwin, Brisbane) - 10-minute
        nadi = RSMC Nadi (FMS) - 10 minute
        wellington = TCWC Wellington (NZMS) - 10-minute
    
    Input: 
        wind: wind speed 
        agency: name of agency
    Output:
        wind_conv: converted wind       
    """
    
    if agency=='hurdat_epa' or agency=='hurdat_atl' or agency=='newdelhi' or agency=='atcf': #1-minute wind
        wind_conv=0.88*wind
    else:
        wind_conv=wind
        
    return wind_conv   

def extract_data(data,end_year):
    """
    Extract different variables from IBTrACS dataset.
    Input:
        *data*: dataset (IBTrACS)
    Output: 
        *LATLIST_INTERP.npy*: interpolated values of latitude, where each entry in the dictionary stands for one TC
        *LONLIST_INTERP.npy*: interpolated values of longitude (0-360 deg)
        *WINDLIST_INTERP.npy*: interpolated values of wind (m/s)
        *PRESLIST_INTERP.npy*: interpolated values of pressure (hPa)
        *RMAXLIST_INTERP.npy*: interpolated values of Rmax (km)
        *MONTHLIST_INTERP.npy*: Month of TC genesis
        *BASINLIST_INTERP.npy*: Basin of TC genesis
        *YEARLIST_INTERP.npy*: Year of TC genesis
    """
    
    basin=data.basin.values
    years=data.season.values
    storms=data.storm.values
    wind=data.wmo_wind.values
    wind=wind*0.51444444    #convert from knots to m/s
    pres=data.wmo_pres.values
    time=data.time.values
    latitude=data.lat.values
    longitude=data.lon.values
    rmax=data.usa_rmw.values*1.85200 #convert from nm to km    
    wmo_agency=data.wmo_agency.values    
    nature=data.nature.values

    TS_cutoff=15.8
    """Create a npy list for each of the items"""    
    latlist={i:[] for i in range(len(storms))}
    lonlist={i:[] for i in range(len(storms))}
    timelist={i:[] for i in range(len(storms))}
    windlist={i:[] for i in range(len(storms))}
    preslist={i:[] for i in range(len(storms))}
    monthlist={i:[] for i in range(len(storms))}
    basinlist={i:[] for i in range(len(storms))}
    basinlist1={i:[] for i in range(len(storms))}
    rmaxlist={i:[] for i in range(len(storms))}
    yearlist={i:[] for i in range(len(storms))} 
    
    
    for i in range(len(storms)):
        if years[i]<=end_year:            
            idx=[x for x,v in enumerate(wmo_agency[i]) if len(v)>1.]            
            if len(idx)>0: #there is data on wind speed and associated agency. 
                #Note that if the wind list solely consists of 'nan',there would be no associated agency. 
                #And that if there is a wind reading, there is also an associated agency.                
                wind_conv=convert_wind_speed(wind[i],wmo_agency[i][idx[0]].decode("utf-8")) #convert to 10 min average
                    
                if np.all(np.isnan(wind_conv))==False and np.nanmax(wind_conv)>=TS_cutoff:
                    
                    """We consider the timesteps between the first and the last moment of maximum wind speed > 15.8 m/s (equal to a tropical storm)"""  
                    ind=[x for x,v in enumerate(wind_conv) if v>=TS_cutoff] #find indices of wind values over cutoff    
                    nature_list=[x.decode("utf-8") for x in nature[i]] #decode the IBTrACS nature list from bytes to strings
                    
                    if 'ET' in nature_list:
                        et_idx=nature_list.index('ET') #if ET in nature list, find index of first occurrence
                        
                        if et_idx>ind[0]:
                            end=max(ii for ii in ind if ii<et_idx) #if storm does not starts as an ET, find last idx before becoming ET
                            ind=ind[:end+1] #set index end to end of tropical phase
                        else:
                            ind=[]
                        
                    if len(ind)>0. and basin[i][ind[0]].decode("utf-8")!='SA': #exclude the south atlantic                    
                        j0=ind[0] #first location at which storm is tropical storm                        
                        if len(ind)>1: #the storm spans multiple time steps               
                            j1=ind[-1] #last location at which storm is tropical storm                                              
                        else:
                            j0=ind[0]
                            j1=j0                        
                        
                        monthlist[i].append(find_date_time(time[i][ind[0]])[0]) #find month of each storm start
                        basinlist[i].append(find_basin(basin[i][ind[0]].decode("utf-8"))) #find basin of each storm at start
                        yearlist[i].append(years[i]) #find season of each storm start
                        
                        idx=[x for x,v in enumerate(wmo_agency[i]) if len(v)>1.]#find index of the WMO agency if it exists              
                        time_idx=[j0+x for x,v in enumerate(time[i][j0:j1+1]) if round(v,3)%0.125==0.] #if timestep at 3/6/9/12, add idx to time idx list                      
                        new_list=np.intersect1d(ind,time_idx) #find all times there is both agency data and it's at 3/6/9/12
             
                        if len(new_list)>1.: #if there is more than 1 location with 3/6/9/12 agency data, find idx of start/end in timelist
                            n0=time_idx.index(new_list[0]) 
                            n1=time_idx.index(new_list[-1])
                           
                            new_time=time_idx[n0:n1+1] #define new idxlist as times with data on 3/6/9/12
                            
                            j_idx=0
                            while j_idx<len(new_time):  #while index remains less than number of timesteps                              
                                j=new_time[j_idx]        
                                latlist[i].append(latitude[i][j]) #find latitude of this storm at that time
                                
                                if longitude[i][j]<0.:
                                    longitude[i][j]+=360. #convert to 360 based lon
                                
                                lonlist[i].append(longitude[i][j]) #find longitude of this storm at that time
                                timelist[i].append(round(time[i][j],3)) #add value of time to list
                                windlist[i].append(wind_conv[j]) #find wspd of this storm at that time                                
                                preslist[i].append(pres[i][j]) #find central pressure of this storm at that time
                                rmaxlist[i].append(rmax[i][j]) #find rmax of this storm at that time
                                j_idx=j_idx+1        
                            
                            check_timelist(timelist[i])          #check that timesteps are 3h apart     
                            
    
    for i in range(len(storms)):
        if len(latlist[i])>0: 
            if (basinlist[i][0]==3 or basinlist[i][0]==4):
                bas_bounds=BOUNDARIES_BASINS(3)
                if lonlist[i][0]<=bas_bounds[3]:
                    basinlist1[i].append(3)
                else:
                    basinlist1[i].append(4)
            else:
                basinlist1[i].append(basinlist[i][0])                           
    
                         
    """This part is for interpolating the missing values"""
    lat_int={i:[] for i in range(len(storms))}
    lon_int={i:[] for i in range(len(storms))}    
    wind_int={i:[] for i in range(len(storms))}
    pres_int={i:[] for i in range(len(storms))}    
    rmax_int={i:[] for i in range(len(storms))}    
    
    for i in range(len(latlist)):
        if len(latlist[i])>0: 
            if np.isnan(windlist[i][-1])==True:  #if the last timestep is a nan,grab only till the one before that
                lat_int[i]=interpolate(latlist[i][:-1])
                lon_int[i]=interpolate(lonlist[i][:-1])                
                wind_int[i]=interpolate(windlist[i][:-1])
                pres_int[i]=interpolate(preslist[i][:-1])        
                rmax_int[i]=interpolate(rmaxlist[i][:-1])
            else:
                lat_int[i]=interpolate(latlist[i])
                lon_int[i]=interpolate(lonlist[i])                
                wind_int[i]=interpolate(windlist[i])
                pres_int[i]=interpolate(preslist[i])        
                rmax_int[i]=interpolate(rmaxlist[i])
            
    """
    Create a directory and save the interpolated datasets as .npy files. These files will be used later on 
    and also come in handy when plotting IBTrACS data
    """       
    # Specify the path of the directory
    ib_path = os.path.join(dir_path,'IBTrACS_extract')
    
    # Check if the directory exists. If not, create it
    if os.path.isdir(ib_path)==False:
        os.mkdir(ib_path)

    
    np.save(os.path.join(ib_path,'LATLIST_INTERP.npy'),lat_int)
    np.save(os.path.join(ib_path,'LONLIST_INTERP.npy'),lon_int)
    np.save(os.path.join(ib_path,'TIMELIST_INTERP.npy'),timelist)
    np.save(os.path.join(ib_path,'WINDLIST_INTERP.npy'),wind_int)
    np.save(os.path.join(ib_path,'PRESLIST_INTERP.npy'),pres_int)
    np.save(os.path.join(ib_path,'RMAXLIST_INTERP.npy'),rmax_int)
    np.save(os.path.join(ib_path,'MONTHLIST_INTERP.npy'),monthlist)
    np.save(os.path.join(ib_path,'BASINLIST_INTERP.npy'),basinlist1)
    np.save(os.path.join(ib_path,'YEARLIST_INTERP.npy'),yearlist)
    

def TC_variables():
    """
    Extract the important variables. 
    """
    ib_path=os.path.join(dir_path,'IBTrACS_extract')
    try:
        latlist=np.load(os.path.join(ib_path,'LATLIST_INTERP.npy'),allow_pickle=True).item()
        lonlist=np.load(os.path.join(ib_path,'LONLIST_INTERP.npy'),allow_pickle=True).item()
        windlist=np.load(os.path.join(ib_path,'WINDLIST_INTERP.npy'),allow_pickle=True).item()
        preslist=np.load(os.path.join(ib_path,'PRESLIST_INTERP.npy'),allow_pickle=True).item()
        rmaxlist=np.load(os.path.join(ib_path,'RMAXLIST_INTERP.npy'),allow_pickle=True).item()
        monthlist=np.load(os.path.join(ib_path,'MONTHLIST_INTERP.npy'),allow_pickle=True).item()
        yearlist=np.load(os.path.join(ib_path,'YEARLIST_INTERP.npy'),allow_pickle=True).item()
        basinlist=np.load(os.path.join(ib_path,'BASINLIST_INTERP.npy'),allow_pickle=True).item()
    except FileNotFoundError:
        print('Files do not exist in '+str(__location__)+', please check directory')
        return 
    
    monthsall=[[6,7,8,9,10,11],[6,7,8,9,10,11],[4,5,6,9,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]
   
    months={i:[] for i in range(0,6)}
    years={i:[] for i in range(0,6)}
    genesis_wind={i:[] for i in range(0,6)}
    genesis_pressure={i:[] for i in range(0,6)}
    genesis_dpres={i:[] for i in range(0,6)}
    genesis_pres_var={i:[] for i in range(0,6)}    
    genesis_loc={i:[] for i in range(0,6)}
    poisson={i:[0] for i in range(0,6)}    #Poisson genesis parameters (avg number of TC formations per year)
    genesis_poisson=[] 
    
    
    track={i:[] for i in range(0,6)}     #All info for the track. 
    #0=dlat0 (backward change in latitude),1=dlat1 (forward change in latitude),2=dlon0,3=dlon1,4=lat,5=lon
    pressure={i:[] for i in range(0,6)}     #All info for the pressure change
    #0=dp0 (backward change in pressure), 1=dp1 (forward change in pressure), 2=pressure, 3=latitude, 4=longitude, 5=month
    for i in range(0,6):
        track[i]={j:[] for j in range(0,6)}
        pressure[i]={j:[] for j in range(0,6)}
        
    for idx in range(0,6):
        genesis_wind[idx]={i:[] for i in monthsall[idx]}        #genesis wind speed
        genesis_pressure[idx]={i:[] for i in monthsall[idx]}    #genesis pressure
        genesis_dpres[idx]={i:[] for i in monthsall[idx]}       #genesis change in pressure
        genesis_pres_var[idx]={i:[] for i in monthsall[idx]}    #genesis variables for pressure.
        genesis_loc[idx]={i:[] for i in monthsall[idx]}         #genesis location
        
    for i in range(len(latlist)): #for all storms in the list
        if len(latlist[i])>0: #if there is actually data for the storms
            idx=basinlist[i][0]     #this is the index for each of the basins. 0=EP, 1=NA, 2=NI, 3=SI, 4=SP, 5=WP
            month=monthlist[i][0]   #genesis month
            
            if month in monthsall[idx]: #if this storm formed inside hurricane season in its basin
                months[idx].append(month) #add its formation month to the monthlist
                genesis_wind[idx][month].append(windlist[i][0]) #add its windspeed at formation to genesis formation list
                genesis_dpres[idx][month].append(preslist[i][1]-preslist[i][0]) #add the first change in pressure to the genesis change in pressure list
                genesis_pressure[idx][month].append(preslist[i][0]) #add its first pressure to the genesis pressure list
                genesis_loc[idx][month].append([latlist[i][0],lonlist[i][0]]) #add its formation coordinates to the genesis coordinates list
                poisson[idx][0]+=1 #add 1 to the number of tc formations in a given year
                years[idx].append(int(yearlist[i][0]))
                
                for j in range(1,len(latlist[i])-1): #for the length of the storm lifetime
                    track[0][idx].append(latlist[i][j]-latlist[i][j-1]) #record change of latitude before each timestep
                    track[1][idx].append(latlist[i][j+1]-latlist[i][j]) #record change of latitude after each timestep
                    track[2][idx].append(lonlist[i][j]-lonlist[i][j-1]) #record change of longitude before each timestep
                    track[3][idx].append(lonlist[i][j+1]-lonlist[i][j]) #record change of longitude after each timestep
                    track[4][idx].append(latlist[i][j]) #record latitude position
                    track[5][idx].append(lonlist[i][j]) #record longitude position
                
                for j in range(1,len(preslist[i])-1):
                    if np.isnan(preslist[i][j-1])==False and np.isnan(preslist[i][j])==False and np.isnan(preslist[i][j+1])==False:
                        #if the previous, current, and next pressure values exist:
                        pressure[0][idx].append(preslist[i][j]-preslist[i][j-1]) #record change of pressure before each timestep
                        pressure[1][idx].append(preslist[i][j+1]-preslist[i][j]) #record change of pressure after each timestep
                        pressure[2][idx].append(preslist[i][j]) #record pressure at that timestep
                        pressure[3][idx].append(latlist[i][j]) #record latitude position
                        pressure[4][idx].append(lonlist[i][j]) #record longitude position
                        pressure[5][idx].append(month) #record month of formation
                 
    for idx in range(0,6): # 
        genesis_poisson.append(round(poisson[idx][0]/len(np.unique(years[idx])),1)) #find average number of storms per year
        
        dp0_neg,dp0_pos=[],[]
        for j in range(len(pressure[0][idx])):
            if pressure[0][idx][j]<0.: #if the storm strengthened getting to this timestep
                dp0_neg.append(pressure[0][idx][j]) #add change in pressure to "strengthening" list
            elif pressure[0][idx][j]>0:#if the storm weakened getting to this timestep
                dp0_pos.append(pressure[0][idx][j]) #add change in pressure to "weakening" list
        
        pneg=np.percentile(dp0_neg,1) #Strengthening threshold
        ppos=np.percentile(dp0_pos,99) #Weakening threshold
        
        for month in monthsall[idx]: #for every month in hurricane season 
            dplist=[v for v in genesis_dpres[idx][month] if np.isnan(v)==False and v>-1000.] #add dpres to dplist if the change in pressure exists and isn't an error
            plist=[v for v in genesis_pressure[idx][month] if np.isnan(v)==False and v>0.] #add pres to plist if the change in pressure exists and isn't negative
            
            mudp0,stddp0=stats.norm.fit(dplist) #fit normal distribution to genesis pressure change data
            mupres,stdpres=stats.norm.fit(plist) #fit normal distribution to genesis pressure data
            
            genesis_pres_var[idx][month]=[mupres,stdpres,mudp0,stddp0,pneg,ppos] #save monthly values of normal distributions and weakening/strengthening thresholds per basin
            
                
    radius={i:[] for i in range(0,3)}
    for i in range(len(rmaxlist)):
        if len(rmaxlist[i])>0.:
            for j in range(len(rmaxlist[i])):
                if np.isnan(rmaxlist[i][j])==False and np.isnan(preslist[i][j])==False:                    
                    if preslist[i][j]<=920.:
                        radius[0].append(rmaxlist[i][j])
                    elif preslist[i][j]>920. and preslist[i][j]<=960.:
                        radius[1].append(rmaxlist[i][j])
                    elif preslist[i][j]>960.:
                        radius[2].append(rmaxlist[i][j])               
     
    print(genesis_poisson)

    #Save all the files in a variable directory
    # Specify the path of the directory
    storm_var_path = os.path.join(dir_path,'STORM_variables')
    
    # Check if the directory exists. If not, create it
    if os.path.isdir(storm_var_path)==False:
        os.mkdir(storm_var_path)
        
    np.save(os.path.join(storm_var_path,'RMAX_PRESSURE.npy'),radius) #radius of maximum wind
    np.savetxt(os.path.join(storm_var_path,'POISSON_GENESIS_PARAMETERS.txt'),genesis_poisson) #average number of formations per year per basin
    np.save(os.path.join(storm_var_path,'TC_TRACK_VARIABLES.npy'),track) #variables describing lat and lon change as well as position
    np.save(os.path.join(storm_var_path,'TC_PRESSURE_VARIABLES.npy'),pressure) #variables describing pressure change and values, as well as storm position
    np.save(os.path.join(storm_var_path,'DP0_PRES_GENESIS.npy'),genesis_pres_var) #values of genesis pressure and genesis pressure change

    np.save(os.path.join(storm_var_path,'DP_GEN.npy'),genesis_dpres) #first change in pressure
    np.save(os.path.join(storm_var_path,'PRES_GEN.npy'),genesis_pressure) #pressure at genesis
    np.save(os.path.join(storm_var_path,'GEN_LOC.npy'),genesis_loc)      #location at genesis
    np.save(os.path.join(storm_var_path,'GENESIS_WIND.npy'),genesis_wind)    #wind speed at genesis
    np.save(os.path.join(storm_var_path,'GENESIS_MONTHS.npy'),months) #month of the year at genesis
    np.save(os.path.join(storm_var_path,'GENESIS_YEARS.npy'),years) #year at genesis
