# -*- coding: utf-8 -*-
"""
@authors: Nadia Bloemendaal, nadia.bloemendaal@vu.nl, Marjolein Ribberink, m.r.s.ribberink@vu.nl

This module is part of the STORM model

For more information, please see 
Bloemendaal, N., Haigh, I.D., de Moel, H. et al. 
Generation of a global synthetic tropical cyclone hazard dataset using STORM. 
Sci Data 7, 40 (2020). https://doi.org/10.1038/s41597-020-0381-2

This is the script for generating the genesis matrices in Python 3 (with Cartopy)

Copyright (C) 2020 Nadia Bloemendaal. All versions released under the GNU General Public License v3.0
"""

import numpy as np
import os
import sys
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep
import pandas as pd 
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from preprocessing import BOUNDARIES_BASINS

dir_path=os.path.dirname(os.path.realpath(sys.argv[0]))
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

land_shp_fname = shpreader.natural_earth(resolution='50m',
                                       category='physical', name='land')

land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
land = prep(land_geom)

def is_land(x, y):
    return land.contains(sgeom.Point(x, y))

print(is_land(150,20))

def create_mask(idx):
    stepsize=10
    lat0,lat1,lon0,lon1=BOUNDARIES_BASINS(idx)
    x=int(abs(lon1-lon0)*stepsize)
    y=int(abs(lat1-lat0)*stepsize)
    if lon0<180: #south pacific
        lon_grid,lat_grid=np.mgrid[lon0:lon1:complex(0,x),lat0:lat1:complex(0,y)]        
    else:  
        lon_grid,lat_grid=np.mgrid[lon0-360:lon1-360:complex(0,x),lat0:lat1:complex(0,y)]
    
    mask=np.ones((len(lon_grid[0]),len(lon_grid)))
    for i in range(len(lon_grid)):
        for j in range(len(lon_grid[i])):
            mask[j][i]=is_land(lon_grid[i][j],lat_grid[i][j])

    mask=np.flipud(mask)
    
    return mask

def create_5deg_grid(locations,month,idx):
    
    step=5.

    lat0,lat1,lon0,lon1=BOUNDARIES_BASINS(idx)
    if idx==1:    
        lonspace=np.linspace(lon0,360.,int(abs(lon0-360.)/step)+1)
    else:
        lonspace=np.linspace(lon0,lon1,int(abs(lon0-lon1)/step)+1)
    
    latspace=np.linspace(lat0,lat1,int(abs(lat0-lat1)/step)+1)
    
    
    lat_list=[locations[month][i][0] for i in range(len(locations[month])) if (lat0<=locations[month][i][0]<=lat1 and lon0<=locations[month][i][1]<=lon1)]
    lon_list=[locations[month][i][1] for i in range(len(locations[month])) if (lat0<=locations[month][i][0]<=lat1 and lon0<=locations[month][i][1]<=lon1)]
    
    df=pd.DataFrame({'Latitude':lat_list,'Longitude':lon_list})
    
    to_bin=lambda x:np.floor(x/step)*step
    df["latbin"]=df.Latitude.map(to_bin)
    df["lonbin"]=df.Longitude.map(to_bin)
    groups=df.groupby(["latbin","lonbin"])
    count_df=pd.DataFrame({'count':groups.size()}).reset_index()
    counts=count_df["count"]       
    latbin=groups.count().index.get_level_values('latbin')
    lonbin=groups.count().index.get_level_values('lonbin')
    count_matrix=np.zeros((len(latspace),int(abs(lon0-lon1)/step)+1))
    
    for lat,lon,count in zip(latbin,lonbin,counts):
          i=latspace.tolist().index(lat)
          j=lonspace.tolist().index(lon)
          count_matrix[i,j]=count
    
    return count_matrix
  
def create_1deg_grid(delta_count_matrix,idx,month):
    step=5.
    
    lat0,lat1,lon0,lon1=BOUNDARIES_BASINS(idx)

    latspace=np.linspace(lat0,lat1,int(abs(lat0-lat1)/step)+1)
    lonspace=np.linspace(lon0,lon1,int(abs(lon0-lon1)/step)+1)
            
    xg=int(abs(lon1-lon0))
    yg=int(abs(lat1-lat0))
    xgrid,ygrid=np.mgrid[lon0:lon1:complex(0,xg),lat0:lat1:complex(0,yg)]
    points=[]
    for i in range(len(lonspace)):
        for j in range(len(latspace)):
            points.append((lonspace[i],latspace[j]))
     
    values=np.reshape(delta_count_matrix.T,int(len(lonspace))*int(len(latspace)))
    grid=griddata(points,values,(xgrid,ygrid),method='cubic')
    grid=np.transpose(grid)
    grid=np.flipud(grid)
    grid[grid<0]=0
            

    #overlay data with a land-sea mask
    mdata=create_mask(idx)
    coarseness=10
    mdata_coarse=mdata.reshape((mdata.shape[0]//coarseness,coarseness,mdata.shape[1]//coarseness,coarseness))
    mdata_coarse=np.mean(mdata_coarse,axis=(1,3))
                         
    (x,y)=mdata_coarse.shape
    
    for i in range(0,x):
        for j in range(0,y):
            if mdata_coarse[i,j]>0.50:
                grid[i,j]='nan'
                
    plt.imshow(grid)
    plt.show()
        

    return grid


def Change_genesis_locations():
    monthsall=[[6,7,8,9,10,11],[6,7,8,9,10,11],[4,5,6,9,10,11],[1,2,3,4,11,12],[1,2,3,4,11,12],[5,6,7,8,9,10,11]]    
    locations=np.load(os.path.join(__location__,'../STORM_variables','GEN_LOC.npy'),allow_pickle=True,encoding='latin1').item()

     #Saving in their own folder
    grid_gen_path = os.path.join(dir_path,'STORM_variables','GRID_GEN')
    if os.path.isdir(grid_gen_path)==False:
        os.mkdir(grid_gen_path) 

    for idx in range(0,6):
        print(idx)
        for month in monthsall[idx]:                
            matrix_dict=create_5deg_grid(locations[idx],month,idx)

            genesis_grids=create_1deg_grid(matrix_dict,idx,month)

            
    
            np.savetxt(os.path.join(grid_gen_path,'GRID_GENESIS_MATRIX_{}_{}.txt'.format(idx,month)),genesis_grids)
