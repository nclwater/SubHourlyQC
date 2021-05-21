# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:11:35 2019

@author: Roberto Villalobos

Automated checks for high precipitation hours 

"""

import os
os.chdir('D:/PhD/13. Scripts/phd-python-code/Intense_QC/')

import intense_Roberto_03 as ex
#import intense_.intense_CW as ex
import scipy.stats as stats2
import statistics as stats
import pandas as pd
import numpy as np
import zipfile
import math
import glob
from joblib import Parallel, delayed

"""
Function 1 of the SHQC process

Reads in subhourly data and examines it's frequency and resolution
Monthly periods with frequencies >= 30 minutes, or where the resolution is
1 mm (usually an indicator of tip counts not tip amounts in the data), are 
replaced with NAN. 

An output file is generated to keep track of changes. 
"""
def freqResChecker(input_file_zip_pair, outdir): #(file, outdir):
    
    # Reading from zipfile
    input_file = input_file_zip_pair[0]
    zip_folder = input_file_zip_pair[1]
    zf_in = zipfile.ZipFile(zip_folder, 'r')
    d = zf_in.open(input_file, mode='r')
    
    try:
        data = pd.read_csv(d)
        
        # Get datetime index
        data.index = pd.DatetimeIndex(data['ob_time'])
        # Get metadata in file
        station_id = data['id'][1]
        station_name = data['src_id'][1]
    except:
        print('Could not read data for '+input_file)
    
    d.close()
    zf_in.close()
    
        # read station data
    """ Old read method
    try:
        data = pd.read_csv(file)
        
        # Get datetime index
        data.index = pd.DatetimeIndex(data['ob_time'])
        # Get metadata in file
        station_id = data['id'][1]
        station_name = data['src_id'][1]
    except:
        print('Could not read data for '+file)
    """
    
    out = pd.DataFrame(columns = ["Station_id","Station_name","Removed","N_months","obs_rem","pobs_rem","mm_rem","pmm_rem"])

    og_data = data.copy()
    # Drop un-needed columns
    data = pd.DataFrame(data['accum'])
    data = data.dropna()
    
    # Calculate time difference vector to identify if data is 15-min or other type
    tdifs = data.index.to_series().diff()/np.timedelta64(1,'s')
    tdifs = tdifs.resample('M').apply(lambda x: stats2.mode(x)[0])
    
    # Calculate data resolution, by month
    res = data.resample('M').apply(lambda x: stats2.mode(x)[0])
    # Concatenate checks
    checks = pd.concat([tdifs,res], axis = 1) 
    
    # If time resolution is >= 30mins or if resolution == 0.5, flag
    checks['remove'] = np.where((checks['ob_time'] >= 1800)|
            ((checks['ob_time'] >= 1800)&(checks['accum'] == 0.5))|
            (checks['accum'] >= 1),1,0)
    
    # If data has been flagged, remove and write     
    if max(checks['remove']) > 0:
        # Create mask to remove data
        months = checks[checks['remove'] == 1].dropna().index
        mask = months.to_period('M')
        
        # Fill erroneous periods with 'NA' values and write
        clean_data = og_data.copy()
        clean_data['accum'] = np.where(clean_data.index.to_period('M').isin(mask), np.nan,clean_data['accum'])
        clean_data.to_csv(outdir+'/'+station_id+'.txt', index = False)
        
        # Calculate data removed
        og_mis = og_data.accum.isnull().sum()
        cl_mis = clean_data.accum.isnull().sum()
        H_rem = cl_mis - og_mis
        ph_rem = H_rem*100/og_data.shape[0] #percentage of data entries replaced with NAN
        
        # Calculate rainfall removed
        r_rem = og_data.accum.sum() - clean_data.accum.sum()
        pr_rem = r_rem*100/og_data.accum.sum()
        
        rem = 'True'
        n_mon = checks[checks['remove']>0].shape[0]
        
    #Otherwise write out un-changed data
    else:
        
        og_data.to_csv(outdir+'/'+station_id+'.txt', index = False)
        rem = 'False'
        n_mon = 0
        H_rem = 0
        ph_rem =0
        r_rem = 0
        pr_rem = 0
        
    out = out.append({"Station_id":station_id,
                      "Station_name":station_name,
                      "Removed":rem,
                     "N_months":n_mon,
                      "obs_rem":H_rem,
                      "pobs_rem":ph_rem,
                      "mm_rem":r_rem,
                      "pmm_rem":pr_rem}, ignore_index = True)
    
    return out

"""
Function 2 of the SHQC process

Here a threshold-based approach is used to examine rainfall data at a sub-hourly 
resolution to identify and discard suspicious periods. Hourly, 15-min and 1-min 
thresholds are used, as well as a fast-tipping frequency check. Suspicious 3-hr
periods are replaced with NAN in the subhourly data. 

A log file is prepared, and every removed interval is registered. 
"""
def subH_checkr(file, metadir, thresholds60, thresholds15, thresholds1, outdir):
    
    # Read station data
    try:
        data = pd.read_csv(file)
        
        # Get datetime index
        data.index = pd.DatetimeIndex(data['ob_time'])
        # Get metadata in file
        station_id = data['id'][1]
        station_name = data['src_id'][1]
    except:
        print('Could not read data for '+file)
    
    # Additional metadata
    try:
        its = ex.readIntense(metadir+station_id+".txt", only_metadata = False)
    except:
        print('Could not read metadata for '+file)
    
    # Copy original data, resample for hourly search
    og_data = data.copy()
    hourly = data['accum'].resample('H', closed = 'right', label = 'right').sum()
    
    # Check for suspect hours using monthly thresholds
    suspect = hourly.loc[((hourly.index.month == 1)&(hourly >= thresholds60[1])) | # or
                            ((hourly.index.month == 2)&(hourly >= thresholds60[2])) |
                            ((hourly.index.month == 3)&(hourly >= thresholds60[3])) |
                            ((hourly.index.month == 4)&(hourly >= thresholds60[4])) |
                            ((hourly.index.month == 5)&(hourly >= thresholds60[5])) |
                            ((hourly.index.month == 6)&(hourly >= thresholds60[6])) |
                            ((hourly.index.month == 7)&(hourly >= thresholds60[7])) |
                            ((hourly.index.month == 8)&(hourly >= thresholds60[8])) |
                            ((hourly.index.month == 9)&(hourly >= thresholds60[9])) |
                            ((hourly.index.month == 10)&(hourly >= thresholds60[10])) |
                            ((hourly.index.month == 11)&(hourly >= thresholds60[11])) |
                            ((hourly.index.month == 12)&(hourly >= thresholds60[12]))]
    
    # The checks only run if we have big hourly values
    if (len(suspect) > 0):
        output = pd.DataFrame(columns = ["Station_ID","Station_Name","Latitude",
                                         "Longitude","datetime","magnitude",
                                         "timestep","QC_status","removed",
                                         "Fast-tips","Large 15s","Large minutes"])
        
        #######################################################################
        
        # Iterate over suspect hours
        for hour in suspect.index:
            # Reset output parameters
            mag = hourly.loc[hour]
            fTips = 'False'
            removed = 'False'
            
            # Get month value
            month = hour.month
            
            # Extract 3 hour window and calculate time differential between tips
            # Extraction is made from un-touched data so deletions won't affect event extraction
            event = og_data[(hour - pd.DateOffset(hours=1)).strftime("%Y-%m-%d %H"):(hour + pd.DateOffset(hours=1)).strftime("%Y-%m-%d %H")].copy()
            
            # Get QC status of event:
            x = int(stats2.mode(event['q'])[0])
            if x== 1:
                event_q = 'S'
            elif x == 2:
                event_q = 'U'
            elif x == 3:
                event_q = 'M'
            else:
                event_q = ''
                
            event = event['accum']
            #event['Time stamp'] = pd.to_datetime(event['Time stamp'],format = '%d/%m/%Y %H:%M:%S')
            tdif = event.index.to_series().diff()/np.timedelta64(1,'s')
            tdif = tdif[~tdif.isna()]
            ###################################################################

            freq = None
            if event.shape[0] >3:
                freq = pd.infer_freq(event.index)  
                try:
                    if (freq == None )& (int(stats2.mode(tdif)[0]) == 900):
                        freq = '15T'
                    #elif(freq == None )& (stats.mode(tdif).total_seconds() == 1800):
                    #    freq = '30T'
                    #elif(freq == None )& (stats.mode(tdif).total_seconds() == 3600):
                    #    freq = '60T'
                except:
                    freq = None
            elif event.shape[0]==1:
                freq = '15T'
            elif event.shape[0]==2: # likely to be a single large 15-min value and a zero
                if (int(math.ceil(stats2.mode(tdif)[0] / 100.0)) * 100) >= 900:
                    freq = '15T'
            elif event.shape[0]==3:
                if round((tdif.sum()/2)) >= 900:
                    freq = '15T'
                    
            # Add catch for hourly data -> interrupt check if hourly or semi-hourly
            # Minute data rules ###############################################
            if freq != '15T': # If tip times are available:
                timestep = '1m'
                try:
                    intertip = int(stats2.mode(tdif)[0])
                except:
                    intertip = round((tdif.sum()/len(event)))
                
                if  intertip < 2: # If most inter-tip times are smaller than 2 seconds, reject event
                    fTips = 'True'
                    removed = 'True' 
                    
                    Tots_m = np.nan
                    Tots_15 = np.nan
                    # remove data
                    data.loc[(hour - pd.DateOffset(hours=1)).strftime("%Y-%m-%d %H"):(hour + pd.DateOffset(hours=1)).strftime("%Y-%m-%d %H"),'accum'] = np.nan
                else:
                    event_min = event.resample('1min').sum()
                    event_15 = event.resample('15min').sum()
                    
                    # Count large minutes and large 15-min values
                    Tots_m = len(event_min[event_min > thresholds1[month] ])
                    Tots_15= len(event_15[event_15 > thresholds15[month] ])
                    
                    if (Tots_m!=0)|(Tots_15!=0): # Winter, more conservative rule, adopted for all months
                    #if (sm!=0)|((sm!=0)&(s15!=0)): # alternative summer rule
                        removed = 'True'
                        data.loc[(hour - pd.DateOffset(hours=1)).strftime("%Y-%m-%d %H"):(hour + pd.DateOffset(hours=1)).strftime("%Y-%m-%d %H"),'accum'] = np.nan
            
            # 15 minute total rules ###########################################
            # Winter
            elif month in [1,2,3,4,11,12]: # If data is 15-minute totals
                    timestep = '15m'
                    event_15 = event.resample('15min').sum()
                    Tots_15 = len(event_15[event_15 > thresholds15[month] ])
                    Tots_m  = np.nan
                     
                    if (Tots_15!= 0):
                        removed = 'True'
                        data.loc[(hour - pd.DateOffset(hours=1)).strftime("%Y-%m-%d %H"):(hour + pd.DateOffset(hours=1)).strftime("%Y-%m-%d %H"),'accum'] = np.nan
            # Summer
            else: # If data is 15-minute totals
                    timestep = '15m'
                    event_15 = event.resample('15min').sum()
                    Tots_15= len(event_15[event_15 > thresholds15[month]])
                    Tots_m = np.nan
                    
                    # Average event intensity for wet 15-minute intervals
                    avg_15 = sum(event_15[event_15>1])/len(event_15[event_15>1])
                    if ((Tots_15==1)&(avg_15>thresholds15[month])):
                        removed = 'True'
                        data.loc[(hour - pd.DateOffset(hours=1)).strftime("%Y-%m-%d %H"):(hour + pd.DateOffset(hours=1)).strftime("%Y-%m-%d %H"),'accum'] = np.nan
                    elif (Tots_15>1):
                        removed = 'True'
                        data.loc[(hour - pd.DateOffset(hours=1)).strftime("%Y-%m-%d %H"):(hour + pd.DateOffset(hours=1)).strftime("%Y-%m-%d %H"),'accum'] = np.nan

            # Append data to output dataframe #################################
            output = output.append({"Station_ID": station_id,
                               "Station_Name": station_name,
                               "Latitude": its.latitude,
                               "Longitude": its.longitude,
                               "datetime": hour,
                               "magnitude": mag,
                               "timestep": timestep,
                               "QC_status": event_q,
                               "removed": removed,
                               "Fast-tips": fTips,
                               "Large 15s": Tots_15,
                               "Large minutes": Tots_m}, 
                              ignore_index = True, sort = True)
            
            # Order columns
            output = output[["Station_ID","Station_Name","Latitude",
                                         "Longitude","datetime","magnitude",
                                         "timestep","QC_status","removed",
                                         "Fast-tips","Large 15s","Large minutes"]]
            
        # Output, still inside if suspect > 0
        data.to_csv(outdir+'/'+station_id+'.txt', index = False)
        
        return output
    
#%%

# Uses SubH files and reads intense metadata inside function
directory = r"C:\QCdRaindata\Ultimate_QC\FM_HQCd_SHData\FL13UK" 
files = glob.glob(os.path.join(directory,"*.txt"))

metadir = "C:/QCdRaindata/Ultimate_QC/QCd_Data/FL13UK/" # important to have slash at the end


# Thresholds are saved as: months = ['','JAN', 'FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
# Iteration 1 (winter1) Thresholds:
#               -  J   F  M  A  M  J  J  A  S  O  N  D 
thresholds60 = [30,30,30,30,30,30,40,40,40,40,40,40,30]
thresholds15 = [10,10,10,10,10,10,20,20,20,20,20,20,10]
thresholds1  = [2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 5, 2] 

# Iteration 2 (winter2) Thresholds:
#               -  J   F  M  A  M  J  J  A  S  O  N  D 
#thresholds60_2 = [30,30,30,30,30,30,40,40,40,40,40,40,30]
#thresholds15_2 = [15,15,15,15,15,15,20,20,20,20,20,20,15]
#thresholds1_2  = [4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 4] 

# FINAL, iteration 3 thresholds:
#               -  J   F  M  A  M  J  J  A  S  O  N  D 
#thresholds60 = [30,30,30,30,30,40,40,40,40,40,30,30,30]
#thresholds15 = [13,15,13,13,13,18,20,20,20,20,17,16,15]
#thresholds1  = [2, 3, 2, 2, 2, 4, 5, 5, 5, 5, 4, 3, 3] 
# The first value is skipped because there is no month 0 in pandas indexes
# The lowest threshold at each duration has been replicated in the first (0) position

outdir = "C:/QCdRaindata/Ultimate_QC/FM_SHQCd_SHData/SHQC_iter1"
logdir = "C:/QCdRaindata/Ultimate_QC/Summary/removedhours_FM_FL13UK_iter1.csv"

raw_ea_15s = Parallel(n_jobs = 4, verbose = 1)(delayed(subH_checkr)(file, metadir, thresholds60, thresholds15, thresholds1, outdir) for file in files)
df1 = pd.concat(raw_ea_15s)
del(raw_ea_15s)

df1.to_csv(logdir, index = False)
del(df1)
