import datetime
import time
import requests
import wget
import os
import pandas as pd
import csv
import sys
#
# function to return SHIP format in CSV format
#
def ship2csv(file,leadtime=0):
    response = requests.get(file)
    b = file.split('/')[-1]
    data = [b[:-19]]
    label = ['Storm']
    if response.status_code == 200:
       print('TC SHIP file: ',file,' exists')
       try:
          TC_ship = wget.download(file)
          df = pd.read_csv(file, skiprows=6, nrows=16,delim_whitespace=True)
          df.columns=['var','unit','0h','6h','12h','18h','24h','30h','36h','42h','48h','54h','60h','66h','72h','78h','84h','90h','96h','102h','108h','114h','120h','126h']
          a = df['var']
          b0 = df['0h']
          for i,j in zip(a,b0):
              label.append(i+"+00h")
              data.append(j)
          b6 = df['6h']
          for i,j in zip(a,b6):
              label.append(i+"+06h")
              data.append(j)
          b12 = df['12h']
          for i,j in zip(a,b12):
              label.append(i+"+12h")
              data.append(j)
          b18 = df['18h']
          for i,j in zip(a,b18):
              label.append(i+"+18h")
              data.append(j)
          b24 = df['24h']
          for i,j in zip(a,b24):
              label.append(i+"+24h")
              data.append(j)
          label.append('class')
          data.append(0)
       except Exception:
          pass
    else:
       print('TC SHIP file: ',file,' does NOT exist')
    return data,label
#
# function to return SHIP format in CSV format using old HWRF SHIP file name format
#
def ship2csv_old(file,leadtime=0):
    response = requests.get(file)
    b = file.split('/')[-1]
    data = [b[:-4]]
    label = ['Storm']
    if response.status_code == 200:
       print('TC SHIP file: ',file,' exists')
       try:
          TC_ship = wget.download(file)
          df = pd.read_csv(file, skiprows=6, nrows=16,delim_whitespace=True)
          df.columns=['var','unit','0h','6h','12h','18h','24h','30h','36h','42h','48h','54h','60h','66h','72h','78h','84h','90h','96h','102h','108h','114h','120h','126h']
          a = df['var']
          b0 = df['0h']
          for i,j in zip(a,b0):
              label.append(i+"+00h")
              data.append(j)
          b6 = df['6h']
          for i,j in zip(a,b6):
              label.append(i+"+06h")
              data.append(j)
          b12 = df['12h']
          for i,j in zip(a,b12):
              label.append(i+"+12h")
              data.append(j)
          b18 = df['18h']
          for i,j in zip(a,b18):
              label.append(i+"+18h")
              data.append(j)
          b24 = df['24h']
          for i,j in zip(a,b24):
              label.append(i+"+24h")
              data.append(j)
          label.append('class')
          data.append(0)
       except Exception:
          pass
    else:
       print('TC SHIP file: ',file,' does NOT exist')
    return data,label

#
# function to return basin acronym for NCEP/HAF system
#
def basin_acronym(basinid):
    if basinid == "L":
        basin = "RT" + yyyy + "_NATL/"
    elif basinid == "E":
        basin = "RT" + yyyy + "_EPAC/"
    elif basinid == "W":
        basin = "RT" + yyyy + "_WPAC/"
    else:
        print("Basin is not support. Stop")
        exit()
    return basin
#
# function to turn real time into a cycle
#
def add_hour(yyyymmddhh,interval=3):
    yyyy = yyyymmddhh[:4]
    mm=yyyymmddhh[4:6]
    dd=yyyymmddhh[6:8]
    hh=yyyymmddhh[-2:]
    a = datetime.datetime(int(yyyy), int(mm), int(dd), int(hh),0,0,0)
    ts = datetime.datetime.timestamp(a)
    b = datetime.datetime.fromtimestamp(ts+interval*3600)
    if b.day < 10:
        new_day = "0" + str(b.day)
    else:
        new_day = str(b.day)
    if b.month < 10:
        new_month = "0" + str(b.month)
    else:
        new_month = str(b.month)
    if b.hour < 10:
        new_hour = "0" + str(b.hour)
    else:
        new_hour = str(b.hour)
    yyyymmddhh_updated = str(b.year) + new_month + new_day + new_hour
    return yyyymmddhh_updated
#
# function to turn real time into a cycle
#
def get_real_cycle(stepback=False):
    get_now = datetime.datetime.now()
    get_now_stamp = get_now.timestamp()
    get_p6h = datetime.datetime.fromtimestamp(get_now_stamp + 3600*6)
    get_m6h = datetime.datetime.fromtimestamp(get_now_stamp - 3600*6)
    if stepback==True:
        get_now = get_m6h
        print(get_now,get_p6h,get_m6h)
    yyyy = get_now.year
    month =  get_now.month
    day = get_now.day
    hour = get_now.hour
    minute = get_now.minute
    if month < 10:
        mm = "0" + str(month)
    else:
        mm = str(month)
    if day < 10:
        dd = "0" + str(day)
    else:
        dd = str(day)
    if hour < 10:
        hh = "0" + str(hour)
    else:
        hh = str(hour)
    yyyymmdd = str(yyyy) + mm + dd
    yyyymmddhh = str(yyyy) + mm + dd + hh
    if hour < 6:
        cycle = "00"
    elif hour < 12:
        cycle = "06"
    elif hour < 18:
        cycle = "12"
    else:
        cycle = "18"
    return yyyymmdd,cycle
#===============================================================================
#MAIN
#===============================================================================
#
# check the input arguments with 3 options for the flag_mode
# 1. realtime: use for real time download
# 2. cycles: a range of TC cycles
# 3. lifetime: an entire TC lifetime
#
n = len(sys.argv)
print("Total arguments input are:", n)
print("Name of Python script:", sys.argv[0])
if n < 2:
    print("Need at least one input argument...Stop")
    print("  Example 1: python RI_getSHIP.py cycles 5 FRANKLIN08L 2023082018")
    print("  Example 2: python RI_getSHIP.py realtime")
    print("  Example 3: python RI_getSHIP.py lifetime FRANKLIN08L 2023082018")
    exit()
flag_mode = sys.argv[1]
if flag_mode == "cycles" and n == 5:
    print("Runing in a specific range of cycle mode")
elif flag_mode == "realtime":
    print("Running in real time mode")
elif flag_mode == "lifetime" and n == 4:
    print("Running in entire TC lifetime mode")
else:
    print("Wrong input argument...Stop")
    print("  Example 1: python RI_getSHIP.py cycles 5 FRANKLIN08L 2023082018")
    print("  Example 2: python RI_getSHIP.py realtime")
    print("  Example 3: python RI_getSHIP.py lifetime FRANKLIN08L 2023082018")
    exit()
#
# setup a downloand link and work on each mode
#
ncep_ftp = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/"
if flag_mode == "realtime":
    yyyymmdd, cycle = get_real_cycle(True)
    check_cycle = ncep_ftp + "gfs." + yyyymmdd + "/" + cycle + "/atmos/" + "gfs.t" + cycle + "z.syndata.tcvitals.tm00"   
    check_file =  "gfs.t" + cycle + "z.syndata.tcvitals.tm00"
    print("TCvital to check is: ",check_cycle) 
    response = requests.get(check_cycle)
    ncep_link="https://www.emc.ncep.noaa.gov/hurricane/HFSAForecast/"
    if response.status_code == 200:
        print('TCvital file exists')
        try:
            TC_cycle = wget.download(check_cycle)
        except Exception:
            pass
        vital_file = open(check_file, 'r')
        lines = vital_file.readlines()
        for line in lines:
            df_line = line.split()
            print(df_line)
            stormid = df_line[2] + df_line[1]
            basinid = stormid[-1:]
            yyyy = df_line[3][:4]
            basin = basin_acronym(basinid)
            storm_cycle = stormid+"."+yyyymmdd+cycle
            ship_file = ncep_link + basin + stormid + "/" + storm_cycle + "/" + storm_cycle + ".HFSA.ship_diag.txt"
            data,label = ship2csv(ship_file,leadtime=24)
            output_file = storm_cycle + '.ship.txt'
            if len(data) > 0:
                f = open(output_file, 'w')
                writer = csv.writer(f)
                writer.writerow(label)
                writer.writerow(data)
                f.close()
            #print(data)  
            #print(label)
    else:
        print('TCvital file does not exist. Will check back in 1 hour')   
elif flag_mode == "cycles":
    ncycles = int(sys.argv[2])
    stormid = sys.argv[3]
    start_cycle = sys.argv[4]
    ncep_link = "https://www.emc.ncep.noaa.gov/hurricane/HFSAForecast/"
    ncep_link2 = "https://www.emc.ncep.noaa.gov/gc_wmb/vxt/RT_EASTPAC/"
    if len(start_cycle) !=10:
        print("Starting cycle must be of the form YYYYMMDDHH. Check again...") 
        exit()
    else:
        yyyy = start_cycle[:4]
        yyyymmdd = start_cycle[:8]
        yyyymmddhh = start_cycle
    storm_name = stormid[:-3]
    basinid = stormid[-1:]
    basin = basin_acronym(basinid)
    loop_cycle = start_cycle[-2:]
    print("Download data for storm: ",stormid," and starting cycle: ",loop_cycle) 
    for icycle in range(ncycles):
        print("\nDownload cycle ... ",yyyymmddhh)
        storm_cycle = stormid + "." + yyyymmddhh
        ship_file = ncep_link + basin + stormid + "/" + storm_cycle + "/" + storm_cycle + ".HFSA.ship_diag.txt"
        ship_file2 = ncep_link2 + stormid + "/" + storm_cycle + "/" + storm_cycle + ".txt"
        data,label = ship2csv_old(ship_file2,leadtime=24)
        output_file = storm_cycle + '.ship.txt'
        if len(data) > 0:
            f = open(output_file, 'w')
            writer = csv.writer(f)
            writer.writerow(label)
            writer.writerow(data)
            f.close()
        yyyymmddhh=add_hour(yyyymmddhh,6)
else:
    print("flag_mode is not yet support...Stop")
    exit()
print("\nFinish downloading data for flag_mode = ",flag_mode)
