# ================================================================
#   ESS, Southern University of Science and Technology
#
#   File Name: setStation.py
#   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
#           Tianhong Xu, 12231218@mail.sustech.edu.cn
#   Created Time: 2025-1-9
#   Discription: Set staions by lon lat file
#
#   Reference:
#      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
#      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
#
# =================================================================*/

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import struct
import sys
import os
from pyproj import Proj
import json
from mpl_toolkits.mplot3d import Axes3D
from pyscripts.GRID import GRID


stationDic = {}

file = open('stationLonLat_picked.txt', 'r')
LonLatTxt = file.readlines()
file.close()

stationNum = len(LonLatTxt)
stationName = []
Lon = np.zeros(stationNum)
Lat = np.zeros(stationNum)
for i in range(stationNum):
    staStr = LonLatTxt[i].split()
    stationName.append(staStr[0])
    Lon[i] = float(staStr[1])
    Lat[i] = float(staStr[2])

jsonsFile = open("params.json")
params = json.load(jsonsFile)
grid = GRID(params)

DH = params["DH"]
centerX = params["centerX"]
centerY = params["centerY"]

nPML = params['nPML']
XRange1 = nPML + 1
XRange2 = params['NX'] - nPML

YRange1 = nPML + 1
YRange2 = params['NY'] - nPML


NX = params['NX']
NY = params['NY']
NZ = params['NZ']


latC = params["centerLatitude"]
lonC = params["centerLongitude"]
proj = Proj(proj='aeqd', lat_0=latC, lon_0=lonC, ellps="WGS84")
NZ = params['NZ']

XCoord = np.zeros(stationNum)
YCoord = np.zeros(stationNum)
XCoord, YCoord = proj(Lon, Lat)

X = np.zeros(stationNum)
Y = np.zeros(stationNum)
n = 0


stationDicTmp = {}
for i in range(stationNum):
    X[i] = round(XCoord[i] / DH) + centerX
    Y[i] = round(YCoord[i] / DH) + centerY
    if X[i] >= XRange1 and X[i] < XRange2 and Y[i] >= YRange1 and Y[i] < YRange2:
        I = int(X[i])
        J = int(Y[i])
        K = int(NZ - 1)
        index = I + J * NX + K * NX * NY
        stationDicTmp[index] = stationName[i]


stationDic = {}


for key in stationDicTmp.keys():
    K = key // (NX * NY)
    J = key % (NX * NY) // NX
    I = key % NX
    stationDic[stationDicTmp[key]] = [I, J, K]


stationParam = {

    "point": 1,
    "lon_lat": 0,

    "station(point)": stationDic
}

print(stationParam)

stationJsonFile = open("station.json", 'w')

json_str = json.dumps(stationParam, indent=4)
stationJsonFile.write(json_str)
stationJsonFile.close()


# json_str = json.dumps( stationParam )
# print( json_str )
