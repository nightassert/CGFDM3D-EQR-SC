# ================================================================
#   ESS, Southern University of Science and Technology
#
#   File Name: plotMultiSeismos.py
#   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
#   Created Time: 2022-11-03
#   Discription: Plot the seismograms of the stations
#
#	Update: Tianhong Xu, 12231218@mail.sustech.edu.cn
#   Update Time: 2023-11-16
#   Update Content: Output the seismograms of the stations to the data directory
#
#   Reference:
#      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
#      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
#
# ================================================================

import json
import numpy as np
from pyscripts.GRID import GRID
import matplotlib.pyplot as plt
import sys
import json

import os

StationLabels = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
                 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                 'O', 'P', 'Q',		 'R', 'S', 'T',
                 'U', 'V', 'W', 	 'X', 'Y', 'Z']

# var = 'Vz'  # Vy Vz
# Uvar = 'Uz'

if len(sys.argv) > 1:
    var = str(sys.argv[1])
    Uvar = str(sys.argv[2])
    scheme = str(sys.argv[3])

data_dir = "./data/" + scheme + "/"

# Keys = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
station_json = open("station.json")
station = json.load(station_json)
Keys = station["station(point)"].keys()

if len(Keys) == 1:
    ampStr = "amplify"
else:
    ampStr = ''


StationTicks = np.arange(0, len(Keys))
lineWidth = 1
IsDiffer = 1


colors = ['k', 'r', 'g', 'b', 'y', 'm']
# lines  = [ '-', '--', '-.', ':', '*', ',']

jsonsFile = open("params.json")
params = json.load(jsonsFile)
grid = GRID(params)


DT = params["DT"]

TMAX = params["TMAX"]

NT = int(TMAX / DT) - 1


t = np.linspace(0, TMAX, NT)
stationFile = open("station.json")
stationDir = json.load(stationFile)

station = stationDir["station(point)"]


zoom = 10

tmp_dir = "./output/" + scheme
NameMode = []
NameMode.append(tmp_dir)  # params["out"]
# NameMode.append(params["out"])  # params["out"]
# NameMode.append( "CGFDM3D-LSRK" )#params["out"]
# NameMode.append( "CGFDM3D-CJM" )#params["out"]
# NameMode.append( "CGFDM3D-CJMVS" )#params["out"]
# NameMode.append( "CGFDM3D-LSRK-CJM" )#params["out"]
# NameMode.append( "CGFDM3D-LSRK-CJMVS" )#params["out"]


NVersion = len(NameMode)

dataPrefix = ""
GRTM3DDir = dataPrefix + "GRTM3D/"


prefix = ""
fileName = []

for i in range(NVersion):
    fileName.append(dataPrefix + NameMode[i] + "/station")


# NameMode = []
# NameMode.append(data_dir)  # params["out"]
# NameMode.append( "LSRK" )#params["out"]
# NameMode.append( "CJM" )#params["out"]
# NameMode.append( "CJMVS" )#params["out"]
# NameMode.append( "LSRK-CJM" )#params["out"]
# NameMode.append( "LSRK-CJMVS" )#params["out"]

# fileName[0] = 'output/station'


WSIZE = 9
CSIZE = 3

mpiX = -1
mpiY = -1
mpiZ = -1

PX = grid.PX
PY = grid.PY
PZ = grid.PZ

stationNum = np.zeros([PZ, PY, PX], dtype='int32')

stationData = {}

seq = 0


varDir = {"Vx": 0, "Vy": 1, "Vz": 2, "Txx": 3,
          "Tyy": 4, "Tzz": 5, "Txy": 6, "Txz": 7, "Tyz": 8}

varId = varDir[var]


for index in station.values():
    X = index[0]
    Y = index[1]
    Z = index[2]

    for mpiZ in range(grid.PZ):
        for mpiY in range(grid.PY):
            for mpiX in range(grid.PX):
                thisX = X - grid.frontNX[mpiX] + grid.halo
                thisY = Y - grid.frontNY[mpiY] + grid.halo
                thisZ = Z - grid.frontNZ[mpiZ] + grid.halo
                if thisX >= grid.halo and thisX < grid._nx[mpiX] and thisY >= grid.halo and thisY < grid._ny[mpiY] and thisZ >= grid.halo and thisZ < grid._nz[mpiZ]:
                    stationNum[mpiZ, mpiY, mpiX] += 1

stationForDrawNum = len(Keys)
U = np.zeros([NVersion, stationForDrawNum, NT])
Ux = np.zeros([NVersion, stationForDrawNum, NT])
Uy = np.zeros([NVersion, stationForDrawNum, NT])
Uz = np.zeros([NVersion, stationForDrawNum, NT])

stationKeyNum = {}


for version in range(NVersion):
    num = 0
    for mpiZ in range(grid.PZ):
        for mpiY in range(grid.PY):
            for mpiX in range(grid.PX):
                if stationNum[mpiZ, mpiY, mpiX] != 0:
                    FileName = "%s_mpi_%d_%d_%d.bin" % (
                        fileName[version], mpiX, mpiY, mpiZ)
                    print(FileName)
                    File = open(FileName, "rb")
                    print(FileName)
                    count = stationNum[mpiZ, mpiY, mpiX] * CSIZE
                    XYZ = np.fromfile(File, dtype='int32', count=count)
                    XYZIndex = np.reshape(
                        XYZ, (stationNum[mpiZ, mpiY, mpiX], CSIZE))
                    count = NT * stationNum[mpiZ, mpiY, mpiX] * WSIZE
                    data = np.fromfile(File, dtype='float32', count=count)
                    dataRe = np.reshape(
                        data, (stationNum[mpiZ, mpiY, mpiX], NT, WSIZE))
                    stationData[(mpiX, mpiY, mpiZ)] = dataRe
                    for key in Keys:
                        xidx = station[key][0]
                        yidx = station[key][1]
                        zidx = station[key][2]
                        # print("key = %s, X = %d, Y = %d, Z = %d" %
                        #       (key, xidx, yidx, zidx))
                        for i in range(stationNum[mpiZ, mpiY, mpiX]):
                            Ux_ = np.zeros(NT)
                            Uy_ = np.zeros(NT)
                            Uz_ = np.zeros(NT)

                            UxSum = 0.0
                            UySum = 0.0
                            UzSum = 0.0
                            '''
							for it in range( 1, NT ):
								UxSum += ( dataRe[0, i, it - 1] + dataRe[0, i, it]) * DT * 0.5
								UySum += ( dataRe[1, i, it - 1] + dataRe[1, i, it]) * DT * 0.5
								UzSum += ( dataRe[2, i, it - 1] + dataRe[2, i, it]) * DT * 0.5
								Ux_[it] = UxSum
								Uy_[it] = UySum
								Uz_[it] = UzSum
							'''
                            Ux_[:] = dataRe[i, :, 0]
                            Uy_[:] = dataRe[i, :, 1]
                            Uz_[:] = dataRe[i, :, 2]

                            if xidx == XYZIndex[i, 0] and yidx == XYZIndex[i, 1] and zidx == XYZIndex[i, 2]:
                                # print(np.shape(Ux))
                                # print(np.shape(Ux_))
                                print("key = %s, X = %d, Y = %d, Z = %d" %
                                      (key, xidx, yidx, zidx))
                                for it in range(NT):
                                    Ux[version, num, it] = Ux_[it]
                                    Uy[version, num, it] = Uy_[it]
                                    Uz[version, num, it] = Uz_[it]

                                stationKeyNum[key] = num
                                num += 1
                        # print(stationKeyNum)

if Uvar == 'Ux':
    U = Ux
else:
    if Uvar == 'Uy':
        U = Uy  # / np.sqrt( np.pi )# 2. * np.pi * np.sqrt( np.pi )
    else:
        if Uvar == 'Uz':
            U = Uz  # / np.sqrt( np.pi )#2. * np.pi *  np.sqrt( np.pi )
        else:
            U = U + 1

vmax = np.max(np.abs(U))


if len(Keys) > 1:
    vmaxUx = np.max(np.abs(Ux))
    vmaxUy = np.max(np.abs(Uy))
    vmaxUz = np.max(np.abs(Uz))
    vmax = np.max([vmaxUx, vmaxUy, vmaxUz])

print("CGFDM3D Max = %f" % vmax)
print(stationKeyNum)


if len(Keys) > 1:
    plt.figure(figsize=(8, 10))
else:
    plt.figure(figsize=(8, 2))


for version in range(NVersion):
    for key in station.keys():
        # print( key )
        for iKey in Keys:
            if key == iKey:
                print(key)
                i = stationKeyNum[key]
                if i == 0:
                    np.savetxt(data_dir + var+str(key)+".txt", U[version, i])
                    plt.plot(t, U[version, i] / vmax + i, color=colors[version],
                             ls='-', label=NameMode[version], linewidth=lineWidth)
                    if version == 0:
                        plt.text(TMAX * 0.85, i + 0.05, "%.2fcm/s" %
                                 np.max(np.abs(U[version, i] * 100)))
                    # if version != 0:
                    # plt.plot( t, (U[version, i] - U[0, i] ) * zoom / vmax + i, color = colors[version], ls = '--', label = NameMode[version] + ':Error $ \\times $ %d' % zoom , linewidth = lineWidth)

                    if len(Keys) > 1:
                        plt.legend(loc=1)
                else:
                    np.savetxt(data_dir + var+str(key)+".txt", U[version, i])
                    plt.plot(t, U[version, i] / vmax + i,
                             color=colors[version], ls='-', linewidth=lineWidth)
                    if version == 0:
                        plt.text(TMAX * 0.85, i + 0.05, "%.2fcm/s" %
                                 np.max(np.abs(U[version, i] * 100)))
                    # if version != 0:
                    # plt.plot( t, ( U[version, i] - U[0, i] ) * zoom / vmax + i, color = colors[version], ls = '--',linewidth = lineWidth )
                # axis[i].plot( t[i], Ux[i] )
                break

# plt.savefig("SCFDM_%s_%s.pdf" % (var, ampStr))
