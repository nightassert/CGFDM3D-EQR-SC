# ================================================================
#   ESS, Southern University of Science and Technology
#
#   File Name: plotWaveform.py
#   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
#           Tianhong Xu, 12231218@mail.sustech.edu.cn
#   Created Time: 2025-1-9
#   Discription: Plot waveforms recorded by the stations
#
#   Reference:
#      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
#      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
#
# =================================================================*/

from matplotlib import rcParams
import scipy.integrate as inte
import json
import numpy as np
from pyscripts.GRID import GRID
import matplotlib.pyplot as plt
import sys
import json

import os

data_dir = "./data/"

station_json = open("station.json")
station = json.load(station_json)
Keys = station["station(point)"].keys()

StationTicks = np.arange(0, len(Keys))
lineWidth = 1
IsDiffer = 1

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

outdir = params["out"]

prefix = ""
fileName = []

fileName = outdir + "/station"


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
Ux = np.zeros([1, stationForDrawNum, NT])
Uy = np.zeros([1, stationForDrawNum, NT])
Uz = np.zeros([1, stationForDrawNum, NT])

stationKeyNum = {}

num = 0
for mpiZ in range(grid.PZ):
    for mpiY in range(grid.PY):
        for mpiX in range(grid.PX):
            if stationNum[mpiZ, mpiY, mpiX] != 0:
                FileName = "%s_mpi_%d_%d_%d.bin" % (
                    fileName, mpiX, mpiY, mpiZ)
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

                    for i in range(stationNum[mpiZ, mpiY, mpiX]):
                        Ux_ = np.zeros(NT)
                        Uy_ = np.zeros(NT)
                        Uz_ = np.zeros(NT)

                        UxSum = 0.0
                        UySum = 0.0
                        UzSum = 0.0

                        Ux_[:] = dataRe[i, :, 0]
                        Uy_[:] = dataRe[i, :, 1]
                        Uz_[:] = dataRe[i, :, 2]

                        if xidx == XYZIndex[i, 0] and yidx == XYZIndex[i, 1] and zidx == XYZIndex[i, 2]:

                            print("key = %s, X = %d, Y = %d, Z = %d" %
                                  (key, xidx, yidx, zidx))
                            for it in range(NT):
                                Ux[0, num, it] = Ux_[it]
                                Uy[0, num, it] = Uy_[it]
                                Uz[0, num, it] = Uz_[it]

                            stationKeyNum[key] = num
                            num += 1

for key in station.keys():
    for iKey in Keys:
        if key == iKey:
            print(key)
            i = stationKeyNum[key]
            np.savetxt(data_dir + "Vx_" + str(key) + ".txt", Ux[0, i])
            np.savetxt(data_dir + "Vy_" + str(key) + ".txt", Uy[0, i])
            np.savetxt(data_dir + "Vz_" + str(key) + ".txt", Uz[0, i])
            break


import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as inte
from matplotlib import rcParams
import json
from scipy import signal

params = json.load(open("params.json"))

dt = params["DT"]
t = np.arange(0, params["TMAX"] - 2 * dt, dt)
print(t.shape)

dt_obs = 0.01

config = {"font.size": 12}
rcParams.update(config)
plt.figure(figsize=(18, 12))
plt.subplots_adjust(wspace=0.08)

station = json.load(open("station.json"))
station_keys = station["station(point)"].keys()
station_num = len(station["station(point)"])

print("station_num = %d" % station_num)

# Bandpass filter: OBS
fl = 0.01
fh = 2.0
wn1 = fl / (1/dt_obs/2)
wn2 = fh / (1/dt_obs/2)
print(wn1, wn2)
b1, a1 = signal.butter(2, [wn1, wn2], 'bandpass')

wn1 = fl / (1/dt/2)
wn2 = fh / (1/dt/2)
print(wn1, wn2)
b2, a2 = signal.butter(2, [wn1, wn2], 'bandpass')

idiswid = 0
for key in station_keys:

    receiver_v_x = np.loadtxt(
        "./data/Vx_" + str(key) + ".txt")
    receiver_v_y = np.loadtxt(
        "./data/Vy_" + str(key) + ".txt")
    receiver_v_z = np.loadtxt(
        "./data/Vz_" + str(key) + ".txt")

    # filter
    receiver_v_x = signal.filtfilt(b2, a2, receiver_v_x)
    receiver_v_y = signal.filtfilt(b2, a2, receiver_v_y)
    receiver_v_z = signal.filtfilt(b2, a2, receiver_v_z)

    # # Displacement
    # receiver_v_x = inte.cumtrapz(receiver_v_x, t, initial=0)
    # receiver_v_y = inte.cumtrapz(receiver_v_y, t, initial=0)
    # receiver_v_z = inte.cumtrapz(receiver_v_z, t, initial=0)

    # Normalize
    # max_value = max(max(abs(receiver_v_x)), max(
    #     abs(receiver_v_y)), max(abs(receiver_v_z)))
    max_value = 1

    disWid = 2

    plt.subplot(1, 3, 1)
    v_x = plt.plot(t, receiver_v_x /
                         max_value + idiswid*disWid, "b-")
    plt.xlim([0, params["TMAX"]])
    plt.ylim(-1.5, -1.5 + 2 * station_num + 1)
    plt.yticks(np.arange(0, 2 * station_num, 2), np.arange(
        0, station_num, 1), fontsize=12)
    plt.yticks(np.arange(0, 2 * station_num, 2), station_keys, fontsize=12)
    plt.xlabel('time(s)')
    plt.title('v_x')
    plt.ylabel("Amplitude (m/s)")

    plt.subplot(1, 3, 2)
    v_y = plt.plot(t, receiver_v_y /
                         max_value + idiswid*disWid, "b-")
    plt.ylim(-1.5, -1.5 + 2 * station_num + 1)
    plt.xlim([0, params["TMAX"]])
    plt.yticks([])
    plt.xlabel('time(s)')
    plt.title('v_y')

    plt.subplot(1, 3, 3)
    v_z = plt.plot(t, receiver_v_z /
                         max_value + idiswid*disWid, "b-")
    plt.ylim(-1.5, -1.5 + 2 * station_num + 1)
    plt.xlim([0, params["TMAX"]])
    plt.yticks([])
    plt.xlabel('time(s)')
    plt.title('v_z')

    idiswid += 1

plt.savefig("./img/Waveforms.png", dpi=300)