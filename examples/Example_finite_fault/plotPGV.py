# ================================================================
#   ESS, Southern University of Science and Technology
#
#   File Name: plotPGV.py
#   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
#           Tianhong Xu, 12231218@mail.sustech.edu.cn
#   Created Time: 2025-1-9
#   Discription: Plot PGVh/PGV/PGAh/PGA and corresponding Intensity
#
#   Reference:
#      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
#      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
# 
# =================================================================*/

import json
import numpy as np
from pyscripts.GRID import GRID
import matplotlib.pyplot as plt
import sys
from scipy.io import savemat

PGVh = 0
PGV = 1
PGAh = 2
PGA = 3

var = PGVh

if len(sys.argv) > 1:
    var = str(sys.argv[1])

if var == "PGVh":
    switch = PGVh
if var == "PGV":
    switch = PGV
if var == "PGAh":
    switch = PGAh
if var == "PGA":
    switch = PGA

jsonsFile = open("params.json")
params = json.load(jsonsFile)
grid = GRID(params)
FAST_AXIS = params["FAST_AXIS"]

sample = 1

outputPath = params["out"]
fileNameX = params["out"] + "/lon"
fileNameY = params["out"] + "/lat"


fileName = outputPath + "/PGV"

PGVSIZE = 4

sliceX = params["sliceX"] - grid.frontNX
sliceY = params["sliceY"] - grid.frontNY
sliceZ = params["sliceZ"] - grid.frontNZ

for mpiSliceX in range(grid.PX):
    if sliceX[mpiSliceX] >= 0 and sliceX[mpiSliceX] < grid.nx[mpiSliceX]:
        break

for mpiSliceY in range(grid.PY):
    if sliceY[mpiSliceY] >= 0 and sliceY[mpiSliceY] < grid.ny[mpiSliceY]:
        break

for mpiSliceZ in range(grid.PZ):
    if sliceZ[mpiSliceZ] >= 0 and sliceZ[mpiSliceZ] < grid.nz[mpiSliceZ]:
        break


if FAST_AXIS == 'Z':
    dataX = np.zeros([grid.NX, grid.NY])
    dataY = np.zeros([grid.NX, grid.NY])
    data = np.zeros([grid.NX, grid.NY])
    Intensity = np.zeros([grid.NX, grid.NY])
else:
    dataX = np.zeros([grid.NY, grid.NX])
    dataY = np.zeros([grid.NY, grid.NX])
    data = np.zeros([grid.NY, grid.NX])
    Intensity = np.zeros([grid.NY, grid.NX])


for mpiY in range(grid.PY):
    for mpiX in range(grid.PX):
        mpiZ = grid.PZ - 1
        XFile = open("%s_mpi_%d_%d_%d.bin" %
                     (fileNameX, mpiX, mpiY, mpiZ), "rb")
        YFile = open("%s_mpi_%d_%d_%d.bin" %
                     (fileNameY, mpiX, mpiY, mpiZ), "rb")
        File = open("%s_Z_mpi_%d_%d_%d.bin" %
                    (fileName, mpiX, mpiY, mpiZ), "rb")

        ny = grid.ny[mpiY]
        nx = grid.nx[mpiX]

        print("ny = %d, nx = %d" % (nx, ny))
        datax = np.fromfile(XFile, dtype='float32', count=ny * nx)
        datay = np.fromfile(YFile, dtype='float32', count=ny * nx)
        data_ = np.fromfile(File, dtype='float32', count=ny * nx * PGVSIZE)
        dataTmp = np.reshape(data_, (ny * nx, PGVSIZE))
        J = grid.frontNY[mpiY]
        J_ = grid.frontNY[mpiY] + ny
        I = grid.frontNX[mpiX]
        I_ = grid.frontNX[mpiX] + nx

        if FAST_AXIS == 'Z':
            dataX[I:I_, J:J_] = np.reshape(datax, (nx, ny))
            dataY[I:I_, J:J_] = np.reshape(datay, (nx, ny))
            data[I:I_, J:J_] = np.reshape(dataTmp[:, switch], (nx, ny))
        else:
            dataX[J:J_, I:I_] = np.reshape(datax, (ny, nx))
            dataY[J:J_, I:I_] = np.reshape(datay, (ny, nx))
            data[J:J_, I:I_] = np.reshape(dataTmp[:, switch], (ny, nx))

switch_data = data

# print(switch_data)

if switch == PGVh or switch == PGV:
    for j in range(grid.NY):
        for i in range(grid.NX):
            if switch_data[j, i] >= 1.76:
                Intensity[j, i] = 11
            if switch_data[j, i] >= 0.815 and switch_data[j, i] < 1.76:
                Intensity[j, i] = 10
            if switch_data[j, i] >= 0.379 and switch_data[j, i] < 0.815:
                Intensity[j, i] = 9
            if switch_data[j, i] >= 0.177 and switch_data[j, i] < 0.379:
                Intensity[j, i] = 8
            if switch_data[j, i] >= 0.0818 and switch_data[j, i] < 0.177:
                Intensity[j, i] = 7
            if switch_data[j, i] >= 0.0381 and switch_data[j, i] < 0.0818:
                Intensity[j, i] = 6
            if switch_data[j, i] >= 0.0177 and switch_data[j, i] < 0.0381:
                Intensity[j, i] = 5
            if switch_data[j, i] >= 0.00820 and switch_data[j, i] < 0.0177:
                Intensity[j, i] = 4
            if switch_data[j, i] >= 0.00382 and switch_data[j, i] < 0.00820:
                Intensity[j, i] = 3
            if switch_data[j, i] >= 0.00178 and switch_data[j, i] < 0.00382:
                Intensity[j, i] = 2
            if switch_data[j, i] < 0.00178:
                Intensity[j, i] = 1

if switch == PGAh or switch == PGA:
    for j in range(grid.NY):
        for i in range(grid.NX):
            if switch_data[j, i] >= 17.3:
                Intensity[j, i] = 11
            if switch_data[j, i] >= 8.31 and switch_data[j, i] < 17.3:
                Intensity[j, i] = 10
            if switch_data[j, i] >= 4.02 and switch_data[j, i] < 8.31:
                Intensity[j, i] = 9
            if switch_data[j, i] >= 1.95 and switch_data[j, i] < 4.02:
                Intensity[j, i] = 8
            if switch_data[j, i] >= 0.937 and switch_data[j, i] < 1.95:
                Intensity[j, i] = 7
            if switch_data[j, i] >= 0.457 and switch_data[j, i] < 0.937:
                Intensity[j, i] = 6
            if switch_data[j, i] >= 0.223 and switch_data[j, i] < 0.457:
                Intensity[j, i] = 5
            if switch_data[j, i] >= 0.109 and switch_data[j, i] < 0.223:
                Intensity[j, i] = 4
            if switch_data[j, i] >= 0.0529 and switch_data[j, i] < 0.109:
                Intensity[j, i] = 3
            if switch_data[j, i] >= 0.0258 and switch_data[j, i] < 0.0529:
                Intensity[j, i] = 2
            if switch_data[j, i] < 0.0258:
                Intensity[j, i] = 1

# nPML = params["nPML"]
nPML = 12

NX = grid.NX
NY = grid.NY
switch_data = switch_data[nPML:NY - nPML, nPML:NX-nPML]

Intensity = Intensity[nPML:NY - nPML, nPML:NX-nPML]
lon = dataX[nPML:NY - nPML, nPML:NX-nPML]
lat = dataY[nPML:NY - nPML, nPML:NX-nPML]

print(np.max(Intensity))

if switch == PGVh:
    name = "PGVh"
if switch == PGV:
    name = "PGV"
if switch == PGAh:
    name = "PGAh"
if switch == PGA:
    name = "PGA"

plt.figure()
dpi = 300
plt.pcolormesh(lon, lat, switch_data, cmap="seismic")
# plt.clim([5, 10])
plt.colorbar()
plt.axis("image")
plt.title(var)
plt.show()
plt.savefig("./img/" + name + ".png", dpi=dpi)

plt.figure()
dpi = 300
plt.pcolormesh(lon, lat, Intensity, cmap="seismic")
# plt.clim([5, 10])
plt.colorbar()
plt.axis("image")
plt.title("Intensity")
plt.show()
plt.savefig("./img/Intensity_" + name + ".png", dpi=dpi)