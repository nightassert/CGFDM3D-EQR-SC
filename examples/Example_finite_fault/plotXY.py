# ================================================================
#   ESS, Southern University of Science and Technology
#
#   File Name: plotXY.py
#   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
#           Tianhong Xu, 12231218@mail.sustech.edu.cn
#   Created Time: 2025-1-9
#   Discription: Plot Wavefield on X-Y plane
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

if len(sys.argv) > 1:
    it = int(sys.argv[1])
    var = str(sys.argv[2])
    FreeSurf = int(sys.argv[3])

jsonsFile = open("params.json")
params = json.load(jsonsFile)
grid = GRID(params)
FAST_AXIS = params["FAST_AXIS"]

sample = 1

outputPath = params["out"]
fileNameX = params["out"] + "/lon"
fileNameY = params["out"] + "/lat"


if FreeSurf == 1:
    fileName = outputPath + "/FreeSurf%s_%d" % (var, it)
    varname = "./img/FreeSurf%s_%d" % (var, it)
else:
    fileName = outputPath + "/%s_%d" % (var, it)
    varname = "./img/SliceZ_%s_%d" % (var, it)

if var == "Vs" or var == "Vp" or var == "rho":
    fileName = outputPath + "/%s" % (var)

print("Draw " + fileName)

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
else:
    dataX = np.zeros([grid.NY, grid.NX])
    dataY = np.zeros([grid.NY, grid.NX])
    data = np.zeros([grid.NY, grid.NX])


for mpiY in range(grid.PY):
    for mpiX in range(grid.PX):
        mpiZ = grid.PZ - 1
        XFile = open("%s_mpi_%d_%d_%d.bin" %
                     (fileNameX, mpiX, mpiY, mpiZ), "rb")
        YFile = open("%s_mpi_%d_%d_%d.bin" %
                     (fileNameY, mpiX, mpiY, mpiZ), "rb")
        if FreeSurf:
            mpiZ = grid.PZ - 1
        else:
            mpiZ = mpiSliceZ
        File = open("%s_Z_mpi_%d_%d_%d.bin" %
                    (fileName, mpiX, mpiY, mpiZ), "rb")

        ny = grid.ny[mpiY]
        nx = grid.nx[mpiX]

        print("ny = %d, nx = %d" % (nx, ny))
        datax = np.fromfile(XFile, dtype='float32', count=ny * nx)
        datay = np.fromfile(YFile, dtype='float32', count=ny * nx)
        data_ = np.fromfile(File, dtype='float32', count=ny * nx)

        J = grid.frontNY[mpiY]
        J_ = grid.frontNY[mpiY] + ny
        I = grid.frontNX[mpiX]
        I_ = grid.frontNX[mpiX] + nx

        if FAST_AXIS == 'Z':
            dataX[I:I_, J:J_] = np.reshape(datax, (nx, ny))
            dataY[I:I_, J:J_] = np.reshape(datay, (nx, ny))
            data[I:I_, J:J_] = np.reshape(data_, (nx, ny))
        else:
            dataX[J:J_, I:I_] = np.reshape(datax, (ny, nx))
            dataY[J:J_, I:I_] = np.reshape(datay, (ny, nx))
            data[J:J_, I:I_] = np.reshape(data_, (ny, nx))


dpi = 300
unit = 1  # 1km = 1000m

fontsize = 10
plt.rcParams['font.size'] = fontsize

vm = np.max(np.abs(data))
dataX = dataX/unit
dataY = dataY/unit
if var == "Vs" or var == "Vp" or var == "rho":
    plt.pcolormesh(dataX, dataY, data, cmap="seismic")
else:
    plt.pcolormesh(dataX, dataY, data, vmax=vm, vmin=-vm, cmap="seismic")
plt.colorbar()
plt.axis("equal")

plt.title(var, fontsize=12)
filename = varname + ".png"
plt.savefig(filename, dpi=300)

# data output
data_dir = "./data/"
np.savetxt(data_dir + "dataX.txt", dataX)
np.savetxt(data_dir + "dataY.txt", dataY)
np.savetxt(data_dir + var + "_" + str(it) + "_" + "data.txt", data)
