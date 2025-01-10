# ================================================================
#   ESS, Southern University of Science and Technology
#
#   File Name: plotYZ.py
#   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
#           Tianhong Xu, 12231218@mail.sustech.edu.cn
#   Created Time: 2025-1-9
#   Discription: Plot Wavefield on Y-Z plane
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


jsonsFile = open("params.json")
params = json.load(jsonsFile)
grid = GRID(params)
FAST_AXIS = params["FAST_AXIS"]

outputPath = params["out"]
fileNameY = params["out"] + "/coordY"
fileNameZ = params["out"] + "/coordZ"

if var == "Vs" or var == "Vp" or var == "rho":
    fileName = outputPath + "/%s" % (var)
else:
    fileName = outputPath + "/%s_%d" % (var, it)
varname = "./img/SliceX_%s_%d" % (var, it)

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
    dataY = np.zeros([grid.NY, grid.NZ])
    dataZ = np.zeros([grid.NY, grid.NZ])
    data = np.zeros([grid.NY, grid.NZ])
else:
    dataY = np.zeros([grid.NZ, grid.NY])
    dataZ = np.zeros([grid.NZ, grid.NY])
    data = np.zeros([grid.NZ, grid.NY])

mpiX = mpiSliceX
for mpiZ in range(grid.PZ):
    for mpiY in range(grid.PY):
        fileY = open("%s_X_mpi_%d_%d_%d.bin" %
                     (fileNameY, mpiX, mpiY, mpiZ), "rb")
        fileZ = open("%s_X_mpi_%d_%d_%d.bin" %
                     (fileNameZ, mpiX, mpiY, mpiZ), "rb")
        file = open("%s_X_mpi_%d_%d_%d.bin" %
                    (fileName, mpiX, mpiY, mpiZ), "rb")
        ny = grid.ny[mpiY]
        nz = grid.nz[mpiZ]
        print("ny = %d, nz = %d" % (ny, nz))
        datay = np.fromfile(fileY, dtype='float32', count=ny * nz)
        dataz = np.fromfile(fileZ, dtype='float32', count=ny * nz)
        data_ = np.fromfile(file, dtype='float32', count=ny * nz)

        J = grid.frontNY[mpiY]
        J_ = grid.frontNY[mpiY] + ny
        K = grid.frontNZ[mpiZ]
        K_ = grid.frontNZ[mpiZ] + nz

        if FAST_AXIS == 'Z':
            dataY[J:J_, K:K_] = np.reshape(datay, (ny, nz))
            dataZ[J:J_, K:K_] = np.reshape(dataz, (ny, nz))
            data[J:J_, K:K_] = np.reshape(data_, (ny, nz))
        else:
            dataY[K:K_, J:J_] = np.reshape(datay, (nz, ny))
            dataZ[K:K_, J:J_] = np.reshape(dataz, (nz, ny))
            data[K:K_, J:J_] = np.reshape(data_, (nz, ny))


dpi = 300
vm = np.max(data)
dataY /= 1000
dataZ /= 1000
if var == "Vs" or var == "Vp" or var == "rho":
    plt.pcolormesh(dataY, dataZ, data, cmap="seismic")
else:
    plt.pcolormesh(dataY, dataZ, data, vmax=vm, vmin=-vm, cmap="seismic")
plt.colorbar()
plt.axis("image")
plt.title(fileName)
plt.savefig(varname + ".png", dpi=dpi)
