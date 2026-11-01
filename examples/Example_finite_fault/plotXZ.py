# ================================================================
#   ESS, Southern University of Science and Technology
#
#   File Name: plotXZ.py
#   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
#           Tianhong Xu, 12231218@mail.sustech.edu.cn
#   Created Time: 2025-1-9
#   Discription: Plot Wavefield on X-Z plane
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

outputPath = params["out"]
fileNameX = params["out"] + "/coordX"
fileNameY = params["out"] + "/coordY"
fileNameZ = params["out"] + "/coordZ"

if var == "Vs" or var == "Vp" or var == "rho":
    fileName = outputPath + "/%s" % (var)
else:
    fileName = outputPath + "/%s_%d" % (var, it)
varname = "./img/SliceY_%s_%d" % (var, it)

print("Draw " + fileName)

FAST_AXIS = params["FAST_AXIS"]

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
    dataX = np.zeros([grid.NX, grid.NZ])
    dataZ = np.zeros([grid.NX, grid.NZ])
    data = np.zeros([grid.NX, grid.NZ])
else:
    dataX = np.zeros([grid.NZ, grid.NX])
    dataZ = np.zeros([grid.NZ, grid.NX])
    data = np.zeros([grid.NZ, grid.NX])


mpiY = mpiSliceY
for mpiZ in range(grid.PZ):
    for mpiX in range(grid.PX):
        fileX = open("%s_Y_mpi_%d_%d_%d.bin" %
                     (fileNameX, mpiX, mpiY, mpiZ), "rb")
        fileZ = open("%s_Y_mpi_%d_%d_%d.bin" %
                     (fileNameZ, mpiX, mpiY, mpiZ), "rb")
        file = open("%s_Y_mpi_%d_%d_%d.bin" %
                    (fileName, mpiX, mpiY, mpiZ), "rb")
        nx = grid.nx[mpiX]
        nz = grid.nz[mpiZ]
        print("nx = %d, nz = %d" % (nx, nz))
        datax = np.fromfile(fileX, dtype='float32', count=nx * nz)
        # print( np.shape( datax ) )
        dataz = np.fromfile(fileZ, dtype='float32', count=nx * nz)
        data_ = np.fromfile(file, dtype='float32', count=nx * nz)
        I = grid.frontNX[mpiX]
        I_ = grid.frontNX[mpiX] + nx
        K = grid.frontNZ[mpiZ]
        K_ = grid.frontNZ[mpiZ] + nz

        if FAST_AXIS == 'Z':
            dataX[I:I_, K:K_] = np.reshape(datax, (nx, nz))
            dataZ[I:I_, K:K_] = np.reshape(dataz, (nx, nz))
            data[I:I_, K:K_] = np.reshape(data_, (nx, nz))
        else:
            dataX[K:K_, I:I_] = np.reshape(datax, (nz, nx))
            dataZ[K:K_, I:I_] = np.reshape(dataz, (nz, nx))
            data[K:K_, I:I_] = np.reshape(data_, (nz, nx))


dpi = 300
vm = np.max(np.abs(data))
dataX /= 1000
dataZ /= 1000
if var == "Vs" or var == "Vp" or var == "rho":
    plt.pcolormesh(dataX, dataZ, data, cmap="seismic")
else:
    plt.pcolormesh(dataX, dataZ, data, vmax=vm, vmin=-vm, cmap="seismic")
plt.plot(dataX[-1, :], dataZ[-1, :], "k-")
plt.colorbar()
plt.axis("image")
plt.title(varname + ".png")
plt.savefig(varname + ".png", dpi=dpi)
