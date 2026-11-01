import json
import numpy as np
from pyscripts.GRID import GRID
import matplotlib.pyplot as plt
import sys

def Extract_XY(params, it):
    grid = GRID(params)
    outputPath = params["out"]
    fileName = outputPath + "/FreeSurf%s_%d" % ("Vx", it)
    fileNameX = params["out"] + "/lon"
    fileNameY = params["out"] + "/lat"
    sliceX = params["sliceX"] - grid.frontNX
    sliceY = params["sliceY"] - grid.frontNY
    sliceZ = params["sliceZ"] - grid.frontNZ
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
        dataX = np.zeros([grid.NY, grid.NX])
        dataY = np.zeros([grid.NY, grid.NX])
        data = np.zeros([grid.NY, grid.NX])
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
            File = open("%s_Z_mpi_%d_%d_%d.bin" %
                        (fileName, mpiX, mpiY, mpiZ), "rb")

            ny = grid.ny[mpiY]
            nx = grid.nx[mpiX]

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

    return dataX, dataY

def Extract_wavefield(params, it, var):
    grid = GRID(params)
    outputPath = params["out"]
    fileName = outputPath + "/FreeSurf%s_%d" % (var, it)
    sliceX = params["sliceX"] - grid.frontNX
    sliceY = params["sliceY"] - grid.frontNY
    sliceZ = params["sliceZ"] - grid.frontNZ
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
        data = np.zeros([grid.NY, grid.NX])
    else:
        data = np.zeros([grid.NY, grid.NX])


    for mpiY in range(grid.PY):
        for mpiX in range(grid.PX):
            mpiZ = grid.PZ - 1
            File = open("%s_Z_mpi_%d_%d_%d.bin" %
                        (fileName, mpiX, mpiY, mpiZ), "rb")

            ny = grid.ny[mpiY]
            nx = grid.nx[mpiX]

            data_ = np.fromfile(File, dtype='float32', count=ny * nx)

            J = grid.frontNY[mpiY]
            J_ = grid.frontNY[mpiY] + ny
            I = grid.frontNX[mpiX]
            I_ = grid.frontNX[mpiX] + nx

            if FAST_AXIS == 'Z':
                data[I:I_, J:J_] = np.reshape(data_, (nx, ny))
            else:
                data[J:J_, I:I_] = np.reshape(data_, (ny, nx))

    print("successfully read data from %s" % fileName)
    return data

def Intensity_by_PGV(PGV, params):
    grid = GRID(params)
    Intensity = np.zeros(PGV.shape)
    for j in range(grid.NY):
        for i in range(grid.NX):
            if PGV[j, i] >= 1.76:
                Intensity[j, i] = 11
            if PGV[j, i] >= 0.815 and PGV[j, i] < 1.76:
                Intensity[j, i] = 10
            if PGV[j, i] >= 0.379 and PGV[j, i] < 0.815:
                Intensity[j, i] = 9
            if PGV[j, i] >= 0.177 and PGV[j, i] < 0.379:
                Intensity[j, i] = 8
            if PGV[j, i] >= 0.0818 and PGV[j, i] < 0.177:
                Intensity[j, i] = 7
            if PGV[j, i] >= 0.0381 and PGV[j, i] < 0.0818:
                Intensity[j, i] = 6
            if PGV[j, i] >= 0.0177 and PGV[j, i] < 0.0381:
                Intensity[j, i] = 5
            if PGV[j, i] >= 0.00820 and PGV[j, i] < 0.0177:
                Intensity[j, i] = 4
            if PGV[j, i] >= 0.00382 and PGV[j, i] < 0.00820:
                Intensity[j, i] = 3
            if PGV[j, i] >= 0.00178 and PGV[j, i] < 0.00382:
                Intensity[j, i] = 2
            if PGV[j, i] < 0.00178:
                Intensity[j, i] = 1
    return Intensity

def Intensity_by_PGA(PGA, params):
    grid = GRID(params)
    Intensity = np.zeros(PGA.shape)
    for j in range(grid.NY):
        for i in range(grid.NX):
            if PGA[j, i] >= 17.3:
                Intensity[j, i] = 11
            if PGA[j, i] >= 8.31 and PGA[j, i] < 17.3:
                Intensity[j, i] = 10
            if PGA[j, i] >= 4.02 and PGA[j, i] < 8.31:
                Intensity[j, i] = 9
            if PGA[j, i] >= 1.95 and PGA[j, i] < 4.02:
                Intensity[j, i] = 8
            if PGA[j, i] >= 0.937 and PGA[j, i] < 1.95:
                Intensity[j, i] = 7
            if PGA[j, i] >= 0.457 and PGA[j, i] < 0.937:
                Intensity[j, i] = 6
            if PGA[j, i] >= 0.223 and PGA[j, i] < 0.457:
                Intensity[j, i] = 5
            if PGA[j, i] >= 0.109 and PGA[j, i] < 0.223:
                Intensity[j, i] = 4
            if PGA[j, i] >= 0.0529 and PGA[j, i] < 0.109:
                Intensity[j, i] = 3
            if PGA[j, i] >= 0.0258 and PGA[j, i] < 0.0529:
                Intensity[j, i] = 2
            if PGA[j, i] < 0.0258:
                Intensity[j, i] = 1
    return Intensity
    
jsonsFile = open("params.json")
params = json.load(jsonsFile)
grid = GRID(params)

NX = params["NX"]
NY = params["NY"]
IT_SKIP = params["IT_SKIP"]
TMAX = params["TMAX"]
DT = params["DT"]
NT = int(TMAX / DT)

PGV = np.zeros([grid.NY, grid.NX])
PGA = np.zeros([grid.NY, grid.NX])

dataX, dataY = Extract_XY(params, 0)

for it in range(0, NT, IT_SKIP):
    if it == 0:
        Vx0 = np.zeros([grid.NY, grid.NX])
        Vy0 = np.zeros([grid.NY, grid.NX])
        Vz0 = np.zeros([grid.NY, grid.NX])
    else:
        Vx0 = Vx
        Vy0 = Vy
        Vz0 = Vz
    
    Vx = Extract_wavefield(params, it, "Vx")
    Vy = Extract_wavefield(params, it, "Vy")
    Vz = Extract_wavefield(params, it, "Vz")
    V = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    PGV = np.maximum(PGV, V)

    Ax = np.abs(Vx - Vx0) / (DT * IT_SKIP)
    Ay = np.abs(Vy - Vy0) / (DT * IT_SKIP)
    Az = np.abs(Vz - Vz0) / (DT * IT_SKIP)
    A = np.sqrt(Ax**2 + Ay**2 + Az**2)
    PGA = np.maximum(PGA, A)

Intensity_PGV = Intensity_by_PGV(PGV, params)
Intensity_PGA = Intensity_by_PGA(PGA, params)

nPML = 12
lon = dataX[nPML:NY - nPML, nPML:NX-nPML]
lat = dataY[nPML:NY - nPML, nPML:NX-nPML]
PGV = PGV[nPML:NY - nPML, nPML:NX-nPML]
PGA = PGA[nPML:NY - nPML, nPML:NX-nPML]
Intensity_PGV = Intensity_PGV[nPML:NY - nPML, nPML:NX-nPML]
Intensity_PGA = Intensity_PGA[nPML:NY - nPML, nPML:NX-nPML]

max_Intensity_PGV = np.max(Intensity_PGV)
max_Intensity_PGA = np.max(Intensity_PGA)
max_Intensity = max(max_Intensity_PGA, max_Intensity_PGV)

dpi = 300
plt.figure()
plt.subplot(221)
plt.pcolormesh(lon, lat, PGV, cmap="seismic")
plt.colorbar()
plt.axis("image")
plt.title("PGV")

plt.subplot(222)
plt.pcolormesh(lon, lat, Intensity_PGV, cmap="seismic")
plt.colorbar()
plt.axis("image")
plt.title("Intensity by PGV")
plt.clim(0, max_Intensity)

plt.subplot(223)
plt.pcolormesh(lon, lat, PGA, cmap="seismic")
plt.colorbar()
plt.axis("image")
plt.title("PGA")

plt.subplot(224)
plt.pcolormesh(lon, lat, Intensity_PGA, cmap="seismic")
plt.colorbar()
plt.axis("image")
plt.title("Intensity by PGA")
plt.clim(0, max_Intensity)

plt.tight_layout()
plt.show()
plt.savefig("./img/Intensity_from_output.png", dpi=dpi)

# Save data
data = np.zeros([(grid.NY - 2 * nPML) * (grid.NX - 2 * nPML), 6])
data[:, 0] = lon.flatten()
data[:, 1] = lat.flatten()
data[:, 2] = PGV.flatten()
data[:, 3] = Intensity_PGV.flatten()
data[:, 4] = PGA.flatten()
data[:, 5] = Intensity_PGA.flatten()
np.savetxt("./data/Intensity_from_output.txt", data, fmt="%10.6f", delimiter="\t",
           header="lon\tlat\tPGV\tIntensity_PGV\tPGA\tIntensity_PGA")