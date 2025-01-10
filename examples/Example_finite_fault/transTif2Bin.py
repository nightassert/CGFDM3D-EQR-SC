# ================================================================
#   ESS, Southern University of Science and Technology
#
#   File Name: transTif2Bin.py
#   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
#           Tianhong Xu, 12231218@mail.sustech.edu.cn
#   Created Time: 2025-1-9
#   Discription: Transfer tif files to bin files
#
#   Reference:
#      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
#      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
#
# =================================================================*/

import json
import struct
import numpy as np
import imageio
import sys
import matplotlib.pyplot as plt


def tif2bin(lonStart, latStart, terrainTifPath, terrainBinPath, FAST_AXIS):
    tifFileName = "%s/srtm_%dN%dE.tif" % (terrainTifPath, latStart, lonStart)
    # terrain = np.zeros( [NY, NX] )
    terrain = np.flipud(imageio.imread(tifFileName))

    (NY, NX) = np.shape(terrain)
    # print( "NY = %d, NX = %d" % ( NY, NX ) )
    binFileName = "%s/srtm_%dN%dE.bin" % (terrainBinPath, latStart, lonStart)
    print("Tif file is being converted to Binary file: %s" % binFileName)
    binFile = open(binFileName, "wb")
    if FAST_AXIS == 'Z':
        for i in range(NX):
            for j in range(NY):
                t = struct.pack("f", float(terrain[j, i]))
                binFile.write(t)
    else:
        for j in range(NY):
            for i in range(NX):
                t = struct.pack("f", float(terrain[j, i]))
                binFile.write(t)


jsonsFile = open("params.json")
params = json.load(jsonsFile)

lonStart = params["lonStart"]
latStart = params["latStart"]

blockX = params["blockX"]
blockY = params["blockY"]
FAST_AXIS = params["FAST_AXIS"]

degreePerBlockX = 5
degreePerBlockY = 5

lonEnd = lonStart + (blockX - 1) * degreePerBlockX
latEnd = latStart + (blockY - 1) * degreePerBlockY

lonList = np.linspace(lonStart, lonEnd, blockX)
latList = np.linspace(latStart, latEnd, blockY)
print(latList)
print(lonList)

terrainTifPath = params["TerrainTif"]
terrainBinPath = params["TerrainDir"]

for lat in latList:
    for lon in lonList:
        tif2bin(int(lon), int(lat), terrainTifPath, terrainBinPath, FAST_AXIS)
