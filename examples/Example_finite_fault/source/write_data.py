# ================================================================
#   ESS, Southern University of Science and Technology
#
#   File Name: write_data.py
#   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
#           Tianhong Xu, 12231218@mail.sustech.edu.cn
#   Created Time: 2025-1-19
#   Discription: Write data to source file .bin
#
#   Reference:
#      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
#      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
#
# =================================================================*/

import struct

NPTS = 10000  # The num of subfaults
NT = 1000
dt = 1e-2

def writeData(sourceFileName, Lon, Lat, Z, Area, Strike, Dip, Rake, Rate):
    sourceFile = open(sourceFileName, "wb")
    value = struct.pack("i", NPTS)
    sourceFile.write(value)
    value = struct.pack("i", NT)
    sourceFile.write(value)
    value = struct.pack("f", dt)
    sourceFile.write(value)

    for i in range(NPTS):
        value = struct.pack("f",  Lon[i])
        sourceFile.write(value)
        value = struct.pack("f",  Lat[i])
        sourceFile.write(value)
        value = struct.pack("f",  Z[i])
        sourceFile.write(value)
        value = struct.pack("f",  Area[i])
        sourceFile.write(value)
        value = struct.pack("f",  Strike[i])
        sourceFile.write(value)
        value = struct.pack("f",  Dip[i])
        sourceFile.write(value)
        tvalue = struct.pack("f" * NT,  *(Rake[i, :]))
        sourceFile.write(tvalue)
        tvalue = struct.pack("f" * NT,  *(Rate[i, :]))
        sourceFile.write(tvalue)

    sourceFile.close()
