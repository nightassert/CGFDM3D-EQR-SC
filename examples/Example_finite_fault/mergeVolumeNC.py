#!/usr/bin/env python3
# ================================================================
#   ESS, Southern University of Science and Technology
#
#   File Name: mergeVolumeNC.py
#   Description: Merge the per-rank output/volume_mpi_X_Y_Z.nc files
#                produced by volume_io.cpp into one NetCDF file.
#
#   Usage:
#       python3 mergeVolumeNC.py output -o volume.nc
#       python3 mergeVolumeNC.py output/volume_mpi_*.nc -o volume.nc
#       python3 mergeVolumeNC.py output -o volume.nc --deflate 1
#
#   The input paths may be files or directories; directories are searched
#   for volume_mpi_*.nc.  The merged file keeps the dimensions
#   (time, z, y, x), the variables vx/vy/vz in m/s, the global grid index
#   coordinates I/J/K and the time axis.  Before writing, the script
#   checks that the sub-blocks tile the requested block exactly (no gap,
#   no overlap) and that all files have the same time axis.
# ================================================================

import argparse
import glob
import os
import sys
from itertools import combinations

import numpy as np

try:
    import netCDF4 as nc
except ImportError:
    sys.exit("netCDF4 is required: pip install netCDF4")


def humanSize(numBytes):
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(numBytes)
    i = 0
    while value >= 1024.0 and i < len(units) - 1:
        value /= 1024.0
        i += 1
    return "%.2f %s" % (value, units[i])


def findFiles(inputs):
    files = []
    for path in inputs:
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, "volume_mpi_*.nc"))))
        elif os.path.isfile(path):
            files.append(path)
        else:
            sys.exit("no such file or directory: %s" % path)

    unique = []
    seen = set()
    for path in files:
        real = os.path.abspath(path)
        if real not in seen:
            seen.add(real)
            unique.append(path)

    if not unique:
        sys.exit("no volume_mpi_*.nc file found")
    return unique


def readMeta(path):
    with nc.Dataset(path, "r") as ds:
        required = ["vx", "vy", "vz", "I", "J", "K", "time"]
        for name in required:
            if name not in ds.variables:
                sys.exit("%s: missing variable '%s'" % (path, name))

        I = np.asarray(ds.variables["I"][:], dtype=np.int64)
        J = np.asarray(ds.variables["J"][:], dtype=np.int64)
        K = np.asarray(ds.variables["K"][:], dtype=np.int64)
        if I.size == 0 or J.size == 0 or K.size == 0:
            sys.exit("%s: empty block" % path)

        for name, arr in (("I", I), ("J", J), ("K", K)):
            if not np.array_equal(arr, np.arange(arr[0], arr[0] + arr.size)):
                sys.exit("%s: coordinate '%s' is not a contiguous range" % (path, name))

        nt = len(ds.dimensions["time"])
        time = np.asarray(ds.variables["time"][:], dtype=np.float64)
        shape = ds.variables["vx"].shape
        expected = (nt, K.size, J.size, I.size)
        if shape != expected:
            sys.exit("%s: vx shape %s does not match (time, z, y, x) = %s"
                     % (path, shape, expected))

        attrs = {}
        for name in ds.ncattrs():
            value = ds.getncattr(name)
            if isinstance(value, np.generic):
                value = value.item()
            attrs[name] = value

    for name in ("global_xStart", "global_xEnd",
                 "global_yStart", "global_yEnd",
                 "global_zStart", "global_zEnd"):
        if name not in attrs:
            sys.exit("%s: missing global attribute '%s'" % (path, name))

    return {
        "path": path,
        "i0": int(I[0]), "nx": int(I.size),
        "j0": int(J[0]), "ny": int(J.size),
        "k0": int(K[0]), "nz": int(K.size),
        "nt": int(nt), "time": time,
        "attrs": attrs,
    }


def checkTiling(blocks, gx, gy, gz):
    gx0 = gx[0]
    gy0 = gy[0]
    gz0 = gz[0]
    gx1, gy1, gz1 = gx[1], gy[1], gz[1]

    for b in blocks:
        if (b["i0"] < gx0 or b["i0"] + b["nx"] - 1 > gx1 or
                b["j0"] < gy0 or b["j0"] + b["ny"] - 1 > gy1 or
                b["k0"] < gz0 or b["k0"] + b["nz"] - 1 > gz1):
            sys.exit("%s: local block is outside the global block" % b["path"])

    total = (gx1 - gx0 + 1) * (gy1 - gy0 + 1) * (gz1 - gz0 + 1)
    covered = sum(b["nx"] * b["ny"] * b["nz"] for b in blocks)
    if covered != total:
        sys.exit("the sub-blocks cover %d points but the global block has %d; "
                 "some files are missing or duplicated" % (covered, total))

    for a, b in combinations(blocks, 2):
        if (a["i0"] <= b["i0"] + b["nx"] - 1 and b["i0"] <= a["i0"] + a["nx"] - 1 and
                a["j0"] <= b["j0"] + b["ny"] - 1 and b["j0"] <= a["j0"] + a["ny"] - 1 and
                a["k0"] <= b["k0"] + b["nz"] - 1 and b["k0"] <= a["k0"] + a["nz"] - 1):
            sys.exit("%s and %s overlap" % (a["path"], b["path"]))


def chunkShape(nx, ny, nz, target=4000000):
    cx = min(nx, 256)
    cy = min(ny, 256)
    cz = min(nz, 32)
    while cx * cy * cz > target:
        if cz > 1:
            cz = max(1, cz // 2)
        elif cy > 1:
            cy = max(1, cy // 2)
        else:
            cx = max(1, cx // 2)
    return (1, cz, cy, cx)


def main():
    parser = argparse.ArgumentParser(
        description="Merge per-rank volume_mpi_X_Y_Z.nc files into one NetCDF file.")
    parser.add_argument("inputs", nargs="+",
                        help="per-rank .nc files or directories containing them")
    parser.add_argument("-o", "--out", default="volume.nc",
                        help="output file (default: volume.nc)")
    parser.add_argument("--deflate", type=int, default=0, choices=range(0, 10),
                        help="zlib compression level 0-9 (default: 0, off)")
    args = parser.parse_args()

    files = findFiles(args.inputs)
    blocks = [readMeta(path) for path in files]

    first = blocks[0]
    attrs = first["attrs"]
    gx = (int(attrs["global_xStart"]), int(attrs["global_xEnd"]))
    gy = (int(attrs["global_yStart"]), int(attrs["global_yEnd"]))
    gz = (int(attrs["global_zStart"]), int(attrs["global_zEnd"]))

    nt = first["nt"]
    for b in blocks:
        for name in ("global_xStart", "global_xEnd", "global_yStart", "global_yEnd",
                     "global_zStart", "global_zEnd", "global_NX", "global_NY", "global_NZ"):
            if b["attrs"].get(name) != attrs.get(name):
                sys.exit("%s: inconsistent attribute '%s'" % (b["path"], name))
        if b["nt"] != nt:
            sys.exit("%s: %d time steps, expected %d" % (b["path"], b["nt"], nt))
        if b["time"].shape != first["time"].shape or \
                not np.allclose(b["time"], first["time"], rtol=0, atol=1e-6):
            sys.exit("%s: time axis differs from the other files" % b["path"])

    checkTiling(blocks, gx, gy, gz)

    nx = gx[1] - gx[0] + 1
    ny = gy[1] - gy[0] + 1
    nz = gz[1] - gz[0] + 1

    print("files       : %d" % len(files))
    print("global block: x %d..%d, y %d..%d, z %d..%d (%d x %d x %d)"
          % (gx[0], gx[1], gy[0], gy[1], gz[0], gz[1], nx, ny, nz))
    print("time steps  : %d" % nt)
    print("output size : %s (uncompressed)"
          % humanSize(np.int64(nt) * nx * ny * nz * 3 * 4))

    # Sort for a deterministic writing order.
    blocks.sort(key=lambda b: (b["k0"], b["j0"], b["i0"]))

    outPath = args.out
    if os.path.abspath(outPath) in [os.path.abspath(f) for f in files]:
        sys.exit("output file would overwrite an input file")

    dsOut = nc.Dataset(outPath, "w", format="NETCDF4")
    try:
        dsOut.createDimension("time", None)
        dsOut.createDimension("z", nz)
        dsOut.createDimension("y", ny)
        dsOut.createDimension("x", nx)

        chunks = chunkShape(nx, ny, nz)
        compress = args.deflate > 0
        kwargs = dict(zlib=compress, shuffle=compress)
        if compress:
            kwargs["complevel"] = args.deflate
            kwargs["chunksizes"] = chunks

        vxVar = dsOut.createVariable("vx", "f4", ("time", "z", "y", "x"), **kwargs)
        vyVar = dsOut.createVariable("vy", "f4", ("time", "z", "y", "x"), **kwargs)
        vzVar = dsOut.createVariable("vz", "f4", ("time", "z", "y", "x"), **kwargs)
        for var in (vxVar, vyVar, vzVar):
            var.units = "m/s"

        IVar = dsOut.createVariable("I", "i4", ("x",))
        JVar = dsOut.createVariable("J", "i4", ("y",))
        KVar = dsOut.createVariable("K", "i4", ("z",))
        IVar[:] = np.arange(gx[0], gx[1] + 1, dtype=np.int32)
        JVar[:] = np.arange(gy[0], gy[1] + 1, dtype=np.int32)
        KVar[:] = np.arange(gz[0], gz[1] + 1, dtype=np.int32)
        IVar.long_name = "global grid index in X"
        JVar.long_name = "global grid index in Y"
        KVar.long_name = "global grid index in Z (NZ-1 is the free surface)"

        timeVar = dsOut.createVariable("time", "f4", ("time",))
        timeVar[:] = first["time"]
        timeVar.units = "s"

        for name, value in attrs.items():
            dsOut.setncattr(name, value)
        dsOut.setncattr("merged_files", len(files))
        dsOut.setncattr("merged_by", "mergeVolumeNC.py")

        dsOut.sync()

        for index, b in enumerate(blocks, start=1):
            i0 = b["i0"] - gx[0]
            j0 = b["j0"] - gy[0]
            k0 = b["k0"] - gz[0]
            with nc.Dataset(b["path"], "r") as dsIn:
                vxi = dsIn.variables["vx"]
                vyi = dsIn.variables["vy"]
                vzi = dsIn.variables["vz"]
                for t in range(nt):
                    vxVar[t, k0:k0 + b["nz"], j0:j0 + b["ny"], i0:i0 + b["nx"]] = \
                        vxi[t, :, :, :]
                    vyVar[t, k0:k0 + b["nz"], j0:j0 + b["ny"], i0:i0 + b["nx"]] = \
                        vyi[t, :, :, :]
                    vzVar[t, k0:k0 + b["nz"], j0:j0 + b["ny"], i0:i0 + b["nx"]] = \
                        vzi[t, :, :, :]
            print("  merged %d/%d: %s (x %d..%d, y %d..%d, z %d..%d)"
                  % (index, len(blocks), b["path"],
                     b["i0"], b["i0"] + b["nx"] - 1,
                     b["j0"], b["j0"] + b["ny"] - 1,
                     b["k0"], b["k0"] + b["nz"] - 1))
    finally:
        dsOut.close()

    print("written     : %s (%s)" % (outPath, humanSize(os.path.getsize(outPath))))


if __name__ == "__main__":
    main()
