#!/usr/bin/env python
# ================================================================
#   ESS, Southern University of Science and Technology
#
#   File Name: plotContrast.py
#   Discription: Batch comparison between a CGFDM run and a WENO run.
#                Reads the raw binary output of the two runs and writes
#                side-by-side / difference figures into img/Contrast.
#
#   Usage:
#       python plotContrast.py
#       python plotContrast.py output/CGFDM output/WENO
#       python plotContrast.py --it-step 1000 --out img/Contrast
#
#   What is compared:
#       - free-surface velocity (and displacement when the build has
#         SOLVE_DISPLACEMENT) on the whole surface
#       - Vx/Vy/Vz on the X, Y and Z slices written by params.json
#       - station waveforms (stacked + one figure per station) and a
#         metrics table (peak, correlation, time shift, relative L2)
#       - PGVh/PGV (and PGAh/PGA when the build has SOLVE_PGA) read from
#         the PGV files, plus the Chinese intensity scale derived from them
#       - PGV/PGA and the intensity scale derived from the free-surface
#         snapshots in the same way as cal_Intensity_from_output.py
#       - the medium on the slices, as a check that both runs used the
#         same model
#
#   PX/PY/PZ are recovered for each run from the size of the plane tiles on
#   disk (params.json is the default candidate), so the two runs may have
#   been decomposed differently, and files left over from an earlier run
#   with another decomposition are detected and ignored.  The runs may even
#   have different time steps: the WENO scheme reduces DT when the 3-D CFL
#   limit demands it, in which case snapshots are paired by physical time,
#   not by step number.  DT is recovered from the number of samples in the
#   station files; --dt-cgfdm / --dt-weno override it.
#
#   Reference:
#      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
#      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
#
# =================================================================*/

import argparse
import csv
import glob
import itertools
import json
import os
import re
import sys
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from scipy import signal

# The terrain-following grid makes the coordinates of a slice non-monotonic,
# so pcolormesh warns about it every time a field is drawn.  The arrays are
# point values of a curvilinear grid, which pcolormesh handles correctly
# here; the warning is only noise.
warnings.filterwarnings(
    "ignore", message=".*not monotonically increasing or decreasing.*")

HERE = os.path.dirname(os.path.abspath(__file__))
HALO = 3
WSIZE = 9
CSIZE = 3

PLANE_TITLE = {"X": "SliceX (Y-Z plane)", "Y": "SliceY (X-Z plane)",
               "Z": "SliceZ (X-Y plane)"}
PLANE_FILE = {"X": "SliceX", "Y": "SliceY", "Z": "SliceZ"}

PGV_NAMES = ["PGVh", "PGV", "PGAh", "PGA"]

# Chinese seismic intensity scale, thresholds in m/s (PGV/PGVh) and m/s^2
# (PGA/PGAh), exactly as in plotPGV.py / cal_Intensity_from_output.py.
# Horizontal PGV is classified on the 4..11 part of the scale, horizontal PGA
# on the same 1..11 table as the three-component PGA, as those scripts do.
PGVH_EDGES = np.array([0.02, 0.04, 0.09, 0.18, 0.35, 0.71, 1.41])
PGV_EDGES = np.array([0.00178, 0.00382, 0.00820, 0.0177, 0.0381,
                      0.0818, 0.177, 0.379, 0.815, 1.76])
PGA_EDGES = np.array([0.0258, 0.0529, 0.109, 0.223, 0.457, 0.937,
                      1.95, 4.02, 8.31, 17.3])


# ---------------------------------------------------------------- utilities

def log(msg=""):
    print(msg, flush=True)


def warn(msg):
    print("WARNING: " + msg, flush=True)


# Files that could not be read are counted here and reported once per
# (directory, field, plane) at the end.  One line per rank file used to bury
# the message when an entire family had been written with an unexpected size.
READ_ISSUES = {}


def note_read_issue(outdir, base, plane, expected, got):
    key = (outdir, base, plane)
    rec = READ_ISSUES.get(key)
    if rec is None:
        READ_ISSUES[key] = [1, expected, got]
    else:
        rec[0] += 1


def read_issue_lines():
    """One line per field family whose rank files did not have the
    expected number of floats."""
    lines = []
    for key in sorted(READ_ISSUES):
        outdir, base, plane = key
        count, expected, got = READ_ISSUES[key]
        lines.append("%s: %d %s tile(s) on the %s plane had a wrong size "
                     "(expected %d floats, first had %d); left blank"
                     % (outdir, count, base, plane, expected, got))
    return lines


def finite_any(a):
    return np.isfinite(a).any()


def safe_max(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.nanmax(a)) if np.isfinite(a).any() else float("nan")


def safe_rms(a):
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).any():
        return float("nan")
    return float(np.sqrt(np.nanmean(a * a)))


def fmt_e(v):
    return "nan" if not np.isfinite(v) else "%.3e" % v


# ---------------------------------------------------------------- parameters

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the output of a CGFDM run and a WENO run.")
    parser.add_argument("cgfdm_dir", nargs="?", default="output/CGFDM",
                        help="CGFDM output folder (default: output/CGFDM)")
    parser.add_argument("weno_dir", nargs="?", default="output/WENO",
                        help="WENO output folder (default: output/WENO)")
    parser.add_argument("--out", default="img/Contrast",
                        help="figure folder (default: img/Contrast)")
    parser.add_argument("--it-step", type=int, default=1000,
                        help="compare every N steps of the reference run "
                             "(default: 1000)")
    parser.add_argument("--its", default="",
                        help="comma-separated list of reference steps, "
                             "overrides --it-step")
    parser.add_argument("--reference", choices=["cgfdm", "weno"],
                        default="cgfdm",
                        help="run whose step numbers define the snapshots")
    parser.add_argument("--nPML", type=int, default=None,
                        help="trim this many cells around the maps "
                             "(default: params.json nPML)")
    parser.add_argument("--no-filter", action="store_true",
                        help="do not band-pass the station waveforms")
    parser.add_argument("--fl", type=float, default=1e-10,
                        help="high-pass corner of the waveform filter (Hz)")
    parser.add_argument("--fh", type=float, default=2.0,
                        help="low-pass corner of the waveform filter (Hz)")
    parser.add_argument("--strict-filter", action="store_true",
                        help="filter every frame with a single fs=1/DT, as "
                             "the original plotWaveform.py does")
    parser.add_argument("--intensity-step", type=int, default=None,
                        help="step between snapshots used for the intensity "
                             "integration (default: every snapshot)")
    parser.add_argument("--dt-cgfdm", type=float, default=None,
                        help="override the CGFDM time step (s)")
    parser.add_argument("--dt-weno", type=float, default=None,
                        help="override the WENO time step (s)")
    parser.add_argument("--max-lag", type=float, default=5.0,
                        help="largest waveform time shift searched (s)")
    parser.add_argument("--skip-slices", action="store_true")
    parser.add_argument("--skip-freesurf", action="store_true")
    parser.add_argument("--skip-stations", action="store_true")
    parser.add_argument("--skip-pgv", action="store_true")
    parser.add_argument("--skip-intensity", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    return parser.parse_args()


def load_json(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def station_positions(params):
    try:
        station = load_json("station.json")
    except (IOError, OSError, ValueError):
        return {}
    points = station.get("station(point)", {})
    out = {}
    for key, pos in points.items():
        try:
            out[str(key)] = (int(pos[0]), int(pos[1]), int(pos[2]))
        except (TypeError, IndexError, ValueError):
            warn("bad station position for key %s" % key)
    return out


# ---------------------------------------------------------------- run / grid

RANK_RE = re.compile(r"_mpi_(\d+)_(\d+)_(\d+)\.bin$")

# Files that hold exactly one float per grid point of the plane tile they
# belong to.  Their size says which PX/PY/PZ wrote them, which is how the
# decomposition of a run is recovered below.  The highest rank index in the
# file names is not reliable on its own: a directory may still hold files of
# an older run that used a different decomposition.
TILE_FILE_RES = [
    # <field>_<it>_<plane>_mpi_x_y_z.bin -- V/T and free-surface snapshots
    (re.compile(r"^(?:V[xyz]|T(?:xx|yy|zz|xy|xz|yz)|FreeSurfV[xyz]|"
                r"FreeSurfU[xyz])_\d+_([XYZ])_mpi_(\d+)_(\d+)_(\d+)\.bin$"),
     1),
    # <field>_<plane>_mpi_x_y_z.bin -- model planes written once at start-up
    (re.compile(r"^(?:coord[XYZ]|Vp|Vs|rho)_([XYZ])_mpi_(\d+)_(\d+)_(\d+)"
                r"\.bin$"), 1),
    # lon_mpi_x_y_z.bin / lat_mpi_x_y_z.bin -- written on the top z rank
    (re.compile(r"^(?:lon|lat)_mpi_(\d+)_(\d+)_(\d+)\.bin$"), 0),
]


def list_files(outdir):
    try:
        return sorted(os.listdir(outdir))
    except OSError:
        raise SystemExit("Can't read output directory: %s" % outdir)


def tile_files(outdir, names):
    """(name, plane, x, y, z, bytes) of the files that hold a whole plane
    tile."""
    out = []
    for name in names:
        for pat, plane_group in TILE_FILE_RES:
            m = pat.match(name)
            if m is None:
                continue
            if plane_group:
                plane = m.group(plane_group)
                g = plane_group + 1
            else:
                plane = "Z"
                g = 1
            try:
                size = os.path.getsize(os.path.join(outdir, name))
            except OSError:
                size = -1
            out.append((name, plane, int(m.group(g)),
                        int(m.group(g + 1)), int(m.group(g + 2)), size))
            break
    return out


def max_nprocs(names):
    mx = [-1, -1, -1]
    for name in names:
        m = RANK_RE.search(name)
        if m is None:
            continue
        for i in range(3):
            v = int(m.group(i + 1))
            if v > mx[i]:
                mx[i] = v
    if min(mx) < 0:
        return None
    return mx[0] + 1, mx[1] + 1, mx[2] + 1


def score_decomposition(tiles, params, nprocs):
    """How many tiles on disk a PX/PY/PZ candidate explains by their size."""
    from pyscripts.GRID import GRID

    p = dict(params)
    p["PX"], p["PY"], p["PZ"] = nprocs
    grid = GRID(p)
    fast = params.get("FAST_AXIS", "X")
    matched = mismatched = outside = 0
    for (name, plane, x, y, z, size) in tiles:
        if x >= grid.PX or y >= grid.PY or z >= grid.PZ:
            outside += 1
            continue
        n1, n2 = tile_shape(plane, fast, x, y, z, grid)
        if size == 4 * n1 * n2:
            matched += 1
        else:
            mismatched += 1
    return matched, mismatched, outside


# Trying every decomposition up to the highest rank index is cheap because
# the tile sizes are read once and then only compared; above this many
# combinations the search falls back to params.json and the maxima.
MAX_CANDIDATES = 512


def candidate_decompositions(base, mx):
    """Decompositions worth testing against the files on disk.

    params.json is the declared one.  A directory can also hold leftovers of
    an older run that used a different decomposition, so the current one may
    be any tuple within the range of rank indices seen -- only the largest
    index per axis is not enough (e.g. an old x=1 file can coexist with a
    current PX=1 run).
    """
    if mx is None:
        return [base]
    if mx[0] * mx[1] * mx[2] > MAX_CANDIDATES:
        return [base] if base == mx else [base, mx]
    cands = [c for c in itertools.product(*[range(1, m + 1) for m in mx])
             if c != base]
    return [base] + cands


def detect_nprocs(outdir, params, names=None):
    """Recover the PX/PY/PZ a run directory was written with.

    The decomposition whose tiles match the file sizes on disk wins;
    params.json is the first candidate and therefore wins ties.  Leftover
    files of an earlier run with another decomposition do not fit the sizes
    and cannot force the wrong grid.
    Returns (PX, PY, PZ, matched, mismatched, outside).
    """
    if names is None:
        names = list_files(outdir)
    base = (int(params["PX"]), int(params["PY"]), int(params["PZ"]))
    candidates = candidate_decompositions(base, max_nprocs(names))

    tiles = tile_files(outdir, names)
    if not tiles:
        warn("no *_mpi_X_Y_Z.bin files in %s, using PX/PY/PZ from "
             "params.json" % outdir)
        return base[0], base[1], base[2], 0, 0, 0

    best = None
    for cand in candidates:
        matched, mismatched, outside = score_decomposition(
            tiles, params, cand)
        key = (matched, -mismatched, -outside)
        if best is None or key > best[0]:
            best = (key, cand, matched, mismatched, outside)
    _, chosen, matched, mismatched, outside = best
    if matched == 0:
        warn("%s: no tile file on disk agrees with any decomposition inside "
             "the rank range %s; the PX/PY/PZ of this run could not be "
             "recovered -- were the files written with other grid "
             "parameters?" % (outdir, "x".join(str(m) for m in
                                               (max_nprocs(names) or base))))
    return (chosen[0], chosen[1], chosen[2], matched, mismatched, outside)


IT_RE = re.compile(
    r"^(?:FreeSurf)?V[xyz]_(\d+)_[XYZ]_mpi_(\d+)_(\d+)_(\d+)\.bin$")
IT_RE_ANY = re.compile(
    r"^[A-Za-z]+_(\d+)_[XYZ]_mpi_(\d+)_(\d+)_(\d+)\.bin$")


def available_its(names, nprocs):
    """Snapshot steps written with the given decomposition; files left over
    from another decomposition are ignored."""
    px, py, pz = nprocs

    def collect(pat):
        its = set()
        for name in names:
            m = pat.match(name)
            if m is None:
                continue
            if (int(m.group(2)) < px and int(m.group(3)) < py
                    and int(m.group(4)) < pz):
                its.add(int(m.group(1)))
        return its

    its = collect(IT_RE)
    if not its:
        its = collect(IT_RE_ANY)
    return sorted(its)


# One entry per output family written by the solver.  The group is the plane
# suffix (X/Y/Z) when the name carries one, so the report shows at a glance
# which slices actually exist.
INVENTORY = [
    ("V snapshots",
     re.compile(r"^V[xyz]_\d+_([XYZ])_mpi_\d+_\d+_\d+\.bin$")),
    ("T snapshots",
     re.compile(r"^T(?:xx|yy|zz|xy|xz|yz)_\d+_([XYZ])_mpi_\d+_\d+_\d+\.bin$")),
    ("FreeSurf V",
     re.compile(r"^FreeSurfV[xyz]_\d+_([XYZ])_mpi_\d+_\d+_\d+\.bin$")),
    ("FreeSurf U",
     re.compile(r"^FreeSurfU[xyz]_\d+_([XYZ])_mpi_\d+_\d+_\d+\.bin$")),
    ("stations", re.compile(r"^station_mpi_\d+_\d+_\d+\.bin$")),
    ("PGV/PGA", re.compile(r"^PGV_([XYZ])_mpi_\d+_\d+_\d+\.bin$")),
    ("coordinates",
     re.compile(r"^coord[XYZ]_([XYZ])_mpi_\d+_\d+_\d+\.bin$")),
    ("medium", re.compile(r"^(?:Vs|Vp|rho)_([XYZ])_mpi_\d+_\d+_\d+\.bin$")),
    ("lon/lat", re.compile(r"^(?:lon|lat)_mpi_\d+_\d+_\d+\.bin$")),
]


def inventory(outdir):
    """Summarise the output families present in one run directory."""
    try:
        names = os.listdir(outdir)
    except OSError:
        return ["unreadable directory"]
    lines = []
    for label, pat in INVENTORY:
        groups = {}
        for name in names:
            m = pat.match(name)
            if not m:
                continue
            plane = m.group(1) if m.groups() else "-"
            groups[plane] = groups.get(plane, 0) + 1
        if not groups:
            continue
        detail = ", ".join("%s plane: %d" % (g, groups[g])
                           for g in sorted(groups))
        lines.append("%-13s %s" % (label + ":", detail))
    if not lines:
        lines.append("no recognised output files")
    return lines


def rank_containing(front, sizes, idx):
    for r in range(len(sizes)):
        if int(front[r]) <= idx < int(front[r]) + int(sizes[r]):
            return r
    return -1


def dt_from_stations(outdir, grid, params, positions):
    """Recover the real DT from the number of samples in a station file.

    propagate() computes NT = TMAX / DT, so DT = TMAX / NT.  The WENO run
    may use a smaller DT than params.json if calc_CFL() had to clamp it;
    the station files still carry the true number of steps.
    """
    if not positions:
        return None
    groups = {}
    for key, (X, Y, Z) in positions.items():
        rx = rank_containing(grid.frontNX, grid.nx, X)
        ry = rank_containing(grid.frontNY, grid.ny, Y)
        rz = rank_containing(grid.frontNZ, grid.nz, Z)
        if rx < 0 or ry < 0 or rz < 0:
            continue
        groups.setdefault((rx, ry, rz), []).append(key)
    for (x, y, z), keys in groups.items():
        fn = os.path.join(outdir, "station_mpi_%d_%d_%d.bin" % (x, y, z))
        if not os.path.isfile(fn):
            continue
        num = len(keys)
        size = os.path.getsize(fn)
        body = size - 4 * CSIZE * num
        per = 4 * WSIZE * num
        if body <= 0 or body % per:
            continue
        nsteps = body // per
        if nsteps > 1:
            return float(params["TMAX"]) / nsteps
    return None


class Run(object):
    def __init__(self, name, outdir, params, dt_override=None):
        from pyscripts.GRID import GRID

        self.name = name
        self.outdir = outdir
        self.params = params
        self.fast = params.get("FAST_AXIS", "X")
        names = list_files(outdir)
        (self.PX, self.PY, self.PZ,
         matched, mismatched, outside) = detect_nprocs(outdir, params, names)
        self.stale_files = outside + mismatched
        declared = (int(params["PX"]), int(params["PY"]), int(params["PZ"]))
        if (self.PX, self.PY, self.PZ) != declared:
            log("  (%s: the files were written with PX/PY/PZ = %d/%d/%d, "
                "not %d/%d/%d from params.json)"
                % ((name,) + (self.PX, self.PY, self.PZ) + declared))
        if outside or mismatched:
            warn("%s: %d file(s) carry rank indices outside PX/PY/PZ = "
                 "%d/%d/%d and %d file(s) have a tile size that does not fit "
                 "that decomposition; they are ignored (typically leftovers "
                 "of an earlier run with another decomposition -- compare "
                 "clean output directories)."
                 % (outdir, outside, self.PX, self.PY, self.PZ, mismatched))
        p = dict(params)
        p["PX"], p["PY"], p["PZ"] = self.PX, self.PY, self.PZ
        self.grid = GRID(p)
        self.its = available_its(names, (self.PX, self.PY, self.PZ))

        positions = station_positions(params)
        dt = dt_override
        if dt is None:
            dt = dt_from_stations(outdir, self.grid, params, positions)
        if dt is None:
            dt = float(params["DT"])
        else:
            nominal = float(params["DT"])
            if dt_override is None and abs(dt - nominal) > 1e-2 * nominal:
                log("  (%s: DT recovered from the station file is %.9g s; "
                    "params.json says %.9g s)" % (name, dt, nominal))
        self.dt = dt

        if not self.its:
            warn("%s contains no wavefield snapshots" % outdir)

    def describe(self):
        return ("%s: %s\n  PX/PY/PZ = %d/%d/%d, DT = %.9g s, "
                "snapshots = %d (%s ... %s)"
                % (self.name, self.outdir, self.PX, self.PY, self.PZ,
                   self.dt, len(self.its),
                   self.its[0] if self.its else "-",
                   self.its[-1] if self.its else "-"))


# ---------------------------------------------------------------- file reads

def global_shape(plane, fast, grid):
    if fast == "X":
        return {"X": (grid.NZ, grid.NY),
                "Y": (grid.NZ, grid.NX),
                "Z": (grid.NY, grid.NX)}[plane]
    return {"X": (grid.NY, grid.NZ),
            "Y": (grid.NX, grid.NZ),
            "Z": (grid.NX, grid.NY)}[plane]


def tile_shape(plane, fast, x, y, z, grid):
    nx = int(grid.nx[x])
    ny = int(grid.ny[y])
    nz = int(grid.nz[z])
    if fast == "X":
        return {"X": (nz, ny), "Y": (nz, nx), "Z": (ny, nx)}[plane]
    return {"X": (ny, nz), "Y": (nx, nz), "Z": (nx, ny)}[plane]


def place_slice(plane, fast, x, y, z, grid):
    nx = int(grid.nx[x])
    ny = int(grid.ny[y])
    nz = int(grid.nz[z])
    I = int(grid.frontNX[x])
    J = int(grid.frontNY[y])
    K = int(grid.frontNZ[z])
    if fast == "X":
        if plane == "X":
            return slice(K, K + nz), slice(J, J + ny)
        if plane == "Y":
            return slice(K, K + nz), slice(I, I + nx)
        return slice(J, J + ny), slice(I, I + nx)
    if plane == "X":
        return slice(J, J + ny), slice(K, K + nz)
    if plane == "Y":
        return slice(I, I + nx), slice(K, K + nz)
    return slice(I, I + nx), slice(J, J + ny)


def plane_ranks(plane, grid, slice_idx, free_surface):
    nx_ranks = len(grid.nx)
    ny_ranks = len(grid.ny)
    nz_ranks = len(grid.nz)
    if free_surface:
        return [(x, y, nz_ranks - 1)
                for y in range(ny_ranks) for x in range(nx_ranks)]
    if plane == "X":
        rx = rank_containing(grid.frontNX, grid.nx, slice_idx)
        if rx < 0:
            return []
        return [(rx, y, z)
                for z in range(nz_ranks) for y in range(ny_ranks)]
    if plane == "Y":
        ry = rank_containing(grid.frontNY, grid.ny, slice_idx)
        if ry < 0:
            return []
        return [(x, ry, z)
                for z in range(nz_ranks) for x in range(nx_ranks)]
    rz = rank_containing(grid.frontNZ, grid.nz, slice_idx)
    if rz < 0:
        return []
    return [(x, y, rz)
            for y in range(ny_ranks) for x in range(nx_ranks)]


def plane_filename(run, base, it, plane, x, y, z, bare=False):
    # data2D_output_bin() names every field <base>_<plane>_mpi_x_y_z.bin, with
    # the step number in front when the field is a snapshot.  lon/lat are the
    # exception: outputLonLat() writes <base>_mpi_x_y_z.bin.
    if bare:
        return os.path.join(run.outdir, "%s_mpi_%d_%d_%d.bin"
                            % (base, x, y, z))
    if it is None:
        return os.path.join(run.outdir, "%s_%s_mpi_%d_%d_%d.bin"
                            % (base, plane, x, y, z))
    return os.path.join(run.outdir, "%s_%d_%s_mpi_%d_%d_%d.bin"
                        % (base, it, plane, x, y, z))


def gather_plane(run, base, it, plane, slice_idx=None, free_surface=False,
                 bare=False):
    """Assemble one global 2-D field from the per-rank binary files."""
    grid = run.grid
    fast = run.fast
    if free_surface:
        plane = "Z"
    shape = global_shape(plane, fast, grid)
    out = np.full(shape, np.nan, dtype=np.float64)
    ranks = plane_ranks(plane, grid, slice_idx, free_surface)
    found = 0
    for (x, y, z) in ranks:
        fn = plane_filename(run, base, it, plane, x, y, z, bare)
        if not os.path.isfile(fn):
            continue
        tshape = tile_shape(plane, fast, x, y, z, grid)
        raw = np.fromfile(fn, dtype=np.float32)
        if raw.size != tshape[0] * tshape[1]:
            note_read_issue(run.outdir, base, plane, tshape[0] * tshape[1],
                            raw.size)
            continue
        out[place_slice(plane, fast, x, y, z, grid)] = \
            raw.reshape(tshape).astype(np.float64)
        found += 1
    return out, found, len(ranks)


def var_exists(run, base, it, plane, free_surface=False, bare=False):
    if bare:
        pat = os.path.join(run.outdir, "%s_mpi_*.bin" % base)
    elif it is None:
        pat = os.path.join(run.outdir, "%s_%s_mpi_*.bin" % (base, plane))
    else:
        pat = os.path.join(run.outdir,
                           "%s_%d_%s_mpi_*.bin" % (base, it, plane))
    for fn in glob.glob(pat):
        m = RANK_RE.search(os.path.basename(fn))
        if m is None:
            return True
        if (int(m.group(1)) < run.PX and int(m.group(2)) < run.PY
                and int(m.group(3)) < run.PZ):
            return True
    return False


def fallback_coords(plane, grid, fast, params):
    """Index-based coordinates used when the coordinate files are gone."""
    DH = float(grid.DH) / 1000.0
    NX, NY, NZ = grid.NX, grid.NY, grid.NZ
    cx = float(params.get("centerX", NX // 2))
    cy = float(params.get("centerY", NY // 2))
    depth = float(params.get("Depth(km)", params.get("Depth", 80.0)))
    xs = (np.arange(NX) - cx) * DH
    ys = (np.arange(NY) - cy) * DH
    zs = -depth + depth * np.arange(NZ) / max(NZ - 1, 1)
    if fast == "X":
        if plane == "X":
            Y, Z = np.meshgrid(ys, zs, indexing="ij")
            return Y.T, Z.T
        if plane == "Y":
            X, Z = np.meshgrid(xs, zs, indexing="ij")
            return X.T, Z.T
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        return X.T, Y.T
    if plane == "X":
        return np.meshgrid(ys, zs, indexing="ij")
    if plane == "Y":
        return np.meshgrid(xs, zs, indexing="ij")
    return np.meshgrid(xs, ys, indexing="ij")


def coords_for_plane(runs, plane, slice_idx, params):
    """Coordinate pair (in km) of a slice, from the first run that has it."""
    names = {"X": ("coordY", "coordZ"),
             "Y": ("coordX", "coordZ"),
             "Z": ("coordX", "coordY")}[plane]
    labels = {"X": ("y (km)", "z (km)"),
              "Y": ("x (km)", "z (km)"),
              "Z": ("x (km)", "y (km)")}[plane]
    for run in runs:
        a, fa, _ = gather_plane(run, names[0], None, plane, slice_idx)
        b, fb, _ = gather_plane(run, names[1], None, plane, slice_idx)
        if fa > 0 and fb > 0 and finite_any(a) and finite_any(b):
            return a / 1000.0, b / 1000.0, labels[0], labels[1]
    for run in runs:
        a, b = fallback_coords(plane, run.grid, run.fast, params)
        return a, b, labels[0], labels[1]
    raise SystemExit("No run available for coordinates")


def surface_coords(runs, slice_idx, params):
    """Longitude/latitude of the free surface (km fallback)."""
    for run in runs:
        lon, fl, _ = gather_plane(run, "lon", None, "Z", free_surface=True,
                                  bare=True)
        lat, ft, _ = gather_plane(run, "lat", None, "Z", free_surface=True,
                                  bare=True)
        if fl > 0 and ft > 0 and finite_any(lon) and finite_any(lat):
            return lon, lat, "longitude (deg)", "latitude (deg)"
    zidx = slice_idx["Z"] if isinstance(slice_idx, dict) else slice_idx
    x, y, xl, yl = coords_for_plane(runs, "Z", zidx, params)
    return x, y, "x (km)", "y (km)"


def read_pgv(run):
    """Return (ncomp, NY, NX) PGV/PGA maps, or None."""
    grid = run.grid
    fast = run.fast
    z = run.PZ - 1
    out = None
    for y in range(run.PY):
        for x in range(run.PX):
            fn = os.path.join(run.outdir, "PGV_Z_mpi_%d_%d_%d.bin"
                              % (x, y, z))
            if not os.path.isfile(fn):
                continue
            nx = int(grid.nx[x])
            ny = int(grid.ny[y])
            raw = np.fromfile(fn, dtype=np.float32)
            if raw.size % (nx * ny) != 0:
                warn("%s: size %d is not a multiple of %d"
                     % (fn, raw.size, nx * ny))
                continue
            nc = raw.size // (nx * ny)
            if out is None:
                shape = ((nc, grid.NY, grid.NX) if fast == "X"
                         else (nc, grid.NX, grid.NY))
                out = np.full(shape, np.nan, dtype=np.float64)
            elif nc != out.shape[0]:
                warn("%s: component count changed, ignored" % fn)
                continue
            # PGV.cpp writes pgv[pos * PGVSIZE + c], i.e. the components of
            # one grid point are contiguous; split them before reshaping.
            comp = raw.reshape(nx * ny, nc).T
            if fast == "X":
                tile = comp.reshape(nc, ny, nx)
                out[:, int(grid.frontNY[y]):int(grid.frontNY[y]) + ny,
                    int(grid.frontNX[x]):int(grid.frontNX[x]) + nx] = tile
            else:
                tile = comp.reshape(nc, nx, ny)
                out[:, int(grid.frontNX[x]):int(grid.frontNX[x]) + nx,
                    int(grid.frontNY[y]):int(grid.frontNY[y]) + ny] = tile
    return out


# ---------------------------------------------------------------- stats

def compare_stats(cg, wn):
    m = np.isfinite(cg) & np.isfinite(wn)
    st = dict(n=int(np.count_nonzero(m)), max_cg=float("nan"),
              max_wn=float("nan"), max_diff=float("nan"),
              rms_diff=float("nan"), corr=float("nan"),
              rel_l2=float("nan"))
    if not m.any():
        return st
    a = cg[m]
    b = wn[m]
    d = a - b
    st["max_cg"] = float(np.max(np.abs(a)))
    st["max_wn"] = float(np.max(np.abs(b)))
    st["max_diff"] = float(np.max(np.abs(d)))
    st["rms_diff"] = float(np.sqrt(np.mean(d * d)))
    den = float(np.sqrt(np.mean(a * a)))
    st["rel_l2"] = float(np.sqrt(np.mean(d * d)) / den) if den > 0 else \
        float("nan")
    if a.std() > 0 and b.std() > 0:
        st["corr"] = float(np.corrcoef(a, b)[0, 1])
    return st


def fmt_stats(st):
    return ("max|CG|=%s  max|WN|=%s  max|d|=%s  RMS(d)=%s  corr=%.4f  "
            "relL2=%s" % (fmt_e(st["max_cg"]), fmt_e(st["max_wn"]),
                          fmt_e(st["max_diff"]), fmt_e(st["rms_diff"]),
                          st["corr"], fmt_e(st["rel_l2"])))


def locate_extreme(arr, X, Y):
    """(x, y, value) of the point where |arr| is largest."""
    if not finite_any(arr):
        return None
    absa = np.where(np.isfinite(arr), np.abs(arr), -np.inf)
    j, i = np.unravel_index(int(np.argmax(absa)), arr.shape)
    try:
        x = float(np.asarray(X)[j, i])
        y = float(np.asarray(Y)[j, i])
    except (IndexError, TypeError, ValueError):
        x = y = float("nan")
    return (x, y, float(arr[j, i]))


def fmt_locs(st):
    """Where the maxima sit, e.g. "maxCG@(38.0,-12.4) 8.50e-02".

    Coordinates are in the same (km) order as the figure axes: the first
    value is the horizontal coordinate of a vertical slice (or x of a
    horizontal slice), the second is depth (or y).
    """
    parts = []
    for key, name in (("loc_cg", "maxCG"), ("loc_wn", "maxWN"),
                      ("loc_diff", "maxD")):
        loc = st.get(key)
        if loc is None:
            parts.append("%s@-" % name)
        else:
            parts.append("%s@(%.1f,%.1f) %s"
                         % (name, loc[0], loc[1], fmt_e(loc[2])))
    return "  ".join(parts)


def class_stats(icg, iwn):
    m = np.isfinite(icg) & np.isfinite(iwn)
    st = dict(equal=float("nan"), within1=float("nan"),
              mean_abs=float("nan"), max_abs=float("nan"), n=0)
    if not m.any():
        return st
    d = icg[m] - iwn[m]
    st["n"] = int(d.size)
    st["equal"] = float(100.0 * np.mean(d == 0))
    st["within1"] = float(100.0 * np.mean(np.abs(d) <= 1))
    st["mean_abs"] = float(np.mean(np.abs(d)))
    st["max_abs"] = float(np.max(np.abs(d)))
    return st


def fmt_class_stats(st):
    return ("identical=%.1f%%  within +-1=%.1f%%  mean|dI|=%.3f  "
            "max|dI|=%.2f" % (st["equal"], st["within1"], st["mean_abs"],
                              st["max_abs"]))


def intensity_map(name, arr):
    """Chinese intensity scale; arr must be finite-scaled (m/s or m/s^2)."""
    with np.errstate(invalid="ignore"):
        if name == "PGVh":
            out = np.digitize(arr, PGVH_EDGES) + 4
            out = np.clip(out, 4, 11)
        else:
            if name == "PGV":
                out = np.digitize(arr, PGV_EDGES) + 1
            else:   # PGA / PGAh
                out = np.digitize(arr, PGA_EDGES) + 1
            out = np.clip(out, 1, 11)
    out = out.astype(np.float64)
    out[~np.isfinite(arr)] = np.nan
    return out


# ---------------------------------------------------------------- figures

def _clean_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("auto")


def save_field_compare(path, title, cg, wn, X, Y, xlabel, ylabel,
                       cmap="seismic", stats=None):
    """Three panels: CGFDM, WENO and CGFDM - WENO."""
    if not (finite_any(cg) or finite_any(wn)):
        warn("skip %s: no finite data" % os.path.basename(path))
        return compare_stats(cg, wn)
    st = compare_stats(cg, wn) if stats is None else stats
    diff = cg - wn
    st["loc_cg"] = locate_extreme(cg, X, Y)
    st["loc_wn"] = locate_extreme(wn, X, Y)
    st["loc_diff"] = locate_extreme(diff, X, Y)

    vmax = np.nanmax(np.abs(np.concatenate([cg.ravel(), wn.ravel()])))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    dmax = np.nanmax(np.abs(diff)) if finite_any(diff) else 0.0
    if not np.isfinite(dmax) or dmax <= 0:
        dmax = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4),
                             constrained_layout=True)
    panels = [(cg, "CGFDM", cmap, -vmax, vmax),
              (wn, "WENO", cmap, -vmax, vmax),
              (diff, "CGFDM - WENO", cmap, -dmax, dmax)]
    for ax, (data, ptitle, pcmap, vmin, vmx) in zip(axes, panels):
        pcm = ax.pcolormesh(X, Y, data, cmap=pcmap, vmin=vmin, vmax=vmx,
                            shading="auto")
        fig.colorbar(pcm, ax=ax, shrink=0.9)
        _clean_axes(ax, xlabel, ylabel, ptitle)
    axes[2].text(0.02, 0.02, fmt_stats(st) + "\n" + fmt_locs(st),
                 transform=axes[2].transAxes, fontsize=7, va="bottom",
                 ha="left", bbox=dict(boxstyle="round", fc="white",
                                      alpha=0.75))
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return st


def save_class_compare(path, title, icg, iwn, X, Y, xlabel, ylabel):
    """Intensity classes: CGFDM, WENO and the class difference."""
    if not (finite_any(icg) or finite_any(iwn)):
        warn("skip %s: no finite data" % os.path.basename(path))
        return class_stats(icg, iwn)
    st = class_stats(icg, iwn)
    diff = icg - iwn
    dmax = np.nanmax(np.abs(diff)) if finite_any(diff) else 0.0
    dmax = max(1.0, float(np.ceil(dmax))) if np.isfinite(dmax) else 1.0

    norm = BoundaryNorm(np.arange(0.5, 12.0, 1.0), 11)
    cmap = discrete_cmap("jet", 11)

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.4),
                             constrained_layout=True)
    for ax, (data, ptitle) in zip(axes[:2], [(icg, "CGFDM"), (iwn, "WENO")]):
        pcm = ax.pcolormesh(X, Y, data, cmap=cmap, norm=norm,
                            shading="auto")
        cb = fig.colorbar(pcm, ax=ax, shrink=0.9, ticks=np.arange(1, 12))
        cb.set_label("intensity")
        _clean_axes(ax, xlabel, ylabel, ptitle)
    pcm = axes[2].pcolormesh(X, Y, diff, cmap="bwr", vmin=-dmax, vmax=dmax,
                             shading="auto")
    fig.colorbar(pcm, ax=axes[2], shrink=0.9)
    _clean_axes(axes[2], xlabel, ylabel, "CGFDM - WENO (class)")
    axes[2].text(0.02, 0.02, fmt_class_stats(st),
                 transform=axes[2].transAxes, fontsize=8, va="bottom",
                 ha="left", bbox=dict(boxstyle="round", fc="white",
                                      alpha=0.75))
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=250)
    plt.close(fig)
    return st


def trim2(arr, n):
    if n is None or n <= 0:
        return arr
    if arr.shape[0] <= 2 * n or arr.shape[1] <= 2 * n:
        return arr
    return arr[n:-n, n:-n]


def discrete_cmap(name, n):
    """n-colour colormap, for both old and new matplotlib versions."""
    cmaps = getattr(matplotlib, "colormaps", None)
    if cmaps is not None:
        return cmaps[name].resampled(n)
    return plt.get_cmap(name, n)


# ---------------------------------------------------------------- snapshots

def pair_label(runs, it_cg, it_wn):
    return ("CGFDM it=%d (t=%.3f s)  |  WENO it=%d (t=%.3f s)"
            % (it_cg, it_cg * runs[0].dt, it_wn, it_wn * runs[1].dt))


def select_snapshots(ref, other, it_step, explicit):
    if explicit:
        its = [it for it in explicit if it in set(ref.its)]
        missing = [it for it in explicit if it not in set(ref.its)]
        if missing:
            warn("reference run has no snapshots at it = %s"
                 % ", ".join(str(i) for i in missing))
    elif it_step and it_step > 0:
        its = [it for it in ref.its if it % it_step == 0]
    else:
        its = list(ref.its)
    if not its:
        warn("no snapshot is a multiple of --it-step %d in the reference "
             "run; using every available snapshot" % it_step)
        its = list(ref.its)
    if not ref.its:
        raise SystemExit("The reference run has no snapshots to compare")

    same_dt = abs(ref.dt - other.dt) <= 1e-6 * max(ref.dt, other.dt)
    pairs = []
    for it in its:
        if same_dt and it in set(other.its):
            ot = it
        elif other.its:
            t = it * ref.dt
            ot = min(other.its, key=lambda o: abs(o * other.dt - t))
        else:
            ot = None
        pairs.append((it, ot))
    return pairs


def compare_snapshots(runs, pairs, outdir, slice_idx, nPML, summary):
    cg, wn = runs
    params = cg.params
    made = 0

    plane_vars = ["Vx", "Vy", "Vz", "Txx", "Tyy", "Tzz",
                  "Txy", "Txz", "Tyz"]

    for it_cg, it_wn in pairs:
        if it_wn is None:
            continue
        label = pair_label(runs, it_cg, it_wn)
        log("[snapshot] " + label)

        if not args.skip_freesurf:
            X, Y, xl, yl = surface_coords(runs, slice_idx, params)
            comps_cg = {}
            comps_wn = {}
            for var in ("Vx", "Vy", "Vz"):
                if not (var_exists(cg, "FreeSurf" + var, it_cg, "Z",
                                   True) and
                        var_exists(wn, "FreeSurf" + var, it_wn, "Z", True)):
                    warn("FreeSurf%s missing at one of the snapshots" % var)
                    continue
                a, fa, _ = gather_plane(cg, "FreeSurf" + var, it_cg, "Z",
                                        free_surface=True)
                b, fb, _ = gather_plane(wn, "FreeSurf" + var, it_wn, "Z",
                                        free_surface=True)
                if fa == 0 or fb == 0:
                    warn("FreeSurf%s: no rank files found" % var)
                    continue
                comps_cg[var] = a
                comps_wn[var] = b
                st = save_field_compare(
                    os.path.join(outdir, "FreeSurf%s_%d.png" % (var, it_cg)),
                    "FreeSurf%s  %s" % (var, label), a, b, X, Y, xl, yl)
                made += 1
                summary.append("FreeSurf%s it=%d/%d: %s  %s"
                               % (var, it_cg, it_wn, fmt_stats(st),
                                  fmt_locs(st)))

            # velocity magnitude at the surface
            if len(comps_cg) == 3 and len(comps_wn) == 3:
                vc = np.sqrt(sum(v * v for v in comps_cg.values()))
                vw = np.sqrt(sum(v * v for v in comps_wn.values()))
                st = save_field_compare(
                    os.path.join(outdir, "FreeSurfV_%d.png" % it_cg),
                    "FreeSurf |V|  %s" % label, vc, vw, X, Y, xl, yl)
                made += 1
                summary.append("FreeSurf|V| it=%d/%d: %s  %s"
                               % (it_cg, it_wn, fmt_stats(st),
                                  fmt_locs(st)))

            # surface displacement, when the build solves for it
            for var in ("Ux", "Uy", "Uz"):
                if not (var_exists(cg, "FreeSurf" + var, it_cg, "Z", True)
                        and var_exists(wn, "FreeSurf" + var, it_wn, "Z",
                                       True)):
                    continue
                a, fa, _ = gather_plane(cg, "FreeSurf" + var, it_cg, "Z",
                                        free_surface=True)
                b, fb, _ = gather_plane(wn, "FreeSurf" + var, it_wn, "Z",
                                        free_surface=True)
                if fa == 0 or fb == 0:
                    continue
                st = save_field_compare(
                    os.path.join(outdir, "FreeSurf%s_%d.png" % (var, it_cg)),
                    "FreeSurf%s  %s" % (var, label), a, b, X, Y, xl, yl)
                made += 1
                summary.append("FreeSurf%s it=%d/%d: %s  %s"
                               % (var, it_cg, it_wn, fmt_stats(st),
                                  fmt_locs(st)))

        if not args.skip_slices:
            for plane in ("X", "Y", "Z"):
                X, Y, xl, yl = coords_for_plane(runs, plane,
                                                slice_idx[plane], params)
                compared = 0
                for var in plane_vars:
                    if not (var_exists(cg, var, it_cg, plane) and
                            var_exists(wn, var, it_wn, plane)):
                        continue
                    a, fa, _ = gather_plane(cg, var, it_cg, plane,
                                            slice_idx[plane])
                    b, fb, _ = gather_plane(wn, var, it_wn, plane,
                                            slice_idx[plane])
                    if fa == 0 or fb == 0:
                        continue
                    name = "%s_%s_%d.png" % (PLANE_FILE[plane], var, it_cg)
                    st = save_field_compare(
                        os.path.join(outdir, name),
                        "%s  %s  %s" % (PLANE_FILE[plane], var, label),
                        a, b, X, Y, xl, yl)
                    made += 1
                    compared += 1
                    summary.append("%s %s it=%d/%d: %s  %s"
                                   % (PLANE_FILE[plane], var, it_cg, it_wn,
                                      fmt_stats(st), fmt_locs(st)))
                if compared == 0:
                    summary.append("%s: nothing compared at it=%d/%d "
                                   "(plane files missing in one of the runs)"
                                   % (PLANE_FILE[plane], it_cg, it_wn))
    return made


# ---------------------------------------------------------------- waveforms

def load_station_traces(run, positions):
    """Return {key: (NT, WSIZE) array} of physical traces."""
    grid = run.grid
    groups = {}
    for key, (X, Y, Z) in positions.items():
        rx = rank_containing(grid.frontNX, grid.nx, X)
        ry = rank_containing(grid.frontNY, grid.ny, Y)
        rz = rank_containing(grid.frontNZ, grid.nz, Z)
        if rx < 0 or ry < 0 or rz < 0:
            warn("station %s lies outside the grid" % key)
            continue
        groups.setdefault((rx, ry, rz), []).append(key)

    traces = {}
    for (x, y, z), keys in groups.items():
        fn = os.path.join(run.outdir, "station_mpi_%d_%d_%d.bin"
                          % (x, y, z))
        if not os.path.isfile(fn):
            warn("missing station file %s" % fn)
            continue
        num = len(keys)
        size = os.path.getsize(fn)
        body = size - 4 * CSIZE * num
        per = 4 * WSIZE * num
        if body <= 0 or body % per:
            warn("unexpected size of %s" % fn)
            continue
        nsteps = body // per
        with open(fn, "rb") as f:
            xyz = np.fromfile(f, dtype=np.int32,
                              count=CSIZE * num).reshape(num, 3)
            wave = np.fromfile(f, dtype=np.float32,
                               count=nsteps * num * WSIZE)
        if wave.size != nsteps * num * WSIZE:
            warn("truncated station file %s" % fn)
            continue
        wave = wave.reshape(num, nsteps, WSIZE)
        lookup = {(int(r[0]), int(r[1]), int(r[2])): i
                  for i, r in enumerate(xyz)}
        for key in keys:
            idx = lookup.get(positions[key])
            if idx is None:
                warn("station %s not found in %s" % (key, fn))
                continue
            traces[key] = wave[idx].astype(np.float64)
    return traces


def bandpass(x, dt, fl, fh):
    nyq = 0.5 / dt
    lo = fl / nyq
    hi = fh / nyq
    if not (0.0 < lo < hi < 1.0):
        return x
    b, a = signal.butter(2, [lo, hi], "bandpass")
    return signal.filtfilt(b, a, x)


def best_shift(a, b, dt, max_seconds):
    """Shift s with b[k] ~ a[k+s]; s > 0 means b (WENO) leads a (CGFDM)."""
    n = min(a.size, b.size)
    a = a[:n] - np.mean(a[:n])
    b = b[:n] - np.mean(b[:n])
    if n < 8 or dt <= 0:
        return 0.0, float("nan")
    smax = min(int(max_seconds / dt), n // 4)
    if smax < 1:
        return 0.0, float("nan")
    best_s = 0
    best_r = np.inf
    for s in range(-smax, smax + 1):
        if s >= 0:
            aa = a[s:]
            bb = b[:n - s]
        else:
            aa = a[:n + s]
            bb = b[-s:]
        if aa.size < n // 2:
            continue
        r = float(np.sqrt(np.mean((aa - bb) ** 2)))
        if r < best_r:
            best_r = r
            best_s = s
    shift = best_s * dt
    if best_s >= 0:
        aa = a[best_s:]
        bb = b[:n - best_s]
    else:
        aa = a[:n + best_s]
        bb = b[-best_s:]
    corr = np.corrcoef(aa, bb)[0, 1] if (aa.std() > 0 and bb.std() > 0) \
        else float("nan")
    return shift, float(corr)


def compare_waveforms(runs, positions, args, outdir, datadir, summary):
    cg, wn = runs
    keys = [k for k in positions.keys()]
    traces_cg = load_station_traces(cg, positions)
    traces_wn = load_station_traces(wn, positions)
    keys = [k for k in keys if k in traces_cg and k in traces_wn]
    if not keys:
        warn("no comparable station waveforms")
        return []

    tmax = float(cg.params["TMAX"])
    proc = {}
    rows = []

    for key in keys:
        for ci, comp in enumerate(("Vx", "Vy", "Vz")):
            a = traces_cg[key][:, ci]
            b = traces_wn[key][:, ci]
            ta = np.arange(a.size) * cg.dt
            tb = np.arange(b.size) * wn.dt
            if not args.no_filter:
                # Filters use each run's own sampling rate, which is the
                # correct thing to do when the two DT differ.  --strict-filter
                # reproduces the original plotWaveformCompare.py, which fed
                # both files through fs = 1/DT of params.json.
                a = bandpass(a, cg.dt, args.fl, args.fh)
                b = bandpass(b, cg.dt if args.strict_filter else wn.dt,
                             args.fl, args.fh)
            t_end = min(tmax, ta[-1], tb[-1])
            if t_end <= 0:
                continue
            t = ta[ta <= t_end]
            if t.size < 8:
                continue
            a_on = np.interp(t, ta, a)
            b_on = np.interp(t, tb, b)

            shift, corr_shift = best_shift(a_on, b_on, cg.dt,
                                           args.max_lag)
            m = (a_on.std() > 0 and b_on.std() > 0)
            corr0 = float(np.corrcoef(a_on, b_on)[0, 1]) if m \
                else float("nan")
            diff = a_on - b_on
            rms = float(np.sqrt(np.mean(diff * diff)))
            pa = float(np.max(np.abs(a_on)))
            pb = float(np.max(np.abs(b_on)))
            rel = rms / np.sqrt(np.mean(a_on * a_on)) \
                if np.sqrt(np.mean(a_on * a_on)) > 0 else float("nan")

            proc[(key, ci)] = (t, a_on, b_on)
            rows.append(dict(station=key, component=comp,
                             peak_cgfdm=pa, peak_weno=pb,
                             rms_diff=rms, rel_l2=rel, corr=corr0,
                             corr_aligned=corr_shift, shift_s=shift,
                             dt_cgfdm=cg.dt, dt_weno=wn.dt))

    if not proc:
        warn("station traces could not be processed")
        return []

    # stacked comparison, in the style of plotWaveformCompare.py
    nkeys = len(keys)
    fig = plt.figure(figsize=(18, 12))
    fig.subplots_adjust(wspace=0.08)
    axes = [fig.add_subplot(1, 3, i + 1) for i in range(3)]
    disWid = 2
    handles = None
    for row, key in enumerate(keys):
        off = row * disWid
        for ci, comp in enumerate(("Vx", "Vy", "Vz")):
            if (key, ci) not in proc:
                continue
            t, a, b = proc[(key, ci)]
            lc, = axes[ci].plot(t, a + off, color="blue", linestyle="-",
                                linewidth=1.0, label="CGFDM")
            lw, = axes[ci].plot(t, b + off, color="red", linestyle="--",
                                linewidth=1.0, label="WENO")
            if handles is None:
                handles = (lc, lw)
    for ci, comp in enumerate(("v_x", "v_y", "v_z")):
        axes[ci].set_xlim([0, tmax])
        axes[ci].set_ylim(-1.5, -1.5 + 2 * nkeys + 1)
        axes[ci].set_xlabel("time (s)")
        axes[ci].set_title(comp)
    axes[0].set_yticks(np.arange(0, 2 * nkeys, 2))
    axes[0].set_yticklabels(keys, fontsize=12)
    axes[0].set_ylabel("Amplitude (m/s)")
    for ax in axes[1:]:
        ax.set_yticks([])
    if handles is not None:
        axes[0].legend(handles=list(handles), labels=["CGFDM", "WENO"],
                       loc="upper right")
    fig.savefig(os.path.join(outdir, "Waveforms_compare.png"), dpi=250)
    plt.close(fig)

    # one figure per station
    for key in keys:
        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                 constrained_layout=True)
        for ci, comp in enumerate(("Vx", "Vy", "Vz")):
            if (key, ci) not in proc:
                continue
            t, a, b = proc[(key, ci)]
            axes[ci].plot(t, a, color="blue", linewidth=1.0,
                          label="CGFDM")
            axes[ci].plot(t, b, color="red", linestyle="--", linewidth=1.0,
                          label="WENO")
            axes[ci].set_ylabel("%s (m/s)" % comp)
            axes[ci].grid(alpha=0.3)
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("time (s)")
        fig.suptitle("Station %s" % key)
        fig.savefig(os.path.join(outdir, "station_%s.png" % key), dpi=250)
        plt.close(fig)

    # metrics table
    csv_path = os.path.join(datadir, "waveform_metrics.csv")
    fields = ["station", "component", "peak_cgfdm", "peak_weno",
              "rms_diff", "rel_l2", "corr", "corr_aligned", "shift_s",
              "dt_cgfdm", "dt_weno"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    good = [r for r in rows if np.isfinite(r["corr"])]
    if good:
        summary.append("waveforms: %d traces, mean corr = %.4f, "
                       "min corr = %.4f, mean |shift| = %.3f s"
                       % (len(good),
                          float(np.mean([r["corr"] for r in good])),
                          float(np.min([r["corr"] for r in good])),
                          float(np.mean([abs(r["shift_s"]) for r in good]))))
    return rows


# ---------------------------------------------------------------- PGV/intensity

def compare_pgv(runs, nPML, outdir, datadir, summary, slice_idx, params):
    cg, wn = runs
    pgv_cg = read_pgv(cg)
    pgv_wn = read_pgv(wn)
    if pgv_cg is None or pgv_wn is None:
        warn("PGV files missing in at least one run, skipping PGV maps")
        return 0
    X, Y, xl, yl = surface_coords(runs, slice_idx, params)
    ncomp = min(pgv_cg.shape[0], pgv_wn.shape[0])
    made = 0
    agreement = []

    for ic in range(ncomp):
        name = PGV_NAMES[ic] if ic < len(PGV_NAMES) else "comp%d" % ic
        a = trim2(pgv_cg[ic], nPML)
        b = trim2(pgv_wn[ic], nPML)
        Xt = trim2(X, nPML)
        Yt = trim2(Y, nPML)

        st = save_field_compare(
            os.path.join(outdir, "%s_compare.png" % name),
            "%s (from the PGV files)" % name, a, b, Xt, Yt, xl, yl)
        made += 1
        summary.append("%s: %s  %s" % (name, fmt_stats(st), fmt_locs(st)))

        ia = intensity_map(name, a)
        ib = intensity_map(name, b)
        cs = save_class_compare(
            os.path.join(outdir, "Intensity_%s_compare.png" % name),
            "Intensity from %s" % name, ia, ib, Xt, Yt, xl, yl)
        made += 1
        agreement.append(dict(quantity=name, **cs))
        summary.append("Intensity(%s): %s" % (name, fmt_class_stats(cs)))

    if agreement:
        with open(os.path.join(datadir, "intensity_agreement.csv"), "w",
                  newline="") as f:
            fields = ["quantity", "n", "equal", "within1", "mean_abs",
                      "max_abs"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(agreement)
    return made


def snapshot_products(run, step=None, label=""):
    """PGV/PGA and intensity inputs integrated over the snapshot series."""
    its = list(run.its)
    if step and step > 0:
        its = [it for it in its if it % step == 0]
    if not its:
        return None
    grid = run.grid
    shape = ((grid.NY, grid.NX) if run.fast == "X"
             else (grid.NX, grid.NY))
    pgvh = np.full(shape, np.nan)
    pgv = np.full(shape, np.nan)
    pgah = np.full(shape, np.nan)
    pga = np.full(shape, np.nan)
    times = []
    max_v = []
    rms_v = []
    prev = None
    prev_it = None
    used = 0

    for k, it in enumerate(its):
        vx, f1, _ = gather_plane(run, "FreeSurfVx", it, "Z",
                                 free_surface=True)
        vy, f2, _ = gather_plane(run, "FreeSurfVy", it, "Z",
                                 free_surface=True)
        vz, f3, _ = gather_plane(run, "FreeSurfVz", it, "Z",
                                 free_surface=True)
        if f1 == 0 or f2 == 0 or f3 == 0:
            if k == 0:
                warn("%s: no free-surface snapshots, skipping the "
                     "snapshot intensity" % run.name)
                return None
            continue
        used += 1
        vh = np.sqrt(vx * vx + vy * vy)
        v = np.sqrt(vx * vx + vy * vy + vz * vz)
        pgvh = np.fmax(pgvh, vh)
        pgv = np.fmax(pgv, v)
        times.append(it * run.dt)
        max_v.append(safe_max(v))
        rms_v.append(safe_rms(v))
        if prev is not None:
            dth = (it - prev_it) * run.dt
            if dth > 0:
                ax = np.abs(vx - prev[0]) / dth
                ay = np.abs(vy - prev[1]) / dth
                az = np.abs(vz - prev[2]) / dth
                pgah = np.fmax(pgah, np.sqrt(ax * ax + ay * ay))
                pga = np.fmax(pga, np.sqrt(ax * ax + ay * ay + az * az))
        prev = (vx, vy, vz)
        prev_it = it
        if used % 10 == 0:
            log("  [%s] %d/%d snapshots integrated (it = %d)"
                % (run.name, k + 1, len(its), it))

    if used == 0:
        return None
    return dict(pgvh=pgvh, pgv=pgv, pgah=pgah, pga=pga,
                times=np.array(times), max_v=np.array(max_v),
                rms_v=np.array(rms_v))


def compare_snapshot_intensity(runs, args, outdir, slice_idx, params,
                               nPML, summary):
    products = []
    for run in runs:
        log("[intensity] integrating free-surface snapshots of %s..."
            % run.name)
        p = snapshot_products(run, args.intensity_step, run.name)
        products.append(p)
    if products[0] is None or products[1] is None:
        warn("snapshot intensity could not be computed for both runs")
        return 0

    X, Y, xl, yl = surface_coords(runs, slice_idx, params)
    made = 0
    for key, name in (("pgvh", "PGVh"), ("pgv", "PGV"),
                      ("pgah", "PGAh"), ("pga", "PGA")):
        a = trim2(products[0][key], nPML)
        b = trim2(products[1][key], nPML)
        if not (finite_any(a) or finite_any(b)):
            continue
        Xt = trim2(X, nPML)
        Yt = trim2(Y, nPML)
        st = save_field_compare(
            os.path.join(outdir, "Snapshot_%s_compare.png" % name),
            "%s from the free-surface snapshots" % name,
            a, b, Xt, Yt, xl, yl)
        made += 1
        summary.append("snapshot %s: %s  %s"
                       % (name, fmt_stats(st), fmt_locs(st)))

        ia = intensity_map(name, a)
        ib = intensity_map(name, b)
        cs = save_class_compare(
            os.path.join(outdir,
                         "Snapshot_Intensity_%s_compare.png" % name),
            "Intensity from snapshot %s" % name, ia, ib, Xt, Yt, xl, yl)
        made += 1
        summary.append("snapshot intensity(%s): %s"
                       % (name, fmt_class_stats(cs)))

    # max / RMS of the surface velocity magnitude against time
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             constrained_layout=True)
    for run, p, color in ((runs[0], products[0], "blue"),
                          (runs[1], products[1], "red")):
        axes[0].plot(p["times"], p["max_v"], color=color, label=run.name)
        axes[1].plot(p["times"], p["rms_v"], color=color, label=run.name,
                     linestyle="--")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("max |V| at the surface (m/s)")
    axes[0].set_title("Surface velocity, maximum")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("RMS |V| at the surface (m/s)")
    axes[1].set_title("Surface velocity, RMS")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.savefig(os.path.join(outdir, "SurfaceVelocity_time.png"), dpi=250)
    plt.close(fig)
    made += 1
    return made


# ---------------------------------------------------------------- model check

def compare_model(runs, outdir, slice_idx, params, summary):
    cg, wn = runs
    made = 0
    for plane in ("X", "Y", "Z"):
        X, Y, xl, yl = coords_for_plane(runs, plane, slice_idx[plane],
                                        params)
        fields = {}
        for var in ("Vs", "Vp", "rho"):
            if not (var_exists(cg, var, None, plane) and
                    var_exists(wn, var, None, plane)):
                continue
            a, fa, _ = gather_plane(cg, var, None, plane, slice_idx[plane])
            b, fb, _ = gather_plane(wn, var, None, plane, slice_idx[plane])
            if fa == 0 or fb == 0 or not (finite_any(a) or finite_any(b)):
                continue
            fields[var] = (a, b)
            st = compare_stats(a, b)
            summary.append("model %s %s: %s"
                           % (PLANE_FILE[plane], var, fmt_stats(st)))
        if not fields:
            summary.append("model %s: nothing compared (medium files missing "
                           "in one of the runs)" % PLANE_FILE[plane])
            continue

        fig, axes = plt.subplots(3, 3, figsize=(16, 13),
                                 constrained_layout=True)
        for row, var in enumerate(("Vs", "Vp", "rho")):
            if var not in fields:
                continue
            a, b = fields[var]
            d = a - b
            vmin = np.nanmin([np.nanmin(a), np.nanmin(b)])
            vmax = np.nanmax([np.nanmax(a), np.nanmax(b)])
            dmax = safe_max(np.abs(d))
            if not np.isfinite(dmax) or dmax <= 0:
                dmax = 1.0
            panels = [(a, "CGFDM", "viridis", vmin, vmax),
                      (b, "WENO", "viridis", vmin, vmax),
                      (d, "CGFDM - WENO", "seismic", -dmax, dmax)]
            for col, (data, ptitle, pcmap, lo, hi) in enumerate(panels):
                ax = axes[row, col]
                pcm = ax.pcolormesh(X, Y, data, cmap=pcmap,
                                    vmin=lo, vmax=hi, shading="auto")
                fig.colorbar(pcm, ax=ax, shrink=0.85)
                _clean_axes(ax, xl, yl, "%s  %s" % (var, ptitle))
        fig.suptitle("Medium on %s" % PLANE_TITLE[plane], fontsize=12)
        fig.savefig(os.path.join(outdir,
                                 "Model_%s_compare.png" % PLANE_FILE[plane]),
                    dpi=250)
        plt.close(fig)
        made += 1
    return made


# ---------------------------------------------------------------- main

def main():
    global args
    args = parse_args()
    cwd0 = os.getcwd()
    cg_dir = os.path.abspath(os.path.join(cwd0, args.cgfdm_dir))
    wn_dir = os.path.abspath(os.path.join(cwd0, args.weno_dir))
    out_dir = os.path.abspath(os.path.join(cwd0, args.out))
    data_dir = os.path.join(out_dir, "data")

    os.chdir(HERE)
    sys.path.insert(0, HERE)

    for d in (cg_dir, wn_dir):
        if not os.path.isdir(d):
            raise SystemExit("Output directory not found: %s" % d)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    params = load_json("params.json")
    if args.nPML is None:
        args.nPML = int(params.get("nPML", 0))

    cg = Run("CGFDM", cg_dir, params, args.dt_cgfdm)
    wn = Run("WENO", wn_dir, params, args.dt_weno)
    runs = (cg, wn)
    slice_idx = {"X": int(params["sliceX"]),
                 "Y": int(params["sliceY"]),
                 "Z": int(params["sliceZ"])}

    log("")
    log("================ CGFDM vs WENO ================")
    log(cg.describe())
    log(wn.describe())

    inv_lines = []
    for run in runs:
        inv_lines.append("%s  %s" % (run.name, run.outdir))
        inv_lines.append("  decomposition: PX/PY/PZ = %d/%d/%d, "
                         "%d stale file(s) ignored"
                         % (run.PX, run.PY, run.PZ, run.stale_files))
        inv_lines.extend("  " + line for line in inventory(run.outdir))
    log("")
    log("[files] output inventory")
    for line in inv_lines:
        log("  " + line)
    if abs(cg.dt - wn.dt) > 1e-6 * max(cg.dt, wn.dt):
        log("NOTE: the two runs use different time steps; snapshots are "
            "paired by physical time.")
    log("figures: %s" % out_dir)

    summary = []
    summary.append(cg.describe())
    summary.append(wn.describe())

    explicit = [int(s) for s in re.split(r"[,\s]+", args.its)
                if s.strip()] if args.its else []

    ref, other = (cg, wn) if args.reference == "cgfdm" else (wn, cg)
    if ref.its:
        pairs = select_snapshots(ref, other, args.it_step, explicit)
        if ref is wn:
            # select_snapshots works on the reference run; the rest of the
            # script always expects (CGFDM step, WENO step).
            pairs = [(o, r) for (r, o) in pairs]
    else:
        pairs = []
        warn("reference run has no snapshots; snapshot comparison skipped")

    log("")
    log("snapshots to compare: %s"
        % ", ".join("%d/%d" % p for p in pairs if p[1] is not None))
    with open(os.path.join(data_dir, "snapshot_pairs.csv"), "w",
              newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cgfdm_it", "cgfdm_t", "weno_it", "weno_t",
                         "dt_minus"])
        for it_cg, it_wn in pairs:
            if it_wn is None:
                continue
            writer.writerow([it_cg, it_cg * cg.dt, it_wn, it_wn * wn.dt,
                             it_wn * wn.dt - it_cg * cg.dt])

    made = 0
    if pairs:
        made += compare_snapshots(runs, pairs, out_dir, slice_idx,
                                  args.nPML, summary)

    positions = station_positions(params)
    if positions and not args.skip_stations:
        log("")
        log("[waveforms] reading station files...")
        rows = compare_waveforms(runs, positions, args, out_dir, data_dir,
                                 summary)
        if rows:
            made += 1 + len(set(r["station"] for r in rows))

    if not args.skip_pgv:
        log("")
        log("[PGV] comparing the PGV maps...")
        made += compare_pgv(runs, args.nPML, out_dir, data_dir, summary,
                            slice_idx, params)

    if not args.skip_intensity:
        log("")
        log("[intensity] comparing snapshot-based PGV/PGA and intensity...")
        made += compare_snapshot_intensity(runs, args, out_dir, slice_idx,
                                           params, args.nPML, summary)

    if not args.skip_model:
        log("")
        log("[model] checking the medium and coordinates...")
        made += compare_model(runs, out_dir, slice_idx, params, summary)

    # summary file
    issues = read_issue_lines()
    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("CGFDM vs WENO comparison\n")
        f.write("generated by plotContrast.py\n\n")
        f.write("files found in the run dirs\n")
        for line in inv_lines:
            f.write(line + "\n")
        f.write("\n")
        for line in summary:
            f.write(line + "\n")
            if line:
                log("  " + line)
        f.write("\n")
        if issues:
            f.write("file read problems\n")
            for line in issues:
                f.write(line + "\n")
        else:
            f.write("file read problems: none\n")

    if issues:
        log("")
        log("[files] fields whose rank files had an unexpected size:")
        for line in issues:
            warn(line)
    else:
        log("[files] every field tile had the expected size")

    log("")
    log("done: figures written to %s" % out_dir)
    log("      summary: %s" % summary_path)


if __name__ == "__main__":
    main()
