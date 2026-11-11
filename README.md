# CGFDM3D-EQR-SC

A platform for rapid response to earthquake disasters in 3D complex media.
The program is developed from CGFDM3D-EQR and adds an alternative-flux
finite-difference WENO scheme. Both numerical schemes are compiled into a
single executable and selected at run time:

- `"CGFDM"` — curvilinear-grid finite-difference method (default)
- `"WENO"` — alternative-flux finite-difference WENO5 scheme
  (`"SCFDM"` is accepted as an alias)

The WENO path uses TVD-RK3 time integration, a C-PML absorbing boundary
condition and a characteristic free-surface condition. The CGFDM path uses
classical RK4 and the velocity-stress C-PML. Both schemes share the same
PML damping profiles (`init_pml_parameter`); the WENO C-PML keeps the split
auxiliary fields in conservative variables and integrates them with the
same TVD-RK3 as the main field (`weno_pml.cpp`). Source, station, PGV,
slice and displacement output work with either scheme.

## Requirements

GPU build:

- CUDA Toolkit 13.x, driver 580 or newer
- An MPI implementation built with CUDA support (CUDA-aware MPI)
- PROJ 8 or newer
- NetCDF (optional, only for the 3D velocity volume output)
- A GPU with compute capability 7.5 or newer
  (A100 = sm_80, RTX 3060 = sm_86, RTX 40-series = sm_89, H100 = sm_90)

CUDA-aware MPI is required because halo exchange (`MPI_Sendrecv`) is performed
directly on device buffers.

CPU-cluster build: only a C++ compiler, MPI and PROJ are needed.  Both
schemes, including the WENO kernels and its C-PML, have complete CPU code
paths and run correctly without any GPU.

## Build

```bash
# A100
make CUDA_ARCH=sm_80

# RTX 3060 / other RTX 30-series cards
make CUDA_ARCH=sm_86

# CPU-only build for MPI clusters without GPUs
make GPU_CUDA=OFF
```

If the toolkit or MPI is not in the default location:

```bash
make CUDA_ARCH=sm_86 \
     CUDA_HOME=/usr/local/cuda-13.0 \
     MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi \
     PROJ_HOME=/usr
```

`make.sh` is a small wrapper:

```bash
./make.sh sm_86
```

The executable is written to `bin/main`; object files go to `obj/`.
Run `make clean` before switching between the GPU and the CPU build.

The optional 3D velocity volume output uses NetCDF. The default build links
`-lnetcdf` (`NETCDF=ON`); set `NETCDF_HOME` when NetCDF is not installed in
`/usr`, or disable the feature explicitly:

```bash
make CUDA_ARCH=sm_86 NETCDF_HOME=/opt/netcdf
make CUDA_ARCH=sm_86 NETCDF=OFF
```

With `NETCDF=OFF` the run is unchanged and `volume.json` is ignored.

## Run

```bash
cd examples/Example_finite_fault
mpirun -np 8 /path/to/CGFDM3D-EQR-SC/bin/main
```

Set the scheme in `params.json`:

```json
{
    "scheme": "WENO-Z",
    ...
}
```

`"CGFDM"` selects the curvilinear-grid scheme. `"WENO-JS"` and `"WENO-Z"`
both run the alternative-flux WENO scheme and differ only in the interface
reconstruction: JS uses the historical Jiang & Shu (1996) nonlinear weights,
Z uses the Borges et al. (2008) weights, which keep the formal order at
smooth extrema and are less dissipative on wave peaks. `"WENO"` and
`"SCFDM"` are aliases of `"WENO-JS"`. Omit the key for CGFDM.

`nPML` selects the thickness of the PML layer for both schemes (default 10).
For WENO, set `"nPML": 0` to fall back to the legacy 20-layer
exponential-decay absorbing layer. Note that with WENO + PML, waves leaving
the top of the model are handled by the characteristic free-surface
condition, so there is no PML on the top face (as in CGFDM).

`wenoLocalAlpha` controls the dissipation speed of the WENO Riemann flux
(WENO only, default 0). `0` uses one global `Vp_max` for every interface;
`1` uses the larger `Vp` of the two cells sharing each interface (never above
the global value). The local choice lowers the artificial dissipation in slow
shallow layers while using no extra memory; the time step is still bounded by
the global `Vp_max`.

A single-GPU workstation does not need a `gpu_nodes` block in `params.json`;
each MPI rank selects a device round-robin. On a cluster, `gpu_nodes` maps
host names and MPI ranks to devices as before.

`PX * PY * PZ` must equal the number of MPI ranks started with `mpirun`
(e.g. `"PX": 1, "PY": 1, "PZ": 1` for one GPU, `"PX": 2, "PY": 1, "PZ": 1`
for two). The program checks this at startup and prints a clear error
instead of failing inside MPI when the numbers do not match.

## 3D velocity volume output

An arbitrary subvolume of the velocity field can be written during a run. It is
disabled by default: create `volume.json` in the run directory to enable it.
Example:

```json
{
    "enabled": 1,
    "it_skip": 20,
    "xStart": 250,
    "xEnd": 550,
    "yStart": 180,
    "yEnd": 420,
    "zStart": 340,
    "zEnd": 400
}
```

The block is given in global grid indices, inclusive, with the same convention
as `station.json` (`K = NZ-1` is the free surface). Missing range keys default
to the whole grid, `it_skip` defaults to 1 and `enabled` defaults to 1. At
startup the program prints the block size, the number of output frames and the
total data volume, so the block or `it_skip` can be adjusted before a long run.

Every rank writes its local part of the block to
`output/volume_mpi_X_Y_Z.nc` (NetCDF-4). Each file has the dimensions
`(time, z, y, x)`, the float32 variables `vx`, `vy`, `vz` in m/s, the integer
grid indices `I`, `J`, `K` and the `time` axis. The CGFDM path stores velocity
directly; the WENO path stores conservative momentum and is converted with
`v = W * buoyancy`, the same conversion used by the station and slice output.

The per-rank files can be merged into a single file with
`examples/Example_finite_fault/mergeVolumeNC.py` (requires `numpy` and
`netCDF4`):

```bash
python3 mergeVolumeNC.py output -o volume.nc
python3 mergeVolumeNC.py output -o volume.nc --deflate 1
```

The script verifies that the sub-blocks tile the requested block without gaps
or overlaps, then writes the full `(time, z, y, x)` volume together with the
`I`, `J`, `K` coordinates and the original global attributes. Remove stale
`output/volume_mpi_*.nc` files from earlier runs before merging.

## Notes on memory

Both schemes use the same unified memory layout. `WAVE` needs four
`WSIZE`-float buffers per grid point and `CJM` needs 19 floats per point,
roughly 220 bytes per grid point (256 bytes with `SOLVE_PGA`). The WENO
output conversion reuses `WAVE.t_W`, so no additional full-grid buffer is
allocated at output time.

The full 801 x 601 x 401 example is about 193 million points, close to 50 GB
of device memory; it does not fit on a single 12 GB card. Use MPI to
distribute it, or reduce `NX`, `NY`, `NZ` for local testing.

## Authors

**Tianhong Xu, Wenqiang Wang, Zhenguo Zhang\***
Southern University of Science and Technology
Shenzhen, China

## Citation

1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D-EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334.
2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512.
3. Xu, T., Yu, H., Wang, W., Wang, Z., & Zhang, Z. (2026). A Mixed Efficient AWENO Scheme to Model Cross‐Fault Ground Motions and Its Applications. Seismological Research Letters, 97(1), 161-172.