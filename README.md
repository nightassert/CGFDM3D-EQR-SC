# CGFDM3D-EQR-SC
A platform for rapid response to earthquake disasters in 3D complex media. \
The program is developed based on CGFDM3D-EQR, adding shock-capturing AWENO scheme. It provides 2 version for seismic wave propagation simulation:

- CGFDM3D-EQR: Curve-Grid Finite-Difference Method (CGFDM)
- CGFDM3D-EQR-SC: Mixed Efficient Alternative Flux Finite-Difference WENO Scheme (ME-AWENO)

You can choose scheme by using different Makefile:
- Makefile_CGFDM
- Makefile_SCFDM
> A convient script make.sh is provided to compile the program. If you use CGFDM, run `bash make.sh CGFDM`, and if you use ME-AWENO, run `bash make.sh SCFDM`.

The folder src_GJI_XU2024 contains the source codes of [Xu & Zhang (2024)](https://doi.org/10.1093/gji/ggae167).
And the folder src contains the newest optimization codes, which has been submitted to JGR: Solid Earth.

## Authors
**Tianhong Xu, Wenqiang Wang, Zhenguo Zhang*** \
Southern University of Science and Technology \
Shenzhen, China

## Citation
1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. 
2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. 
