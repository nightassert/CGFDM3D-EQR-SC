/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: printInfo.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2021-11-03
*   Discription: Print the information of the simulation
*
*	Update: Tianhong Xu, 12231218@mail.sustech.edu.cn
*   Update Time: 2023-11-16
*   Update Content: Add SCFDM
*
*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*
=================================================================*/

#include "header.h"
void printInfo(GRID grid, PARAMS params)
{
	float DT = params.DT;
	int NT = params.TMAX / DT;

	printf(
		"=============================================\n"
		"MPI:  PX = %5d, PY = %5d, PZ = %5d\n"
		"GRID: NX = %5d, NY = %5d, NZ = %5d\n"
		"NT: %5d\n"
		"DH = %5.2e\n",
		grid.PX, grid.PY, grid.PZ,
		grid.NX, grid.NY, grid.NZ,
		NT,
		grid.DH);

	if (params.scheme == SCHEME_WENO)
	{
		// scheme == SCHEME_WENO covers both reconstructions; the "scheme" key
		// in params.json picks which one.
		if (params.wenoReconstruction == WENO_RECONSTRUCTION_Z)
			printf("Scheme: WENO-Z alternative-flux finite difference (Tianhong Xu)\n");
		else
			printf("Scheme: WENO-JS alternative-flux finite difference (Tianhong Xu)\n");
		// wenoLocalAlpha selects the LF dissipation speed: 0 = one global
		// Vp_max (historical), 1 = max(Vp) of the two cells at each interface.
		if (params.wenoLocalAlpha)
			printf("Riemann flux: local max(Vp) per interface (wenoLocalAlpha = 1)\n");
		else
			printf("Riemann flux: global Vp_max (wenoLocalAlpha = 0)\n");
		// The WENO branch uses the C-PML in weno_pml.cpp whenever nPML > 0;
		// the exponential-decay layers are only the fallback for nPML == 0.
#ifdef PML
		if (params.nPML > 0)
			printf("Boundary: C-PML absorbing layers (nPML = %d)\n", params.nPML);
		else
			printf("Boundary: exponential-decay absorbing layers (nPML = 0)\n");
#else
		printf("Boundary: exponential-decay absorbing layers\n");
#endif
	}
	else
	{
		printf("Scheme: CGFDM curvilinear-grid finite difference (Wenqiang Wang)\n");
#ifdef PML
		printf("Boundary: perfectly matched layer (PML)\n");
#endif
	}

	printf("=============================================\n");
}
