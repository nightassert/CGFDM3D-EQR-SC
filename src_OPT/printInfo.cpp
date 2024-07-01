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
		"DH = %5.2e\n"
#ifdef SCFDM
		"You are using SCFDM by Tianhong Xu\n"
#else
		"You are using CGFDM by Wenqiang Wang\n"
#endif

#ifdef EXP_DECAY
		"EXP_DECAY is defined\n"
#endif
		"=============================================\n",
		grid.PX, grid.PY, grid.PZ,
		grid.NX, grid.NY, grid.NZ,
		NT,
		grid.DH);
}
