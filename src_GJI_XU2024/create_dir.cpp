/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: create_dir.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2021-11-30
*   Discription: Solve the contravariant and Jacobian

*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*
=================================================================*/

#include "header.h"

void createDir(PARAMS params)
{
	int thisRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &thisRank);

	if (0 == thisRank)
	{
#if __GNUC__
		mkdir(params.OUT, 0777);
#elif _MSC_VER
		_mkdir(params.OUT);
#endif
	}

	MPI_Barrier(MPI_COMM_WORLD);
}
