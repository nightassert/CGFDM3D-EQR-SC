/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: main.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2021-11-01
*   Discription: Main function
*
*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*
=================================================================*/

#include "header.h"

int main(int argc, char **argv)
{

	PARAMS params;
	GRID grid = {0};

	MPI_COORD thisMPICoord = {0};
	MPI_NEIGHBOR mpiNeighbor = {0};
	MPI_Comm comm_cart;

	getParams(&params);
	init_MPI(&argc, &argv, params, &comm_cart, &thisMPICoord, &mpiNeighbor);
	init_grid(params, &grid, thisMPICoord);

	createDir(params);

#ifdef GPU_CUDA
	init_gpu(grid.PX, grid.PY, grid.PZ);
#endif

	MPI_Barrier(comm_cart);

	run(comm_cart, thisMPICoord, mpiNeighbor, grid, params);

	MPI_Barrier(comm_cart);
	MPI_Comm_free(&comm_cart);
	MPI_Finalize();
	return 0;
}
