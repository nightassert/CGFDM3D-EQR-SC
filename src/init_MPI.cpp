/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: init_MPI.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2021-11-03
*   Discription: Initialize MPI
*
*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*
=================================================================*/
#include "header.h"

void finalize_MPI(MPI_Comm *comm_cart)
{
	MPI_Comm_free(comm_cart);
	MPI_Finalize();
}

void init_MPI(int *argc, char ***argv, PARAMS params, MPI_Comm *comm_cart,
			  MPI_COORD *thisMPICoord, MPI_NEIGHBOR *mpiNeighbor)
{
	int PX = params.PX;
	int PY = params.PY;
	int PZ = params.PZ;

	int thisRank, thisMPICoordXYZ[3];

	int nDim = 3;
	int mpiDims[3] = {PX, PY, PZ};
	int periods[3] = {0, 0, 0};
	int reorder = 0;

	MPI_Init(argc, argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &thisRank);

	// The Cartesian topology needs exactly PX * PY * PZ MPI ranks.  A common
	// single- or dual-GPU mistake is to launch fewer ranks than params.json
	// declares; fail with a clear message instead of an MPI error.
	int worldSize = 0;
	MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
	if (PX * PY * PZ != worldSize)
	{
		if (0 == thisRank)
			printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				   "params.json declares PX * PY * PZ = %d * %d * %d = %d MPI ranks, but\n"
				   "mpirun started %d.  Make PX, PY, PZ match the number of MPI ranks\n"
				   "(e.g. PX = 1, PY = 1, PZ = 1 for one GPU).  The program will abort!\n"
				   "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",
				   PX, PY, PZ, PX * PY * PZ, worldSize);
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Abort(MPI_COMM_WORLD, 110);
	}

	MPI_Cart_create(MPI_COMM_WORLD, nDim, mpiDims, periods, reorder, comm_cart);

	MPI_Cart_shift(*comm_cart, 0, 1, &(mpiNeighbor->X1), &(mpiNeighbor->X2));
	MPI_Cart_shift(*comm_cart, 1, 1, &(mpiNeighbor->Y1), &(mpiNeighbor->Y2));
	MPI_Cart_shift(*comm_cart, 2, 1, &(mpiNeighbor->Z1), &(mpiNeighbor->Z2));


	MPI_Cart_coords(*comm_cart, thisRank, 3, thisMPICoordXYZ);

	thisMPICoord->X = thisMPICoordXYZ[0];
	thisMPICoord->Y = thisMPICoordXYZ[1];
	thisMPICoord->Z = thisMPICoordXYZ[2];
}
