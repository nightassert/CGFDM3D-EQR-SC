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

// #include "/public/software/mpich/include/mpi.h"

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
	// MPI_Comm_size( MPI_COMM_WORLD, &nProcs );

	MPI_Cart_create(MPI_COMM_WORLD, nDim, mpiDims, periods, reorder, comm_cart);

	MPI_Cart_shift(*comm_cart, 0, 1, &(mpiNeighbor->X1), &(mpiNeighbor->X2));
	MPI_Cart_shift(*comm_cart, 1, 1, &(mpiNeighbor->Y1), &(mpiNeighbor->Y2));
	MPI_Cart_shift(*comm_cart, 2, 1, &(mpiNeighbor->Z1), &(mpiNeighbor->Z2));

	// if ( mpiNeighbor->X1 < 0 ) mpiNeighbor->X1 = MPI_PROC_NULL;
	// if ( mpiNeighbor->Y1 < 0 ) mpiNeighbor->Y1 = MPI_PROC_NULL;
	// if ( mpiNeighbor->Z1 < 0 ) mpiNeighbor->Z1 = MPI_PROC_NULL;
	// if ( mpiNeighbor->X2 < 0 ) mpiNeighbor->X1 = MPI_PROC_NULL;
	// if ( mpiNeighbor->Y2 < 0 ) mpiNeighbor->Y1 = MPI_PROC_NULL;
	// if ( mpiNeighbor->Z2 < 0 ) mpiNeighbor->Z1 = MPI_PROC_NULL;

	MPI_Cart_coords(*comm_cart, thisRank, 3, thisMPICoordXYZ);

	thisMPICoord->X = thisMPICoordXYZ[0];
	thisMPICoord->Y = thisMPICoordXYZ[1];
	thisMPICoord->Z = thisMPICoordXYZ[2];
}
