/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: run.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2021-11-03
*   Discription: Run the simulation
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
PARAMS g_params;
MPI_COORD g_thisMPICoord;
float g_vp_max = 0.0f;

void run(MPI_Comm comm_cart, MPI_COORD thisMPICoord, MPI_NEIGHBOR mpiNeighbor, GRID grid, PARAMS params)
{
	int thisRank;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Comm_rank(MPI_COMM_WORLD, &thisRank);
	if (thisRank == 0)
		printInfo(grid, params);
	MPI_Barrier(MPI_COMM_WORLD);

	modelChecking(params);

	SLICE_DATA sliceData, sliceDataCpu;
	SLICE slice = {0};
	locateSlice(params, grid, &slice);
	allocSliceData(grid, slice, &sliceData, &sliceDataCpu);

	g_params = params;
	g_thisMPICoord = thisMPICoord;

	// construct coordinate
	if (thisRank == 0)
		printf("Construct coordinate including processing terrian data...\n");
	MPI_Barrier(MPI_COMM_WORLD);

	float *coord, *cpu_coord;
	float *terrain, *cpu_terrain;
	allocTerrain(grid, &terrain, &cpu_terrain);
	allocCoord(grid, &coord, &cpu_coord);
	constructCoord(comm_cart, thisMPICoord, grid, params, coord, cpu_coord, terrain, cpu_terrain);

	// construct medium
	if (thisRank == 0)
		printf("Construct medium including processing Vs Vp Rho...\n");
	MPI_Barrier(MPI_COMM_WORLD);

	float *medium, *cpu_medium;
	allocMedium(grid, &medium, &cpu_medium);
	constructMedium(thisMPICoord, params, grid, cpu_coord, cpu_terrain, medium, cpu_medium);

	if (thisRank == 0)
		printf("Slice Position Coordinate(x, y, z) and Medium(Vp, Vs, Rho) data output...\n");
	MPI_Barrier(MPI_COMM_WORLD);
	data2D_Model_out(thisMPICoord, params, grid, coord, medium, slice, sliceData, sliceDataCpu);

	// calculate CFL condition
	calc_CFL(grid, cpu_coord, cpu_medium, params);

	FLOAT *CJM; // Contravariant Jacbian Medium
	allocCJM(grid, &CJM);

	Mat_rDZ mat_rDZ = {0};
#ifdef FREE_SURFACE
	// Mat_rDZ is only used by the CGFDM free-surface derivative.
	if (params.scheme == SCHEME_CGFDM)
		allocMat_rDZ(grid, &mat_rDZ);
#endif

	if (thisRank == 0)
		printf("Geometrical Information Calculating: Contravariant and Jacobian...\n");
	MPI_Barrier(MPI_COMM_WORLD);
	constructCJM(comm_cart, mpiNeighbor, grid, CJM, coord, medium, mat_rDZ);
	freeMedium(medium, cpu_medium);
	MPI_Barrier(MPI_COMM_WORLD);

	// multi source
	long long *srcIndex = NULL;
	float *momentRate = NULL, *momentRateSlice = NULL;
	SOURCE_FILE_INPUT src_in = {0};

	if (params.useMultiSource)
	{
		init_MultiSource(params, grid, thisMPICoord, cpu_coord, cpu_terrain, &srcIndex, &momentRate, &momentRateSlice, &src_in);
	}
	freeCoord(coord, cpu_coord);
	freeTerrain(terrain, cpu_terrain);

	MPI_Barrier(MPI_COMM_WORLD);

	WAVE wave;
	allocWave(grid, &wave);

	if (thisRank == 0)
		printf("Start calculating Wave Field:\n");
	MPI_Barrier(MPI_COMM_WORLD);

	propagate(comm_cart, thisMPICoord, mpiNeighbor,
			  grid, params,
			  wave, mat_rDZ, CJM,
			  src_in, srcIndex, momentRate, momentRateSlice,
			  slice, sliceData, sliceDataCpu);

	MPI_Barrier(MPI_COMM_WORLD);

	freeWave(wave);
#ifdef FREE_SURFACE
	if (params.scheme == SCHEME_CGFDM)
		freeMat_rDZ(mat_rDZ);
#endif
	freeCJM(CJM);
	freeSliceData(grid, slice, sliceData, sliceDataCpu);
}
