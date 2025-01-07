/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: modelChecking.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2022-11-04
*   Discription: Check if the model is set correctly
*
*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*
=================================================================*/

#include "header.h"

void modelChecking(PARAMS params)
{
	int thisRank;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Comm_rank(MPI_COMM_WORLD, &thisRank);

	if (params.useSingleSource_ricker && params.useSingleSource_double_couple)
	{
		if (0 == thisRank)
			printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				   "You set \"useSingleSource(ricker)\" and \"useSingleSource(double_couple)\" at the same time. The program will abort!\n"
				   "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Abort(MPI_COMM_WORLD, 180);
	}

	if (params.useMultiSource && (params.useSingleSource_ricker || params.useSingleSource_double_couple))
	{
		params.useMultiSource = 0;
		if (0 == thisRank)
			printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				   "You set \"useMultiSource\" and \"useSingleSource\" at the same time. We set \"useSingleSource:\" = 0.\n"
				   "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
	}
	if (!(params.useMultiSource || params.useSingleSource_ricker || params.useSingleSource_double_couple))
	{

		if (0 == thisRank)
			printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				   "You did not set any source. The program will abort!\n"
				   "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Abort(MPI_COMM_WORLD, 180);
	}
	if (params.ShenModel && params.Crust_1Model)
	{

		if (0 == thisRank)
			printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				   "You set ShenModel and Crust_1Model both to be 1. We will use homogenourse model!\n"
				   "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

		params.ShenModel = 0;
		params.Crust_1Model = 0;
		MPI_Barrier(MPI_COMM_WORLD);
	}

	if ((params.ShenModel || params.Crust_1Model) && params.LayeredModel)
	{

		if (0 == thisRank)
			printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				   "You set ShenModel, Crust_1Model and LayeredModel both to be 1. We will use Layed model!\n"
				   "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

		params.ShenModel = 0;
		params.Crust_1Model = 0;
		params.LayeredModel = 1;
		MPI_Barrier(MPI_COMM_WORLD);
	}

	if (params.useTerrain && params.gauss_hill)
	{
		printf(
			"!!!!!!!!!!!!!!!!!!!!!!!WORNING!!!!!!!!!!!!!!!!!!!!!!!"
			"You set \"useTerrain\" and \"gauss_hill\" can not be 1 at the same time. You should configue your \"./paramsDir/paramsCGFDM3D-CJMVS.json\" file. Or, we will use \"gauss_hill\" model to run the simulation\n"
			"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
		params.gauss_hill = 1;
		params.useTerrain = 0;
		params.Depth = params.DH * params.NZ * 1e-3;
	}

	if (params.useTerrain == 0)
	{

		if (0 == thisRank)
			printf("No SRTM90 Terrain model is used!\n");
	}
	if (params.useTerrain == 1)
	{

		if (0 == thisRank)
			printf("SRTM90 Terrain model is used!\n");
	}

	if (params.useMultiSource == 0)
	{

		if (0 == thisRank)
			printf("No Multi-source model is used!\n");
	}
	if (params.useMultiSource == 1)
	{
		if (0 == thisRank)
			printf("Multi-source model is used!\n");
	}

	if (params.ShenModel == 0)
	{

		if (0 == thisRank)
			printf("No ShenModel is used!\n");
	}
	if (params.ShenModel == 1)
	{

		if (0 == thisRank)
			printf("ShenModel is used!\n");
	}
	if (params.Crust_1Model == 0)
	{

		if (0 == thisRank)
			printf("No Crust_1Model is used!\n");
	}
	if (params.Crust_1Model == 1)
	{

		if (0 == thisRank)
			printf("Crust_1Model is used!\n");
	}

	if (params.LayeredModel == 0)
	{

		if (0 == thisRank)
			printf("No Layered Model is used!\n");
	}
	if (params.LayeredModel == 1)
	{

		if (0 == thisRank)
			printf("Layered Model is used!\n");
	}

	MPI_Barrier(MPI_COMM_WORLD);
}
