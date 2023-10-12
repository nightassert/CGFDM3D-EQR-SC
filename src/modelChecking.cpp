/***********************************************************
  File Name: modelChecking.cpp
  Author: Wenqiang Wang
  Mail: 11849528@mail.sustech.edu.cn
  Created Time: Fri 04 Nov 2022 12:20:38 AM CST
 *******************************************************/

#include "header.h"

void modelChecking(PARAMS params)
{
	int thisRank;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Comm_rank(MPI_COMM_WORLD, &thisRank);

	if (params.useMultiSource && params.useSingleSource)
	{
		params.useMultiSource = 0;
		if (0 == thisRank)
			printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				   "You set \"useMultiSource\" and \"useSingleSource\" at the same time. We set \"useSingleSource:\" = 0.\n"
				   "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
	}
	if (!(params.useMultiSource || params.useSingleSource))
	{

		if (0 == thisRank)
			printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				   "You did not set any source. The program will abort!\n"
				   "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Abort(MPI_COMM_WORLD, 180);
	}
	if (params.ShenModel && params.Crust_1Medel)
	{

		if (0 == thisRank)
			printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				   "You set ShenModel and Crust_1Medel both to be 1. We will use homogenourse model!\n"
				   "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

		params.ShenModel = 0;
		params.Crust_1Medel = 0;
		MPI_Barrier(MPI_COMM_WORLD);
	}

	if ((params.ShenModel || params.Crust_1Medel) && params.LayeredModel)
	{

		if (0 == thisRank)
			printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
				   "You set ShenModel, Crust_1Medel and LayeredModel both to be 1. We will use Layed model!\n"
				   "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

		params.ShenModel = 0;
		params.Crust_1Medel = 0;
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
	if (params.Crust_1Medel == 0)
	{

		if (0 == thisRank)
			printf("No Crust_1Medel is used!\n");
	}
	if (params.Crust_1Medel == 1)
	{

		if (0 == thisRank)
			printf("Crust_1Medel is used!\n");
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
