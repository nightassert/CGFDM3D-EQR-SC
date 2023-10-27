/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name:propagate.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time:2021-11-03
*   Discription:
*
================================================================*/

#include "header.h"

void isMPIBorder(GRID grid, MPI_COORD thisMPICoord, MPI_BORDER *border)
{

	if (0 == thisMPICoord.X)
		border->isx1 = 1;
	if ((grid.PX - 1) == thisMPICoord.X)
		border->isx2 = 1;
	if (0 == thisMPICoord.Y)
		border->isy1 = 1;
	if ((grid.PY - 1) == thisMPICoord.Y)
		border->isy2 = 1;
	if (0 == thisMPICoord.Z)
		border->isz1 = 1;
	if ((grid.PZ - 1) == thisMPICoord.Z)
		border->isz2 = 1;
}

void propagate(
	MPI_Comm comm_cart, MPI_COORD thisMPICoord, MPI_NEIGHBOR mpiNeighbor,
	GRID grid, PARAMS params,
	WAVE wave, Mat_rDZ mat_rDZ, FLOAT *CJM,
	SOURCE_FILE_INPUT src_in, long long *srcIndex, float *momentRate,
	float *momentRateSlice, SLICE slice, SLICE_DATA sliceData, SLICE_DATA sliceDataCpu)
{
	int thisRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &thisRank);

	float DT = params.DT;
	int NT = params.TMAX / DT;
	float DH = grid.DH;

	int IT_SKIP = params.IT_SKIP;

	int sliceFreeSurf = params.sliceFreeSurf;

	SLICE freeSurfSlice;
	float *pgv, *cpuPgv;

	locateFreeSurfSlice(grid, &freeSurfSlice);
	SLICE_DATA freeSurfData, freeSurfDataCpu;

	int IsFreeSurface = 0;
#ifdef FREE_SURFACE
	if (thisMPICoord.Z == grid.PZ - 1)
		IsFreeSurface = 1;
#endif
	if (IsFreeSurface)
	{
		if (sliceFreeSurf)
			allocSliceData(grid, freeSurfSlice, &freeSurfData, &freeSurfDataCpu);

		allocatePGV(grid, &pgv, &cpuPgv);
	}

	MPI_BORDER border = {0};
	isMPIBorder(grid, thisMPICoord, &border);

	AUX6SURF Aux6;
#ifdef PML
	allocPML(grid, &Aux6, border);

	PML_ALPHA pml_alpha;
	PML_BETA pml_beta;
	PML_D pml_d;

	allocPMLParameter(grid, &pml_alpha, &pml_beta, &pml_d);
	init_pml_parameter(params, grid, border, pml_alpha, pml_beta, pml_d);
#endif

	int stationNum;
	STATION station, cpu_station;
	stationNum = readStationIndex(grid);

	if (stationNum > 0)
	{
		allocStation(&station, &cpu_station, stationNum, NT);
		initStationIndex(grid, cpu_station);
		stationCPU2GPU(station, cpu_station, stationNum);
	}

	SOURCE S = {0}; //{ _nx_ / 2, _ny_ / 2, _nz_ / 2 };
	locateSource(params, grid, &S);

	int useMultiSource = params.useMultiSource;
	int useSingleSource = params.useSingleSource;

	int it = 0, irk = 0;
	int FB1 = 0;
	int FB2 = 0;
	int FB3 = 0;

	int FB[8][3] =
		{
			{-1, -1, -1},
			{1, 1, -1},
			{1, 1, 1},
			{-1, -1, 1},
			{-1, 1, -1},
			{1, -1, -1},
			{1, -1, 1},
			{-1, 1, 1},
		}; // F = 1, B = -1

	// GaussField( grid, wave.W );
	// useSingleSource = 0;

	float *gaussFactor;
	int nGauss = 3;
	if (useMultiSource)
	{
		allocGaussFactor(&gaussFactor, nGauss);
		calculateMomentRate(src_in, CJM, momentRate, srcIndex, DH);
	}

	MPI_Barrier(comm_cart);
	long long midClock = clock(), stepClock = 0;

	SEND_RECV_DATA_FLOAT sr_wave;
	FLOAT_allocSendRecv(grid, mpiNeighbor, &sr_wave, WSIZE);
	for (it = 0; it < NT; it++)
	{
		FB1 = FB[it % 8][0];
		FB2 = FB[it % 8][1];
		FB3 = FB[it % 8][2];
		if (useSingleSource)
			loadPointSource(grid, S, wave.W, CJM, it, 0, DT, DH, params.rickerfc);

		if (useMultiSource)
			addMomenteRate(grid, src_in, wave.W, srcIndex, momentRate, momentRateSlice, it, 0, DT, DH, gaussFactor, nGauss, IsFreeSurface);

		for (irk = 0; irk < 4; irk++)
		{
			MPI_Barrier(comm_cart);
			FLOAT_mpiSendRecv(comm_cart, mpiNeighbor, grid, wave.W, sr_wave, WSIZE);
#ifdef PML
#ifdef SCFDM
			// ! For alternative flux finite difference by Tianhong Xu
			waveDeriv_alternative_flux_FD(grid, wave, CJM, pml_beta, FB1, FB2, FB3, DT, thisMPICoord, params);
#else
			waveDeriv(grid, wave, CJM, pml_beta, FB1, FB2, FB3, DT);
#endif // SCFDM
			if (IsFreeSurface)
				freeSurfaceDeriv(grid, wave, CJM, mat_rDZ, pml_beta, FB1, FB2, FB3, DT);
			pmlDeriv(grid, wave, CJM, Aux6, pml_alpha, pml_beta, pml_d, border, FB1, FB2, FB3, DT);
			if (IsFreeSurface)
				pmlFreeSurfaceDeriv(grid, wave, CJM, Aux6, mat_rDZ, pml_d, border, FB1, FB2, DT);
#else // PML
#ifdef SCFDM
			// ! For alternative flux finite difference by Tianhong Xu
			waveDeriv_alternative_flux_FD(grid, wave, CJM, FB1, FB2, FB3, DT, thisMPICoord, params);
#else
			waveDeriv(grid, wave, CJM, FB1, FB2, FB3, DT);
			if (IsFreeSurface)
				freeSurfaceDeriv(grid, wave, CJM, mat_rDZ, FB1, FB2, FB3, DT);
#endif // SCFDM

#endif // PML
			waveRk(grid, irk, wave);
#ifdef PML
			pmlRk(grid, border, irk, Aux6);
#endif
			FB1 *= -1;
			FB2 *= -1;
			FB3 *= -1; // reverse
		}			   // for loop of irk: Range Kutta Four Step
#ifdef SCFDM
// ! For alternative flux finite difference by Tianhong Xu
#ifdef FREE_SURFACE
		if (IsFreeSurface)
			charfreeSurfaceDeriv(grid, wave, CJM, mat_rDZ, FB1, FB2, FB3, DT);
#endif // FREE_SURFACE
#endif // SCFDM

#ifdef EXP_DECAY
		expDecayLayers(grid, wave);
#endif // EXP_DECAY

		if (stationNum > 0)
			storageStation(grid, NT, stationNum, station, wave.W, it
#ifdef SCFDM
						   ,
						   CJM
#endif
			);
		if (IsFreeSurface)
			comparePGV(grid, thisMPICoord, wave.W, pgv
#ifdef SCFDM
					   ,
					   CJM
#endif
			);

		if (it % IT_SKIP == 0)
		{
			data2D_XYZ_out(thisMPICoord, params, grid, wave.W, slice, sliceData, sliceDataCpu, 'V', it
#ifdef SCFDM
						   ,
						   CJM
#endif
			);
			if (IsFreeSurface && sliceFreeSurf)
				data2D_XYZ_out(thisMPICoord, params, grid, wave.W, freeSurfSlice, freeSurfData, freeSurfDataCpu, 'F', it
#ifdef SCFDM
							   ,
							   CJM
#endif
				);
		}
		/*
		 */
		MPI_Barrier(comm_cart);
		if ((0 == thisRank) && (it % 10 == 0))
		{
			printf("it = %8d. ", it);
			stepClock = clock() - midClock;
			midClock = stepClock + midClock;
			printf("Step time loss: %8.3lfs. Total time loss: %8.3lfs.\n", stepClock * 1.0 / (CLOCKS_PER_SEC * 1.0), midClock * 1.0 / (CLOCKS_PER_SEC * 1.0));
		}

	} // for loop of it: The time iterator of NT steps
	if (useMultiSource)
		freeGaussFactor(gaussFactor);

#ifdef PML
	freePML(border, Aux6);
	freePMLParamter(pml_alpha, pml_beta, pml_d);
#endif

	FLOAT_freeSendRecv(mpiNeighbor, sr_wave);
	MPI_Barrier(comm_cart);
	if (stationNum > 0)
	{
		stationGPU2CPU(station, cpu_station, stationNum, NT);
		stationWrite(params, grid, thisMPICoord, cpu_station, NT, stationNum);
		freeStation(station, cpu_station);
	}

	if (IsFreeSurface)
	{
		outputPGV(params, grid, thisMPICoord, pgv, cpuPgv);
		freePGV(pgv, cpuPgv);
		if (sliceFreeSurf)
			freeSliceData(grid, freeSurfSlice, freeSurfData, freeSurfDataCpu);
	}
}
