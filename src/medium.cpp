/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name:medium.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time:2021-10-31
*   Modified Time:2022-11-04
*   Discription:
*
================================================================*/
#include "header.h"

void allocMedium(GRID grid, float **medium, float **cpu_medium)
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	long long num = _nx_ * _ny_ * _nz_;

	long long size = sizeof(float) * num * MSIZE;

	CHECK(Malloc((void **)medium, size));
	if (*medium == NULL)
	{
		printf("can't allocate Medium memory!\n");
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}
	CHECK(Memset(*medium, 0, size));

#ifdef GPU_CUDA
	*cpu_medium = (float *)malloc(size);

	if (*cpu_medium == NULL)
	{
		printf("can't allocate Medium memory!\n");
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}

	memset(*cpu_medium, 0, size);
#else
	*cpu_medium = *medium;
#endif
}

void freeMedium(float *medium, float *cpu_medium)
{

	Free(medium);
#ifdef GPU_CUDA
	free(cpu_medium);
#endif
}

// homogeneous medium
__GLOBAL__
void construct_homo_medium(float *medium, int _nx_, int _ny_, int _nz_)
{
	long long index = 0;
#ifdef GPU_CUDA
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
#else
	int i = 0;
	int j = 0;
	int k = 0;
#endif

	CALCULATE3D(i, j, k, 0, _nx_, 0, _ny_, 0, _nz_)
	index = INDEX(i, j, k) * MSIZE;
	medium[index + 0] = 3464.0f;
	medium[index + 1] = 6000.0f;
	medium[index + 2] = 2670.0f;
	END_CALCULATE3D()
}

// 1D Model
void generate_1D_medium(PARAMS params, GRID grid, float *cpu_medium, float *cpu_coord, float *cpu_terrain)
{
	long long index = 0, indexC = 0, pos;
	int i = 0;
	int j = 0;
	int k = 0;

	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	float coordz;

	FILE *file = fopen(params.LayeredFileName, "r");
	if (NULL == file)
	{
		printf("Can't open Layered Model file %s \n", params.LayeredFileName);
	}
	int F = 0;
	int layers = 0;
	fread(&(layers), sizeof(int), 1, file);

	float *heightData = (float *)malloc(sizeof(float) * layers);
	float *VsData = (float *)malloc(sizeof(float) * layers);
	float *VpData = (float *)malloc(sizeof(float) * layers);
	float *RhoData = (float *)malloc(sizeof(float) * layers);

	for (F = 0; F < layers; ++F)
	{
		fread(heightData + F, sizeof(float), 1, file);
		fread(VsData + F, sizeof(float), 1, file);
		fread(VpData + F, sizeof(float), 1, file);
		fread(RhoData + F, sizeof(float), 1, file);

		heightData[F] *= 1000.0f;
		if (F != 0)
		{
			heightData[F] += 0.0f;
		}
		VsData[F] *= 1000.f;
		VpData[F] *= 1000.f;
		RhoData[F] *= 1000.f;
		// printf("height = %f, VsData = %f\n", heightData[F], VsData[F]);
	}

	fclose(file);

	int K = 0;
	float tolerance = 1.0f;
	float vs = 0.0f;
	float vp = 0.0f;
	float rho = 0.0f;

	int NZ = grid.NZ;
	double Depth = params.Depth * 1e3;
	double terrain = 0.0;
	int index1 = 0, index2 = 0;

	float h0 = 0.0, h1 = 0.0, h2 = 0.0;

	FOR_LOOP3D(i, j, k, 0, _nx_, 0, _ny_, 0, _nz_)
	// CALCULATE3D( i, j, k, 0, _nx_, 0, _ny_, 0, _nz_ )
	index = INDEX(i, j, k) * MSIZE;
#ifdef LayeredStructureTerrain
	pos = Index2D(i, j, _nx_, _ny_);
	terrain = cpu_terrain[pos];
	// printf( "terrain = %f\n", terrain  );
#else
	terrain = 0.0;
#endif

	K = grid.frontNZ + k;
	indexC = INDEX(i, j, k) * CSIZE;
	coordz = cpu_coord[indexC + 2];
	for (F = 1; F < layers; ++F)
	{
		h0 = -heightData[0] + terrain;
		h1 = -heightData[F - 1] + terrain;
		h2 = -heightData[F] + terrain;

		if (coordz > (h0 - tolerance))
		{
			vs = VsData[0];
			vp = VpData[0];
			rho = RhoData[0];
			// if( i == 100 && j == 100) {
			//	printf("coordZ = %f, Vs = %f, Vp = %f, rho = %f\n", coordz, vs, vp, rho );
			// }
			cpu_medium[index + 0] = vs;
			cpu_medium[index + 1] = vp;
			cpu_medium[index + 2] = rho;
		}
		if ((h1 - tolerance) > coordz && coordz > (h2 - tolerance))
		{
			vs = VsData[F - 1];
			vp = VpData[F - 1];
			rho = RhoData[F - 1];
			// if( i == 100 && j == 100) {
			//	printf("coordZ = %f, Vs = %f, Vp = %f, rho = %f\n", coordZ, Vs, Vp, rho );
			// }
			cpu_medium[index + 0] = vs;
			cpu_medium[index + 1] = vp;
			cpu_medium[index + 2] = rho;
		}
		if ((h2 - tolerance) > coordz)
		{
			vs = VsData[F];
			vp = VpData[F];
			rho = RhoData[F];
			// if( i == 100 && j == 100) {
			//	printf("coordZ = %f, Vs = %f, Vp = %f, rho = %f\n", coordZ, Vs, Vp, rho );
			// }
			cpu_medium[index + 0] = vs;
			cpu_medium[index + 1] = vp;
			cpu_medium[index + 2] = rho;
		}
	}

	END_LOOP3D()

	free(heightData);
	free(VsData);
	free(VpData);
	free(RhoData);
}

void setBasin(MPI_COORD thisMPICoord, GRID grid, float *coord, float *structure)
{
	if (thisMPICoord.Z != grid.PZ - 1)
		return;
	long long indexZ = 0;
	long long indexM = 0;
	long long indexSurf = 0;
	float coordZ = 0.0, coordSurf = 0.0;

	int i, j, k;
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;
	int _nz = grid._nz;

	float VS, VP, RHO;

	FOR_LOOP3D(i, j, k, 0, _nx_, 0, _ny_, 0, _nz_)
	indexZ = INDEX(i, j, k) * CSIZE;
	coordZ = coord[indexZ + 2];

	indexSurf = INDEX(i, j, _nz - 1) * CSIZE;
	coordSurf = coord[indexSurf + 2];
	if (coordZ >= coordSurf - 1000.0 && coordZ > 0.0)
	{
		VS = 300 + 36 * powf(coordZ, 0.43);
		VP = powf(4.57, 0.5) * VS;
		RHO = 1900;
		indexM = INDEX(i, j, k) * MSIZE;
		structure[indexM + 0] = VS;	 // LAM;
		structure[indexM + 1] = VP;	 // MU;
		structure[indexM + 2] = RHO; // RHO;
	}
	END_LOOP3D()
}

void constructMedium(MPI_COORD thisMPICoord, PARAMS params, GRID grid, float *cpu_coord, float *cpu_terrain, float *medium, float *cpu_medium)
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	long long num = _nx_ * _ny_ * _nz_;
	long long size = num * MSIZE * sizeof(float);

	// printf( "Vs: %p, Vp: %p, rho: %p\n", structure.Vs, structure.Vp, structure.rho  );
#ifdef GPU_CUDA
	dim3 threads(32, 16, 1);
	dim3 blocks;
	blocks.x = (_nx_ + threads.x - 1) / threads.x;
	blocks.y = (_ny_ + threads.y - 1) / threads.y;
	blocks.z = (_nz_ + threads.z - 1) / threads.z;

	construct_homo_medium<<<blocks, threads>>>(medium, _nx_, _ny_, _nz_);
	// CHECK( cudaDeviceSynchronize( )  );
	CHECK(cudaMemcpy(cpu_medium, medium, size, cudaMemcpyDeviceToHost));
	CHECK(cudaDeviceSynchronize());
#else
	construct_homo_medium(medium, _nx_, _ny_, _nz_);
#endif

	// printf( "1: medium = %p\n", medium );
	MPI_Barrier(MPI_COMM_WORLD);
	if (params.useMedium)
	{
		if (params.Crust_1Medel)
			readCrustal_1(params, grid, thisMPICoord, cpu_coord, cpu_medium);
		if (params.ShenModel)
			readWeisenShenModel(params, grid, thisMPICoord, cpu_coord, cpu_terrain, cpu_medium);
		if (params.LayeredModel)
			generate_1D_medium(params, grid, cpu_medium, cpu_coord, cpu_terrain);
#ifdef SET_BASIN
		setBasin(thisMPICoord, grid, cpu_coord, cpu_medium);
#endif

#ifdef GPU_CUDA
		CHECK(cudaMemcpy(medium, cpu_medium, size, cudaMemcpyHostToDevice));
		CHECK(cudaDeviceSynchronize());
#endif
		// printf( "2: cpu_medium = %p\n", cpu_medium );
	}

	MPI_Barrier(MPI_COMM_WORLD);

	/*
	int i = HALO;
	int j = HALO;
	int k = HALO;

	int _nx = grid._nx;
	int _ny = grid._ny;
	int _nz = grid._nz;

	long long index = 0;
	FOR_LOOP3D( i, j, k, HALO, _nx, HALO, _ny, HALO, _nz )
		index = INDEX( i, j, k ) * MSIZE;
		if ( cpu_medium[index+0] < 1000.0 )
			printf( "%f\n", cpu_medium[index+0] );
	END_LOOP3D( )
	//printf( "2: num = %d\n", _nx_ * _ );
	//printf( "2: cpu_medium = %p\n", cpu_medium );
	*/
}
