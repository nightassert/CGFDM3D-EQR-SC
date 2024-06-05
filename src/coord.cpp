/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: coord.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2021-09-06
*   Discription: Construct Coordinate

*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*
=================================================================*/

#include "header.h"

void allocCoord(GRID grid, float **coord, float **cpu_coord)
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	long long num = _nx_ * _ny_ * _nz_;

	long long size = sizeof(float) * num * CSIZE;

	CHECK(Malloc((void **)coord, size));
	if (*coord == NULL)
	{
		printf("can't allocate Coordinate memory!\n");
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}
	CHECK(Memset(*coord, 0, size));

#ifdef GPU_CUDA
	*cpu_coord = (float *)malloc(size);

	if (*cpu_coord == NULL)
	{
		printf("can't allocate Coordinate memory!\n");
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}

	memset(*cpu_coord, 0, size);
#else
	*cpu_coord = *coord;
#endif
}

void freeCoord(float *coord, float *cpu_coord)
{
	Free(coord);
#ifdef GPU_CUDA
	free(cpu_coord);
#endif
}

void allocTerrain(GRID grid, float **terrain, float **cpu_terrain)
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	long long num = _nx_ * _ny_;

	long long size = sizeof(float) * num;

	CHECK(Malloc((void **)terrain, size));
	if (*terrain == NULL)
	{
		printf("can't allocate Terrain memory!\n");
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}
	CHECK(Memset(*terrain, 0, size));

#ifdef GPU_CUDA
	*cpu_terrain = (float *)malloc(size);

	if (*cpu_terrain == NULL)
	{
		printf("can't allocate Terrain memory!\n");
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}

	memset(*cpu_terrain, 0, size);
#else
	*cpu_terrain = *terrain;
#endif
}

void freeTerrain(float *terrain, float *cpu_terrain)
{
	Free(terrain);
#ifdef GPU_CUDA
	free(cpu_terrain);
#endif
}

__GLOBAL__
void construct_flat_coord(
	float *coord, float *terrain,
	int _nx_, int _ny_, int _nz_,
	int frontNX, int frontNY, int frontNZ,
	int originalX, int originalY,
	int NZ,
	float DH)
{
	long long index = 0;
	int I = 0, J = 0, K = 0;
#ifdef GPU_CUDA
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
#else
	int i = 0;
	int j = 0;
	int k = 0;
#endif
	long long pos = 0;
	CALCULATE3D(i, j, k, 0, _nx_, 0, _ny_, 0, _nz_)
	index = INDEX(i, j, k) * CSIZE;
	I = frontNX + i;
	J = frontNY + j;
	K = frontNZ + k;
	if (K == NZ + HALO - 1)
	{
		pos = Index2D(i, j, _nx_, _ny_);
		terrain[pos] = (K - HALO + 1) * DH - NZ * DH;
	}
	coord[index + 0] = (I - HALO) * DH - originalX * DH;
	coord[index + 1] = (J - HALO) * DH - originalY * DH;
	coord[index + 2] = (K - HALO + 1) * DH - NZ * DH;
	END_CALCULATE3D()
}

__GLOBAL__
void construct_gauss_hill_surface(
	float *terrain,
	int _nx_, int _ny_,
	int frontNX, int frontNY,
	int originalX, int originalY,
	int NZ,
	float DH, float cal_depth)
{
#ifdef GPU_CUDA
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
#else
	int i = 0;
	int j = 0;
#endif

	float x, y;

	float height = 0.0;

	float h = 0.2 * cal_depth;
	float a = 0.1 * cal_depth, b = 0.1 * cal_depth;

	int I = 0, J = 0;

	long long index, pos;

	CALCULATE2D(i, j, 0, _nx_, 0, _ny_)
	index = Index2D(i, j, _nx_, _ny_);
	I = frontNX + i;
	J = frontNY + j;
	x = (I - HALO) * DH - originalX * DH;
	y = (J - HALO) * DH - originalY * DH;

	height = h * exp(-0.5f * (x * x / (a * a) + y * y / (b * b)));
	pos = Index2D(i, j, _nx_, _ny_);
	terrain[pos] = height;
	// DZ[index] = double( height + abs( cal_depth ) ) / double( NZ - 1 );

	END_CALCULATE2D()
}

__GLOBAL__
void verifyDZ(float *DZ, int _nx_, int _ny_, int _nz_)
{
#ifdef GPU_CUDA
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
#else
	int i = 0;
	int j = 0;
	int k = 0;
#endif

	long long index;

	CALCULATE3D(i, j, k, 0, _nx_, 0, _ny_, 0, _nz_)
	index = Index2D(i, j, _nx_, _ny_);
	if (index == _nx_ * _ny_ - 2)
	{
		printf("2: &DZ = %p\n", DZ + index);
	}

	END_CALCULATE3D()
}

__GLOBAL__
void construct_terrain_coord(
	float *coord, float *terrain,
	int _nx_, int _ny_, int _nz_,
	int frontNZ,
	int NZ,
	float cal_depth)
{
	long long index = 0, pos = 0;
#ifdef GPU_CUDA
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
#else
	int i = 0;
	int j = 0;
	int k = 0;
#endif
	int K;
	float DZ, height;
	int _nx = _nx_ - HALO;
	int _ny = _ny_ - HALO;
	int _nz = _nz_ - HALO;

	CALCULATE3D(i, j, k, 0, _nx_, 0, _ny_, 0, _nz_)
	index = INDEX(i, j, k);
	pos = Index2D(i, j, _nx_, _ny_);
	K = frontNZ + k - HALO;
	height = terrain[pos];
	DZ = double(height + abs(cal_depth)) / double(NZ - 1);
	coord[index * CSIZE + 2] = -abs(cal_depth) + DZ * K;
	// printf( "========================\n"  );
	/*
	if ( index == INDEX( _nx/2 - 10, _ny / 2 - 9, _nz - 1 ) )
	{
		printf( "coordZ = %3.10e\n", coord[index * CSIZE+2]  );
		printf( "K = %d\n", K  );
		printf( "height = %3.10e\n", height  );
		printf( "DZ = %3.10e\n", DZ  );
		printf( "z-1 = %3.10e\n", coord[INDEX_et( i, j, k, -1 )*CSIZE+2]  );
		printf( "z+0 = %3.10e\n", coord[INDEX_et( i, j, k, +0 )*CSIZE+2]  );
		printf( "z+1 = %3.10e\n", coord[INDEX_et( i, j, k, +1 )*CSIZE+2]  );
		printf( "z+2 = %3.10e\n", coord[INDEX_et( i, j, k, +2 )*CSIZE+2]  );
		printf( "z+3 = %3.10e\n", coord[INDEX_et( i, j, k, +3 )*CSIZE+2]  );

		printf( "z-2 = %3.10e\n", coord[INDEX_et( i, j, k, -2 )*CSIZE+2]  );
		printf( "z-3 = %3.10e\n", coord[INDEX_et( i, j, k, -3 )*CSIZE+2]  );

	}
	*/
	END_CALCULATE3D()
}

// void calculate_range( float * data, float range[2], long long num );

void constructCoord(MPI_Comm comm_cart, MPI_COORD thisMPICoord, GRID grid, PARAMS params, float *coord, float *cpu_coord, float *terrain, float *cpu_terrain)
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	int frontNX = grid.frontNX;
	int frontNY = grid.frontNY;
	int frontNZ = grid.frontNZ;

	int NZ = grid.NZ;

	int originalX = grid.originalX;
	int originalY = grid.originalY;

	long long num = _nx_ * _ny_ * _nz_;
	long long size = num * CSIZE * sizeof(float);
	long long ter_size = _nx_ * _ny_ * sizeof(float);

	float DH = grid.DH;

	float cal_depth = params.Depth * 1000;

#ifdef GPU_CUDA
	dim3 threads(32, 16, 1);
	dim3 blocks;
	blocks.x = (_nx_ + threads.x - 1) / threads.x;
	blocks.y = (_ny_ + threads.y - 1) / threads.y;
	blocks.z = (_nz_ + threads.z - 1) / threads.z;

	construct_flat_coord<<<blocks, threads>>>(coord, terrain, _nx_, _ny_, _nz_,
											  frontNX, frontNY, frontNZ, originalX, originalY, NZ, DH);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(cpu_coord, coord, size, cudaMemcpyDeviceToHost));
	CHECK(cudaMemcpy(cpu_terrain, terrain, ter_size, cudaMemcpyDeviceToHost));
	CHECK(cudaDeviceSynchronize());
#else
	construct_flat_coord(coord, terrain, _nx_, _ny_, _nz_,
						 frontNX, frontNY, frontNZ, originalX, originalY, NZ, DH);
#endif

	int useTerrain = params.useTerrain;
	int gauss_hill = params.gauss_hill;

	if (gauss_hill)
	{
#ifdef GPU_CUDA
		dim3 threadXY(32, 16, 1);
		dim3 blockXY;
		blockXY.x = (_nx_ + threadXY.x - 1) / threadXY.x;
		blockXY.y = (_ny_ + threadXY.y - 1) / threadXY.y;
		blockXY.z = 1;
		construct_gauss_hill_surface<<<blockXY, threadXY>>>(terrain, _nx_, _ny_, frontNX, frontNY, originalX, originalY, NZ, DH, cal_depth);

		construct_terrain_coord<<<blocks, threads>>>(coord, terrain, _nx_, _ny_, _nz_, frontNZ, NZ, cal_depth);
		// printf( "========================\n"  );
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(cpu_coord, coord, size, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(cpu_terrain, terrain, ter_size, cudaMemcpyDeviceToHost));
		CHECK(cudaDeviceSynchronize());
#else
		construct_gauss_hill_surface(terrain, _nx_, _ny_, frontNX, frontNY, originalX, originalY, NZ, DH, cal_depth);
		construct_terrain_coord(coord, terrain, _nx_, _ny_, _nz_, frontNZ, NZ, cal_depth);
#endif
	}

	if (useTerrain)
	{
		preprocessTerrain(params, comm_cart, thisMPICoord, grid, cpu_coord, cpu_terrain);
#ifdef GPU_CUDA
		CHECK(cudaMemcpy(coord, cpu_coord, size, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(terrain, cpu_terrain, ter_size, cudaMemcpyHostToDevice));
#endif
	}
}
