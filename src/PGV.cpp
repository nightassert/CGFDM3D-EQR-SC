/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name:propagate.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time:2021-11-20
*   Discription:
*
================================================================*/
#include "header.h"

void allocatePGV(GRID grid, float **pgv, float **cpu_pgv)
{
	int nx = grid.nx;
	int ny = grid.ny;

	int len = sizeof(float) * nx * ny * PGVSIZE;
	CHECK(Malloc((void **)pgv, len));
	if (*pgv == NULL)
	{
		printf("can't allocate pgv memory!\n");
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}
	CHECK(Memset(*pgv, 0, len));

#ifdef GPU_CUDA
	*cpu_pgv = (float *)malloc(len);
	if (*cpu_pgv == NULL)
	{
		printf("can't allocate pgv memory!\n");
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}
	memset(*cpu_pgv, 0, len);
#else
	*cpu_pgv = *pgv;
#endif
}

void freePGV(float *pgv, float *cpu_pgv)
{
	Free(pgv);
#ifdef GPU_CUDA
	free(cpu_pgv);
#endif
}

__GLOBAL__
void compare_pgv(float *pgv, FLOAT *W, int nx, int ny, int nz
#ifdef SCFDM
				 ,
				 FLOAT *CJM
#endif
)
{
	int _nx_ = nx + 2 * HALO;
	int _ny_ = ny + 2 * HALO;
	int _nz_ = nz + 2 * HALO;

#ifdef GPU_CUDA
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
#else
	int i0 = 0;
	int j0 = 0;
#endif

	// printf( "==========compare pgv===============\n"  );

	int i = i0 + HALO;
	int j = j0 + HALO;
	int k = _nz_ - HALO - 1;

	long long index, pos;

#ifdef SCFDM
	float buoyancy;
#endif

	float Vx = 0.0f, Vy = 0.0f, Vz = 0.0f, Vh = 0.0f, V = 0.0f;

	double c = 1.0 / Cv;
	CALCULATE2D(i0, j0, 0, nx, 0, ny)
	i = i0 + HALO;
	j = j0 + HALO;
	index = INDEX(i, j, k);
	pos = Index2D(i0, j0, nx, ny);

	// if ( i0 == nx - 1 && j0 == ny - 1 )
	//	printf( "nx = %d, ny = %d, pos = %d, pgvh = %p, pgv = %p\n", nx, ny, pos, pgv.pgvh, pgv.pgv );

#ifdef SCFDM
	buoyancy = CJM[index * CJMSIZE + 12];
	buoyancy *= Crho;

	Vx = (float)W[index * WSIZE + 6] * c * buoyancy;
	Vy = (float)W[index * WSIZE + 7] * c * buoyancy;
	Vz = (float)W[index * WSIZE + 8] * c * buoyancy;
#else
	Vx = (float)W[index * WSIZE + 0] * c;
	Vy = (float)W[index * WSIZE + 1] * c;
	Vz = (float)W[index * WSIZE + 2] * c;
#endif

	Vh = sqrtf(Vx * Vx + Vy * Vy);
	V = sqrtf(Vx * Vx + Vy * Vy + Vz * Vz);

	if (pgv[pos * PGVSIZE + 0] < Vh) // PGVh
	{
		pgv[pos * PGVSIZE + 0] = Vh;
	}
	if (pgv[pos * PGVSIZE + 1] < V) // PGV
	{
		pgv[pos * PGVSIZE + 1] = V;
	}
	END_CALCULATE2D()
}

void outputPgvData(PARAMS params, MPI_COORD thisMPICoord, float *cpuPgv, int nx, int ny)
{
	char fileName[1024] = {0};
	FILE *filePgv = NULL;
	sprintf(fileName, "%s/PGV_Z_mpi_%d_%d_%d.bin", params.OUT, thisMPICoord.X, thisMPICoord.Y, thisMPICoord.Z);
	filePgv = fopen(fileName, "wb");
	fwrite(cpuPgv, sizeof(float), nx * ny * PGVSIZE, filePgv);
	fclose(filePgv);
}

void comparePGV(GRID grid, MPI_COORDINATE thisMPICoord, FLOAT *W, float *pgv
#ifdef SCFDM
				,
				FLOAT *CJM
#endif
)
{
	int nx = grid.nx;
	int ny = grid.ny;
	int nz = grid.nz;
#ifdef GPU_CUDA
	dim3 threads(32, 16, 1);
	dim3 blocks;
	blocks.x = (nx + threads.x - 1) / threads.x;
	blocks.y = (ny + threads.y - 1) / threads.y;
	blocks.z = 1;

	compare_pgv<<<blocks, threads>>>(pgv, W, nx, ny, nz
#ifdef SCFDM
									 ,
									 CJM
#endif
	);
	CHECK(cudaDeviceSynchronize());
#else
	compare_pgv(pgv, W, nx, ny, nz);
#endif
}

void outputPGV(PARAMS params, GRID grid, MPI_COORDINATE thisMPICoord, float *pgv, float *cpuPgv)
{
	int nx = grid.nx;
	int ny = grid.ny;

	int size;
	size = sizeof(float) * nx * ny * PGVSIZE;
#ifdef GPU_CUDA
	CHECK(cudaMemcpy(cpuPgv, pgv, size, cudaMemcpyDeviceToHost));
#endif
	outputPgvData(params, thisMPICoord, cpuPgv, nx, ny);
}
