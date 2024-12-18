/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: PGV.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2021-11-20
*   Discription: Compare PGV
*
*	Update: Tianhong Xu, 12231218@mail.sustech.edu.cn
*   Update Time: 2023-11-16
*   Update Content: Add SCFDM
*
*	Update: Tianhong Xu, 12231218@mail.sustech.edu.cn
*   Update Time: 2024-06-11
*   Update Content: Modify the equations to Wenqiang Zhang (2023)
*
*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*      3. Zhang, W., Liu, Y., & Chen, X. (2023). A Mixed‐Flux‐Based Nodal Discontinuous Galerkin Method for 3D Dynamic Rupture Modeling. Journal of Geophysical Research: Solid Earth, e2022JB025817.
*
=================================================================*/

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
void compare_pgv(float *pgv, FLOAT *W, int nx, int ny, int nz, float DT, int it
#ifdef SCFDM
				 ,
				 FLOAT *CJM
#endif

#ifdef SOLVE_PGA
				 ,
				 FLOAT *W_pre

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
#ifdef SOLVE_PGA
	float Ax = 0.0f, Ay = 0.0f, Az = 0.0f, Ah = 0.0f, A = 0.0f;
	float Vx_pre = 0.0f, Vy_pre = 0.0f, Vz_pre = 0.0f;
#endif

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

	Vx = (float)W[index * WSIZE + 0] * c * buoyancy;
	Vy = (float)W[index * WSIZE + 1] * c * buoyancy;
	Vz = (float)W[index * WSIZE + 2] * c * buoyancy;
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

#ifdef SOLVE_PGA

	if (it % 10 == 0)
	{
#ifdef SCFDM
		buoyancy = CJM[index * CJMSIZE + 12];
		buoyancy *= Crho;

		Vx_pre = (float)W_pre[index * WSIZE + 0] * c * buoyancy;
		Vy_pre = (float)W_pre[index * WSIZE + 1] * c * buoyancy;
		Vz_pre = (float)W_pre[index * WSIZE + 2] * c * buoyancy;
#else
		Vx_pre = (float)W_pre[index * WSIZE + 0] * c;
		Vy_pre = (float)W_pre[index * WSIZE + 1] * c;
		Vz_pre = (float)W_pre[index * WSIZE + 2] * c;
#endif

		Ax = (Vx - Vx_pre) / (10 * DT);
		Ay = (Vy - Vy_pre) / (10 * DT);
		Az = (Vz - Vz_pre) / (10 * DT);

		Ah = sqrtf(Ax * Ax + Ay * Ay);
		A = sqrtf(Ax * Ax + Ay * Ay + Az * Az);

		if (pgv[pos * PGVSIZE + 2] < Ah) // PGAh
		{
			pgv[pos * PGVSIZE + 2] = Ah;
		}
		if (pgv[pos * PGVSIZE + 3] < A) // PGA
		{
			pgv[pos * PGVSIZE + 3] = A;
		}
	}
#endif
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

void comparePGV(GRID grid, MPI_COORDINATE thisMPICoord, FLOAT *W, float *pgv, float DT, int it
#ifdef SCFDM
				,
				FLOAT *CJM
#endif

#ifdef SOLVE_PGA
				,
				FLOAT *W_pre

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

	compare_pgv<<<blocks, threads>>>(pgv, W, nx, ny, nz, DT, it
#ifdef SCFDM
									 ,
									 CJM
#endif

#ifdef SOLVE_PGA
									 ,
									 W_pre

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
