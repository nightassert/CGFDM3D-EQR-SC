/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name:calc_CFL.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time:2021-11-01
*   Discription:
*
================================================================*/

#include "header.h"

typedef struct STRUCTURE
{
	float *Vs;
	float *Vp;
	float *rho;
} STRUCTURE;

void allocStructure(GRID grid, STRUCTURE *structure)
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	long long num = _nx_ * _ny_ * _nz_;

	float *pStructure = NULL;
	long long size = sizeof(float) * num * MSIZE;

	pStructure = (float *)malloc(size);
	if (pStructure == NULL)
	{
		printf("can't allocate STRUCTURE memory!\n");
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}

	structure->Vs = pStructure;
	structure->Vp = pStructure + num;
	structure->rho = pStructure + num * 2;
}

void freeStructure(STRUCTURE structure)
{
	free(structure.Vs);
}

double distance_point2plane(double P[3], double A[3], double B[3], double C[3])
{
	double AB[3] = {
		B[0] - A[0],
		B[1] - A[1],
		B[2] - A[2]};

	double AC[3] = {
		C[0] - A[0],
		C[1] - A[1],
		C[2] - A[2]};

	// double BC[3] = {
	//	C[0] - B[0],
	//	C[1] - B[1],
	//	C[2] - B[2]
	// };

	double n[3] = {
		AB[1] * AC[2] - AB[2] * AC[1],
		AB[2] * AC[0] - AB[0] * AC[2],
		AB[0] * AC[1] - AB[1] * AC[0]};

	double PA[3] = {
		P[0] - A[0],
		P[1] - A[1],
		P[2] - A[2]};

	double n_dis = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);

	n[0] = n[0] / n_dis;
	n[1] = n[1] / n_dis;
	n[2] = n[2] / n_dis;

	double d = abs(PA[0] * n[0] + PA[1] * n[1] + PA[2] * n[2]);
	return d;
}

void calculate_DH_Range(
	float *coord,
	float h_min_max[2],
	int _nx_, int _ny_, int _nz_)
{
	int i = HALO;
	int j = HALO;
	int k = HALO;
	long long index = 0;

	int _nx = _nx_ - HALO;
	int _ny = _ny_ - HALO;
	int _nz = _nz_ - HALO;

	int nx = _nx - HALO;
	int ny = _ny - HALO;
	int nz = _nz - HALO;

	int ii, jj, kk;

	double P[3], A[3], B[3], C[3];
	long long posA = 0;
	long long posB = 0;
	long long posC = 0;

	double d = 0.0;
	double hmin = 1.0e20;
	double hmax = 0.0;

	FOR_LOOP3D(i, j, k, HALO, _nx, HALO, _ny, HALO, _nz)
	index = INDEX(i, j, k) * CSIZE;

	P[0] = coord[index + 0];
	P[1] = coord[index + 1];
	P[2] = coord[index + 2];

	// if ( coord[index+2] > -10e3 )
	// printf( "coord = %f\n", coord[index+2] );

	for (ii = -1; ii <= 1; ii += 2)
		for (jj = -1; jj <= 1; jj += 2)
			for (kk = -1; kk <= 1; kk += 2)
			{
				posA = INDEX(i - ii, j, k) * CSIZE; // 前后
				posB = INDEX(i, j - jj, k) * CSIZE; // 左右
				posC = INDEX(i, j, k - kk) * CSIZE; // 上下

				A[0] = coord[posA + 0];
				A[1] = coord[posA + 1];
				A[2] = coord[posA + 2];

				B[0] = coord[posB + 0];
				B[1] = coord[posB + 1];
				B[2] = coord[posB + 2];

				C[0] = coord[posC + 0];
				C[1] = coord[posC + 1];
				C[2] = coord[posC + 2];

				d = distance_point2plane(P, A, B, C);
				// d = 100.0f / sqrt( 3.0 );
				hmin = MIN(hmin, d);
				hmax = MAX(hmax, d);
			}

	END_LOOP3D()

	h_min_max[0] = hmin;
	h_min_max[1] = hmax;
}

void calculate_range(float *data, float range[2], long long num)
{
	long long i = 0;
	float maxValue = -1.0e20, minValue = 1.0e20;
	for (i = 0; i < num; i++)
	{

		if (data[i] < minValue)
			minValue = data[i];

		if (data[i] > maxValue)
			maxValue = data[i];
	}
	range[0] = minValue;
	range[1] = maxValue;
}

/*
__GLOBAL__
void medium2structure( GRID grid, STRUCTURE structure, float * medium )
{
	//printf( "3: medium = %p\n", medium );
#ifdef GPU_CUDA
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
#else
	int i = 0;
	int j = 0;
	int k = 0;
#endif
	long long index = 0, idx = 0;

	int nx = grid.nx;
	int ny = grid.ny;
	int nz = grid.nz;

	int _nx = grid._nx;
	int _ny = grid._ny;
	int _nz = grid._nz;

	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;


	FOR_LOOP3D( i, j, k, 0, _nx_, 0, _ny_, 0, _nz_ )
		index = INDEX( i, j, k ) * MSIZE;
		structure.Vs [index] = medium[index+0];
		structure.Vp [index] = medium[index+1];
		structure.rho[index] = medium[index+2];

		if ( medium[index+0] < 1000.0 )
			printf( "%f\n", medium[index+0] );

	END_LOOP3D( )
}
*/

void medium2structure(GRID grid, STRUCTURE structure, float *medium)
{
	// printf( "3: medium = %p\n", medium );
	long long i = 0;
	int nx = grid.nx;
	int ny = grid.ny;
	int nz = grid.nz;

	int _nx = grid._nx;
	int _ny = grid._ny;
	int _nz = grid._nz;

	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	long long num = grid._nx_ * grid._ny_ * grid._nz_;
	// printf( "3: num = %d\n", num );

	FOR_LOOP1D(i, 0, num)
	structure.Vs[i] = medium[i * MSIZE + 0];
	structure.Vp[i] = medium[i * MSIZE + 1];
	structure.rho[i] = medium[i * MSIZE + 2];
	END_LOOP1D()
}

#ifdef SCFDM
#ifdef LF
extern float vp_max_for_SCFDM;
#endif
#endif
void calc_CFL(GRID grid, float *coord, float *medium, PARAMS params)
{

	int nx = grid.nx;
	int ny = grid.ny;
	int nz = grid.nz;

	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	float h_min_max[2];

	float Vs_min_max[2];
	float Vp_min_max[2];
	float rho_min_max[2];

	long long num = nx * ny * nz;

	calculate_DH_Range(coord, h_min_max, _nx_, _ny_, _nz_);

	MPI_Barrier(MPI_COMM_WORLD);

	STRUCTURE structure;
	allocStructure(grid, &structure);
	medium2structure(grid, structure, medium);

	num = _nx_ * _ny_ * _nz_;
	calculate_range(structure.Vs, Vs_min_max, num);
	calculate_range(structure.Vp, Vp_min_max, num);
	calculate_range(structure.rho, rho_min_max, num);

	freeStructure(structure);

	MPI_Barrier(MPI_COMM_WORLD);

	int thisRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &thisRank);
	// printf( "thisRank = %d, H   Range: %5.f ~ %5.f\n", thisRank, h_min_max[0], h_min_max[1] );
	// printf( "thisRank = %d, Vs  Range: %5.f ~ %5.f\n", thisRank, Vs_min_max[0], Vs_min_max[1] );
	// printf( "thisRank = %d, Vp  Range: %5.f ~ %5.f\n", thisRank, Vp_min_max[0], Vp_min_max[1] );
	// printf( "thisRank = %d, rho Range: %5.f ~ %5.f\n", thisRank, rho_min_max[0], rho_min_max[1] );

	float H_min, H_max;
	float Vs_min, Vs_max;
	float Vp_min, Vp_max;
	float rho_min, rho_max;

	MPI_Allreduce(&h_min_max[0], &H_min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&h_min_max[1], &H_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

	MPI_Allreduce(&Vs_min_max[0], &Vs_min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&Vs_min_max[1], &Vs_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

	MPI_Allreduce(&Vp_min_max[0], &Vp_min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&Vp_min_max[1], &Vp_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

	MPI_Allreduce(&rho_min_max[0], &rho_min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&rho_min_max[1], &rho_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

#ifdef SCFDM
	// ! For alternative flux finite difference by Tianhong Xu
#ifdef LF
	// ! For Lax-Friedriches Riemann solver: alpha = vp_max
	vp_max_for_SCFDM = Vp_max;
#endif
#endif

	float dtmax = 1.34 * H_min / Vp_max;

	if (0 == thisRank)
	{

		// cout << "H   : min = " << H_min    << ", max = " << H_max   << endl;
		// cout << "Vs  : min = " << Vs_min   << ", max = " << Vs_max  << endl;
		// cout << "Vp  : min = " << Vp_min   << ", max = " << Vp_max  << endl;
		// cout << "rho : min = " << rho_min  << ", max = " << rho_max << endl;

		printf("H   Range: %5.4e ~ %5.4e\n", H_min, H_max);
		printf("Vs  Range: %5.4e ~ %5.4e\n", Vs_min, Vs_max);
		printf("Vp  Range: %5.4e ~ %5.4e\n", Vp_min, Vp_max);
		printf("rho Range: %5.4e ~ %5.4e\n", rho_min, rho_max);

		printf("dtmax = %5.2e, DT = %5.2e\n", dtmax, params.DT);
		if (params.DT > dtmax)
		{
			printf("The parameters can't afford the CFL condition!\n");
			MPI_Abort(MPI_COMM_WORLD, 110);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
}
