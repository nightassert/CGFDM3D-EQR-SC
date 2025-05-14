/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: CJM.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2022-11-03
*   Discription: Construct Contravariant Jacobian Medium (CJM)
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

void solveContravariantJac(MPI_Comm comm_cart, GRID grid, float *con, float *coord, float *Jac, float *medium, Mat_rDZ mat_rDZ);

void allocContravariant(GRID grid, float **contravariant);
void freeContravariant(float *contravariant);
void allocJac(GRID grid, float **Jac);
void freeJac(float *Jac);

void allocCJM(GRID grid, FLOAT **CJM)
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	long long num = _nx_ * _ny_ * _nz_;

	long long size = sizeof(FLOAT) * num * CJMSIZE;

	CHECK(Malloc((void **)CJM, size));
	if (*CJM == NULL)
	{
		printf("can't allocate CJM memory!\n");
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}
	CHECK(Memset(*CJM, 0, size));
}

void freeCJM(FLOAT *CJM)
{
	Free(CJM);
}

__GLOBAL__
void calculate_medium(float *medium, int num)
{

#ifdef GPU_CUDA
	int i = threadIdx.x + blockIdx.x * blockDim.x;
#else
	int i = 0;
#endif
	float Vp, Vs, rho;
	double c = 1.0 * Cs / Cv;

	float lambda;
	float mu;
	float buoyancy;

	CALCULATE1D(i, 0, num)
	Vs = medium[i * MSIZE + 0];
	Vp = medium[i * MSIZE + 1];
	rho = medium[i * MSIZE + 2];

	medium[i * MSIZE + 0] = rho * (Vs * Vs);
	medium[i * MSIZE + 1] = rho * (Vp * Vp - 2.0f * Vs * Vs);
	medium[i * MSIZE + 2] = 1.0f / rho;
	END_CALCULATE1D()
}

void vs_vp_rho2lam_mu_bou(GRID grid, float *medium)
{

	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	long long num = _nx_ * _ny_ * _nz_;

#ifdef GPU_CUDA

	dim3 threadX;
	dim3 blockX;
	threadX.x = 512;
	threadX.y = 1;
	threadX.z = 1;
	blockX.x = (num + threadX.x - 1) / threadX.x;
	blockX.y = 1;
	blockX.z = 1;

	calculate_medium<<<blockX, threadX>>>(medium, num);

#else
	calculate_medium(medium, num);
#endif
}

__GLOBAL__
void set_CJM(FLOAT *CJM, float *medium, float *con, float *Jac, int num)
{
#ifdef GPU_CUDA
	int i = threadIdx.x + blockIdx.x * blockDim.x;
#else
	int i = 0;
#endif
	float Vp, Vs, rho;

	CALCULATE1D(i, 0, num)

	CJM[i * CJMSIZE + 0] = con[i * CONSIZE + 0]; // xi_x_J
	CJM[i * CJMSIZE + 1] = con[i * CONSIZE + 1]; // xi_y_J
	CJM[i * CJMSIZE + 2] = con[i * CONSIZE + 2]; // xi_z_J
	CJM[i * CJMSIZE + 3] = con[i * CONSIZE + 3]; // et_x_J
	CJM[i * CJMSIZE + 4] = con[i * CONSIZE + 4]; // et_y_J
	CJM[i * CJMSIZE + 5] = con[i * CONSIZE + 5]; // et_z_J
	CJM[i * CJMSIZE + 6] = con[i * CONSIZE + 6]; // zt_x_J
	CJM[i * CJMSIZE + 7] = con[i * CONSIZE + 7]; // zt_y_J
	CJM[i * CJMSIZE + 8] = con[i * CONSIZE + 8]; // zt_z_J

	CJM[i * CJMSIZE + 9] = Jac[i]; // ! This is actually the inverse of Jacobian if you use SCFDM, which is just a different formula but the same value

	CJM[i * CJMSIZE + 10] = medium[i * MSIZE + 0];
	CJM[i * CJMSIZE + 11] = medium[i * MSIZE + 1];
	CJM[i * CJMSIZE + 12] = medium[i * MSIZE + 2];

	END_CALCULATE1D()
}

void setCJM(GRID grid, FLOAT *CJM, float *medium, float *con, float *Jac)
{

	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	long long num = _nx_ * _ny_ * _nz_;

#ifdef GPU_CUDA

	dim3 threadX;
	dim3 blockX;
	threadX.x = 512;
	threadX.y = 1;
	threadX.z = 1;
	blockX.x = (num + threadX.x - 1) / threadX.x;
	blockX.y = 1;
	blockX.z = 1;

	set_CJM<<<blockX, threadX>>>(CJM, medium, con, Jac, num);

#else
	set_CJM(CJM, medium, con, Jac, num);
#endif
}
void data2D_output_bin(GRID grid, SLICE slice,
					   MPI_COORD thisMPICoord,
					   void *datain, int VAR, int VARSIZE, SLICE_DATA sliceData, SLICE_DATA sliceDataCpu,
					   const char *name, int CVS, int FP_TYPE);
extern PARAMS g_params;
extern MPI_COORD g_thisMPICoord;
void constructCJM(MPI_Comm comm_cart, MPI_NEIGHBOR mpiNeighbor, GRID grid, FLOAT *CJM, float *coord, float *medium, Mat_rDZ mat_rDZ)
{
	float *con, *Jac;
	allocContravariant(grid, &con);
	allocJac(grid, &Jac);

	vs_vp_rho2lam_mu_bou(grid, medium);

	solveContravariantJac(comm_cart, grid, con, coord, Jac, medium, mat_rDZ);
	MPI_Barrier(MPI_COMM_WORLD);

	// char name[256];
	// sprintf( name, "%s/ZT_X", g_params.OUT );

	// SLICE_DATA sliceData, sliceDataCpu;
	// SLICE slice = { 0 };
	// locateSlice( g_params, grid, &slice );
	// allocSliceData( grid, slice, &sliceData, &sliceDataCpu );

	// Jacobian Transmit
	SEND_RECV_DATA sr_jac;
	int VARSIZE = 1;
	allocSendRecv(grid, mpiNeighbor, &sr_jac, VARSIZE);
	mpiSendRecv(comm_cart, mpiNeighbor, grid, Jac, sr_jac, VARSIZE);
	freeSendRecv(mpiNeighbor, sr_jac);

	// int FP_TYPE = 1;
	// data2D_output_bin( grid, slice, g_thisMPICoord, Jac, 0, 1, sliceData, sliceDataCpu, name, 0, FP_TYPE );

	MPI_Barrier(MPI_COMM_WORLD);

	// Contravariant Transmit
	SEND_RECV_DATA sr_con;
	VARSIZE = CONSIZE;
	allocSendRecv(grid, mpiNeighbor, &sr_con, VARSIZE);
	mpiSendRecv(comm_cart, mpiNeighbor, grid, con, sr_con, VARSIZE);
	freeSendRecv(mpiNeighbor, sr_con);
	MPI_Barrier(MPI_COMM_WORLD);

#ifdef SCFDM
	// ! For alternative flux finite difference by Tianhong Xu
	// Medium Transmit
	SEND_RECV_DATA sr_medium;
	VARSIZE = MEDIUMSIZE;
	allocSendRecv(grid, mpiNeighbor, &sr_medium, VARSIZE);
	mpiSendRecv(comm_cart, mpiNeighbor, grid, medium, sr_medium, VARSIZE);
	freeSendRecv(mpiNeighbor, sr_medium);
	MPI_Barrier(MPI_COMM_WORLD);
#endif

	// data2D_output_bin( grid, slice, g_thisMPICoord, con, 6, CONSIZE, sliceData, sliceDataCpu, name, 0, FP_TYPE );

	setCJM(grid, CJM, medium, con, Jac);

	freeJac(Jac);
	freeContravariant(con);
	// freeSliceData( grid, slice, sliceData, sliceDataCpu );

	// SEND_RECV_DATA sr_coord;
	// VARSIZE = CSIZE;
	// allocSendRecv( grid, mpiNeighbor, &sr_coord, VARSIZE );
	// mpiSendRecv( comm_cart, mpiNeighbor, grid, coord, sr_coord, VARSIZE );
	// freeSendRecv( mpiNeighbor, sr_coord );
}
