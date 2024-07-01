/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: pml_rk.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2021-11-05
*   Discription: PML Runge-Kutta Time Integration
*
*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*
=================================================================*/

#include "header.h"
typedef void (*WAVE_RK_FUNC)(float *h_W, float *W, float *t_W, float *m_W, long long num);

__GLOBAL__
void wave_pml_rk0(float *h_W, float *W, float *t_W, float *m_W, long long WStride)
{
#ifdef GPU_CUDA
	int i = threadIdx.x + blockIdx.x * blockDim.x;
#else
	int i = 0;
#endif
	CALCULATE1D(i, 0, WStride)
	// if( i == WStride )
	//	printf( "WStride = %d\n", WStride );
	m_W[i] = W[i];
	t_W[i] = m_W[i] + beta1 * h_W[i];
	W[i] = m_W[i] + alpha2 * h_W[i];
	END_CALCULATE1D()
}

__GLOBAL__
void wave_pml_rk1(float *h_W, float *W, float *t_W, float *m_W, long long WStride)
{
#ifdef GPU_CUDA
	int i = threadIdx.x + blockIdx.x * blockDim.x;
#else
	int i = 0;
#endif
	CALCULATE1D(i, 0, WStride)
	t_W[i] += beta2 * h_W[i];
	W[i] = m_W[i] + alpha3 * h_W[i];
	END_CALCULATE1D()
}

__GLOBAL__
void wave_pml_rk2(float *h_W, float *W, float *t_W, float *m_W, long long WStride)
{
#ifdef GPU_CUDA
	int i = threadIdx.x + blockIdx.x * blockDim.x;
#else
	int i = 0;
#endif
	CALCULATE1D(i, 0, WStride)
	t_W[i] += beta3 * h_W[i];
	W[i] = m_W[i] + h_W[i];
	END_CALCULATE1D()
}

__GLOBAL__
void wave_pml_rk3(float *h_W, float *W, float *t_W, float *m_W, long long WStride)
{
#ifdef GPU_CUDA
	int i = threadIdx.x + blockIdx.x * blockDim.x;
#else
	int i = 0;
#endif
	CALCULATE1D(i, 0, WStride)
	W[i] = t_W[i] + beta4 * h_W[i];
	END_CALCULATE1D()
}
void pmlRk(GRID grid, MPI_BORDER border, int irk, AUX6 Aux6)
{
	int nx = grid.nx;
	int ny = grid.ny;
	int nz = grid.nz;

	int nPML = grid.nPML;

	long long num = 0;

	WAVE_RK_FUNC pml_rk[4] = {wave_pml_rk0, wave_pml_rk1, wave_pml_rk2, wave_pml_rk3};
	long long numx = nPML * ny * nz * WSIZE;
	long long numy = nPML * nx * nz * WSIZE;
	long long numz = nPML * nx * ny * WSIZE;

#ifdef GPU_CUDA

	dim3 thread(1024, 1, 1);
	dim3 blockX;

	blockX.x = (numx + thread.x - 1) / thread.x;
	blockX.y = 1;
	blockX.z = 1;

	dim3 blockY;
	blockY.x = (numy + thread.x - 1) / thread.x;
	blockY.y = 1;
	blockY.z = 1;

	dim3 blockZ;
	blockZ.x = (numz + thread.x - 1) / thread.x;
	blockZ.y = 1;
	blockZ.z = 1;

	if (border.isx1)
		pml_rk[irk]<<<blockX, thread>>>(Aux6.Aux1x.h_Aux, Aux6.Aux1x.Aux, Aux6.Aux1x.t_Aux, Aux6.Aux1x.m_Aux, numx);
	if (border.isy1)
		pml_rk[irk]<<<blockY, thread>>>(Aux6.Aux1y.h_Aux, Aux6.Aux1y.Aux, Aux6.Aux1y.t_Aux, Aux6.Aux1y.m_Aux, numy);
	if (border.isz1)
		pml_rk[irk]<<<blockZ, thread>>>(Aux6.Aux1z.h_Aux, Aux6.Aux1z.Aux, Aux6.Aux1z.t_Aux, Aux6.Aux1z.m_Aux, numz);

	if (border.isx2)
		pml_rk[irk]<<<blockX, thread>>>(Aux6.Aux2x.h_Aux, Aux6.Aux2x.Aux, Aux6.Aux2x.t_Aux, Aux6.Aux2x.m_Aux, numx);
	if (border.isy2)
		pml_rk[irk]<<<blockY, thread>>>(Aux6.Aux2y.h_Aux, Aux6.Aux2y.Aux, Aux6.Aux2y.t_Aux, Aux6.Aux2y.m_Aux, numy);
		// CHECK( cudaDeviceSynchronize( ) );
#ifndef FREE_SURFACE
	if (border.isz2)
		pml_rk[irk]<<<blockZ, thread>>>(Aux6.Aux2z.h_Aux, Aux6.Aux2z.Aux, Aux6.Aux2z.t_Aux, Aux6.Aux2z.m_Aux, numz);
#endif

#else

	if (border.isx1)
		pml_rk[irk](Aux6.Aux1x.h_Aux, Aux6.Aux1x.Aux, Aux6.Aux1x.t_Aux, Aux6.Aux1x.m_Aux, numx);
	if (border.isy1)
		pml_rk[irk](Aux6.Aux1y.h_Aux, Aux6.Aux1y.Aux, Aux6.Aux1y.t_Aux, Aux6.Aux1y.m_Aux, numy);
	if (border.isz1)
		pml_rk[irk](Aux6.Aux1z.h_Aux, Aux6.Aux1z.Aux, Aux6.Aux1z.t_Aux, Aux6.Aux1z.m_Aux, numz);
	//
	if (border.isx2)
		pml_rk[irk](Aux6.Aux2x.h_Aux, Aux6.Aux2x.Aux, Aux6.Aux2x.t_Aux, Aux6.Aux2x.m_Aux, numx);
	if (border.isy2)
		pml_rk[irk](Aux6.Aux2y.h_Aux, Aux6.Aux2y.Aux, Aux6.Aux2y.t_Aux, Aux6.Aux2y.m_Aux, numy);
#ifndef FREE_SURFACE
	if (border.isz2)
		pml_rk[irk](Aux6.Aux2z.h_Aux, Aux6.Aux2z.Aux, Aux6.Aux2z.t_Aux, Aux6.Aux2z.m_Aux, numz);
	;
#endif

#endif
}
