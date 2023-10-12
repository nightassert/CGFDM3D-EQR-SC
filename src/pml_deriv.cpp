/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name:propagate.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time:2021-11-04
*   Discription:
*
================================================================*/
#include "header.h"

void allocAux(int nPML, int N1, int N2, AUX *Aux)
{
	long long num = nPML * N1 * N2;

	float *pAux = NULL;
	long long size = sizeof(float) * num * WSIZE * 4;

	CHECK(Malloc((void **)&pAux, size));

	if (pAux == NULL)
	{
		printf("can't allocate PML Auxiliry memory!\n");
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}
	CHECK(Memset(pAux, 0, size));

	Aux->h_Aux = pAux + 0 * WSIZE * num;
	Aux->Aux = pAux + 1 * WSIZE * num;
	Aux->t_Aux = pAux + 2 * WSIZE * num;
	Aux->m_Aux = pAux + 3 * WSIZE * num;
}

void freeAux(AUX Aux)
{
	Free(Aux.h_Aux);
}

void allocPML(GRID grid, AUX6SURF *Aux6, MPI_BORDER border)
{

	int nx = grid.nx;
	int ny = grid.ny;
	int nz = grid.nz;

	int nPML = grid.nPML;

	memset((void *)Aux6, 0, sizeof(AUX));

	if (border.isx1 && nPML >= nx)
	{
		printf("The PML layer(nPML) just bigger than nx(%d)\n", nPML, nx);
		MPI_Abort(MPI_COMM_WORLD, 130);
	}
	if (border.isy1 && nPML >= ny)
	{
		printf("The PML layer(nPML) just bigger than ny(%d)\n", nPML, ny);
		MPI_Abort(MPI_COMM_WORLD, 130);
	}
	if (border.isz1 && nPML >= nz)
	{
		printf("The PML layer(nPML) just bigger than nz(%d)\n", nPML, nz);
		MPI_Abort(MPI_COMM_WORLD, 130);
	}

	if (border.isx2 && nPML >= nx)
	{
		printf("The PML layer(nPML) just bigger than nx(%d)\n", nPML, nx);
		MPI_Abort(MPI_COMM_WORLD, 130);
	}
	if (border.isy2 && nPML >= ny)
	{
		printf("The PML layer(nPML) just bigger than ny(%d)\n", nPML, ny);
		MPI_Abort(MPI_COMM_WORLD, 130);
	}
	if (border.isz2 && nPML >= nz)
	{
		printf("The PML layer(nPML) just bigger than nz(%d)\n", nPML, nz);
		MPI_Abort(MPI_COMM_WORLD, 130);
	}

	if (border.isx1)
		allocAux(nPML, ny, nz, &(Aux6->Aux1x));
	if (border.isy1)
		allocAux(nPML, nx, nz, &(Aux6->Aux1y));
	if (border.isz1)
		allocAux(nPML, nx, ny, &(Aux6->Aux1z));

	if (border.isx2)
		allocAux(nPML, ny, nz, &(Aux6->Aux2x));
	if (border.isy2)
		allocAux(nPML, nx, nz, &(Aux6->Aux2y));

#ifndef FREE_SURFACE
	if (border.isz2)
		allocAux(nPML, nx, ny, &(Aux6->Aux2z));
#endif
}

void freePML(MPI_BORDER border, AUX6SURF Aux6)
{

	if (border.isx1)
		freeAux(Aux6.Aux1x);
	if (border.isy1)
		freeAux(Aux6.Aux1y);
	if (border.isz1)
		freeAux(Aux6.Aux1z);

	if (border.isx2)
		freeAux(Aux6.Aux2x);
	if (border.isy2)
		freeAux(Aux6.Aux2y);
#ifndef FREE_SURFACE
	if (border.isz2)
		freeAux(Aux6.Aux2z);
#endif
}

__GLOBAL__
void pml_deriv_x(
	FLOAT *h_W, FLOAT *W, float *h_Aux_x, float *Aux_x,
	FLOAT *CJM,
	float *pml_alpha_x, float *pml_beta_x, float *pml_d_x,
	int nPML, int _nx_, int _ny_, int _nz_, int FLAG, float rDH, int FB1, float DT)
{
#ifdef GPU_CUDA
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;

#else
	int i0 = 0;
	int j0 = 0;
	int k0 = 0;
#endif

	int i, j, k;
	long long index;
	long long pos;

	int nx = _nx_ - HALO - HALO;
	int ny = _ny_ - HALO - HALO;
	int nz = _nz_ - HALO - HALO;

	float mu = 0.0f;
	float lambda = 0.0f;
	float buoyancy = 0.0f;

	float beta_x = 0.0f;
	float d_x = 0.0f;
	float alpha_d_x = 0.0f;

	float xi_x = 0.0f;
	float xi_y = 0.0f;
	float xi_z = 0.0f;

	float Txx_xi = 0.0f;
	float Tyy_xi = 0.0f;
	float Txy_xi = 0.0f;
	float Txz_xi = 0.0f;
	float Tyz_xi = 0.0f;
	float Tzz_xi = 0.0f;
	float Vx_xi = 0.0f;
	float Vy_xi = 0.0f;
	float Vz_xi = 0.0f;

	float Vx1 = 0.0;
	float Vy1 = 0.0;
	float Vz1 = 0.0;
	float Txx1 = 0.0;
	float Tyy1 = 0.0;
	float Tzz1 = 0.0;
	float Txy1 = 0.0;
	float Txz1 = 0.0;
	float Tyz1 = 0.0;

	float h_WVx;
	float h_WVy;
	float h_WVz;
	float h_WTxx;
	float h_WTyy;
	float h_WTzz;
	float h_WTxy;
	float h_WTxz;
	float h_WTyz;

	int stride = FLAG * (nx - nPML);
	CALCULATE3D(i0, j0, k0, 0, nPML, 0, ny, 0, nz)
	i = i0 + HALO + stride;
	j = j0 + HALO;
	k = k0 + HALO;
	index = INDEX(i, j, k);
	pos = Index3D(i0, j0, k0, nPML, ny, nz); // i0 + j0 * nPML + k0 * nPML * ny;
	mu = CJM[index * CJMSIZE + 10];
	lambda = CJM[index * CJMSIZE + 11];
	buoyancy = CJM[index * CJMSIZE + 12];
	buoyancy *= Crho;

	beta_x = pml_beta_x[i];
	d_x = pml_d_x[i];
	alpha_d_x = d_x + pml_alpha_x[i];

	xi_x = CJM[index * CJMSIZE + 0];
	xi_y = CJM[index * CJMSIZE + 1];
	xi_z = CJM[index * CJMSIZE + 2];

	Vx_xi = L((float)W, 0, WSIZE, FB1, xi) * d_x;
	Vy_xi = L((float)W, 1, WSIZE, FB1, xi) * d_x;
	Vz_xi = L((float)W, 2, WSIZE, FB1, xi) * d_x;
	Txx_xi = L((float)W, 3, WSIZE, FB1, xi) * d_x;
	Tyy_xi = L((float)W, 4, WSIZE, FB1, xi) * d_x;
	Tzz_xi = L((float)W, 5, WSIZE, FB1, xi) * d_x;
	Txy_xi = L((float)W, 6, WSIZE, FB1, xi) * d_x;
	Txz_xi = L((float)W, 7, WSIZE, FB1, xi) * d_x;
	Tyz_xi = L((float)W, 8, WSIZE, FB1, xi) * d_x;

	Vx1 = DOT_PRODUCT3D(xi_x, xi_y, xi_z, Txx_xi, Txy_xi, Txz_xi) * buoyancy;
	Vy1 = DOT_PRODUCT3D(xi_x, xi_y, xi_z, Txy_xi, Tyy_xi, Tyz_xi) * buoyancy;
	Vz1 = DOT_PRODUCT3D(xi_x, xi_y, xi_z, Txz_xi, Tyz_xi, Tzz_xi) * buoyancy;

	Txx1 = DOT_PRODUCT3D(xi_x, xi_y, xi_z, Vx_xi, Vy_xi, Vz_xi) * lambda + 2.0f * mu * (xi_x * Vx_xi);
	Tyy1 = DOT_PRODUCT3D(xi_x, xi_y, xi_z, Vx_xi, Vy_xi, Vz_xi) * lambda + 2.0f * mu * (xi_y * Vy_xi);
	Tzz1 = DOT_PRODUCT3D(xi_x, xi_y, xi_z, Vx_xi, Vy_xi, Vz_xi) * lambda + 2.0f * mu * (xi_z * Vz_xi);

	Txy1 = DOT_PRODUCT2D(xi_y, xi_x, Vx_xi, Vy_xi) * mu;
	Txz1 = DOT_PRODUCT2D(xi_z, xi_x, Vx_xi, Vz_xi) * mu;
	Tyz1 = DOT_PRODUCT2D(xi_z, xi_y, Vy_xi, Vz_xi) * mu;

	h_Aux_x[pos * WSIZE + 0] = (Vx1 - alpha_d_x * Aux_x[pos * WSIZE + 0]) * DT;
	h_Aux_x[pos * WSIZE + 1] = (Vy1 - alpha_d_x * Aux_x[pos * WSIZE + 1]) * DT;
	h_Aux_x[pos * WSIZE + 2] = (Vz1 - alpha_d_x * Aux_x[pos * WSIZE + 2]) * DT;
	h_Aux_x[pos * WSIZE + 3] = (Txx1 - alpha_d_x * Aux_x[pos * WSIZE + 3]) * DT;
	h_Aux_x[pos * WSIZE + 4] = (Tyy1 - alpha_d_x * Aux_x[pos * WSIZE + 4]) * DT;
	h_Aux_x[pos * WSIZE + 5] = (Tzz1 - alpha_d_x * Aux_x[pos * WSIZE + 5]) * DT;
	h_Aux_x[pos * WSIZE + 6] = (Txy1 - alpha_d_x * Aux_x[pos * WSIZE + 6]) * DT;
	h_Aux_x[pos * WSIZE + 7] = (Txz1 - alpha_d_x * Aux_x[pos * WSIZE + 7]) * DT;
	h_Aux_x[pos * WSIZE + 8] = (Tyz1 - alpha_d_x * Aux_x[pos * WSIZE + 8]) * DT;

	h_WVx = h_W[index * WSIZE + 0];
	h_WVy = h_W[index * WSIZE + 1];
	h_WVz = h_W[index * WSIZE + 2];
	h_WTxx = h_W[index * WSIZE + 3];
	h_WTyy = h_W[index * WSIZE + 4];
	h_WTzz = h_W[index * WSIZE + 5];
	h_WTxy = h_W[index * WSIZE + 6];
	h_WTxz = h_W[index * WSIZE + 7];
	h_WTyz = h_W[index * WSIZE + 8];

	h_WVx += -beta_x * Aux_x[pos * WSIZE + 0] * DT;
	h_WVy += -beta_x * Aux_x[pos * WSIZE + 1] * DT;
	h_WVz += -beta_x * Aux_x[pos * WSIZE + 2] * DT;
	h_WTxx += -beta_x * Aux_x[pos * WSIZE + 3] * DT;
	h_WTyy += -beta_x * Aux_x[pos * WSIZE + 4] * DT;
	h_WTzz += -beta_x * Aux_x[pos * WSIZE + 5] * DT;
	h_WTxy += -beta_x * Aux_x[pos * WSIZE + 6] * DT;
	h_WTxz += -beta_x * Aux_x[pos * WSIZE + 7] * DT;
	h_WTyz += -beta_x * Aux_x[pos * WSIZE + 8] * DT;

	h_W[index * WSIZE + 0] = h_WVx;
	h_W[index * WSIZE + 1] = h_WVy;
	h_W[index * WSIZE + 2] = h_WVz;
	h_W[index * WSIZE + 3] = h_WTxx;
	h_W[index * WSIZE + 4] = h_WTyy;
	h_W[index * WSIZE + 5] = h_WTzz;
	h_W[index * WSIZE + 6] = h_WTxy;
	h_W[index * WSIZE + 7] = h_WTxz;
	h_W[index * WSIZE + 8] = h_WTyz;

	END_CALCULATE3D()
}

__GLOBAL__
void pml_deriv_y(
	FLOAT *h_W, FLOAT *W, float *h_Aux_y, float *Aux_y,
	FLOAT *CJM,
	float *pml_alpha_y, float *pml_beta_y, float *pml_d_y,
	int nPML, int _nx_, int _ny_, int _nz_, int FLAG, float rDH, int FB2, float DT)
{

#ifdef GPU_CUDA
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;
#else
	int i0 = 0;
	int j0 = 0;
	int k0 = 0;
#endif

	int i, j, k;
	long long index;
	long long pos;

	int nx = _nx_ - HALO - HALO;
	int ny = _ny_ - HALO - HALO;
	int nz = _nz_ - HALO - HALO;

	float mu = 0.0f;
	float lambda = 0.0f;
	float buoyancy = 0.0f;

	float beta_y = 0.0f;
	float d_y = 0.0f;
	float alpha_d_y = 0.0f;

	float et_x = 0.0f;
	float et_y = 0.0f;
	float et_z = 0.0f;

	float Txx_et = 0.0f;
	float Tyy_et = 0.0f;
	float Txy_et = 0.0f;
	float Txz_et = 0.0f;
	float Tyz_et = 0.0f;
	float Tzz_et = 0.0f;
	float Vx_et = 0.0f;
	float Vy_et = 0.0f;
	float Vz_et = 0.0f;

	float Vx2 = 0.0f;
	float Vy2 = 0.0f;
	float Vz2 = 0.0f;
	float Txx2 = 0.0f;
	float Tyy2 = 0.0f;
	float Tzz2 = 0.0f;
	float Txy2 = 0.0f;
	float Txz2 = 0.0f;
	float Tyz2 = 0.0f;

	float h_WVx;
	float h_WVy;
	float h_WVz;
	float h_WTxx;
	float h_WTyy;
	float h_WTzz;
	float h_WTxy;
	float h_WTxz;
	float h_WTyz;

	int stride = FLAG * (ny - nPML);
	CALCULATE3D(i0, j0, k0, 0, nx, 0, nPML, 0, nz)
	i = i0 + HALO;
	j = j0 + HALO + stride;
	k = k0 + HALO;
	index = INDEX(i, j, k);
	pos = Index3D(i0, j0, k0, nx, nPML, nz); // i0 + j0 * nx + k0 * nx * nPML;

	mu = CJM[index * CJMSIZE + 10];
	lambda = CJM[index * CJMSIZE + 11];
	buoyancy = CJM[index * CJMSIZE + 12];
	buoyancy *= Crho;

	beta_y = pml_beta_y[j];
	d_y = pml_d_y[j];
	alpha_d_y = d_y + pml_alpha_y[j];

	et_x = CJM[index * CJMSIZE + 3];
	et_y = CJM[index * CJMSIZE + 4];
	et_z = CJM[index * CJMSIZE + 5];

	Vx_et = L((float)W, 0, WSIZE, FB2, et) * d_y;
	Vy_et = L((float)W, 1, WSIZE, FB2, et) * d_y;
	Vz_et = L((float)W, 2, WSIZE, FB2, et) * d_y;
	Txx_et = L((float)W, 3, WSIZE, FB2, et) * d_y;
	Tyy_et = L((float)W, 4, WSIZE, FB2, et) * d_y;
	Tzz_et = L((float)W, 5, WSIZE, FB2, et) * d_y;
	Txy_et = L((float)W, 6, WSIZE, FB2, et) * d_y;
	Txz_et = L((float)W, 7, WSIZE, FB2, et) * d_y;
	Tyz_et = L((float)W, 8, WSIZE, FB2, et) * d_y;

	Vx2 = DOT_PRODUCT3D(et_x, et_y, et_z, Txx_et, Txy_et, Txz_et) * buoyancy;
	Vy2 = DOT_PRODUCT3D(et_x, et_y, et_z, Txy_et, Tyy_et, Tyz_et) * buoyancy;
	Vz2 = DOT_PRODUCT3D(et_x, et_y, et_z, Txz_et, Tyz_et, Tzz_et) * buoyancy;

	Txx2 = DOT_PRODUCT3D(et_x, et_y, et_z, Vx_et, Vy_et, Vz_et) * lambda + 2.0f * mu * (et_x * Vx_et);
	Tyy2 = DOT_PRODUCT3D(et_x, et_y, et_z, Vx_et, Vy_et, Vz_et) * lambda + 2.0f * mu * (et_y * Vy_et);
	Tzz2 = DOT_PRODUCT3D(et_x, et_y, et_z, Vx_et, Vy_et, Vz_et) * lambda + 2.0f * mu * (et_z * Vz_et);

	Txy2 = DOT_PRODUCT2D(et_y, et_x, Vx_et, Vy_et) * mu;
	Txz2 = DOT_PRODUCT2D(et_z, et_x, Vx_et, Vz_et) * mu;
	Tyz2 = DOT_PRODUCT2D(et_z, et_y, Vy_et, Vz_et) * mu;

	h_Aux_y[pos * WSIZE + 0] = (Vx2 - alpha_d_y * Aux_y[pos * WSIZE + 0]) * DT;
	h_Aux_y[pos * WSIZE + 1] = (Vy2 - alpha_d_y * Aux_y[pos * WSIZE + 1]) * DT;
	h_Aux_y[pos * WSIZE + 2] = (Vz2 - alpha_d_y * Aux_y[pos * WSIZE + 2]) * DT;
	h_Aux_y[pos * WSIZE + 3] = (Txx2 - alpha_d_y * Aux_y[pos * WSIZE + 3]) * DT;
	h_Aux_y[pos * WSIZE + 4] = (Tyy2 - alpha_d_y * Aux_y[pos * WSIZE + 4]) * DT;
	h_Aux_y[pos * WSIZE + 5] = (Tzz2 - alpha_d_y * Aux_y[pos * WSIZE + 5]) * DT;
	h_Aux_y[pos * WSIZE + 6] = (Txy2 - alpha_d_y * Aux_y[pos * WSIZE + 6]) * DT;
	h_Aux_y[pos * WSIZE + 7] = (Txz2 - alpha_d_y * Aux_y[pos * WSIZE + 7]) * DT;
	h_Aux_y[pos * WSIZE + 8] = (Tyz2 - alpha_d_y * Aux_y[pos * WSIZE + 8]) * DT;

	h_WVx = h_W[index * WSIZE + 0];
	h_WVy = h_W[index * WSIZE + 1];
	h_WVz = h_W[index * WSIZE + 2];
	h_WTxx = h_W[index * WSIZE + 3];
	h_WTyy = h_W[index * WSIZE + 4];
	h_WTzz = h_W[index * WSIZE + 5];
	h_WTxy = h_W[index * WSIZE + 6];
	h_WTxz = h_W[index * WSIZE + 7];
	h_WTyz = h_W[index * WSIZE + 8];

	h_WVx += -beta_y * Aux_y[pos * WSIZE + 0] * DT;
	h_WVy += -beta_y * Aux_y[pos * WSIZE + 1] * DT;
	h_WVz += -beta_y * Aux_y[pos * WSIZE + 2] * DT;
	h_WTxx += -beta_y * Aux_y[pos * WSIZE + 3] * DT;
	h_WTyy += -beta_y * Aux_y[pos * WSIZE + 4] * DT;
	h_WTzz += -beta_y * Aux_y[pos * WSIZE + 5] * DT;
	h_WTxy += -beta_y * Aux_y[pos * WSIZE + 6] * DT;
	h_WTxz += -beta_y * Aux_y[pos * WSIZE + 7] * DT;
	h_WTyz += -beta_y * Aux_y[pos * WSIZE + 8] * DT;

	h_W[index * WSIZE + 0] = h_WVx;
	h_W[index * WSIZE + 1] = h_WVy;
	h_W[index * WSIZE + 2] = h_WVz;
	h_W[index * WSIZE + 3] = h_WTxx;
	h_W[index * WSIZE + 4] = h_WTyy;
	h_W[index * WSIZE + 5] = h_WTzz;
	h_W[index * WSIZE + 6] = h_WTxy;
	h_W[index * WSIZE + 7] = h_WTxz;
	h_W[index * WSIZE + 8] = h_WTyz;

	END_CALCULATE3D()
}

__GLOBAL__
void pml_deriv_z(
	FLOAT *h_W, FLOAT *W, float *h_Aux_z, float *Aux_z,
	FLOAT *CJM,
	float *pml_alpha_z, float *pml_beta_z, float *pml_d_z,
	int nPML, int _nx_, int _ny_, int _nz_, int FLAG, float rDH, int FB3, float DT)
{

#ifdef GPU_CUDA
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;
#else
	int i0 = 0;
	int j0 = 0;
	int k0 = 0;
#endif

	int i, j, k;
	long long index;
	long long pos;

	int nx = _nx_ - HALO - HALO;
	int ny = _ny_ - HALO - HALO;
	int nz = _nz_ - HALO - HALO;

	float mu = 0.0f;
	float lambda = 0.0f;
	float buoyancy = 0.0f;

	float beta_z = 0.0f;
	float d_z = 0.0f;
	float alpha_d_z = 0.0f;

	float zt_x = 0.0f;
	float zt_y = 0.0f;
	float zt_z = 0.0f;

	float Txx_zt = 0.0f;
	float Tyy_zt = 0.0f;
	float Txy_zt = 0.0f;
	float Txz_zt = 0.0f;
	float Tyz_zt = 0.0f;
	float Tzz_zt = 0.0f;
	float Vx_zt = 0.0f;
	float Vy_zt = 0.0f;
	float Vz_zt = 0.0f;

	float Vx3 = 0.0f;
	float Vy3 = 0.0f;
	float Vz3 = 0.0f;
	float Txx3 = 0.0f;
	float Tyy3 = 0.0f;
	float Tzz3 = 0.0f;
	float Txy3 = 0.0f;
	float Txz3 = 0.0f;
	float Tyz3 = 0.0f;

	float h_WVx;
	float h_WVy;
	float h_WVz;
	float h_WTxx;
	float h_WTyy;
	float h_WTzz;
	float h_WTxy;
	float h_WTxz;
	float h_WTyz;

	int stride = FLAG * (nz - nPML);
	CALCULATE3D(i0, j0, k0, 0, nx, 0, ny, 0, nPML)
	i = i0 + HALO;
	j = j0 + HALO;
	k = k0 + HALO + stride;
	index = INDEX(i, j, k);
	pos = Index3D(i0, j0, k0, nx, ny, nPML); // i0 + j0 * nx + k0 * nx * ny;

	mu = CJM[index * CJMSIZE + 10];
	lambda = CJM[index * CJMSIZE + 11];
	buoyancy = CJM[index * CJMSIZE + 12];
	buoyancy *= Crho;

	beta_z = pml_beta_z[k];
	d_z = pml_d_z[k];
	alpha_d_z = d_z + pml_alpha_z[k];

	zt_x = CJM[index * CJMSIZE + 6];
	zt_y = CJM[index * CJMSIZE + 7];
	zt_z = CJM[index * CJMSIZE + 8];

	Vx_zt = L((float)W, 0, WSIZE, FB3, zt) * d_z;
	Vy_zt = L((float)W, 1, WSIZE, FB3, zt) * d_z;
	Vz_zt = L((float)W, 2, WSIZE, FB3, zt) * d_z;
	Txx_zt = L((float)W, 3, WSIZE, FB3, zt) * d_z;
	Tyy_zt = L((float)W, 4, WSIZE, FB3, zt) * d_z;
	Tzz_zt = L((float)W, 5, WSIZE, FB3, zt) * d_z;
	Txy_zt = L((float)W, 6, WSIZE, FB3, zt) * d_z;
	Txz_zt = L((float)W, 7, WSIZE, FB3, zt) * d_z;
	Tyz_zt = L((float)W, 8, WSIZE, FB3, zt) * d_z;

	Vx3 = DOT_PRODUCT3D(zt_x, zt_y, zt_z, Txx_zt, Txy_zt, Txz_zt) * buoyancy;
	Vy3 = DOT_PRODUCT3D(zt_x, zt_y, zt_z, Txy_zt, Tyy_zt, Tyz_zt) * buoyancy;
	Vz3 = DOT_PRODUCT3D(zt_x, zt_y, zt_z, Txz_zt, Tyz_zt, Tzz_zt) * buoyancy;

	Txx3 = DOT_PRODUCT3D(zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt) * lambda + 2.0f * mu * (zt_x * Vx_zt);
	Tyy3 = DOT_PRODUCT3D(zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt) * lambda + 2.0f * mu * (zt_y * Vy_zt);
	Tzz3 = DOT_PRODUCT3D(zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt) * lambda + 2.0f * mu * (zt_z * Vz_zt);

	Txy3 = DOT_PRODUCT2D(zt_y, zt_x, Vx_zt, Vy_zt) * mu;
	Txz3 = DOT_PRODUCT2D(zt_z, zt_x, Vx_zt, Vz_zt) * mu;
	Tyz3 = DOT_PRODUCT2D(zt_z, zt_y, Vy_zt, Vz_zt) * mu;

	h_Aux_z[pos * WSIZE + 0] = (Vx3 - alpha_d_z * Aux_z[pos * WSIZE + 0]) * DT;
	h_Aux_z[pos * WSIZE + 1] = (Vy3 - alpha_d_z * Aux_z[pos * WSIZE + 1]) * DT;
	h_Aux_z[pos * WSIZE + 2] = (Vz3 - alpha_d_z * Aux_z[pos * WSIZE + 2]) * DT;
	h_Aux_z[pos * WSIZE + 3] = (Txx3 - alpha_d_z * Aux_z[pos * WSIZE + 3]) * DT;
	h_Aux_z[pos * WSIZE + 4] = (Tyy3 - alpha_d_z * Aux_z[pos * WSIZE + 4]) * DT;
	h_Aux_z[pos * WSIZE + 5] = (Tzz3 - alpha_d_z * Aux_z[pos * WSIZE + 5]) * DT;
	h_Aux_z[pos * WSIZE + 6] = (Txy3 - alpha_d_z * Aux_z[pos * WSIZE + 6]) * DT;
	h_Aux_z[pos * WSIZE + 7] = (Txz3 - alpha_d_z * Aux_z[pos * WSIZE + 7]) * DT;
	h_Aux_z[pos * WSIZE + 8] = (Tyz3 - alpha_d_z * Aux_z[pos * WSIZE + 8]) * DT;

	h_WVx = h_W[index * WSIZE + 0];
	h_WVy = h_W[index * WSIZE + 1];
	h_WVz = h_W[index * WSIZE + 2];
	h_WTxx = h_W[index * WSIZE + 3];
	h_WTyy = h_W[index * WSIZE + 4];
	h_WTzz = h_W[index * WSIZE + 5];
	h_WTxy = h_W[index * WSIZE + 6];
	h_WTxz = h_W[index * WSIZE + 7];
	h_WTyz = h_W[index * WSIZE + 8];

	h_WVx += -beta_z * Aux_z[pos * WSIZE + 0] * DT;
	h_WVy += -beta_z * Aux_z[pos * WSIZE + 1] * DT;
	h_WVz += -beta_z * Aux_z[pos * WSIZE + 2] * DT;
	h_WTxx += -beta_z * Aux_z[pos * WSIZE + 3] * DT;
	h_WTyy += -beta_z * Aux_z[pos * WSIZE + 4] * DT;
	h_WTzz += -beta_z * Aux_z[pos * WSIZE + 5] * DT;
	h_WTxy += -beta_z * Aux_z[pos * WSIZE + 6] * DT;
	h_WTxz += -beta_z * Aux_z[pos * WSIZE + 7] * DT;
	h_WTyz += -beta_z * Aux_z[pos * WSIZE + 8] * DT;

	h_W[index * WSIZE + 0] = h_WVx;
	h_W[index * WSIZE + 1] = h_WVy;
	h_W[index * WSIZE + 2] = h_WVz;
	h_W[index * WSIZE + 3] = h_WTxx;
	h_W[index * WSIZE + 4] = h_WTyy;
	h_W[index * WSIZE + 5] = h_WTzz;
	h_W[index * WSIZE + 6] = h_WTxy;
	h_W[index * WSIZE + 7] = h_WTxz;
	h_W[index * WSIZE + 8] = h_WTyz;

	END_CALCULATE3D()
}

void pmlDeriv(GRID grid, WAVE wave, FLOAT *CJM, AUX6 Aux6, PML_ALPHA pml_alpha,
			  PML_BETA pml_beta, PML_D pml_d, MPI_BORDER border, int FB1, int FB2, int FB3, float DT)

{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	float rDH = grid.rDH;

	int nPML = grid.nPML;

	float *pml_alpha_x = pml_alpha.x;
	float *pml_beta_x = pml_beta.x;
	float *pml_d_x = pml_d.x;
	float *pml_alpha_y = pml_alpha.y;
	float *pml_beta_y = pml_beta.y;
	float *pml_d_y = pml_d.y;
	float *pml_alpha_z = pml_alpha.z;
	float *pml_beta_z = pml_beta.z;
	float *pml_d_z = pml_d.z;

	FLOAT *h_W = wave.h_W;
	FLOAT *W = wave.W;

#ifdef GPU_CUDA

	int nx = _nx_ - 2 * HALO;
	int ny = _ny_ - 2 * HALO;
	int nz = _nz_ - 2 * HALO;

	dim3 thread(16, 16, 1);
	dim3 blockX;
	blockX.x = (nPML + thread.x - 1) / thread.x;
	blockX.y = (ny + thread.y - 1) / thread.y;
	blockX.z = (nz + thread.z - 1) / thread.z;

	dim3 blockY;
	blockY.x = (nx + thread.x - 1) / thread.x;
	blockY.y = (nPML + thread.y - 1) / thread.y;
	blockY.z = (nz + thread.z - 1) / thread.z;

	dim3 blockZ;
	blockZ.x = (nx + thread.x - 1) / thread.x;
	blockZ.y = (ny + thread.y - 1) / thread.y;
	blockZ.z = (nPML + thread.z - 1) / thread.z;

	if (border.isx1)
		pml_deriv_x<<<blockX, thread>>>(h_W, W, Aux6.Aux1x.h_Aux, Aux6.Aux1x.Aux, CJM, pml_alpha_x, pml_beta_x, pml_d_x, nPML, _nx_, _ny_, _nz_, 0, rDH, FB1, DT);
	if (border.isy1)
		pml_deriv_y<<<blockY, thread>>>(h_W, W, Aux6.Aux1y.h_Aux, Aux6.Aux1y.Aux, CJM, pml_alpha_y, pml_beta_y, pml_d_y, nPML, _nx_, _ny_, _nz_, 0, rDH, FB2, DT);
	if (border.isz1)
		pml_deriv_z<<<blockZ, thread>>>(h_W, W, Aux6.Aux1z.h_Aux, Aux6.Aux1z.Aux, CJM, pml_alpha_z, pml_beta_z, pml_d_z, nPML, _nx_, _ny_, _nz_, 0, rDH, FB3, DT);

	if (border.isx2)
		pml_deriv_x<<<blockX, thread>>>(h_W, W, Aux6.Aux2x.h_Aux, Aux6.Aux2x.Aux, CJM, pml_alpha_x, pml_beta_x, pml_d_x, nPML, _nx_, _ny_, _nz_, 1, rDH, FB1, DT);
	if (border.isy2)
		pml_deriv_y<<<blockY, thread>>>(h_W, W, Aux6.Aux2y.h_Aux, Aux6.Aux2y.Aux, CJM, pml_alpha_y, pml_beta_y, pml_d_y, nPML, _nx_, _ny_, _nz_, 1, rDH, FB2, DT);
#ifndef FREE_SURFACE
	if (border.isz2)
		pml_deriv_z<<<blockZ, thread>>>(h_W, W, Aux6.Aux2z.h_Aux, Aux6.Aux2z.Aux, CJM, pml_alpha_z, pml_beta_z, pml_d_z, nPML, _nx_, _ny_, _nz_, 1, rDH, FB3, DT);
#endif

	CHECK(cudaDeviceSynchronize());

#else

	if (border.isx1)
		pml_deriv_x(h_W, W, Aux6.Aux1x.h_Aux, Aux6.Aux1x.Aux, CJM, pml_alpha_x, pml_beta_x, pml_d_x, nPML, _nx_, _ny_, _nz_, 0, rDH, FB1, DT);
	if (border.isy1)
		pml_deriv_y(h_W, W, Aux6.Aux1y.h_Aux, Aux6.Aux1y.Aux, CJM, pml_alpha_y, pml_beta_y, pml_d_y, nPML, _nx_, _ny_, _nz_, 0, rDH, FB2, DT);
	if (border.isz1)
		pml_deriv_z(h_W, W, Aux6.Aux1z.h_Aux, Aux6.Aux1z.Aux, CJM, pml_alpha_z, pml_beta_z, pml_d_z, nPML, _nx_, _ny_, _nz_, 0, rDH, FB3, DT);

	if (border.isx2)
		pml_deriv_x(h_W, W, Aux6.Aux2x.h_Aux, Aux6.Aux2x.Aux, CJM, pml_alpha_x, pml_beta_x, pml_d_x, nPML, _nx_, _ny_, _nz_, 1, rDH, FB1, DT);
	if (border.isy2)
		pml_deriv_y(h_W, W, Aux6.Aux2y.h_Aux, Aux6.Aux2y.Aux, CJM, pml_alpha_y, pml_beta_y, pml_d_y, nPML, _nx_, _ny_, _nz_, 1, rDH, FB2, DT);
#ifndef FREE_SURFACE
	if (border.isz2)
		pml_deriv_z(h_W, W, Aux6.Aux2z.h_Aux, Aux6.Aux2z.Aux, CJM, pml_alpha_z, pml_beta_z, pml_d_z, nPML, _nx_, _ny_, _nz_, 1, rDH, FB3, DT);
#endif

#endif // GPU_CUDA
}
