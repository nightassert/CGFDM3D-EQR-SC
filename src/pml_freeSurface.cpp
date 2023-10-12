/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name:pml_freeSurface.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time:2021-11-17
*   Discription:
*
================================================================*/

#include "header.h"
__GLOBAL__
void pml_free_surface_x(
	FLOAT *h_W, FLOAT *W, float *h_Aux_x, float *Aux_x,
	FLOAT *CJM, float *_rDZ_DX, float *_rDZ_DY,
	float *pml_d_x, int nPML,
	int _nx_, int _ny_, int _nz_, int FLAG, float rDH, int FB1, float DT)
{
#ifdef GPU_CUDA
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
#else
	int i0 = 0;
	int j0 = 0;
#endif

	int k0;
	int i, j, k;
	long long index;
	long long pos;

	int nx = _nx_ - HALO - HALO;
	int ny = _ny_ - HALO - HALO;
	int nz = _nz_ - HALO - HALO;

	int indexOnSurf;
	float mu = 0.0f;
	float lambda = 0.0f;

	float d_x = 0.0f;

	float Vx_xi = 0.0f;
	float Vy_xi = 0.0f;
	float Vz_xi = 0.0f;
	float Vx_zt = 0.0f;
	float Vy_zt = 0.0f;
	float Vz_zt = 0.0f;
	float zt_x = 0.0f;
	float zt_y = 0.0f;
	float zt_z = 0.0f;

	float Txx3 = 0.0;
	float Tyy3 = 0.0;
	float Tzz3 = 0.0;
	float Txy3 = 0.0;
	float Txz3 = 0.0;
	float Tyz3 = 0.0;

	float h_Aux_xTxx;
	float h_Aux_xTyy;
	float h_Aux_xTzz;
	float h_Aux_xTxy;
	float h_Aux_xTxz;
	float h_Aux_xTyz;

	int stride = FLAG * (nx - nPML);
	k0 = nz - 1;
	CALCULATE2D(i0, j0, 0, nPML, 0, ny)
	i = i0 + HALO + stride;
	j = j0 + HALO;
	k = k0 + HALO;
	index = INDEX(i, j, k);
	indexOnSurf = Index2D(i, j, _nx_, _ny_);
	// pos	= i0 + j0 * nPML + k0 * nPML * ny;
	pos = Index3D(i0, j0, k0, nPML, ny, nz); // i0 + j0 * nPML + k0 * nPML * ny;

	mu = CJM[index * CJMSIZE + 10];
	lambda = CJM[index * CJMSIZE + 11];

	d_x = pml_d_x[i];

	zt_x = CJM[index * CJMSIZE + 6];
	zt_y = CJM[index * CJMSIZE + 7];
	zt_z = CJM[index * CJMSIZE + 8];

	Vx_xi = L((float)W, 0, WSIZE, FB1, xi) * d_x;
	Vy_xi = L((float)W, 1, WSIZE, FB1, xi) * d_x;
	Vz_xi = L((float)W, 2, WSIZE, FB1, xi) * d_x;

	Vx_zt = DOT_PRODUCT3D(_rDZ_DX[indexOnSurf * MATSIZE + 0], _rDZ_DX[indexOnSurf * MATSIZE + 1], _rDZ_DX[indexOnSurf * MATSIZE + 2], Vx_xi, Vy_xi, Vz_xi);
	Vy_zt = DOT_PRODUCT3D(_rDZ_DX[indexOnSurf * MATSIZE + 3], _rDZ_DX[indexOnSurf * MATSIZE + 4], _rDZ_DX[indexOnSurf * MATSIZE + 5], Vx_xi, Vy_xi, Vz_xi);
	Vz_zt = DOT_PRODUCT3D(_rDZ_DX[indexOnSurf * MATSIZE + 6], _rDZ_DX[indexOnSurf * MATSIZE + 7], _rDZ_DX[indexOnSurf * MATSIZE + 8], Vx_xi, Vy_xi, Vz_xi);

	Txx3 = DOT_PRODUCT3D(zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt) * lambda + 2.0f * mu * (zt_x * Vx_zt);
	Tyy3 = DOT_PRODUCT3D(zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt) * lambda + 2.0f * mu * (zt_y * Vy_zt);
	Tzz3 = DOT_PRODUCT3D(zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt) * lambda + 2.0f * mu * (zt_z * Vz_zt);

	Txy3 = DOT_PRODUCT2D(zt_y, zt_x, Vx_zt, Vy_zt) * mu;
	Txz3 = DOT_PRODUCT2D(zt_z, zt_x, Vx_zt, Vz_zt) * mu;
	Tyz3 = DOT_PRODUCT2D(zt_z, zt_y, Vy_zt, Vz_zt) * mu;

	h_Aux_xTxx = h_Aux_x[pos * WSIZE + 3];
	h_Aux_xTyy = h_Aux_x[pos * WSIZE + 4];
	h_Aux_xTzz = h_Aux_x[pos * WSIZE + 5];
	h_Aux_xTxy = h_Aux_x[pos * WSIZE + 6];
	h_Aux_xTxz = h_Aux_x[pos * WSIZE + 7];
	h_Aux_xTyz = h_Aux_x[pos * WSIZE + 8];

	h_Aux_xTxx += Txx3 * DT;
	h_Aux_xTyy += Tyy3 * DT;
	h_Aux_xTzz += Tzz3 * DT;
	h_Aux_xTxy += Txy3 * DT;
	h_Aux_xTxz += Txz3 * DT;
	h_Aux_xTyz += Tyz3 * DT;

	h_Aux_x[pos * WSIZE + 3] = h_Aux_xTxx;
	h_Aux_x[pos * WSIZE + 4] = h_Aux_xTyy;
	h_Aux_x[pos * WSIZE + 5] = h_Aux_xTzz;
	h_Aux_x[pos * WSIZE + 6] = h_Aux_xTxy;
	h_Aux_x[pos * WSIZE + 7] = h_Aux_xTxz;
	h_Aux_x[pos * WSIZE + 8] = h_Aux_xTyz;

	END_CALCULATE2D()
}

__GLOBAL__
void pml_free_surface_y(
	FLOAT *h_W, FLOAT *W, float *h_Aux_y, float *Aux_y,
	FLOAT *CJM, float *_rDZ_DX, float *_rDZ_DY,
	float *pml_d_y, int nPML,
	int _nx_, int _ny_, int _nz_, int FLAG, float rDH, int FB2, float DT)
{
#ifdef GPU_CUDA
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
#else
	int i0 = 0;
	int j0 = 0;
#endif

	int k0;
	int i, j, k;
	long long index;
	long long pos;

	int nx = _nx_ - HALO - HALO;
	int ny = _ny_ - HALO - HALO;
	int nz = _nz_ - HALO - HALO;

	int indexOnSurf;
	float mu = 0.0f;
	float lambda = 0.0f;

	float d_y = 0.0f;

	float Vx_et = 0.0f;
	float Vy_et = 0.0f;
	float Vz_et = 0.0f;
	float Vx_zt = 0.0f;
	float Vy_zt = 0.0f;
	float Vz_zt = 0.0f;
	float zt_x = 0.0f;
	float zt_y = 0.0f;
	float zt_z = 0.0f;

	float Txx3 = 0.0f;
	float Tyy3 = 0.0f;
	float Tzz3 = 0.0f;
	float Txy3 = 0.0f;
	float Txz3 = 0.0f;
	float Tyz3 = 0.0f;

	float h_Aux_yTxx;
	float h_Aux_yTyy;
	float h_Aux_yTzz;
	float h_Aux_yTxy;
	float h_Aux_yTxz;
	float h_Aux_yTyz;

	int stride = FLAG * (ny - nPML);
	k0 = nz - 1;
	CALCULATE2D(i0, j0, 0, nx, 0, nPML)
	i = i0 + HALO;
	j = j0 + HALO + stride;
	k = k0 + HALO;
	index = INDEX(i, j, k);
	indexOnSurf = Index2D(i, j, _nx_, _ny_);
	// pos	= i0 + j0 * nx + k0 * nx * nPML;
	pos = Index3D(i0, j0, k0, nx, nPML, nz); // i0 + j0 * nx + k0 * nx * nPML;

	mu = CJM[index * CJMSIZE + 10];
	lambda = CJM[index * CJMSIZE + 11];

	d_y = pml_d_y[j];

	zt_x = CJM[index * CJMSIZE + 6];
	zt_y = CJM[index * CJMSIZE + 7];
	zt_z = CJM[index * CJMSIZE + 8];

	Vx_et = L((float)W, 0, WSIZE, FB2, et) * d_y;
	Vy_et = L((float)W, 1, WSIZE, FB2, et) * d_y;
	Vz_et = L((float)W, 2, WSIZE, FB2, et) * d_y;

	Vx_zt = DOT_PRODUCT3D(_rDZ_DY[indexOnSurf * MATSIZE + 0], _rDZ_DY[indexOnSurf * MATSIZE + 1], _rDZ_DY[indexOnSurf * MATSIZE + 2], Vx_et, Vy_et, Vz_et);
	Vy_zt = DOT_PRODUCT3D(_rDZ_DY[indexOnSurf * MATSIZE + 3], _rDZ_DY[indexOnSurf * MATSIZE + 4], _rDZ_DY[indexOnSurf * MATSIZE + 5], Vx_et, Vy_et, Vz_et);
	Vz_zt = DOT_PRODUCT3D(_rDZ_DY[indexOnSurf * MATSIZE + 6], _rDZ_DY[indexOnSurf * MATSIZE + 7], _rDZ_DY[indexOnSurf * MATSIZE + 8], Vx_et, Vy_et, Vz_et);

	Txx3 = DOT_PRODUCT3D(zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt) * lambda + 2.0f * mu * (zt_x * Vx_zt);
	Tyy3 = DOT_PRODUCT3D(zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt) * lambda + 2.0f * mu * (zt_y * Vy_zt);
	Tzz3 = DOT_PRODUCT3D(zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt) * lambda + 2.0f * mu * (zt_z * Vz_zt);

	Txy3 = DOT_PRODUCT2D(zt_y, zt_x, Vx_zt, Vy_zt) * mu;
	Txz3 = DOT_PRODUCT2D(zt_z, zt_x, Vx_zt, Vz_zt) * mu;
	Tyz3 = DOT_PRODUCT2D(zt_z, zt_y, Vy_zt, Vz_zt) * mu;

	h_Aux_yTxx = h_Aux_y[pos * WSIZE + 3];
	h_Aux_yTyy = h_Aux_y[pos * WSIZE + 4];
	h_Aux_yTzz = h_Aux_y[pos * WSIZE + 5];
	h_Aux_yTxy = h_Aux_y[pos * WSIZE + 6];
	h_Aux_yTxz = h_Aux_y[pos * WSIZE + 7];
	h_Aux_yTyz = h_Aux_y[pos * WSIZE + 8];

	h_Aux_yTxx += Txx3 * DT;
	h_Aux_yTyy += Tyy3 * DT;
	h_Aux_yTzz += Tzz3 * DT;
	h_Aux_yTxy += Txy3 * DT;
	h_Aux_yTxz += Txz3 * DT;
	h_Aux_yTyz += Tyz3 * DT;

	h_Aux_y[pos * WSIZE + 3] = h_Aux_yTxx;
	h_Aux_y[pos * WSIZE + 4] = h_Aux_yTyy;
	h_Aux_y[pos * WSIZE + 5] = h_Aux_yTzz;
	h_Aux_y[pos * WSIZE + 6] = h_Aux_yTxy;
	h_Aux_y[pos * WSIZE + 7] = h_Aux_yTxz;
	h_Aux_y[pos * WSIZE + 8] = h_Aux_yTyz;

	END_CALCULATE2D()
}

void pmlFreeSurfaceDeriv(GRID grid, WAVE wave,
						 FLOAT *CJM, AUX6 Aux6, Mat_rDZ mat_rDZ,
						 PML_D pml_d, MPI_BORDER border, int FB1, int FB2, float DT)
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	int nPML = grid.nPML;

	float rDH = grid.rDH;

	float *pml_d_x = pml_d.x;
	float *pml_d_y = pml_d.y;

	float *_rDZ_DX = mat_rDZ._rDZ_DX;
	float *_rDZ_DY = mat_rDZ._rDZ_DY;

	FLOAT *h_W = wave.h_W;
	FLOAT *W = wave.W;

#ifdef GPU_CUDA
	int nx = _nx_ - 2 * HALO;
	int ny = _ny_ - 2 * HALO;
	int nz = _nz_ - 2 * HALO;

	dim3 thread(8, 8, 1);
	dim3 blockX;
	blockX.x = (nPML + thread.x - 1) / thread.x;
	blockX.y = (ny + thread.y - 1) / thread.y;
	blockX.z = 1;

	dim3 blockY;
	blockY.x = (nx + thread.x - 1) / thread.x;
	blockY.y = (nPML + thread.y - 1) / thread.y;
	blockY.z = 1;

	if (border.isx1)
		pml_free_surface_x<<<blockX, thread>>>(h_W, W,
											   Aux6.Aux1x.h_Aux, Aux6.Aux1x.Aux,
											   CJM, _rDZ_DX, _rDZ_DY,
											   pml_d_x, nPML,
											   _nx_, _ny_, _nz_, 0, rDH, FB1, DT);

	if (border.isy1)
		pml_free_surface_y<<<blockY, thread>>>(h_W, W,
											   Aux6.Aux1y.h_Aux, Aux6.Aux1y.Aux,
											   CJM, _rDZ_DX, _rDZ_DY,
											   pml_d_y, nPML,
											   _nx_, _ny_, _nz_, 0, rDH, FB2, DT);

	if (border.isx2)
		pml_free_surface_x<<<blockX, thread>>>(h_W, W,
											   Aux6.Aux2x.h_Aux, Aux6.Aux2x.Aux,
											   CJM, _rDZ_DX, _rDZ_DY,
											   pml_d_x, nPML,
											   _nx_, _ny_, _nz_, 1, rDH, FB1, DT);

	if (border.isy2)
		pml_free_surface_y<<<blockY, thread>>>(h_W, W,
											   Aux6.Aux2y.h_Aux, Aux6.Aux2y.Aux,
											   CJM, _rDZ_DX, _rDZ_DY,
											   pml_d_y, nPML,
											   _nx_, _ny_, _nz_, 1, rDH, FB2, DT);
#else
	if (border.isx1)
		pml_free_surface_x(h_W, W,
						   Aux6.Aux1x.h_Aux, Aux6.Aux1x.Aux,
						   CJM, _rDZ_DX, _rDZ_DY,
						   pml_d_x, nPML,
						   _nx_, _ny_, _nz_, 0, rDH, FB1, DT);

	if (border.isy1)
		pml_free_surface_y(h_W, W,
						   Aux6.Aux1y.h_Aux, Aux6.Aux1y.Aux,
						   CJM, _rDZ_DX, _rDZ_DY,
						   pml_d_y, nPML,
						   _nx_, _ny_, _nz_, 0, rDH, FB2, DT);

	if (border.isx2)
		pml_free_surface_x(h_W, W,
						   Aux6.Aux2x.h_Aux, Aux6.Aux2x.Aux,
						   CJM, _rDZ_DX, _rDZ_DY,
						   pml_d_x, nPML,
						   _nx_, _ny_, _nz_, 1, rDH, FB1, DT);

	if (border.isy2)
		pml_free_surface_y(h_W, W,
						   Aux6.Aux2y.h_Aux, Aux6.Aux2y.Aux,
						   CJM, _rDZ_DX, _rDZ_DY,
						   pml_d_y, nPML,
						   _nx_, _ny_, _nz_, 1, rDH, FB2, DT);

#endif // GPU_CUDA
}
