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
#ifdef PML
#define times_pml_beta_x *pml_beta_x
#define times_pml_beta_y *pml_beta_y
#define times_pml_beta_z *pml_beta_z
#else
#define times_pml_beta_x
#define times_pml_beta_y
#define times_pml_beta_z
#endif

#define extrap3(k1, k2, k3, k4) (4.0f * k1 - 6.0f * k2 + 4.0f * k3 - k4)

__GLOBAL__
void char_free_surface_deriv(
	FLOAT *h_W, FLOAT *W, FLOAT *CJM, Mat_rDZ mat_rDZ,
#ifdef PML
	PML_BETA pml_beta,
#endif
	int _nx_, int _ny_, int _nz_, float rDH, int FB1, int FB2, int FB3, float DT)
{
	int _nx = _nx_ - HALO;
	int _ny = _ny_ - HALO;
	int _nz = _nz_ - HALO;

#ifdef GPU_CUDA
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = 0;
#else
	// int i = HALO;
	// int j = HALO;
	// int k = _nz - HALO;
#endif

	long long index;

	float mu = 0.0f;
	float lambda = 0.0f;
	float buoyancy = 0.0f;
	float vs = 0.0f;
	float vp = 0.0f;

#ifdef PML
	float pml_beta_x = 0.0f;
	float pml_beta_y = 0.0f;
#endif

	float nx = 0.0f;
	float ny = 0.0f;
	float nz = 1.0f;
	float sx = 1.0f;
	float sy = 0.0f;
	float sz = 0.0f;
	float tx = 0.0f;
	float ty = 1.0f;
	float tz = 0.0f;

	float u_conserv[9], u_phy[9], u_phy_T[9];

	CALCULATE3D(i, j, k, HALO, _nx, HALO, _ny, 0, 1)
	index = INDEX(i, j, _nz - 1);
	mu = CJM[index * CJMSIZE + 10];
	lambda = CJM[index * CJMSIZE + 11];
	buoyancy = CJM[index * CJMSIZE + 12];
	buoyancy *= Crho;
	vs = sqrt(mu * buoyancy);
	vp = sqrt((lambda + 2 * mu) * buoyancy);

#ifdef PML
	pml_beta_x = pml_beta.x[i];
	pml_beta_y = pml_beta.y[j];
#endif

	// Store conservative variables
	for (int n = 0; n < 9; n++)
	{
		u_conserv[n] = W[index * WSIZE + n];
	}

	// Calculate physical variables
	u_phy[0] = lambda * u_conserv[1] + lambda * u_conserv[2] + u_conserv[0] * (lambda + 2 * mu);
	u_phy[1] = lambda * u_conserv[0] + lambda * u_conserv[2] + u_conserv[1] * (lambda + 2 * mu);
	u_phy[2] = lambda * u_conserv[0] + lambda * u_conserv[1] + u_conserv[2] * (lambda + 2 * mu);
	u_phy[3] = 2 * mu * u_conserv[3];
	u_phy[4] = 2 * mu * u_conserv[5];
	u_phy[5] = 2 * mu * u_conserv[4];
	u_phy[6] = u_conserv[6] * buoyancy;
	u_phy[7] = u_conserv[7] * buoyancy;
	u_phy[8] = u_conserv[8] * buoyancy;

	// Rotate physical variables
	u_phy_T[0] = (u_phy[2] * (sx * sx) * (ty * ty) - 2 * u_phy[5] * (sx * sx) * ty * tz + u_phy[1] * (sx * sx) * (tz * tz) - 2 * u_phy[2] * sx * sy * tx * ty + 2 * u_phy[5] * sx * sy * tx * tz + 2 * u_phy[4] * sx * sy * ty * tz - 2 * u_phy[3] * sx * sy * (tz * tz) + 2 * u_phy[5] * sx * sz * tx * ty - 2 * u_phy[1] * sx * sz * tx * tz - 2 * u_phy[4] * sx * sz * (ty * ty) + 2 * u_phy[3] * sx * sz * ty * tz + u_phy[2] * (sy * sy) * (tx * tx) - 2 * u_phy[4] * (sy * sy) * tx * tz + u_phy[0] * (sy * sy) * (tz * tz) - 2 * u_phy[5] * sy * sz * (tx * tx) + 2 * u_phy[4] * sy * sz * tx * ty + 2 * u_phy[3] * sy * sz * tx * tz - 2 * u_phy[0] * sy * sz * ty * tz + u_phy[1] * (sz * sz) * (tx * tx) - 2 * u_phy[3] * (sz * sz) * tx * ty + u_phy[0] * (sz * sz) * (ty * ty)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
	u_phy_T[1] = (u_phy[2] * (nx * nx) * (ty * ty) - 2 * u_phy[5] * (nx * nx) * ty * tz + u_phy[1] * (nx * nx) * (tz * tz) - 2 * u_phy[2] * nx * ny * tx * ty + 2 * u_phy[5] * nx * ny * tx * tz + 2 * u_phy[4] * nx * ny * ty * tz - 2 * u_phy[3] * nx * ny * (tz * tz) + 2 * u_phy[5] * nx * nz * tx * ty - 2 * u_phy[1] * nx * nz * tx * tz - 2 * u_phy[4] * nx * nz * (ty * ty) + 2 * u_phy[3] * nx * nz * ty * tz + u_phy[2] * (ny * ny) * (tx * tx) - 2 * u_phy[4] * (ny * ny) * tx * tz + u_phy[0] * (ny * ny) * (tz * tz) - 2 * u_phy[5] * ny * nz * (tx * tx) + 2 * u_phy[4] * ny * nz * tx * ty + 2 * u_phy[3] * ny * nz * tx * tz - 2 * u_phy[0] * ny * nz * ty * tz + u_phy[1] * (nz * nz) * (tx * tx) - 2 * u_phy[3] * (nz * nz) * tx * ty + u_phy[0] * (nz * nz) * (ty * ty)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
	u_phy_T[2] = (u_phy[2] * (nx * nx) * (sy * sy) - 2 * u_phy[5] * (nx * nx) * sy * sz + u_phy[1] * (nx * nx) * (sz * sz) - 2 * u_phy[2] * nx * ny * sx * sy + 2 * u_phy[5] * nx * ny * sx * sz + 2 * u_phy[4] * nx * ny * sy * sz - 2 * u_phy[3] * nx * ny * (sz * sz) + 2 * u_phy[5] * nx * nz * sx * sy - 2 * u_phy[1] * nx * nz * sx * sz - 2 * u_phy[4] * nx * nz * (sy * sy) + 2 * u_phy[3] * nx * nz * sy * sz + u_phy[2] * (ny * ny) * (sx * sx) - 2 * u_phy[4] * (ny * ny) * sx * sz + u_phy[0] * (ny * ny) * (sz * sz) - 2 * u_phy[5] * ny * nz * (sx * sx) + 2 * u_phy[4] * ny * nz * sx * sy + 2 * u_phy[3] * ny * nz * sx * sz - 2 * u_phy[0] * ny * nz * sy * sz + u_phy[1] * (nz * nz) * (sx * sx) - 2 * u_phy[3] * (nz * nz) * sx * sy + u_phy[0] * (nz * nz) * (sy * sy)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
	u_phy_T[3] = -(nx * sx * (ty * ty) * u_phy[2] + nx * sx * (tz * tz) * u_phy[1] + ny * sy * (tx * tx) * u_phy[2] - nx * sy * (tz * tz) * u_phy[3] - ny * sx * (tz * tz) * u_phy[3] - nx * sz * (ty * ty) * u_phy[4] - nz * sx * (ty * ty) * u_phy[4] - ny * sz * (tx * tx) * u_phy[5] - nz * sy * (tx * tx) * u_phy[5] + ny * sy * (tz * tz) * u_phy[0] + nz * sz * (tx * tx) * u_phy[1] + nz * sz * (ty * ty) * u_phy[0] - nx * sy * tx * ty * u_phy[2] - ny * sx * tx * ty * u_phy[2] - 2 * nx * sx * ty * tz * u_phy[5] + nx * sy * tx * tz * u_phy[5] + nx * sz * tx * ty * u_phy[5] + ny * sx * tx * tz * u_phy[5] + nz * sx * tx * ty * u_phy[5] - nx * sz * tx * tz * u_phy[1] - nz * sx * tx * tz * u_phy[1] + nx * sy * ty * tz * u_phy[4] + ny * sx * ty * tz * u_phy[4] - 2 * ny * sy * tx * tz * u_phy[4] + ny * sz * tx * ty * u_phy[4] + nz * sy * tx * ty * u_phy[4] + nx * sz * ty * tz * u_phy[3] + ny * sz * tx * tz * u_phy[3] + nz * sx * ty * tz * u_phy[3] + nz * sy * tx * tz * u_phy[3] - 2 * nz * sz * tx * ty * u_phy[3] - ny * sz * ty * tz * u_phy[0] - nz * sy * ty * tz * u_phy[0]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
	u_phy_T[4] = -(nx * (sy * sy) * tx * u_phy[2] + nx * (sz * sz) * tx * u_phy[1] + ny * (sx * sx) * ty * u_phy[2] - nx * (sz * sz) * ty * u_phy[3] - ny * (sz * sz) * tx * u_phy[3] - nx * (sy * sy) * tz * u_phy[4] - nz * (sy * sy) * tx * u_phy[4] - ny * (sx * sx) * tz * u_phy[5] - nz * (sx * sx) * ty * u_phy[5] + ny * (sz * sz) * ty * u_phy[0] + nz * (sx * sx) * tz * u_phy[1] + nz * (sy * sy) * tz * u_phy[0] - nx * sx * sy * ty * u_phy[2] - ny * sx * sy * tx * u_phy[2] + nx * sx * sy * tz * u_phy[5] + nx * sx * sz * ty * u_phy[5] - 2 * nx * sy * sz * tx * u_phy[5] + ny * sx * sz * tx * u_phy[5] + nz * sx * sy * tx * u_phy[5] - nx * sx * sz * tz * u_phy[1] - nz * sx * sz * tx * u_phy[1] + nx * sy * sz * ty * u_phy[4] + ny * sx * sy * tz * u_phy[4] - 2 * ny * sx * sz * ty * u_phy[4] + ny * sy * sz * tx * u_phy[4] + nz * sx * sy * ty * u_phy[4] + nx * sy * sz * tz * u_phy[3] + ny * sx * sz * tz * u_phy[3] - 2 * nz * sx * sy * tz * u_phy[3] + nz * sx * sz * ty * u_phy[3] + nz * sy * sz * tx * u_phy[3] - ny * sy * sz * tz * u_phy[0] - nz * sy * sz * ty * u_phy[0]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
	u_phy_T[5] = -((ny * ny) * sx * tx * u_phy[2] + (nz * nz) * sx * tx * u_phy[1] + (nx * nx) * sy * ty * u_phy[2] - (nz * nz) * sx * ty * u_phy[3] - (nz * nz) * sy * tx * u_phy[3] - (ny * ny) * sx * tz * u_phy[4] - (ny * ny) * sz * tx * u_phy[4] - (nx * nx) * sy * tz * u_phy[5] - (nx * nx) * sz * ty * u_phy[5] + (nz * nz) * sy * ty * u_phy[0] + (nx * nx) * sz * tz * u_phy[1] + (ny * ny) * sz * tz * u_phy[0] - nx * ny * sx * ty * u_phy[2] - nx * ny * sy * tx * u_phy[2] + nx * ny * sx * tz * u_phy[5] + nx * ny * sz * tx * u_phy[5] + nx * nz * sx * ty * u_phy[5] + nx * nz * sy * tx * u_phy[5] - 2 * ny * nz * sx * tx * u_phy[5] - nx * nz * sx * tz * u_phy[1] - nx * nz * sz * tx * u_phy[1] + nx * ny * sy * tz * u_phy[4] + nx * ny * sz * ty * u_phy[4] - 2 * nx * nz * sy * ty * u_phy[4] + ny * nz * sx * ty * u_phy[4] + ny * nz * sy * tx * u_phy[4] - 2 * nx * ny * sz * tz * u_phy[3] + nx * nz * sy * tz * u_phy[3] + nx * nz * sz * ty * u_phy[3] + ny * nz * sx * tz * u_phy[3] + ny * nz * sz * tx * u_phy[3] - ny * nz * sy * tz * u_phy[0] - ny * nz * sz * ty * u_phy[0]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
	u_phy_T[6] = (sx * ty * u_phy[8] - sy * tx * u_phy[8] - sx * tz * u_phy[7] + sz * tx * u_phy[7] + sy * tz * u_phy[6] - sz * ty * u_phy[6]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
	u_phy_T[7] = -(nx * ty * u_phy[8] - ny * tx * u_phy[8] - nx * tz * u_phy[7] + nz * tx * u_phy[7] + ny * tz * u_phy[6] - nz * ty * u_phy[6]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
	u_phy_T[8] = (nx * sy * u_phy[8] - ny * sx * u_phy[8] - nx * sz * u_phy[7] + nz * sx * u_phy[7] + ny * sz * u_phy[6] - nz * sy * u_phy[6]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);

	// Apply characteristic free surface boundary conditions
	u_phy_T[1] -= u_phy_T[0] * lambda / (lambda + 2 * mu);
	u_phy_T[2] -= u_phy_T[0] * lambda / (lambda + 2 * mu);
	u_phy_T[7] -= u_phy_T[3] / (vs / buoyancy);
	u_phy_T[8] -= u_phy_T[4] / (vs / buoyancy);
	u_phy_T[6] -= u_phy_T[0] / (vp / buoyancy);
	u_phy_T[0] = 0;
	u_phy_T[3] = 0;
	u_phy_T[4] = 0;

	// Rotate back physical variables
	u_phy[0] = u_phy_T[0] * (nx * nx) + 2 * u_phy_T[3] * nx * sx + 2 * u_phy_T[4] * nx * tx + u_phy_T[1] * (sx * sx) + 2 * u_phy_T[5] * sx * tx + u_phy_T[2] * (tx * tx);
	u_phy[1] = u_phy_T[0] * (ny * ny) + 2 * u_phy_T[3] * ny * sy + 2 * u_phy_T[4] * ny * ty + u_phy_T[1] * (sy * sy) + 2 * u_phy_T[5] * sy * ty + u_phy_T[2] * (ty * ty);
	u_phy[2] = u_phy_T[0] * (nz * nz) + 2 * u_phy_T[3] * nz * sz + 2 * u_phy_T[4] * nz * tz + u_phy_T[1] * (sz * sz) + 2 * u_phy_T[5] * sz * tz + u_phy_T[2] * (tz * tz);
	u_phy[3] = u_phy_T[3] * (nx * sy + ny * sx) + u_phy_T[4] * (nx * ty + ny * tx) + u_phy_T[5] * (sx * ty + sy * tx) + nx * ny * u_phy_T[0] + sx * sy * u_phy_T[1] + tx * ty * u_phy_T[2];
	u_phy[4] = u_phy_T[3] * (nx * sz + nz * sx) + u_phy_T[4] * (nx * tz + nz * tx) + u_phy_T[5] * (sx * tz + sz * tx) + nx * nz * u_phy_T[0] + sx * sz * u_phy_T[1] + tx * tz * u_phy_T[2];
	u_phy[5] = u_phy_T[3] * (ny * sz + nz * sy) + u_phy_T[4] * (ny * tz + nz * ty) + u_phy_T[5] * (sy * tz + sz * ty) + ny * nz * u_phy_T[0] + sy * sz * u_phy_T[1] + ty * tz * u_phy_T[2];
	u_phy[6] = nx * u_phy_T[6] + sx * u_phy_T[7] + tx * u_phy_T[8];
	u_phy[7] = ny * u_phy_T[6] + sy * u_phy_T[7] + ty * u_phy_T[8];
	u_phy[8] = nz * u_phy_T[6] + sz * u_phy_T[7] + tz * u_phy_T[8];

	// Calculate conservative variables
	u_conserv[0] = (u_phy[0] * (lambda + mu)) / (2 * (mu * mu) + 3 * lambda * mu) - (lambda * u_phy[1]) / (2 * (2 * (mu * mu) + 3 * lambda * mu)) - (lambda * u_phy[2]) / (2 * (2 * (mu * mu) + 3 * lambda * mu));
	u_conserv[1] = (u_phy[1] * (lambda + mu)) / (2 * (mu * mu) + 3 * lambda * mu) - (lambda * u_phy[0]) / (2 * (2 * (mu * mu) + 3 * lambda * mu)) - (lambda * u_phy[2]) / (2 * (2 * (mu * mu) + 3 * lambda * mu));
	u_conserv[2] = (u_phy[2] * (lambda + mu)) / (2 * (mu * mu) + 3 * lambda * mu) - (lambda * u_phy[0]) / (2 * (2 * (mu * mu) + 3 * lambda * mu)) - (lambda * u_phy[1]) / (2 * (2 * (mu * mu) + 3 * lambda * mu));
	u_conserv[3] = u_phy[3] / (2 * mu);
	u_conserv[4] = u_phy[5] / (2 * mu);
	u_conserv[5] = u_phy[4] / (2 * mu);
	u_conserv[6] = u_phy[6] / buoyancy;
	u_conserv[7] = u_phy[7] / buoyancy;
	u_conserv[8] = u_phy[8] / buoyancy;

	// Put conservative variables back to wave.u
	for (int n = 0; n < 9; n++)
	{
		W[index * WSIZE + n] = u_conserv[n];
	}

	for (int n = 0; n < 9; n++)
	{
		for (int h = 1; h <= HALO; h++)
		{
			W[INDEX(i, j, _nz - 1 + h) * WSIZE + n] = extrap3(W[INDEX(i, j, _nz - 1 + h - 1) * WSIZE + n],
															  W[INDEX(i, j, _nz - 1 + h - 2) * WSIZE + n],
															  W[INDEX(i, j, _nz - 1 + h - 3) * WSIZE + n],
															  W[INDEX(i, j, _nz - 1 + h - 4) * WSIZE + n]);
		}
	}

	END_CALCULATE3D()
}

void charfreeSurfaceDeriv(
	GRID grid, WAVE wave, FLOAT *CJM, Mat_rDZ mat_rDZ,
#ifdef PML
	PML_BETA pml_beta,
#endif
	int FB1, int FB2, int FB3, float DT)
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	float rDH = grid.rDH;

	FLOAT *h_W = wave.h_W;
	FLOAT *W = wave.W;

#ifdef GPU_CUDA
	int nx = _nx_ - 2 * HALO;
	int ny = _ny_ - 2 * HALO;
	// int nz = _nz_ - 2 * HALO;

	dim3 threads(32, 4, 1);
	dim3 blocks;

	blocks.x = (_nx_ + threads.x - 1) / threads.x;
	blocks.y = (_ny_ + threads.y - 1) / threads.y;
	blocks.z = 1;

	char_free_surface_deriv<<<blocks, threads>>>(h_W, W, CJM, mat_rDZ,
#ifdef PML
												 pml_beta,
#endif
												 _nx_, _ny_, _nz_, rDH, FB1, FB2, FB3, DT);

#else

	char_free_surface_deriv(h_W, W, CJM, mat_rDZ,
#ifdef PML
							pml_beta,
#endif
							_nx_, _ny_, _nz_, rDH, FB1, FB2, FB3, DT);

#endif // GPU_CUDA
}

__GLOBAL__
void free_surface_deriv(
	FLOAT *h_W, FLOAT *W, FLOAT *CJM, Mat_rDZ mat_rDZ,
#ifdef PML
	PML_BETA pml_beta,
#endif
	int _nx_, int _ny_, int _nz_, float rDH, int FB1, int FB2, int FB3, float DT)
{
	int _nx = _nx_ - HALO;
	int _ny = _ny_ - HALO;
	int _nz = _nz_ - HALO;

#ifdef GPU_CUDA
	int i = threadIdx.x + blockIdx.x * blockDim.x + HALO;
	int j = threadIdx.y + blockIdx.y * blockDim.y + HALO;
	int k = threadIdx.z + blockIdx.z * blockDim.z + _nz - HALO;
#else
	int i = HALO;
	int j = HALO;
	int k = _nz - HALO;
#endif

	long long index;

	float mu = 0.0f;
	float lambda = 0.0f;
	float buoyancy = 0.0f;

#ifdef PML
	float pml_beta_x = 0.0f;
	float pml_beta_y = 0.0f;
#endif

	float xi_x = 0.0f;
	float xi_y = 0.0f;
	float xi_z = 0.0f;
	float et_x = 0.0f;
	float et_y = 0.0f;
	float et_z = 0.0f;
	float zt_x = 0.0f;
	float zt_y = 0.0f;
	float zt_z = 0.0f;

	float Vx_xi = 0.0f;
	float Vx_et = 0.0f;
	float Vx_zt = 0.0f;
	float Vy_xi = 0.0f;
	float Vy_et = 0.0f;
	float Vy_zt = 0.0f;
	float Vz_xi = 0.0f;
	float Vz_et = 0.0f;
	float Vz_zt = 0.0f;

	float Jinv = 0.0f;
	float jacb = 0.0f;
	float J_T1x[7] = {0.0f};
	float J_T2x[7] = {0.0f};
	float J_T3x[7] = {0.0f};
	float J_T1y[7] = {0.0f};
	float J_T2y[7] = {0.0f};
	float J_T3y[7] = {0.0f};
	float J_T1z[7] = {0.0f};
	float J_T2z[7] = {0.0f};
	float J_T3z[7] = {0.0f};

	float Txx1 = 0.0;
	float Txx2 = 0.0f;
	float Txx3 = 0.0f;
	float Tyy1 = 0.0;
	float Tyy2 = 0.0f;
	float Tyy3 = 0.0f;
	float Tzz1 = 0.0;
	float Tzz2 = 0.0f;
	float Tzz3 = 0.0f;
	float Txy1 = 0.0;
	float Txy2 = 0.0f;
	float Txy3 = 0.0f;
	float Txz1 = 0.0;
	float Txz2 = 0.0f;
	float Txz3 = 0.0f;
	float Tyz1 = 0.0;
	float Tyz2 = 0.0f;
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

	int l = 0;
	int k_s = 0; /*relative index on the surface*/
	long long pos = 0;

	int indexOnSurf;

	float *_rDZ_DX = mat_rDZ._rDZ_DX;
	float *_rDZ_DY = mat_rDZ._rDZ_DY;

	CALCULATE3D(i, j, k, HALO, _nx, HALO, _ny, _nz - HALO, _nz)
	index = INDEX(i, j, k);
	mu = CJM[index * CJMSIZE + 10];
	lambda = CJM[index * CJMSIZE + 11];
	buoyancy = CJM[index * CJMSIZE + 12];
	buoyancy *= Crho;

#ifdef PML
	pml_beta_x = pml_beta.x[i];
	pml_beta_y = pml_beta.y[j];
#endif

	k_s = (_nz - 1) - k + HALO; /*relative index on the surface*/
	for (l = 0; l <= (2 * HALO); l++)
	{
		pos = INDEX(i + (l - HALO), j, k);

		xi_x = CJM[pos * CJMSIZE + 0];
		xi_y = CJM[pos * CJMSIZE + 1];
		xi_z = CJM[pos * CJMSIZE + 2];

		jacb = CJM[pos * CJMSIZE + 9];

		J_T1x[l] = (xi_x * (float)W[pos * WSIZE + 3] + xi_y * (float)W[pos * WSIZE + 6] + xi_z * (float)W[pos * WSIZE + 7]) * jacb;
		J_T2x[l] = (xi_x * (float)W[pos * WSIZE + 6] + xi_y * (float)W[pos * WSIZE + 4] + xi_z * (float)W[pos * WSIZE + 8]) * jacb;
		J_T3x[l] = (xi_x * (float)W[pos * WSIZE + 7] + xi_y * (float)W[pos * WSIZE + 8] + xi_z * (float)W[pos * WSIZE + 5]) * jacb;

		pos = INDEX(i, j + (l - HALO), k);

		et_x = CJM[pos * CJMSIZE + 3];
		et_y = CJM[pos * CJMSIZE + 4];
		et_z = CJM[pos * CJMSIZE + 5];

		jacb = CJM[pos * CJMSIZE + 9];

		J_T1y[l] = (et_x * (float)W[pos * WSIZE + 3] + et_y * (float)W[pos * WSIZE + 6] + et_z * (float)W[pos * WSIZE + 7]) * jacb;
		J_T2y[l] = (et_x * (float)W[pos * WSIZE + 6] + et_y * (float)W[pos * WSIZE + 4] + et_z * (float)W[pos * WSIZE + 8]) * jacb;
		J_T3y[l] = (et_x * (float)W[pos * WSIZE + 7] + et_y * (float)W[pos * WSIZE + 8] + et_z * (float)W[pos * WSIZE + 5]) * jacb;
	}
	for (l = 0; l < k_s; l++)
	{
		pos = INDEX(i, j, k + (l - HALO));
		zt_x = CJM[pos * CJMSIZE + 6];
		zt_y = CJM[pos * CJMSIZE + 7];
		zt_z = CJM[pos * CJMSIZE + 8];

		jacb = CJM[pos * CJMSIZE + 9];
		J_T1z[l] = (zt_x * (float)W[pos * WSIZE + 3] + zt_y * (float)W[pos * WSIZE + 6] + zt_z * (float)W[pos * WSIZE + 7]) * jacb;
		J_T2z[l] = (zt_x * (float)W[pos * WSIZE + 6] + zt_y * (float)W[pos * WSIZE + 4] + zt_z * (float)W[pos * WSIZE + 8]) * jacb;
		J_T3z[l] = (zt_x * (float)W[pos * WSIZE + 7] + zt_y * (float)W[pos * WSIZE + 8] + zt_z * (float)W[pos * WSIZE + 5]) * jacb;
	}
	// The T on the surface is 0.
	J_T1z[k_s] = 0.0f;
	J_T2z[k_s] = 0.0f;
	J_T3z[k_s] = 0.0f;
	for (l = k_s + 1; l <= 2 * HALO; l++)
	{
		J_T1z[l] = -J_T1z[2 * k_s - l];
		J_T2z[l] = -J_T2z[2 * k_s - l];
		J_T3z[l] = -J_T3z[2 * k_s - l];
	}
	jacb = CJM[index * CJMSIZE + 9];
	Jinv = 1.0f / jacb;

	h_WVx = buoyancy * Jinv * (L_J_T(J_T1x, FB1) times_pml_beta_x + L_J_T(J_T1y, FB2) times_pml_beta_y + L_J_T(J_T1z, FB3));
	h_WVy = buoyancy * Jinv * (L_J_T(J_T2x, FB1) times_pml_beta_x + L_J_T(J_T2y, FB2) times_pml_beta_y + L_J_T(J_T2z, FB3));
	h_WVz = buoyancy * Jinv * (L_J_T(J_T3x, FB1) times_pml_beta_x + L_J_T(J_T3y, FB2) times_pml_beta_y + L_J_T(J_T3z, FB3));

	Vx_xi = L((float)W, 0, WSIZE, FB1, xi) times_pml_beta_x;
	Vx_et = L((float)W, 0, WSIZE, FB2, et) times_pml_beta_y;

	Vy_xi = L((float)W, 1, WSIZE, FB1, xi) times_pml_beta_x;
	Vy_et = L((float)W, 1, WSIZE, FB2, et) times_pml_beta_y;

	Vz_xi = L((float)W, 2, WSIZE, FB1, xi) times_pml_beta_x;
	Vz_et = L((float)W, 2, WSIZE, FB2, et) times_pml_beta_y;

	xi_x = CJM[index * CJMSIZE + 0];
	xi_y = CJM[index * CJMSIZE + 1];
	xi_z = CJM[index * CJMSIZE + 2];
	et_x = CJM[index * CJMSIZE + 3];
	et_y = CJM[index * CJMSIZE + 4];
	et_z = CJM[index * CJMSIZE + 5];
	zt_x = CJM[index * CJMSIZE + 6];
	zt_y = CJM[index * CJMSIZE + 7];
	zt_z = CJM[index * CJMSIZE + 8];

	//=======================================================
	// When change the HALO, BE CAREFUL!!!!
	//=======================================================

	if (k == _nz - 1)
	{
		indexOnSurf = Index2D(i, j, _nx_, _ny_);
		Vx_zt = DOT_PRODUCT3D(_rDZ_DX[indexOnSurf * MATSIZE + 0], _rDZ_DX[indexOnSurf * MATSIZE + 1], _rDZ_DX[indexOnSurf * MATSIZE + 2], Vx_xi, Vy_xi, Vz_xi) + DOT_PRODUCT3D(_rDZ_DY[indexOnSurf * MATSIZE + 0], _rDZ_DY[indexOnSurf * MATSIZE + 1], _rDZ_DY[indexOnSurf * MATSIZE + 2], Vx_et, Vy_et, Vz_et);
		Vy_zt = DOT_PRODUCT3D(_rDZ_DX[indexOnSurf * MATSIZE + 3], _rDZ_DX[indexOnSurf * MATSIZE + 4], _rDZ_DX[indexOnSurf * MATSIZE + 5], Vx_xi, Vy_xi, Vz_xi) + DOT_PRODUCT3D(_rDZ_DY[indexOnSurf * MATSIZE + 3], _rDZ_DY[indexOnSurf * MATSIZE + 4], _rDZ_DY[indexOnSurf * MATSIZE + 5], Vx_et, Vy_et, Vz_et);
		Vz_zt = DOT_PRODUCT3D(_rDZ_DX[indexOnSurf * MATSIZE + 6], _rDZ_DX[indexOnSurf * MATSIZE + 7], _rDZ_DX[indexOnSurf * MATSIZE + 8], Vx_xi, Vy_xi, Vz_xi) + DOT_PRODUCT3D(_rDZ_DY[indexOnSurf * MATSIZE + 6], _rDZ_DY[indexOnSurf * MATSIZE + 7], _rDZ_DY[indexOnSurf * MATSIZE + 8], Vx_et, Vy_et, Vz_et);
	}

	if (k == _nz - 2)
	{
		Vx_zt = L2((float)W, 0, WSIZE, FB3, zt);
		Vy_zt = L2((float)W, 1, WSIZE, FB3, zt);
		Vz_zt = L2((float)W, 2, WSIZE, FB3, zt);
	}
	if (k == _nz - 3)
	{
		Vx_zt = L3((float)W, 0, WSIZE, FB3, zt);
		Vy_zt = L3((float)W, 1, WSIZE, FB3, zt);
		Vz_zt = L3((float)W, 2, WSIZE, FB3, zt);
	}

	Txx1 = DOT_PRODUCT3D(xi_x, xi_y, xi_z, Vx_xi, Vy_xi, Vz_xi) * lambda + 2.0f * mu * (xi_x * Vx_xi);
	Txx2 = DOT_PRODUCT3D(et_x, et_y, et_z, Vx_et, Vy_et, Vz_et) * lambda + 2.0f * mu * (et_x * Vx_et);
	Txx3 = DOT_PRODUCT3D(zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt) * lambda + 2.0f * mu * (zt_x * Vx_zt);
	Tyy1 = DOT_PRODUCT3D(xi_x, xi_y, xi_z, Vx_xi, Vy_xi, Vz_xi) * lambda + 2.0f * mu * (xi_y * Vy_xi);
	Tyy2 = DOT_PRODUCT3D(et_x, et_y, et_z, Vx_et, Vy_et, Vz_et) * lambda + 2.0f * mu * (et_y * Vy_et);
	Tyy3 = DOT_PRODUCT3D(zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt) * lambda + 2.0f * mu * (zt_y * Vy_zt);
	Tzz1 = DOT_PRODUCT3D(xi_x, xi_y, xi_z, Vx_xi, Vy_xi, Vz_xi) * lambda + 2.0f * mu * (xi_z * Vz_xi);
	Tzz2 = DOT_PRODUCT3D(et_x, et_y, et_z, Vx_et, Vy_et, Vz_et) * lambda + 2.0f * mu * (et_z * Vz_et);
	Tzz3 = DOT_PRODUCT3D(zt_x, zt_y, zt_z, Vx_zt, Vy_zt, Vz_zt) * lambda + 2.0f * mu * (zt_z * Vz_zt);

	Txy1 = DOT_PRODUCT2D(xi_y, xi_x, Vx_xi, Vy_xi) * mu;
	Txy2 = DOT_PRODUCT2D(et_y, et_x, Vx_et, Vy_et) * mu;
	Txy3 = DOT_PRODUCT2D(zt_y, zt_x, Vx_zt, Vy_zt) * mu;
	Txz1 = DOT_PRODUCT2D(xi_z, xi_x, Vx_xi, Vz_xi) * mu;
	Txz2 = DOT_PRODUCT2D(et_z, et_x, Vx_et, Vz_et) * mu;
	Txz3 = DOT_PRODUCT2D(zt_z, zt_x, Vx_zt, Vz_zt) * mu;
	Tyz1 = DOT_PRODUCT2D(xi_z, xi_y, Vy_xi, Vz_xi) * mu;
	Tyz2 = DOT_PRODUCT2D(et_z, et_y, Vy_et, Vz_et) * mu;
	Tyz3 = DOT_PRODUCT2D(zt_z, zt_y, Vy_zt, Vz_zt) * mu;

	h_WTxx = Txx1 + Txx2 + Txx3;
	h_WTyy = Tyy1 + Tyy2 + Tyy3;
	h_WTzz = Tzz1 + Tzz2 + Tzz3;
	h_WTxy = Txy1 + Txy2 + Txy3;
	h_WTxz = Txz1 + Txz2 + Txz3;
	h_WTyz = Tyz1 + Tyz2 + Tyz3;

	h_W[index * WSIZE + 0] = h_WVx * DT;
	h_W[index * WSIZE + 1] = h_WVy * DT;
	h_W[index * WSIZE + 2] = h_WVz * DT;
	h_W[index * WSIZE + 3] = h_WTxx * DT;
	h_W[index * WSIZE + 4] = h_WTyy * DT;
	h_W[index * WSIZE + 5] = h_WTzz * DT;
	h_W[index * WSIZE + 6] = h_WTxy * DT;
	h_W[index * WSIZE + 7] = h_WTxz * DT;
	h_W[index * WSIZE + 8] = h_WTyz * DT;
	END_CALCULATE3D()
}

void freeSurfaceDeriv(
	GRID grid, WAVE wave, FLOAT *CJM, Mat_rDZ mat_rDZ,
#ifdef PML
	PML_BETA pml_beta,
#endif
	int FB1, int FB2, int FB3, float DT)
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	float rDH = grid.rDH;

	FLOAT *h_W = wave.h_W;
	FLOAT *W = wave.W;

#ifdef GPU_CUDA
	int nx = _nx_ - 2 * HALO;
	int ny = _ny_ - 2 * HALO;
	// int nz = _nz_ - 2 * HALO;

	dim3 threads(32, 4, 1);
	dim3 blocks;

	blocks.x = (nx + threads.x - 1) / threads.x;
	blocks.y = (ny + threads.y - 1) / threads.y;
	blocks.z = HALO / threads.z;

	free_surface_deriv<<<blocks, threads>>>(h_W, W, CJM, mat_rDZ,
#ifdef PML
											pml_beta,
#endif
											_nx_, _ny_, _nz_, rDH, FB1, FB2, FB3, DT);

#else

	free_surface_deriv(h_W, W, CJM, mat_rDZ,
#ifdef PML
					   pml_beta,
#endif
					   _nx_, _ny_, _nz_, rDH, FB1, FB2, FB3, DT);

#endif // GPU_CUDA
}
