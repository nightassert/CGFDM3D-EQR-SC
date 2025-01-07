/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: singleSource.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2021-11-06
*   Discription: Single source
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
void locateSource(PARAMS params, GRID grid, SOURCE *source)
{
	int sourceX = params.sourceX;
	int sourceY = params.sourceY;
	int sourceZ = params.sourceZ;

	int frontNX = grid.frontNX;
	int frontNY = grid.frontNY;
	int frontNZ = grid.frontNZ;

	source->X = sourceX - frontNX + HALO;
	source->Y = sourceY - frontNY + HALO;
	source->Z = sourceZ - frontNZ + HALO;

	// printf( "source.X = %d, source.Y = %d, source.Z = %d\n", source->X, source->Y, source->Z );
}
__DEVICE__
float sourceFunction(float rickerfc, int it, int irk, float DT)
{
	float t = 0.0f;
	float tdelay = 1.0f / rickerfc;
	// tdelay = 1.0;

	if (0 == irk)
	{
		t = (it + 0.0f) * DT;
	}
	else if (1 == irk || 2 == irk)
	{
		t = (it + 0.5f) * DT;
	}
	else if (3 == irk)
	{
		t = (it + 1.0f) * DT;
	}
	float r = PI * rickerfc * (t - tdelay);
	float rr = r * r;
	float s = r * (3.0 - 2.0f * rr) * exp(-rr) * sqrt(PI) * 0.5 * rickerfc * PI;

	float M0 = 1e9; // ! When M0 > 1e9, S wave residue in SCFDM
	s *= M0;

	return s;
}

__DEVICE__
float gaussFunction(float duration, int it, int irk, float DT)
{
	float t = 0.0f;
	float tdelay = duration / 2 + 1;
	// tdelay = 1.0;

	if (0 == irk)
	{
		t = (it + 0.0f) * DT;
	}
	else if (1 == irk || 2 == irk)
	{
		t = (it + 0.5f) * DT;
	}
	else if (3 == irk)
	{
		t = (it + 1.0f) * DT;
	}
	float s = exp(-(t - tdelay) * (t - tdelay) / (duration * duration)) / (sqrt(PI) * duration);

	return s;
}

__GLOBAL__
void load_point_source(SOURCE S, FLOAT *h_W, int _nx_, int _ny_, int _nz_,
					   FLOAT *CJM, int it, int irk, float DT, float DH)
{
	float amp = 1.0;
	float s = sourceFunction(2.0f, it, irk, DT);

	float jacb = CJM[INDEX(S.X, S.Y, S.Z) * CJMSIZE + 9];

	float value = -1.0f * s * amp / (jacb * (DH * DH * DH));

	// printf("index = %d, value = %f\n",  INDEX( S.X, S.Y, S.Z ), value );
	h_W[INDEX(S.X, S.Y, S.Z) * WSIZE + 3] += value;
	h_W[INDEX(S.X, S.Y, S.Z) * WSIZE + 4] += value;
	h_W[INDEX(S.X, S.Y, S.Z) * WSIZE + 5] += value;
}

__GLOBAL__
void load_smooth_source_ricker(SOURCE S, FLOAT *h_W, int _nx_, int _ny_, int _nz_,
							   FLOAT *CJM, int it, int iRK, float DT, float DH, int nGauss, float rickerfc)
{
#ifdef GPU_CUDA
	int i = threadIdx.x + blockIdx.x * blockDim.x + S.X - nGauss;
	int j = threadIdx.y + blockIdx.y * blockDim.y + S.Y - nGauss;
	int k = threadIdx.z + blockIdx.z * blockDim.z + S.Z - nGauss;
#else
	int i = 0;
	int j = 0;
	int k = 0;
#endif
	long long index = 0;
	float amp = 0.0f;
	float s = 0.0f;
	float value = 0.0f;

	CALCULATE3D(i, j, k, S.X - nGauss, S.X + nGauss + 1, S.Y - nGauss, S.Y + nGauss + 1, S.Z - nGauss, S.Z + nGauss + 1)
	index = INDEX(i, j, k);
	if (iRK == 0)
		s = 0.0f;
	s = sourceFunction(rickerfc, it, iRK, DT);
	float ra = nGauss * 0.5;
	float D1 = GAUSS_FUN(i - S.X, ra, 0.0);
	float D2 = GAUSS_FUN(j - S.Y, ra, 0.0);
	float D3 = GAUSS_FUN(k - S.Z, ra, 0.0);
	float amp = D1 * D2 * D3;

	amp /= 0.998125703461425; // # 3
	// amp /= 0.9951563131100551; // # 5

	float jacb = CJM[INDEX(S.X, S.Y, S.Z) * CJMSIZE + 9];

	value = -1.0f * s * amp / (jacb * (DH * DH * DH));
	// value = -1.0f * s * amp  / (  DH * DH * DH );
	// printf( "value = %f\n", value );

	// if ( i == S.X && j == S.Y && k == S.Z )
	// printf( "index = %d\n", index  );
	// printf( "value = %10.10lf\n", value  );
	value *= Cs * DT;

#ifdef SCFDM

	float mu, lambda, buoyancy;

	mu = CJM[index * CJMSIZE + 10];
	lambda = CJM[index * CJMSIZE + 11];

	value = value / (3 * lambda + 2 * mu);

	h_W[index * WSIZE + 3] = ((float)h_W[index * WSIZE + 3] + value);
	h_W[index * WSIZE + 4] = ((float)h_W[index * WSIZE + 4] + value);
	h_W[index * WSIZE + 5] = ((float)h_W[index * WSIZE + 5] + value);
#else
	h_W[index * WSIZE + 3] = ((float)h_W[index * WSIZE + 3] + value);
	h_W[index * WSIZE + 4] = ((float)h_W[index * WSIZE + 4] + value);
	h_W[index * WSIZE + 5] = ((float)h_W[index * WSIZE + 5] + value);
#endif

	END_CALCULATE3D()
}

void loadPointSource_ricker(GRID grid, SOURCE S, FLOAT *h_W, FLOAT *CJM, int it, int irk, float DT, float DH, float rickerfc)
{
	// load_point_source<<< 1, 1 >>>( S, h_W, _nx_, _ny_, _nz_, Jac, it, irk, DT, DH );

	int nGauss = 3;

	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	int _nx = _nx_ - HALO;
	int _ny = _ny_ - HALO;
	int _nz = _nz_ - HALO;

#ifdef GPU_CUDA

	dim3 threads(4, 4, 4);
	dim3 blocks;
	blocks.x = (2 * nGauss + 1 + threads.x - 1) / threads.x;
	blocks.y = (2 * nGauss + 1 + threads.y - 1) / threads.y;
	blocks.z = (2 * nGauss + 1 + threads.z - 1) / threads.z;

	if (S.X >= HALO && S.X < _nx &&
		S.Y >= HALO && S.Y < _ny &&
		S.Z >= HALO && S.Z < _nz)
	{
		load_smooth_source_ricker<<<blocks, threads>>>(S, h_W, _nx_, _ny_, _nz_, CJM, it, irk, DT, DH, nGauss, rickerfc);
	}
	CHECK(cudaDeviceSynchronize());
#else

	if (S.X >= HALO && S.X < _nx &&
		S.Y >= HALO && S.Y < _ny &&
		S.Z >= HALO && S.Z < _nz)
	{
		load_smooth_source(S, h_W, _nx_, _ny_, _nz_, CJM, it, irk, DT, DH, nGauss, rickerfc);
	}

#endif
}

__GLOBAL__
void load_smooth_source_double_couple(SOURCE S, FLOAT *h_W, int _nx_, int _ny_, int _nz_,
									  FLOAT *CJM, int it, int iRK, float DT, float DH, int nGauss, float strike, float dip, float rake, float Mw, float duration)
{
#ifdef GPU_CUDA
	int i = threadIdx.x + blockIdx.x * blockDim.x + S.X - nGauss;
	int j = threadIdx.y + blockIdx.y * blockDim.y + S.Y - nGauss;
	int k = threadIdx.z + blockIdx.z * blockDim.z + S.Z - nGauss;
#else
	int i = 0;
	int j = 0;
	int k = 0;
#endif
	long long index = 0;
	float amp = 0.0f;
	float s = 0.0f;
	float rate = 0.0f;

	strike = strike * DEGREE2RADIAN;
	dip = dip * DEGREE2RADIAN;
	rake = rake * DEGREE2RADIAN;

	float m0 = 1.5 * powf(10, Mw + 6);

	float M11 = 0.0f;
	float M22 = 0.0f;
	float M33 = 0.0f;
	float M12 = 0.0f;
	float M13 = 0.0f;
	float M23 = 0.0f;

	float momentrate[6];
	float momentrate_strain[6];

	CALCULATE3D(i, j, k, S.X - nGauss, S.X + nGauss + 1, S.Y - nGauss, S.Y + nGauss + 1, S.Z - nGauss, S.Z + nGauss + 1)
	index = INDEX(i, j, k);
	if (iRK == 0)
		s = 0.0f;
	s = gaussFunction(duration, it, iRK, DT);
	float ra = nGauss * 0.5;
	float D1 = GAUSS_FUN(i - S.X, ra, 0.0);
	float D2 = GAUSS_FUN(j - S.Y, ra, 0.0);
	float D3 = GAUSS_FUN(k - S.Z, ra, 0.0);
	float amp = D1 * D2 * D3;

	amp /= 0.998125703461425; // # 3
	// amp /= 0.9951563131100551; // # 5

	float jacb = CJM[INDEX(S.X, S.Y, S.Z) * CJMSIZE + 9];

	rate = -1.0f * s * amp / (jacb * (DH * DH * DH)) * m0;
	// value = -1.0f * s * amp  / (  DH * DH * DH );
	// printf( "value = %f\n", value );

	// if ( i == S.X && j == S.Y && k == S.Z )
	// printf( "index = %d\n", index  );
	// printf( "value = %10.10lf\n", value  );
	rate *= Cs * DT;

	M11 = -(sin(dip) * cos(rake) * sin(2.0 * strike) + sin(2.0 * dip) * sin(rake) * sin(strike) * sin(strike));
	M22 = sin(dip) * cos(rake) * sin(2.0 * strike) - sin(2.0 * dip) * sin(rake) * cos(strike) * cos(strike);
	M33 = -(M11 + M22);
	M12 = sin(dip) * cos(rake) * cos(2.0 * strike) + 0.5 * sin(2.0 * dip) * sin(rake) * sin(2.0 * strike);
	M13 = -(cos(dip) * cos(rake) * cos(strike) + cos(2.0 * dip) * sin(rake) * sin(strike));
	M23 = -(cos(dip) * cos(rake) * sin(strike) - cos(2.0 * dip) * sin(rake) * cos(strike));

	momentrate[0] = M22 * rate;
	momentrate[1] = M11 * rate;
	momentrate[2] = M33 * rate;
	momentrate[3] = M12 * rate;
	momentrate[4] = -M23 * rate;
	momentrate[5] = -M13 * rate;

#ifdef SCFDM
	float mu, lambda, buoyancy;

	mu = CJM[index * CJMSIZE + 10];
	lambda = CJM[index * CJMSIZE + 11];

	momentrate_strain[0] = (momentrate[0] * (lambda + mu)) / (2 * (mu * mu) + 3 * lambda * mu) - (lambda * momentrate[1]) / (2 * (2 * (mu * mu) + 3 * lambda * mu)) - (lambda * momentrate[2]) / (2 * (2 * (mu * mu) + 3 * lambda * mu));
	momentrate_strain[1] = (momentrate[1] * (lambda + mu)) / (2 * (mu * mu) + 3 * lambda * mu) - (lambda * momentrate[0]) / (2 * (2 * (mu * mu) + 3 * lambda * mu)) - (lambda * momentrate[2]) / (2 * (2 * (mu * mu) + 3 * lambda * mu));
	momentrate_strain[2] = (momentrate[2] * (lambda + mu)) / (2 * (mu * mu) + 3 * lambda * mu) - (lambda * momentrate[0]) / (2 * (2 * (mu * mu) + 3 * lambda * mu)) - (lambda * momentrate[1]) / (2 * (2 * (mu * mu) + 3 * lambda * mu));
	momentrate_strain[3] = momentrate[5] / (mu);
	momentrate_strain[4] = momentrate[4] / (mu);
	momentrate_strain[5] = momentrate[3] / (mu);

	h_W[index * WSIZE + 3] = ((float)h_W[index * WSIZE + 3] - momentrate_strain[0]);
	h_W[index * WSIZE + 4] = ((float)h_W[index * WSIZE + 4] - momentrate_strain[1]);
	h_W[index * WSIZE + 5] = ((float)h_W[index * WSIZE + 5] - momentrate_strain[2]);
	h_W[index * WSIZE + 6] = ((float)h_W[index * WSIZE + 6] - momentrate_strain[3]);
	h_W[index * WSIZE + 7] = ((float)h_W[index * WSIZE + 7] - momentrate_strain[4]);
	h_W[index * WSIZE + 8] = ((float)h_W[index * WSIZE + 8] - momentrate_strain[5]);
#else
	h_W[index * WSIZE + 3] = ((float)h_W[index * WSIZE + 3] - momentrate[0]);
	h_W[index * WSIZE + 4] = ((float)h_W[index * WSIZE + 4] - momentrate[1]);
	h_W[index * WSIZE + 5] = ((float)h_W[index * WSIZE + 5] - momentrate[2]);
	h_W[index * WSIZE + 6] = ((float)h_W[index * WSIZE + 6] - momentrate[3]);
	h_W[index * WSIZE + 7] = ((float)h_W[index * WSIZE + 7] - momentrate[4]);
	h_W[index * WSIZE + 8] = ((float)h_W[index * WSIZE + 8] - momentrate[5]);
#endif

	END_CALCULATE3D()
}

void loadPointSource_double_couple(GRID grid, SOURCE S, FLOAT *h_W, FLOAT *CJM, int it, int irk, float DT, float DH, float strike, float dip, float rake, float Mw, float duration)
{
	// load_point_source<<< 1, 1 >>>( S, h_W, _nx_, _ny_, _nz_, Jac, it, irk, DT, DH );

	int nGauss = 3;

	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	int _nx = _nx_ - HALO;
	int _ny = _ny_ - HALO;
	int _nz = _nz_ - HALO;

#ifdef GPU_CUDA

	dim3 threads(4, 4, 4);
	dim3 blocks;
	blocks.x = (2 * nGauss + 1 + threads.x - 1) / threads.x;
	blocks.y = (2 * nGauss + 1 + threads.y - 1) / threads.y;
	blocks.z = (2 * nGauss + 1 + threads.z - 1) / threads.z;

	if (S.X >= HALO && S.X < _nx &&
		S.Y >= HALO && S.Y < _ny &&
		S.Z >= HALO && S.Z < _nz)
	{
		load_smooth_source_double_couple<<<blocks, threads>>>(S, h_W, _nx_, _ny_, _nz_, CJM, it, irk, DT, DH, nGauss, strike, dip, rake, Mw, duration);
	}
	CHECK(cudaDeviceSynchronize());
#else

	if (S.X >= HALO && S.X < _nx &&
		S.Y >= HALO && S.Y < _ny &&
		S.Z >= HALO && S.Z < _nz)
	{
		load_smooth_source(S, h_W, _nx_, _ny_, _nz_, CJM, it, irk, DT, DH, nGauss, rickerfc);
	}

#endif
}

__GLOBAL__
void Gaussian_pluse(FLOAT *W,
					int _nx_, int _ny_, int _nz_,
					int frontNX, int frontNY, int frontNZ,
					int _NX_, int _NY_, int _NZ_)
{
	float sigma = 10.0f;
#ifdef GPU_CUDA
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	int k = threadIdx.z + blockIdx.z * blockDim.z;
#else
	int i = 0;
	int j = 0;
	int k = 0;
#endif

	long long index = 0;

	int _nx = _nx_ - HALO;
	int _ny = _ny_ - HALO;
	int _nz = _nz_ - HALO;

	int I = frontNX + i;
	int J = frontNY + j;
	int K = frontNZ + k;

	// printf( "===============\n" );
	CALCULATE3D(i, j, k, HALO, _nx, HALO, _ny, HALO, _nz)
	index = INDEX(i, j, k);
	float r2 = -((I - _NX_ / 2) * (I - _NX_ / 2) + (J - _NY_ / 2) * (J - _NY_ / 2) + (K - _NZ_ / 2) * (K - _NZ_ / 2)); // powf( i - _nx_ * 0.5, 2 ) + powf( j - _ny_ * 0.5, 2 ) + powf( k - _nz_ * 0.5, 2 );
	float a2 = 2.0f * sigma * sigma;
	float M0 = 1e2f * expf(r2 / a2);

	W[index * WSIZE + 3] = 1e19f * (1.0f / (PI * a2)) * exp(r2 / a2);
	W[index * WSIZE + 4] = 1e19f * (1.0f / (PI * a2)) * exp(r2 / a2);
	W[index * WSIZE + 5] = 1e19f * (1.0f / (PI * a2)) * exp(r2 / a2);

	END_CALCULATE3D()
}

void GaussField(GRID grid, FLOAT *W)
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	int frontNX = grid.frontNX;
	int frontNY = grid.frontNY;
	int frontNZ = grid.frontNZ;

	int _NX_ = grid._NX_;
	int _NY_ = grid._NY_;
	int _NZ_ = grid._NZ_;

	int originalX = grid.originalX;
	int originalY = grid.originalY;

#ifdef GPU_CUDA

	printf("W = %p\n", W);

	dim3 threads(32, 4, 4);
	dim3 blocks;
	blocks.x = (_nx_ + threads.x - 1) / threads.x;
	blocks.y = (_ny_ + threads.y - 1) / threads.y;
	blocks.z = (_nz_ + threads.z - 1) / threads.z;

	Gaussian_pluse<<<blocks, threads>>>(W, _nx_, _ny_, _nz_, frontNX, frontNY, frontNZ, _NX_, _NY_, _NZ_);
	CHECK(cudaDeviceSynchronize());
#else
	Gaussian_pluse(W, _nx_, _ny_, _nz_, frontNX, frontNY, frontNZ, _NX_, _NY_, _NZ_);

#endif
}
