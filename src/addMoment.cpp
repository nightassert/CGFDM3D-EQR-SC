/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: addMoment.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2022-11-16
*   Discription: Moment rate source
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

// The WENO scheme evolves strain, while the moment-tensor source is given
// in stress.  Both source kernels use this conversion.
__DEVICE__ inline void stress_to_strain(const float *m_stress, float *m_strain,
										float mu, float lambda)
{
	float det = 2.0f * (mu * mu) + 3.0f * lambda * mu;
	float lam_term = lambda / (2.0f * det);

	m_strain[0] = m_stress[0] * (lambda + mu) / det - lam_term * m_stress[1] - lam_term * m_stress[2];
	m_strain[1] = m_stress[1] * (lambda + mu) / det - lam_term * m_stress[0] - lam_term * m_stress[2];
	m_strain[2] = m_stress[2] * (lambda + mu) / det - lam_term * m_stress[0] - lam_term * m_stress[1];
	m_strain[3] = m_stress[5] / mu;
	m_strain[4] = m_stress[4] / mu;
	m_strain[5] = m_stress[3] / mu;
}

__GLOBAL__
void calculate_MomentRate(long long npts, long long nt, FLOAT *CJM, float *momentRate, long long *srcIndex, float DH)
{

#ifdef GPU_CUDA
	long long i = threadIdx.x + blockIdx.x * blockDim.x;
	long long j = threadIdx.y + blockIdx.y * blockDim.y;
#else
	long long i = 0;
	long long j = 0;
#endif
	long long idx = 0;
	long long pos = 0;

	float V = 1.0;

	float jacb = 0.0f;

	float mu = 0.0f;

	CALCULATE2D(i, j, 0, npts, 0, nt)
	idx = srcIndex[i];
	jacb = CJM[idx * CJMSIZE + 9];
	mu = CJM[idx * CJMSIZE + 10];

	pos = i + j * npts;
	V = jacb * DH * DH * DH;
	V = 1.0 / V;
	V = V * Cv;
	momentRate[pos * MOMSIZE + 0] *= mu * V;
	momentRate[pos * MOMSIZE + 1] *= mu * V;
	momentRate[pos * MOMSIZE + 2] *= mu * V;
	momentRate[pos * MOMSIZE + 3] *= mu * V;
	momentRate[pos * MOMSIZE + 4] *= mu * V;
	momentRate[pos * MOMSIZE + 5] *= mu * V;
	END_CALCULATE2D()
}

void calculateMomentRate(SOURCE_FILE_INPUT src_in, FLOAT *CJM, float *momentRate, long long *srcIndex, float DH)
{

	long long npts = src_in.npts;
	long long nt = src_in.nt;

	if (0 == npts)
		return;

#ifdef GPU_CUDA
	dim3 threads(16, 16, 1);
	dim3 blocks;
	blocks.x = (npts + threads.x - 1) / threads.x;
	blocks.y = (nt + threads.y - 1) / threads.y;
	blocks.z = 1;
	calculate_MomentRate<<<blocks, threads>>>(npts, nt, CJM, momentRate, srcIndex, DH);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

#else
	calculate_MomentRate(npts, nt, CJM, momentRate, srcIndex, DH);
#endif
}

__GLOBAL__
void interp_momentRate(long long npts, float *momentRate, float *momentRateSlice, float t_weight, int srcIt)
{
#ifdef GPU_CUDA
	long long i = threadIdx.x + blockIdx.x * blockDim.x;
#else
	long long i = 0;
#endif
	long long pos0 = 0, pos1 = 0;

	CALCULATE1D(i, 0, npts)
	pos0 = srcIt * npts + i;
	pos1 = (srcIt + 1) * npts + i;
	momentRateSlice[i * MOMSIZE + 0] = (momentRate[pos1 * MOMSIZE + 0] - momentRate[pos0 * MOMSIZE + 0]) * t_weight + momentRate[pos0 * MOMSIZE + 0];
	momentRateSlice[i * MOMSIZE + 1] = (momentRate[pos1 * MOMSIZE + 1] - momentRate[pos0 * MOMSIZE + 1]) * t_weight + momentRate[pos0 * MOMSIZE + 1];
	momentRateSlice[i * MOMSIZE + 2] = (momentRate[pos1 * MOMSIZE + 2] - momentRate[pos0 * MOMSIZE + 2]) * t_weight + momentRate[pos0 * MOMSIZE + 2];
	momentRateSlice[i * MOMSIZE + 3] = (momentRate[pos1 * MOMSIZE + 3] - momentRate[pos0 * MOMSIZE + 3]) * t_weight + momentRate[pos0 * MOMSIZE + 3];
	momentRateSlice[i * MOMSIZE + 4] = (momentRate[pos1 * MOMSIZE + 4] - momentRate[pos0 * MOMSIZE + 4]) * t_weight + momentRate[pos0 * MOMSIZE + 4];
	momentRateSlice[i * MOMSIZE + 5] = (momentRate[pos1 * MOMSIZE + 5] - momentRate[pos0 * MOMSIZE + 5]) * t_weight + momentRate[pos0 * MOMSIZE + 5];
	END_CALCULATE1D()
}

__GLOBAL__
void addSource(FLOAT *hW, float *momentRateSlice, long long *srcIndex, int npts,
			   int gaussI, int gaussJ, int gaussK, float factorGauss,
			   int _nx_, int _ny_, int _nz_, int flagSurf, float DT,
			   FLOAT *CJM, int scheme)
{

#ifdef GPU_CUDA
	long long i = threadIdx.x + blockIdx.x * blockDim.x;
#else
	long long i = 0;
#endif
	long long idx = 0;

	int Z = 0;
	CALCULATE1D(i, 0, npts)
	idx = srcIndex[i];

	if (flagSurf == 1)
	{
		Z = idx / _nx_ / _ny_;
		if (Z + gaussK > _nz_ - 4)
			factorGauss = 0.0;

		if ((Z == _nz_ - 4 && gaussK < 0) ||
			(Z == _nz_ - 5 && gaussK < -1) ||
			(Z == _nz_ - 6 && gaussK < -2))
		{
			factorGauss = factorGauss * 2;
		}
	}

	if (scheme == SCHEME_WENO)
	{
		float mu = CJM[idx * CJMSIZE + 10];
		float lambda = CJM[idx * CJMSIZE + 11];
		float m_stress[6], m_strain[6];

		for (int n = 0; n < 6; n++)
			m_stress[n] = momentRateSlice[i * MOMSIZE + n];

		stress_to_strain(m_stress, m_strain, mu, lambda);

		for (int n = 0; n < 6; n++)
			hW[idx * WSIZE + 3 + n] -= m_strain[n] * factorGauss * DT;
	}
	else
	{
		hW[idx * WSIZE + 3] -= momentRateSlice[i * MOMSIZE + 0] * factorGauss * DT;
		hW[idx * WSIZE + 4] -= momentRateSlice[i * MOMSIZE + 1] * factorGauss * DT;
		hW[idx * WSIZE + 5] -= momentRateSlice[i * MOMSIZE + 2] * factorGauss * DT;
		hW[idx * WSIZE + 6] -= momentRateSlice[i * MOMSIZE + 3] * factorGauss * DT;
		hW[idx * WSIZE + 7] -= momentRateSlice[i * MOMSIZE + 4] * factorGauss * DT;
		hW[idx * WSIZE + 8] -= momentRateSlice[i * MOMSIZE + 5] * factorGauss * DT;
	}
	END_CALCULATE1D()
}

__GLOBAL__
void addSource1(FLOAT *hW, float *momentRateSlice, long long *srcIndex, int npts, float DT,
				FLOAT *CJM, int scheme)
{

#ifdef GPU_CUDA
	long long i = threadIdx.x + blockIdx.x * blockDim.x;
#else
	long long i = 0;
#endif
	long long idx = 0;

	CALCULATE1D(i, 0, npts)
	idx = srcIndex[i];

	if (scheme == SCHEME_WENO)
	{
		float mu = CJM[idx * CJMSIZE + 10];
		float lambda = CJM[idx * CJMSIZE + 11];
		float m_stress[6], m_strain[6];

		for (int n = 0; n < 6; n++)
			m_stress[n] = momentRateSlice[i * MOMSIZE + n];

		stress_to_strain(m_stress, m_strain, mu, lambda);

		for (int n = 0; n < 6; n++)
			hW[idx * WSIZE + 3 + n] -= m_strain[n] * DT;
	}
	else
	{
		hW[idx * WSIZE + 3] -= momentRateSlice[i * MOMSIZE + 0] * DT;
		hW[idx * WSIZE + 4] -= momentRateSlice[i * MOMSIZE + 1] * DT;
		hW[idx * WSIZE + 5] -= momentRateSlice[i * MOMSIZE + 2] * DT;
		hW[idx * WSIZE + 6] -= momentRateSlice[i * MOMSIZE + 3] * DT;
		hW[idx * WSIZE + 7] -= momentRateSlice[i * MOMSIZE + 4] * DT;
		hW[idx * WSIZE + 8] -= momentRateSlice[i * MOMSIZE + 5] * DT;
	}
	END_CALCULATE1D()
}

void addMomenteRate(GRID grid, SOURCE_FILE_INPUT src_in,
					FLOAT *hW, long long *&srcIndex,
					float *&momentRate, float *&momentRateSlice,
					int it, int irk, float DT, float DH,
					float *gaussFactor, int nGauss, int flagSurf,
					FLOAT *CJM)
{
	// This routine is collective: it opens with a barrier and closes with a
	// second one at the end, and *every* rank has to reach both of them on
	// every call.  Two per-rank conditions make the body a no-op:
	//   - the subdomain holds no fault point (npts == 0), so the allocators in
	//     init_MultiSource() leave all three handles NULL;
	//   - the fault source has already run out and finish_MultiSource()
	//     released the buffers and cleared the handles.
	// Neither of them may leave through a bare "return" -- that skips the
	// closing barrier, and the ranks that still own fault points then wait
	// there forever.  That is a deadlock on it = 0 with every GPU idle.
	MPI_Barrier(MPI_COMM_WORLD);

	if (NULL == srcIndex || NULL == momentRate || NULL == momentRateSlice)
	{
		MPI_Barrier(MPI_COMM_WORLD);
		return;
	}

	long long npts = src_in.npts;
	int nt = src_in.nt;
	float dt = src_in.dt;

	float tmpT, t1, t_weight;

	if (0 == irk)
	{
		tmpT = (it + 0.0f) * DT;
	}
	else if (1 == irk || 2 == irk)
	{
		tmpT = (it + 0.5f) * DT;
	}
	else if (3 == irk)
	{
		tmpT = (it + 1.0f) * DT;
	}

	int srcIt = int(tmpT / dt);

	if ((srcIt + 1) >= nt)
	{
		finish_MultiSource(srcIndex, momentRate, momentRateSlice, src_in.npts);

		// Same rule as at the top: leave through the closing barrier, not
		// around it.
		MPI_Barrier(MPI_COMM_WORLD);
		return;
	}

	t1 = float(srcIt) * dt;
	t_weight = (tmpT - t1) / dt;

	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;
	int gaussI, gaussJ, gaussK, gPos;
	int lenGauss = nGauss * 2 + 1;
	float factorGauss = 0.0f;
#ifdef GPU_CUDA
	// A rank whose subdomain holds no fault point has npts == 0, so every
	// launch below would use blocks.x == 0.  CUDA rejects such a launch with
	// cudaErrorInvalidValue ("invalid argument") and cudaDeviceSynchronize
	// does not report it, so the stale error would only surface at the next
	// cudaGetLastError() -- the one in the WENO derivative kernels -- and
	// abort the whole run.  Nothing to do on these ranks: the kernel bodies
	// would not execute a single thread anyway.
	if (npts > 0)
	{
		long long num = npts;
		dim3 threads(256, 1, 1);
		dim3 blocks;
		blocks.x = (num + threads.x - 1) / threads.x;
		blocks.y = 1;
		blocks.z = 1;
		CHECK(cudaDeviceSynchronize());
		interp_momentRate<<<blocks, threads>>>(npts, momentRate, momentRateSlice, t_weight, srcIt);
		CHECK(cudaGetLastError());
		CHECK(cudaDeviceSynchronize());
#ifdef NO_SOURCE_SMOOTH
		addSource1<<<blocks, threads>>>(hW, momentRateSlice, srcIndex, npts, DT, CJM, g_params.scheme);
		CHECK(cudaGetLastError());
		CHECK(cudaDeviceSynchronize());

#else

		for (gaussK = -nGauss; gaussK < nGauss + 1; gaussK++)
		{
			for (gaussJ = -nGauss; gaussJ < nGauss + 1; gaussJ++)
			{
				for (gaussI = -nGauss; gaussI < nGauss + 1; gaussI++)
				{
					gPos = (gaussI + nGauss) + (gaussJ + nGauss) * lenGauss + (gaussK + nGauss) * lenGauss * lenGauss;
					factorGauss = gaussFactor[gPos];
					addSource<<<blocks, threads>>>(hW,
												   momentRateSlice,
												   srcIndex, npts,
												   gaussI, gaussJ, gaussK, factorGauss, _nx_, _ny_, _nz_, flagSurf, DT,
												   CJM, g_params.scheme);
					CHECK(cudaGetLastError());
					CHECK(cudaDeviceSynchronize());
				}
			}
		}
#endif
	}

#else
	interp_momentRate(npts, momentRate, momentRateSlice, t_weight, srcIt);
#ifdef NO_SOURCE_SMOOTH
	addSource1(hW, momentRateSlice, srcIndex, npts, DT, CJM, g_params.scheme);

#else

	for (gaussK = -nGauss; gaussK < nGauss + 1; gaussK++)
	{
		for (gaussJ = -nGauss; gaussJ < nGauss + 1; gaussJ++)
		{
			for (gaussI = -nGauss; gaussI < nGauss + 1; gaussI++)
			{
				gPos = (gaussI + nGauss) + (gaussJ + nGauss) * lenGauss + (gaussK + nGauss) * lenGauss * lenGauss;
				factorGauss = gaussFactor[gPos];
				addSource(hW, momentRateSlice, srcIndex, npts, gaussI, gaussJ, gaussK,
						  factorGauss, _nx_, _ny_, _nz_, flagSurf, DT,
						  CJM, g_params.scheme);
			}
		}
	}
#endif

#endif
	MPI_Barrier(MPI_COMM_WORLD);
}

void allocGaussFactor(float **gaussFactor, int nGauss)
{
	int lenGauss = nGauss * 2 + 1;
	int gaussPoints = lenGauss * lenGauss * lenGauss;
	*gaussFactor = (float *)malloc(gaussPoints * sizeof(float));
}

void gaussSmooth(float *gaussFactor, int nGauss)
{
	int lenGauss = nGauss * 2 + 1;
	int gaussPoints = lenGauss * lenGauss * lenGauss;
	int gPos = 0;
	float sumGauss = 0.0;
	float factorGauss = 0.0;
	int gaussI = 0, gaussJ = 0, gaussK = 0;
	float ra = 0.5 * nGauss;
	for (gaussK = -nGauss; gaussK < nGauss + 1; gaussK++)
	{
		for (gaussJ = -nGauss; gaussJ < nGauss + 1; gaussJ++)
		{
			for (gaussI = -nGauss; gaussI < nGauss + 1; gaussI++)
			{
				gPos = (gaussI + nGauss) + (gaussJ + nGauss) * lenGauss + (gaussK + nGauss) * lenGauss * lenGauss;
				float D1 = GAUSS_FUN(gaussI, ra, 0.0);
				float D2 = GAUSS_FUN(gaussJ, ra, 0.0);
				float D3 = GAUSS_FUN(gaussK, ra, 0.0);
				float amp = D1 * D2 * D3 / 0.998125703461425;
				gaussFactor[gPos] = amp;
				sumGauss += amp;
			}
		}
	}
}

void freeGaussFactor(float *gaussFactor)
{
	free(gaussFactor);
}
