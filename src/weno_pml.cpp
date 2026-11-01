/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: weno_pml.cpp
*   Created Time: 2026-10-27
*   Discription: C-PML absorbing boundary condition for the WENO scheme
*
*   The WENO scheme (alternative_flux_FD.cpp) evolves conservative variables
*       U = [Px, Py, Pz, Exx, Eyy, Ezz, Eyz, Exz, Exy],
*   while CGFDM evolves velocity-stress variables.  This module ports the
*   CGFDM C-PML (init_pml_para.cpp, pml_deriv.cpp, pml_rk.cpp) to the WENO
*   conservative system:
*
*     - The same one-dimensional damping profiles d, alpha and beta (kappa)
*       are reused unchanged (init_pml_parameter).
*     - Per surface m, the split auxiliary field Aux_m holds the nine
*       conservative components and follows the CFS-PML recursion
*           Aux_m^{n+1} = b_m * Aux_m^n + (d_m / (d_m + alpha_m)) * (1 - b_m) * D_m,
*           b_m = exp( -(d_m + alpha_m) * DT ),
*       which is the exact integration of  dAux/dt = d_m * D_m - (d_m + alpha_m) * Aux_m
*       for piecewise-constant D_m.  d_m here is already d/kappa and
*       (d_m + alpha_m) = d/kappa + alpha is the complex-frequency-shifted
*       decay rate, i.e. this is the Komatitsch & Martin (2007) recursion.
*       D_m is the m-directional WENO flux contribution computed by
*       cal_du_m in alternative_flux_FD.cpp.  D_m already contains the
*       factor DT * rDH * jac, so it needs no rescaling.
*     - Inside the slab the interior contribution is replaced by the C-PML
*       form
*           h_W += beta_m * ( D_m - Aux_m * DT ),    beta_m = 1 / kappa_m.
*       cal_du_x / cal_du_y / cal_du_z stash D_x / D_y / D_z into the h_Aux
*       slot of the corresponding surface instead of writing them to h_W,
*       and the kernels below read the stash back before overwriting it.
*     - Aux_m is integrated with the same TVD-RK3 as the main field, so
*       only three arrays per surface (h_Aux, Aux, m_Aux) are needed;
*       CGFDM's RK4 needs a fourth one (t_Aux, left unused here).
*     - Under FREE_SURFACE there is no top z-PML, exactly as in CGFDM.  The
*       WENO characteristic free surface (freeSurface.cpp) runs after the
*       RK stages and replaces the pml_freeSurface.cpp correction used by
*       CGFDM.
*
*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*      3. Komatitsch, D., & Martin, R. (2007). An unsplit convolutional perfectly matched layer improved at grazing incidence for the seismic wave equation. Geophysics, 72(5), SM155-SM167. https://doi.org/https://doi.org/10.1190/1.2757586
*
=================================================================*/

#include "header.h"

void allocWenoAux(int nPML, int N1, int N2, AUX *Aux)
{
	long long num = nPML * N1 * N2;

	float *pAux = NULL;
	long long size = sizeof(float) * num * WSIZE * 3;

	CHECK(Malloc((void **)&pAux, size));

	if (pAux == NULL)
	{
		printf("can't allocate WENO PML Auxiliry memory!\n");
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}
	CHECK(Memset(pAux, 0, size));

	Aux->h_Aux = pAux + 0 * WSIZE * num; // TVD-RK3 increment (also the D_m stash)
	Aux->Aux = pAux + 1 * WSIZE * num;   // current split field
	Aux->m_Aux = pAux + 2 * WSIZE * num; // stage-0 snapshot
	Aux->t_Aux = NULL;					 // unused by TVD-RK3
}

void allocWenoPML(GRID grid, AUX6SURF *Aux6, MPI_BORDER border)
{

	int nx = grid.nx;
	int ny = grid.ny;
	int nz = grid.nz;

	int nPML = grid.nPML;

	memset((void *)Aux6, 0, sizeof(AUX6SURF));

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
		allocWenoAux(nPML, ny, nz, &(Aux6->Aux1x));
	if (border.isy1)
		allocWenoAux(nPML, nx, nz, &(Aux6->Aux1y));
	if (border.isz1)
		allocWenoAux(nPML, nx, ny, &(Aux6->Aux1z));

	if (border.isx2)
		allocWenoAux(nPML, ny, nz, &(Aux6->Aux2x));
	if (border.isy2)
		allocWenoAux(nPML, nx, nz, &(Aux6->Aux2y));

#ifndef FREE_SURFACE
	if (border.isz2)
		allocWenoAux(nPML, nx, ny, &(Aux6->Aux2z));
#endif
}

__GLOBAL__
void weno_pml_x(
	FLOAT *h_W, float *h_Aux_x, float *Aux_x,
	float *pml_alpha_x, float *pml_beta_x, float *pml_d_x,
	int nPML, int _nx_, int _ny_, int _nz_, int FLAG, float DT)
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

	float beta_x = 0.0f;
	float d_x = 0.0f;
	float alpha_d_x = 0.0f;

	int stride = FLAG * (nx - nPML);
	CALCULATE3D(i0, j0, k0, 0, nPML, 0, ny, 0, nz)
	i = i0 + HALO + stride;
	j = j0 + HALO;
	k = k0 + HALO;
	index = INDEX(i, j, k);
	pos = Index3D(i0, j0, k0, nPML, ny, nz);

	beta_x = pml_beta_x[i];
	d_x = pml_d_x[i];
	alpha_d_x = d_x + pml_alpha_x[i];

	// CFS-PML memory variable, exact recursive-convolution update
	//     psi^{n+1} = b * psi^n + (d / (d + alpha)) * (1 - b) * D,
	//     b = exp( -(d + alpha) * DT ).
	// d already holds d/kappa and beta is 1/kappa, so (d + alpha) is the
	// complex-frequency-shifted decay rate d/kappa + alpha and this is the
	// Komatitsch & Martin (2007) recursion.  Replacing the forward-Euler
	// increment  d*D - (d+alpha)*psi*DT  by this form makes the memory
	// variable unconditionally stable: the old one grows whenever
	// (d + alpha) * DT exceeds 2 at the outer edge of a thick PML.
	float b_x = expf(-alpha_d_x * DT);
	float a_x = (alpha_d_x > 1.0e-20f) ? (d_x / alpha_d_x) * (1.0f - b_x) : 0.0f;
	float rDT = 1.0f / DT;

	for (int n = 0; n < 9; n++)
	{
		// D_x was stashed into h_Aux_x by cal_du_x and does not enter h_W
		// inside the slab, so the C-PML form can simply be added.
		float du = h_Aux_x[pos * WSIZE + n];
		float aux = Aux_x[pos * WSIZE + n];

		h_Aux_x[pos * WSIZE + n] = (b_x - 1.0f) * aux + a_x * du * rDT;
		h_W[index * WSIZE + n] += beta_x * (du - aux * DT);
	}
	END_CALCULATE3D()
}

__GLOBAL__
void weno_pml_y(
	FLOAT *h_W, float *h_Aux_y, float *Aux_y,
	float *pml_alpha_y, float *pml_beta_y, float *pml_d_y,
	int nPML, int _nx_, int _ny_, int _nz_, int FLAG, float DT)
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

	float beta_y = 0.0f;
	float d_y = 0.0f;
	float alpha_d_y = 0.0f;

	int stride = FLAG * (ny - nPML);
	CALCULATE3D(i0, j0, k0, 0, nx, 0, nPML, 0, nz)
	i = i0 + HALO;
	j = j0 + HALO + stride;
	k = k0 + HALO;
	index = INDEX(i, j, k);
	pos = Index3D(i0, j0, k0, nx, nPML, nz);

	beta_y = pml_beta_y[j];
	d_y = pml_d_y[j];
	alpha_d_y = d_y + pml_alpha_y[j];

	// CFS-PML exact recursion, see weno_pml_x
	float b_y = expf(-alpha_d_y * DT);
	float a_y = (alpha_d_y > 1.0e-20f) ? (d_y / alpha_d_y) * (1.0f - b_y) : 0.0f;
	float rDT = 1.0f / DT;

	for (int n = 0; n < 9; n++)
	{
		// D_y was stashed into h_Aux_y by cal_du_y
		float du = h_Aux_y[pos * WSIZE + n];
		float aux = Aux_y[pos * WSIZE + n];

		h_Aux_y[pos * WSIZE + n] = (b_y - 1.0f) * aux + a_y * du * rDT;
		h_W[index * WSIZE + n] += beta_y * (du - aux * DT);
	}
	END_CALCULATE3D()
}

__GLOBAL__
void weno_pml_z(
	FLOAT *h_W, float *h_Aux_z, float *Aux_z,
	float *pml_alpha_z, float *pml_beta_z, float *pml_d_z,
	int nPML, int _nx_, int _ny_, int _nz_, int FLAG, float DT)
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

	float beta_z = 0.0f;
	float d_z = 0.0f;
	float alpha_d_z = 0.0f;

	int stride = FLAG * (nz - nPML);
	CALCULATE3D(i0, j0, k0, 0, nx, 0, ny, 0, nPML)
	i = i0 + HALO;
	j = j0 + HALO;
	k = k0 + HALO + stride;
	index = INDEX(i, j, k);
	pos = Index3D(i0, j0, k0, nx, ny, nPML);

	beta_z = pml_beta_z[k];
	d_z = pml_d_z[k];
	alpha_d_z = d_z + pml_alpha_z[k];

	// CFS-PML exact recursion, see weno_pml_x
	float b_z = expf(-alpha_d_z * DT);
	float a_z = (alpha_d_z > 1.0e-20f) ? (d_z / alpha_d_z) * (1.0f - b_z) : 0.0f;
	float rDT = 1.0f / DT;

	for (int n = 0; n < 9; n++)
	{
		// D_z was stashed into h_Aux_z by cal_du_z
		float du = h_Aux_z[pos * WSIZE + n];
		float aux = Aux_z[pos * WSIZE + n];

		h_Aux_z[pos * WSIZE + n] = (b_z - 1.0f) * aux + a_z * du * rDT;
		h_W[index * WSIZE + n] += beta_z * (du - aux * DT);
	}
	END_CALCULATE3D()
}

void wenoPmlDeriv(GRID grid, WAVE wave, AUX6 Aux6, PML_ALPHA pml_alpha,
				  PML_BETA pml_beta, PML_D pml_d, MPI_BORDER border, float DT)

{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

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
		weno_pml_x<<<blockX, thread>>>(h_W, Aux6.Aux1x.h_Aux, Aux6.Aux1x.Aux, pml_alpha_x, pml_beta_x, pml_d_x, nPML, _nx_, _ny_, _nz_, 0, DT);
	if (border.isy1)
		weno_pml_y<<<blockY, thread>>>(h_W, Aux6.Aux1y.h_Aux, Aux6.Aux1y.Aux, pml_alpha_y, pml_beta_y, pml_d_y, nPML, _nx_, _ny_, _nz_, 0, DT);
	if (border.isz1)
		weno_pml_z<<<blockZ, thread>>>(h_W, Aux6.Aux1z.h_Aux, Aux6.Aux1z.Aux, pml_alpha_z, pml_beta_z, pml_d_z, nPML, _nx_, _ny_, _nz_, 0, DT);

	if (border.isx2)
		weno_pml_x<<<blockX, thread>>>(h_W, Aux6.Aux2x.h_Aux, Aux6.Aux2x.Aux, pml_alpha_x, pml_beta_x, pml_d_x, nPML, _nx_, _ny_, _nz_, 1, DT);
	if (border.isy2)
		weno_pml_y<<<blockY, thread>>>(h_W, Aux6.Aux2y.h_Aux, Aux6.Aux2y.Aux, pml_alpha_y, pml_beta_y, pml_d_y, nPML, _nx_, _ny_, _nz_, 1, DT);
#ifndef FREE_SURFACE
	if (border.isz2)
		weno_pml_z<<<blockZ, thread>>>(h_W, Aux6.Aux2z.h_Aux, Aux6.Aux2z.Aux, pml_alpha_z, pml_beta_z, pml_d_z, nPML, _nx_, _ny_, _nz_, 1, DT);
#endif

	CHECK(cudaDeviceSynchronize());

#else

	if (border.isx1)
		weno_pml_x(h_W, Aux6.Aux1x.h_Aux, Aux6.Aux1x.Aux, pml_alpha_x, pml_beta_x, pml_d_x, nPML, _nx_, _ny_, _nz_, 0, DT);
	if (border.isy1)
		weno_pml_y(h_W, Aux6.Aux1y.h_Aux, Aux6.Aux1y.Aux, pml_alpha_y, pml_beta_y, pml_d_y, nPML, _nx_, _ny_, _nz_, 0, DT);
	if (border.isz1)
		weno_pml_z(h_W, Aux6.Aux1z.h_Aux, Aux6.Aux1z.Aux, pml_alpha_z, pml_beta_z, pml_d_z, nPML, _nx_, _ny_, _nz_, 0, DT);

	if (border.isx2)
		weno_pml_x(h_W, Aux6.Aux2x.h_Aux, Aux6.Aux2x.Aux, pml_alpha_x, pml_beta_x, pml_d_x, nPML, _nx_, _ny_, _nz_, 1, DT);
	if (border.isy2)
		weno_pml_y(h_W, Aux6.Aux2y.h_Aux, Aux6.Aux2y.Aux, pml_alpha_y, pml_beta_y, pml_d_y, nPML, _nx_, _ny_, _nz_, 1, DT);
#ifndef FREE_SURFACE
	if (border.isz2)
		weno_pml_z(h_W, Aux6.Aux2z.h_Aux, Aux6.Aux2z.Aux, pml_alpha_z, pml_beta_z, pml_d_z, nPML, _nx_, _ny_, _nz_, 1, DT);
#endif

#endif // GPU_CUDA
}

typedef void (*WENO_PML_RK_FUNC)(float *h_Aux, float *Aux, float *m_Aux, long long num);

__GLOBAL__
void weno_pml_rk0(float *h_Aux, float *Aux, float *m_Aux, long long num)
{
#ifdef GPU_CUDA
	long long i = threadIdx.x + blockIdx.x * blockDim.x;
#else
	long long i = 0;
#endif
	float h_a, a;
	CALCULATE1D(i, 0, num)
	h_a = h_Aux[i];
	a = Aux[i];

	m_Aux[i] = a;
	Aux[i] = a + h_a;
	END_CALCULATE1D()
}

__GLOBAL__
void weno_pml_rk1(float *h_Aux, float *Aux, float *m_Aux, long long num)
{
#ifdef GPU_CUDA
	long long i = threadIdx.x + blockIdx.x * blockDim.x;
#else
	long long i = 0;
#endif
	float h_a, a;
	CALCULATE1D(i, 0, num)
	h_a = h_Aux[i];
	a = Aux[i];

	Aux[i] = 0.75 * m_Aux[i] + 0.25 * a + 0.25 * h_a;
	END_CALCULATE1D()
}

__GLOBAL__
void weno_pml_rk2(float *h_Aux, float *Aux, float *m_Aux, long long num)
{
#ifdef GPU_CUDA
	long long i = threadIdx.x + blockIdx.x * blockDim.x;
#else
	long long i = 0;
#endif
	float h_a, a;
	CALCULATE1D(i, 0, num)
	h_a = h_Aux[i];
	a = Aux[i];

	Aux[i] = (1.0 / 3.0) * m_Aux[i] + (2.0 / 3.0) * a + (2.0 / 3.0) * h_a;
	END_CALCULATE1D()
}

void wenoPmlRk(GRID grid, MPI_BORDER border, int irk, AUX6 Aux6)
{
	int nx = grid.nx;
	int ny = grid.ny;
	int nz = grid.nz;

	int nPML = grid.nPML;

	WENO_PML_RK_FUNC weno_pml_rk[3] = {weno_pml_rk0, weno_pml_rk1, weno_pml_rk2};
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
		weno_pml_rk[irk]<<<blockX, thread>>>(Aux6.Aux1x.h_Aux, Aux6.Aux1x.Aux, Aux6.Aux1x.m_Aux, numx);
	if (border.isy1)
		weno_pml_rk[irk]<<<blockY, thread>>>(Aux6.Aux1y.h_Aux, Aux6.Aux1y.Aux, Aux6.Aux1y.m_Aux, numy);
	if (border.isz1)
		weno_pml_rk[irk]<<<blockZ, thread>>>(Aux6.Aux1z.h_Aux, Aux6.Aux1z.Aux, Aux6.Aux1z.m_Aux, numz);

	if (border.isx2)
		weno_pml_rk[irk]<<<blockX, thread>>>(Aux6.Aux2x.h_Aux, Aux6.Aux2x.Aux, Aux6.Aux2x.m_Aux, numx);
	if (border.isy2)
		weno_pml_rk[irk]<<<blockY, thread>>>(Aux6.Aux2y.h_Aux, Aux6.Aux2y.Aux, Aux6.Aux2y.m_Aux, numy);
#ifndef FREE_SURFACE
	if (border.isz2)
		weno_pml_rk[irk]<<<blockZ, thread>>>(Aux6.Aux2z.h_Aux, Aux6.Aux2z.Aux, Aux6.Aux2z.m_Aux, numz);
#endif

	// Without this the first error raised here only surfaces at the next
	// cudaGetLastError(), i.e. at alternative_flux_FD.cpp:682 in the next RK
	// stage, which makes the crash look like it comes from the flux kernel.
	CHECK(cudaGetLastError());

#else

	if (border.isx1)
		weno_pml_rk[irk](Aux6.Aux1x.h_Aux, Aux6.Aux1x.Aux, Aux6.Aux1x.m_Aux, numx);
	if (border.isy1)
		weno_pml_rk[irk](Aux6.Aux1y.h_Aux, Aux6.Aux1y.Aux, Aux6.Aux1y.m_Aux, numy);
	if (border.isz1)
		weno_pml_rk[irk](Aux6.Aux1z.h_Aux, Aux6.Aux1z.Aux, Aux6.Aux1z.m_Aux, numz);

	if (border.isx2)
		weno_pml_rk[irk](Aux6.Aux2x.h_Aux, Aux6.Aux2x.Aux, Aux6.Aux2x.m_Aux, numx);
	if (border.isy2)
		weno_pml_rk[irk](Aux6.Aux2y.h_Aux, Aux6.Aux2y.Aux, Aux6.Aux2y.m_Aux, numy);
#ifndef FREE_SURFACE
	if (border.isz2)
		weno_pml_rk[irk](Aux6.Aux2z.h_Aux, Aux6.Aux2z.Aux, Aux6.Aux2z.m_Aux, numz);
#endif

#endif
}
