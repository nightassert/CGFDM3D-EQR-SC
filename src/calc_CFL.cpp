/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: cal_CFL.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2022-11-01
*   Discription: Calculate the CFL condition
*
*   Update: Tianhong Xu, 12231218@mail.sustech.edu.cn
*   Update Time: 2023-11-16
*   Update Content: Add SCFDM
*
*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*
=================================================================*/

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
				hmin = MIN(hmin, d);
				hmax = MAX(hmax, d);
			}

	END_LOOP3D()

	h_min_max[0] = hmin;
	h_min_max[1] = hmax;
}

// Smallest distance between the grid planes of each coordinate direction:
//
//     h_d = J / |con_d|,      con_d = cofactor row of d(x,y,z)/d(xi,eta,zt).
//
// These are the same covariant components and cofactors that
// constructCJM() builds in contravariant.cpp, but here the central
// differences are taken without the 1/DH factors, so h_d comes out in meters
// and no DH argument is needed.
//
// calculate_DH_Range() above measures the distance from a point to a slanted
// plane through one neighbour in each direction; that is not the distance a
// wave travels between two xi = const planes, which is what the directional
// CFL number of the WENO flux uses (du = DT*con/J * (F_i - F_{i-1}) / DH, so
// the effective spacing along xi is J/|con_xi|).
void calculate_direction_spacing(
	float *coord,
	float h_min[3],
	int _nx_, int _ny_, int _nz_)
{
	int i, j, k, m;
	long long ip = 0, im = 0, jp = 0, jm = 0, kp = 0, km = 0;

	int _nx = _nx_ - HALO;
	int _ny = _ny_ - HALO;
	int _nz = _nz_ - HALO;

	double dv[3][3]; // dv[d][c] = d(coord[c]) / d(direction d), per index step
	double con[3][3];
	double J = 0.0;
	double d = 0.0;
	double h[3] = {1.0e20, 1.0e20, 1.0e20};

	FOR_LOOP3D(i, j, k, HALO, _nx, HALO, _ny, HALO, _nz)

	ip = INDEX(i + 1, j, k) * CSIZE;
	im = INDEX(i - 1, j, k) * CSIZE;
	jp = INDEX(i, j + 1, k) * CSIZE;
	jm = INDEX(i, j - 1, k) * CSIZE;
	kp = INDEX(i, j, k + 1) * CSIZE;
	km = INDEX(i, j, k - 1) * CSIZE;

	for (m = 0; m < 3; m++)
	{
		dv[0][m] = 0.5 * ((double)coord[ip + m] - (double)coord[im + m]);
		dv[1][m] = 0.5 * ((double)coord[jp + m] - (double)coord[jm + m]);
		dv[2][m] = 0.5 * ((double)coord[kp + m] - (double)coord[km + m]);
	}

	con[0][0] = dv[1][1] * dv[2][2] - dv[1][2] * dv[2][1]; // xi_x * J
	con[0][1] = dv[1][2] * dv[2][0] - dv[1][0] * dv[2][2]; // xi_y * J
	con[0][2] = dv[1][0] * dv[2][1] - dv[1][1] * dv[2][0]; // xi_z * J

	con[1][0] = dv[2][1] * dv[0][2] - dv[2][2] * dv[0][1]; // et_x * J
	con[1][1] = dv[2][2] * dv[0][0] - dv[2][0] * dv[0][2]; // et_y * J
	con[1][2] = dv[2][0] * dv[0][1] - dv[2][1] * dv[0][0]; // et_z * J

	con[2][0] = dv[0][1] * dv[1][2] - dv[0][2] * dv[1][1]; // zt_x * J
	con[2][1] = dv[0][2] * dv[1][0] - dv[0][0] * dv[1][2]; // zt_y * J
	con[2][2] = dv[0][0] * dv[1][1] - dv[0][1] * dv[1][0]; // zt_z * J

	J = dv[0][0] * con[0][0] + dv[0][1] * con[0][1] + dv[0][2] * con[0][2];

	for (m = 0; m < 3; m++)
	{
		d = sqrt(con[m][0] * con[m][0] + con[m][1] * con[m][1] + con[m][2] * con[m][2]);
		if (d > 1.0e-12)
		{
			d = J / d;
			if (d > 0.0 && d < h[m])
				h[m] = d;
		}
	}

	END_LOOP3D()

	h_min[0] = (float)h[0];
	h_min[1] = (float)h[1];
	h_min[2] = (float)h[2];
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

void medium2structure(GRID grid, STRUCTURE structure, float *medium)
{
	long long i = 0;

	long long num = grid._nx_ * grid._ny_ * grid._nz_;

	FOR_LOOP1D(i, 0, num)
	structure.Vs[i] = medium[i * MSIZE + 0];
	structure.Vp[i] = medium[i * MSIZE + 1];
	structure.rho[i] = medium[i * MSIZE + 2];
	END_LOOP1D()
}

void calc_CFL(GRID grid, float *coord, float *medium, PARAMS &params)
{

	int nx = grid.nx;
	int ny = grid.ny;
	int nz = grid.nz;

	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	float h_min_max[2];

	float h_dir[3];
	float h_dir_min[3];

	float Vs_min_max[2];
	float Vp_min_max[2];
	float rho_min_max[2];

	long long num = nx * ny * nz;

	calculate_DH_Range(coord, h_min_max, _nx_, _ny_, _nz_);
	calculate_direction_spacing(coord, h_dir, _nx_, _ny_, _nz_);

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

	float H_min, H_max;
	float Vs_min, Vs_max;
	float Vp_min, Vp_max;
	float rho_min, rho_max;

	MPI_Allreduce(&h_min_max[0], &H_min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&h_min_max[1], &H_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

	MPI_Allreduce(h_dir, h_dir_min, 3, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);

	// A degenerate cell can make one directional spacing vanish or come out
	// non-finite; fall back to the cell-size estimate rather than clamping the
	// time step to an absurd value.
	for (int m = 0; m < 3; m++)
	{
		if (!(h_dir_min[m] > 0.0f) || !(h_dir_min[m] < 1.0e19f))
			h_dir_min[m] = H_min;
	}

	MPI_Allreduce(&Vs_min_max[0], &Vs_min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&Vs_min_max[1], &Vs_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

	MPI_Allreduce(&Vp_min_max[0], &Vp_min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&Vp_min_max[1], &Vp_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

	MPI_Allreduce(&rho_min_max[0], &rho_min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
	MPI_Allreduce(&rho_min_max[1], &rho_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	// Lax-Friedrichs wave-speed bound used by the WENO flux.
	g_vp_max = Vp_max;

	// Time-step bound.
	//
	// CGFDM keeps the historical 1-D bound, 1.34 * H_min / Vp_max, with H_min
	// the plane distance from calculate_DH_Range().  Runs with 2/2/2 and this
	// bound have always been stable, so it is left untouched.
	//
	// The WENO5 + TVD-RK3 scheme is different: the Rusanov (Lax-Friedrichs)
	// flux uses alpha = Vp_max in *all three* directions, and the directional
	// operators add, so the quantity that must stay below the 1-D limit of
	// the integrator is the SUM of the three CFL numbers:
	//
	//     Vp_max * DT * (1/h_xi + 1/h_eta + 1/h_zeta) <= 1.34.
	//
	// The 1-D stability limit of this discrete operator was measured to be
	// CFL = 1.34 (stable) / 1.40 (unstable); in 3-D the same operator turns
	// unstable at a summed CFL of roughly 1.7-1.8, so 1.34 leaves margin.
	// With only the vertical term checked, the fault model (DH = 200 m,
	// h_zeta ~ 1.2e2 m, Vp_max = 8056 m/s, DT = 1.3e-2 s) passes with a 3-D
	// sum of ~2, the scheme grows the grid-scale content injected by the
	// finite-fault source at ~1.2-1.5x per step, and the field overflows a
	// few hundred steps in -- which looks like the source blowing up.
	float slow3D = 1.0f / h_dir_min[0] + 1.0f / h_dir_min[1] + 1.0f / h_dir_min[2];

	float dtmax;
	if (SCHEME_WENO == params.scheme)
		dtmax = 1.34f / (Vp_max * slow3D);
	else
		dtmax = 1.34f * H_min / Vp_max;

	float DT_input = (float)params.DT;

	// Clamp the step for the WENO scheme.  dtmax was reduced over the whole
	// communicator, so every rank takes the same branch and keeps the same
	// DT -> the MPI exchange stays consistent.  The reduction of TMAX in
	// propagate() then produces more, smaller steps for the same duration.
	if (SCHEME_WENO == params.scheme && params.DT > dtmax)
	{
		params.DT = dtmax;
		g_params.DT = dtmax;
	}

	if (0 == thisRank)
	{
		printf("H   Range: %5.4e ~ %5.4e\n", H_min, H_max);
		printf("Vs  Range: %5.4e ~ %5.4e\n", Vs_min, Vs_max);
		printf("Vp  Range: %5.4e ~ %5.4e\n", Vp_min, Vp_max);
		printf("rho Range: %5.4e ~ %5.4e\n", rho_min, rho_max);

		if (SCHEME_WENO == params.scheme)
		{
			printf("h   min per direction (x, y, z): %5.4e %5.4e %5.4e\n",
				   h_dir_min[0], h_dir_min[1], h_dir_min[2]);
			printf("3-D CFL: Vp_max*DT*(1/hx+1/hy+1/hz) = %.3f  (limit 1.34)\n",
				   Vp_max * (float)params.DT * slow3D);
		}

		printf("dtmax = %5.2e, DT = %5.2e\n", dtmax, params.DT);

		if ((float)params.DT < DT_input)
		{
			printf("\n");
			printf("****************************************************************\n");
			printf("  WARNING: the WENO scheme cannot run stably at DT = %5.2e s.\n", DT_input);
			printf("  The 3-D CFL number would be %.3f (limit 1.34); DT has been\n",
				   Vp_max * DT_input * slow3D);
			printf("  reduced to %5.2e s for this run, so the number of time steps\n", (float)params.DT);
			printf("  grows by %.2f x (NT = %d) and TMAX is unchanged.\n",
				   DT_input / (float)params.DT, (int)(params.TMAX / params.DT));
			printf("  Put \"DT\": %5.2e (or smaller) in params.json to silence this.\n", dtmax);
			printf("****************************************************************\n");
			printf("\n");
		}

		if (params.DT > dtmax)
		{
			printf("The parameters can't afford the CFL condition!\n");
			MPI_Abort(MPI_COMM_WORLD, 110);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
}
