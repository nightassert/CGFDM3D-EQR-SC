/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: alternative_flux_FD.cpp
*   Author: Tianhong Xu, 12231218@mail.sustech.edu.cn
*   Created Time: 2023-10-22
*   Discription: Shock Capturing Finite Difference Method kernel file
*
*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*
=================================================================*/

// ! For alternative flux finite difference by Tianhong Xu
#ifdef SCFDM
#include "header.h"

#ifdef PML
#define TIMES_PML_BETA_X *pml_beta_x
#define TIMES_PML_BETA_Y *pml_beta_y
#define TIMES_PML_BETA_Z *pml_beta_z
#else
#define TIMES_PML_BETA_X
#define TIMES_PML_BETA_Y
#define TIMES_PML_BETA_Z
#endif

#define order2_approximation(u1, u2, u3, u4, u5, u6) (1.0f / 48 * (-5 * u1 + 39 * u2 - 34 * u3 - 34 * u4 + 39 * u5 - 5 * u6))
#define order4_approximation(u1, u2, u3, u4, u5, u6) (1.0f / 2 * (u1 - 3 * u2 + 2 * u3 + 2 * u4 - 3 * u5 + u6))
#define WENO5_interp_weights(u1, u2, u3, u4, u5, w) ((0.375f * u3 + 0.75f * u4 - 0.125f * u5) * w[0] + (-0.125f * u2 + 0.75f * u3 + 0.375f * u4) * w[1] + (0.375f * u1 - 1.25f * u2 + 1.875f * u3) * w[2])

#ifdef LF
extern float vp_max_for_SCFDM;
#endif

// WENO5 interpolation scheme
__DEVICE__
float WENO5_interpolation(float u1, float u2, float u3, float u4, float u5)
{
    // small stencils
    float v1 = 0.375f * u3 + 0.75f * u4 - 0.125f * u5;
    float v2 = -0.125f * u2 + 0.75f * u3 + 0.375f * u4;
    float v3 = 0.375f * u1 - 1.25f * u2 + 1.875f * u3;

    // linear weights
    float d1 = 0.3125f, d2 = 0.625f, d3 = 0.0625f;

    // smoothness indicators
    float WENO_beta1 = 0.3333333333f * (10 * u3 * u3 - 31 * u3 * u4 + 25 * u4 * u4 + 11 * u3 * u5 - 19 * u4 * u5 + 4 * u5 * u5);
    float WENO_beta2 = 0.3333333333f * (4 * u2 * u2 - 13 * u2 * u3 + 13 * u3 * u3 + 5 * u2 * u4 - 13 * u3 * u4 + 4 * u4 * u4);
    float WENO_beta3 = 0.3333333333f * (4 * u1 * u1 - 19 * u1 * u2 + 25 * u2 * u2 + 11 * u1 * u3 - 31 * u2 * u3 + 10 * u3 * u3);

    float epsilon = 1e-6;

    float WENO_alpha1 = d1 / ((WENO_beta1 + epsilon) * (WENO_beta1 + epsilon));
    float WENO_alpha2 = d2 / ((WENO_beta2 + epsilon) * (WENO_beta2 + epsilon));
    float WENO_alpha3 = d3 / ((WENO_beta3 + epsilon) * (WENO_beta3 + epsilon));

    // nonlinear weights
    float w1 = WENO_alpha1 / (WENO_alpha1 + WENO_alpha2 + WENO_alpha3);
    float w2 = WENO_alpha2 / (WENO_alpha1 + WENO_alpha2 + WENO_alpha3);
    float w3 = WENO_alpha3 / (WENO_alpha1 + WENO_alpha2 + WENO_alpha3);

    // WENO interpolation
    return w1 * v1 + w2 * v2 + w3 * v3;
}

// Calculate flux
__GLOBAL__
void cal_flux(FLOAT *fu, FLOAT *gu, FLOAT *hu, FLOAT *u, FLOAT *CJM, int _nx_, int _ny_, int _nz_, MPI_COORD thisMPICoord, PARAMS params)
{
#ifdef GPU_CUDA
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
#else
    int i = 0;
    int j = 0;
    int k = 0;
#endif
    long long index;
    long long index_on_surf;

    float mu;
    float lambda;
    float buoyancy;

    float xi_x_J;
    float xi_y_J;
    float xi_z_J;
    float et_x_J;
    float et_y_J;
    float et_z_J;
    float zt_x_J;
    float zt_y_J;
    float zt_z_J;

    CALCULATE3D(i, j, k, 0, _nx_, 0, _ny_, 0, _nz_)
    index = INDEX(i, j, k);
    // index * VARSIZE + VAR
    xi_x_J = CJM[index * CJMSIZE + 0];
    xi_y_J = CJM[index * CJMSIZE + 1];
    xi_z_J = CJM[index * CJMSIZE + 2];
    et_x_J = CJM[index * CJMSIZE + 3];
    et_y_J = CJM[index * CJMSIZE + 4];
    et_z_J = CJM[index * CJMSIZE + 5];
    zt_x_J = CJM[index * CJMSIZE + 6];
    zt_y_J = CJM[index * CJMSIZE + 7];
    zt_z_J = CJM[index * CJMSIZE + 8];

    mu = CJM[index * CJMSIZE + 10];
    lambda = CJM[index * CJMSIZE + 11];
    buoyancy = CJM[index * CJMSIZE + 12];
    buoyancy *= Crho;

    fu[index * WSIZE + 0] = -(xi_x_J * u[index * WSIZE + 6]) * buoyancy;
    fu[index * WSIZE + 1] = -(xi_y_J * u[index * WSIZE + 7]) * buoyancy;
    fu[index * WSIZE + 2] = -(xi_z_J * u[index * WSIZE + 8]) * buoyancy;
    fu[index * WSIZE + 3] = -(xi_x_J * u[index * WSIZE + 7]) * 0.5 * buoyancy - (xi_y_J * u[index * WSIZE + 6]) * 0.5 * buoyancy;
    fu[index * WSIZE + 4] = -(xi_y_J * u[index * WSIZE + 8]) * 0.5 * buoyancy - (xi_z_J * u[index * WSIZE + 7]) * 0.5 * buoyancy;
    fu[index * WSIZE + 5] = -(xi_x_J * u[index * WSIZE + 8]) * 0.5 * buoyancy - (xi_z_J * u[index * WSIZE + 6]) * 0.5 * buoyancy;
    fu[index * WSIZE + 6] = -xi_x_J * (2 * mu * u[index * WSIZE + 0] + lambda * (u[index * WSIZE + 0] + u[index * WSIZE + 1] + u[index * WSIZE + 2])) - 2 * mu * xi_y_J * u[index * WSIZE + 3] - 2 * mu * xi_z_J * u[index * WSIZE + 5];
    fu[index * WSIZE + 7] = -xi_y_J * (2 * mu * u[index * WSIZE + 1] + lambda * (u[index * WSIZE + 0] + u[index * WSIZE + 1] + u[index * WSIZE + 2])) - 2 * mu * xi_x_J * u[index * WSIZE + 3] - 2 * mu * xi_z_J * u[index * WSIZE + 4];
    fu[index * WSIZE + 8] = -xi_z_J * (2 * mu * u[index * WSIZE + 2] + lambda * (u[index * WSIZE + 0] + u[index * WSIZE + 1] + u[index * WSIZE + 2])) - 2 * mu * xi_x_J * u[index * WSIZE + 5] - 2 * mu * xi_y_J * u[index * WSIZE + 4];

    gu[index * WSIZE + 0] = -(et_x_J * u[index * WSIZE + 6]) * buoyancy;
    gu[index * WSIZE + 1] = -(et_y_J * u[index * WSIZE + 7]) * buoyancy;
    gu[index * WSIZE + 2] = -(et_z_J * u[index * WSIZE + 8]) * buoyancy;
    gu[index * WSIZE + 3] = -(et_x_J * u[index * WSIZE + 7]) * 0.5 * buoyancy - (et_y_J * u[index * WSIZE + 6]) * 0.5 * buoyancy;
    gu[index * WSIZE + 4] = -(et_y_J * u[index * WSIZE + 8]) * 0.5 * buoyancy - (et_z_J * u[index * WSIZE + 7]) * 0.5 * buoyancy;
    gu[index * WSIZE + 5] = -(et_x_J * u[index * WSIZE + 8]) * 0.5 * buoyancy - (et_z_J * u[index * WSIZE + 6]) * 0.5 * buoyancy;
    gu[index * WSIZE + 6] = -et_x_J * (2 * mu * u[index * WSIZE + 0] + lambda * (u[index * WSIZE + 0] + u[index * WSIZE + 1] + u[index * WSIZE + 2])) - 2 * mu * et_y_J * u[index * WSIZE + 3] - 2 * mu * et_z_J * u[index * WSIZE + 5];
    gu[index * WSIZE + 7] = -et_y_J * (2 * mu * u[index * WSIZE + 1] + lambda * (u[index * WSIZE + 0] + u[index * WSIZE + 1] + u[index * WSIZE + 2])) - 2 * mu * et_x_J * u[index * WSIZE + 3] - 2 * mu * et_z_J * u[index * WSIZE + 4];
    gu[index * WSIZE + 8] = -et_z_J * (2 * mu * u[index * WSIZE + 2] + lambda * (u[index * WSIZE + 0] + u[index * WSIZE + 1] + u[index * WSIZE + 2])) - 2 * mu * et_x_J * u[index * WSIZE + 5] - 2 * mu * et_y_J * u[index * WSIZE + 4];

    hu[index * WSIZE + 0] = -(zt_x_J * u[index * WSIZE + 6]) * buoyancy;
    hu[index * WSIZE + 1] = -(zt_y_J * u[index * WSIZE + 7]) * buoyancy;
    hu[index * WSIZE + 2] = -(zt_z_J * u[index * WSIZE + 8]) * buoyancy;
    hu[index * WSIZE + 3] = -(zt_x_J * u[index * WSIZE + 7]) * 0.5 * buoyancy - (zt_y_J * u[index * WSIZE + 6]) * 0.5 * buoyancy;
    hu[index * WSIZE + 4] = -(zt_y_J * u[index * WSIZE + 8]) * 0.5 * buoyancy - (zt_z_J * u[index * WSIZE + 7]) * 0.5 * buoyancy;
    hu[index * WSIZE + 5] = -(zt_x_J * u[index * WSIZE + 8]) * 0.5 * buoyancy - (zt_z_J * u[index * WSIZE + 6]) * 0.5 * buoyancy;
    hu[index * WSIZE + 6] = -zt_x_J * (2 * mu * u[index * WSIZE + 0] + lambda * (u[index * WSIZE + 0] + u[index * WSIZE + 1] + u[index * WSIZE + 2])) - 2 * mu * zt_y_J * u[index * WSIZE + 3] - 2 * mu * zt_z_J * u[index * WSIZE + 5];
    hu[index * WSIZE + 7] = -zt_y_J * (2 * mu * u[index * WSIZE + 1] + lambda * (u[index * WSIZE + 0] + u[index * WSIZE + 1] + u[index * WSIZE + 2])) - 2 * mu * zt_x_J * u[index * WSIZE + 3] - 2 * mu * zt_z_J * u[index * WSIZE + 4];
    hu[index * WSIZE + 8] = -zt_z_J * (2 * mu * u[index * WSIZE + 2] + lambda * (u[index * WSIZE + 0] + u[index * WSIZE + 1] + u[index * WSIZE + 2])) - 2 * mu * zt_x_J * u[index * WSIZE + 5] - 2 * mu * zt_y_J * u[index * WSIZE + 4];

    END_CALCULATE3D()
}

__GLOBAL__
void wave_deriv_alternative_flux_FD(FLOAT *Fu_ip12x, FLOAT *Fu_ip12y, FLOAT *Fu_ip12z,
                                    FLOAT *Fu, FLOAT *Gu, FLOAT *Hu, FLOAT *h_W, FLOAT *W, FLOAT *CJM,
#ifdef PML
                                    PML_BETA pml_beta,
#endif
                                    int _nx_, int _ny_, int _nz_, float rDH, int FB1, int FB2, int FB3, float DT
#ifdef LF
                                    ,
                                    float alpha
#endif
)
{

    int _nx = _nx_ - HALO;
    int _ny = _ny_ - HALO;
    int _nz = _nz_ - HALO;

#ifdef GPU_CUDA
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
#else
    int i = 0;
    int j = 0;
    int k = 0;
#endif

#ifdef PML
    float pml_beta_x;
    float pml_beta_y;
    float pml_beta_z;
#endif

    float mu;
    float lambda;
    float buoyancy;

    float xi_x_J_h;
    float xi_y_J_h;
    float xi_z_J_h;
    float et_x_J_h;
    float et_y_J_h;
    float et_z_J_h;
    float zt_x_J_h;
    float zt_y_J_h;
    float zt_z_J_h;

    long long idx_n2, idx_n1, idx, idx_p1, idx_p2, idx_p3;

    float u_ip12n[9], u_ip12p[9];
    float fu_ip12n[9], fu_ip12p[9];
    float weights_n[3], weights_p[3];

    // * X direction
    CALCULATE3D(i, j, k, HALO - 1, _nx, HALO, _ny, HALO, _nz)

#ifdef PML
    pml_beta_x = pml_beta.x[i];
    pml_beta_y = pml_beta.y[j];
    pml_beta_z = pml_beta.z[k];
#endif

    idx_n2 = INDEX(i - 2, j, k);
    idx_n1 = INDEX(i - 1, j, k);
    idx = INDEX(i, j, k);
    idx_p1 = INDEX(i + 1, j, k);
    idx_p2 = INDEX(i + 2, j, k);
    idx_p3 = INDEX(i + 3, j, k);

    xi_x_J_h = CJM[idx * CJMSIZE + 0];
    xi_y_J_h = CJM[idx * CJMSIZE + 1];
    xi_z_J_h = CJM[idx * CJMSIZE + 2];

    mu = CJM[idx * CJMSIZE + 10];
    lambda = CJM[idx * CJMSIZE + 11];
    buoyancy = CJM[idx * CJMSIZE + 12];
    buoyancy *= Crho;

    // Shock capture interpolation
    for (int n = 0; n < 9; n++)
    {
        u_ip12n[n] = WENO5_interpolation(W[idx_n2 * WSIZE + n], W[idx_n1 * WSIZE + n], W[idx * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx_p2 * WSIZE + n]);
        u_ip12p[n] = WENO5_interpolation(W[idx_p3 * WSIZE + n], W[idx_p2 * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx * WSIZE + n], W[idx_n1 * WSIZE + n]);
    }

#ifdef LF
    // Riemann solver: Lax-Friedrichs
    fu_ip12n[0] = -(xi_x_J_h * u_ip12n[6]) * buoyancy;
    fu_ip12n[1] = -(xi_y_J_h * u_ip12n[7]) * buoyancy;
    fu_ip12n[2] = -(xi_z_J_h * u_ip12n[8]) * buoyancy;
    fu_ip12n[3] = -(xi_x_J_h * u_ip12n[7]) * 0.5 * buoyancy - (xi_y_J_h * u_ip12n[6]) * 0.5 * buoyancy;
    fu_ip12n[4] = -(xi_y_J_h * u_ip12n[8]) * 0.5 * buoyancy - (xi_z_J_h * u_ip12n[7]) * 0.5 * buoyancy;
    fu_ip12n[5] = -(xi_x_J_h * u_ip12n[8]) * 0.5 * buoyancy - (xi_z_J_h * u_ip12n[6]) * 0.5 * buoyancy;
    fu_ip12n[6] = -xi_x_J_h * (2 * mu * u_ip12n[0] + lambda * (u_ip12n[0] + u_ip12n[1] + u_ip12n[2])) - 2 * mu * xi_y_J_h * u_ip12n[3] - 2 * mu * xi_z_J_h * u_ip12n[5];
    fu_ip12n[7] = -xi_y_J_h * (2 * mu * u_ip12n[1] + lambda * (u_ip12n[0] + u_ip12n[1] + u_ip12n[2])) - 2 * mu * xi_x_J_h * u_ip12n[3] - 2 * mu * xi_z_J_h * u_ip12n[4];
    fu_ip12n[8] = -xi_z_J_h * (2 * mu * u_ip12n[2] + lambda * (u_ip12n[0] + u_ip12n[1] + u_ip12n[2])) - 2 * mu * xi_x_J_h * u_ip12n[5] - 2 * mu * xi_y_J_h * u_ip12n[4];

    fu_ip12p[0] = -(xi_x_J_h * u_ip12p[6]) * buoyancy;
    fu_ip12p[1] = -(xi_y_J_h * u_ip12p[7]) * buoyancy;
    fu_ip12p[2] = -(xi_z_J_h * u_ip12p[8]) * buoyancy;
    fu_ip12p[3] = -(xi_x_J_h * u_ip12p[7]) * 0.5 * buoyancy - (xi_y_J_h * u_ip12p[6]) * 0.5 * buoyancy;
    fu_ip12p[4] = -(xi_y_J_h * u_ip12p[8]) * 0.5 * buoyancy - (xi_z_J_h * u_ip12p[7]) * 0.5 * buoyancy;
    fu_ip12p[5] = -(xi_x_J_h * u_ip12p[8]) * 0.5 * buoyancy - (xi_z_J_h * u_ip12p[6]) * 0.5 * buoyancy;
    fu_ip12p[6] = -xi_x_J_h * (2 * mu * u_ip12p[0] + lambda * (u_ip12p[0] + u_ip12p[1] + u_ip12p[2])) - 2 * mu * xi_y_J_h * u_ip12p[3] - 2 * mu * xi_z_J_h * u_ip12p[5];
    fu_ip12p[7] = -xi_y_J_h * (2 * mu * u_ip12p[1] + lambda * (u_ip12p[0] + u_ip12p[1] + u_ip12p[2])) - 2 * mu * xi_x_J_h * u_ip12p[3] - 2 * mu * xi_z_J_h * u_ip12p[4];
    fu_ip12p[8] = -xi_z_J_h * (2 * mu * u_ip12p[2] + lambda * (u_ip12p[0] + u_ip12p[1] + u_ip12p[2])) - 2 * mu * xi_x_J_h * u_ip12p[5] - 2 * mu * xi_y_J_h * u_ip12p[4];
#endif

    for (int n = 0; n < 9; n++)
    {
#ifdef LF
        Fu_ip12x[idx * WSIZE + n] = 0.5f * (fu_ip12p[n] + fu_ip12n[n] - alpha * (u_ip12p[n] - u_ip12n[n])) - 1.0f / 24 * order2_approximation(Fu[idx_n2 * WSIZE + n], Fu[idx_n1 * WSIZE + n], Fu[idx * WSIZE + n], Fu[idx_p1 * WSIZE + n], Fu[idx_p2 * WSIZE + n], Fu[idx_p3 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Fu[idx_n2 * WSIZE + n], Fu[idx_n1 * WSIZE + n], Fu[idx * WSIZE + n], Fu[idx_p1 * WSIZE + n], Fu[idx_p2 * WSIZE + n], Fu[idx_p3 * WSIZE + n]);
#endif // LF
    }
    END_CALCULATE3D()

    // * Y direction
    CALCULATE3D(i, j, k, HALO, _nx, HALO - 1, _ny, HALO, _nz)

#ifdef PML
    pml_beta_x = pml_beta.x[i];
    pml_beta_y = pml_beta.y[j];
    pml_beta_z = pml_beta.z[k];
#endif

    idx_n2 = INDEX(i, j - 2, k);
    idx_n1 = INDEX(i, j - 1, k);
    idx = INDEX(i, j, k);
    idx_p1 = INDEX(i, j + 1, k);
    idx_p2 = INDEX(i, j + 2, k);
    idx_p3 = INDEX(i, j + 3, k);

    et_x_J_h = CJM[idx * CJMSIZE + 3];
    et_y_J_h = CJM[idx * CJMSIZE + 4];
    et_z_J_h = CJM[idx * CJMSIZE + 5];

    mu = CJM[idx * CJMSIZE + 10];
    lambda = CJM[idx * CJMSIZE + 11];
    buoyancy = CJM[idx * CJMSIZE + 12];
    buoyancy *= Crho;

    // Shock capture interpolation
    for (int n = 0; n < 9; n++)
    {
        u_ip12n[n] = WENO5_interpolation(W[idx_n2 * WSIZE + n], W[idx_n1 * WSIZE + n], W[idx * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx_p2 * WSIZE + n]);
        u_ip12p[n] = WENO5_interpolation(W[idx_p3 * WSIZE + n], W[idx_p2 * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx * WSIZE + n], W[idx_n1 * WSIZE + n]);
    }

#ifdef LF
    // Riemann solver: Lax-Friedrichs
    fu_ip12n[0] = -(et_x_J_h * u_ip12n[6]) * buoyancy;
    fu_ip12n[1] = -(et_y_J_h * u_ip12n[7]) * buoyancy;
    fu_ip12n[2] = -(et_z_J_h * u_ip12n[8]) * buoyancy;
    fu_ip12n[3] = -(et_x_J_h * u_ip12n[7]) * 0.5 * buoyancy - (et_y_J_h * u_ip12n[6]) * 0.5 * buoyancy;
    fu_ip12n[4] = -(et_y_J_h * u_ip12n[8]) * 0.5 * buoyancy - (et_z_J_h * u_ip12n[7]) * 0.5 * buoyancy;
    fu_ip12n[5] = -(et_x_J_h * u_ip12n[8]) * 0.5 * buoyancy - (et_z_J_h * u_ip12n[6]) * 0.5 * buoyancy;
    fu_ip12n[6] = -et_x_J_h * (2 * mu * u_ip12n[0] + lambda * (u_ip12n[0] + u_ip12n[1] + u_ip12n[2])) - 2 * mu * et_y_J_h * u_ip12n[3] - 2 * mu * et_z_J_h * u_ip12n[5];
    fu_ip12n[7] = -et_y_J_h * (2 * mu * u_ip12n[1] + lambda * (u_ip12n[0] + u_ip12n[1] + u_ip12n[2])) - 2 * mu * et_x_J_h * u_ip12n[3] - 2 * mu * et_z_J_h * u_ip12n[4];
    fu_ip12n[8] = -et_z_J_h * (2 * mu * u_ip12n[2] + lambda * (u_ip12n[0] + u_ip12n[1] + u_ip12n[2])) - 2 * mu * et_x_J_h * u_ip12n[5] - 2 * mu * et_y_J_h * u_ip12n[4];

    fu_ip12p[0] = -(et_x_J_h * u_ip12p[6]) * buoyancy;
    fu_ip12p[1] = -(et_y_J_h * u_ip12p[7]) * buoyancy;
    fu_ip12p[2] = -(et_z_J_h * u_ip12p[8]) * buoyancy;
    fu_ip12p[3] = -(et_x_J_h * u_ip12p[7]) * 0.5 * buoyancy - (et_y_J_h * u_ip12p[6]) * 0.5 * buoyancy;
    fu_ip12p[4] = -(et_y_J_h * u_ip12p[8]) * 0.5 * buoyancy - (et_z_J_h * u_ip12p[7]) * 0.5 * buoyancy;
    fu_ip12p[5] = -(et_x_J_h * u_ip12p[8]) * 0.5 * buoyancy - (et_z_J_h * u_ip12p[6]) * 0.5 * buoyancy;
    fu_ip12p[6] = -et_x_J_h * (2 * mu * u_ip12p[0] + lambda * (u_ip12p[0] + u_ip12p[1] + u_ip12p[2])) - 2 * mu * et_y_J_h * u_ip12p[3] - 2 * mu * et_z_J_h * u_ip12p[5];
    fu_ip12p[7] = -et_y_J_h * (2 * mu * u_ip12p[1] + lambda * (u_ip12p[0] + u_ip12p[1] + u_ip12p[2])) - 2 * mu * et_x_J_h * u_ip12p[3] - 2 * mu * et_z_J_h * u_ip12p[4];
    fu_ip12p[8] = -et_z_J_h * (2 * mu * u_ip12p[2] + lambda * (u_ip12p[0] + u_ip12p[1] + u_ip12p[2])) - 2 * mu * et_x_J_h * u_ip12p[5] - 2 * mu * et_y_J_h * u_ip12p[4];
#endif

    for (int n = 0; n < 9; n++)
    {
#ifdef LF
        Fu_ip12y[idx * WSIZE + n] = 0.5f * (fu_ip12p[n] + fu_ip12n[n] - alpha * (u_ip12p[n] - u_ip12n[n])) - 1.0f / 24 * order2_approximation(Gu[idx_n2 * WSIZE + n], Gu[idx_n1 * WSIZE + n], Gu[idx * WSIZE + n], Gu[idx_p1 * WSIZE + n], Gu[idx_p2 * WSIZE + n], Gu[idx_p3 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Gu[idx_n2 * WSIZE + n], Gu[idx_n1 * WSIZE + n], Gu[idx * WSIZE + n], Gu[idx_p1 * WSIZE + n], Gu[idx_p2 * WSIZE + n], Gu[idx_p3 * WSIZE + n]);
#endif // LF
    }
    END_CALCULATE3D()

    // * Z direction
    CALCULATE3D(i, j, k, HALO, _nx, HALO, _ny, HALO - 1, _nz)

#ifdef PML
    pml_beta_x = pml_beta.x[i];
    pml_beta_y = pml_beta.y[j];
    pml_beta_z = pml_beta.z[k];
#endif

    idx_n2 = INDEX(i, j, k - 2);
    idx_n1 = INDEX(i, j, k - 1);
    idx = INDEX(i, j, k);
    idx_p1 = INDEX(i, j, k + 1);
    idx_p2 = INDEX(i, j, k + 2);
    idx_p3 = INDEX(i, j, k + 3);

    zt_x_J_h = CJM[idx * CJMSIZE + 6];
    zt_y_J_h = CJM[idx * CJMSIZE + 7];
    zt_z_J_h = CJM[idx * CJMSIZE + 8];

    mu = CJM[idx * CJMSIZE + 10];
    lambda = CJM[idx * CJMSIZE + 11];
    buoyancy = CJM[idx * CJMSIZE + 12];
    buoyancy *= Crho;

    // Shock capture interpolation
    for (int n = 0; n < 9; n++)
    {
        u_ip12n[n] = WENO5_interpolation(W[idx_n2 * WSIZE + n], W[idx_n1 * WSIZE + n], W[idx * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx_p2 * WSIZE + n]);
        u_ip12p[n] = WENO5_interpolation(W[idx_p3 * WSIZE + n], W[idx_p2 * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx * WSIZE + n], W[idx_n1 * WSIZE + n]);
    }

#ifdef LF
    // Riemann solver: Lax-Friedrichs
    fu_ip12n[0] = -(zt_x_J_h * u_ip12n[6]) * buoyancy;
    fu_ip12n[1] = -(zt_y_J_h * u_ip12n[7]) * buoyancy;
    fu_ip12n[2] = -(zt_z_J_h * u_ip12n[8]) * buoyancy;
    fu_ip12n[3] = -(zt_x_J_h * u_ip12n[7]) * 0.5 * buoyancy - (zt_y_J_h * u_ip12n[6]) * 0.5 * buoyancy;
    fu_ip12n[4] = -(zt_y_J_h * u_ip12n[8]) * 0.5 * buoyancy - (zt_z_J_h * u_ip12n[7]) * 0.5 * buoyancy;
    fu_ip12n[5] = -(zt_x_J_h * u_ip12n[8]) * 0.5 * buoyancy - (zt_z_J_h * u_ip12n[6]) * 0.5 * buoyancy;
    fu_ip12n[6] = -zt_x_J_h * (2 * mu * u_ip12n[0] + lambda * (u_ip12n[0] + u_ip12n[1] + u_ip12n[2])) - 2 * mu * zt_y_J_h * u_ip12n[3] - 2 * mu * zt_z_J_h * u_ip12n[5];
    fu_ip12n[7] = -zt_y_J_h * (2 * mu * u_ip12n[1] + lambda * (u_ip12n[0] + u_ip12n[1] + u_ip12n[2])) - 2 * mu * zt_x_J_h * u_ip12n[3] - 2 * mu * zt_z_J_h * u_ip12n[4];
    fu_ip12n[8] = -zt_z_J_h * (2 * mu * u_ip12n[2] + lambda * (u_ip12n[0] + u_ip12n[1] + u_ip12n[2])) - 2 * mu * zt_x_J_h * u_ip12n[5] - 2 * mu * zt_y_J_h * u_ip12n[4];

    fu_ip12p[0] = -(zt_x_J_h * u_ip12p[6]) * buoyancy;
    fu_ip12p[1] = -(zt_y_J_h * u_ip12p[7]) * buoyancy;
    fu_ip12p[2] = -(zt_z_J_h * u_ip12p[8]) * buoyancy;
    fu_ip12p[3] = -(zt_x_J_h * u_ip12p[7]) * 0.5 * buoyancy - (zt_y_J_h * u_ip12p[6]) * 0.5 * buoyancy;
    fu_ip12p[4] = -(zt_y_J_h * u_ip12p[8]) * 0.5 * buoyancy - (zt_z_J_h * u_ip12p[7]) * 0.5 * buoyancy;
    fu_ip12p[5] = -(zt_x_J_h * u_ip12p[8]) * 0.5 * buoyancy - (zt_z_J_h * u_ip12p[6]) * 0.5 * buoyancy;
    fu_ip12p[6] = -zt_x_J_h * (2 * mu * u_ip12p[0] + lambda * (u_ip12p[0] + u_ip12p[1] + u_ip12p[2])) - 2 * mu * zt_y_J_h * u_ip12p[3] - 2 * mu * zt_z_J_h * u_ip12p[5];
    fu_ip12p[7] = -zt_y_J_h * (2 * mu * u_ip12p[1] + lambda * (u_ip12p[0] + u_ip12p[1] + u_ip12p[2])) - 2 * mu * zt_x_J_h * u_ip12p[3] - 2 * mu * zt_z_J_h * u_ip12p[4];
    fu_ip12p[8] = -zt_z_J_h * (2 * mu * u_ip12p[2] + lambda * (u_ip12p[0] + u_ip12p[1] + u_ip12p[2])) - 2 * mu * zt_x_J_h * u_ip12p[5] - 2 * mu * zt_y_J_h * u_ip12p[4];
#endif

    for (int n = 0; n < 9; n++)
    {
#ifdef LF
        Fu_ip12z[idx * WSIZE + n] = 0.5f * (fu_ip12p[n] + fu_ip12n[n] - alpha * (u_ip12p[n] - u_ip12n[n])) - 1.0f / 24 * order2_approximation(Hu[idx_n2 * WSIZE + n], Hu[idx_n1 * WSIZE + n], Hu[idx * WSIZE + n], Hu[idx_p1 * WSIZE + n], Hu[idx_p2 * WSIZE + n], Hu[idx_p3 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Hu[idx_n2 * WSIZE + n], Hu[idx_n1 * WSIZE + n], Hu[idx * WSIZE + n], Hu[idx_p1 * WSIZE + n], Hu[idx_p2 * WSIZE + n], Hu[idx_p3 * WSIZE + n]);
#endif // LF
    }
    END_CALCULATE3D()
}

__GLOBAL__
void cal_du(FLOAT *Fu_ip12x, FLOAT *Fu_ip12y, FLOAT *Fu_ip12z, FLOAT *h_W, FLOAT *CJM,
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
    int k = threadIdx.z + blockIdx.z * blockDim.z;
#else
    int i = 0;
    int j = 0;
    int k = 0;
#endif

#ifdef PML
    float pml_beta_x;
    float pml_beta_y;
    float pml_beta_z;
#endif

    long long idx, idx_n1, idy_n1, idz_n1;
    float jac_inv;

    CALCULATE3D(i, j, k, HALO, _nx, HALO, _ny, HALO, _nz)

    idx = INDEX(i, j, k);
    idx_n1 = INDEX(i - 1, j, k);
    idy_n1 = INDEX(i, j - 1, k);
    idz_n1 = INDEX(i, j, k - 1);

    jac_inv = CJM[idx * CJMSIZE + 9];

    for (int n = 0; n < 9; n++)
    {
        h_W[idx * WSIZE + n] = DT * (-(Fu_ip12x[idx * WSIZE + n] - Fu_ip12x[idx_n1 * WSIZE + n]) * rDH - (Fu_ip12y[idx * WSIZE + n] - Fu_ip12y[idy_n1 * WSIZE + n]) * rDH - (Fu_ip12z[idx * WSIZE + n] - Fu_ip12z[idz_n1 * WSIZE + n]) * rDH) / jac_inv;
    }
    END_CALCULATE3D()
}

void waveDeriv_alternative_flux_FD(GRID grid, WAVE wave, FLOAT *CJM,
#ifdef PML
                                   PML_BETA pml_beta,
#endif
                                   int FB1, int FB2, int FB3, float DT, MPI_COORD thisMPICoord, PARAMS params)
{
    int _nx_ = grid._nx_;
    int _ny_ = grid._ny_;
    int _nz_ = grid._nz_;

    float rDH = grid.rDH;

#ifdef GPU_CUDA
    int nx = _nx_ - 2 * HALO;
    int ny = _ny_ - 2 * HALO;
    int nz = _nz_ - 2 * HALO;

#ifdef XFAST
    dim3 threads(32, 4, 4);
#endif
#ifdef ZFAST
    dim3 threads(1, 8, 64);
#endif
    dim3 blocks;
    blocks.x = (_nx_ + threads.x - 1) / threads.x;
    blocks.y = (_ny_ + threads.y - 1) / threads.y;
    blocks.z = (_nz_ + threads.z - 1) / threads.z;

    // cout << "X = " << blocks.x << "Y = " << blocks.y << "Z = " << blocks.z << endl;
    cal_flux<<<blocks, threads>>>(wave.Fu, wave.Gu, wave.Hu, wave.W, CJM, _nx_, _ny_, _nz_, thisMPICoord, params);
    wave_deriv_alternative_flux_FD<<<blocks, threads>>>(wave.fu_ip12x, wave.fu_ip12y, wave.fu_ip12z,
                                                        wave.Fu, wave.Gu, wave.Hu, wave.h_W, wave.W, CJM,
#ifdef PML
                                                        pml_beta,
#endif // PML
                                                        _nx_, _ny_, _nz_, rDH, FB1, FB2, FB3, DT
#ifdef LF
                                                        ,
                                                        vp_max_for_SCFDM
#endif // LF
    );

    CHECK(cudaDeviceSynchronize());
    cal_du<<<blocks, threads>>>(wave.fu_ip12x, wave.fu_ip12y, wave.fu_ip12z, wave.h_W, CJM,
#ifdef PML
                                pml_beta,
#endif // PML
                                _nx_, _ny_, _nz_, rDH, FB1, FB2, FB3, DT);
    CHECK(cudaDeviceSynchronize());

#else // GPU_CUDA
    cal_flux(wave.Fu, wave.Gu, wave.Hu, wave.W, CJM, _nx_, _ny_, _nz_);
    wave_deriv_alternative_flux_FD(wave.h_W, wave.W, CJM,
#ifdef PML
                                   pml_beta,
#endif // PML
                                   _nx_, _ny_, _nz_, rDH, FB1, FB2, FB3, DT
#ifdef LF
                                   ,
                                   vp_max_for_SCFDM
#endif // LF
    );
    cal_du(wave.fu_ip12x, wave.fu_ip12y, wave.fu_ip12z, wave.h_W, CJM,
#ifdef PML
           pml_beta,
#endif // PML
           _nx_, _ny_, _nz_, rDH, FB1, FB2, FB3, DT);

#endif // GPU_CUDA
}

#endif // SCFDM