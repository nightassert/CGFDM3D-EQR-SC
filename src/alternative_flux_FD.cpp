/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: alternative_flux_FD.cpp
*   Author: Tianhong Xu, 12231218@mail.sustech.edu.cn
*   Created Time: 2023-10-22
*   Discription: Shock Capturing Finite Difference Method kernel file
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
    // return d1 * v1 + d2 * v2 + d3 * v3;
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
    mu = CJM[index * CJMSIZE + 10];
    lambda = CJM[index * CJMSIZE + 11];
    buoyancy = CJM[index * CJMSIZE + 12];

    // calculate flux fu
    xi_x_J = CJM[index * CJMSIZE + 0];
    xi_y_J = CJM[index * CJMSIZE + 1];
    xi_z_J = CJM[index * CJMSIZE + 2];

    fu[index * WSIZE + 0] = -(xi_x_J * (lambda * u[index * WSIZE + 4] + lambda * u[index * WSIZE + 5] + u[index * WSIZE + 3] * (lambda + 2 * mu)) + mu * xi_y_J * u[index * WSIZE + 8] + mu * xi_z_J * u[index * WSIZE + 7]);
    fu[index * WSIZE + 1] = -(xi_y_J * (lambda * u[index * WSIZE + 3] + lambda * u[index * WSIZE + 5] + u[index * WSIZE + 4] * (lambda + 2 * mu)) + mu * xi_x_J * u[index * WSIZE + 8] + mu * xi_z_J * u[index * WSIZE + 6]);
    fu[index * WSIZE + 2] = -(xi_z_J * (lambda * u[index * WSIZE + 3] + lambda * u[index * WSIZE + 4] + u[index * WSIZE + 5] * (lambda + 2 * mu)) + mu * xi_x_J * u[index * WSIZE + 7] + mu * xi_y_J * u[index * WSIZE + 6]);
    fu[index * WSIZE + 3] = -((xi_x_J * u[index * WSIZE + 0]) * buoyancy);
    fu[index * WSIZE + 4] = -((xi_y_J * u[index * WSIZE + 1]) * buoyancy);
    fu[index * WSIZE + 5] = -((xi_z_J * u[index * WSIZE + 2]) * buoyancy);
    fu[index * WSIZE + 6] = -((xi_y_J * u[index * WSIZE + 2]) * buoyancy + (xi_z_J * u[index * WSIZE + 1]) * buoyancy);
    fu[index * WSIZE + 7] = -((xi_x_J * u[index * WSIZE + 2]) * buoyancy + (xi_z_J * u[index * WSIZE + 0]) * buoyancy);
    fu[index * WSIZE + 8] = -((xi_x_J * u[index * WSIZE + 1]) * buoyancy + (xi_y_J * u[index * WSIZE + 0]) * buoyancy);

    // calculate flux gu
    et_x_J = CJM[index * CJMSIZE + 3];
    et_y_J = CJM[index * CJMSIZE + 4];
    et_z_J = CJM[index * CJMSIZE + 5];

    gu[index * WSIZE + 0] = -(et_x_J * (lambda * u[index * WSIZE + 4] + lambda * u[index * WSIZE + 5] + u[index * WSIZE + 3] * (lambda + 2 * mu)) + mu * et_y_J * u[index * WSIZE + 8] + mu * et_z_J * u[index * WSIZE + 7]);
    gu[index * WSIZE + 1] = -(et_y_J * (lambda * u[index * WSIZE + 3] + lambda * u[index * WSIZE + 5] + u[index * WSIZE + 4] * (lambda + 2 * mu)) + mu * et_x_J * u[index * WSIZE + 8] + mu * et_z_J * u[index * WSIZE + 6]);
    gu[index * WSIZE + 2] = -(et_z_J * (lambda * u[index * WSIZE + 3] + lambda * u[index * WSIZE + 4] + u[index * WSIZE + 5] * (lambda + 2 * mu)) + mu * et_x_J * u[index * WSIZE + 7] + mu * et_y_J * u[index * WSIZE + 6]);
    gu[index * WSIZE + 3] = -((et_x_J * u[index * WSIZE + 0]) * buoyancy);
    gu[index * WSIZE + 4] = -((et_y_J * u[index * WSIZE + 1]) * buoyancy);
    gu[index * WSIZE + 5] = -((et_z_J * u[index * WSIZE + 2]) * buoyancy);
    gu[index * WSIZE + 6] = -((et_y_J * u[index * WSIZE + 2]) * buoyancy + (et_z_J * u[index * WSIZE + 1]) * buoyancy);
    gu[index * WSIZE + 7] = -((et_x_J * u[index * WSIZE + 2]) * buoyancy + (et_z_J * u[index * WSIZE + 0]) * buoyancy);
    gu[index * WSIZE + 8] = -((et_x_J * u[index * WSIZE + 1]) * buoyancy + (et_y_J * u[index * WSIZE + 0]) * buoyancy);

    // calculate flux hu
    zt_x_J = CJM[index * CJMSIZE + 6];
    zt_y_J = CJM[index * CJMSIZE + 7];
    zt_z_J = CJM[index * CJMSIZE + 8];

    hu[index * WSIZE + 0] = -(zt_x_J * (lambda * u[index * WSIZE + 4] + lambda * u[index * WSIZE + 5] + u[index * WSIZE + 3] * (lambda + 2 * mu)) + mu * zt_y_J * u[index * WSIZE + 8] + mu * zt_z_J * u[index * WSIZE + 7]);
    hu[index * WSIZE + 1] = -(zt_y_J * (lambda * u[index * WSIZE + 3] + lambda * u[index * WSIZE + 5] + u[index * WSIZE + 4] * (lambda + 2 * mu)) + mu * zt_x_J * u[index * WSIZE + 8] + mu * zt_z_J * u[index * WSIZE + 6]);
    hu[index * WSIZE + 2] = -(zt_z_J * (lambda * u[index * WSIZE + 3] + lambda * u[index * WSIZE + 4] + u[index * WSIZE + 5] * (lambda + 2 * mu)) + mu * zt_x_J * u[index * WSIZE + 7] + mu * zt_y_J * u[index * WSIZE + 6]);
    hu[index * WSIZE + 3] = -((zt_x_J * u[index * WSIZE + 0]) * buoyancy);
    hu[index * WSIZE + 4] = -((zt_y_J * u[index * WSIZE + 1]) * buoyancy);
    hu[index * WSIZE + 5] = -((zt_z_J * u[index * WSIZE + 2]) * buoyancy);
    hu[index * WSIZE + 6] = -((zt_y_J * u[index * WSIZE + 2]) * buoyancy + (zt_z_J * u[index * WSIZE + 1]) * buoyancy);
    hu[index * WSIZE + 7] = -((zt_x_J * u[index * WSIZE + 2]) * buoyancy + (zt_z_J * u[index * WSIZE + 0]) * buoyancy);
    hu[index * WSIZE + 8] = -((zt_x_J * u[index * WSIZE + 1]) * buoyancy + (zt_y_J * u[index * WSIZE + 0]) * buoyancy);
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
    float Riemann_flux[9];

#ifdef LF
    float fu_ip12n[9], fu_ip12p[9];
#endif

#ifdef UW
    float u_ip12n_T[9], u_ip12p_T[9];
    float fu_ip12n[9], fu_ip12p[9];
    float Riemann_flux_T[9];
    float zs_n, zs_p, zp_n, zp_p;
    float lambda_n, mu_n, rho_n, lambda_p, mu_p, rho_p;
    float tau_xx, tau_xy, tau_xz, v_x, v_y, v_z;
    float nx, ny, nz, sx, sy, sz, tx, ty, tz;
#endif

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

    // Shock capture interpolation
    for (int n = 0; n < 9; n++)
    {
        u_ip12n[n] = WENO5_interpolation(W[idx_n2 * WSIZE + n], W[idx_n1 * WSIZE + n], W[idx * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx_p2 * WSIZE + n]);
        u_ip12p[n] = WENO5_interpolation(W[idx_p3 * WSIZE + n], W[idx_p2 * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx * WSIZE + n], W[idx_n1 * WSIZE + n]);
    }

#ifdef LF
    // Riemann solver: Lax-Friedrichs
    fu_ip12n[0] = -(xi_x_J_h * (lambda * u_ip12n[4] + lambda * u_ip12n[5] + u_ip12n[3] * (lambda + 2 * mu)) + mu * xi_y_J_h * u_ip12n[8] + mu * xi_z_J_h * u_ip12n[7]);
    fu_ip12n[1] = -(xi_y_J_h * (lambda * u_ip12n[3] + lambda * u_ip12n[5] + u_ip12n[4] * (lambda + 2 * mu)) + mu * xi_x_J_h * u_ip12n[8] + mu * xi_z_J_h * u_ip12n[6]);
    fu_ip12n[2] = -(xi_z_J_h * (lambda * u_ip12n[3] + lambda * u_ip12n[4] + u_ip12n[5] * (lambda + 2 * mu)) + mu * xi_x_J_h * u_ip12n[7] + mu * xi_y_J_h * u_ip12n[6]);
    fu_ip12n[3] = -((xi_x_J_h * u_ip12n[0]) * buoyancy);
    fu_ip12n[4] = -((xi_y_J_h * u_ip12n[1]) * buoyancy);
    fu_ip12n[5] = -((xi_z_J_h * u_ip12n[2]) * buoyancy);
    fu_ip12n[6] = -((xi_y_J_h * u_ip12n[2]) * buoyancy + (xi_z_J_h * u_ip12n[1]) * buoyancy);
    fu_ip12n[7] = -((xi_x_J_h * u_ip12n[2]) * buoyancy + (xi_z_J_h * u_ip12n[0]) * buoyancy);
    fu_ip12n[8] = -((xi_x_J_h * u_ip12n[1]) * buoyancy + (xi_y_J_h * u_ip12n[0]) * buoyancy);

    fu_ip12p[0] = -(xi_x_J_h * (lambda * u_ip12p[4] + lambda * u_ip12p[5] + u_ip12p[3] * (lambda + 2 * mu)) + mu * xi_y_J_h * u_ip12p[8] + mu * xi_z_J_h * u_ip12p[7]);
    fu_ip12p[1] = -(xi_y_J_h * (lambda * u_ip12p[3] + lambda * u_ip12p[5] + u_ip12p[4] * (lambda + 2 * mu)) + mu * xi_x_J_h * u_ip12p[8] + mu * xi_z_J_h * u_ip12p[6]);
    fu_ip12p[2] = -(xi_z_J_h * (lambda * u_ip12p[3] + lambda * u_ip12p[4] + u_ip12p[5] * (lambda + 2 * mu)) + mu * xi_x_J_h * u_ip12p[7] + mu * xi_y_J_h * u_ip12p[6]);
    fu_ip12p[3] = -((xi_x_J_h * u_ip12p[0]) * buoyancy);
    fu_ip12p[4] = -((xi_y_J_h * u_ip12p[1]) * buoyancy);
    fu_ip12p[5] = -((xi_z_J_h * u_ip12p[2]) * buoyancy);
    fu_ip12p[6] = -((xi_y_J_h * u_ip12p[2]) * buoyancy + (xi_z_J_h * u_ip12p[1]) * buoyancy);
    fu_ip12p[7] = -((xi_x_J_h * u_ip12p[2]) * buoyancy + (xi_z_J_h * u_ip12p[0]) * buoyancy);
    fu_ip12p[8] = -((xi_x_J_h * u_ip12p[1]) * buoyancy + (xi_y_J_h * u_ip12p[0]) * buoyancy);

    Riemann_flux[0] = 0.5f * (fu_ip12p[0] + fu_ip12n[0] - alpha * (u_ip12p[0] - u_ip12n[0]));
    Riemann_flux[1] = 0.5f * (fu_ip12p[1] + fu_ip12n[1] - alpha * (u_ip12p[1] - u_ip12n[1]));
    Riemann_flux[2] = 0.5f * (fu_ip12p[2] + fu_ip12n[2] - alpha * (u_ip12p[2] - u_ip12n[2]));
    Riemann_flux[3] = 0.5f * (fu_ip12p[3] + fu_ip12n[3] - alpha * (u_ip12p[3] - u_ip12n[3]));
    Riemann_flux[4] = 0.5f * (fu_ip12p[4] + fu_ip12n[4] - alpha * (u_ip12p[4] - u_ip12n[4]));
    Riemann_flux[5] = 0.5f * (fu_ip12p[5] + fu_ip12n[5] - alpha * (u_ip12p[5] - u_ip12n[5]));
    Riemann_flux[6] = 0.5f * (fu_ip12p[6] + fu_ip12n[6] - alpha * (u_ip12p[6] - u_ip12n[6]));
    Riemann_flux[7] = 0.5f * (fu_ip12p[7] + fu_ip12n[7] - alpha * (u_ip12p[7] - u_ip12n[7]));
    Riemann_flux[8] = 0.5f * (fu_ip12p[8] + fu_ip12n[8] - alpha * (u_ip12p[8] - u_ip12n[8]));
#endif

#ifdef UW // ! Still not correct
    mu_n = CJM[idx * CJMSIZE + 10];
    lambda_n = CJM[idx * CJMSIZE + 11];
    rho_n = 1.0 / CJM[idx * CJMSIZE + 12];

    mu_p = CJM[idx_p1 * CJMSIZE + 10];
    lambda_p = CJM[idx_p1 * CJMSIZE + 11];
    rho_p = 1.0 / CJM[idx_p1 * CJMSIZE + 12];

    zs_n = sqrt(CJM[idx * CJMSIZE + 10] / CJM[idx * CJMSIZE + 12]);
    zs_p = sqrt(CJM[idx_p1 * CJMSIZE + 10] / CJM[idx_p1 * CJMSIZE + 12]);
    zp_n = sqrt((CJM[idx * CJMSIZE + 11] + 2 * CJM[idx * CJMSIZE + 10]) / CJM[idx * CJMSIZE + 12]);
    zp_p = sqrt((CJM[idx_p1 * CJMSIZE + 11] + 2 * CJM[idx_p1 * CJMSIZE + 10]) / CJM[idx_p1 * CJMSIZE + 12]);

    // Calculate T
    nx = (CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) / (sqrt((CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) + (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) + (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]))); // nx;
    ny = (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) / (sqrt((CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) + (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) + (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]))); // ny;
    nz = (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]) / (sqrt((CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) + (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) + (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]))); // nz;
    sx = CJM[idx * CJMSIZE + 16] / (sqrt(CJM[idx * CJMSIZE + 16] * CJM[idx * CJMSIZE + 16] + CJM[idx * CJMSIZE + 17] * CJM[idx * CJMSIZE + 17] + CJM[idx * CJMSIZE + 18] * CJM[idx * CJMSIZE + 18]));                                                                                                                                                                                       // sx;
    sy = CJM[idx * CJMSIZE + 17] / (sqrt(CJM[idx * CJMSIZE + 16] * CJM[idx * CJMSIZE + 16] + CJM[idx * CJMSIZE + 17] * CJM[idx * CJMSIZE + 17] + CJM[idx * CJMSIZE + 18] * CJM[idx * CJMSIZE + 18]));                                                                                                                                                                                       // sy;
    sz = CJM[idx * CJMSIZE + 18] / (sqrt(CJM[idx * CJMSIZE + 16] * CJM[idx * CJMSIZE + 16] + CJM[idx * CJMSIZE + 17] * CJM[idx * CJMSIZE + 17] + CJM[idx * CJMSIZE + 18] * CJM[idx * CJMSIZE + 18]));                                                                                                                                                                                       // sz;
    tx = CJM[idx * CJMSIZE + 19] / (sqrt(CJM[idx * CJMSIZE + 19] * CJM[idx * CJMSIZE + 19] + CJM[idx * CJMSIZE + 20] * CJM[idx * CJMSIZE + 20] + CJM[idx * CJMSIZE + 21] * CJM[idx * CJMSIZE + 21]));                                                                                                                                                                                       // tx;
    ty = CJM[idx * CJMSIZE + 20] / (sqrt(CJM[idx * CJMSIZE + 19] * CJM[idx * CJMSIZE + 19] + CJM[idx * CJMSIZE + 20] * CJM[idx * CJMSIZE + 20] + CJM[idx * CJMSIZE + 21] * CJM[idx * CJMSIZE + 21]));                                                                                                                                                                                       // ty;
    tz = CJM[idx * CJMSIZE + 21] / (sqrt(CJM[idx * CJMSIZE + 19] * CJM[idx * CJMSIZE + 19] + CJM[idx * CJMSIZE + 20] * CJM[idx * CJMSIZE + 20] + CJM[idx * CJMSIZE + 21] * CJM[idx * CJMSIZE + 21]));                                                                                                                                                                                       // tz;

    // Rotate u
    u_ip12n_T[0] = nx * u_ip12n[0] + ny * u_ip12n[1] + nz * u_ip12n[2];
    u_ip12n_T[1] = sx * u_ip12n[0] + sy * u_ip12n[1] + sz * u_ip12n[2];
    u_ip12n_T[2] = tx * u_ip12n[0] + ty * u_ip12n[1] + tz * u_ip12n[2];
    u_ip12n_T[3] = u_ip12n[3] * nx * nx + u_ip12n[8] * nx * ny + u_ip12n[7] * nx * nz + u_ip12n[4] * ny * ny + u_ip12n[6] * ny * nz + u_ip12n[5] * nz * nz;
    u_ip12n_T[4] = u_ip12n[3] * sx * sx + u_ip12n[8] * sx * sy + u_ip12n[7] * sx * sz + u_ip12n[4] * sy * sy + u_ip12n[6] * sy * sz + u_ip12n[5] * sz * sz;
    u_ip12n_T[5] = u_ip12n[3] * tx * tx + u_ip12n[8] * tx * ty + u_ip12n[7] * tx * tz + u_ip12n[4] * ty * ty + u_ip12n[6] * ty * tz + u_ip12n[5] * tz * tz;
    u_ip12n_T[6] = u_ip12n[8] * (sx * ty + sy * tx) + u_ip12n[7] * (sx * tz + sz * tx) + u_ip12n[6] * (sy * tz + sz * ty) + 2 * sx * tx * u_ip12n[3] + 2 * sy * ty * u_ip12n[4] + 2 * sz * tz * u_ip12n[5];
    u_ip12n_T[7] = u_ip12n[8] * (nx * ty + ny * tx) + u_ip12n[7] * (nx * tz + nz * tx) + u_ip12n[6] * (ny * tz + nz * ty) + 2 * nx * tx * u_ip12n[3] + 2 * ny * ty * u_ip12n[4] + 2 * nz * tz * u_ip12n[5];
    u_ip12n_T[8] = u_ip12n[8] * (nx * sy + ny * sx) + u_ip12n[7] * (nx * sz + nz * sx) + u_ip12n[6] * (ny * sz + nz * sy) + 2 * nx * sx * u_ip12n[3] + 2 * ny * sy * u_ip12n[4] + 2 * nz * sz * u_ip12n[5];
    // u_ip12n_T[0] = (ny * sz * u_ip12n[2] - nz * sy * u_ip12n[2] - ny * tz * u_ip12n[1] + nz * ty * u_ip12n[1] + sy * tz * u_ip12n[0] - sz * ty * u_ip12n[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    // u_ip12n_T[1] = -(nx * sz * u_ip12n[2] - nz * sx * u_ip12n[2] - nx * tz * u_ip12n[1] + nz * tx * u_ip12n[1] + sx * tz * u_ip12n[0] - sz * tx * u_ip12n[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    // u_ip12n_T[2] = (nx * sy * u_ip12n[2] - ny * sx * u_ip12n[2] - nx * ty * u_ip12n[1] + ny * tx * u_ip12n[1] + sx * ty * u_ip12n[0] - sy * tx * u_ip12n[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    // u_ip12n_T[3] = (u_ip12n[5] * (ny * ny) * (sz * sz) - u_ip12n[6] * (ny * ny) * sz * tz + u_ip12n[4] * (ny * ny) * (tz * tz) - 2 * u_ip12n[5] * ny * nz * sy * sz + u_ip12n[6] * ny * nz * sy * tz + u_ip12n[6] * ny * nz * sz * ty - 2 * u_ip12n[4] * ny * nz * ty * tz + u_ip12n[7] * ny * sy * sz * tz - u_ip12n[8] * ny * sy * (tz * tz) - u_ip12n[7] * ny * (sz * sz) * ty + u_ip12n[8] * ny * sz * ty * tz + u_ip12n[5] * (nz * nz) * (sy * sy) - u_ip12n[6] * (nz * nz) * sy * ty + u_ip12n[4] * (nz * nz) * (ty * ty) - u_ip12n[7] * nz * (sy * sy) * tz + u_ip12n[7] * nz * sy * sz * ty + u_ip12n[8] * nz * sy * ty * tz - u_ip12n[8] * nz * sz * (ty * ty) + u_ip12n[3] * (sy * sy) * (tz * tz) - 2 * u_ip12n[3] * sy * sz * ty * tz + u_ip12n[3] * (sz * sz) * (ty * ty)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12n_T[4] = (u_ip12n[5] * (nx * nx) * (sz * sz) - u_ip12n[6] * (nx * nx) * sz * tz + u_ip12n[4] * (nx * nx) * (tz * tz) - 2 * u_ip12n[5] * nx * nz * sx * sz + u_ip12n[6] * nx * nz * sx * tz + u_ip12n[6] * nx * nz * sz * tx - 2 * u_ip12n[4] * nx * nz * tx * tz + u_ip12n[7] * nx * sx * sz * tz - u_ip12n[8] * nx * sx * (tz * tz) - u_ip12n[7] * nx * (sz * sz) * tx + u_ip12n[8] * nx * sz * tx * tz + u_ip12n[5] * (nz * nz) * (sx * sx) - u_ip12n[6] * (nz * nz) * sx * tx + u_ip12n[4] * (nz * nz) * (tx * tx) - u_ip12n[7] * nz * (sx * sx) * tz + u_ip12n[7] * nz * sx * sz * tx + u_ip12n[8] * nz * sx * tx * tz - u_ip12n[8] * nz * sz * (tx * tx) + u_ip12n[3] * (sx * sx) * (tz * tz) - 2 * u_ip12n[3] * sx * sz * tx * tz + u_ip12n[3] * (sz * sz) * (tx * tx)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12n_T[5] = (u_ip12n[5] * (nx * nx) * (sy * sy) - u_ip12n[6] * (nx * nx) * sy * ty + u_ip12n[4] * (nx * nx) * (ty * ty) - 2 * u_ip12n[5] * nx * ny * sx * sy + u_ip12n[6] * nx * ny * sx * ty + u_ip12n[6] * nx * ny * sy * tx - 2 * u_ip12n[4] * nx * ny * tx * ty + u_ip12n[7] * nx * sx * sy * ty - u_ip12n[8] * nx * sx * (ty * ty) - u_ip12n[7] * nx * (sy * sy) * tx + u_ip12n[8] * nx * sy * tx * ty + u_ip12n[5] * (ny * ny) * (sx * sx) - u_ip12n[6] * (ny * ny) * sx * tx + u_ip12n[4] * (ny * ny) * (tx * tx) - u_ip12n[7] * ny * (sx * sx) * ty + u_ip12n[7] * ny * sx * sy * tx + u_ip12n[8] * ny * sx * tx * ty - u_ip12n[8] * ny * sy * (tx * tx) + u_ip12n[3] * (sx * sx) * (ty * ty) - 2 * u_ip12n[3] * sx * sy * tx * ty + u_ip12n[3] * (sy * sy) * (tx * tx)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12n_T[6] = -(2 * ny * nz * (sx * sx) * u_ip12n[5] + 2 * ny * nz * (tx * tx) * u_ip12n[4] + 2 * (nx * nx) * sy * sz * u_ip12n[5] - (nx * nx) * sy * tz * u_ip12n[6] - (nx * nx) * sz * ty * u_ip12n[6] - ny * (sx * sx) * tz * u_ip12n[7] - nz * (sx * sx) * ty * u_ip12n[7] - ny * sz * (tx * tx) * u_ip12n[8] - nz * sy * (tx * tx) * u_ip12n[8] + 2 * (nx * nx) * ty * tz * u_ip12n[4] + 2 * sy * sz * (tx * tx) * u_ip12n[3] + 2 * (sx * sx) * ty * tz * u_ip12n[3] - 2 * nx * ny * sx * sz * u_ip12n[5] - 2 * nx * nz * sx * sy * u_ip12n[5] + nx * ny * sx * tz * u_ip12n[6] + nx * ny * sz * tx * u_ip12n[6] + nx * nz * sx * ty * u_ip12n[6] + nx * nz * sy * tx * u_ip12n[6] - 2 * ny * nz * sx * tx * u_ip12n[6] - 2 * nx * ny * tx * tz * u_ip12n[4] - 2 * nx * nz * tx * ty * u_ip12n[4] + nx * sx * sy * tz * u_ip12n[7] + nx * sx * sz * ty * u_ip12n[7] - 2 * nx * sy * sz * tx * u_ip12n[7] + ny * sx * sz * tx * u_ip12n[7] + nz * sx * sy * tx * u_ip12n[7] - 2 * nx * sx * ty * tz * u_ip12n[8] + nx * sy * tx * tz * u_ip12n[8] + nx * sz * tx * ty * u_ip12n[8] + ny * sx * tx * tz * u_ip12n[8] + nz * sx * tx * ty * u_ip12n[8] - 2 * sx * sy * tx * tz * u_ip12n[3] - 2 * sx * sz * tx * ty * u_ip12n[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12n_T[7] = -(2 * nx * nz * (sy * sy) * u_ip12n[5] + 2 * nx * nz * (ty * ty) * u_ip12n[4] + 2 * (ny * ny) * sx * sz * u_ip12n[5] - (ny * ny) * sx * tz * u_ip12n[6] - (ny * ny) * sz * tx * u_ip12n[6] - nx * (sy * sy) * tz * u_ip12n[7] - nz * (sy * sy) * tx * u_ip12n[7] - nx * sz * (ty * ty) * u_ip12n[8] - nz * sx * (ty * ty) * u_ip12n[8] + 2 * (ny * ny) * tx * tz * u_ip12n[4] + 2 * sx * sz * (ty * ty) * u_ip12n[3] + 2 * (sy * sy) * tx * tz * u_ip12n[3] - 2 * nx * ny * sy * sz * u_ip12n[5] - 2 * ny * nz * sx * sy * u_ip12n[5] + nx * ny * sy * tz * u_ip12n[6] + nx * ny * sz * ty * u_ip12n[6] - 2 * nx * nz * sy * ty * u_ip12n[6] + ny * nz * sx * ty * u_ip12n[6] + ny * nz * sy * tx * u_ip12n[6] - 2 * nx * ny * ty * tz * u_ip12n[4] - 2 * ny * nz * tx * ty * u_ip12n[4] + nx * sy * sz * ty * u_ip12n[7] + ny * sx * sy * tz * u_ip12n[7] - 2 * ny * sx * sz * ty * u_ip12n[7] + ny * sy * sz * tx * u_ip12n[7] + nz * sx * sy * ty * u_ip12n[7] + nx * sy * ty * tz * u_ip12n[8] + ny * sx * ty * tz * u_ip12n[8] - 2 * ny * sy * tx * tz * u_ip12n[8] + ny * sz * tx * ty * u_ip12n[8] + nz * sy * tx * ty * u_ip12n[8] - 2 * sx * sy * ty * tz * u_ip12n[3] - 2 * sy * sz * tx * ty * u_ip12n[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12n_T[8] = -(2 * nx * ny * (sz * sz) * u_ip12n[5] + 2 * nx * ny * (tz * tz) * u_ip12n[4] + 2 * (nz * nz) * sx * sy * u_ip12n[5] - (nz * nz) * sx * ty * u_ip12n[6] - (nz * nz) * sy * tx * u_ip12n[6] - nx * (sz * sz) * ty * u_ip12n[7] - ny * (sz * sz) * tx * u_ip12n[7] - nx * sy * (tz * tz) * u_ip12n[8] - ny * sx * (tz * tz) * u_ip12n[8] + 2 * (nz * nz) * tx * ty * u_ip12n[4] + 2 * sx * sy * (tz * tz) * u_ip12n[3] + 2 * (sz * sz) * tx * ty * u_ip12n[3] - 2 * nx * nz * sy * sz * u_ip12n[5] - 2 * ny * nz * sx * sz * u_ip12n[5] - 2 * nx * ny * sz * tz * u_ip12n[6] + nx * nz * sy * tz * u_ip12n[6] + nx * nz * sz * ty * u_ip12n[6] + ny * nz * sx * tz * u_ip12n[6] + ny * nz * sz * tx * u_ip12n[6] - 2 * nx * nz * ty * tz * u_ip12n[4] - 2 * ny * nz * tx * tz * u_ip12n[4] + nx * sy * sz * tz * u_ip12n[7] + ny * sx * sz * tz * u_ip12n[7] - 2 * nz * sx * sy * tz * u_ip12n[7] + nz * sx * sz * ty * u_ip12n[7] + nz * sy * sz * tx * u_ip12n[7] + nx * sz * ty * tz * u_ip12n[8] + ny * sz * tx * tz * u_ip12n[8] + nz * sx * ty * tz * u_ip12n[8] + nz * sy * tx * tz * u_ip12n[8] - 2 * nz * sz * tx * ty * u_ip12n[8] - 2 * sx * sz * ty * tz * u_ip12n[3] - 2 * sy * sz * tx * tz * u_ip12n[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));

    u_ip12p_T[0] = nx * u_ip12p[0] + ny * u_ip12p[1] + nz * u_ip12p[2];
    u_ip12p_T[1] = sx * u_ip12p[0] + sy * u_ip12p[1] + sz * u_ip12p[2];
    u_ip12p_T[2] = tx * u_ip12p[0] + ty * u_ip12p[1] + tz * u_ip12p[2];
    u_ip12p_T[3] = u_ip12p[3] * nx * nx + u_ip12p[8] * nx * ny + u_ip12p[7] * nx * nz + u_ip12p[4] * ny * ny + u_ip12p[6] * ny * nz + u_ip12p[5] * nz * nz;
    u_ip12p_T[4] = u_ip12p[3] * sx * sx + u_ip12p[8] * sx * sy + u_ip12p[7] * sx * sz + u_ip12p[4] * sy * sy + u_ip12p[6] * sy * sz + u_ip12p[5] * sz * sz;
    u_ip12p_T[5] = u_ip12p[3] * tx * tx + u_ip12p[8] * tx * ty + u_ip12p[7] * tx * tz + u_ip12p[4] * ty * ty + u_ip12p[6] * ty * tz + u_ip12p[5] * tz * tz;
    u_ip12p_T[6] = u_ip12p[8] * (sx * ty + sy * tx) + u_ip12p[7] * (sx * tz + sz * tx) + u_ip12p[6] * (sy * tz + sz * ty) + 2 * sx * tx * u_ip12p[3] + 2 * sy * ty * u_ip12p[4] + 2 * sz * tz * u_ip12p[5];
    u_ip12p_T[7] = u_ip12p[8] * (nx * ty + ny * tx) + u_ip12p[7] * (nx * tz + nz * tx) + u_ip12p[6] * (ny * tz + nz * ty) + 2 * nx * tx * u_ip12p[3] + 2 * ny * ty * u_ip12p[4] + 2 * nz * tz * u_ip12p[5];
    u_ip12p_T[8] = u_ip12p[8] * (nx * sy + ny * sx) + u_ip12p[7] * (nx * sz + nz * sx) + u_ip12p[6] * (ny * sz + nz * sy) + 2 * nx * sx * u_ip12p[3] + 2 * ny * sy * u_ip12p[4] + 2 * nz * sz * u_ip12p[5];
    // u_ip12p_T[0] = (ny * sz * u_ip12p[2] - nz * sy * u_ip12p[2] - ny * tz * u_ip12p[1] + nz * ty * u_ip12p[1] + sy * tz * u_ip12p[0] - sz * ty * u_ip12p[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    // u_ip12p_T[1] = -(nx * sz * u_ip12p[2] - nz * sx * u_ip12p[2] - nx * tz * u_ip12p[1] + nz * tx * u_ip12p[1] + sx * tz * u_ip12p[0] - sz * tx * u_ip12p[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    // u_ip12p_T[2] = (nx * sy * u_ip12p[2] - ny * sx * u_ip12p[2] - nx * ty * u_ip12p[1] + ny * tx * u_ip12p[1] + sx * ty * u_ip12p[0] - sy * tx * u_ip12p[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    // u_ip12p_T[3] = (u_ip12p[5] * (ny * ny) * (sz * sz) - u_ip12p[6] * (ny * ny) * sz * tz + u_ip12p[4] * (ny * ny) * (tz * tz) - 2 * u_ip12p[5] * ny * nz * sy * sz + u_ip12p[6] * ny * nz * sy * tz + u_ip12p[6] * ny * nz * sz * ty - 2 * u_ip12p[4] * ny * nz * ty * tz + u_ip12p[7] * ny * sy * sz * tz - u_ip12p[8] * ny * sy * (tz * tz) - u_ip12p[7] * ny * (sz * sz) * ty + u_ip12p[8] * ny * sz * ty * tz + u_ip12p[5] * (nz * nz) * (sy * sy) - u_ip12p[6] * (nz * nz) * sy * ty + u_ip12p[4] * (nz * nz) * (ty * ty) - u_ip12p[7] * nz * (sy * sy) * tz + u_ip12p[7] * nz * sy * sz * ty + u_ip12p[8] * nz * sy * ty * tz - u_ip12p[8] * nz * sz * (ty * ty) + u_ip12p[3] * (sy * sy) * (tz * tz) - 2 * u_ip12p[3] * sy * sz * ty * tz + u_ip12p[3] * (sz * sz) * (ty * ty)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12p_T[4] = (u_ip12p[5] * (nx * nx) * (sz * sz) - u_ip12p[6] * (nx * nx) * sz * tz + u_ip12p[4] * (nx * nx) * (tz * tz) - 2 * u_ip12p[5] * nx * nz * sx * sz + u_ip12p[6] * nx * nz * sx * tz + u_ip12p[6] * nx * nz * sz * tx - 2 * u_ip12p[4] * nx * nz * tx * tz + u_ip12p[7] * nx * sx * sz * tz - u_ip12p[8] * nx * sx * (tz * tz) - u_ip12p[7] * nx * (sz * sz) * tx + u_ip12p[8] * nx * sz * tx * tz + u_ip12p[5] * (nz * nz) * (sx * sx) - u_ip12p[6] * (nz * nz) * sx * tx + u_ip12p[4] * (nz * nz) * (tx * tx) - u_ip12p[7] * nz * (sx * sx) * tz + u_ip12p[7] * nz * sx * sz * tx + u_ip12p[8] * nz * sx * tx * tz - u_ip12p[8] * nz * sz * (tx * tx) + u_ip12p[3] * (sx * sx) * (tz * tz) - 2 * u_ip12p[3] * sx * sz * tx * tz + u_ip12p[3] * (sz * sz) * (tx * tx)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12p_T[5] = (u_ip12p[5] * (nx * nx) * (sy * sy) - u_ip12p[6] * (nx * nx) * sy * ty + u_ip12p[4] * (nx * nx) * (ty * ty) - 2 * u_ip12p[5] * nx * ny * sx * sy + u_ip12p[6] * nx * ny * sx * ty + u_ip12p[6] * nx * ny * sy * tx - 2 * u_ip12p[4] * nx * ny * tx * ty + u_ip12p[7] * nx * sx * sy * ty - u_ip12p[8] * nx * sx * (ty * ty) - u_ip12p[7] * nx * (sy * sy) * tx + u_ip12p[8] * nx * sy * tx * ty + u_ip12p[5] * (ny * ny) * (sx * sx) - u_ip12p[6] * (ny * ny) * sx * tx + u_ip12p[4] * (ny * ny) * (tx * tx) - u_ip12p[7] * ny * (sx * sx) * ty + u_ip12p[7] * ny * sx * sy * tx + u_ip12p[8] * ny * sx * tx * ty - u_ip12p[8] * ny * sy * (tx * tx) + u_ip12p[3] * (sx * sx) * (ty * ty) - 2 * u_ip12p[3] * sx * sy * tx * ty + u_ip12p[3] * (sy * sy) * (tx * tx)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12p_T[6] = -(2 * ny * nz * (sx * sx) * u_ip12p[5] + 2 * ny * nz * (tx * tx) * u_ip12p[4] + 2 * (nx * nx) * sy * sz * u_ip12p[5] - (nx * nx) * sy * tz * u_ip12p[6] - (nx * nx) * sz * ty * u_ip12p[6] - ny * (sx * sx) * tz * u_ip12p[7] - nz * (sx * sx) * ty * u_ip12p[7] - ny * sz * (tx * tx) * u_ip12p[8] - nz * sy * (tx * tx) * u_ip12p[8] + 2 * (nx * nx) * ty * tz * u_ip12p[4] + 2 * sy * sz * (tx * tx) * u_ip12p[3] + 2 * (sx * sx) * ty * tz * u_ip12p[3] - 2 * nx * ny * sx * sz * u_ip12p[5] - 2 * nx * nz * sx * sy * u_ip12p[5] + nx * ny * sx * tz * u_ip12p[6] + nx * ny * sz * tx * u_ip12p[6] + nx * nz * sx * ty * u_ip12p[6] + nx * nz * sy * tx * u_ip12p[6] - 2 * ny * nz * sx * tx * u_ip12p[6] - 2 * nx * ny * tx * tz * u_ip12p[4] - 2 * nx * nz * tx * ty * u_ip12p[4] + nx * sx * sy * tz * u_ip12p[7] + nx * sx * sz * ty * u_ip12p[7] - 2 * nx * sy * sz * tx * u_ip12p[7] + ny * sx * sz * tx * u_ip12p[7] + nz * sx * sy * tx * u_ip12p[7] - 2 * nx * sx * ty * tz * u_ip12p[8] + nx * sy * tx * tz * u_ip12p[8] + nx * sz * tx * ty * u_ip12p[8] + ny * sx * tx * tz * u_ip12p[8] + nz * sx * tx * ty * u_ip12p[8] - 2 * sx * sy * tx * tz * u_ip12p[3] - 2 * sx * sz * tx * ty * u_ip12p[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12p_T[7] = -(2 * nx * nz * (sy * sy) * u_ip12p[5] + 2 * nx * nz * (ty * ty) * u_ip12p[4] + 2 * (ny * ny) * sx * sz * u_ip12p[5] - (ny * ny) * sx * tz * u_ip12p[6] - (ny * ny) * sz * tx * u_ip12p[6] - nx * (sy * sy) * tz * u_ip12p[7] - nz * (sy * sy) * tx * u_ip12p[7] - nx * sz * (ty * ty) * u_ip12p[8] - nz * sx * (ty * ty) * u_ip12p[8] + 2 * (ny * ny) * tx * tz * u_ip12p[4] + 2 * sx * sz * (ty * ty) * u_ip12p[3] + 2 * (sy * sy) * tx * tz * u_ip12p[3] - 2 * nx * ny * sy * sz * u_ip12p[5] - 2 * ny * nz * sx * sy * u_ip12p[5] + nx * ny * sy * tz * u_ip12p[6] + nx * ny * sz * ty * u_ip12p[6] - 2 * nx * nz * sy * ty * u_ip12p[6] + ny * nz * sx * ty * u_ip12p[6] + ny * nz * sy * tx * u_ip12p[6] - 2 * nx * ny * ty * tz * u_ip12p[4] - 2 * ny * nz * tx * ty * u_ip12p[4] + nx * sy * sz * ty * u_ip12p[7] + ny * sx * sy * tz * u_ip12p[7] - 2 * ny * sx * sz * ty * u_ip12p[7] + ny * sy * sz * tx * u_ip12p[7] + nz * sx * sy * ty * u_ip12p[7] + nx * sy * ty * tz * u_ip12p[8] + ny * sx * ty * tz * u_ip12p[8] - 2 * ny * sy * tx * tz * u_ip12p[8] + ny * sz * tx * ty * u_ip12p[8] + nz * sy * tx * ty * u_ip12p[8] - 2 * sx * sy * ty * tz * u_ip12p[3] - 2 * sy * sz * tx * ty * u_ip12p[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12p_T[8] = -(2 * nx * ny * (sz * sz) * u_ip12p[5] + 2 * nx * ny * (tz * tz) * u_ip12p[4] + 2 * (nz * nz) * sx * sy * u_ip12p[5] - (nz * nz) * sx * ty * u_ip12p[6] - (nz * nz) * sy * tx * u_ip12p[6] - nx * (sz * sz) * ty * u_ip12p[7] - ny * (sz * sz) * tx * u_ip12p[7] - nx * sy * (tz * tz) * u_ip12p[8] - ny * sx * (tz * tz) * u_ip12p[8] + 2 * (nz * nz) * tx * ty * u_ip12p[4] + 2 * sx * sy * (tz * tz) * u_ip12p[3] + 2 * (sz * sz) * tx * ty * u_ip12p[3] - 2 * nx * nz * sy * sz * u_ip12p[5] - 2 * ny * nz * sx * sz * u_ip12p[5] - 2 * nx * ny * sz * tz * u_ip12p[6] + nx * nz * sy * tz * u_ip12p[6] + nx * nz * sz * ty * u_ip12p[6] + ny * nz * sx * tz * u_ip12p[6] + ny * nz * sz * tx * u_ip12p[6] - 2 * nx * nz * ty * tz * u_ip12p[4] - 2 * ny * nz * tx * tz * u_ip12p[4] + nx * sy * sz * tz * u_ip12p[7] + ny * sx * sz * tz * u_ip12p[7] - 2 * nz * sx * sy * tz * u_ip12p[7] + nz * sx * sz * ty * u_ip12p[7] + nz * sy * sz * tx * u_ip12p[7] + nx * sz * ty * tz * u_ip12p[8] + ny * sz * tx * tz * u_ip12p[8] + nz * sx * ty * tz * u_ip12p[8] + nz * sy * tx * tz * u_ip12p[8] - 2 * nz * sz * tx * ty * u_ip12p[8] - 2 * sx * sz * ty * tz * u_ip12p[3] - 2 * sy * sz * tx * tz * u_ip12p[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));

    // Calculate physical variables
    fu_ip12n[0] = (lambda_n * u_ip12n_T[4] + lambda_n * u_ip12n_T[5] + u_ip12n_T[3] * (lambda_n + 2 * mu_n));
    fu_ip12n[1] = (mu_n * u_ip12n_T[8]);
    fu_ip12n[2] = (mu_n * u_ip12n_T[7]);
    fu_ip12n[3] = (u_ip12n_T[0] / rho_n);
    fu_ip12n[4] = (0);
    fu_ip12n[5] = (0);
    fu_ip12n[6] = (0);
    fu_ip12n[7] = (u_ip12n_T[2] / rho_n);
    fu_ip12n[8] = (u_ip12n_T[1] / rho_n);

    fu_ip12p[0] = (lambda_p * u_ip12p_T[4] + lambda_p * u_ip12p_T[5] + u_ip12p_T[3] * (lambda_p + 2 * mu_p));
    fu_ip12p[1] = (mu_p * u_ip12p_T[8]);
    fu_ip12p[2] = (mu_p * u_ip12p_T[7]);
    fu_ip12p[3] = (u_ip12p_T[0] / rho_p);
    fu_ip12p[4] = (0);
    fu_ip12p[5] = (0);
    fu_ip12p[6] = (0);
    fu_ip12p[7] = (u_ip12p_T[2] / rho_p);
    fu_ip12p[8] = (u_ip12p_T[1] / rho_p);

    // Calculate Riemann flux
    tau_xx = -(fu_ip12p[0] - fu_ip12n[0]);
    tau_xy = -(fu_ip12p[1] - fu_ip12n[1]);
    tau_xz = -(fu_ip12p[2] - fu_ip12n[2]);
    v_x = -(fu_ip12p[3] - fu_ip12n[3]);
    v_y = -(fu_ip12p[8] - fu_ip12n[8]);
    v_z = -(fu_ip12p[7] - fu_ip12n[7]);

    Riemann_flux_T[0] = zp_n * (tau_xx + zp_p * v_x) / (zp_n + zp_p);
    Riemann_flux_T[1] = zs_n * (tau_xy + zs_p * v_y) / (zs_n + zs_p);
    Riemann_flux_T[2] = zs_n * (tau_xz + zs_p * v_z) / (zs_n + zs_p);
    Riemann_flux_T[3] = (tau_xx + zp_p * v_x) / (zp_n + zp_p);
    Riemann_flux_T[4] = 0;
    Riemann_flux_T[5] = 0;
    Riemann_flux_T[6] = 0;
    Riemann_flux_T[7] = (tau_xz + zs_p * v_z) / (zs_n + zs_p);
    Riemann_flux_T[8] = (tau_xy + zs_p * v_y) / (zs_n + zs_p);

    // Rotate back
    Riemann_flux[0] = (ny * sz * Riemann_flux_T[2] - nz * sy * Riemann_flux_T[2] - ny * tz * Riemann_flux_T[1] + nz * ty * Riemann_flux_T[1] + sy * tz * Riemann_flux_T[0] - sz * ty * Riemann_flux_T[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    Riemann_flux[1] = -(nx * sz * Riemann_flux_T[2] - nz * sx * Riemann_flux_T[2] - nx * tz * Riemann_flux_T[1] + nz * tx * Riemann_flux_T[1] + sx * tz * Riemann_flux_T[0] - sz * tx * Riemann_flux_T[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    Riemann_flux[2] = (nx * sy * Riemann_flux_T[2] - ny * sx * Riemann_flux_T[2] - nx * ty * Riemann_flux_T[1] + ny * tx * Riemann_flux_T[1] + sx * ty * Riemann_flux_T[0] - sy * tx * Riemann_flux_T[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    Riemann_flux[3] = (Riemann_flux_T[5] * (ny * ny) * (sz * sz) - Riemann_flux_T[6] * (ny * ny) * sz * tz + Riemann_flux_T[4] * (ny * ny) * (tz * tz) - 2 * Riemann_flux_T[5] * ny * nz * sy * sz + Riemann_flux_T[6] * ny * nz * sy * tz + Riemann_flux_T[6] * ny * nz * sz * ty - 2 * Riemann_flux_T[4] * ny * nz * ty * tz + Riemann_flux_T[7] * ny * sy * sz * tz - Riemann_flux_T[8] * ny * sy * (tz * tz) - Riemann_flux_T[7] * ny * (sz * sz) * ty + Riemann_flux_T[8] * ny * sz * ty * tz + Riemann_flux_T[5] * (nz * nz) * (sy * sy) - Riemann_flux_T[6] * (nz * nz) * sy * ty + Riemann_flux_T[4] * (nz * nz) * (ty * ty) - Riemann_flux_T[7] * nz * (sy * sy) * tz + Riemann_flux_T[7] * nz * sy * sz * ty + Riemann_flux_T[8] * nz * sy * ty * tz - Riemann_flux_T[8] * nz * sz * (ty * ty) + Riemann_flux_T[3] * (sy * sy) * (tz * tz) - 2 * Riemann_flux_T[3] * sy * sz * ty * tz + Riemann_flux_T[3] * (sz * sz) * (ty * ty)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    Riemann_flux[4] = (Riemann_flux_T[5] * (nx * nx) * (sz * sz) - Riemann_flux_T[6] * (nx * nx) * sz * tz + Riemann_flux_T[4] * (nx * nx) * (tz * tz) - 2 * Riemann_flux_T[5] * nx * nz * sx * sz + Riemann_flux_T[6] * nx * nz * sx * tz + Riemann_flux_T[6] * nx * nz * sz * tx - 2 * Riemann_flux_T[4] * nx * nz * tx * tz + Riemann_flux_T[7] * nx * sx * sz * tz - Riemann_flux_T[8] * nx * sx * (tz * tz) - Riemann_flux_T[7] * nx * (sz * sz) * tx + Riemann_flux_T[8] * nx * sz * tx * tz + Riemann_flux_T[5] * (nz * nz) * (sx * sx) - Riemann_flux_T[6] * (nz * nz) * sx * tx + Riemann_flux_T[4] * (nz * nz) * (tx * tx) - Riemann_flux_T[7] * nz * (sx * sx) * tz + Riemann_flux_T[7] * nz * sx * sz * tx + Riemann_flux_T[8] * nz * sx * tx * tz - Riemann_flux_T[8] * nz * sz * (tx * tx) + Riemann_flux_T[3] * (sx * sx) * (tz * tz) - 2 * Riemann_flux_T[3] * sx * sz * tx * tz + Riemann_flux_T[3] * (sz * sz) * (tx * tx)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    Riemann_flux[5] = (Riemann_flux_T[5] * (nx * nx) * (sy * sy) - Riemann_flux_T[6] * (nx * nx) * sy * ty + Riemann_flux_T[4] * (nx * nx) * (ty * ty) - 2 * Riemann_flux_T[5] * nx * ny * sx * sy + Riemann_flux_T[6] * nx * ny * sx * ty + Riemann_flux_T[6] * nx * ny * sy * tx - 2 * Riemann_flux_T[4] * nx * ny * tx * ty + Riemann_flux_T[7] * nx * sx * sy * ty - Riemann_flux_T[8] * nx * sx * (ty * ty) - Riemann_flux_T[7] * nx * (sy * sy) * tx + Riemann_flux_T[8] * nx * sy * tx * ty + Riemann_flux_T[5] * (ny * ny) * (sx * sx) - Riemann_flux_T[6] * (ny * ny) * sx * tx + Riemann_flux_T[4] * (ny * ny) * (tx * tx) - Riemann_flux_T[7] * ny * (sx * sx) * ty + Riemann_flux_T[7] * ny * sx * sy * tx + Riemann_flux_T[8] * ny * sx * tx * ty - Riemann_flux_T[8] * ny * sy * (tx * tx) + Riemann_flux_T[3] * (sx * sx) * (ty * ty) - 2 * Riemann_flux_T[3] * sx * sy * tx * ty + Riemann_flux_T[3] * (sy * sy) * (tx * tx)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    Riemann_flux[6] = -(2 * ny * nz * (sx * sx) * Riemann_flux_T[5] + 2 * ny * nz * (tx * tx) * Riemann_flux_T[4] + 2 * (nx * nx) * sy * sz * Riemann_flux_T[5] - (nx * nx) * sy * tz * Riemann_flux_T[6] - (nx * nx) * sz * ty * Riemann_flux_T[6] - ny * (sx * sx) * tz * Riemann_flux_T[7] - nz * (sx * sx) * ty * Riemann_flux_T[7] - ny * sz * (tx * tx) * Riemann_flux_T[8] - nz * sy * (tx * tx) * Riemann_flux_T[8] + 2 * (nx * nx) * ty * tz * Riemann_flux_T[4] + 2 * sy * sz * (tx * tx) * Riemann_flux_T[3] + 2 * (sx * sx) * ty * tz * Riemann_flux_T[3] - 2 * nx * ny * sx * sz * Riemann_flux_T[5] - 2 * nx * nz * sx * sy * Riemann_flux_T[5] + nx * ny * sx * tz * Riemann_flux_T[6] + nx * ny * sz * tx * Riemann_flux_T[6] + nx * nz * sx * ty * Riemann_flux_T[6] + nx * nz * sy * tx * Riemann_flux_T[6] - 2 * ny * nz * sx * tx * Riemann_flux_T[6] - 2 * nx * ny * tx * tz * Riemann_flux_T[4] - 2 * nx * nz * tx * ty * Riemann_flux_T[4] + nx * sx * sy * tz * Riemann_flux_T[7] + nx * sx * sz * ty * Riemann_flux_T[7] - 2 * nx * sy * sz * tx * Riemann_flux_T[7] + ny * sx * sz * tx * Riemann_flux_T[7] + nz * sx * sy * tx * Riemann_flux_T[7] - 2 * nx * sx * ty * tz * Riemann_flux_T[8] + nx * sy * tx * tz * Riemann_flux_T[8] + nx * sz * tx * ty * Riemann_flux_T[8] + ny * sx * tx * tz * Riemann_flux_T[8] + nz * sx * tx * ty * Riemann_flux_T[8] - 2 * sx * sy * tx * tz * Riemann_flux_T[3] - 2 * sx * sz * tx * ty * Riemann_flux_T[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    Riemann_flux[7] = -(2 * nx * nz * (sy * sy) * Riemann_flux_T[5] + 2 * nx * nz * (ty * ty) * Riemann_flux_T[4] + 2 * (ny * ny) * sx * sz * Riemann_flux_T[5] - (ny * ny) * sx * tz * Riemann_flux_T[6] - (ny * ny) * sz * tx * Riemann_flux_T[6] - nx * (sy * sy) * tz * Riemann_flux_T[7] - nz * (sy * sy) * tx * Riemann_flux_T[7] - nx * sz * (ty * ty) * Riemann_flux_T[8] - nz * sx * (ty * ty) * Riemann_flux_T[8] + 2 * (ny * ny) * tx * tz * Riemann_flux_T[4] + 2 * sx * sz * (ty * ty) * Riemann_flux_T[3] + 2 * (sy * sy) * tx * tz * Riemann_flux_T[3] - 2 * nx * ny * sy * sz * Riemann_flux_T[5] - 2 * ny * nz * sx * sy * Riemann_flux_T[5] + nx * ny * sy * tz * Riemann_flux_T[6] + nx * ny * sz * ty * Riemann_flux_T[6] - 2 * nx * nz * sy * ty * Riemann_flux_T[6] + ny * nz * sx * ty * Riemann_flux_T[6] + ny * nz * sy * tx * Riemann_flux_T[6] - 2 * nx * ny * ty * tz * Riemann_flux_T[4] - 2 * ny * nz * tx * ty * Riemann_flux_T[4] + nx * sy * sz * ty * Riemann_flux_T[7] + ny * sx * sy * tz * Riemann_flux_T[7] - 2 * ny * sx * sz * ty * Riemann_flux_T[7] + ny * sy * sz * tx * Riemann_flux_T[7] + nz * sx * sy * ty * Riemann_flux_T[7] + nx * sy * ty * tz * Riemann_flux_T[8] + ny * sx * ty * tz * Riemann_flux_T[8] - 2 * ny * sy * tx * tz * Riemann_flux_T[8] + ny * sz * tx * ty * Riemann_flux_T[8] + nz * sy * tx * ty * Riemann_flux_T[8] - 2 * sx * sy * ty * tz * Riemann_flux_T[3] - 2 * sy * sz * tx * ty * Riemann_flux_T[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    Riemann_flux[8] = -(2 * nx * ny * (sz * sz) * Riemann_flux_T[5] + 2 * nx * ny * (tz * tz) * Riemann_flux_T[4] + 2 * (nz * nz) * sx * sy * Riemann_flux_T[5] - (nz * nz) * sx * ty * Riemann_flux_T[6] - (nz * nz) * sy * tx * Riemann_flux_T[6] - nx * (sz * sz) * ty * Riemann_flux_T[7] - ny * (sz * sz) * tx * Riemann_flux_T[7] - nx * sy * (tz * tz) * Riemann_flux_T[8] - ny * sx * (tz * tz) * Riemann_flux_T[8] + 2 * (nz * nz) * tx * ty * Riemann_flux_T[4] + 2 * sx * sy * (tz * tz) * Riemann_flux_T[3] + 2 * (sz * sz) * tx * ty * Riemann_flux_T[3] - 2 * nx * nz * sy * sz * Riemann_flux_T[5] - 2 * ny * nz * sx * sz * Riemann_flux_T[5] - 2 * nx * ny * sz * tz * Riemann_flux_T[6] + nx * nz * sy * tz * Riemann_flux_T[6] + nx * nz * sz * ty * Riemann_flux_T[6] + ny * nz * sx * tz * Riemann_flux_T[6] + ny * nz * sz * tx * Riemann_flux_T[6] - 2 * nx * nz * ty * tz * Riemann_flux_T[4] - 2 * ny * nz * tx * tz * Riemann_flux_T[4] + nx * sy * sz * tz * Riemann_flux_T[7] + ny * sx * sz * tz * Riemann_flux_T[7] - 2 * nz * sx * sy * tz * Riemann_flux_T[7] + nz * sx * sz * ty * Riemann_flux_T[7] + nz * sy * sz * tx * Riemann_flux_T[7] + nx * sz * ty * tz * Riemann_flux_T[8] + ny * sz * tx * tz * Riemann_flux_T[8] + nz * sx * ty * tz * Riemann_flux_T[8] + nz * sy * tx * tz * Riemann_flux_T[8] - 2 * nz * sz * tx * ty * Riemann_flux_T[8] - 2 * sx * sz * ty * tz * Riemann_flux_T[3] - 2 * sy * sz * tx * tz * Riemann_flux_T[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // Riemann_flux[0] = nx * Riemann_flux_T[0] + ny * Riemann_flux_T[1] + nz * Riemann_flux_T[2];
    // Riemann_flux[1] = sx * Riemann_flux_T[0] + sy * Riemann_flux_T[1] + sz * Riemann_flux_T[2];
    // Riemann_flux[2] = tx * Riemann_flux_T[0] + ty * Riemann_flux_T[1] + tz * Riemann_flux_T[2];
    // Riemann_flux[3] = Riemann_flux_T[3] * nx * nx + Riemann_flux_T[8] * nx * ny + Riemann_flux_T[7] * nx * nz + Riemann_flux_T[4] * ny * ny + Riemann_flux_T[6] * ny * nz + Riemann_flux_T[5] * nz * nz;
    // Riemann_flux[4] = Riemann_flux_T[3] * sx * sx + Riemann_flux_T[8] * sx * sy + Riemann_flux_T[7] * sx * sz + Riemann_flux_T[4] * sy * sy + Riemann_flux_T[6] * sy * sz + Riemann_flux_T[5] * sz * sz;
    // Riemann_flux[5] = Riemann_flux_T[3] * tx * tx + Riemann_flux_T[8] * tx * ty + Riemann_flux_T[7] * tx * tz + Riemann_flux_T[4] * ty * ty + Riemann_flux_T[6] * ty * tz + Riemann_flux_T[5] * tz * tz;
    // Riemann_flux[6] = Riemann_flux_T[8] * (sx * ty + sy * tx) + Riemann_flux_T[7] * (sx * tz + sz * tx) + Riemann_flux_T[6] * (sy * tz + sz * ty) + 2 * sx * tx * Riemann_flux_T[3] + 2 * sy * ty * Riemann_flux_T[4] + 2 * sz * tz * Riemann_flux_T[5];
    // Riemann_flux[7] = Riemann_flux_T[8] * (nx * ty + ny * tx) + Riemann_flux_T[7] * (nx * tz + nz * tx) + Riemann_flux_T[6] * (ny * tz + nz * ty) + 2 * nx * tx * Riemann_flux_T[3] + 2 * ny * ty * Riemann_flux_T[4] + 2 * nz * tz * Riemann_flux_T[5];
    // Riemann_flux[8] = Riemann_flux_T[8] * (nx * sy + ny * sx) + Riemann_flux_T[7] * (nx * sz + nz * sx) + Riemann_flux_T[6] * (ny * sz + nz * sy) + 2 * nx * sx * Riemann_flux_T[3] + 2 * ny * sy * Riemann_flux_T[4] + 2 * nz * sz * Riemann_flux_T[5];

#endif // UW

    for (int n = 0; n < 9; n++)
    {
        Fu_ip12x[idx * WSIZE + n] = Riemann_flux[n] - 1.0f / 24 * order2_approximation(Fu[idx_n2 * WSIZE + n], Fu[idx_n1 * WSIZE + n], Fu[idx * WSIZE + n], Fu[idx_p1 * WSIZE + n], Fu[idx_p2 * WSIZE + n], Fu[idx_p3 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Fu[idx_n2 * WSIZE + n], Fu[idx_n1 * WSIZE + n], Fu[idx * WSIZE + n], Fu[idx_p1 * WSIZE + n], Fu[idx_p2 * WSIZE + n], Fu[idx_p3 * WSIZE + n]);
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

    // Shock capture interpolation
    for (int n = 0; n < 9; n++)
    {
        u_ip12n[n] = WENO5_interpolation(W[idx_n2 * WSIZE + n], W[idx_n1 * WSIZE + n], W[idx * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx_p2 * WSIZE + n]);
        u_ip12p[n] = WENO5_interpolation(W[idx_p3 * WSIZE + n], W[idx_p2 * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx * WSIZE + n], W[idx_n1 * WSIZE + n]);
    }

#ifdef LF
    // Riemann solver: Lax-Friedrichs
    fu_ip12n[0] = -(et_x_J_h * (lambda * u_ip12n[4] + lambda * u_ip12n[5] + u_ip12n[3] * (lambda + 2 * mu)) + mu * et_y_J_h * u_ip12n[8] + mu * et_z_J_h * u_ip12n[7]);
    fu_ip12n[1] = -(et_y_J_h * (lambda * u_ip12n[3] + lambda * u_ip12n[5] + u_ip12n[4] * (lambda + 2 * mu)) + mu * et_x_J_h * u_ip12n[8] + mu * et_z_J_h * u_ip12n[6]);
    fu_ip12n[2] = -(et_z_J_h * (lambda * u_ip12n[3] + lambda * u_ip12n[4] + u_ip12n[5] * (lambda + 2 * mu)) + mu * et_x_J_h * u_ip12n[7] + mu * et_y_J_h * u_ip12n[6]);
    fu_ip12n[3] = -((et_x_J_h * u_ip12n[0]) * buoyancy);
    fu_ip12n[4] = -((et_y_J_h * u_ip12n[1]) * buoyancy);
    fu_ip12n[5] = -((et_z_J_h * u_ip12n[2]) * buoyancy);
    fu_ip12n[6] = -((et_y_J_h * u_ip12n[2]) * buoyancy + (et_z_J_h * u_ip12n[1]) * buoyancy);
    fu_ip12n[7] = -((et_x_J_h * u_ip12n[2]) * buoyancy + (et_z_J_h * u_ip12n[0]) * buoyancy);
    fu_ip12n[8] = -((et_x_J_h * u_ip12n[1]) * buoyancy + (et_y_J_h * u_ip12n[0]) * buoyancy);

    fu_ip12p[0] = -(et_x_J_h * (lambda * u_ip12p[4] + lambda * u_ip12p[5] + u_ip12p[3] * (lambda + 2 * mu)) + mu * et_y_J_h * u_ip12p[8] + mu * et_z_J_h * u_ip12p[7]);
    fu_ip12p[1] = -(et_y_J_h * (lambda * u_ip12p[3] + lambda * u_ip12p[5] + u_ip12p[4] * (lambda + 2 * mu)) + mu * et_x_J_h * u_ip12p[8] + mu * et_z_J_h * u_ip12p[6]);
    fu_ip12p[2] = -(et_z_J_h * (lambda * u_ip12p[3] + lambda * u_ip12p[4] + u_ip12p[5] * (lambda + 2 * mu)) + mu * et_x_J_h * u_ip12p[7] + mu * et_y_J_h * u_ip12p[6]);
    fu_ip12p[3] = -((et_x_J_h * u_ip12p[0]) * buoyancy);
    fu_ip12p[4] = -((et_y_J_h * u_ip12p[1]) * buoyancy);
    fu_ip12p[5] = -((et_z_J_h * u_ip12p[2]) * buoyancy);
    fu_ip12p[6] = -((et_y_J_h * u_ip12p[2]) * buoyancy + (et_z_J_h * u_ip12p[1]) * buoyancy);
    fu_ip12p[7] = -((et_x_J_h * u_ip12p[2]) * buoyancy + (et_z_J_h * u_ip12p[0]) * buoyancy);
    fu_ip12p[8] = -((et_x_J_h * u_ip12p[1]) * buoyancy + (et_y_J_h * u_ip12p[0]) * buoyancy);

    Riemann_flux[0] = 0.5f * (fu_ip12p[0] + fu_ip12n[0] - alpha * (u_ip12p[0] - u_ip12n[0]));
    Riemann_flux[1] = 0.5f * (fu_ip12p[1] + fu_ip12n[1] - alpha * (u_ip12p[1] - u_ip12n[1]));
    Riemann_flux[2] = 0.5f * (fu_ip12p[2] + fu_ip12n[2] - alpha * (u_ip12p[2] - u_ip12n[2]));
    Riemann_flux[3] = 0.5f * (fu_ip12p[3] + fu_ip12n[3] - alpha * (u_ip12p[3] - u_ip12n[3]));
    Riemann_flux[4] = 0.5f * (fu_ip12p[4] + fu_ip12n[4] - alpha * (u_ip12p[4] - u_ip12n[4]));
    Riemann_flux[5] = 0.5f * (fu_ip12p[5] + fu_ip12n[5] - alpha * (u_ip12p[5] - u_ip12n[5]));
    Riemann_flux[6] = 0.5f * (fu_ip12p[6] + fu_ip12n[6] - alpha * (u_ip12p[6] - u_ip12n[6]));
    Riemann_flux[7] = 0.5f * (fu_ip12p[7] + fu_ip12n[7] - alpha * (u_ip12p[7] - u_ip12n[7]));
    Riemann_flux[8] = 0.5f * (fu_ip12p[8] + fu_ip12n[8] - alpha * (u_ip12p[8] - u_ip12n[8]));
#endif

#ifdef UW // ! Still not correct
    mu_n = CJM[idx * CJMSIZE + 10];
    lambda_n = CJM[idx * CJMSIZE + 11];
    rho_n = 1.0 / CJM[idx * CJMSIZE + 12];

    mu_p = CJM[idx_p1 * CJMSIZE + 10];
    lambda_p = CJM[idx_p1 * CJMSIZE + 11];
    rho_p = 1.0 / CJM[idx_p1 * CJMSIZE + 12];

    zs_n = sqrt(CJM[idx * CJMSIZE + 10] / CJM[idx * CJMSIZE + 12]);
    zs_p = sqrt(CJM[idx_p1 * CJMSIZE + 10] / CJM[idx_p1 * CJMSIZE + 12]);
    zp_n = sqrt((CJM[idx * CJMSIZE + 11] + 2 * CJM[idx * CJMSIZE + 10]) / CJM[idx * CJMSIZE + 12]);
    zp_p = sqrt((CJM[idx_p1 * CJMSIZE + 11] + 2 * CJM[idx_p1 * CJMSIZE + 10]) / CJM[idx_p1 * CJMSIZE + 12]);

    // Calculate T
    nx = (CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) / (sqrt((CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) + (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) + (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]))); // nx;
    ny = (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) / (sqrt((CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) + (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) + (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]))); // ny;
    nz = (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]) / (sqrt((CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) + (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) + (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]))); // nz;
    sx = CJM[idx * CJMSIZE + 16] / (sqrt(CJM[idx * CJMSIZE + 16] * CJM[idx * CJMSIZE + 16] + CJM[idx * CJMSIZE + 17] * CJM[idx * CJMSIZE + 17] + CJM[idx * CJMSIZE + 18] * CJM[idx * CJMSIZE + 18]));                                                                                                                                                                                       // sx;
    sy = CJM[idx * CJMSIZE + 17] / (sqrt(CJM[idx * CJMSIZE + 16] * CJM[idx * CJMSIZE + 16] + CJM[idx * CJMSIZE + 17] * CJM[idx * CJMSIZE + 17] + CJM[idx * CJMSIZE + 18] * CJM[idx * CJMSIZE + 18]));                                                                                                                                                                                       // sy;
    sz = CJM[idx * CJMSIZE + 18] / (sqrt(CJM[idx * CJMSIZE + 16] * CJM[idx * CJMSIZE + 16] + CJM[idx * CJMSIZE + 17] * CJM[idx * CJMSIZE + 17] + CJM[idx * CJMSIZE + 18] * CJM[idx * CJMSIZE + 18]));                                                                                                                                                                                       // sz;
    tx = CJM[idx * CJMSIZE + 19] / (sqrt(CJM[idx * CJMSIZE + 19] * CJM[idx * CJMSIZE + 19] + CJM[idx * CJMSIZE + 20] * CJM[idx * CJMSIZE + 20] + CJM[idx * CJMSIZE + 21] * CJM[idx * CJMSIZE + 21]));                                                                                                                                                                                       // tx;
    ty = CJM[idx * CJMSIZE + 20] / (sqrt(CJM[idx * CJMSIZE + 19] * CJM[idx * CJMSIZE + 19] + CJM[idx * CJMSIZE + 20] * CJM[idx * CJMSIZE + 20] + CJM[idx * CJMSIZE + 21] * CJM[idx * CJMSIZE + 21]));                                                                                                                                                                                       // ty;
    tz = CJM[idx * CJMSIZE + 21] / (sqrt(CJM[idx * CJMSIZE + 19] * CJM[idx * CJMSIZE + 19] + CJM[idx * CJMSIZE + 20] * CJM[idx * CJMSIZE + 20] + CJM[idx * CJMSIZE + 21] * CJM[idx * CJMSIZE + 21]));                                                                                                                                                                                       // tz;

    // Rotate u
    u_ip12n_T[0] = nx * u_ip12n[0] + ny * u_ip12n[1] + nz * u_ip12n[2];
    u_ip12n_T[1] = sx * u_ip12n[0] + sy * u_ip12n[1] + sz * u_ip12n[2];
    u_ip12n_T[2] = tx * u_ip12n[0] + ty * u_ip12n[1] + tz * u_ip12n[2];
    u_ip12n_T[3] = u_ip12n[3] * nx * nx + u_ip12n[8] * nx * ny + u_ip12n[7] * nx * nz + u_ip12n[4] * ny * ny + u_ip12n[6] * ny * nz + u_ip12n[5] * nz * nz;
    u_ip12n_T[4] = u_ip12n[3] * sx * sx + u_ip12n[8] * sx * sy + u_ip12n[7] * sx * sz + u_ip12n[4] * sy * sy + u_ip12n[6] * sy * sz + u_ip12n[5] * sz * sz;
    u_ip12n_T[5] = u_ip12n[3] * tx * tx + u_ip12n[8] * tx * ty + u_ip12n[7] * tx * tz + u_ip12n[4] * ty * ty + u_ip12n[6] * ty * tz + u_ip12n[5] * tz * tz;
    u_ip12n_T[6] = u_ip12n[8] * (sx * ty + sy * tx) + u_ip12n[7] * (sx * tz + sz * tx) + u_ip12n[6] * (sy * tz + sz * ty) + 2 * sx * tx * u_ip12n[3] + 2 * sy * ty * u_ip12n[4] + 2 * sz * tz * u_ip12n[5];
    u_ip12n_T[7] = u_ip12n[8] * (nx * ty + ny * tx) + u_ip12n[7] * (nx * tz + nz * tx) + u_ip12n[6] * (ny * tz + nz * ty) + 2 * nx * tx * u_ip12n[3] + 2 * ny * ty * u_ip12n[4] + 2 * nz * tz * u_ip12n[5];
    u_ip12n_T[8] = u_ip12n[8] * (nx * sy + ny * sx) + u_ip12n[7] * (nx * sz + nz * sx) + u_ip12n[6] * (ny * sz + nz * sy) + 2 * nx * sx * u_ip12n[3] + 2 * ny * sy * u_ip12n[4] + 2 * nz * sz * u_ip12n[5];
    // u_ip12n_T[0] = (ny * sz * u_ip12n[2] - nz * sy * u_ip12n[2] - ny * tz * u_ip12n[1] + nz * ty * u_ip12n[1] + sy * tz * u_ip12n[0] - sz * ty * u_ip12n[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    // u_ip12n_T[1] = -(nx * sz * u_ip12n[2] - nz * sx * u_ip12n[2] - nx * tz * u_ip12n[1] + nz * tx * u_ip12n[1] + sx * tz * u_ip12n[0] - sz * tx * u_ip12n[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    // u_ip12n_T[2] = (nx * sy * u_ip12n[2] - ny * sx * u_ip12n[2] - nx * ty * u_ip12n[1] + ny * tx * u_ip12n[1] + sx * ty * u_ip12n[0] - sy * tx * u_ip12n[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    // u_ip12n_T[3] = (u_ip12n[5] * (ny * ny) * (sz * sz) - u_ip12n[6] * (ny * ny) * sz * tz + u_ip12n[4] * (ny * ny) * (tz * tz) - 2 * u_ip12n[5] * ny * nz * sy * sz + u_ip12n[6] * ny * nz * sy * tz + u_ip12n[6] * ny * nz * sz * ty - 2 * u_ip12n[4] * ny * nz * ty * tz + u_ip12n[7] * ny * sy * sz * tz - u_ip12n[8] * ny * sy * (tz * tz) - u_ip12n[7] * ny * (sz * sz) * ty + u_ip12n[8] * ny * sz * ty * tz + u_ip12n[5] * (nz * nz) * (sy * sy) - u_ip12n[6] * (nz * nz) * sy * ty + u_ip12n[4] * (nz * nz) * (ty * ty) - u_ip12n[7] * nz * (sy * sy) * tz + u_ip12n[7] * nz * sy * sz * ty + u_ip12n[8] * nz * sy * ty * tz - u_ip12n[8] * nz * sz * (ty * ty) + u_ip12n[3] * (sy * sy) * (tz * tz) - 2 * u_ip12n[3] * sy * sz * ty * tz + u_ip12n[3] * (sz * sz) * (ty * ty)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12n_T[4] = (u_ip12n[5] * (nx * nx) * (sz * sz) - u_ip12n[6] * (nx * nx) * sz * tz + u_ip12n[4] * (nx * nx) * (tz * tz) - 2 * u_ip12n[5] * nx * nz * sx * sz + u_ip12n[6] * nx * nz * sx * tz + u_ip12n[6] * nx * nz * sz * tx - 2 * u_ip12n[4] * nx * nz * tx * tz + u_ip12n[7] * nx * sx * sz * tz - u_ip12n[8] * nx * sx * (tz * tz) - u_ip12n[7] * nx * (sz * sz) * tx + u_ip12n[8] * nx * sz * tx * tz + u_ip12n[5] * (nz * nz) * (sx * sx) - u_ip12n[6] * (nz * nz) * sx * tx + u_ip12n[4] * (nz * nz) * (tx * tx) - u_ip12n[7] * nz * (sx * sx) * tz + u_ip12n[7] * nz * sx * sz * tx + u_ip12n[8] * nz * sx * tx * tz - u_ip12n[8] * nz * sz * (tx * tx) + u_ip12n[3] * (sx * sx) * (tz * tz) - 2 * u_ip12n[3] * sx * sz * tx * tz + u_ip12n[3] * (sz * sz) * (tx * tx)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12n_T[5] = (u_ip12n[5] * (nx * nx) * (sy * sy) - u_ip12n[6] * (nx * nx) * sy * ty + u_ip12n[4] * (nx * nx) * (ty * ty) - 2 * u_ip12n[5] * nx * ny * sx * sy + u_ip12n[6] * nx * ny * sx * ty + u_ip12n[6] * nx * ny * sy * tx - 2 * u_ip12n[4] * nx * ny * tx * ty + u_ip12n[7] * nx * sx * sy * ty - u_ip12n[8] * nx * sx * (ty * ty) - u_ip12n[7] * nx * (sy * sy) * tx + u_ip12n[8] * nx * sy * tx * ty + u_ip12n[5] * (ny * ny) * (sx * sx) - u_ip12n[6] * (ny * ny) * sx * tx + u_ip12n[4] * (ny * ny) * (tx * tx) - u_ip12n[7] * ny * (sx * sx) * ty + u_ip12n[7] * ny * sx * sy * tx + u_ip12n[8] * ny * sx * tx * ty - u_ip12n[8] * ny * sy * (tx * tx) + u_ip12n[3] * (sx * sx) * (ty * ty) - 2 * u_ip12n[3] * sx * sy * tx * ty + u_ip12n[3] * (sy * sy) * (tx * tx)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12n_T[6] = -(2 * ny * nz * (sx * sx) * u_ip12n[5] + 2 * ny * nz * (tx * tx) * u_ip12n[4] + 2 * (nx * nx) * sy * sz * u_ip12n[5] - (nx * nx) * sy * tz * u_ip12n[6] - (nx * nx) * sz * ty * u_ip12n[6] - ny * (sx * sx) * tz * u_ip12n[7] - nz * (sx * sx) * ty * u_ip12n[7] - ny * sz * (tx * tx) * u_ip12n[8] - nz * sy * (tx * tx) * u_ip12n[8] + 2 * (nx * nx) * ty * tz * u_ip12n[4] + 2 * sy * sz * (tx * tx) * u_ip12n[3] + 2 * (sx * sx) * ty * tz * u_ip12n[3] - 2 * nx * ny * sx * sz * u_ip12n[5] - 2 * nx * nz * sx * sy * u_ip12n[5] + nx * ny * sx * tz * u_ip12n[6] + nx * ny * sz * tx * u_ip12n[6] + nx * nz * sx * ty * u_ip12n[6] + nx * nz * sy * tx * u_ip12n[6] - 2 * ny * nz * sx * tx * u_ip12n[6] - 2 * nx * ny * tx * tz * u_ip12n[4] - 2 * nx * nz * tx * ty * u_ip12n[4] + nx * sx * sy * tz * u_ip12n[7] + nx * sx * sz * ty * u_ip12n[7] - 2 * nx * sy * sz * tx * u_ip12n[7] + ny * sx * sz * tx * u_ip12n[7] + nz * sx * sy * tx * u_ip12n[7] - 2 * nx * sx * ty * tz * u_ip12n[8] + nx * sy * tx * tz * u_ip12n[8] + nx * sz * tx * ty * u_ip12n[8] + ny * sx * tx * tz * u_ip12n[8] + nz * sx * tx * ty * u_ip12n[8] - 2 * sx * sy * tx * tz * u_ip12n[3] - 2 * sx * sz * tx * ty * u_ip12n[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12n_T[7] = -(2 * nx * nz * (sy * sy) * u_ip12n[5] + 2 * nx * nz * (ty * ty) * u_ip12n[4] + 2 * (ny * ny) * sx * sz * u_ip12n[5] - (ny * ny) * sx * tz * u_ip12n[6] - (ny * ny) * sz * tx * u_ip12n[6] - nx * (sy * sy) * tz * u_ip12n[7] - nz * (sy * sy) * tx * u_ip12n[7] - nx * sz * (ty * ty) * u_ip12n[8] - nz * sx * (ty * ty) * u_ip12n[8] + 2 * (ny * ny) * tx * tz * u_ip12n[4] + 2 * sx * sz * (ty * ty) * u_ip12n[3] + 2 * (sy * sy) * tx * tz * u_ip12n[3] - 2 * nx * ny * sy * sz * u_ip12n[5] - 2 * ny * nz * sx * sy * u_ip12n[5] + nx * ny * sy * tz * u_ip12n[6] + nx * ny * sz * ty * u_ip12n[6] - 2 * nx * nz * sy * ty * u_ip12n[6] + ny * nz * sx * ty * u_ip12n[6] + ny * nz * sy * tx * u_ip12n[6] - 2 * nx * ny * ty * tz * u_ip12n[4] - 2 * ny * nz * tx * ty * u_ip12n[4] + nx * sy * sz * ty * u_ip12n[7] + ny * sx * sy * tz * u_ip12n[7] - 2 * ny * sx * sz * ty * u_ip12n[7] + ny * sy * sz * tx * u_ip12n[7] + nz * sx * sy * ty * u_ip12n[7] + nx * sy * ty * tz * u_ip12n[8] + ny * sx * ty * tz * u_ip12n[8] - 2 * ny * sy * tx * tz * u_ip12n[8] + ny * sz * tx * ty * u_ip12n[8] + nz * sy * tx * ty * u_ip12n[8] - 2 * sx * sy * ty * tz * u_ip12n[3] - 2 * sy * sz * tx * ty * u_ip12n[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12n_T[8] = -(2 * nx * ny * (sz * sz) * u_ip12n[5] + 2 * nx * ny * (tz * tz) * u_ip12n[4] + 2 * (nz * nz) * sx * sy * u_ip12n[5] - (nz * nz) * sx * ty * u_ip12n[6] - (nz * nz) * sy * tx * u_ip12n[6] - nx * (sz * sz) * ty * u_ip12n[7] - ny * (sz * sz) * tx * u_ip12n[7] - nx * sy * (tz * tz) * u_ip12n[8] - ny * sx * (tz * tz) * u_ip12n[8] + 2 * (nz * nz) * tx * ty * u_ip12n[4] + 2 * sx * sy * (tz * tz) * u_ip12n[3] + 2 * (sz * sz) * tx * ty * u_ip12n[3] - 2 * nx * nz * sy * sz * u_ip12n[5] - 2 * ny * nz * sx * sz * u_ip12n[5] - 2 * nx * ny * sz * tz * u_ip12n[6] + nx * nz * sy * tz * u_ip12n[6] + nx * nz * sz * ty * u_ip12n[6] + ny * nz * sx * tz * u_ip12n[6] + ny * nz * sz * tx * u_ip12n[6] - 2 * nx * nz * ty * tz * u_ip12n[4] - 2 * ny * nz * tx * tz * u_ip12n[4] + nx * sy * sz * tz * u_ip12n[7] + ny * sx * sz * tz * u_ip12n[7] - 2 * nz * sx * sy * tz * u_ip12n[7] + nz * sx * sz * ty * u_ip12n[7] + nz * sy * sz * tx * u_ip12n[7] + nx * sz * ty * tz * u_ip12n[8] + ny * sz * tx * tz * u_ip12n[8] + nz * sx * ty * tz * u_ip12n[8] + nz * sy * tx * tz * u_ip12n[8] - 2 * nz * sz * tx * ty * u_ip12n[8] - 2 * sx * sz * ty * tz * u_ip12n[3] - 2 * sy * sz * tx * tz * u_ip12n[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));

    u_ip12p_T[0] = nx * u_ip12p[0] + ny * u_ip12p[1] + nz * u_ip12p[2];
    u_ip12p_T[1] = sx * u_ip12p[0] + sy * u_ip12p[1] + sz * u_ip12p[2];
    u_ip12p_T[2] = tx * u_ip12p[0] + ty * u_ip12p[1] + tz * u_ip12p[2];
    u_ip12p_T[3] = u_ip12p[3] * nx * nx + u_ip12p[8] * nx * ny + u_ip12p[7] * nx * nz + u_ip12p[4] * ny * ny + u_ip12p[6] * ny * nz + u_ip12p[5] * nz * nz;
    u_ip12p_T[4] = u_ip12p[3] * sx * sx + u_ip12p[8] * sx * sy + u_ip12p[7] * sx * sz + u_ip12p[4] * sy * sy + u_ip12p[6] * sy * sz + u_ip12p[5] * sz * sz;
    u_ip12p_T[5] = u_ip12p[3] * tx * tx + u_ip12p[8] * tx * ty + u_ip12p[7] * tx * tz + u_ip12p[4] * ty * ty + u_ip12p[6] * ty * tz + u_ip12p[5] * tz * tz;
    u_ip12p_T[6] = u_ip12p[8] * (sx * ty + sy * tx) + u_ip12p[7] * (sx * tz + sz * tx) + u_ip12p[6] * (sy * tz + sz * ty) + 2 * sx * tx * u_ip12p[3] + 2 * sy * ty * u_ip12p[4] + 2 * sz * tz * u_ip12p[5];
    u_ip12p_T[7] = u_ip12p[8] * (nx * ty + ny * tx) + u_ip12p[7] * (nx * tz + nz * tx) + u_ip12p[6] * (ny * tz + nz * ty) + 2 * nx * tx * u_ip12p[3] + 2 * ny * ty * u_ip12p[4] + 2 * nz * tz * u_ip12p[5];
    u_ip12p_T[8] = u_ip12p[8] * (nx * sy + ny * sx) + u_ip12p[7] * (nx * sz + nz * sx) + u_ip12p[6] * (ny * sz + nz * sy) + 2 * nx * sx * u_ip12p[3] + 2 * ny * sy * u_ip12p[4] + 2 * nz * sz * u_ip12p[5];
    // u_ip12p_T[0] = (ny * sz * u_ip12p[2] - nz * sy * u_ip12p[2] - ny * tz * u_ip12p[1] + nz * ty * u_ip12p[1] + sy * tz * u_ip12p[0] - sz * ty * u_ip12p[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    // u_ip12p_T[1] = -(nx * sz * u_ip12p[2] - nz * sx * u_ip12p[2] - nx * tz * u_ip12p[1] + nz * tx * u_ip12p[1] + sx * tz * u_ip12p[0] - sz * tx * u_ip12p[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    // u_ip12p_T[2] = (nx * sy * u_ip12p[2] - ny * sx * u_ip12p[2] - nx * ty * u_ip12p[1] + ny * tx * u_ip12p[1] + sx * ty * u_ip12p[0] - sy * tx * u_ip12p[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    // u_ip12p_T[3] = (u_ip12p[5] * (ny * ny) * (sz * sz) - u_ip12p[6] * (ny * ny) * sz * tz + u_ip12p[4] * (ny * ny) * (tz * tz) - 2 * u_ip12p[5] * ny * nz * sy * sz + u_ip12p[6] * ny * nz * sy * tz + u_ip12p[6] * ny * nz * sz * ty - 2 * u_ip12p[4] * ny * nz * ty * tz + u_ip12p[7] * ny * sy * sz * tz - u_ip12p[8] * ny * sy * (tz * tz) - u_ip12p[7] * ny * (sz * sz) * ty + u_ip12p[8] * ny * sz * ty * tz + u_ip12p[5] * (nz * nz) * (sy * sy) - u_ip12p[6] * (nz * nz) * sy * ty + u_ip12p[4] * (nz * nz) * (ty * ty) - u_ip12p[7] * nz * (sy * sy) * tz + u_ip12p[7] * nz * sy * sz * ty + u_ip12p[8] * nz * sy * ty * tz - u_ip12p[8] * nz * sz * (ty * ty) + u_ip12p[3] * (sy * sy) * (tz * tz) - 2 * u_ip12p[3] * sy * sz * ty * tz + u_ip12p[3] * (sz * sz) * (ty * ty)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12p_T[4] = (u_ip12p[5] * (nx * nx) * (sz * sz) - u_ip12p[6] * (nx * nx) * sz * tz + u_ip12p[4] * (nx * nx) * (tz * tz) - 2 * u_ip12p[5] * nx * nz * sx * sz + u_ip12p[6] * nx * nz * sx * tz + u_ip12p[6] * nx * nz * sz * tx - 2 * u_ip12p[4] * nx * nz * tx * tz + u_ip12p[7] * nx * sx * sz * tz - u_ip12p[8] * nx * sx * (tz * tz) - u_ip12p[7] * nx * (sz * sz) * tx + u_ip12p[8] * nx * sz * tx * tz + u_ip12p[5] * (nz * nz) * (sx * sx) - u_ip12p[6] * (nz * nz) * sx * tx + u_ip12p[4] * (nz * nz) * (tx * tx) - u_ip12p[7] * nz * (sx * sx) * tz + u_ip12p[7] * nz * sx * sz * tx + u_ip12p[8] * nz * sx * tx * tz - u_ip12p[8] * nz * sz * (tx * tx) + u_ip12p[3] * (sx * sx) * (tz * tz) - 2 * u_ip12p[3] * sx * sz * tx * tz + u_ip12p[3] * (sz * sz) * (tx * tx)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12p_T[5] = (u_ip12p[5] * (nx * nx) * (sy * sy) - u_ip12p[6] * (nx * nx) * sy * ty + u_ip12p[4] * (nx * nx) * (ty * ty) - 2 * u_ip12p[5] * nx * ny * sx * sy + u_ip12p[6] * nx * ny * sx * ty + u_ip12p[6] * nx * ny * sy * tx - 2 * u_ip12p[4] * nx * ny * tx * ty + u_ip12p[7] * nx * sx * sy * ty - u_ip12p[8] * nx * sx * (ty * ty) - u_ip12p[7] * nx * (sy * sy) * tx + u_ip12p[8] * nx * sy * tx * ty + u_ip12p[5] * (ny * ny) * (sx * sx) - u_ip12p[6] * (ny * ny) * sx * tx + u_ip12p[4] * (ny * ny) * (tx * tx) - u_ip12p[7] * ny * (sx * sx) * ty + u_ip12p[7] * ny * sx * sy * tx + u_ip12p[8] * ny * sx * tx * ty - u_ip12p[8] * ny * sy * (tx * tx) + u_ip12p[3] * (sx * sx) * (ty * ty) - 2 * u_ip12p[3] * sx * sy * tx * ty + u_ip12p[3] * (sy * sy) * (tx * tx)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12p_T[6] = -(2 * ny * nz * (sx * sx) * u_ip12p[5] + 2 * ny * nz * (tx * tx) * u_ip12p[4] + 2 * (nx * nx) * sy * sz * u_ip12p[5] - (nx * nx) * sy * tz * u_ip12p[6] - (nx * nx) * sz * ty * u_ip12p[6] - ny * (sx * sx) * tz * u_ip12p[7] - nz * (sx * sx) * ty * u_ip12p[7] - ny * sz * (tx * tx) * u_ip12p[8] - nz * sy * (tx * tx) * u_ip12p[8] + 2 * (nx * nx) * ty * tz * u_ip12p[4] + 2 * sy * sz * (tx * tx) * u_ip12p[3] + 2 * (sx * sx) * ty * tz * u_ip12p[3] - 2 * nx * ny * sx * sz * u_ip12p[5] - 2 * nx * nz * sx * sy * u_ip12p[5] + nx * ny * sx * tz * u_ip12p[6] + nx * ny * sz * tx * u_ip12p[6] + nx * nz * sx * ty * u_ip12p[6] + nx * nz * sy * tx * u_ip12p[6] - 2 * ny * nz * sx * tx * u_ip12p[6] - 2 * nx * ny * tx * tz * u_ip12p[4] - 2 * nx * nz * tx * ty * u_ip12p[4] + nx * sx * sy * tz * u_ip12p[7] + nx * sx * sz * ty * u_ip12p[7] - 2 * nx * sy * sz * tx * u_ip12p[7] + ny * sx * sz * tx * u_ip12p[7] + nz * sx * sy * tx * u_ip12p[7] - 2 * nx * sx * ty * tz * u_ip12p[8] + nx * sy * tx * tz * u_ip12p[8] + nx * sz * tx * ty * u_ip12p[8] + ny * sx * tx * tz * u_ip12p[8] + nz * sx * tx * ty * u_ip12p[8] - 2 * sx * sy * tx * tz * u_ip12p[3] - 2 * sx * sz * tx * ty * u_ip12p[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12p_T[7] = -(2 * nx * nz * (sy * sy) * u_ip12p[5] + 2 * nx * nz * (ty * ty) * u_ip12p[4] + 2 * (ny * ny) * sx * sz * u_ip12p[5] - (ny * ny) * sx * tz * u_ip12p[6] - (ny * ny) * sz * tx * u_ip12p[6] - nx * (sy * sy) * tz * u_ip12p[7] - nz * (sy * sy) * tx * u_ip12p[7] - nx * sz * (ty * ty) * u_ip12p[8] - nz * sx * (ty * ty) * u_ip12p[8] + 2 * (ny * ny) * tx * tz * u_ip12p[4] + 2 * sx * sz * (ty * ty) * u_ip12p[3] + 2 * (sy * sy) * tx * tz * u_ip12p[3] - 2 * nx * ny * sy * sz * u_ip12p[5] - 2 * ny * nz * sx * sy * u_ip12p[5] + nx * ny * sy * tz * u_ip12p[6] + nx * ny * sz * ty * u_ip12p[6] - 2 * nx * nz * sy * ty * u_ip12p[6] + ny * nz * sx * ty * u_ip12p[6] + ny * nz * sy * tx * u_ip12p[6] - 2 * nx * ny * ty * tz * u_ip12p[4] - 2 * ny * nz * tx * ty * u_ip12p[4] + nx * sy * sz * ty * u_ip12p[7] + ny * sx * sy * tz * u_ip12p[7] - 2 * ny * sx * sz * ty * u_ip12p[7] + ny * sy * sz * tx * u_ip12p[7] + nz * sx * sy * ty * u_ip12p[7] + nx * sy * ty * tz * u_ip12p[8] + ny * sx * ty * tz * u_ip12p[8] - 2 * ny * sy * tx * tz * u_ip12p[8] + ny * sz * tx * ty * u_ip12p[8] + nz * sy * tx * ty * u_ip12p[8] - 2 * sx * sy * ty * tz * u_ip12p[3] - 2 * sy * sz * tx * ty * u_ip12p[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12p_T[8] = -(2 * nx * ny * (sz * sz) * u_ip12p[5] + 2 * nx * ny * (tz * tz) * u_ip12p[4] + 2 * (nz * nz) * sx * sy * u_ip12p[5] - (nz * nz) * sx * ty * u_ip12p[6] - (nz * nz) * sy * tx * u_ip12p[6] - nx * (sz * sz) * ty * u_ip12p[7] - ny * (sz * sz) * tx * u_ip12p[7] - nx * sy * (tz * tz) * u_ip12p[8] - ny * sx * (tz * tz) * u_ip12p[8] + 2 * (nz * nz) * tx * ty * u_ip12p[4] + 2 * sx * sy * (tz * tz) * u_ip12p[3] + 2 * (sz * sz) * tx * ty * u_ip12p[3] - 2 * nx * nz * sy * sz * u_ip12p[5] - 2 * ny * nz * sx * sz * u_ip12p[5] - 2 * nx * ny * sz * tz * u_ip12p[6] + nx * nz * sy * tz * u_ip12p[6] + nx * nz * sz * ty * u_ip12p[6] + ny * nz * sx * tz * u_ip12p[6] + ny * nz * sz * tx * u_ip12p[6] - 2 * nx * nz * ty * tz * u_ip12p[4] - 2 * ny * nz * tx * tz * u_ip12p[4] + nx * sy * sz * tz * u_ip12p[7] + ny * sx * sz * tz * u_ip12p[7] - 2 * nz * sx * sy * tz * u_ip12p[7] + nz * sx * sz * ty * u_ip12p[7] + nz * sy * sz * tx * u_ip12p[7] + nx * sz * ty * tz * u_ip12p[8] + ny * sz * tx * tz * u_ip12p[8] + nz * sx * ty * tz * u_ip12p[8] + nz * sy * tx * tz * u_ip12p[8] - 2 * nz * sz * tx * ty * u_ip12p[8] - 2 * sx * sz * ty * tz * u_ip12p[3] - 2 * sy * sz * tx * tz * u_ip12p[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));

    // Calculate physical variables
    fu_ip12n[0] = (lambda_n * u_ip12n_T[4] + lambda_n * u_ip12n_T[5] + u_ip12n_T[3] * (lambda_n + 2 * mu_n));
    fu_ip12n[1] = (mu_n * u_ip12n_T[8]);
    fu_ip12n[2] = (mu_n * u_ip12n_T[7]);
    fu_ip12n[3] = (u_ip12n_T[0] / rho_n);
    fu_ip12n[4] = (0);
    fu_ip12n[5] = (0);
    fu_ip12n[6] = (0);
    fu_ip12n[7] = (u_ip12n_T[2] / rho_n);
    fu_ip12n[8] = (u_ip12n_T[1] / rho_n);

    fu_ip12p[0] = (lambda_p * u_ip12p_T[4] + lambda_p * u_ip12p_T[5] + u_ip12p_T[3] * (lambda_p + 2 * mu_p));
    fu_ip12p[1] = (mu_p * u_ip12p_T[8]);
    fu_ip12p[2] = (mu_p * u_ip12p_T[7]);
    fu_ip12p[3] = (u_ip12p_T[0] / rho_p);
    fu_ip12p[4] = (0);
    fu_ip12p[5] = (0);
    fu_ip12p[6] = (0);
    fu_ip12p[7] = (u_ip12p_T[2] / rho_p);
    fu_ip12p[8] = (u_ip12p_T[1] / rho_p);

    // Calculate Riemann flux
    tau_xx = -(fu_ip12p[0] - fu_ip12n[0]);
    tau_xy = -(fu_ip12p[1] - fu_ip12n[1]);
    tau_xz = -(fu_ip12p[2] - fu_ip12n[2]);
    v_x = -(fu_ip12p[3] - fu_ip12n[3]);
    v_y = -(fu_ip12p[8] - fu_ip12n[8]);
    v_z = -(fu_ip12p[7] - fu_ip12n[7]);

    Riemann_flux_T[0] = zp_n * (tau_xx + zp_p * v_x) / (zp_n + zp_p);
    Riemann_flux_T[1] = zs_n * (tau_xy + zs_p * v_y) / (zs_n + zs_p);
    Riemann_flux_T[2] = zs_n * (tau_xz + zs_p * v_z) / (zs_n + zs_p);
    Riemann_flux_T[3] = (tau_xx + zp_p * v_x) / (zp_n + zp_p);
    Riemann_flux_T[4] = 0;
    Riemann_flux_T[5] = 0;
    Riemann_flux_T[6] = 0;
    Riemann_flux_T[7] = (tau_xz + zs_p * v_z) / (zs_n + zs_p);
    Riemann_flux_T[8] = (tau_xy + zs_p * v_y) / (zs_n + zs_p);

    // Rotate back
    Riemann_flux[0] = (ny * sz * Riemann_flux_T[2] - nz * sy * Riemann_flux_T[2] - ny * tz * Riemann_flux_T[1] + nz * ty * Riemann_flux_T[1] + sy * tz * Riemann_flux_T[0] - sz * ty * Riemann_flux_T[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    Riemann_flux[1] = -(nx * sz * Riemann_flux_T[2] - nz * sx * Riemann_flux_T[2] - nx * tz * Riemann_flux_T[1] + nz * tx * Riemann_flux_T[1] + sx * tz * Riemann_flux_T[0] - sz * tx * Riemann_flux_T[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    Riemann_flux[2] = (nx * sy * Riemann_flux_T[2] - ny * sx * Riemann_flux_T[2] - nx * ty * Riemann_flux_T[1] + ny * tx * Riemann_flux_T[1] + sx * ty * Riemann_flux_T[0] - sy * tx * Riemann_flux_T[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    Riemann_flux[3] = (Riemann_flux_T[5] * (ny * ny) * (sz * sz) - Riemann_flux_T[6] * (ny * ny) * sz * tz + Riemann_flux_T[4] * (ny * ny) * (tz * tz) - 2 * Riemann_flux_T[5] * ny * nz * sy * sz + Riemann_flux_T[6] * ny * nz * sy * tz + Riemann_flux_T[6] * ny * nz * sz * ty - 2 * Riemann_flux_T[4] * ny * nz * ty * tz + Riemann_flux_T[7] * ny * sy * sz * tz - Riemann_flux_T[8] * ny * sy * (tz * tz) - Riemann_flux_T[7] * ny * (sz * sz) * ty + Riemann_flux_T[8] * ny * sz * ty * tz + Riemann_flux_T[5] * (nz * nz) * (sy * sy) - Riemann_flux_T[6] * (nz * nz) * sy * ty + Riemann_flux_T[4] * (nz * nz) * (ty * ty) - Riemann_flux_T[7] * nz * (sy * sy) * tz + Riemann_flux_T[7] * nz * sy * sz * ty + Riemann_flux_T[8] * nz * sy * ty * tz - Riemann_flux_T[8] * nz * sz * (ty * ty) + Riemann_flux_T[3] * (sy * sy) * (tz * tz) - 2 * Riemann_flux_T[3] * sy * sz * ty * tz + Riemann_flux_T[3] * (sz * sz) * (ty * ty)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    Riemann_flux[4] = (Riemann_flux_T[5] * (nx * nx) * (sz * sz) - Riemann_flux_T[6] * (nx * nx) * sz * tz + Riemann_flux_T[4] * (nx * nx) * (tz * tz) - 2 * Riemann_flux_T[5] * nx * nz * sx * sz + Riemann_flux_T[6] * nx * nz * sx * tz + Riemann_flux_T[6] * nx * nz * sz * tx - 2 * Riemann_flux_T[4] * nx * nz * tx * tz + Riemann_flux_T[7] * nx * sx * sz * tz - Riemann_flux_T[8] * nx * sx * (tz * tz) - Riemann_flux_T[7] * nx * (sz * sz) * tx + Riemann_flux_T[8] * nx * sz * tx * tz + Riemann_flux_T[5] * (nz * nz) * (sx * sx) - Riemann_flux_T[6] * (nz * nz) * sx * tx + Riemann_flux_T[4] * (nz * nz) * (tx * tx) - Riemann_flux_T[7] * nz * (sx * sx) * tz + Riemann_flux_T[7] * nz * sx * sz * tx + Riemann_flux_T[8] * nz * sx * tx * tz - Riemann_flux_T[8] * nz * sz * (tx * tx) + Riemann_flux_T[3] * (sx * sx) * (tz * tz) - 2 * Riemann_flux_T[3] * sx * sz * tx * tz + Riemann_flux_T[3] * (sz * sz) * (tx * tx)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    Riemann_flux[5] = (Riemann_flux_T[5] * (nx * nx) * (sy * sy) - Riemann_flux_T[6] * (nx * nx) * sy * ty + Riemann_flux_T[4] * (nx * nx) * (ty * ty) - 2 * Riemann_flux_T[5] * nx * ny * sx * sy + Riemann_flux_T[6] * nx * ny * sx * ty + Riemann_flux_T[6] * nx * ny * sy * tx - 2 * Riemann_flux_T[4] * nx * ny * tx * ty + Riemann_flux_T[7] * nx * sx * sy * ty - Riemann_flux_T[8] * nx * sx * (ty * ty) - Riemann_flux_T[7] * nx * (sy * sy) * tx + Riemann_flux_T[8] * nx * sy * tx * ty + Riemann_flux_T[5] * (ny * ny) * (sx * sx) - Riemann_flux_T[6] * (ny * ny) * sx * tx + Riemann_flux_T[4] * (ny * ny) * (tx * tx) - Riemann_flux_T[7] * ny * (sx * sx) * ty + Riemann_flux_T[7] * ny * sx * sy * tx + Riemann_flux_T[8] * ny * sx * tx * ty - Riemann_flux_T[8] * ny * sy * (tx * tx) + Riemann_flux_T[3] * (sx * sx) * (ty * ty) - 2 * Riemann_flux_T[3] * sx * sy * tx * ty + Riemann_flux_T[3] * (sy * sy) * (tx * tx)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    Riemann_flux[6] = -(2 * ny * nz * (sx * sx) * Riemann_flux_T[5] + 2 * ny * nz * (tx * tx) * Riemann_flux_T[4] + 2 * (nx * nx) * sy * sz * Riemann_flux_T[5] - (nx * nx) * sy * tz * Riemann_flux_T[6] - (nx * nx) * sz * ty * Riemann_flux_T[6] - ny * (sx * sx) * tz * Riemann_flux_T[7] - nz * (sx * sx) * ty * Riemann_flux_T[7] - ny * sz * (tx * tx) * Riemann_flux_T[8] - nz * sy * (tx * tx) * Riemann_flux_T[8] + 2 * (nx * nx) * ty * tz * Riemann_flux_T[4] + 2 * sy * sz * (tx * tx) * Riemann_flux_T[3] + 2 * (sx * sx) * ty * tz * Riemann_flux_T[3] - 2 * nx * ny * sx * sz * Riemann_flux_T[5] - 2 * nx * nz * sx * sy * Riemann_flux_T[5] + nx * ny * sx * tz * Riemann_flux_T[6] + nx * ny * sz * tx * Riemann_flux_T[6] + nx * nz * sx * ty * Riemann_flux_T[6] + nx * nz * sy * tx * Riemann_flux_T[6] - 2 * ny * nz * sx * tx * Riemann_flux_T[6] - 2 * nx * ny * tx * tz * Riemann_flux_T[4] - 2 * nx * nz * tx * ty * Riemann_flux_T[4] + nx * sx * sy * tz * Riemann_flux_T[7] + nx * sx * sz * ty * Riemann_flux_T[7] - 2 * nx * sy * sz * tx * Riemann_flux_T[7] + ny * sx * sz * tx * Riemann_flux_T[7] + nz * sx * sy * tx * Riemann_flux_T[7] - 2 * nx * sx * ty * tz * Riemann_flux_T[8] + nx * sy * tx * tz * Riemann_flux_T[8] + nx * sz * tx * ty * Riemann_flux_T[8] + ny * sx * tx * tz * Riemann_flux_T[8] + nz * sx * tx * ty * Riemann_flux_T[8] - 2 * sx * sy * tx * tz * Riemann_flux_T[3] - 2 * sx * sz * tx * ty * Riemann_flux_T[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    Riemann_flux[7] = -(2 * nx * nz * (sy * sy) * Riemann_flux_T[5] + 2 * nx * nz * (ty * ty) * Riemann_flux_T[4] + 2 * (ny * ny) * sx * sz * Riemann_flux_T[5] - (ny * ny) * sx * tz * Riemann_flux_T[6] - (ny * ny) * sz * tx * Riemann_flux_T[6] - nx * (sy * sy) * tz * Riemann_flux_T[7] - nz * (sy * sy) * tx * Riemann_flux_T[7] - nx * sz * (ty * ty) * Riemann_flux_T[8] - nz * sx * (ty * ty) * Riemann_flux_T[8] + 2 * (ny * ny) * tx * tz * Riemann_flux_T[4] + 2 * sx * sz * (ty * ty) * Riemann_flux_T[3] + 2 * (sy * sy) * tx * tz * Riemann_flux_T[3] - 2 * nx * ny * sy * sz * Riemann_flux_T[5] - 2 * ny * nz * sx * sy * Riemann_flux_T[5] + nx * ny * sy * tz * Riemann_flux_T[6] + nx * ny * sz * ty * Riemann_flux_T[6] - 2 * nx * nz * sy * ty * Riemann_flux_T[6] + ny * nz * sx * ty * Riemann_flux_T[6] + ny * nz * sy * tx * Riemann_flux_T[6] - 2 * nx * ny * ty * tz * Riemann_flux_T[4] - 2 * ny * nz * tx * ty * Riemann_flux_T[4] + nx * sy * sz * ty * Riemann_flux_T[7] + ny * sx * sy * tz * Riemann_flux_T[7] - 2 * ny * sx * sz * ty * Riemann_flux_T[7] + ny * sy * sz * tx * Riemann_flux_T[7] + nz * sx * sy * ty * Riemann_flux_T[7] + nx * sy * ty * tz * Riemann_flux_T[8] + ny * sx * ty * tz * Riemann_flux_T[8] - 2 * ny * sy * tx * tz * Riemann_flux_T[8] + ny * sz * tx * ty * Riemann_flux_T[8] + nz * sy * tx * ty * Riemann_flux_T[8] - 2 * sx * sy * ty * tz * Riemann_flux_T[3] - 2 * sy * sz * tx * ty * Riemann_flux_T[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    Riemann_flux[8] = -(2 * nx * ny * (sz * sz) * Riemann_flux_T[5] + 2 * nx * ny * (tz * tz) * Riemann_flux_T[4] + 2 * (nz * nz) * sx * sy * Riemann_flux_T[5] - (nz * nz) * sx * ty * Riemann_flux_T[6] - (nz * nz) * sy * tx * Riemann_flux_T[6] - nx * (sz * sz) * ty * Riemann_flux_T[7] - ny * (sz * sz) * tx * Riemann_flux_T[7] - nx * sy * (tz * tz) * Riemann_flux_T[8] - ny * sx * (tz * tz) * Riemann_flux_T[8] + 2 * (nz * nz) * tx * ty * Riemann_flux_T[4] + 2 * sx * sy * (tz * tz) * Riemann_flux_T[3] + 2 * (sz * sz) * tx * ty * Riemann_flux_T[3] - 2 * nx * nz * sy * sz * Riemann_flux_T[5] - 2 * ny * nz * sx * sz * Riemann_flux_T[5] - 2 * nx * ny * sz * tz * Riemann_flux_T[6] + nx * nz * sy * tz * Riemann_flux_T[6] + nx * nz * sz * ty * Riemann_flux_T[6] + ny * nz * sx * tz * Riemann_flux_T[6] + ny * nz * sz * tx * Riemann_flux_T[6] - 2 * nx * nz * ty * tz * Riemann_flux_T[4] - 2 * ny * nz * tx * tz * Riemann_flux_T[4] + nx * sy * sz * tz * Riemann_flux_T[7] + ny * sx * sz * tz * Riemann_flux_T[7] - 2 * nz * sx * sy * tz * Riemann_flux_T[7] + nz * sx * sz * ty * Riemann_flux_T[7] + nz * sy * sz * tx * Riemann_flux_T[7] + nx * sz * ty * tz * Riemann_flux_T[8] + ny * sz * tx * tz * Riemann_flux_T[8] + nz * sx * ty * tz * Riemann_flux_T[8] + nz * sy * tx * tz * Riemann_flux_T[8] - 2 * nz * sz * tx * ty * Riemann_flux_T[8] - 2 * sx * sz * ty * tz * Riemann_flux_T[3] - 2 * sy * sz * tx * tz * Riemann_flux_T[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // Riemann_flux[0] = nx * Riemann_flux_T[0] + ny * Riemann_flux_T[1] + nz * Riemann_flux_T[2];
    // Riemann_flux[1] = sx * Riemann_flux_T[0] + sy * Riemann_flux_T[1] + sz * Riemann_flux_T[2];
    // Riemann_flux[2] = tx * Riemann_flux_T[0] + ty * Riemann_flux_T[1] + tz * Riemann_flux_T[2];
    // Riemann_flux[3] = Riemann_flux_T[3] * nx * nx + Riemann_flux_T[8] * nx * ny + Riemann_flux_T[7] * nx * nz + Riemann_flux_T[4] * ny * ny + Riemann_flux_T[6] * ny * nz + Riemann_flux_T[5] * nz * nz;
    // Riemann_flux[4] = Riemann_flux_T[3] * sx * sx + Riemann_flux_T[8] * sx * sy + Riemann_flux_T[7] * sx * sz + Riemann_flux_T[4] * sy * sy + Riemann_flux_T[6] * sy * sz + Riemann_flux_T[5] * sz * sz;
    // Riemann_flux[5] = Riemann_flux_T[3] * tx * tx + Riemann_flux_T[8] * tx * ty + Riemann_flux_T[7] * tx * tz + Riemann_flux_T[4] * ty * ty + Riemann_flux_T[6] * ty * tz + Riemann_flux_T[5] * tz * tz;
    // Riemann_flux[6] = Riemann_flux_T[8] * (sx * ty + sy * tx) + Riemann_flux_T[7] * (sx * tz + sz * tx) + Riemann_flux_T[6] * (sy * tz + sz * ty) + 2 * sx * tx * Riemann_flux_T[3] + 2 * sy * ty * Riemann_flux_T[4] + 2 * sz * tz * Riemann_flux_T[5];
    // Riemann_flux[7] = Riemann_flux_T[8] * (nx * ty + ny * tx) + Riemann_flux_T[7] * (nx * tz + nz * tx) + Riemann_flux_T[6] * (ny * tz + nz * ty) + 2 * nx * tx * Riemann_flux_T[3] + 2 * ny * ty * Riemann_flux_T[4] + 2 * nz * tz * Riemann_flux_T[5];
    // Riemann_flux[8] = Riemann_flux_T[8] * (nx * sy + ny * sx) + Riemann_flux_T[7] * (nx * sz + nz * sx) + Riemann_flux_T[6] * (ny * sz + nz * sy) + 2 * nx * sx * Riemann_flux_T[3] + 2 * ny * sy * Riemann_flux_T[4] + 2 * nz * sz * Riemann_flux_T[5];

#endif // UW

    for (int n = 0; n < 9; n++)
    {
        Fu_ip12y[idx * WSIZE + n] = Riemann_flux[n] - 1.0f / 24 * order2_approximation(Gu[idx_n2 * WSIZE + n], Gu[idx_n1 * WSIZE + n], Gu[idx * WSIZE + n], Gu[idx_p1 * WSIZE + n], Gu[idx_p2 * WSIZE + n], Gu[idx_p3 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Gu[idx_n2 * WSIZE + n], Gu[idx_n1 * WSIZE + n], Gu[idx * WSIZE + n], Gu[idx_p1 * WSIZE + n], Gu[idx_p2 * WSIZE + n], Gu[idx_p3 * WSIZE + n]);
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

    // Shock capture interpolation
    for (int n = 0; n < 9; n++)
    {
        u_ip12n[n] = WENO5_interpolation(W[idx_n2 * WSIZE + n], W[idx_n1 * WSIZE + n], W[idx * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx_p2 * WSIZE + n]);
        u_ip12p[n] = WENO5_interpolation(W[idx_p3 * WSIZE + n], W[idx_p2 * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx * WSIZE + n], W[idx_n1 * WSIZE + n]);
    }

#ifdef LF
    // Riemann solver: Lax-Friedrichs
    fu_ip12n[0] = -(zt_x_J_h * (lambda * u_ip12n[4] + lambda * u_ip12n[5] + u_ip12n[3] * (lambda + 2 * mu)) + mu * zt_y_J_h * u_ip12n[8] + mu * zt_z_J_h * u_ip12n[7]);
    fu_ip12n[1] = -(zt_y_J_h * (lambda * u_ip12n[3] + lambda * u_ip12n[5] + u_ip12n[4] * (lambda + 2 * mu)) + mu * zt_x_J_h * u_ip12n[8] + mu * zt_z_J_h * u_ip12n[6]);
    fu_ip12n[2] = -(zt_z_J_h * (lambda * u_ip12n[3] + lambda * u_ip12n[4] + u_ip12n[5] * (lambda + 2 * mu)) + mu * zt_x_J_h * u_ip12n[7] + mu * zt_y_J_h * u_ip12n[6]);
    fu_ip12n[3] = -((zt_x_J_h * u_ip12n[0]) * buoyancy);
    fu_ip12n[4] = -((zt_y_J_h * u_ip12n[1]) * buoyancy);
    fu_ip12n[5] = -((zt_z_J_h * u_ip12n[2]) * buoyancy);
    fu_ip12n[6] = -((zt_y_J_h * u_ip12n[2]) * buoyancy + (zt_z_J_h * u_ip12n[1]) * buoyancy);
    fu_ip12n[7] = -((zt_x_J_h * u_ip12n[2]) * buoyancy + (zt_z_J_h * u_ip12n[0]) * buoyancy);
    fu_ip12n[8] = -((zt_x_J_h * u_ip12n[1]) * buoyancy + (zt_y_J_h * u_ip12n[0]) * buoyancy);

    fu_ip12p[0] = -(zt_x_J_h * (lambda * u_ip12p[4] + lambda * u_ip12p[5] + u_ip12p[3] * (lambda + 2 * mu)) + mu * zt_y_J_h * u_ip12p[8] + mu * zt_z_J_h * u_ip12p[7]);
    fu_ip12p[1] = -(zt_y_J_h * (lambda * u_ip12p[3] + lambda * u_ip12p[5] + u_ip12p[4] * (lambda + 2 * mu)) + mu * zt_x_J_h * u_ip12p[8] + mu * zt_z_J_h * u_ip12p[6]);
    fu_ip12p[2] = -(zt_z_J_h * (lambda * u_ip12p[3] + lambda * u_ip12p[4] + u_ip12p[5] * (lambda + 2 * mu)) + mu * zt_x_J_h * u_ip12p[7] + mu * zt_y_J_h * u_ip12p[6]);
    fu_ip12p[3] = -((zt_x_J_h * u_ip12p[0]) * buoyancy);
    fu_ip12p[4] = -((zt_y_J_h * u_ip12p[1]) * buoyancy);
    fu_ip12p[5] = -((zt_z_J_h * u_ip12p[2]) * buoyancy);
    fu_ip12p[6] = -((zt_y_J_h * u_ip12p[2]) * buoyancy + (zt_z_J_h * u_ip12p[1]) * buoyancy);
    fu_ip12p[7] = -((zt_x_J_h * u_ip12p[2]) * buoyancy + (zt_z_J_h * u_ip12p[0]) * buoyancy);
    fu_ip12p[8] = -((zt_x_J_h * u_ip12p[1]) * buoyancy + (zt_y_J_h * u_ip12p[0]) * buoyancy);

    Riemann_flux[0] = 0.5f * (fu_ip12p[0] + fu_ip12n[0] - alpha * (u_ip12p[0] - u_ip12n[0]));
    Riemann_flux[1] = 0.5f * (fu_ip12p[1] + fu_ip12n[1] - alpha * (u_ip12p[1] - u_ip12n[1]));
    Riemann_flux[2] = 0.5f * (fu_ip12p[2] + fu_ip12n[2] - alpha * (u_ip12p[2] - u_ip12n[2]));
    Riemann_flux[3] = 0.5f * (fu_ip12p[3] + fu_ip12n[3] - alpha * (u_ip12p[3] - u_ip12n[3]));
    Riemann_flux[4] = 0.5f * (fu_ip12p[4] + fu_ip12n[4] - alpha * (u_ip12p[4] - u_ip12n[4]));
    Riemann_flux[5] = 0.5f * (fu_ip12p[5] + fu_ip12n[5] - alpha * (u_ip12p[5] - u_ip12n[5]));
    Riemann_flux[6] = 0.5f * (fu_ip12p[6] + fu_ip12n[6] - alpha * (u_ip12p[6] - u_ip12n[6]));
    Riemann_flux[7] = 0.5f * (fu_ip12p[7] + fu_ip12n[7] - alpha * (u_ip12p[7] - u_ip12n[7]));
    Riemann_flux[8] = 0.5f * (fu_ip12p[8] + fu_ip12n[8] - alpha * (u_ip12p[8] - u_ip12n[8]));
#endif

#ifdef UW // ! Still not correct
    mu_n = CJM[idx * CJMSIZE + 10];
    lambda_n = CJM[idx * CJMSIZE + 11];
    rho_n = 1.0 / CJM[idx * CJMSIZE + 12];

    mu_p = CJM[idx_p1 * CJMSIZE + 10];
    lambda_p = CJM[idx_p1 * CJMSIZE + 11];
    rho_p = 1.0 / CJM[idx_p1 * CJMSIZE + 12];

    zs_n = sqrt(CJM[idx * CJMSIZE + 10] / CJM[idx * CJMSIZE + 12]);
    zs_p = sqrt(CJM[idx_p1 * CJMSIZE + 10] / CJM[idx_p1 * CJMSIZE + 12]);
    zp_n = sqrt((CJM[idx * CJMSIZE + 11] + 2 * CJM[idx * CJMSIZE + 10]) / CJM[idx * CJMSIZE + 12]);
    zp_p = sqrt((CJM[idx_p1 * CJMSIZE + 11] + 2 * CJM[idx_p1 * CJMSIZE + 10]) / CJM[idx_p1 * CJMSIZE + 12]);

    // Calculate T
    nx = (CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) / (sqrt((CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) + (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) + (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]))); // nx;
    ny = (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) / (sqrt((CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) + (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) + (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]))); // ny;
    nz = (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]) / (sqrt((CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 0] / CJM[idx * CJMSIZE + 9]) + (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 1] / CJM[idx * CJMSIZE + 9]) + (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]) * (CJM[idx * CJMSIZE + 2] / CJM[idx * CJMSIZE + 9]))); // nz;
    sx = CJM[idx * CJMSIZE + 16] / (sqrt(CJM[idx * CJMSIZE + 16] * CJM[idx * CJMSIZE + 16] + CJM[idx * CJMSIZE + 17] * CJM[idx * CJMSIZE + 17] + CJM[idx * CJMSIZE + 18] * CJM[idx * CJMSIZE + 18]));                                                                                                                                                                                       // sx;
    sy = CJM[idx * CJMSIZE + 17] / (sqrt(CJM[idx * CJMSIZE + 16] * CJM[idx * CJMSIZE + 16] + CJM[idx * CJMSIZE + 17] * CJM[idx * CJMSIZE + 17] + CJM[idx * CJMSIZE + 18] * CJM[idx * CJMSIZE + 18]));                                                                                                                                                                                       // sy;
    sz = CJM[idx * CJMSIZE + 18] / (sqrt(CJM[idx * CJMSIZE + 16] * CJM[idx * CJMSIZE + 16] + CJM[idx * CJMSIZE + 17] * CJM[idx * CJMSIZE + 17] + CJM[idx * CJMSIZE + 18] * CJM[idx * CJMSIZE + 18]));                                                                                                                                                                                       // sz;
    tx = CJM[idx * CJMSIZE + 19] / (sqrt(CJM[idx * CJMSIZE + 19] * CJM[idx * CJMSIZE + 19] + CJM[idx * CJMSIZE + 20] * CJM[idx * CJMSIZE + 20] + CJM[idx * CJMSIZE + 21] * CJM[idx * CJMSIZE + 21]));                                                                                                                                                                                       // tx;
    ty = CJM[idx * CJMSIZE + 20] / (sqrt(CJM[idx * CJMSIZE + 19] * CJM[idx * CJMSIZE + 19] + CJM[idx * CJMSIZE + 20] * CJM[idx * CJMSIZE + 20] + CJM[idx * CJMSIZE + 21] * CJM[idx * CJMSIZE + 21]));                                                                                                                                                                                       // ty;
    tz = CJM[idx * CJMSIZE + 21] / (sqrt(CJM[idx * CJMSIZE + 19] * CJM[idx * CJMSIZE + 19] + CJM[idx * CJMSIZE + 20] * CJM[idx * CJMSIZE + 20] + CJM[idx * CJMSIZE + 21] * CJM[idx * CJMSIZE + 21]));                                                                                                                                                                                       // tz;

    // Rotate u
    u_ip12n_T[0] = nx * u_ip12n[0] + ny * u_ip12n[1] + nz * u_ip12n[2];
    u_ip12n_T[1] = sx * u_ip12n[0] + sy * u_ip12n[1] + sz * u_ip12n[2];
    u_ip12n_T[2] = tx * u_ip12n[0] + ty * u_ip12n[1] + tz * u_ip12n[2];
    u_ip12n_T[3] = u_ip12n[3] * nx * nx + u_ip12n[8] * nx * ny + u_ip12n[7] * nx * nz + u_ip12n[4] * ny * ny + u_ip12n[6] * ny * nz + u_ip12n[5] * nz * nz;
    u_ip12n_T[4] = u_ip12n[3] * sx * sx + u_ip12n[8] * sx * sy + u_ip12n[7] * sx * sz + u_ip12n[4] * sy * sy + u_ip12n[6] * sy * sz + u_ip12n[5] * sz * sz;
    u_ip12n_T[5] = u_ip12n[3] * tx * tx + u_ip12n[8] * tx * ty + u_ip12n[7] * tx * tz + u_ip12n[4] * ty * ty + u_ip12n[6] * ty * tz + u_ip12n[5] * tz * tz;
    u_ip12n_T[6] = u_ip12n[8] * (sx * ty + sy * tx) + u_ip12n[7] * (sx * tz + sz * tx) + u_ip12n[6] * (sy * tz + sz * ty) + 2 * sx * tx * u_ip12n[3] + 2 * sy * ty * u_ip12n[4] + 2 * sz * tz * u_ip12n[5];
    u_ip12n_T[7] = u_ip12n[8] * (nx * ty + ny * tx) + u_ip12n[7] * (nx * tz + nz * tx) + u_ip12n[6] * (ny * tz + nz * ty) + 2 * nx * tx * u_ip12n[3] + 2 * ny * ty * u_ip12n[4] + 2 * nz * tz * u_ip12n[5];
    u_ip12n_T[8] = u_ip12n[8] * (nx * sy + ny * sx) + u_ip12n[7] * (nx * sz + nz * sx) + u_ip12n[6] * (ny * sz + nz * sy) + 2 * nx * sx * u_ip12n[3] + 2 * ny * sy * u_ip12n[4] + 2 * nz * sz * u_ip12n[5];
    // u_ip12n_T[0] = (ny * sz * u_ip12n[2] - nz * sy * u_ip12n[2] - ny * tz * u_ip12n[1] + nz * ty * u_ip12n[1] + sy * tz * u_ip12n[0] - sz * ty * u_ip12n[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    // u_ip12n_T[1] = -(nx * sz * u_ip12n[2] - nz * sx * u_ip12n[2] - nx * tz * u_ip12n[1] + nz * tx * u_ip12n[1] + sx * tz * u_ip12n[0] - sz * tx * u_ip12n[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    // u_ip12n_T[2] = (nx * sy * u_ip12n[2] - ny * sx * u_ip12n[2] - nx * ty * u_ip12n[1] + ny * tx * u_ip12n[1] + sx * ty * u_ip12n[0] - sy * tx * u_ip12n[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    // u_ip12n_T[3] = (u_ip12n[5] * (ny * ny) * (sz * sz) - u_ip12n[6] * (ny * ny) * sz * tz + u_ip12n[4] * (ny * ny) * (tz * tz) - 2 * u_ip12n[5] * ny * nz * sy * sz + u_ip12n[6] * ny * nz * sy * tz + u_ip12n[6] * ny * nz * sz * ty - 2 * u_ip12n[4] * ny * nz * ty * tz + u_ip12n[7] * ny * sy * sz * tz - u_ip12n[8] * ny * sy * (tz * tz) - u_ip12n[7] * ny * (sz * sz) * ty + u_ip12n[8] * ny * sz * ty * tz + u_ip12n[5] * (nz * nz) * (sy * sy) - u_ip12n[6] * (nz * nz) * sy * ty + u_ip12n[4] * (nz * nz) * (ty * ty) - u_ip12n[7] * nz * (sy * sy) * tz + u_ip12n[7] * nz * sy * sz * ty + u_ip12n[8] * nz * sy * ty * tz - u_ip12n[8] * nz * sz * (ty * ty) + u_ip12n[3] * (sy * sy) * (tz * tz) - 2 * u_ip12n[3] * sy * sz * ty * tz + u_ip12n[3] * (sz * sz) * (ty * ty)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12n_T[4] = (u_ip12n[5] * (nx * nx) * (sz * sz) - u_ip12n[6] * (nx * nx) * sz * tz + u_ip12n[4] * (nx * nx) * (tz * tz) - 2 * u_ip12n[5] * nx * nz * sx * sz + u_ip12n[6] * nx * nz * sx * tz + u_ip12n[6] * nx * nz * sz * tx - 2 * u_ip12n[4] * nx * nz * tx * tz + u_ip12n[7] * nx * sx * sz * tz - u_ip12n[8] * nx * sx * (tz * tz) - u_ip12n[7] * nx * (sz * sz) * tx + u_ip12n[8] * nx * sz * tx * tz + u_ip12n[5] * (nz * nz) * (sx * sx) - u_ip12n[6] * (nz * nz) * sx * tx + u_ip12n[4] * (nz * nz) * (tx * tx) - u_ip12n[7] * nz * (sx * sx) * tz + u_ip12n[7] * nz * sx * sz * tx + u_ip12n[8] * nz * sx * tx * tz - u_ip12n[8] * nz * sz * (tx * tx) + u_ip12n[3] * (sx * sx) * (tz * tz) - 2 * u_ip12n[3] * sx * sz * tx * tz + u_ip12n[3] * (sz * sz) * (tx * tx)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12n_T[5] = (u_ip12n[5] * (nx * nx) * (sy * sy) - u_ip12n[6] * (nx * nx) * sy * ty + u_ip12n[4] * (nx * nx) * (ty * ty) - 2 * u_ip12n[5] * nx * ny * sx * sy + u_ip12n[6] * nx * ny * sx * ty + u_ip12n[6] * nx * ny * sy * tx - 2 * u_ip12n[4] * nx * ny * tx * ty + u_ip12n[7] * nx * sx * sy * ty - u_ip12n[8] * nx * sx * (ty * ty) - u_ip12n[7] * nx * (sy * sy) * tx + u_ip12n[8] * nx * sy * tx * ty + u_ip12n[5] * (ny * ny) * (sx * sx) - u_ip12n[6] * (ny * ny) * sx * tx + u_ip12n[4] * (ny * ny) * (tx * tx) - u_ip12n[7] * ny * (sx * sx) * ty + u_ip12n[7] * ny * sx * sy * tx + u_ip12n[8] * ny * sx * tx * ty - u_ip12n[8] * ny * sy * (tx * tx) + u_ip12n[3] * (sx * sx) * (ty * ty) - 2 * u_ip12n[3] * sx * sy * tx * ty + u_ip12n[3] * (sy * sy) * (tx * tx)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12n_T[6] = -(2 * ny * nz * (sx * sx) * u_ip12n[5] + 2 * ny * nz * (tx * tx) * u_ip12n[4] + 2 * (nx * nx) * sy * sz * u_ip12n[5] - (nx * nx) * sy * tz * u_ip12n[6] - (nx * nx) * sz * ty * u_ip12n[6] - ny * (sx * sx) * tz * u_ip12n[7] - nz * (sx * sx) * ty * u_ip12n[7] - ny * sz * (tx * tx) * u_ip12n[8] - nz * sy * (tx * tx) * u_ip12n[8] + 2 * (nx * nx) * ty * tz * u_ip12n[4] + 2 * sy * sz * (tx * tx) * u_ip12n[3] + 2 * (sx * sx) * ty * tz * u_ip12n[3] - 2 * nx * ny * sx * sz * u_ip12n[5] - 2 * nx * nz * sx * sy * u_ip12n[5] + nx * ny * sx * tz * u_ip12n[6] + nx * ny * sz * tx * u_ip12n[6] + nx * nz * sx * ty * u_ip12n[6] + nx * nz * sy * tx * u_ip12n[6] - 2 * ny * nz * sx * tx * u_ip12n[6] - 2 * nx * ny * tx * tz * u_ip12n[4] - 2 * nx * nz * tx * ty * u_ip12n[4] + nx * sx * sy * tz * u_ip12n[7] + nx * sx * sz * ty * u_ip12n[7] - 2 * nx * sy * sz * tx * u_ip12n[7] + ny * sx * sz * tx * u_ip12n[7] + nz * sx * sy * tx * u_ip12n[7] - 2 * nx * sx * ty * tz * u_ip12n[8] + nx * sy * tx * tz * u_ip12n[8] + nx * sz * tx * ty * u_ip12n[8] + ny * sx * tx * tz * u_ip12n[8] + nz * sx * tx * ty * u_ip12n[8] - 2 * sx * sy * tx * tz * u_ip12n[3] - 2 * sx * sz * tx * ty * u_ip12n[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12n_T[7] = -(2 * nx * nz * (sy * sy) * u_ip12n[5] + 2 * nx * nz * (ty * ty) * u_ip12n[4] + 2 * (ny * ny) * sx * sz * u_ip12n[5] - (ny * ny) * sx * tz * u_ip12n[6] - (ny * ny) * sz * tx * u_ip12n[6] - nx * (sy * sy) * tz * u_ip12n[7] - nz * (sy * sy) * tx * u_ip12n[7] - nx * sz * (ty * ty) * u_ip12n[8] - nz * sx * (ty * ty) * u_ip12n[8] + 2 * (ny * ny) * tx * tz * u_ip12n[4] + 2 * sx * sz * (ty * ty) * u_ip12n[3] + 2 * (sy * sy) * tx * tz * u_ip12n[3] - 2 * nx * ny * sy * sz * u_ip12n[5] - 2 * ny * nz * sx * sy * u_ip12n[5] + nx * ny * sy * tz * u_ip12n[6] + nx * ny * sz * ty * u_ip12n[6] - 2 * nx * nz * sy * ty * u_ip12n[6] + ny * nz * sx * ty * u_ip12n[6] + ny * nz * sy * tx * u_ip12n[6] - 2 * nx * ny * ty * tz * u_ip12n[4] - 2 * ny * nz * tx * ty * u_ip12n[4] + nx * sy * sz * ty * u_ip12n[7] + ny * sx * sy * tz * u_ip12n[7] - 2 * ny * sx * sz * ty * u_ip12n[7] + ny * sy * sz * tx * u_ip12n[7] + nz * sx * sy * ty * u_ip12n[7] + nx * sy * ty * tz * u_ip12n[8] + ny * sx * ty * tz * u_ip12n[8] - 2 * ny * sy * tx * tz * u_ip12n[8] + ny * sz * tx * ty * u_ip12n[8] + nz * sy * tx * ty * u_ip12n[8] - 2 * sx * sy * ty * tz * u_ip12n[3] - 2 * sy * sz * tx * ty * u_ip12n[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12n_T[8] = -(2 * nx * ny * (sz * sz) * u_ip12n[5] + 2 * nx * ny * (tz * tz) * u_ip12n[4] + 2 * (nz * nz) * sx * sy * u_ip12n[5] - (nz * nz) * sx * ty * u_ip12n[6] - (nz * nz) * sy * tx * u_ip12n[6] - nx * (sz * sz) * ty * u_ip12n[7] - ny * (sz * sz) * tx * u_ip12n[7] - nx * sy * (tz * tz) * u_ip12n[8] - ny * sx * (tz * tz) * u_ip12n[8] + 2 * (nz * nz) * tx * ty * u_ip12n[4] + 2 * sx * sy * (tz * tz) * u_ip12n[3] + 2 * (sz * sz) * tx * ty * u_ip12n[3] - 2 * nx * nz * sy * sz * u_ip12n[5] - 2 * ny * nz * sx * sz * u_ip12n[5] - 2 * nx * ny * sz * tz * u_ip12n[6] + nx * nz * sy * tz * u_ip12n[6] + nx * nz * sz * ty * u_ip12n[6] + ny * nz * sx * tz * u_ip12n[6] + ny * nz * sz * tx * u_ip12n[6] - 2 * nx * nz * ty * tz * u_ip12n[4] - 2 * ny * nz * tx * tz * u_ip12n[4] + nx * sy * sz * tz * u_ip12n[7] + ny * sx * sz * tz * u_ip12n[7] - 2 * nz * sx * sy * tz * u_ip12n[7] + nz * sx * sz * ty * u_ip12n[7] + nz * sy * sz * tx * u_ip12n[7] + nx * sz * ty * tz * u_ip12n[8] + ny * sz * tx * tz * u_ip12n[8] + nz * sx * ty * tz * u_ip12n[8] + nz * sy * tx * tz * u_ip12n[8] - 2 * nz * sz * tx * ty * u_ip12n[8] - 2 * sx * sz * ty * tz * u_ip12n[3] - 2 * sy * sz * tx * tz * u_ip12n[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));

    u_ip12p_T[0] = nx * u_ip12p[0] + ny * u_ip12p[1] + nz * u_ip12p[2];
    u_ip12p_T[1] = sx * u_ip12p[0] + sy * u_ip12p[1] + sz * u_ip12p[2];
    u_ip12p_T[2] = tx * u_ip12p[0] + ty * u_ip12p[1] + tz * u_ip12p[2];
    u_ip12p_T[3] = u_ip12p[3] * nx * nx + u_ip12p[8] * nx * ny + u_ip12p[7] * nx * nz + u_ip12p[4] * ny * ny + u_ip12p[6] * ny * nz + u_ip12p[5] * nz * nz;
    u_ip12p_T[4] = u_ip12p[3] * sx * sx + u_ip12p[8] * sx * sy + u_ip12p[7] * sx * sz + u_ip12p[4] * sy * sy + u_ip12p[6] * sy * sz + u_ip12p[5] * sz * sz;
    u_ip12p_T[5] = u_ip12p[3] * tx * tx + u_ip12p[8] * tx * ty + u_ip12p[7] * tx * tz + u_ip12p[4] * ty * ty + u_ip12p[6] * ty * tz + u_ip12p[5] * tz * tz;
    u_ip12p_T[6] = u_ip12p[8] * (sx * ty + sy * tx) + u_ip12p[7] * (sx * tz + sz * tx) + u_ip12p[6] * (sy * tz + sz * ty) + 2 * sx * tx * u_ip12p[3] + 2 * sy * ty * u_ip12p[4] + 2 * sz * tz * u_ip12p[5];
    u_ip12p_T[7] = u_ip12p[8] * (nx * ty + ny * tx) + u_ip12p[7] * (nx * tz + nz * tx) + u_ip12p[6] * (ny * tz + nz * ty) + 2 * nx * tx * u_ip12p[3] + 2 * ny * ty * u_ip12p[4] + 2 * nz * tz * u_ip12p[5];
    u_ip12p_T[8] = u_ip12p[8] * (nx * sy + ny * sx) + u_ip12p[7] * (nx * sz + nz * sx) + u_ip12p[6] * (ny * sz + nz * sy) + 2 * nx * sx * u_ip12p[3] + 2 * ny * sy * u_ip12p[4] + 2 * nz * sz * u_ip12p[5];
    // u_ip12p_T[0] = (ny * sz * u_ip12p[2] - nz * sy * u_ip12p[2] - ny * tz * u_ip12p[1] + nz * ty * u_ip12p[1] + sy * tz * u_ip12p[0] - sz * ty * u_ip12p[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    // u_ip12p_T[1] = -(nx * sz * u_ip12p[2] - nz * sx * u_ip12p[2] - nx * tz * u_ip12p[1] + nz * tx * u_ip12p[1] + sx * tz * u_ip12p[0] - sz * tx * u_ip12p[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    // u_ip12p_T[2] = (nx * sy * u_ip12p[2] - ny * sx * u_ip12p[2] - nx * ty * u_ip12p[1] + ny * tx * u_ip12p[1] + sx * ty * u_ip12p[0] - sy * tx * u_ip12p[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    // u_ip12p_T[3] = (u_ip12p[5] * (ny * ny) * (sz * sz) - u_ip12p[6] * (ny * ny) * sz * tz + u_ip12p[4] * (ny * ny) * (tz * tz) - 2 * u_ip12p[5] * ny * nz * sy * sz + u_ip12p[6] * ny * nz * sy * tz + u_ip12p[6] * ny * nz * sz * ty - 2 * u_ip12p[4] * ny * nz * ty * tz + u_ip12p[7] * ny * sy * sz * tz - u_ip12p[8] * ny * sy * (tz * tz) - u_ip12p[7] * ny * (sz * sz) * ty + u_ip12p[8] * ny * sz * ty * tz + u_ip12p[5] * (nz * nz) * (sy * sy) - u_ip12p[6] * (nz * nz) * sy * ty + u_ip12p[4] * (nz * nz) * (ty * ty) - u_ip12p[7] * nz * (sy * sy) * tz + u_ip12p[7] * nz * sy * sz * ty + u_ip12p[8] * nz * sy * ty * tz - u_ip12p[8] * nz * sz * (ty * ty) + u_ip12p[3] * (sy * sy) * (tz * tz) - 2 * u_ip12p[3] * sy * sz * ty * tz + u_ip12p[3] * (sz * sz) * (ty * ty)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12p_T[4] = (u_ip12p[5] * (nx * nx) * (sz * sz) - u_ip12p[6] * (nx * nx) * sz * tz + u_ip12p[4] * (nx * nx) * (tz * tz) - 2 * u_ip12p[5] * nx * nz * sx * sz + u_ip12p[6] * nx * nz * sx * tz + u_ip12p[6] * nx * nz * sz * tx - 2 * u_ip12p[4] * nx * nz * tx * tz + u_ip12p[7] * nx * sx * sz * tz - u_ip12p[8] * nx * sx * (tz * tz) - u_ip12p[7] * nx * (sz * sz) * tx + u_ip12p[8] * nx * sz * tx * tz + u_ip12p[5] * (nz * nz) * (sx * sx) - u_ip12p[6] * (nz * nz) * sx * tx + u_ip12p[4] * (nz * nz) * (tx * tx) - u_ip12p[7] * nz * (sx * sx) * tz + u_ip12p[7] * nz * sx * sz * tx + u_ip12p[8] * nz * sx * tx * tz - u_ip12p[8] * nz * sz * (tx * tx) + u_ip12p[3] * (sx * sx) * (tz * tz) - 2 * u_ip12p[3] * sx * sz * tx * tz + u_ip12p[3] * (sz * sz) * (tx * tx)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12p_T[5] = (u_ip12p[5] * (nx * nx) * (sy * sy) - u_ip12p[6] * (nx * nx) * sy * ty + u_ip12p[4] * (nx * nx) * (ty * ty) - 2 * u_ip12p[5] * nx * ny * sx * sy + u_ip12p[6] * nx * ny * sx * ty + u_ip12p[6] * nx * ny * sy * tx - 2 * u_ip12p[4] * nx * ny * tx * ty + u_ip12p[7] * nx * sx * sy * ty - u_ip12p[8] * nx * sx * (ty * ty) - u_ip12p[7] * nx * (sy * sy) * tx + u_ip12p[8] * nx * sy * tx * ty + u_ip12p[5] * (ny * ny) * (sx * sx) - u_ip12p[6] * (ny * ny) * sx * tx + u_ip12p[4] * (ny * ny) * (tx * tx) - u_ip12p[7] * ny * (sx * sx) * ty + u_ip12p[7] * ny * sx * sy * tx + u_ip12p[8] * ny * sx * tx * ty - u_ip12p[8] * ny * sy * (tx * tx) + u_ip12p[3] * (sx * sx) * (ty * ty) - 2 * u_ip12p[3] * sx * sy * tx * ty + u_ip12p[3] * (sy * sy) * (tx * tx)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12p_T[6] = -(2 * ny * nz * (sx * sx) * u_ip12p[5] + 2 * ny * nz * (tx * tx) * u_ip12p[4] + 2 * (nx * nx) * sy * sz * u_ip12p[5] - (nx * nx) * sy * tz * u_ip12p[6] - (nx * nx) * sz * ty * u_ip12p[6] - ny * (sx * sx) * tz * u_ip12p[7] - nz * (sx * sx) * ty * u_ip12p[7] - ny * sz * (tx * tx) * u_ip12p[8] - nz * sy * (tx * tx) * u_ip12p[8] + 2 * (nx * nx) * ty * tz * u_ip12p[4] + 2 * sy * sz * (tx * tx) * u_ip12p[3] + 2 * (sx * sx) * ty * tz * u_ip12p[3] - 2 * nx * ny * sx * sz * u_ip12p[5] - 2 * nx * nz * sx * sy * u_ip12p[5] + nx * ny * sx * tz * u_ip12p[6] + nx * ny * sz * tx * u_ip12p[6] + nx * nz * sx * ty * u_ip12p[6] + nx * nz * sy * tx * u_ip12p[6] - 2 * ny * nz * sx * tx * u_ip12p[6] - 2 * nx * ny * tx * tz * u_ip12p[4] - 2 * nx * nz * tx * ty * u_ip12p[4] + nx * sx * sy * tz * u_ip12p[7] + nx * sx * sz * ty * u_ip12p[7] - 2 * nx * sy * sz * tx * u_ip12p[7] + ny * sx * sz * tx * u_ip12p[7] + nz * sx * sy * tx * u_ip12p[7] - 2 * nx * sx * ty * tz * u_ip12p[8] + nx * sy * tx * tz * u_ip12p[8] + nx * sz * tx * ty * u_ip12p[8] + ny * sx * tx * tz * u_ip12p[8] + nz * sx * tx * ty * u_ip12p[8] - 2 * sx * sy * tx * tz * u_ip12p[3] - 2 * sx * sz * tx * ty * u_ip12p[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12p_T[7] = -(2 * nx * nz * (sy * sy) * u_ip12p[5] + 2 * nx * nz * (ty * ty) * u_ip12p[4] + 2 * (ny * ny) * sx * sz * u_ip12p[5] - (ny * ny) * sx * tz * u_ip12p[6] - (ny * ny) * sz * tx * u_ip12p[6] - nx * (sy * sy) * tz * u_ip12p[7] - nz * (sy * sy) * tx * u_ip12p[7] - nx * sz * (ty * ty) * u_ip12p[8] - nz * sx * (ty * ty) * u_ip12p[8] + 2 * (ny * ny) * tx * tz * u_ip12p[4] + 2 * sx * sz * (ty * ty) * u_ip12p[3] + 2 * (sy * sy) * tx * tz * u_ip12p[3] - 2 * nx * ny * sy * sz * u_ip12p[5] - 2 * ny * nz * sx * sy * u_ip12p[5] + nx * ny * sy * tz * u_ip12p[6] + nx * ny * sz * ty * u_ip12p[6] - 2 * nx * nz * sy * ty * u_ip12p[6] + ny * nz * sx * ty * u_ip12p[6] + ny * nz * sy * tx * u_ip12p[6] - 2 * nx * ny * ty * tz * u_ip12p[4] - 2 * ny * nz * tx * ty * u_ip12p[4] + nx * sy * sz * ty * u_ip12p[7] + ny * sx * sy * tz * u_ip12p[7] - 2 * ny * sx * sz * ty * u_ip12p[7] + ny * sy * sz * tx * u_ip12p[7] + nz * sx * sy * ty * u_ip12p[7] + nx * sy * ty * tz * u_ip12p[8] + ny * sx * ty * tz * u_ip12p[8] - 2 * ny * sy * tx * tz * u_ip12p[8] + ny * sz * tx * ty * u_ip12p[8] + nz * sy * tx * ty * u_ip12p[8] - 2 * sx * sy * ty * tz * u_ip12p[3] - 2 * sy * sz * tx * ty * u_ip12p[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // u_ip12p_T[8] = -(2 * nx * ny * (sz * sz) * u_ip12p[5] + 2 * nx * ny * (tz * tz) * u_ip12p[4] + 2 * (nz * nz) * sx * sy * u_ip12p[5] - (nz * nz) * sx * ty * u_ip12p[6] - (nz * nz) * sy * tx * u_ip12p[6] - nx * (sz * sz) * ty * u_ip12p[7] - ny * (sz * sz) * tx * u_ip12p[7] - nx * sy * (tz * tz) * u_ip12p[8] - ny * sx * (tz * tz) * u_ip12p[8] + 2 * (nz * nz) * tx * ty * u_ip12p[4] + 2 * sx * sy * (tz * tz) * u_ip12p[3] + 2 * (sz * sz) * tx * ty * u_ip12p[3] - 2 * nx * nz * sy * sz * u_ip12p[5] - 2 * ny * nz * sx * sz * u_ip12p[5] - 2 * nx * ny * sz * tz * u_ip12p[6] + nx * nz * sy * tz * u_ip12p[6] + nx * nz * sz * ty * u_ip12p[6] + ny * nz * sx * tz * u_ip12p[6] + ny * nz * sz * tx * u_ip12p[6] - 2 * nx * nz * ty * tz * u_ip12p[4] - 2 * ny * nz * tx * tz * u_ip12p[4] + nx * sy * sz * tz * u_ip12p[7] + ny * sx * sz * tz * u_ip12p[7] - 2 * nz * sx * sy * tz * u_ip12p[7] + nz * sx * sz * ty * u_ip12p[7] + nz * sy * sz * tx * u_ip12p[7] + nx * sz * ty * tz * u_ip12p[8] + ny * sz * tx * tz * u_ip12p[8] + nz * sx * ty * tz * u_ip12p[8] + nz * sy * tx * tz * u_ip12p[8] - 2 * nz * sz * tx * ty * u_ip12p[8] - 2 * sx * sz * ty * tz * u_ip12p[3] - 2 * sy * sz * tx * tz * u_ip12p[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));

    // Calculate physical variables
    fu_ip12n[0] = (lambda_n * u_ip12n_T[4] + lambda_n * u_ip12n_T[5] + u_ip12n_T[3] * (lambda_n + 2 * mu_n));
    fu_ip12n[1] = (mu_n * u_ip12n_T[8]);
    fu_ip12n[2] = (mu_n * u_ip12n_T[7]);
    fu_ip12n[3] = (u_ip12n_T[0] / rho_n);
    fu_ip12n[4] = (0);
    fu_ip12n[5] = (0);
    fu_ip12n[6] = (0);
    fu_ip12n[7] = (u_ip12n_T[2] / rho_n);
    fu_ip12n[8] = (u_ip12n_T[1] / rho_n);

    fu_ip12p[0] = (lambda_p * u_ip12p_T[4] + lambda_p * u_ip12p_T[5] + u_ip12p_T[3] * (lambda_p + 2 * mu_p));
    fu_ip12p[1] = (mu_p * u_ip12p_T[8]);
    fu_ip12p[2] = (mu_p * u_ip12p_T[7]);
    fu_ip12p[3] = (u_ip12p_T[0] / rho_p);
    fu_ip12p[4] = (0);
    fu_ip12p[5] = (0);
    fu_ip12p[6] = (0);
    fu_ip12p[7] = (u_ip12p_T[2] / rho_p);
    fu_ip12p[8] = (u_ip12p_T[1] / rho_p);

    // Calculate Riemann flux
    tau_xx = -(fu_ip12p[0] - fu_ip12n[0]);
    tau_xy = -(fu_ip12p[1] - fu_ip12n[1]);
    tau_xz = -(fu_ip12p[2] - fu_ip12n[2]);
    v_x = -(fu_ip12p[3] - fu_ip12n[3]);
    v_y = -(fu_ip12p[8] - fu_ip12n[8]);
    v_z = -(fu_ip12p[7] - fu_ip12n[7]);

    Riemann_flux_T[0] = zp_n * (tau_xx + zp_p * v_x) / (zp_n + zp_p);
    Riemann_flux_T[1] = zs_n * (tau_xy + zs_p * v_y) / (zs_n + zs_p);
    Riemann_flux_T[2] = zs_n * (tau_xz + zs_p * v_z) / (zs_n + zs_p);
    Riemann_flux_T[3] = (tau_xx + zp_p * v_x) / (zp_n + zp_p);
    Riemann_flux_T[4] = 0;
    Riemann_flux_T[5] = 0;
    Riemann_flux_T[6] = 0;
    Riemann_flux_T[7] = (tau_xz + zs_p * v_z) / (zs_n + zs_p);
    Riemann_flux_T[8] = (tau_xy + zs_p * v_y) / (zs_n + zs_p);

    // Rotate back
    Riemann_flux[0] = (ny * sz * Riemann_flux_T[2] - nz * sy * Riemann_flux_T[2] - ny * tz * Riemann_flux_T[1] + nz * ty * Riemann_flux_T[1] + sy * tz * Riemann_flux_T[0] - sz * ty * Riemann_flux_T[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    Riemann_flux[1] = -(nx * sz * Riemann_flux_T[2] - nz * sx * Riemann_flux_T[2] - nx * tz * Riemann_flux_T[1] + nz * tx * Riemann_flux_T[1] + sx * tz * Riemann_flux_T[0] - sz * tx * Riemann_flux_T[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    Riemann_flux[2] = (nx * sy * Riemann_flux_T[2] - ny * sx * Riemann_flux_T[2] - nx * ty * Riemann_flux_T[1] + ny * tx * Riemann_flux_T[1] + sx * ty * Riemann_flux_T[0] - sy * tx * Riemann_flux_T[0]) / (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx);
    Riemann_flux[3] = (Riemann_flux_T[5] * (ny * ny) * (sz * sz) - Riemann_flux_T[6] * (ny * ny) * sz * tz + Riemann_flux_T[4] * (ny * ny) * (tz * tz) - 2 * Riemann_flux_T[5] * ny * nz * sy * sz + Riemann_flux_T[6] * ny * nz * sy * tz + Riemann_flux_T[6] * ny * nz * sz * ty - 2 * Riemann_flux_T[4] * ny * nz * ty * tz + Riemann_flux_T[7] * ny * sy * sz * tz - Riemann_flux_T[8] * ny * sy * (tz * tz) - Riemann_flux_T[7] * ny * (sz * sz) * ty + Riemann_flux_T[8] * ny * sz * ty * tz + Riemann_flux_T[5] * (nz * nz) * (sy * sy) - Riemann_flux_T[6] * (nz * nz) * sy * ty + Riemann_flux_T[4] * (nz * nz) * (ty * ty) - Riemann_flux_T[7] * nz * (sy * sy) * tz + Riemann_flux_T[7] * nz * sy * sz * ty + Riemann_flux_T[8] * nz * sy * ty * tz - Riemann_flux_T[8] * nz * sz * (ty * ty) + Riemann_flux_T[3] * (sy * sy) * (tz * tz) - 2 * Riemann_flux_T[3] * sy * sz * ty * tz + Riemann_flux_T[3] * (sz * sz) * (ty * ty)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    Riemann_flux[4] = (Riemann_flux_T[5] * (nx * nx) * (sz * sz) - Riemann_flux_T[6] * (nx * nx) * sz * tz + Riemann_flux_T[4] * (nx * nx) * (tz * tz) - 2 * Riemann_flux_T[5] * nx * nz * sx * sz + Riemann_flux_T[6] * nx * nz * sx * tz + Riemann_flux_T[6] * nx * nz * sz * tx - 2 * Riemann_flux_T[4] * nx * nz * tx * tz + Riemann_flux_T[7] * nx * sx * sz * tz - Riemann_flux_T[8] * nx * sx * (tz * tz) - Riemann_flux_T[7] * nx * (sz * sz) * tx + Riemann_flux_T[8] * nx * sz * tx * tz + Riemann_flux_T[5] * (nz * nz) * (sx * sx) - Riemann_flux_T[6] * (nz * nz) * sx * tx + Riemann_flux_T[4] * (nz * nz) * (tx * tx) - Riemann_flux_T[7] * nz * (sx * sx) * tz + Riemann_flux_T[7] * nz * sx * sz * tx + Riemann_flux_T[8] * nz * sx * tx * tz - Riemann_flux_T[8] * nz * sz * (tx * tx) + Riemann_flux_T[3] * (sx * sx) * (tz * tz) - 2 * Riemann_flux_T[3] * sx * sz * tx * tz + Riemann_flux_T[3] * (sz * sz) * (tx * tx)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    Riemann_flux[5] = (Riemann_flux_T[5] * (nx * nx) * (sy * sy) - Riemann_flux_T[6] * (nx * nx) * sy * ty + Riemann_flux_T[4] * (nx * nx) * (ty * ty) - 2 * Riemann_flux_T[5] * nx * ny * sx * sy + Riemann_flux_T[6] * nx * ny * sx * ty + Riemann_flux_T[6] * nx * ny * sy * tx - 2 * Riemann_flux_T[4] * nx * ny * tx * ty + Riemann_flux_T[7] * nx * sx * sy * ty - Riemann_flux_T[8] * nx * sx * (ty * ty) - Riemann_flux_T[7] * nx * (sy * sy) * tx + Riemann_flux_T[8] * nx * sy * tx * ty + Riemann_flux_T[5] * (ny * ny) * (sx * sx) - Riemann_flux_T[6] * (ny * ny) * sx * tx + Riemann_flux_T[4] * (ny * ny) * (tx * tx) - Riemann_flux_T[7] * ny * (sx * sx) * ty + Riemann_flux_T[7] * ny * sx * sy * tx + Riemann_flux_T[8] * ny * sx * tx * ty - Riemann_flux_T[8] * ny * sy * (tx * tx) + Riemann_flux_T[3] * (sx * sx) * (ty * ty) - 2 * Riemann_flux_T[3] * sx * sy * tx * ty + Riemann_flux_T[3] * (sy * sy) * (tx * tx)) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    Riemann_flux[6] = -(2 * ny * nz * (sx * sx) * Riemann_flux_T[5] + 2 * ny * nz * (tx * tx) * Riemann_flux_T[4] + 2 * (nx * nx) * sy * sz * Riemann_flux_T[5] - (nx * nx) * sy * tz * Riemann_flux_T[6] - (nx * nx) * sz * ty * Riemann_flux_T[6] - ny * (sx * sx) * tz * Riemann_flux_T[7] - nz * (sx * sx) * ty * Riemann_flux_T[7] - ny * sz * (tx * tx) * Riemann_flux_T[8] - nz * sy * (tx * tx) * Riemann_flux_T[8] + 2 * (nx * nx) * ty * tz * Riemann_flux_T[4] + 2 * sy * sz * (tx * tx) * Riemann_flux_T[3] + 2 * (sx * sx) * ty * tz * Riemann_flux_T[3] - 2 * nx * ny * sx * sz * Riemann_flux_T[5] - 2 * nx * nz * sx * sy * Riemann_flux_T[5] + nx * ny * sx * tz * Riemann_flux_T[6] + nx * ny * sz * tx * Riemann_flux_T[6] + nx * nz * sx * ty * Riemann_flux_T[6] + nx * nz * sy * tx * Riemann_flux_T[6] - 2 * ny * nz * sx * tx * Riemann_flux_T[6] - 2 * nx * ny * tx * tz * Riemann_flux_T[4] - 2 * nx * nz * tx * ty * Riemann_flux_T[4] + nx * sx * sy * tz * Riemann_flux_T[7] + nx * sx * sz * ty * Riemann_flux_T[7] - 2 * nx * sy * sz * tx * Riemann_flux_T[7] + ny * sx * sz * tx * Riemann_flux_T[7] + nz * sx * sy * tx * Riemann_flux_T[7] - 2 * nx * sx * ty * tz * Riemann_flux_T[8] + nx * sy * tx * tz * Riemann_flux_T[8] + nx * sz * tx * ty * Riemann_flux_T[8] + ny * sx * tx * tz * Riemann_flux_T[8] + nz * sx * tx * ty * Riemann_flux_T[8] - 2 * sx * sy * tx * tz * Riemann_flux_T[3] - 2 * sx * sz * tx * ty * Riemann_flux_T[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    Riemann_flux[7] = -(2 * nx * nz * (sy * sy) * Riemann_flux_T[5] + 2 * nx * nz * (ty * ty) * Riemann_flux_T[4] + 2 * (ny * ny) * sx * sz * Riemann_flux_T[5] - (ny * ny) * sx * tz * Riemann_flux_T[6] - (ny * ny) * sz * tx * Riemann_flux_T[6] - nx * (sy * sy) * tz * Riemann_flux_T[7] - nz * (sy * sy) * tx * Riemann_flux_T[7] - nx * sz * (ty * ty) * Riemann_flux_T[8] - nz * sx * (ty * ty) * Riemann_flux_T[8] + 2 * (ny * ny) * tx * tz * Riemann_flux_T[4] + 2 * sx * sz * (ty * ty) * Riemann_flux_T[3] + 2 * (sy * sy) * tx * tz * Riemann_flux_T[3] - 2 * nx * ny * sy * sz * Riemann_flux_T[5] - 2 * ny * nz * sx * sy * Riemann_flux_T[5] + nx * ny * sy * tz * Riemann_flux_T[6] + nx * ny * sz * ty * Riemann_flux_T[6] - 2 * nx * nz * sy * ty * Riemann_flux_T[6] + ny * nz * sx * ty * Riemann_flux_T[6] + ny * nz * sy * tx * Riemann_flux_T[6] - 2 * nx * ny * ty * tz * Riemann_flux_T[4] - 2 * ny * nz * tx * ty * Riemann_flux_T[4] + nx * sy * sz * ty * Riemann_flux_T[7] + ny * sx * sy * tz * Riemann_flux_T[7] - 2 * ny * sx * sz * ty * Riemann_flux_T[7] + ny * sy * sz * tx * Riemann_flux_T[7] + nz * sx * sy * ty * Riemann_flux_T[7] + nx * sy * ty * tz * Riemann_flux_T[8] + ny * sx * ty * tz * Riemann_flux_T[8] - 2 * ny * sy * tx * tz * Riemann_flux_T[8] + ny * sz * tx * ty * Riemann_flux_T[8] + nz * sy * tx * ty * Riemann_flux_T[8] - 2 * sx * sy * ty * tz * Riemann_flux_T[3] - 2 * sy * sz * tx * ty * Riemann_flux_T[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    Riemann_flux[8] = -(2 * nx * ny * (sz * sz) * Riemann_flux_T[5] + 2 * nx * ny * (tz * tz) * Riemann_flux_T[4] + 2 * (nz * nz) * sx * sy * Riemann_flux_T[5] - (nz * nz) * sx * ty * Riemann_flux_T[6] - (nz * nz) * sy * tx * Riemann_flux_T[6] - nx * (sz * sz) * ty * Riemann_flux_T[7] - ny * (sz * sz) * tx * Riemann_flux_T[7] - nx * sy * (tz * tz) * Riemann_flux_T[8] - ny * sx * (tz * tz) * Riemann_flux_T[8] + 2 * (nz * nz) * tx * ty * Riemann_flux_T[4] + 2 * sx * sy * (tz * tz) * Riemann_flux_T[3] + 2 * (sz * sz) * tx * ty * Riemann_flux_T[3] - 2 * nx * nz * sy * sz * Riemann_flux_T[5] - 2 * ny * nz * sx * sz * Riemann_flux_T[5] - 2 * nx * ny * sz * tz * Riemann_flux_T[6] + nx * nz * sy * tz * Riemann_flux_T[6] + nx * nz * sz * ty * Riemann_flux_T[6] + ny * nz * sx * tz * Riemann_flux_T[6] + ny * nz * sz * tx * Riemann_flux_T[6] - 2 * nx * nz * ty * tz * Riemann_flux_T[4] - 2 * ny * nz * tx * tz * Riemann_flux_T[4] + nx * sy * sz * tz * Riemann_flux_T[7] + ny * sx * sz * tz * Riemann_flux_T[7] - 2 * nz * sx * sy * tz * Riemann_flux_T[7] + nz * sx * sz * ty * Riemann_flux_T[7] + nz * sy * sz * tx * Riemann_flux_T[7] + nx * sz * ty * tz * Riemann_flux_T[8] + ny * sz * tx * tz * Riemann_flux_T[8] + nz * sx * ty * tz * Riemann_flux_T[8] + nz * sy * tx * tz * Riemann_flux_T[8] - 2 * nz * sz * tx * ty * Riemann_flux_T[8] - 2 * sx * sz * ty * tz * Riemann_flux_T[3] - 2 * sy * sz * tx * tz * Riemann_flux_T[3]) / ((nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx) * (nx * sy * tz - nx * sz * ty - ny * sx * tz + ny * sz * tx + nz * sx * ty - nz * sy * tx));
    // Riemann_flux[0] = nx * Riemann_flux_T[0] + ny * Riemann_flux_T[1] + nz * Riemann_flux_T[2];
    // Riemann_flux[1] = sx * Riemann_flux_T[0] + sy * Riemann_flux_T[1] + sz * Riemann_flux_T[2];
    // Riemann_flux[2] = tx * Riemann_flux_T[0] + ty * Riemann_flux_T[1] + tz * Riemann_flux_T[2];
    // Riemann_flux[3] = Riemann_flux_T[3] * nx * nx + Riemann_flux_T[8] * nx * ny + Riemann_flux_T[7] * nx * nz + Riemann_flux_T[4] * ny * ny + Riemann_flux_T[6] * ny * nz + Riemann_flux_T[5] * nz * nz;
    // Riemann_flux[4] = Riemann_flux_T[3] * sx * sx + Riemann_flux_T[8] * sx * sy + Riemann_flux_T[7] * sx * sz + Riemann_flux_T[4] * sy * sy + Riemann_flux_T[6] * sy * sz + Riemann_flux_T[5] * sz * sz;
    // Riemann_flux[5] = Riemann_flux_T[3] * tx * tx + Riemann_flux_T[8] * tx * ty + Riemann_flux_T[7] * tx * tz + Riemann_flux_T[4] * ty * ty + Riemann_flux_T[6] * ty * tz + Riemann_flux_T[5] * tz * tz;
    // Riemann_flux[6] = Riemann_flux_T[8] * (sx * ty + sy * tx) + Riemann_flux_T[7] * (sx * tz + sz * tx) + Riemann_flux_T[6] * (sy * tz + sz * ty) + 2 * sx * tx * Riemann_flux_T[3] + 2 * sy * ty * Riemann_flux_T[4] + 2 * sz * tz * Riemann_flux_T[5];
    // Riemann_flux[7] = Riemann_flux_T[8] * (nx * ty + ny * tx) + Riemann_flux_T[7] * (nx * tz + nz * tx) + Riemann_flux_T[6] * (ny * tz + nz * ty) + 2 * nx * tx * Riemann_flux_T[3] + 2 * ny * ty * Riemann_flux_T[4] + 2 * nz * tz * Riemann_flux_T[5];
    // Riemann_flux[8] = Riemann_flux_T[8] * (nx * sy + ny * sx) + Riemann_flux_T[7] * (nx * sz + nz * sx) + Riemann_flux_T[6] * (ny * sz + nz * sy) + 2 * nx * sx * Riemann_flux_T[3] + 2 * ny * sy * Riemann_flux_T[4] + 2 * nz * sz * Riemann_flux_T[5];

#endif // UW

    for (int n = 0; n < 9; n++)
    {
        Fu_ip12z[idx * WSIZE + n] = Riemann_flux[n] - 1.0f / 24 * order2_approximation(Hu[idx_n2 * WSIZE + n], Hu[idx_n1 * WSIZE + n], Hu[idx * WSIZE + n], Hu[idx_p1 * WSIZE + n], Hu[idx_p2 * WSIZE + n], Hu[idx_p3 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Hu[idx_n2 * WSIZE + n], Hu[idx_n1 * WSIZE + n], Hu[idx * WSIZE + n], Hu[idx_p1 * WSIZE + n], Hu[idx_p2 * WSIZE + n], Hu[idx_p3 * WSIZE + n]);
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