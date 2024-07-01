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
*	Update: Tianhong Xu, 12231218@mail.sustech.edu.cn
*   Update Time: 2024-07-02
*   Update Content: Modify the high order approximation to Chu et al. (2023)
*
*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*      3. Zhang, W., Liu, Y., & Chen, X. (2023). A Mixed‐Flux‐Based Nodal Discontinuous Galerkin Method for 3D Dynamic Rupture Modeling. Journal of Geophysical Research: Solid Earth, e2022JB025817.
*      4. Chu, Shaoshuai and Kurganov, Alexander and Xin, Ruixiao, New More Efficient A-Weno Schemes. Available at SSRN: https://ssrn.com/abstract=4486288 or http://dx.doi.org/10.2139/ssrn.4486288
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

#define order2_approximation(u1, u2, u3, u4, u5) (1.0f / 12 * (-u1 + 16 * u2 - 30 * u3 + 16 * u4 - u5))
#define order4_approximation(u1, u2, u3, u4, u5) ((u1 - 4 * u2 + 6 * u3 - 4 * u4 + u5))

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

__GLOBAL__
void wave_deriv_alternative_flux_FD_x(FLOAT *Riemann_flux, FLOAT *h_W, FLOAT *W, FLOAT *CJM,
#ifdef PML
                                      PML_BETA pml_beta,
#endif
                                      int _nx_, int _ny_, int _nz_, float rDH, float DT
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
    float jac;

    long long idx_n3, idx_n2, idx_n1, idx, idx_p1, idx_p2, idx_p3;

    float u_ip12n[9], u_ip12p[9];

#ifdef LF
    float fu_ip12n[9], fu_ip12p[9];
#endif

    // * X direction
    CALCULATE3D(i, j, k, HALO - 1, _nx, HALO, _ny, HALO, _nz)

#ifdef PML
    pml_beta_x = pml_beta.x[i];
    pml_beta_y = pml_beta.y[j];
    pml_beta_z = pml_beta.z[k];
#endif

    idx_n3 = INDEX(i - 3, j, k);
    idx_n2 = INDEX(i - 2, j, k);
    idx_n1 = INDEX(i - 1, j, k);
    idx = INDEX(i, j, k);
    idx_p1 = INDEX(i + 1, j, k);
    idx_p2 = INDEX(i + 2, j, k);
    idx_p3 = INDEX(i + 3, j, k);

    xi_x_J_h = CJM[idx * CJMSIZE + 0];
    xi_y_J_h = CJM[idx * CJMSIZE + 1];
    xi_z_J_h = CJM[idx * CJMSIZE + 2];
    jac = 1.0 / CJM[idx * CJMSIZE + 9];

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

    for (int n = 0; n < 9; n++)
    {
        Riemann_flux[idx * WSIZE + n] = 0.5f * (fu_ip12p[n] + fu_ip12n[n] - alpha * (u_ip12p[n] - u_ip12n[n]));
    }
#endif

    END_CALCULATE3D()
}

__GLOBAL__
void wave_deriv_alternative_flux_FD_y(FLOAT *Riemann_flux, FLOAT *h_W, FLOAT *W, FLOAT *CJM,
#ifdef PML
                                      PML_BETA pml_beta,
#endif
                                      int _nx_, int _ny_, int _nz_, float rDH, float DT
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
    float jac;

    long long idx_n3, idx_n2, idx_n1, idx, idx_p1, idx_p2, idx_p3;

    float u_ip12n[9], u_ip12p[9];

#ifdef LF
    float fu_ip12n[9], fu_ip12p[9];
#endif

    // * Y direction
    CALCULATE3D(i, j, k, HALO, _nx, HALO - 1, _ny, HALO, _nz)

#ifdef PML
    pml_beta_x = pml_beta.x[i];
    pml_beta_y = pml_beta.y[j];
    pml_beta_z = pml_beta.z[k];
#endif

    idx_n3 = INDEX(i, j - 3, k);
    idx_n2 = INDEX(i, j - 2, k);
    idx_n1 = INDEX(i, j - 1, k);
    idx = INDEX(i, j, k);
    idx_p1 = INDEX(i, j + 1, k);
    idx_p2 = INDEX(i, j + 2, k);
    idx_p3 = INDEX(i, j + 3, k);

    et_x_J_h = CJM[idx * CJMSIZE + 3];
    et_y_J_h = CJM[idx * CJMSIZE + 4];
    et_z_J_h = CJM[idx * CJMSIZE + 5];
    jac = 1.0 / CJM[idx * CJMSIZE + 9];

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

    for (int n = 0; n < 9; n++)
    {
        Riemann_flux[idx * WSIZE + n] = 0.5f * (fu_ip12p[n] + fu_ip12n[n] - alpha * (u_ip12p[n] - u_ip12n[n]));
    }
#endif
    END_CALCULATE3D()
}

__GLOBAL__
void wave_deriv_alternative_flux_FD_z(FLOAT *Riemann_flux, FLOAT *h_W, FLOAT *W, FLOAT *CJM,
#ifdef PML
                                      PML_BETA pml_beta,
#endif
                                      int _nx_, int _ny_, int _nz_, float rDH, float DT
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
    float jac;

    long long idx_n3, idx_n2, idx_n1, idx, idx_p1, idx_p2, idx_p3;

    float u_ip12n[9], u_ip12p[9];

#ifdef LF
    float fu_ip12n[9], fu_ip12p[9];
#endif

    // * Z direction
    CALCULATE3D(i, j, k, HALO, _nx, HALO, _ny, HALO - 1, _nz)

#ifdef PML
    pml_beta_x = pml_beta.x[i];
    pml_beta_y = pml_beta.y[j];
    pml_beta_z = pml_beta.z[k];
#endif

    idx_n3 = INDEX(i, j, k - 3);
    idx_n2 = INDEX(i, j, k - 2);
    idx_n1 = INDEX(i, j, k - 1);
    idx = INDEX(i, j, k);
    idx_p1 = INDEX(i, j, k + 1);
    idx_p2 = INDEX(i, j, k + 2);
    idx_p3 = INDEX(i, j, k + 3);

    zt_x_J_h = CJM[idx * CJMSIZE + 6];
    zt_y_J_h = CJM[idx * CJMSIZE + 7];
    zt_z_J_h = CJM[idx * CJMSIZE + 8];
    jac = 1.0 / CJM[idx * CJMSIZE + 9];

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

    for (int n = 0; n < 9; n++)
    {
        Riemann_flux[idx * WSIZE + n] = 0.5f * (fu_ip12p[n] + fu_ip12n[n] - alpha * (u_ip12p[n] - u_ip12n[n]));
    }
#endif
    END_CALCULATE3D()
}

__GLOBAL__
void cal_du_x(FLOAT *Riemann_flux, FLOAT *h_W, FLOAT *CJM,
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

    long long idx_n3, idx_n2, idx_n1, idx, idx_p1, idx_p2, idx_p3;
    float jac;

    CALCULATE3D(i, j, k, HALO, _nx, HALO, _ny, HALO, _nz)

    idx_n3 = INDEX(i - 3, j, k);
    idx_n2 = INDEX(i - 2, j, k);
    idx_n1 = INDEX(i - 1, j, k);
    idx = INDEX(i, j, k);
    idx_p1 = INDEX(i + 1, j, k);
    idx_p2 = INDEX(i + 2, j, k);
    idx_p3 = INDEX(i + 3, j, k);

    jac = 1.0 / CJM[idx * CJMSIZE + 9];

    for (int n = 0; n < 9; n++)
    {
        h_W[idx * WSIZE + n] = DT * rDH * jac * (-((Riemann_flux[idx * WSIZE + n] - 1.0f / 24 * order2_approximation(Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n], Riemann_flux[idx_p2 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n], Riemann_flux[idx_p2 * WSIZE + n])) - (Riemann_flux[idx_n1 * WSIZE + n] - 1.0f / 24 * order2_approximation(Riemann_flux[idx_n3 * WSIZE + n], Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Riemann_flux[idx_n3 * WSIZE + n], Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n]))));
    }
    END_CALCULATE3D()
}

__GLOBAL__
void cal_du_y(FLOAT *Riemann_flux, FLOAT *h_W, FLOAT *CJM,
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

    long long idx_n3, idx_n2, idx_n1, idx, idx_p1, idx_p2, idx_p3;
    float jac;

    CALCULATE3D(i, j, k, HALO, _nx, HALO, _ny, HALO, _nz)

    idx_n3 = INDEX(i, j - 3, k);
    idx_n2 = INDEX(i, j - 2, k);
    idx_n1 = INDEX(i, j - 1, k);
    idx = INDEX(i, j, k);
    idx_p1 = INDEX(i, j + 1, k);
    idx_p2 = INDEX(i, j + 2, k);
    idx_p3 = INDEX(i, j + 3, k);

    jac = 1.0 / CJM[idx * CJMSIZE + 9];

    for (int n = 0; n < 9; n++)
    {
        h_W[idx * WSIZE + n] += DT * rDH * jac * (-((Riemann_flux[idx * WSIZE + n] - 1.0f / 24 * order2_approximation(Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n], Riemann_flux[idx_p2 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n], Riemann_flux[idx_p2 * WSIZE + n])) - (Riemann_flux[idx_n1 * WSIZE + n] - 1.0f / 24 * order2_approximation(Riemann_flux[idx_n3 * WSIZE + n], Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Riemann_flux[idx_n3 * WSIZE + n], Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n]))));
    }
    END_CALCULATE3D()
}

__GLOBAL__
void cal_du_z(FLOAT *Riemann_flux, FLOAT *h_W, FLOAT *CJM,
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

    long long idx_n3, idx_n2, idx_n1, idx, idx_p1, idx_p2, idx_p3;
    float jac;

    CALCULATE3D(i, j, k, HALO, _nx, HALO, _ny, HALO, _nz)

    idx_n3 = INDEX(i, j, k - 3);
    idx_n2 = INDEX(i, j, k - 2);
    idx_n1 = INDEX(i, j, k - 1);
    idx = INDEX(i, j, k);
    idx_p1 = INDEX(i, j, k + 1);
    idx_p2 = INDEX(i, j, k + 2);
    idx_p3 = INDEX(i, j, k + 3);

    jac = 1.0 / CJM[idx * CJMSIZE + 9];

    for (int n = 0; n < 9; n++)
    {
        h_W[idx * WSIZE + n] += DT * rDH * jac * (-((Riemann_flux[idx * WSIZE + n] - 1.0f / 24 * order2_approximation(Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n], Riemann_flux[idx_p2 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n], Riemann_flux[idx_p2 * WSIZE + n])) - (Riemann_flux[idx_n1 * WSIZE + n] - 1.0f / 24 * order2_approximation(Riemann_flux[idx_n3 * WSIZE + n], Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Riemann_flux[idx_n3 * WSIZE + n], Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n]))));
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
    wave_deriv_alternative_flux_FD_x<<<blocks, threads>>>(wave.Riemann_flux, wave.h_W, wave.W, CJM,
#ifdef PML
                                                          pml_beta,
#endif // PML
                                                          _nx_, _ny_, _nz_, rDH, DT
#ifdef LF
                                                          ,
                                                          vp_max_for_SCFDM
#endif // LF
    );

    cal_du_x<<<blocks, threads>>>(wave.Riemann_flux, wave.h_W, CJM,
#ifdef PML
                                  pml_beta,
#endif // PML
                                  _nx_, _ny_, _nz_, rDH, FB1, FB2, FB3, DT);

    wave_deriv_alternative_flux_FD_y<<<blocks, threads>>>(wave.Riemann_flux, wave.h_W, wave.W, CJM,
#ifdef PML
                                                          pml_beta,
#endif // PML
                                                          _nx_, _ny_, _nz_, rDH, DT
#ifdef LF
                                                          ,
                                                          vp_max_for_SCFDM
#endif // LF
    );

    cal_du_y<<<blocks, threads>>>(wave.Riemann_flux, wave.h_W, CJM,
#ifdef PML
                                  pml_beta,
#endif // PML
                                  _nx_, _ny_, _nz_, rDH, FB1, FB2, FB3, DT);

    wave_deriv_alternative_flux_FD_z<<<blocks, threads>>>(wave.Riemann_flux, wave.h_W, wave.W, CJM,
#ifdef PML
                                                          pml_beta,
#endif // PML
                                                          _nx_, _ny_, _nz_, rDH, DT
#ifdef LF
                                                          ,
                                                          vp_max_for_SCFDM
#endif // LF
    );

    cal_du_z<<<blocks, threads>>>(wave.Riemann_flux, wave.h_W, CJM,
#ifdef PML
                                  pml_beta,
#endif // PML
                                  _nx_, _ny_, _nz_, rDH, FB1, FB2, FB3, DT);

    // CHECK(cudaDeviceSynchronize());

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