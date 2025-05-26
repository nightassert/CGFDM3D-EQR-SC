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
*   Update Content: Modify the high order approximation to Chu et al. (2023) and optimize the code
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

#define order2_approximation(u1, u2, u3, u4, u5) (1.0f / 12 * (-u1 + 16 * u2 - 30 * u3 + 16 * u4 - u5))
#define order4_approximation(u1, u2, u3, u4, u5) ((u1 - 4 * u2 + 6 * u3 - 4 * u4 + u5))

#define order2_approximation_f(u1, u2, u3, u4, u5, u6) (1.0f / 48 * (-5 * u1 + 39 * u2 - 34 * u3 - 34 * u4 + 39 * u5 - 5 * u6))
#define order4_approximation_f(u1, u2, u3, u4, u5, u6) (1.0f / 2 * (u1 - 3 * u2 + 2 * u3 + 2 * u4 - 3 * u5 + u6))

#ifdef LF
extern float vp_max_for_SCFDM;
#endif

#define WENO5_interpolation WENO5_JS

// WENO5-JS interpolation scheme
__DEVICE__
float WENO5_JS(float u1, float u2, float u3, float u4, float u5)
{
    // smoothness indicators
    float WENO_beta1 = 0.3333333333f * (10 * u3 * u3 - 31 * u3 * u4 + 25 * u4 * u4 + 11 * u3 * u5 - 19 * u4 * u5 + 4 * u5 * u5);
    float WENO_beta2 = 0.3333333333f * (4 * u2 * u2 - 13 * u2 * u3 + 13 * u3 * u3 + 5 * u2 * u4 - 13 * u3 * u4 + 4 * u4 * u4);
    float WENO_beta3 = 0.3333333333f * (4 * u1 * u1 - 19 * u1 * u2 + 25 * u2 * u2 + 11 * u1 * u3 - 31 * u2 * u3 + 10 * u3 * u3);

    WENO_beta1 = 0.3125f / ((WENO_beta1 + 1e-6) * (WENO_beta1 + 1e-6));
    WENO_beta2 = 0.625f / ((WENO_beta2 + 1e-6) * (WENO_beta2 + 1e-6));
    WENO_beta3 = 0.0625f / ((WENO_beta3 + 1e-6) * (WENO_beta3 + 1e-6));

    // nonlinear weights
    float total_weights = 1.0f / (WENO_beta1 + WENO_beta2 + WENO_beta3);

    // WENO interpolation
    return (WENO_beta1 * total_weights) * (0.375f * u3 + 0.75f * u4 - 0.125f * u5) + (WENO_beta2 * total_weights) * (-0.125f * u2 + 0.75f * u3 + 0.375f * u4) + (WENO_beta3 * total_weights) * (0.375f * u1 - 1.25f * u2 + 1.875f * u3);
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

    float nx;
    float ny;
    float nz;
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

    nx = CJM[idx * CJMSIZE + 0];
    ny = CJM[idx * CJMSIZE + 1];
    nz = CJM[idx * CJMSIZE + 2];
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
    fu_ip12n[0] = -(nx * (lambda * u_ip12n[4] + lambda * u_ip12n[5] + u_ip12n[3] * (lambda + 2 * mu)) + mu * ny * u_ip12n[8] + mu * nz * u_ip12n[7]);
    fu_ip12n[1] = -(ny * (lambda * u_ip12n[3] + lambda * u_ip12n[5] + u_ip12n[4] * (lambda + 2 * mu)) + mu * nx * u_ip12n[8] + mu * nz * u_ip12n[6]);
    fu_ip12n[2] = -(nz * (lambda * u_ip12n[3] + lambda * u_ip12n[4] + u_ip12n[5] * (lambda + 2 * mu)) + mu * nx * u_ip12n[7] + mu * ny * u_ip12n[6]);
    fu_ip12n[3] = -((nx * u_ip12n[0]) * buoyancy);
    fu_ip12n[4] = -((ny * u_ip12n[1]) * buoyancy);
    fu_ip12n[5] = -((nz * u_ip12n[2]) * buoyancy);
    fu_ip12n[6] = -((ny * u_ip12n[2]) * buoyancy + (nz * u_ip12n[1]) * buoyancy);
    fu_ip12n[7] = -((nx * u_ip12n[2]) * buoyancy + (nz * u_ip12n[0]) * buoyancy);
    fu_ip12n[8] = -((nx * u_ip12n[1]) * buoyancy + (ny * u_ip12n[0]) * buoyancy);

    fu_ip12p[0] = -(nx * (lambda * u_ip12p[4] + lambda * u_ip12p[5] + u_ip12p[3] * (lambda + 2 * mu)) + mu * ny * u_ip12p[8] + mu * nz * u_ip12p[7]);
    fu_ip12p[1] = -(ny * (lambda * u_ip12p[3] + lambda * u_ip12p[5] + u_ip12p[4] * (lambda + 2 * mu)) + mu * nx * u_ip12p[8] + mu * nz * u_ip12p[6]);
    fu_ip12p[2] = -(nz * (lambda * u_ip12p[3] + lambda * u_ip12p[4] + u_ip12p[5] * (lambda + 2 * mu)) + mu * nx * u_ip12p[7] + mu * ny * u_ip12p[6]);
    fu_ip12p[3] = -((nx * u_ip12p[0]) * buoyancy);
    fu_ip12p[4] = -((ny * u_ip12p[1]) * buoyancy);
    fu_ip12p[5] = -((nz * u_ip12p[2]) * buoyancy);
    fu_ip12p[6] = -((ny * u_ip12p[2]) * buoyancy + (nz * u_ip12p[1]) * buoyancy);
    fu_ip12p[7] = -((nx * u_ip12p[2]) * buoyancy + (nz * u_ip12p[0]) * buoyancy);
    fu_ip12p[8] = -((nx * u_ip12p[1]) * buoyancy + (ny * u_ip12p[0]) * buoyancy);

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

    float nx;
    float ny;
    float nz;
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

    nx = CJM[idx * CJMSIZE + 3];
    ny = CJM[idx * CJMSIZE + 4];
    nz = CJM[idx * CJMSIZE + 5];
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
    fu_ip12n[0] = -(nx * (lambda * u_ip12n[4] + lambda * u_ip12n[5] + u_ip12n[3] * (lambda + 2 * mu)) + mu * ny * u_ip12n[8] + mu * nz * u_ip12n[7]);
    fu_ip12n[1] = -(ny * (lambda * u_ip12n[3] + lambda * u_ip12n[5] + u_ip12n[4] * (lambda + 2 * mu)) + mu * nx * u_ip12n[8] + mu * nz * u_ip12n[6]);
    fu_ip12n[2] = -(nz * (lambda * u_ip12n[3] + lambda * u_ip12n[4] + u_ip12n[5] * (lambda + 2 * mu)) + mu * nx * u_ip12n[7] + mu * ny * u_ip12n[6]);
    fu_ip12n[3] = -((nx * u_ip12n[0]) * buoyancy);
    fu_ip12n[4] = -((ny * u_ip12n[1]) * buoyancy);
    fu_ip12n[5] = -((nz * u_ip12n[2]) * buoyancy);
    fu_ip12n[6] = -((ny * u_ip12n[2]) * buoyancy + (nz * u_ip12n[1]) * buoyancy);
    fu_ip12n[7] = -((nx * u_ip12n[2]) * buoyancy + (nz * u_ip12n[0]) * buoyancy);
    fu_ip12n[8] = -((nx * u_ip12n[1]) * buoyancy + (ny * u_ip12n[0]) * buoyancy);

    fu_ip12p[0] = -(nx * (lambda * u_ip12p[4] + lambda * u_ip12p[5] + u_ip12p[3] * (lambda + 2 * mu)) + mu * ny * u_ip12p[8] + mu * nz * u_ip12p[7]);
    fu_ip12p[1] = -(ny * (lambda * u_ip12p[3] + lambda * u_ip12p[5] + u_ip12p[4] * (lambda + 2 * mu)) + mu * nx * u_ip12p[8] + mu * nz * u_ip12p[6]);
    fu_ip12p[2] = -(nz * (lambda * u_ip12p[3] + lambda * u_ip12p[4] + u_ip12p[5] * (lambda + 2 * mu)) + mu * nx * u_ip12p[7] + mu * ny * u_ip12p[6]);
    fu_ip12p[3] = -((nx * u_ip12p[0]) * buoyancy);
    fu_ip12p[4] = -((ny * u_ip12p[1]) * buoyancy);
    fu_ip12p[5] = -((nz * u_ip12p[2]) * buoyancy);
    fu_ip12p[6] = -((ny * u_ip12p[2]) * buoyancy + (nz * u_ip12p[1]) * buoyancy);
    fu_ip12p[7] = -((nx * u_ip12p[2]) * buoyancy + (nz * u_ip12p[0]) * buoyancy);
    fu_ip12p[8] = -((nx * u_ip12p[1]) * buoyancy + (ny * u_ip12p[0]) * buoyancy);

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

    float nx;
    float ny;
    float nz;
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

    nx = CJM[idx * CJMSIZE + 6];
    ny = CJM[idx * CJMSIZE + 7];
    nz = CJM[idx * CJMSIZE + 8];
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
    fu_ip12n[0] = -(nx * (lambda * u_ip12n[4] + lambda * u_ip12n[5] + u_ip12n[3] * (lambda + 2 * mu)) + mu * ny * u_ip12n[8] + mu * nz * u_ip12n[7]);
    fu_ip12n[1] = -(ny * (lambda * u_ip12n[3] + lambda * u_ip12n[5] + u_ip12n[4] * (lambda + 2 * mu)) + mu * nx * u_ip12n[8] + mu * nz * u_ip12n[6]);
    fu_ip12n[2] = -(nz * (lambda * u_ip12n[3] + lambda * u_ip12n[4] + u_ip12n[5] * (lambda + 2 * mu)) + mu * nx * u_ip12n[7] + mu * ny * u_ip12n[6]);
    fu_ip12n[3] = -((nx * u_ip12n[0]) * buoyancy);
    fu_ip12n[4] = -((ny * u_ip12n[1]) * buoyancy);
    fu_ip12n[5] = -((nz * u_ip12n[2]) * buoyancy);
    fu_ip12n[6] = -((ny * u_ip12n[2]) * buoyancy + (nz * u_ip12n[1]) * buoyancy);
    fu_ip12n[7] = -((nx * u_ip12n[2]) * buoyancy + (nz * u_ip12n[0]) * buoyancy);
    fu_ip12n[8] = -((nx * u_ip12n[1]) * buoyancy + (ny * u_ip12n[0]) * buoyancy);

    fu_ip12p[0] = -(nx * (lambda * u_ip12p[4] + lambda * u_ip12p[5] + u_ip12p[3] * (lambda + 2 * mu)) + mu * ny * u_ip12p[8] + mu * nz * u_ip12p[7]);
    fu_ip12p[1] = -(ny * (lambda * u_ip12p[3] + lambda * u_ip12p[5] + u_ip12p[4] * (lambda + 2 * mu)) + mu * nx * u_ip12p[8] + mu * nz * u_ip12p[6]);
    fu_ip12p[2] = -(nz * (lambda * u_ip12p[3] + lambda * u_ip12p[4] + u_ip12p[5] * (lambda + 2 * mu)) + mu * nx * u_ip12p[7] + mu * ny * u_ip12p[6]);
    fu_ip12p[3] = -((nx * u_ip12p[0]) * buoyancy);
    fu_ip12p[4] = -((ny * u_ip12p[1]) * buoyancy);
    fu_ip12p[5] = -((nz * u_ip12p[2]) * buoyancy);
    fu_ip12p[6] = -((ny * u_ip12p[2]) * buoyancy + (nz * u_ip12p[1]) * buoyancy);
    fu_ip12p[7] = -((nx * u_ip12p[2]) * buoyancy + (nz * u_ip12p[0]) * buoyancy);
    fu_ip12p[8] = -((nx * u_ip12p[1]) * buoyancy + (ny * u_ip12p[0]) * buoyancy);

    for (int n = 0; n < 9; n++)
    {
        Riemann_flux[idx * WSIZE + n] = 0.5f * (fu_ip12p[n] + fu_ip12n[n] - alpha * (u_ip12p[n] - u_ip12n[n]));
    }
#endif
    END_CALCULATE3D()
}

__GLOBAL__
void wave_deriv_alternative_flux_FD_char_x(FLOAT *Riemann_flux, FLOAT *h_W, FLOAT *W, FLOAT *CJM,
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
    float rho;

    float nx;
    float ny;
    float nz;
    float jac;
    float sum_square;

    long long idx_n3, idx_n2, idx_n1, idx, idx_p1, idx_p2, idx_p3;

    float u_ip12n[9], u_ip12p[9];

    float u[6][9], v[6][9], v_ip12n[9], v_ip12p[9];

#ifdef LF
    float fu_ip12n[9], fu_ip12p[9];
#endif

    // * X direction
    CALCULATE3D(i, j, k, HALO - 1, HALO + 3, HALO, _ny, HALO, _nz)

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

    nx = CJM[idx * CJMSIZE + 0] + 1e-6;
    ny = CJM[idx * CJMSIZE + 1] + 1e-6;
    nz = CJM[idx * CJMSIZE + 2] + 1e-6;
    sum_square = nx * nx + ny * ny + nz * nz;
    jac = 1.0 / CJM[idx * CJMSIZE + 9];

    mu = CJM[idx * CJMSIZE + 10];
    lambda = CJM[idx * CJMSIZE + 11];
    buoyancy = CJM[idx * CJMSIZE + 12];
    rho = 1.0 / buoyancy;

    // Store original variables
    for (int n = 0; n < 6; n++)
    {
        u[n][0] = W[INDEX(i - 2 + n, j, k) * WSIZE + 0];
        u[n][1] = W[INDEX(i - 2 + n, j, k) * WSIZE + 1];
        u[n][2] = W[INDEX(i - 2 + n, j, k) * WSIZE + 2];
        u[n][3] = W[INDEX(i - 2 + n, j, k) * WSIZE + 3];
        u[n][4] = W[INDEX(i - 2 + n, j, k) * WSIZE + 4];
        u[n][5] = W[INDEX(i - 2 + n, j, k) * WSIZE + 5];
        u[n][6] = W[INDEX(i - 2 + n, j, k) * WSIZE + 6];
        u[n][7] = W[INDEX(i - 2 + n, j, k) * WSIZE + 7];
        u[n][8] = W[INDEX(i - 2 + n, j, k) * WSIZE + 8];
    }

    // Transform to characteristic domain
    for (int n = 0; n < 6; n++)
    {
        v[n][0] = -(lambda * nx * (ny * ny * ny) * u[n][7] - 2 * mu * (nx * nx * nx * nx) * u[n][6] - lambda * (nx * nx * nx * nx) * u[n][6] + lambda * (nx * nx * nx) * ny * u[n][7] + lambda * nx * (nz * nz * nz) * u[n][8] + lambda * (nx * nx * nx) * nz * u[n][8] + 2 * lambda * ny * (nz * nz * nz) * u[n][3] + 2 * lambda * (ny * ny * ny) * nz * u[n][3] + 4 * lambda * ny * (nz * nz * nz) * u[n][4] + 4 * lambda * (ny * ny * ny) * nz * u[n][5] + 2 * mu * nx * (ny * ny * ny) * u[n][7] + 2 * mu * (nx * nx * nx) * ny * u[n][7] + 2 * mu * nx * (nz * nz * nz) * u[n][8] + 2 * mu * (nx * nx * nx) * nz * u[n][8] + 4 * mu * ny * (nz * nz * nz) * u[n][4] + 4 * mu * (ny * ny * ny) * nz * u[n][5] - lambda * (nx * nx) * (ny * ny) * u[n][6] - lambda * (nx * nx) * (nz * nz) * u[n][6] - 4 * lambda * (ny * ny) * (nz * nz) * u[n][6] - 2 * mu * (nx * nx) * (ny * ny) * u[n][6] - 2 * mu * (nx * nx) * (nz * nz) * u[n][6] - 4 * mu * (ny * ny) * (nz * nz) * u[n][6] - 2 * lambda * (nx * nx) * ny * nz * u[n][3] + 4 * lambda * (nx * nx) * ny * nz * u[n][4] + 4 * lambda * (nx * nx) * ny * nz * u[n][5] - 3 * lambda * nx * ny * (nz * nz) * u[n][7] - 3 * lambda * nx * (ny * ny) * nz * u[n][8] - 4 * mu * (nx * nx) * ny * nz * u[n][3] + 4 * mu * (nx * nx) * ny * nz * u[n][4] + 4 * mu * (nx * nx) * ny * nz * u[n][5] - 2 * mu * nx * ny * (nz * nz) * u[n][7] - 2 * mu * nx * (ny * ny) * nz * u[n][8]) / ((lambda + 2 * mu) * (sum_square * sum_square));
        v[n][1] = -(lambda * nx * (ny * ny * ny) * u[n][6] - 2 * mu * (ny * ny * ny * ny) * u[n][7] - lambda * (ny * ny * ny * ny) * u[n][7] + lambda * (nx * nx * nx) * ny * u[n][6] + 4 * lambda * nx * (nz * nz * nz) * u[n][3] + 2 * lambda * nx * (nz * nz * nz) * u[n][4] + 2 * lambda * (nx * nx * nx) * nz * u[n][4] + 4 * lambda * (nx * nx * nx) * nz * u[n][5] + lambda * ny * (nz * nz * nz) * u[n][8] + lambda * (ny * ny * ny) * nz * u[n][8] + 2 * mu * nx * (ny * ny * ny) * u[n][6] + 2 * mu * (nx * nx * nx) * ny * u[n][6] + 4 * mu * nx * (nz * nz * nz) * u[n][3] + 4 * mu * (nx * nx * nx) * nz * u[n][5] + 2 * mu * ny * (nz * nz * nz) * u[n][8] + 2 * mu * (ny * ny * ny) * nz * u[n][8] - lambda * (nx * nx) * (ny * ny) * u[n][7] - 4 * lambda * (nx * nx) * (nz * nz) * u[n][7] - lambda * (ny * ny) * (nz * nz) * u[n][7] - 2 * mu * (nx * nx) * (ny * ny) * u[n][7] - 4 * mu * (nx * nx) * (nz * nz) * u[n][7] - 2 * mu * (ny * ny) * (nz * nz) * u[n][7] + 4 * lambda * nx * (ny * ny) * nz * u[n][3] - 2 * lambda * nx * (ny * ny) * nz * u[n][4] + 4 * lambda * nx * (ny * ny) * nz * u[n][5] - 3 * lambda * nx * ny * (nz * nz) * u[n][6] - 3 * lambda * (nx * nx) * ny * nz * u[n][8] + 4 * mu * nx * (ny * ny) * nz * u[n][3] - 4 * mu * nx * (ny * ny) * nz * u[n][4] + 4 * mu * nx * (ny * ny) * nz * u[n][5] - 2 * mu * nx * ny * (nz * nz) * u[n][6] - 2 * mu * (nx * nx) * ny * nz * u[n][8]) / ((lambda + 2 * mu) * (sum_square * sum_square));
        v[n][2] = -(4 * lambda * nx * (ny * ny * ny) * u[n][3] - 2 * mu * (nz * nz * nz * nz) * u[n][8] - lambda * (nz * nz * nz * nz) * u[n][8] + 4 * lambda * (nx * nx * nx) * ny * u[n][4] + 2 * lambda * nx * (ny * ny * ny) * u[n][5] + 2 * lambda * (nx * nx * nx) * ny * u[n][5] + lambda * nx * (nz * nz * nz) * u[n][6] + lambda * (nx * nx * nx) * nz * u[n][6] + lambda * ny * (nz * nz * nz) * u[n][7] + lambda * (ny * ny * ny) * nz * u[n][7] + 4 * mu * nx * (ny * ny * ny) * u[n][3] + 4 * mu * (nx * nx * nx) * ny * u[n][4] + 2 * mu * nx * (nz * nz * nz) * u[n][6] + 2 * mu * (nx * nx * nx) * nz * u[n][6] + 2 * mu * ny * (nz * nz * nz) * u[n][7] + 2 * mu * (ny * ny * ny) * nz * u[n][7] - 4 * lambda * (nx * nx) * (ny * ny) * u[n][8] - lambda * (nx * nx) * (nz * nz) * u[n][8] - lambda * (ny * ny) * (nz * nz) * u[n][8] - 4 * mu * (nx * nx) * (ny * ny) * u[n][8] - 2 * mu * (nx * nx) * (nz * nz) * u[n][8] - 2 * mu * (ny * ny) * (nz * nz) * u[n][8] + 4 * lambda * nx * ny * (nz * nz) * u[n][3] + 4 * lambda * nx * ny * (nz * nz) * u[n][4] - 2 * lambda * nx * ny * (nz * nz) * u[n][5] - 3 * lambda * nx * (ny * ny) * nz * u[n][6] - 3 * lambda * (nx * nx) * ny * nz * u[n][7] + 4 * mu * nx * ny * (nz * nz) * u[n][3] + 4 * mu * nx * ny * (nz * nz) * u[n][4] - 4 * mu * nx * ny * (nz * nz) * u[n][5] - 2 * mu * nx * (ny * ny) * nz * u[n][6] - 2 * mu * (nx * nx) * ny * nz * u[n][7]) / ((lambda + 2 * mu) * (sum_square * sum_square));
        v[n][3] = ((nx * nx * nx) * u[n][2] * sqrt(mu * rho * sum_square) + (nz * nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) + nx * (ny * ny) * u[n][2] * sqrt(mu * rho * sum_square) - (nx * nx) * nz * u[n][0] * sqrt(mu * rho * sum_square) - nx * (nz * nz) * u[n][2] * sqrt(mu * rho * sum_square) + (ny * ny) * nz * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][7] + mu * (nz * nz * nz * nz) * rho * u[n][7] + mu * (nx * nx) * (ny * ny) * rho * u[n][7] - 2 * mu * (nx * nx) * (nz * nz) * rho * u[n][7] + mu * (ny * ny) * (nz * nz) * rho * u[n][7] - 2 * nx * ny * nz * u[n][1] * sqrt(mu * rho * sum_square) + mu * nx * (ny * ny * ny) * rho * u[n][6] + mu * (nx * nx * nx) * ny * rho * u[n][6] + 2 * mu * nx * (nz * nz * nz) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * nz * rho * u[n][3] - 2 * mu * nx * (nz * nz * nz) * rho * u[n][5] + 2 * mu * (nx * nx * nx) * nz * rho * u[n][5] + mu * ny * (nz * nz * nz) * rho * u[n][8] + mu * (ny * ny * ny) * nz * rho * u[n][8] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][3] - 4 * mu * nx * (ny * ny) * nz * rho * u[n][4] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][5] - 3 * mu * nx * ny * (nz * nz) * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][8]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][4] = ((nx * nx * nx) * u[n][1] * sqrt(mu * rho * sum_square) + (ny * ny * ny) * u[n][0] * sqrt(mu * rho * sum_square) - (nx * nx) * ny * u[n][0] * sqrt(mu * rho * sum_square) - nx * (ny * ny) * u[n][1] * sqrt(mu * rho * sum_square) + nx * (nz * nz) * u[n][1] * sqrt(mu * rho * sum_square) + ny * (nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][8] + mu * (ny * ny * ny * ny) * rho * u[n][8] - 2 * mu * (nx * nx) * (ny * ny) * rho * u[n][8] + mu * (nx * nx) * (nz * nz) * rho * u[n][8] + mu * (ny * ny) * (nz * nz) * rho * u[n][8] - 2 * nx * ny * nz * u[n][2] * sqrt(mu * rho * sum_square) + 2 * mu * nx * (ny * ny * ny) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * ny * rho * u[n][3] - 2 * mu * nx * (ny * ny * ny) * rho * u[n][4] + 2 * mu * (nx * nx * nx) * ny * rho * u[n][4] + mu * nx * (nz * nz * nz) * rho * u[n][6] + mu * (nx * nx * nx) * nz * rho * u[n][6] + mu * ny * (nz * nz * nz) * rho * u[n][7] + mu * (ny * ny * ny) * nz * rho * u[n][7] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][3] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][4] - 4 * mu * nx * ny * (nz * nz) * rho * u[n][5] - 3 * mu * nx * (ny * ny) * nz * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][7]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][5] = (rho * rho * (lambda + 2 * mu) * (lambda * (nx * nx * nx * nx) * ny * u[n][0] + lambda * nx * (ny * ny * ny * ny) * u[n][1] + 2 * mu * (nx * nx * nx * nx) * ny * u[n][0] + 2 * mu * nx * (ny * ny * ny * ny) * u[n][1] + lambda * (nx * nx) * (ny * ny * ny) * u[n][0] + lambda * (nx * nx * nx) * (ny * ny) * u[n][1] + 2 * mu * (nx * nx) * (ny * ny * ny) * u[n][0] + 2 * mu * (nx * nx * nx) * (ny * ny) * u[n][1] + lambda * nx * (ny * ny * ny) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx * nx) * ny * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * (ny * ny * ny) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx * nx) * ny * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx) * ny * (nz * nz) * u[n][0] + lambda * nx * (ny * ny) * (nz * nz) * u[n][1] + 2 * mu * (nx * nx) * ny * (nz * nz) * u[n][0] + 2 * mu * nx * (ny * ny) * (nz * nz) * u[n][1] + 2 * mu * (nx * nx) * (ny * ny) * u[n][8] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz * nz) * u[n][2] + lambda * nx * (ny * ny * ny) * nz * u[n][2] + lambda * (nx * nx * nx) * ny * nz * u[n][2] + 2 * mu * nx * ny * (nz * nz * nz) * u[n][2] + 2 * mu * nx * (ny * ny * ny) * nz * u[n][2] + 2 * mu * (nx * nx * nx) * ny * nz * u[n][2] + lambda * nx * ny * (nz * nz) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * nx * (ny * ny) * nz * u[n][6] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * (nx * nx) * ny * nz * u[n][7] * sqrt(rho * (lambda + 2 * mu) * sum_square))) / ((rho * (lambda + 2 * mu) * sum_square) * (rho * (lambda + 2 * mu) * sum_square) * sqrt(rho * (lambda + 2 * mu) * sum_square));
        v[n][6] = ((nx * nx) * nz * u[n][0] * sqrt(mu * rho * sum_square) - (nz * nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) - nx * (ny * ny) * u[n][2] * sqrt(mu * rho * sum_square) - (nx * nx * nx) * u[n][2] * sqrt(mu * rho * sum_square) + nx * (nz * nz) * u[n][2] * sqrt(mu * rho * sum_square) - (ny * ny) * nz * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][7] + mu * (nz * nz * nz * nz) * rho * u[n][7] + mu * (nx * nx) * (ny * ny) * rho * u[n][7] - 2 * mu * (nx * nx) * (nz * nz) * rho * u[n][7] + mu * (ny * ny) * (nz * nz) * rho * u[n][7] + 2 * nx * ny * nz * u[n][1] * sqrt(mu * rho * sum_square) + mu * nx * (ny * ny * ny) * rho * u[n][6] + mu * (nx * nx * nx) * ny * rho * u[n][6] + 2 * mu * nx * (nz * nz * nz) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * nz * rho * u[n][3] - 2 * mu * nx * (nz * nz * nz) * rho * u[n][5] + 2 * mu * (nx * nx * nx) * nz * rho * u[n][5] + mu * ny * (nz * nz * nz) * rho * u[n][8] + mu * (ny * ny * ny) * nz * rho * u[n][8] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][3] - 4 * mu * nx * (ny * ny) * nz * rho * u[n][4] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][5] - 3 * mu * nx * ny * (nz * nz) * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][8]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][7] = ((nx * nx) * ny * u[n][0] * sqrt(mu * rho * sum_square) - (ny * ny * ny) * u[n][0] * sqrt(mu * rho * sum_square) - (nx * nx * nx) * u[n][1] * sqrt(mu * rho * sum_square) + nx * (ny * ny) * u[n][1] * sqrt(mu * rho * sum_square) - nx * (nz * nz) * u[n][1] * sqrt(mu * rho * sum_square) - ny * (nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][8] + mu * (ny * ny * ny * ny) * rho * u[n][8] - 2 * mu * (nx * nx) * (ny * ny) * rho * u[n][8] + mu * (nx * nx) * (nz * nz) * rho * u[n][8] + mu * (ny * ny) * (nz * nz) * rho * u[n][8] + 2 * nx * ny * nz * u[n][2] * sqrt(mu * rho * sum_square) + 2 * mu * nx * (ny * ny * ny) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * ny * rho * u[n][3] - 2 * mu * nx * (ny * ny * ny) * rho * u[n][4] + 2 * mu * (nx * nx * nx) * ny * rho * u[n][4] + mu * nx * (nz * nz * nz) * rho * u[n][6] + mu * (nx * nx * nx) * nz * rho * u[n][6] + mu * ny * (nz * nz * nz) * rho * u[n][7] + mu * (ny * ny * ny) * nz * rho * u[n][7] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][3] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][4] - 4 * mu * nx * ny * (nz * nz) * rho * u[n][5] - 3 * mu * nx * (ny * ny) * nz * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][7]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][8] = -(rho * rho * (lambda + 2 * mu) * (lambda * (nx * nx * nx * nx) * ny * u[n][0] + lambda * nx * (ny * ny * ny * ny) * u[n][1] + 2 * mu * (nx * nx * nx * nx) * ny * u[n][0] + 2 * mu * nx * (ny * ny * ny * ny) * u[n][1] + lambda * (nx * nx) * (ny * ny * ny) * u[n][0] + lambda * (nx * nx * nx) * (ny * ny) * u[n][1] + 2 * mu * (nx * nx) * (ny * ny * ny) * u[n][0] + 2 * mu * (nx * nx * nx) * (ny * ny) * u[n][1] - lambda * nx * (ny * ny * ny) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * (nx * nx * nx) * ny * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * (ny * ny * ny) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * (nx * nx * nx) * ny * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx) * ny * (nz * nz) * u[n][0] + lambda * nx * (ny * ny) * (nz * nz) * u[n][1] + 2 * mu * (nx * nx) * ny * (nz * nz) * u[n][0] + 2 * mu * nx * (ny * ny) * (nz * nz) * u[n][1] - 2 * mu * (nx * nx) * (ny * ny) * u[n][8] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz * nz) * u[n][2] + lambda * nx * (ny * ny * ny) * nz * u[n][2] + lambda * (nx * nx * nx) * ny * nz * u[n][2] + 2 * mu * nx * ny * (nz * nz * nz) * u[n][2] + 2 * mu * nx * (ny * ny * ny) * nz * u[n][2] + 2 * mu * (nx * nx * nx) * ny * nz * u[n][2] - lambda * nx * ny * (nz * nz) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * ny * (nz * nz) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * nx * (ny * ny) * nz * u[n][6] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * (nx * nx) * ny * nz * u[n][7] * sqrt(rho * (lambda + 2 * mu) * sum_square))) / ((rho * (lambda + 2 * mu) * sum_square) * (rho * (lambda + 2 * mu) * sum_square) * sqrt(rho * (lambda + 2 * mu) * sum_square));
    }

    // Shock capture interpolation
    for (int n = 0; n < 9; n++)
    {
        v_ip12n[n] = WENO5_interpolation(v[0][n], v[1][n], v[2][n], v[3][n], v[4][n]);
        v_ip12p[n] = WENO5_interpolation(v[5][n], v[4][n], v[3][n], v[2][n], v[1][n]);
    }

    // Transform back to physical domain
    u_ip12n[0] = (v_ip12n[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) - (v_ip12n[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) + (ny * v_ip12n[4] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (ny * v_ip12n[7] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nz * v_ip12n[3] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (nz * v_ip12n[6] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz));
    u_ip12n[1] = (v_ip12n[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12n[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12n[4] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12n[7] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12n[3] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12n[6] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12n[2] = (nz * v_ip12n[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (nz * v_ip12n[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (v_ip12n[3] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12n[6] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12n[4] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12n[7] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12n[3] = (nx * v_ip12n[5]) / (2 * ny) + (nx * v_ip12n[8]) / (2 * ny) + (nx * ny * v_ip12n[4]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * ny * v_ip12n[7]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12n[3]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12n[6]) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (v_ip12n[2] * (2 * lambda * (ny * ny) - lambda * (nx * nx) + 2 * mu * (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12n[1] * (2 * lambda * (nz * nz) - lambda * (nx * nx) + 2 * mu * (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12n[0] * ((ny * ny) + (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu));
    u_ip12n[4] = (ny * v_ip12n[5]) / (2 * nx) + (ny * v_ip12n[8]) / (2 * nx) - ((ny * ny) * nz * v_ip12n[3]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - ((ny * ny) * nz * v_ip12n[6]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12n[4] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12n[7] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12n[2] * (2 * lambda * (nx * nx) - lambda * (ny * ny) + 2 * mu * (nx * nx))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12n[0] * (2 * lambda * (nz * nz) - lambda * (ny * ny) + 2 * mu * (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12n[1] * ((nx * nx) + (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu));
    u_ip12n[5] = ((nz * nz) * v_ip12n[5]) / (2 * nx * ny) + ((nz * nz) * v_ip12n[8]) / (2 * nx * ny) - (ny * (nz * nz) * v_ip12n[4]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * (nz * nz) * v_ip12n[7]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12n[3] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12n[6] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12n[1] * (2 * lambda * (nx * nx) - lambda * (nz * nz) + 2 * mu * (nx * nx))) / (2 * nx * nz * (3 * lambda + 2 * mu)) - (v_ip12n[0] * (2 * lambda * (ny * ny) - lambda * (nz * nz) + 2 * mu * (ny * ny))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12n[2] * ((nx * nx) + (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu));
    u_ip12n[6] = -((nx * nx * nx) * v_ip12n[0] - (ny * ny * ny) * v_ip12n[3] - (ny * ny * ny) * v_ip12n[6] - (nz * nz * nz) * v_ip12n[4] - (nz * nz * nz) * v_ip12n[5] - (nz * nz * nz) * v_ip12n[7] - (nz * nz * nz) * v_ip12n[8] - nx * (ny * ny) * v_ip12n[0] + (nx * nx) * ny * v_ip12n[3] + (nx * nx) * ny * v_ip12n[6] - nx * (nz * nz) * v_ip12n[0] + (nx * nx) * nz * v_ip12n[4] + (nx * nx) * nz * v_ip12n[5] + (nx * nx) * nz * v_ip12n[7] + (nx * nx) * nz * v_ip12n[8] + ny * (nz * nz) * v_ip12n[3] + (ny * ny) * nz * v_ip12n[4] - (ny * ny) * nz * v_ip12n[5] + ny * (nz * nz) * v_ip12n[6] + (ny * ny) * nz * v_ip12n[7] - (ny * ny) * nz * v_ip12n[8]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12n[7] = v_ip12n[1] + v_ip12n[3] + v_ip12n[6] + (nz * v_ip12n[5]) / ny + (nz * v_ip12n[8]) / ny;
    u_ip12n[8] = v_ip12n[2] + v_ip12n[4] + v_ip12n[5] + v_ip12n[7] + v_ip12n[8];

    u_ip12p[0] = (v_ip12p[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) - (v_ip12p[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) + (ny * v_ip12p[4] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (ny * v_ip12p[7] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nz * v_ip12p[3] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (nz * v_ip12p[6] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz));
    u_ip12p[1] = (v_ip12p[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12p[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12p[4] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12p[7] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12p[3] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12p[6] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12p[2] = (nz * v_ip12p[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (nz * v_ip12p[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (v_ip12p[3] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12p[6] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12p[4] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12p[7] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12p[3] = (nx * v_ip12p[5]) / (2 * ny) + (nx * v_ip12p[8]) / (2 * ny) + (nx * ny * v_ip12p[4]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * ny * v_ip12p[7]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12p[3]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12p[6]) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (v_ip12p[2] * (2 * lambda * (ny * ny) - lambda * (nx * nx) + 2 * mu * (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12p[1] * (2 * lambda * (nz * nz) - lambda * (nx * nx) + 2 * mu * (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12p[0] * ((ny * ny) + (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu));
    u_ip12p[4] = (ny * v_ip12p[5]) / (2 * nx) + (ny * v_ip12p[8]) / (2 * nx) - ((ny * ny) * nz * v_ip12p[3]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - ((ny * ny) * nz * v_ip12p[6]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12p[4] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12p[7] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12p[2] * (2 * lambda * (nx * nx) - lambda * (ny * ny) + 2 * mu * (nx * nx))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12p[0] * (2 * lambda * (nz * nz) - lambda * (ny * ny) + 2 * mu * (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12p[1] * ((nx * nx) + (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu));
    u_ip12p[5] = ((nz * nz) * v_ip12p[5]) / (2 * nx * ny) + ((nz * nz) * v_ip12p[8]) / (2 * nx * ny) - (ny * (nz * nz) * v_ip12p[4]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * (nz * nz) * v_ip12p[7]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12p[3] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12p[6] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12p[1] * (2 * lambda * (nx * nx) - lambda * (nz * nz) + 2 * mu * (nx * nx))) / (2 * nx * nz * (3 * lambda + 2 * mu)) - (v_ip12p[0] * (2 * lambda * (ny * ny) - lambda * (nz * nz) + 2 * mu * (ny * ny))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12p[2] * ((nx * nx) + (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu));
    u_ip12p[6] = -((nx * nx * nx) * v_ip12p[0] - (ny * ny * ny) * v_ip12p[3] - (ny * ny * ny) * v_ip12p[6] - (nz * nz * nz) * v_ip12p[4] - (nz * nz * nz) * v_ip12p[5] - (nz * nz * nz) * v_ip12p[7] - (nz * nz * nz) * v_ip12p[8] - nx * (ny * ny) * v_ip12p[0] + (nx * nx) * ny * v_ip12p[3] + (nx * nx) * ny * v_ip12p[6] - nx * (nz * nz) * v_ip12p[0] + (nx * nx) * nz * v_ip12p[4] + (nx * nx) * nz * v_ip12p[5] + (nx * nx) * nz * v_ip12p[7] + (nx * nx) * nz * v_ip12p[8] + ny * (nz * nz) * v_ip12p[3] + (ny * ny) * nz * v_ip12p[4] - (ny * ny) * nz * v_ip12p[5] + ny * (nz * nz) * v_ip12p[6] + (ny * ny) * nz * v_ip12p[7] - (ny * ny) * nz * v_ip12p[8]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12p[7] = v_ip12p[1] + v_ip12p[3] + v_ip12p[6] + (nz * v_ip12p[5]) / ny + (nz * v_ip12p[8]) / ny;
    u_ip12p[8] = v_ip12p[2] + v_ip12p[4] + v_ip12p[5] + v_ip12p[7] + v_ip12p[8];

#ifdef LF
    // Riemann solver: Lax-Friedrichs
    fu_ip12n[0] = -(nx * (lambda * u_ip12n[4] + lambda * u_ip12n[5] + u_ip12n[3] * (lambda + 2 * mu)) + mu * ny * u_ip12n[8] + mu * nz * u_ip12n[7]);
    fu_ip12n[1] = -(ny * (lambda * u_ip12n[3] + lambda * u_ip12n[5] + u_ip12n[4] * (lambda + 2 * mu)) + mu * nx * u_ip12n[8] + mu * nz * u_ip12n[6]);
    fu_ip12n[2] = -(nz * (lambda * u_ip12n[3] + lambda * u_ip12n[4] + u_ip12n[5] * (lambda + 2 * mu)) + mu * nx * u_ip12n[7] + mu * ny * u_ip12n[6]);
    fu_ip12n[3] = -((nx * u_ip12n[0]) * buoyancy);
    fu_ip12n[4] = -((ny * u_ip12n[1]) * buoyancy);
    fu_ip12n[5] = -((nz * u_ip12n[2]) * buoyancy);
    fu_ip12n[6] = -((ny * u_ip12n[2]) * buoyancy + (nz * u_ip12n[1]) * buoyancy);
    fu_ip12n[7] = -((nx * u_ip12n[2]) * buoyancy + (nz * u_ip12n[0]) * buoyancy);
    fu_ip12n[8] = -((nx * u_ip12n[1]) * buoyancy + (ny * u_ip12n[0]) * buoyancy);

    fu_ip12p[0] = -(nx * (lambda * u_ip12p[4] + lambda * u_ip12p[5] + u_ip12p[3] * (lambda + 2 * mu)) + mu * ny * u_ip12p[8] + mu * nz * u_ip12p[7]);
    fu_ip12p[1] = -(ny * (lambda * u_ip12p[3] + lambda * u_ip12p[5] + u_ip12p[4] * (lambda + 2 * mu)) + mu * nx * u_ip12p[8] + mu * nz * u_ip12p[6]);
    fu_ip12p[2] = -(nz * (lambda * u_ip12p[3] + lambda * u_ip12p[4] + u_ip12p[5] * (lambda + 2 * mu)) + mu * nx * u_ip12p[7] + mu * ny * u_ip12p[6]);
    fu_ip12p[3] = -((nx * u_ip12p[0]) * buoyancy);
    fu_ip12p[4] = -((ny * u_ip12p[1]) * buoyancy);
    fu_ip12p[5] = -((nz * u_ip12p[2]) * buoyancy);
    fu_ip12p[6] = -((ny * u_ip12p[2]) * buoyancy + (nz * u_ip12p[1]) * buoyancy);
    fu_ip12p[7] = -((nx * u_ip12p[2]) * buoyancy + (nz * u_ip12p[0]) * buoyancy);
    fu_ip12p[8] = -((nx * u_ip12p[1]) * buoyancy + (ny * u_ip12p[0]) * buoyancy);

    for (int n = 0; n < 9; n++)
    {
        Riemann_flux[idx * WSIZE + n] = 0.5f * (fu_ip12p[n] + fu_ip12n[n] - alpha * (u_ip12p[n] - u_ip12n[n]));
    }
#endif

    END_CALCULATE3D()

    CALCULATE3D(i, j, k, _nx - HALO, _nx, HALO, _ny, HALO, _nz)

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

    nx = CJM[idx * CJMSIZE + 0] + 1e-6;
    ny = CJM[idx * CJMSIZE + 1] + 1e-6;
    nz = CJM[idx * CJMSIZE + 2] + 1e-6;
    sum_square = nx * nx + ny * ny + nz * nz;
    jac = 1.0 / CJM[idx * CJMSIZE + 9];

    mu = CJM[idx * CJMSIZE + 10];
    lambda = CJM[idx * CJMSIZE + 11];
    buoyancy = CJM[idx * CJMSIZE + 12];
    rho = 1.0 / buoyancy;

    // Store original variables
    for (int n = 0; n < 6; n++)
    {
        u[n][0] = W[INDEX(i - 2 + n, j, k) * WSIZE + 0];
        u[n][1] = W[INDEX(i - 2 + n, j, k) * WSIZE + 1];
        u[n][2] = W[INDEX(i - 2 + n, j, k) * WSIZE + 2];
        u[n][3] = W[INDEX(i - 2 + n, j, k) * WSIZE + 3];
        u[n][4] = W[INDEX(i - 2 + n, j, k) * WSIZE + 4];
        u[n][5] = W[INDEX(i - 2 + n, j, k) * WSIZE + 5];
        u[n][6] = W[INDEX(i - 2 + n, j, k) * WSIZE + 6];
        u[n][7] = W[INDEX(i - 2 + n, j, k) * WSIZE + 7];
        u[n][8] = W[INDEX(i - 2 + n, j, k) * WSIZE + 8];
    }

    // Transform to characteristic domain
    for (int n = 0; n < 6; n++)
    {
        v[n][0] = -(lambda * nx * (ny * ny * ny) * u[n][7] - 2 * mu * (nx * nx * nx * nx) * u[n][6] - lambda * (nx * nx * nx * nx) * u[n][6] + lambda * (nx * nx * nx) * ny * u[n][7] + lambda * nx * (nz * nz * nz) * u[n][8] + lambda * (nx * nx * nx) * nz * u[n][8] + 2 * lambda * ny * (nz * nz * nz) * u[n][3] + 2 * lambda * (ny * ny * ny) * nz * u[n][3] + 4 * lambda * ny * (nz * nz * nz) * u[n][4] + 4 * lambda * (ny * ny * ny) * nz * u[n][5] + 2 * mu * nx * (ny * ny * ny) * u[n][7] + 2 * mu * (nx * nx * nx) * ny * u[n][7] + 2 * mu * nx * (nz * nz * nz) * u[n][8] + 2 * mu * (nx * nx * nx) * nz * u[n][8] + 4 * mu * ny * (nz * nz * nz) * u[n][4] + 4 * mu * (ny * ny * ny) * nz * u[n][5] - lambda * (nx * nx) * (ny * ny) * u[n][6] - lambda * (nx * nx) * (nz * nz) * u[n][6] - 4 * lambda * (ny * ny) * (nz * nz) * u[n][6] - 2 * mu * (nx * nx) * (ny * ny) * u[n][6] - 2 * mu * (nx * nx) * (nz * nz) * u[n][6] - 4 * mu * (ny * ny) * (nz * nz) * u[n][6] - 2 * lambda * (nx * nx) * ny * nz * u[n][3] + 4 * lambda * (nx * nx) * ny * nz * u[n][4] + 4 * lambda * (nx * nx) * ny * nz * u[n][5] - 3 * lambda * nx * ny * (nz * nz) * u[n][7] - 3 * lambda * nx * (ny * ny) * nz * u[n][8] - 4 * mu * (nx * nx) * ny * nz * u[n][3] + 4 * mu * (nx * nx) * ny * nz * u[n][4] + 4 * mu * (nx * nx) * ny * nz * u[n][5] - 2 * mu * nx * ny * (nz * nz) * u[n][7] - 2 * mu * nx * (ny * ny) * nz * u[n][8]) / ((lambda + 2 * mu) * (sum_square * sum_square));
        v[n][1] = -(lambda * nx * (ny * ny * ny) * u[n][6] - 2 * mu * (ny * ny * ny * ny) * u[n][7] - lambda * (ny * ny * ny * ny) * u[n][7] + lambda * (nx * nx * nx) * ny * u[n][6] + 4 * lambda * nx * (nz * nz * nz) * u[n][3] + 2 * lambda * nx * (nz * nz * nz) * u[n][4] + 2 * lambda * (nx * nx * nx) * nz * u[n][4] + 4 * lambda * (nx * nx * nx) * nz * u[n][5] + lambda * ny * (nz * nz * nz) * u[n][8] + lambda * (ny * ny * ny) * nz * u[n][8] + 2 * mu * nx * (ny * ny * ny) * u[n][6] + 2 * mu * (nx * nx * nx) * ny * u[n][6] + 4 * mu * nx * (nz * nz * nz) * u[n][3] + 4 * mu * (nx * nx * nx) * nz * u[n][5] + 2 * mu * ny * (nz * nz * nz) * u[n][8] + 2 * mu * (ny * ny * ny) * nz * u[n][8] - lambda * (nx * nx) * (ny * ny) * u[n][7] - 4 * lambda * (nx * nx) * (nz * nz) * u[n][7] - lambda * (ny * ny) * (nz * nz) * u[n][7] - 2 * mu * (nx * nx) * (ny * ny) * u[n][7] - 4 * mu * (nx * nx) * (nz * nz) * u[n][7] - 2 * mu * (ny * ny) * (nz * nz) * u[n][7] + 4 * lambda * nx * (ny * ny) * nz * u[n][3] - 2 * lambda * nx * (ny * ny) * nz * u[n][4] + 4 * lambda * nx * (ny * ny) * nz * u[n][5] - 3 * lambda * nx * ny * (nz * nz) * u[n][6] - 3 * lambda * (nx * nx) * ny * nz * u[n][8] + 4 * mu * nx * (ny * ny) * nz * u[n][3] - 4 * mu * nx * (ny * ny) * nz * u[n][4] + 4 * mu * nx * (ny * ny) * nz * u[n][5] - 2 * mu * nx * ny * (nz * nz) * u[n][6] - 2 * mu * (nx * nx) * ny * nz * u[n][8]) / ((lambda + 2 * mu) * (sum_square * sum_square));
        v[n][2] = -(4 * lambda * nx * (ny * ny * ny) * u[n][3] - 2 * mu * (nz * nz * nz * nz) * u[n][8] - lambda * (nz * nz * nz * nz) * u[n][8] + 4 * lambda * (nx * nx * nx) * ny * u[n][4] + 2 * lambda * nx * (ny * ny * ny) * u[n][5] + 2 * lambda * (nx * nx * nx) * ny * u[n][5] + lambda * nx * (nz * nz * nz) * u[n][6] + lambda * (nx * nx * nx) * nz * u[n][6] + lambda * ny * (nz * nz * nz) * u[n][7] + lambda * (ny * ny * ny) * nz * u[n][7] + 4 * mu * nx * (ny * ny * ny) * u[n][3] + 4 * mu * (nx * nx * nx) * ny * u[n][4] + 2 * mu * nx * (nz * nz * nz) * u[n][6] + 2 * mu * (nx * nx * nx) * nz * u[n][6] + 2 * mu * ny * (nz * nz * nz) * u[n][7] + 2 * mu * (ny * ny * ny) * nz * u[n][7] - 4 * lambda * (nx * nx) * (ny * ny) * u[n][8] - lambda * (nx * nx) * (nz * nz) * u[n][8] - lambda * (ny * ny) * (nz * nz) * u[n][8] - 4 * mu * (nx * nx) * (ny * ny) * u[n][8] - 2 * mu * (nx * nx) * (nz * nz) * u[n][8] - 2 * mu * (ny * ny) * (nz * nz) * u[n][8] + 4 * lambda * nx * ny * (nz * nz) * u[n][3] + 4 * lambda * nx * ny * (nz * nz) * u[n][4] - 2 * lambda * nx * ny * (nz * nz) * u[n][5] - 3 * lambda * nx * (ny * ny) * nz * u[n][6] - 3 * lambda * (nx * nx) * ny * nz * u[n][7] + 4 * mu * nx * ny * (nz * nz) * u[n][3] + 4 * mu * nx * ny * (nz * nz) * u[n][4] - 4 * mu * nx * ny * (nz * nz) * u[n][5] - 2 * mu * nx * (ny * ny) * nz * u[n][6] - 2 * mu * (nx * nx) * ny * nz * u[n][7]) / ((lambda + 2 * mu) * (sum_square * sum_square));
        v[n][3] = ((nx * nx * nx) * u[n][2] * sqrt(mu * rho * sum_square) + (nz * nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) + nx * (ny * ny) * u[n][2] * sqrt(mu * rho * sum_square) - (nx * nx) * nz * u[n][0] * sqrt(mu * rho * sum_square) - nx * (nz * nz) * u[n][2] * sqrt(mu * rho * sum_square) + (ny * ny) * nz * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][7] + mu * (nz * nz * nz * nz) * rho * u[n][7] + mu * (nx * nx) * (ny * ny) * rho * u[n][7] - 2 * mu * (nx * nx) * (nz * nz) * rho * u[n][7] + mu * (ny * ny) * (nz * nz) * rho * u[n][7] - 2 * nx * ny * nz * u[n][1] * sqrt(mu * rho * sum_square) + mu * nx * (ny * ny * ny) * rho * u[n][6] + mu * (nx * nx * nx) * ny * rho * u[n][6] + 2 * mu * nx * (nz * nz * nz) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * nz * rho * u[n][3] - 2 * mu * nx * (nz * nz * nz) * rho * u[n][5] + 2 * mu * (nx * nx * nx) * nz * rho * u[n][5] + mu * ny * (nz * nz * nz) * rho * u[n][8] + mu * (ny * ny * ny) * nz * rho * u[n][8] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][3] - 4 * mu * nx * (ny * ny) * nz * rho * u[n][4] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][5] - 3 * mu * nx * ny * (nz * nz) * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][8]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][4] = ((nx * nx * nx) * u[n][1] * sqrt(mu * rho * sum_square) + (ny * ny * ny) * u[n][0] * sqrt(mu * rho * sum_square) - (nx * nx) * ny * u[n][0] * sqrt(mu * rho * sum_square) - nx * (ny * ny) * u[n][1] * sqrt(mu * rho * sum_square) + nx * (nz * nz) * u[n][1] * sqrt(mu * rho * sum_square) + ny * (nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][8] + mu * (ny * ny * ny * ny) * rho * u[n][8] - 2 * mu * (nx * nx) * (ny * ny) * rho * u[n][8] + mu * (nx * nx) * (nz * nz) * rho * u[n][8] + mu * (ny * ny) * (nz * nz) * rho * u[n][8] - 2 * nx * ny * nz * u[n][2] * sqrt(mu * rho * sum_square) + 2 * mu * nx * (ny * ny * ny) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * ny * rho * u[n][3] - 2 * mu * nx * (ny * ny * ny) * rho * u[n][4] + 2 * mu * (nx * nx * nx) * ny * rho * u[n][4] + mu * nx * (nz * nz * nz) * rho * u[n][6] + mu * (nx * nx * nx) * nz * rho * u[n][6] + mu * ny * (nz * nz * nz) * rho * u[n][7] + mu * (ny * ny * ny) * nz * rho * u[n][7] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][3] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][4] - 4 * mu * nx * ny * (nz * nz) * rho * u[n][5] - 3 * mu * nx * (ny * ny) * nz * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][7]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][5] = (rho * rho * (lambda + 2 * mu) * (lambda * (nx * nx * nx * nx) * ny * u[n][0] + lambda * nx * (ny * ny * ny * ny) * u[n][1] + 2 * mu * (nx * nx * nx * nx) * ny * u[n][0] + 2 * mu * nx * (ny * ny * ny * ny) * u[n][1] + lambda * (nx * nx) * (ny * ny * ny) * u[n][0] + lambda * (nx * nx * nx) * (ny * ny) * u[n][1] + 2 * mu * (nx * nx) * (ny * ny * ny) * u[n][0] + 2 * mu * (nx * nx * nx) * (ny * ny) * u[n][1] + lambda * nx * (ny * ny * ny) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx * nx) * ny * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * (ny * ny * ny) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx * nx) * ny * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx) * ny * (nz * nz) * u[n][0] + lambda * nx * (ny * ny) * (nz * nz) * u[n][1] + 2 * mu * (nx * nx) * ny * (nz * nz) * u[n][0] + 2 * mu * nx * (ny * ny) * (nz * nz) * u[n][1] + 2 * mu * (nx * nx) * (ny * ny) * u[n][8] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz * nz) * u[n][2] + lambda * nx * (ny * ny * ny) * nz * u[n][2] + lambda * (nx * nx * nx) * ny * nz * u[n][2] + 2 * mu * nx * ny * (nz * nz * nz) * u[n][2] + 2 * mu * nx * (ny * ny * ny) * nz * u[n][2] + 2 * mu * (nx * nx * nx) * ny * nz * u[n][2] + lambda * nx * ny * (nz * nz) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * nx * (ny * ny) * nz * u[n][6] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * (nx * nx) * ny * nz * u[n][7] * sqrt(rho * (lambda + 2 * mu) * sum_square))) / ((rho * (lambda + 2 * mu) * sum_square) * (rho * (lambda + 2 * mu) * sum_square) * sqrt(rho * (lambda + 2 * mu) * sum_square));
        v[n][6] = ((nx * nx) * nz * u[n][0] * sqrt(mu * rho * sum_square) - (nz * nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) - nx * (ny * ny) * u[n][2] * sqrt(mu * rho * sum_square) - (nx * nx * nx) * u[n][2] * sqrt(mu * rho * sum_square) + nx * (nz * nz) * u[n][2] * sqrt(mu * rho * sum_square) - (ny * ny) * nz * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][7] + mu * (nz * nz * nz * nz) * rho * u[n][7] + mu * (nx * nx) * (ny * ny) * rho * u[n][7] - 2 * mu * (nx * nx) * (nz * nz) * rho * u[n][7] + mu * (ny * ny) * (nz * nz) * rho * u[n][7] + 2 * nx * ny * nz * u[n][1] * sqrt(mu * rho * sum_square) + mu * nx * (ny * ny * ny) * rho * u[n][6] + mu * (nx * nx * nx) * ny * rho * u[n][6] + 2 * mu * nx * (nz * nz * nz) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * nz * rho * u[n][3] - 2 * mu * nx * (nz * nz * nz) * rho * u[n][5] + 2 * mu * (nx * nx * nx) * nz * rho * u[n][5] + mu * ny * (nz * nz * nz) * rho * u[n][8] + mu * (ny * ny * ny) * nz * rho * u[n][8] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][3] - 4 * mu * nx * (ny * ny) * nz * rho * u[n][4] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][5] - 3 * mu * nx * ny * (nz * nz) * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][8]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][7] = ((nx * nx) * ny * u[n][0] * sqrt(mu * rho * sum_square) - (ny * ny * ny) * u[n][0] * sqrt(mu * rho * sum_square) - (nx * nx * nx) * u[n][1] * sqrt(mu * rho * sum_square) + nx * (ny * ny) * u[n][1] * sqrt(mu * rho * sum_square) - nx * (nz * nz) * u[n][1] * sqrt(mu * rho * sum_square) - ny * (nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][8] + mu * (ny * ny * ny * ny) * rho * u[n][8] - 2 * mu * (nx * nx) * (ny * ny) * rho * u[n][8] + mu * (nx * nx) * (nz * nz) * rho * u[n][8] + mu * (ny * ny) * (nz * nz) * rho * u[n][8] + 2 * nx * ny * nz * u[n][2] * sqrt(mu * rho * sum_square) + 2 * mu * nx * (ny * ny * ny) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * ny * rho * u[n][3] - 2 * mu * nx * (ny * ny * ny) * rho * u[n][4] + 2 * mu * (nx * nx * nx) * ny * rho * u[n][4] + mu * nx * (nz * nz * nz) * rho * u[n][6] + mu * (nx * nx * nx) * nz * rho * u[n][6] + mu * ny * (nz * nz * nz) * rho * u[n][7] + mu * (ny * ny * ny) * nz * rho * u[n][7] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][3] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][4] - 4 * mu * nx * ny * (nz * nz) * rho * u[n][5] - 3 * mu * nx * (ny * ny) * nz * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][7]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][8] = -(rho * rho * (lambda + 2 * mu) * (lambda * (nx * nx * nx * nx) * ny * u[n][0] + lambda * nx * (ny * ny * ny * ny) * u[n][1] + 2 * mu * (nx * nx * nx * nx) * ny * u[n][0] + 2 * mu * nx * (ny * ny * ny * ny) * u[n][1] + lambda * (nx * nx) * (ny * ny * ny) * u[n][0] + lambda * (nx * nx * nx) * (ny * ny) * u[n][1] + 2 * mu * (nx * nx) * (ny * ny * ny) * u[n][0] + 2 * mu * (nx * nx * nx) * (ny * ny) * u[n][1] - lambda * nx * (ny * ny * ny) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * (nx * nx * nx) * ny * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * (ny * ny * ny) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * (nx * nx * nx) * ny * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx) * ny * (nz * nz) * u[n][0] + lambda * nx * (ny * ny) * (nz * nz) * u[n][1] + 2 * mu * (nx * nx) * ny * (nz * nz) * u[n][0] + 2 * mu * nx * (ny * ny) * (nz * nz) * u[n][1] - 2 * mu * (nx * nx) * (ny * ny) * u[n][8] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz * nz) * u[n][2] + lambda * nx * (ny * ny * ny) * nz * u[n][2] + lambda * (nx * nx * nx) * ny * nz * u[n][2] + 2 * mu * nx * ny * (nz * nz * nz) * u[n][2] + 2 * mu * nx * (ny * ny * ny) * nz * u[n][2] + 2 * mu * (nx * nx * nx) * ny * nz * u[n][2] - lambda * nx * ny * (nz * nz) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * ny * (nz * nz) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * nx * (ny * ny) * nz * u[n][6] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * (nx * nx) * ny * nz * u[n][7] * sqrt(rho * (lambda + 2 * mu) * sum_square))) / ((rho * (lambda + 2 * mu) * sum_square) * (rho * (lambda + 2 * mu) * sum_square) * sqrt(rho * (lambda + 2 * mu) * sum_square));
    }

    // Shock capture interpolation
    for (int n = 0; n < 9; n++)
    {
        v_ip12n[n] = WENO5_interpolation(v[0][n], v[1][n], v[2][n], v[3][n], v[4][n]);
        v_ip12p[n] = WENO5_interpolation(v[5][n], v[4][n], v[3][n], v[2][n], v[1][n]);
    }

    // Transform back to physical domain
    u_ip12n[0] = (v_ip12n[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) - (v_ip12n[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) + (ny * v_ip12n[4] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (ny * v_ip12n[7] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nz * v_ip12n[3] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (nz * v_ip12n[6] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz));
    u_ip12n[1] = (v_ip12n[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12n[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12n[4] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12n[7] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12n[3] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12n[6] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12n[2] = (nz * v_ip12n[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (nz * v_ip12n[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (v_ip12n[3] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12n[6] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12n[4] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12n[7] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12n[3] = (nx * v_ip12n[5]) / (2 * ny) + (nx * v_ip12n[8]) / (2 * ny) + (nx * ny * v_ip12n[4]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * ny * v_ip12n[7]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12n[3]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12n[6]) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (v_ip12n[2] * (2 * lambda * (ny * ny) - lambda * (nx * nx) + 2 * mu * (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12n[1] * (2 * lambda * (nz * nz) - lambda * (nx * nx) + 2 * mu * (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12n[0] * ((ny * ny) + (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu));
    u_ip12n[4] = (ny * v_ip12n[5]) / (2 * nx) + (ny * v_ip12n[8]) / (2 * nx) - ((ny * ny) * nz * v_ip12n[3]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - ((ny * ny) * nz * v_ip12n[6]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12n[4] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12n[7] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12n[2] * (2 * lambda * (nx * nx) - lambda * (ny * ny) + 2 * mu * (nx * nx))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12n[0] * (2 * lambda * (nz * nz) - lambda * (ny * ny) + 2 * mu * (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12n[1] * ((nx * nx) + (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu));
    u_ip12n[5] = ((nz * nz) * v_ip12n[5]) / (2 * nx * ny) + ((nz * nz) * v_ip12n[8]) / (2 * nx * ny) - (ny * (nz * nz) * v_ip12n[4]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * (nz * nz) * v_ip12n[7]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12n[3] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12n[6] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12n[1] * (2 * lambda * (nx * nx) - lambda * (nz * nz) + 2 * mu * (nx * nx))) / (2 * nx * nz * (3 * lambda + 2 * mu)) - (v_ip12n[0] * (2 * lambda * (ny * ny) - lambda * (nz * nz) + 2 * mu * (ny * ny))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12n[2] * ((nx * nx) + (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu));
    u_ip12n[6] = -((nx * nx * nx) * v_ip12n[0] - (ny * ny * ny) * v_ip12n[3] - (ny * ny * ny) * v_ip12n[6] - (nz * nz * nz) * v_ip12n[4] - (nz * nz * nz) * v_ip12n[5] - (nz * nz * nz) * v_ip12n[7] - (nz * nz * nz) * v_ip12n[8] - nx * (ny * ny) * v_ip12n[0] + (nx * nx) * ny * v_ip12n[3] + (nx * nx) * ny * v_ip12n[6] - nx * (nz * nz) * v_ip12n[0] + (nx * nx) * nz * v_ip12n[4] + (nx * nx) * nz * v_ip12n[5] + (nx * nx) * nz * v_ip12n[7] + (nx * nx) * nz * v_ip12n[8] + ny * (nz * nz) * v_ip12n[3] + (ny * ny) * nz * v_ip12n[4] - (ny * ny) * nz * v_ip12n[5] + ny * (nz * nz) * v_ip12n[6] + (ny * ny) * nz * v_ip12n[7] - (ny * ny) * nz * v_ip12n[8]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12n[7] = v_ip12n[1] + v_ip12n[3] + v_ip12n[6] + (nz * v_ip12n[5]) / ny + (nz * v_ip12n[8]) / ny;
    u_ip12n[8] = v_ip12n[2] + v_ip12n[4] + v_ip12n[5] + v_ip12n[7] + v_ip12n[8];

    u_ip12p[0] = (v_ip12p[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) - (v_ip12p[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) + (ny * v_ip12p[4] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (ny * v_ip12p[7] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nz * v_ip12p[3] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (nz * v_ip12p[6] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz));
    u_ip12p[1] = (v_ip12p[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12p[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12p[4] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12p[7] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12p[3] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12p[6] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12p[2] = (nz * v_ip12p[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (nz * v_ip12p[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (v_ip12p[3] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12p[6] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12p[4] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12p[7] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12p[3] = (nx * v_ip12p[5]) / (2 * ny) + (nx * v_ip12p[8]) / (2 * ny) + (nx * ny * v_ip12p[4]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * ny * v_ip12p[7]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12p[3]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12p[6]) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (v_ip12p[2] * (2 * lambda * (ny * ny) - lambda * (nx * nx) + 2 * mu * (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12p[1] * (2 * lambda * (nz * nz) - lambda * (nx * nx) + 2 * mu * (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12p[0] * ((ny * ny) + (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu));
    u_ip12p[4] = (ny * v_ip12p[5]) / (2 * nx) + (ny * v_ip12p[8]) / (2 * nx) - ((ny * ny) * nz * v_ip12p[3]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - ((ny * ny) * nz * v_ip12p[6]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12p[4] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12p[7] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12p[2] * (2 * lambda * (nx * nx) - lambda * (ny * ny) + 2 * mu * (nx * nx))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12p[0] * (2 * lambda * (nz * nz) - lambda * (ny * ny) + 2 * mu * (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12p[1] * ((nx * nx) + (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu));
    u_ip12p[5] = ((nz * nz) * v_ip12p[5]) / (2 * nx * ny) + ((nz * nz) * v_ip12p[8]) / (2 * nx * ny) - (ny * (nz * nz) * v_ip12p[4]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * (nz * nz) * v_ip12p[7]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12p[3] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12p[6] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12p[1] * (2 * lambda * (nx * nx) - lambda * (nz * nz) + 2 * mu * (nx * nx))) / (2 * nx * nz * (3 * lambda + 2 * mu)) - (v_ip12p[0] * (2 * lambda * (ny * ny) - lambda * (nz * nz) + 2 * mu * (ny * ny))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12p[2] * ((nx * nx) + (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu));
    u_ip12p[6] = -((nx * nx * nx) * v_ip12p[0] - (ny * ny * ny) * v_ip12p[3] - (ny * ny * ny) * v_ip12p[6] - (nz * nz * nz) * v_ip12p[4] - (nz * nz * nz) * v_ip12p[5] - (nz * nz * nz) * v_ip12p[7] - (nz * nz * nz) * v_ip12p[8] - nx * (ny * ny) * v_ip12p[0] + (nx * nx) * ny * v_ip12p[3] + (nx * nx) * ny * v_ip12p[6] - nx * (nz * nz) * v_ip12p[0] + (nx * nx) * nz * v_ip12p[4] + (nx * nx) * nz * v_ip12p[5] + (nx * nx) * nz * v_ip12p[7] + (nx * nx) * nz * v_ip12p[8] + ny * (nz * nz) * v_ip12p[3] + (ny * ny) * nz * v_ip12p[4] - (ny * ny) * nz * v_ip12p[5] + ny * (nz * nz) * v_ip12p[6] + (ny * ny) * nz * v_ip12p[7] - (ny * ny) * nz * v_ip12p[8]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12p[7] = v_ip12p[1] + v_ip12p[3] + v_ip12p[6] + (nz * v_ip12p[5]) / ny + (nz * v_ip12p[8]) / ny;
    u_ip12p[8] = v_ip12p[2] + v_ip12p[4] + v_ip12p[5] + v_ip12p[7] + v_ip12p[8];

#ifdef LF
    // Riemann solver: Lax-Friedrichs
    fu_ip12n[0] = -(nx * (lambda * u_ip12n[4] + lambda * u_ip12n[5] + u_ip12n[3] * (lambda + 2 * mu)) + mu * ny * u_ip12n[8] + mu * nz * u_ip12n[7]);
    fu_ip12n[1] = -(ny * (lambda * u_ip12n[3] + lambda * u_ip12n[5] + u_ip12n[4] * (lambda + 2 * mu)) + mu * nx * u_ip12n[8] + mu * nz * u_ip12n[6]);
    fu_ip12n[2] = -(nz * (lambda * u_ip12n[3] + lambda * u_ip12n[4] + u_ip12n[5] * (lambda + 2 * mu)) + mu * nx * u_ip12n[7] + mu * ny * u_ip12n[6]);
    fu_ip12n[3] = -((nx * u_ip12n[0]) * buoyancy);
    fu_ip12n[4] = -((ny * u_ip12n[1]) * buoyancy);
    fu_ip12n[5] = -((nz * u_ip12n[2]) * buoyancy);
    fu_ip12n[6] = -((ny * u_ip12n[2]) * buoyancy + (nz * u_ip12n[1]) * buoyancy);
    fu_ip12n[7] = -((nx * u_ip12n[2]) * buoyancy + (nz * u_ip12n[0]) * buoyancy);
    fu_ip12n[8] = -((nx * u_ip12n[1]) * buoyancy + (ny * u_ip12n[0]) * buoyancy);

    fu_ip12p[0] = -(nx * (lambda * u_ip12p[4] + lambda * u_ip12p[5] + u_ip12p[3] * (lambda + 2 * mu)) + mu * ny * u_ip12p[8] + mu * nz * u_ip12p[7]);
    fu_ip12p[1] = -(ny * (lambda * u_ip12p[3] + lambda * u_ip12p[5] + u_ip12p[4] * (lambda + 2 * mu)) + mu * nx * u_ip12p[8] + mu * nz * u_ip12p[6]);
    fu_ip12p[2] = -(nz * (lambda * u_ip12p[3] + lambda * u_ip12p[4] + u_ip12p[5] * (lambda + 2 * mu)) + mu * nx * u_ip12p[7] + mu * ny * u_ip12p[6]);
    fu_ip12p[3] = -((nx * u_ip12p[0]) * buoyancy);
    fu_ip12p[4] = -((ny * u_ip12p[1]) * buoyancy);
    fu_ip12p[5] = -((nz * u_ip12p[2]) * buoyancy);
    fu_ip12p[6] = -((ny * u_ip12p[2]) * buoyancy + (nz * u_ip12p[1]) * buoyancy);
    fu_ip12p[7] = -((nx * u_ip12p[2]) * buoyancy + (nz * u_ip12p[0]) * buoyancy);
    fu_ip12p[8] = -((nx * u_ip12p[1]) * buoyancy + (ny * u_ip12p[0]) * buoyancy);

    for (int n = 0; n < 9; n++)
    {
        Riemann_flux[idx * WSIZE + n] = 0.5f * (fu_ip12p[n] + fu_ip12n[n] - alpha * (u_ip12p[n] - u_ip12n[n]));
    }
#endif

    END_CALCULATE3D()
}

__GLOBAL__
void wave_deriv_alternative_flux_FD_char_y(FLOAT *Riemann_flux, FLOAT *h_W, FLOAT *W, FLOAT *CJM,
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
    float rho;

    float nx;
    float ny;
    float nz;
    float jac;
    float sum_square;

    long long idx_n3, idx_n2, idx_n1, idx, idx_p1, idx_p2, idx_p3;

    float u_ip12n[9], u_ip12p[9];

    float u[6][9], v[6][9], v_ip12n[9], v_ip12p[9];

#ifdef LF
    float fu_ip12n[9], fu_ip12p[9];
#endif

    // * Y direction
    CALCULATE3D(i, j, k, HALO, _nx, HALO - 1, HALO + 3, HALO, _nz)

#ifdef PML
    pml_beta_x = pml_beta.x[i];
    pml_beta_y = pml_beta.y[j];
    pml_beta_z = pml_beta.z[k];
#endif

    nx = CJM[idx * CJMSIZE + 3] + 1e-6;
    ny = CJM[idx * CJMSIZE + 4] + 1e-6;
    nz = CJM[idx * CJMSIZE + 5] + 1e-6;
    sum_square = nx * nx + ny * ny + nz * nz;
    jac = 1.0 / CJM[idx * CJMSIZE + 9];

    mu = CJM[idx * CJMSIZE + 10];
    lambda = CJM[idx * CJMSIZE + 11];
    buoyancy = CJM[idx * CJMSIZE + 12];
    rho = 1.0 / buoyancy;

    // Store original variables
    for (int n = 0; n < 6; n++)
    {
        u[n][0] = W[INDEX(i, j - 2 + n, k) * WSIZE + 0];
        u[n][1] = W[INDEX(i, j - 2 + n, k) * WSIZE + 1];
        u[n][2] = W[INDEX(i, j - 2 + n, k) * WSIZE + 2];
        u[n][3] = W[INDEX(i, j - 2 + n, k) * WSIZE + 3];
        u[n][4] = W[INDEX(i, j - 2 + n, k) * WSIZE + 4];
        u[n][5] = W[INDEX(i, j - 2 + n, k) * WSIZE + 5];
        u[n][6] = W[INDEX(i, j - 2 + n, k) * WSIZE + 6];
        u[n][7] = W[INDEX(i, j - 2 + n, k) * WSIZE + 7];
        u[n][8] = W[INDEX(i, j - 2 + n, k) * WSIZE + 8];
    }

    // Transform to characteristic domain
    for (int n = 0; n < 6; n++)
    {
        v[n][0] = -(lambda * nx * (ny * ny * ny) * u[n][7] - 2 * mu * (nx * nx * nx * nx) * u[n][6] - lambda * (nx * nx * nx * nx) * u[n][6] + lambda * (nx * nx * nx) * ny * u[n][7] + lambda * nx * (nz * nz * nz) * u[n][8] + lambda * (nx * nx * nx) * nz * u[n][8] + 2 * lambda * ny * (nz * nz * nz) * u[n][3] + 2 * lambda * (ny * ny * ny) * nz * u[n][3] + 4 * lambda * ny * (nz * nz * nz) * u[n][4] + 4 * lambda * (ny * ny * ny) * nz * u[n][5] + 2 * mu * nx * (ny * ny * ny) * u[n][7] + 2 * mu * (nx * nx * nx) * ny * u[n][7] + 2 * mu * nx * (nz * nz * nz) * u[n][8] + 2 * mu * (nx * nx * nx) * nz * u[n][8] + 4 * mu * ny * (nz * nz * nz) * u[n][4] + 4 * mu * (ny * ny * ny) * nz * u[n][5] - lambda * (nx * nx) * (ny * ny) * u[n][6] - lambda * (nx * nx) * (nz * nz) * u[n][6] - 4 * lambda * (ny * ny) * (nz * nz) * u[n][6] - 2 * mu * (nx * nx) * (ny * ny) * u[n][6] - 2 * mu * (nx * nx) * (nz * nz) * u[n][6] - 4 * mu * (ny * ny) * (nz * nz) * u[n][6] - 2 * lambda * (nx * nx) * ny * nz * u[n][3] + 4 * lambda * (nx * nx) * ny * nz * u[n][4] + 4 * lambda * (nx * nx) * ny * nz * u[n][5] - 3 * lambda * nx * ny * (nz * nz) * u[n][7] - 3 * lambda * nx * (ny * ny) * nz * u[n][8] - 4 * mu * (nx * nx) * ny * nz * u[n][3] + 4 * mu * (nx * nx) * ny * nz * u[n][4] + 4 * mu * (nx * nx) * ny * nz * u[n][5] - 2 * mu * nx * ny * (nz * nz) * u[n][7] - 2 * mu * nx * (ny * ny) * nz * u[n][8]) / ((lambda + 2 * mu) * (sum_square * sum_square));
        v[n][1] = -(lambda * nx * (ny * ny * ny) * u[n][6] - 2 * mu * (ny * ny * ny * ny) * u[n][7] - lambda * (ny * ny * ny * ny) * u[n][7] + lambda * (nx * nx * nx) * ny * u[n][6] + 4 * lambda * nx * (nz * nz * nz) * u[n][3] + 2 * lambda * nx * (nz * nz * nz) * u[n][4] + 2 * lambda * (nx * nx * nx) * nz * u[n][4] + 4 * lambda * (nx * nx * nx) * nz * u[n][5] + lambda * ny * (nz * nz * nz) * u[n][8] + lambda * (ny * ny * ny) * nz * u[n][8] + 2 * mu * nx * (ny * ny * ny) * u[n][6] + 2 * mu * (nx * nx * nx) * ny * u[n][6] + 4 * mu * nx * (nz * nz * nz) * u[n][3] + 4 * mu * (nx * nx * nx) * nz * u[n][5] + 2 * mu * ny * (nz * nz * nz) * u[n][8] + 2 * mu * (ny * ny * ny) * nz * u[n][8] - lambda * (nx * nx) * (ny * ny) * u[n][7] - 4 * lambda * (nx * nx) * (nz * nz) * u[n][7] - lambda * (ny * ny) * (nz * nz) * u[n][7] - 2 * mu * (nx * nx) * (ny * ny) * u[n][7] - 4 * mu * (nx * nx) * (nz * nz) * u[n][7] - 2 * mu * (ny * ny) * (nz * nz) * u[n][7] + 4 * lambda * nx * (ny * ny) * nz * u[n][3] - 2 * lambda * nx * (ny * ny) * nz * u[n][4] + 4 * lambda * nx * (ny * ny) * nz * u[n][5] - 3 * lambda * nx * ny * (nz * nz) * u[n][6] - 3 * lambda * (nx * nx) * ny * nz * u[n][8] + 4 * mu * nx * (ny * ny) * nz * u[n][3] - 4 * mu * nx * (ny * ny) * nz * u[n][4] + 4 * mu * nx * (ny * ny) * nz * u[n][5] - 2 * mu * nx * ny * (nz * nz) * u[n][6] - 2 * mu * (nx * nx) * ny * nz * u[n][8]) / ((lambda + 2 * mu) * (sum_square * sum_square));
        v[n][2] = -(4 * lambda * nx * (ny * ny * ny) * u[n][3] - 2 * mu * (nz * nz * nz * nz) * u[n][8] - lambda * (nz * nz * nz * nz) * u[n][8] + 4 * lambda * (nx * nx * nx) * ny * u[n][4] + 2 * lambda * nx * (ny * ny * ny) * u[n][5] + 2 * lambda * (nx * nx * nx) * ny * u[n][5] + lambda * nx * (nz * nz * nz) * u[n][6] + lambda * (nx * nx * nx) * nz * u[n][6] + lambda * ny * (nz * nz * nz) * u[n][7] + lambda * (ny * ny * ny) * nz * u[n][7] + 4 * mu * nx * (ny * ny * ny) * u[n][3] + 4 * mu * (nx * nx * nx) * ny * u[n][4] + 2 * mu * nx * (nz * nz * nz) * u[n][6] + 2 * mu * (nx * nx * nx) * nz * u[n][6] + 2 * mu * ny * (nz * nz * nz) * u[n][7] + 2 * mu * (ny * ny * ny) * nz * u[n][7] - 4 * lambda * (nx * nx) * (ny * ny) * u[n][8] - lambda * (nx * nx) * (nz * nz) * u[n][8] - lambda * (ny * ny) * (nz * nz) * u[n][8] - 4 * mu * (nx * nx) * (ny * ny) * u[n][8] - 2 * mu * (nx * nx) * (nz * nz) * u[n][8] - 2 * mu * (ny * ny) * (nz * nz) * u[n][8] + 4 * lambda * nx * ny * (nz * nz) * u[n][3] + 4 * lambda * nx * ny * (nz * nz) * u[n][4] - 2 * lambda * nx * ny * (nz * nz) * u[n][5] - 3 * lambda * nx * (ny * ny) * nz * u[n][6] - 3 * lambda * (nx * nx) * ny * nz * u[n][7] + 4 * mu * nx * ny * (nz * nz) * u[n][3] + 4 * mu * nx * ny * (nz * nz) * u[n][4] - 4 * mu * nx * ny * (nz * nz) * u[n][5] - 2 * mu * nx * (ny * ny) * nz * u[n][6] - 2 * mu * (nx * nx) * ny * nz * u[n][7]) / ((lambda + 2 * mu) * (sum_square * sum_square));
        v[n][3] = ((nx * nx * nx) * u[n][2] * sqrt(mu * rho * sum_square) + (nz * nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) + nx * (ny * ny) * u[n][2] * sqrt(mu * rho * sum_square) - (nx * nx) * nz * u[n][0] * sqrt(mu * rho * sum_square) - nx * (nz * nz) * u[n][2] * sqrt(mu * rho * sum_square) + (ny * ny) * nz * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][7] + mu * (nz * nz * nz * nz) * rho * u[n][7] + mu * (nx * nx) * (ny * ny) * rho * u[n][7] - 2 * mu * (nx * nx) * (nz * nz) * rho * u[n][7] + mu * (ny * ny) * (nz * nz) * rho * u[n][7] - 2 * nx * ny * nz * u[n][1] * sqrt(mu * rho * sum_square) + mu * nx * (ny * ny * ny) * rho * u[n][6] + mu * (nx * nx * nx) * ny * rho * u[n][6] + 2 * mu * nx * (nz * nz * nz) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * nz * rho * u[n][3] - 2 * mu * nx * (nz * nz * nz) * rho * u[n][5] + 2 * mu * (nx * nx * nx) * nz * rho * u[n][5] + mu * ny * (nz * nz * nz) * rho * u[n][8] + mu * (ny * ny * ny) * nz * rho * u[n][8] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][3] - 4 * mu * nx * (ny * ny) * nz * rho * u[n][4] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][5] - 3 * mu * nx * ny * (nz * nz) * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][8]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][4] = ((nx * nx * nx) * u[n][1] * sqrt(mu * rho * sum_square) + (ny * ny * ny) * u[n][0] * sqrt(mu * rho * sum_square) - (nx * nx) * ny * u[n][0] * sqrt(mu * rho * sum_square) - nx * (ny * ny) * u[n][1] * sqrt(mu * rho * sum_square) + nx * (nz * nz) * u[n][1] * sqrt(mu * rho * sum_square) + ny * (nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][8] + mu * (ny * ny * ny * ny) * rho * u[n][8] - 2 * mu * (nx * nx) * (ny * ny) * rho * u[n][8] + mu * (nx * nx) * (nz * nz) * rho * u[n][8] + mu * (ny * ny) * (nz * nz) * rho * u[n][8] - 2 * nx * ny * nz * u[n][2] * sqrt(mu * rho * sum_square) + 2 * mu * nx * (ny * ny * ny) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * ny * rho * u[n][3] - 2 * mu * nx * (ny * ny * ny) * rho * u[n][4] + 2 * mu * (nx * nx * nx) * ny * rho * u[n][4] + mu * nx * (nz * nz * nz) * rho * u[n][6] + mu * (nx * nx * nx) * nz * rho * u[n][6] + mu * ny * (nz * nz * nz) * rho * u[n][7] + mu * (ny * ny * ny) * nz * rho * u[n][7] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][3] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][4] - 4 * mu * nx * ny * (nz * nz) * rho * u[n][5] - 3 * mu * nx * (ny * ny) * nz * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][7]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][5] = (rho * rho * (lambda + 2 * mu) * (lambda * (nx * nx * nx * nx) * ny * u[n][0] + lambda * nx * (ny * ny * ny * ny) * u[n][1] + 2 * mu * (nx * nx * nx * nx) * ny * u[n][0] + 2 * mu * nx * (ny * ny * ny * ny) * u[n][1] + lambda * (nx * nx) * (ny * ny * ny) * u[n][0] + lambda * (nx * nx * nx) * (ny * ny) * u[n][1] + 2 * mu * (nx * nx) * (ny * ny * ny) * u[n][0] + 2 * mu * (nx * nx * nx) * (ny * ny) * u[n][1] + lambda * nx * (ny * ny * ny) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx * nx) * ny * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * (ny * ny * ny) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx * nx) * ny * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx) * ny * (nz * nz) * u[n][0] + lambda * nx * (ny * ny) * (nz * nz) * u[n][1] + 2 * mu * (nx * nx) * ny * (nz * nz) * u[n][0] + 2 * mu * nx * (ny * ny) * (nz * nz) * u[n][1] + 2 * mu * (nx * nx) * (ny * ny) * u[n][8] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz * nz) * u[n][2] + lambda * nx * (ny * ny * ny) * nz * u[n][2] + lambda * (nx * nx * nx) * ny * nz * u[n][2] + 2 * mu * nx * ny * (nz * nz * nz) * u[n][2] + 2 * mu * nx * (ny * ny * ny) * nz * u[n][2] + 2 * mu * (nx * nx * nx) * ny * nz * u[n][2] + lambda * nx * ny * (nz * nz) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * nx * (ny * ny) * nz * u[n][6] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * (nx * nx) * ny * nz * u[n][7] * sqrt(rho * (lambda + 2 * mu) * sum_square))) / ((rho * (lambda + 2 * mu) * sum_square) * (rho * (lambda + 2 * mu) * sum_square) * sqrt(rho * (lambda + 2 * mu) * sum_square));
        v[n][6] = ((nx * nx) * nz * u[n][0] * sqrt(mu * rho * sum_square) - (nz * nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) - nx * (ny * ny) * u[n][2] * sqrt(mu * rho * sum_square) - (nx * nx * nx) * u[n][2] * sqrt(mu * rho * sum_square) + nx * (nz * nz) * u[n][2] * sqrt(mu * rho * sum_square) - (ny * ny) * nz * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][7] + mu * (nz * nz * nz * nz) * rho * u[n][7] + mu * (nx * nx) * (ny * ny) * rho * u[n][7] - 2 * mu * (nx * nx) * (nz * nz) * rho * u[n][7] + mu * (ny * ny) * (nz * nz) * rho * u[n][7] + 2 * nx * ny * nz * u[n][1] * sqrt(mu * rho * sum_square) + mu * nx * (ny * ny * ny) * rho * u[n][6] + mu * (nx * nx * nx) * ny * rho * u[n][6] + 2 * mu * nx * (nz * nz * nz) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * nz * rho * u[n][3] - 2 * mu * nx * (nz * nz * nz) * rho * u[n][5] + 2 * mu * (nx * nx * nx) * nz * rho * u[n][5] + mu * ny * (nz * nz * nz) * rho * u[n][8] + mu * (ny * ny * ny) * nz * rho * u[n][8] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][3] - 4 * mu * nx * (ny * ny) * nz * rho * u[n][4] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][5] - 3 * mu * nx * ny * (nz * nz) * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][8]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][7] = ((nx * nx) * ny * u[n][0] * sqrt(mu * rho * sum_square) - (ny * ny * ny) * u[n][0] * sqrt(mu * rho * sum_square) - (nx * nx * nx) * u[n][1] * sqrt(mu * rho * sum_square) + nx * (ny * ny) * u[n][1] * sqrt(mu * rho * sum_square) - nx * (nz * nz) * u[n][1] * sqrt(mu * rho * sum_square) - ny * (nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][8] + mu * (ny * ny * ny * ny) * rho * u[n][8] - 2 * mu * (nx * nx) * (ny * ny) * rho * u[n][8] + mu * (nx * nx) * (nz * nz) * rho * u[n][8] + mu * (ny * ny) * (nz * nz) * rho * u[n][8] + 2 * nx * ny * nz * u[n][2] * sqrt(mu * rho * sum_square) + 2 * mu * nx * (ny * ny * ny) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * ny * rho * u[n][3] - 2 * mu * nx * (ny * ny * ny) * rho * u[n][4] + 2 * mu * (nx * nx * nx) * ny * rho * u[n][4] + mu * nx * (nz * nz * nz) * rho * u[n][6] + mu * (nx * nx * nx) * nz * rho * u[n][6] + mu * ny * (nz * nz * nz) * rho * u[n][7] + mu * (ny * ny * ny) * nz * rho * u[n][7] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][3] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][4] - 4 * mu * nx * ny * (nz * nz) * rho * u[n][5] - 3 * mu * nx * (ny * ny) * nz * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][7]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][8] = -(rho * rho * (lambda + 2 * mu) * (lambda * (nx * nx * nx * nx) * ny * u[n][0] + lambda * nx * (ny * ny * ny * ny) * u[n][1] + 2 * mu * (nx * nx * nx * nx) * ny * u[n][0] + 2 * mu * nx * (ny * ny * ny * ny) * u[n][1] + lambda * (nx * nx) * (ny * ny * ny) * u[n][0] + lambda * (nx * nx * nx) * (ny * ny) * u[n][1] + 2 * mu * (nx * nx) * (ny * ny * ny) * u[n][0] + 2 * mu * (nx * nx * nx) * (ny * ny) * u[n][1] - lambda * nx * (ny * ny * ny) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * (nx * nx * nx) * ny * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * (ny * ny * ny) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * (nx * nx * nx) * ny * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx) * ny * (nz * nz) * u[n][0] + lambda * nx * (ny * ny) * (nz * nz) * u[n][1] + 2 * mu * (nx * nx) * ny * (nz * nz) * u[n][0] + 2 * mu * nx * (ny * ny) * (nz * nz) * u[n][1] - 2 * mu * (nx * nx) * (ny * ny) * u[n][8] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz * nz) * u[n][2] + lambda * nx * (ny * ny * ny) * nz * u[n][2] + lambda * (nx * nx * nx) * ny * nz * u[n][2] + 2 * mu * nx * ny * (nz * nz * nz) * u[n][2] + 2 * mu * nx * (ny * ny * ny) * nz * u[n][2] + 2 * mu * (nx * nx * nx) * ny * nz * u[n][2] - lambda * nx * ny * (nz * nz) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * ny * (nz * nz) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * nx * (ny * ny) * nz * u[n][6] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * (nx * nx) * ny * nz * u[n][7] * sqrt(rho * (lambda + 2 * mu) * sum_square))) / ((rho * (lambda + 2 * mu) * sum_square) * (rho * (lambda + 2 * mu) * sum_square) * sqrt(rho * (lambda + 2 * mu) * sum_square));
    }

    // Shock capture interpolation
    for (int n = 0; n < 9; n++)
    {
        v_ip12n[n] = WENO5_interpolation(v[0][n], v[1][n], v[2][n], v[3][n], v[4][n]);
        v_ip12p[n] = WENO5_interpolation(v[5][n], v[4][n], v[3][n], v[2][n], v[1][n]);
    }

    // Transform back to physical domain
    u_ip12n[0] = (v_ip12n[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) - (v_ip12n[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) + (ny * v_ip12n[4] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (ny * v_ip12n[7] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nz * v_ip12n[3] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (nz * v_ip12n[6] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz));
    u_ip12n[1] = (v_ip12n[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12n[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12n[4] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12n[7] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12n[3] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12n[6] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12n[2] = (nz * v_ip12n[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (nz * v_ip12n[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (v_ip12n[3] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12n[6] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12n[4] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12n[7] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12n[3] = (nx * v_ip12n[5]) / (2 * ny) + (nx * v_ip12n[8]) / (2 * ny) + (nx * ny * v_ip12n[4]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * ny * v_ip12n[7]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12n[3]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12n[6]) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (v_ip12n[2] * (2 * lambda * (ny * ny) - lambda * (nx * nx) + 2 * mu * (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12n[1] * (2 * lambda * (nz * nz) - lambda * (nx * nx) + 2 * mu * (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12n[0] * ((ny * ny) + (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu));
    u_ip12n[4] = (ny * v_ip12n[5]) / (2 * nx) + (ny * v_ip12n[8]) / (2 * nx) - ((ny * ny) * nz * v_ip12n[3]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - ((ny * ny) * nz * v_ip12n[6]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12n[4] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12n[7] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12n[2] * (2 * lambda * (nx * nx) - lambda * (ny * ny) + 2 * mu * (nx * nx))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12n[0] * (2 * lambda * (nz * nz) - lambda * (ny * ny) + 2 * mu * (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12n[1] * ((nx * nx) + (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu));
    u_ip12n[5] = ((nz * nz) * v_ip12n[5]) / (2 * nx * ny) + ((nz * nz) * v_ip12n[8]) / (2 * nx * ny) - (ny * (nz * nz) * v_ip12n[4]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * (nz * nz) * v_ip12n[7]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12n[3] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12n[6] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12n[1] * (2 * lambda * (nx * nx) - lambda * (nz * nz) + 2 * mu * (nx * nx))) / (2 * nx * nz * (3 * lambda + 2 * mu)) - (v_ip12n[0] * (2 * lambda * (ny * ny) - lambda * (nz * nz) + 2 * mu * (ny * ny))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12n[2] * ((nx * nx) + (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu));
    u_ip12n[6] = -((nx * nx * nx) * v_ip12n[0] - (ny * ny * ny) * v_ip12n[3] - (ny * ny * ny) * v_ip12n[6] - (nz * nz * nz) * v_ip12n[4] - (nz * nz * nz) * v_ip12n[5] - (nz * nz * nz) * v_ip12n[7] - (nz * nz * nz) * v_ip12n[8] - nx * (ny * ny) * v_ip12n[0] + (nx * nx) * ny * v_ip12n[3] + (nx * nx) * ny * v_ip12n[6] - nx * (nz * nz) * v_ip12n[0] + (nx * nx) * nz * v_ip12n[4] + (nx * nx) * nz * v_ip12n[5] + (nx * nx) * nz * v_ip12n[7] + (nx * nx) * nz * v_ip12n[8] + ny * (nz * nz) * v_ip12n[3] + (ny * ny) * nz * v_ip12n[4] - (ny * ny) * nz * v_ip12n[5] + ny * (nz * nz) * v_ip12n[6] + (ny * ny) * nz * v_ip12n[7] - (ny * ny) * nz * v_ip12n[8]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12n[7] = v_ip12n[1] + v_ip12n[3] + v_ip12n[6] + (nz * v_ip12n[5]) / ny + (nz * v_ip12n[8]) / ny;
    u_ip12n[8] = v_ip12n[2] + v_ip12n[4] + v_ip12n[5] + v_ip12n[7] + v_ip12n[8];

    u_ip12p[0] = (v_ip12p[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) - (v_ip12p[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) + (ny * v_ip12p[4] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (ny * v_ip12p[7] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nz * v_ip12p[3] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (nz * v_ip12p[6] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz));
    u_ip12p[1] = (v_ip12p[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12p[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12p[4] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12p[7] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12p[3] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12p[6] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12p[2] = (nz * v_ip12p[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (nz * v_ip12p[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (v_ip12p[3] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12p[6] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12p[4] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12p[7] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12p[3] = (nx * v_ip12p[5]) / (2 * ny) + (nx * v_ip12p[8]) / (2 * ny) + (nx * ny * v_ip12p[4]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * ny * v_ip12p[7]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12p[3]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12p[6]) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (v_ip12p[2] * (2 * lambda * (ny * ny) - lambda * (nx * nx) + 2 * mu * (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12p[1] * (2 * lambda * (nz * nz) - lambda * (nx * nx) + 2 * mu * (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12p[0] * ((ny * ny) + (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu));
    u_ip12p[4] = (ny * v_ip12p[5]) / (2 * nx) + (ny * v_ip12p[8]) / (2 * nx) - ((ny * ny) * nz * v_ip12p[3]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - ((ny * ny) * nz * v_ip12p[6]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12p[4] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12p[7] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12p[2] * (2 * lambda * (nx * nx) - lambda * (ny * ny) + 2 * mu * (nx * nx))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12p[0] * (2 * lambda * (nz * nz) - lambda * (ny * ny) + 2 * mu * (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12p[1] * ((nx * nx) + (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu));
    u_ip12p[5] = ((nz * nz) * v_ip12p[5]) / (2 * nx * ny) + ((nz * nz) * v_ip12p[8]) / (2 * nx * ny) - (ny * (nz * nz) * v_ip12p[4]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * (nz * nz) * v_ip12p[7]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12p[3] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12p[6] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12p[1] * (2 * lambda * (nx * nx) - lambda * (nz * nz) + 2 * mu * (nx * nx))) / (2 * nx * nz * (3 * lambda + 2 * mu)) - (v_ip12p[0] * (2 * lambda * (ny * ny) - lambda * (nz * nz) + 2 * mu * (ny * ny))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12p[2] * ((nx * nx) + (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu));
    u_ip12p[6] = -((nx * nx * nx) * v_ip12p[0] - (ny * ny * ny) * v_ip12p[3] - (ny * ny * ny) * v_ip12p[6] - (nz * nz * nz) * v_ip12p[4] - (nz * nz * nz) * v_ip12p[5] - (nz * nz * nz) * v_ip12p[7] - (nz * nz * nz) * v_ip12p[8] - nx * (ny * ny) * v_ip12p[0] + (nx * nx) * ny * v_ip12p[3] + (nx * nx) * ny * v_ip12p[6] - nx * (nz * nz) * v_ip12p[0] + (nx * nx) * nz * v_ip12p[4] + (nx * nx) * nz * v_ip12p[5] + (nx * nx) * nz * v_ip12p[7] + (nx * nx) * nz * v_ip12p[8] + ny * (nz * nz) * v_ip12p[3] + (ny * ny) * nz * v_ip12p[4] - (ny * ny) * nz * v_ip12p[5] + ny * (nz * nz) * v_ip12p[6] + (ny * ny) * nz * v_ip12p[7] - (ny * ny) * nz * v_ip12p[8]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12p[7] = v_ip12p[1] + v_ip12p[3] + v_ip12p[6] + (nz * v_ip12p[5]) / ny + (nz * v_ip12p[8]) / ny;
    u_ip12p[8] = v_ip12p[2] + v_ip12p[4] + v_ip12p[5] + v_ip12p[7] + v_ip12p[8];

#ifdef LF
    // Riemann solver: Lax-Friedrichs
    fu_ip12n[0] = -(nx * (lambda * u_ip12n[4] + lambda * u_ip12n[5] + u_ip12n[3] * (lambda + 2 * mu)) + mu * ny * u_ip12n[8] + mu * nz * u_ip12n[7]);
    fu_ip12n[1] = -(ny * (lambda * u_ip12n[3] + lambda * u_ip12n[5] + u_ip12n[4] * (lambda + 2 * mu)) + mu * nx * u_ip12n[8] + mu * nz * u_ip12n[6]);
    fu_ip12n[2] = -(nz * (lambda * u_ip12n[3] + lambda * u_ip12n[4] + u_ip12n[5] * (lambda + 2 * mu)) + mu * nx * u_ip12n[7] + mu * ny * u_ip12n[6]);
    fu_ip12n[3] = -((nx * u_ip12n[0]) * buoyancy);
    fu_ip12n[4] = -((ny * u_ip12n[1]) * buoyancy);
    fu_ip12n[5] = -((nz * u_ip12n[2]) * buoyancy);
    fu_ip12n[6] = -((ny * u_ip12n[2]) * buoyancy + (nz * u_ip12n[1]) * buoyancy);
    fu_ip12n[7] = -((nx * u_ip12n[2]) * buoyancy + (nz * u_ip12n[0]) * buoyancy);
    fu_ip12n[8] = -((nx * u_ip12n[1]) * buoyancy + (ny * u_ip12n[0]) * buoyancy);

    fu_ip12p[0] = -(nx * (lambda * u_ip12p[4] + lambda * u_ip12p[5] + u_ip12p[3] * (lambda + 2 * mu)) + mu * ny * u_ip12p[8] + mu * nz * u_ip12p[7]);
    fu_ip12p[1] = -(ny * (lambda * u_ip12p[3] + lambda * u_ip12p[5] + u_ip12p[4] * (lambda + 2 * mu)) + mu * nx * u_ip12p[8] + mu * nz * u_ip12p[6]);
    fu_ip12p[2] = -(nz * (lambda * u_ip12p[3] + lambda * u_ip12p[4] + u_ip12p[5] * (lambda + 2 * mu)) + mu * nx * u_ip12p[7] + mu * ny * u_ip12p[6]);
    fu_ip12p[3] = -((nx * u_ip12p[0]) * buoyancy);
    fu_ip12p[4] = -((ny * u_ip12p[1]) * buoyancy);
    fu_ip12p[5] = -((nz * u_ip12p[2]) * buoyancy);
    fu_ip12p[6] = -((ny * u_ip12p[2]) * buoyancy + (nz * u_ip12p[1]) * buoyancy);
    fu_ip12p[7] = -((nx * u_ip12p[2]) * buoyancy + (nz * u_ip12p[0]) * buoyancy);
    fu_ip12p[8] = -((nx * u_ip12p[1]) * buoyancy + (ny * u_ip12p[0]) * buoyancy);

    for (int n = 0; n < 9; n++)
    {
        Riemann_flux[idx * WSIZE + n] = 0.5f * (fu_ip12p[n] + fu_ip12n[n] - alpha * (u_ip12p[n] - u_ip12n[n]));
    }
#endif

    END_CALCULATE3D()

    CALCULATE3D(i, j, k, HALO, _nx, _ny - HALO, _ny, HALO, _nz)

#ifdef PML
    pml_beta_x = pml_beta.x[i];
    pml_beta_y = pml_beta.y[j];
    pml_beta_z = pml_beta.z[k];
#endif

    nx = CJM[idx * CJMSIZE + 3] + 1e-6;
    ny = CJM[idx * CJMSIZE + 4] + 1e-6;
    nz = CJM[idx * CJMSIZE + 5] + 1e-6;
    sum_square = nx * nx + ny * ny + nz * nz;
    jac = 1.0 / CJM[idx * CJMSIZE + 9];

    mu = CJM[idx * CJMSIZE + 10];
    lambda = CJM[idx * CJMSIZE + 11];
    buoyancy = CJM[idx * CJMSIZE + 12];
    rho = 1.0 / buoyancy;

    // Store original variables
    for (int n = 0; n < 6; n++)
    {
        u[n][0] = W[INDEX(i, j - 2 + n, k) * WSIZE + 0];
        u[n][1] = W[INDEX(i, j - 2 + n, k) * WSIZE + 1];
        u[n][2] = W[INDEX(i, j - 2 + n, k) * WSIZE + 2];
        u[n][3] = W[INDEX(i, j - 2 + n, k) * WSIZE + 3];
        u[n][4] = W[INDEX(i, j - 2 + n, k) * WSIZE + 4];
        u[n][5] = W[INDEX(i, j - 2 + n, k) * WSIZE + 5];
        u[n][6] = W[INDEX(i, j - 2 + n, k) * WSIZE + 6];
        u[n][7] = W[INDEX(i, j - 2 + n, k) * WSIZE + 7];
        u[n][8] = W[INDEX(i, j - 2 + n, k) * WSIZE + 8];
    }

    // Transform to characteristic domain
    for (int n = 0; n < 6; n++)
    {
        v[n][0] = -(lambda * nx * (ny * ny * ny) * u[n][7] - 2 * mu * (nx * nx * nx * nx) * u[n][6] - lambda * (nx * nx * nx * nx) * u[n][6] + lambda * (nx * nx * nx) * ny * u[n][7] + lambda * nx * (nz * nz * nz) * u[n][8] + lambda * (nx * nx * nx) * nz * u[n][8] + 2 * lambda * ny * (nz * nz * nz) * u[n][3] + 2 * lambda * (ny * ny * ny) * nz * u[n][3] + 4 * lambda * ny * (nz * nz * nz) * u[n][4] + 4 * lambda * (ny * ny * ny) * nz * u[n][5] + 2 * mu * nx * (ny * ny * ny) * u[n][7] + 2 * mu * (nx * nx * nx) * ny * u[n][7] + 2 * mu * nx * (nz * nz * nz) * u[n][8] + 2 * mu * (nx * nx * nx) * nz * u[n][8] + 4 * mu * ny * (nz * nz * nz) * u[n][4] + 4 * mu * (ny * ny * ny) * nz * u[n][5] - lambda * (nx * nx) * (ny * ny) * u[n][6] - lambda * (nx * nx) * (nz * nz) * u[n][6] - 4 * lambda * (ny * ny) * (nz * nz) * u[n][6] - 2 * mu * (nx * nx) * (ny * ny) * u[n][6] - 2 * mu * (nx * nx) * (nz * nz) * u[n][6] - 4 * mu * (ny * ny) * (nz * nz) * u[n][6] - 2 * lambda * (nx * nx) * ny * nz * u[n][3] + 4 * lambda * (nx * nx) * ny * nz * u[n][4] + 4 * lambda * (nx * nx) * ny * nz * u[n][5] - 3 * lambda * nx * ny * (nz * nz) * u[n][7] - 3 * lambda * nx * (ny * ny) * nz * u[n][8] - 4 * mu * (nx * nx) * ny * nz * u[n][3] + 4 * mu * (nx * nx) * ny * nz * u[n][4] + 4 * mu * (nx * nx) * ny * nz * u[n][5] - 2 * mu * nx * ny * (nz * nz) * u[n][7] - 2 * mu * nx * (ny * ny) * nz * u[n][8]) / ((lambda + 2 * mu) * (sum_square * sum_square));
        v[n][1] = -(lambda * nx * (ny * ny * ny) * u[n][6] - 2 * mu * (ny * ny * ny * ny) * u[n][7] - lambda * (ny * ny * ny * ny) * u[n][7] + lambda * (nx * nx * nx) * ny * u[n][6] + 4 * lambda * nx * (nz * nz * nz) * u[n][3] + 2 * lambda * nx * (nz * nz * nz) * u[n][4] + 2 * lambda * (nx * nx * nx) * nz * u[n][4] + 4 * lambda * (nx * nx * nx) * nz * u[n][5] + lambda * ny * (nz * nz * nz) * u[n][8] + lambda * (ny * ny * ny) * nz * u[n][8] + 2 * mu * nx * (ny * ny * ny) * u[n][6] + 2 * mu * (nx * nx * nx) * ny * u[n][6] + 4 * mu * nx * (nz * nz * nz) * u[n][3] + 4 * mu * (nx * nx * nx) * nz * u[n][5] + 2 * mu * ny * (nz * nz * nz) * u[n][8] + 2 * mu * (ny * ny * ny) * nz * u[n][8] - lambda * (nx * nx) * (ny * ny) * u[n][7] - 4 * lambda * (nx * nx) * (nz * nz) * u[n][7] - lambda * (ny * ny) * (nz * nz) * u[n][7] - 2 * mu * (nx * nx) * (ny * ny) * u[n][7] - 4 * mu * (nx * nx) * (nz * nz) * u[n][7] - 2 * mu * (ny * ny) * (nz * nz) * u[n][7] + 4 * lambda * nx * (ny * ny) * nz * u[n][3] - 2 * lambda * nx * (ny * ny) * nz * u[n][4] + 4 * lambda * nx * (ny * ny) * nz * u[n][5] - 3 * lambda * nx * ny * (nz * nz) * u[n][6] - 3 * lambda * (nx * nx) * ny * nz * u[n][8] + 4 * mu * nx * (ny * ny) * nz * u[n][3] - 4 * mu * nx * (ny * ny) * nz * u[n][4] + 4 * mu * nx * (ny * ny) * nz * u[n][5] - 2 * mu * nx * ny * (nz * nz) * u[n][6] - 2 * mu * (nx * nx) * ny * nz * u[n][8]) / ((lambda + 2 * mu) * (sum_square * sum_square));
        v[n][2] = -(4 * lambda * nx * (ny * ny * ny) * u[n][3] - 2 * mu * (nz * nz * nz * nz) * u[n][8] - lambda * (nz * nz * nz * nz) * u[n][8] + 4 * lambda * (nx * nx * nx) * ny * u[n][4] + 2 * lambda * nx * (ny * ny * ny) * u[n][5] + 2 * lambda * (nx * nx * nx) * ny * u[n][5] + lambda * nx * (nz * nz * nz) * u[n][6] + lambda * (nx * nx * nx) * nz * u[n][6] + lambda * ny * (nz * nz * nz) * u[n][7] + lambda * (ny * ny * ny) * nz * u[n][7] + 4 * mu * nx * (ny * ny * ny) * u[n][3] + 4 * mu * (nx * nx * nx) * ny * u[n][4] + 2 * mu * nx * (nz * nz * nz) * u[n][6] + 2 * mu * (nx * nx * nx) * nz * u[n][6] + 2 * mu * ny * (nz * nz * nz) * u[n][7] + 2 * mu * (ny * ny * ny) * nz * u[n][7] - 4 * lambda * (nx * nx) * (ny * ny) * u[n][8] - lambda * (nx * nx) * (nz * nz) * u[n][8] - lambda * (ny * ny) * (nz * nz) * u[n][8] - 4 * mu * (nx * nx) * (ny * ny) * u[n][8] - 2 * mu * (nx * nx) * (nz * nz) * u[n][8] - 2 * mu * (ny * ny) * (nz * nz) * u[n][8] + 4 * lambda * nx * ny * (nz * nz) * u[n][3] + 4 * lambda * nx * ny * (nz * nz) * u[n][4] - 2 * lambda * nx * ny * (nz * nz) * u[n][5] - 3 * lambda * nx * (ny * ny) * nz * u[n][6] - 3 * lambda * (nx * nx) * ny * nz * u[n][7] + 4 * mu * nx * ny * (nz * nz) * u[n][3] + 4 * mu * nx * ny * (nz * nz) * u[n][4] - 4 * mu * nx * ny * (nz * nz) * u[n][5] - 2 * mu * nx * (ny * ny) * nz * u[n][6] - 2 * mu * (nx * nx) * ny * nz * u[n][7]) / ((lambda + 2 * mu) * (sum_square * sum_square));
        v[n][3] = ((nx * nx * nx) * u[n][2] * sqrt(mu * rho * sum_square) + (nz * nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) + nx * (ny * ny) * u[n][2] * sqrt(mu * rho * sum_square) - (nx * nx) * nz * u[n][0] * sqrt(mu * rho * sum_square) - nx * (nz * nz) * u[n][2] * sqrt(mu * rho * sum_square) + (ny * ny) * nz * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][7] + mu * (nz * nz * nz * nz) * rho * u[n][7] + mu * (nx * nx) * (ny * ny) * rho * u[n][7] - 2 * mu * (nx * nx) * (nz * nz) * rho * u[n][7] + mu * (ny * ny) * (nz * nz) * rho * u[n][7] - 2 * nx * ny * nz * u[n][1] * sqrt(mu * rho * sum_square) + mu * nx * (ny * ny * ny) * rho * u[n][6] + mu * (nx * nx * nx) * ny * rho * u[n][6] + 2 * mu * nx * (nz * nz * nz) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * nz * rho * u[n][3] - 2 * mu * nx * (nz * nz * nz) * rho * u[n][5] + 2 * mu * (nx * nx * nx) * nz * rho * u[n][5] + mu * ny * (nz * nz * nz) * rho * u[n][8] + mu * (ny * ny * ny) * nz * rho * u[n][8] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][3] - 4 * mu * nx * (ny * ny) * nz * rho * u[n][4] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][5] - 3 * mu * nx * ny * (nz * nz) * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][8]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][4] = ((nx * nx * nx) * u[n][1] * sqrt(mu * rho * sum_square) + (ny * ny * ny) * u[n][0] * sqrt(mu * rho * sum_square) - (nx * nx) * ny * u[n][0] * sqrt(mu * rho * sum_square) - nx * (ny * ny) * u[n][1] * sqrt(mu * rho * sum_square) + nx * (nz * nz) * u[n][1] * sqrt(mu * rho * sum_square) + ny * (nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][8] + mu * (ny * ny * ny * ny) * rho * u[n][8] - 2 * mu * (nx * nx) * (ny * ny) * rho * u[n][8] + mu * (nx * nx) * (nz * nz) * rho * u[n][8] + mu * (ny * ny) * (nz * nz) * rho * u[n][8] - 2 * nx * ny * nz * u[n][2] * sqrt(mu * rho * sum_square) + 2 * mu * nx * (ny * ny * ny) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * ny * rho * u[n][3] - 2 * mu * nx * (ny * ny * ny) * rho * u[n][4] + 2 * mu * (nx * nx * nx) * ny * rho * u[n][4] + mu * nx * (nz * nz * nz) * rho * u[n][6] + mu * (nx * nx * nx) * nz * rho * u[n][6] + mu * ny * (nz * nz * nz) * rho * u[n][7] + mu * (ny * ny * ny) * nz * rho * u[n][7] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][3] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][4] - 4 * mu * nx * ny * (nz * nz) * rho * u[n][5] - 3 * mu * nx * (ny * ny) * nz * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][7]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][5] = (rho * rho * (lambda + 2 * mu) * (lambda * (nx * nx * nx * nx) * ny * u[n][0] + lambda * nx * (ny * ny * ny * ny) * u[n][1] + 2 * mu * (nx * nx * nx * nx) * ny * u[n][0] + 2 * mu * nx * (ny * ny * ny * ny) * u[n][1] + lambda * (nx * nx) * (ny * ny * ny) * u[n][0] + lambda * (nx * nx * nx) * (ny * ny) * u[n][1] + 2 * mu * (nx * nx) * (ny * ny * ny) * u[n][0] + 2 * mu * (nx * nx * nx) * (ny * ny) * u[n][1] + lambda * nx * (ny * ny * ny) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx * nx) * ny * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * (ny * ny * ny) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx * nx) * ny * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx) * ny * (nz * nz) * u[n][0] + lambda * nx * (ny * ny) * (nz * nz) * u[n][1] + 2 * mu * (nx * nx) * ny * (nz * nz) * u[n][0] + 2 * mu * nx * (ny * ny) * (nz * nz) * u[n][1] + 2 * mu * (nx * nx) * (ny * ny) * u[n][8] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz * nz) * u[n][2] + lambda * nx * (ny * ny * ny) * nz * u[n][2] + lambda * (nx * nx * nx) * ny * nz * u[n][2] + 2 * mu * nx * ny * (nz * nz * nz) * u[n][2] + 2 * mu * nx * (ny * ny * ny) * nz * u[n][2] + 2 * mu * (nx * nx * nx) * ny * nz * u[n][2] + lambda * nx * ny * (nz * nz) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * nx * (ny * ny) * nz * u[n][6] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * (nx * nx) * ny * nz * u[n][7] * sqrt(rho * (lambda + 2 * mu) * sum_square))) / ((rho * (lambda + 2 * mu) * sum_square) * (rho * (lambda + 2 * mu) * sum_square) * sqrt(rho * (lambda + 2 * mu) * sum_square));
        v[n][6] = ((nx * nx) * nz * u[n][0] * sqrt(mu * rho * sum_square) - (nz * nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) - nx * (ny * ny) * u[n][2] * sqrt(mu * rho * sum_square) - (nx * nx * nx) * u[n][2] * sqrt(mu * rho * sum_square) + nx * (nz * nz) * u[n][2] * sqrt(mu * rho * sum_square) - (ny * ny) * nz * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][7] + mu * (nz * nz * nz * nz) * rho * u[n][7] + mu * (nx * nx) * (ny * ny) * rho * u[n][7] - 2 * mu * (nx * nx) * (nz * nz) * rho * u[n][7] + mu * (ny * ny) * (nz * nz) * rho * u[n][7] + 2 * nx * ny * nz * u[n][1] * sqrt(mu * rho * sum_square) + mu * nx * (ny * ny * ny) * rho * u[n][6] + mu * (nx * nx * nx) * ny * rho * u[n][6] + 2 * mu * nx * (nz * nz * nz) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * nz * rho * u[n][3] - 2 * mu * nx * (nz * nz * nz) * rho * u[n][5] + 2 * mu * (nx * nx * nx) * nz * rho * u[n][5] + mu * ny * (nz * nz * nz) * rho * u[n][8] + mu * (ny * ny * ny) * nz * rho * u[n][8] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][3] - 4 * mu * nx * (ny * ny) * nz * rho * u[n][4] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][5] - 3 * mu * nx * ny * (nz * nz) * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][8]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][7] = ((nx * nx) * ny * u[n][0] * sqrt(mu * rho * sum_square) - (ny * ny * ny) * u[n][0] * sqrt(mu * rho * sum_square) - (nx * nx * nx) * u[n][1] * sqrt(mu * rho * sum_square) + nx * (ny * ny) * u[n][1] * sqrt(mu * rho * sum_square) - nx * (nz * nz) * u[n][1] * sqrt(mu * rho * sum_square) - ny * (nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][8] + mu * (ny * ny * ny * ny) * rho * u[n][8] - 2 * mu * (nx * nx) * (ny * ny) * rho * u[n][8] + mu * (nx * nx) * (nz * nz) * rho * u[n][8] + mu * (ny * ny) * (nz * nz) * rho * u[n][8] + 2 * nx * ny * nz * u[n][2] * sqrt(mu * rho * sum_square) + 2 * mu * nx * (ny * ny * ny) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * ny * rho * u[n][3] - 2 * mu * nx * (ny * ny * ny) * rho * u[n][4] + 2 * mu * (nx * nx * nx) * ny * rho * u[n][4] + mu * nx * (nz * nz * nz) * rho * u[n][6] + mu * (nx * nx * nx) * nz * rho * u[n][6] + mu * ny * (nz * nz * nz) * rho * u[n][7] + mu * (ny * ny * ny) * nz * rho * u[n][7] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][3] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][4] - 4 * mu * nx * ny * (nz * nz) * rho * u[n][5] - 3 * mu * nx * (ny * ny) * nz * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][7]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][8] = -(rho * rho * (lambda + 2 * mu) * (lambda * (nx * nx * nx * nx) * ny * u[n][0] + lambda * nx * (ny * ny * ny * ny) * u[n][1] + 2 * mu * (nx * nx * nx * nx) * ny * u[n][0] + 2 * mu * nx * (ny * ny * ny * ny) * u[n][1] + lambda * (nx * nx) * (ny * ny * ny) * u[n][0] + lambda * (nx * nx * nx) * (ny * ny) * u[n][1] + 2 * mu * (nx * nx) * (ny * ny * ny) * u[n][0] + 2 * mu * (nx * nx * nx) * (ny * ny) * u[n][1] - lambda * nx * (ny * ny * ny) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * (nx * nx * nx) * ny * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * (ny * ny * ny) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * (nx * nx * nx) * ny * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx) * ny * (nz * nz) * u[n][0] + lambda * nx * (ny * ny) * (nz * nz) * u[n][1] + 2 * mu * (nx * nx) * ny * (nz * nz) * u[n][0] + 2 * mu * nx * (ny * ny) * (nz * nz) * u[n][1] - 2 * mu * (nx * nx) * (ny * ny) * u[n][8] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz * nz) * u[n][2] + lambda * nx * (ny * ny * ny) * nz * u[n][2] + lambda * (nx * nx * nx) * ny * nz * u[n][2] + 2 * mu * nx * ny * (nz * nz * nz) * u[n][2] + 2 * mu * nx * (ny * ny * ny) * nz * u[n][2] + 2 * mu * (nx * nx * nx) * ny * nz * u[n][2] - lambda * nx * ny * (nz * nz) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * ny * (nz * nz) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * nx * (ny * ny) * nz * u[n][6] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * (nx * nx) * ny * nz * u[n][7] * sqrt(rho * (lambda + 2 * mu) * sum_square))) / ((rho * (lambda + 2 * mu) * sum_square) * (rho * (lambda + 2 * mu) * sum_square) * sqrt(rho * (lambda + 2 * mu) * sum_square));
    }

    // Shock capture interpolation
    for (int n = 0; n < 9; n++)
    {
        v_ip12n[n] = WENO5_interpolation(v[0][n], v[1][n], v[2][n], v[3][n], v[4][n]);
        v_ip12p[n] = WENO5_interpolation(v[5][n], v[4][n], v[3][n], v[2][n], v[1][n]);
    }

    // Transform back to physical domain
    u_ip12n[0] = (v_ip12n[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) - (v_ip12n[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) + (ny * v_ip12n[4] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (ny * v_ip12n[7] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nz * v_ip12n[3] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (nz * v_ip12n[6] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz));
    u_ip12n[1] = (v_ip12n[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12n[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12n[4] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12n[7] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12n[3] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12n[6] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12n[2] = (nz * v_ip12n[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (nz * v_ip12n[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (v_ip12n[3] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12n[6] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12n[4] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12n[7] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12n[3] = (nx * v_ip12n[5]) / (2 * ny) + (nx * v_ip12n[8]) / (2 * ny) + (nx * ny * v_ip12n[4]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * ny * v_ip12n[7]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12n[3]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12n[6]) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (v_ip12n[2] * (2 * lambda * (ny * ny) - lambda * (nx * nx) + 2 * mu * (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12n[1] * (2 * lambda * (nz * nz) - lambda * (nx * nx) + 2 * mu * (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12n[0] * ((ny * ny) + (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu));
    u_ip12n[4] = (ny * v_ip12n[5]) / (2 * nx) + (ny * v_ip12n[8]) / (2 * nx) - ((ny * ny) * nz * v_ip12n[3]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - ((ny * ny) * nz * v_ip12n[6]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12n[4] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12n[7] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12n[2] * (2 * lambda * (nx * nx) - lambda * (ny * ny) + 2 * mu * (nx * nx))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12n[0] * (2 * lambda * (nz * nz) - lambda * (ny * ny) + 2 * mu * (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12n[1] * ((nx * nx) + (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu));
    u_ip12n[5] = ((nz * nz) * v_ip12n[5]) / (2 * nx * ny) + ((nz * nz) * v_ip12n[8]) / (2 * nx * ny) - (ny * (nz * nz) * v_ip12n[4]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * (nz * nz) * v_ip12n[7]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12n[3] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12n[6] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12n[1] * (2 * lambda * (nx * nx) - lambda * (nz * nz) + 2 * mu * (nx * nx))) / (2 * nx * nz * (3 * lambda + 2 * mu)) - (v_ip12n[0] * (2 * lambda * (ny * ny) - lambda * (nz * nz) + 2 * mu * (ny * ny))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12n[2] * ((nx * nx) + (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu));
    u_ip12n[6] = -((nx * nx * nx) * v_ip12n[0] - (ny * ny * ny) * v_ip12n[3] - (ny * ny * ny) * v_ip12n[6] - (nz * nz * nz) * v_ip12n[4] - (nz * nz * nz) * v_ip12n[5] - (nz * nz * nz) * v_ip12n[7] - (nz * nz * nz) * v_ip12n[8] - nx * (ny * ny) * v_ip12n[0] + (nx * nx) * ny * v_ip12n[3] + (nx * nx) * ny * v_ip12n[6] - nx * (nz * nz) * v_ip12n[0] + (nx * nx) * nz * v_ip12n[4] + (nx * nx) * nz * v_ip12n[5] + (nx * nx) * nz * v_ip12n[7] + (nx * nx) * nz * v_ip12n[8] + ny * (nz * nz) * v_ip12n[3] + (ny * ny) * nz * v_ip12n[4] - (ny * ny) * nz * v_ip12n[5] + ny * (nz * nz) * v_ip12n[6] + (ny * ny) * nz * v_ip12n[7] - (ny * ny) * nz * v_ip12n[8]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12n[7] = v_ip12n[1] + v_ip12n[3] + v_ip12n[6] + (nz * v_ip12n[5]) / ny + (nz * v_ip12n[8]) / ny;
    u_ip12n[8] = v_ip12n[2] + v_ip12n[4] + v_ip12n[5] + v_ip12n[7] + v_ip12n[8];

    u_ip12p[0] = (v_ip12p[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) - (v_ip12p[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) + (ny * v_ip12p[4] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (ny * v_ip12p[7] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nz * v_ip12p[3] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (nz * v_ip12p[6] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz));
    u_ip12p[1] = (v_ip12p[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12p[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12p[4] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12p[7] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12p[3] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12p[6] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12p[2] = (nz * v_ip12p[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (nz * v_ip12p[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (v_ip12p[3] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12p[6] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12p[4] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12p[7] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12p[3] = (nx * v_ip12p[5]) / (2 * ny) + (nx * v_ip12p[8]) / (2 * ny) + (nx * ny * v_ip12p[4]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * ny * v_ip12p[7]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12p[3]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12p[6]) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (v_ip12p[2] * (2 * lambda * (ny * ny) - lambda * (nx * nx) + 2 * mu * (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12p[1] * (2 * lambda * (nz * nz) - lambda * (nx * nx) + 2 * mu * (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12p[0] * ((ny * ny) + (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu));
    u_ip12p[4] = (ny * v_ip12p[5]) / (2 * nx) + (ny * v_ip12p[8]) / (2 * nx) - ((ny * ny) * nz * v_ip12p[3]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - ((ny * ny) * nz * v_ip12p[6]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12p[4] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12p[7] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12p[2] * (2 * lambda * (nx * nx) - lambda * (ny * ny) + 2 * mu * (nx * nx))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12p[0] * (2 * lambda * (nz * nz) - lambda * (ny * ny) + 2 * mu * (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12p[1] * ((nx * nx) + (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu));
    u_ip12p[5] = ((nz * nz) * v_ip12p[5]) / (2 * nx * ny) + ((nz * nz) * v_ip12p[8]) / (2 * nx * ny) - (ny * (nz * nz) * v_ip12p[4]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * (nz * nz) * v_ip12p[7]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12p[3] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12p[6] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12p[1] * (2 * lambda * (nx * nx) - lambda * (nz * nz) + 2 * mu * (nx * nx))) / (2 * nx * nz * (3 * lambda + 2 * mu)) - (v_ip12p[0] * (2 * lambda * (ny * ny) - lambda * (nz * nz) + 2 * mu * (ny * ny))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12p[2] * ((nx * nx) + (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu));
    u_ip12p[6] = -((nx * nx * nx) * v_ip12p[0] - (ny * ny * ny) * v_ip12p[3] - (ny * ny * ny) * v_ip12p[6] - (nz * nz * nz) * v_ip12p[4] - (nz * nz * nz) * v_ip12p[5] - (nz * nz * nz) * v_ip12p[7] - (nz * nz * nz) * v_ip12p[8] - nx * (ny * ny) * v_ip12p[0] + (nx * nx) * ny * v_ip12p[3] + (nx * nx) * ny * v_ip12p[6] - nx * (nz * nz) * v_ip12p[0] + (nx * nx) * nz * v_ip12p[4] + (nx * nx) * nz * v_ip12p[5] + (nx * nx) * nz * v_ip12p[7] + (nx * nx) * nz * v_ip12p[8] + ny * (nz * nz) * v_ip12p[3] + (ny * ny) * nz * v_ip12p[4] - (ny * ny) * nz * v_ip12p[5] + ny * (nz * nz) * v_ip12p[6] + (ny * ny) * nz * v_ip12p[7] - (ny * ny) * nz * v_ip12p[8]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12p[7] = v_ip12p[1] + v_ip12p[3] + v_ip12p[6] + (nz * v_ip12p[5]) / ny + (nz * v_ip12p[8]) / ny;
    u_ip12p[8] = v_ip12p[2] + v_ip12p[4] + v_ip12p[5] + v_ip12p[7] + v_ip12p[8];

#ifdef LF
    // Riemann solver: Lax-Friedrichs
    fu_ip12n[0] = -(nx * (lambda * u_ip12n[4] + lambda * u_ip12n[5] + u_ip12n[3] * (lambda + 2 * mu)) + mu * ny * u_ip12n[8] + mu * nz * u_ip12n[7]);
    fu_ip12n[1] = -(ny * (lambda * u_ip12n[3] + lambda * u_ip12n[5] + u_ip12n[4] * (lambda + 2 * mu)) + mu * nx * u_ip12n[8] + mu * nz * u_ip12n[6]);
    fu_ip12n[2] = -(nz * (lambda * u_ip12n[3] + lambda * u_ip12n[4] + u_ip12n[5] * (lambda + 2 * mu)) + mu * nx * u_ip12n[7] + mu * ny * u_ip12n[6]);
    fu_ip12n[3] = -((nx * u_ip12n[0]) * buoyancy);
    fu_ip12n[4] = -((ny * u_ip12n[1]) * buoyancy);
    fu_ip12n[5] = -((nz * u_ip12n[2]) * buoyancy);
    fu_ip12n[6] = -((ny * u_ip12n[2]) * buoyancy + (nz * u_ip12n[1]) * buoyancy);
    fu_ip12n[7] = -((nx * u_ip12n[2]) * buoyancy + (nz * u_ip12n[0]) * buoyancy);
    fu_ip12n[8] = -((nx * u_ip12n[1]) * buoyancy + (ny * u_ip12n[0]) * buoyancy);

    fu_ip12p[0] = -(nx * (lambda * u_ip12p[4] + lambda * u_ip12p[5] + u_ip12p[3] * (lambda + 2 * mu)) + mu * ny * u_ip12p[8] + mu * nz * u_ip12p[7]);
    fu_ip12p[1] = -(ny * (lambda * u_ip12p[3] + lambda * u_ip12p[5] + u_ip12p[4] * (lambda + 2 * mu)) + mu * nx * u_ip12p[8] + mu * nz * u_ip12p[6]);
    fu_ip12p[2] = -(nz * (lambda * u_ip12p[3] + lambda * u_ip12p[4] + u_ip12p[5] * (lambda + 2 * mu)) + mu * nx * u_ip12p[7] + mu * ny * u_ip12p[6]);
    fu_ip12p[3] = -((nx * u_ip12p[0]) * buoyancy);
    fu_ip12p[4] = -((ny * u_ip12p[1]) * buoyancy);
    fu_ip12p[5] = -((nz * u_ip12p[2]) * buoyancy);
    fu_ip12p[6] = -((ny * u_ip12p[2]) * buoyancy + (nz * u_ip12p[1]) * buoyancy);
    fu_ip12p[7] = -((nx * u_ip12p[2]) * buoyancy + (nz * u_ip12p[0]) * buoyancy);
    fu_ip12p[8] = -((nx * u_ip12p[1]) * buoyancy + (ny * u_ip12p[0]) * buoyancy);

    for (int n = 0; n < 9; n++)
    {
        Riemann_flux[idx * WSIZE + n] = 0.5f * (fu_ip12p[n] + fu_ip12n[n] - alpha * (u_ip12p[n] - u_ip12n[n]));
    }
#endif

    END_CALCULATE3D()
}

__GLOBAL__
void wave_deriv_alternative_flux_FD_char_z(FLOAT *Riemann_flux, FLOAT *h_W, FLOAT *W, FLOAT *CJM,
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
    float rho;

    float nx;
    float ny;
    float nz;
    float jac;
    float sum_square;

    long long idx_n3, idx_n2, idx_n1, idx, idx_p1, idx_p2, idx_p3;

    float u_ip12n[9], u_ip12p[9];

    float u[6][9], v[6][9], v_ip12n[9], v_ip12p[9];

#ifdef LF
    float fu_ip12n[9], fu_ip12p[9];
#endif

    // * Z direction
    //     CALCULATE3D(i, j, k, HALO, _nx, HALO, _ny, HALO - 1, HALO + 3)

    // #ifdef PML
    //     pml_beta_x = pml_beta.x[i];
    //     pml_beta_y = pml_beta.y[j];
    //     pml_beta_z = pml_beta.z[k];
    // #endif

    //     nx = CJM[idx * CJMSIZE + 6] + 1e-6;
    //     ny = CJM[idx * CJMSIZE + 7] + 1e-6;
    //     nz = CJM[idx * CJMSIZE + 8] + 1e-6;
    //     sum_square = nx * nx + ny * ny + nz * nz;
    //     jac = 1.0 / CJM[idx * CJMSIZE + 9];

    //     mu = CJM[idx * CJMSIZE + 10];
    //     lambda = CJM[idx * CJMSIZE + 11];
    //     buoyancy = CJM[idx * CJMSIZE + 12];
    //     rho = 1.0 / buoyancy;

    //     // Store original variables
    //     for (int n = 0; n < 6; n++)
    //     {
    //         u[n][0] = W[INDEX(i, j, k - 2 + n) * WSIZE + 0];
    //         u[n][1] = W[INDEX(i, j, k - 2 + n) * WSIZE + 1];
    //         u[n][2] = W[INDEX(i, j, k - 2 + n) * WSIZE + 2];
    //         u[n][3] = W[INDEX(i, j, k - 2 + n) * WSIZE + 3];
    //         u[n][4] = W[INDEX(i, j, k - 2 + n) * WSIZE + 4];
    //         u[n][5] = W[INDEX(i, j, k - 2 + n) * WSIZE + 5];
    //         u[n][6] = W[INDEX(i, j, k - 2 + n) * WSIZE + 6];
    //         u[n][7] = W[INDEX(i, j, k - 2 + n) * WSIZE + 7];
    //         u[n][8] = W[INDEX(i, j, k - 2 + n) * WSIZE + 8];
    //     }

    //     // Transform to characteristic domain
    //     for (int n = 0; n < 6; n++)
    //     {
    //         v[n][0] = -(lambda * nx * (ny * ny * ny) * u[n][7] - 2 * mu * (nx * nx * nx * nx) * u[n][6] - lambda * (nx * nx * nx * nx) * u[n][6] + lambda * (nx * nx * nx) * ny * u[n][7] + lambda * nx * (nz * nz * nz) * u[n][8] + lambda * (nx * nx * nx) * nz * u[n][8] + 2 * lambda * ny * (nz * nz * nz) * u[n][3] + 2 * lambda * (ny * ny * ny) * nz * u[n][3] + 4 * lambda * ny * (nz * nz * nz) * u[n][4] + 4 * lambda * (ny * ny * ny) * nz * u[n][5] + 2 * mu * nx * (ny * ny * ny) * u[n][7] + 2 * mu * (nx * nx * nx) * ny * u[n][7] + 2 * mu * nx * (nz * nz * nz) * u[n][8] + 2 * mu * (nx * nx * nx) * nz * u[n][8] + 4 * mu * ny * (nz * nz * nz) * u[n][4] + 4 * mu * (ny * ny * ny) * nz * u[n][5] - lambda * (nx * nx) * (ny * ny) * u[n][6] - lambda * (nx * nx) * (nz * nz) * u[n][6] - 4 * lambda * (ny * ny) * (nz * nz) * u[n][6] - 2 * mu * (nx * nx) * (ny * ny) * u[n][6] - 2 * mu * (nx * nx) * (nz * nz) * u[n][6] - 4 * mu * (ny * ny) * (nz * nz) * u[n][6] - 2 * lambda * (nx * nx) * ny * nz * u[n][3] + 4 * lambda * (nx * nx) * ny * nz * u[n][4] + 4 * lambda * (nx * nx) * ny * nz * u[n][5] - 3 * lambda * nx * ny * (nz * nz) * u[n][7] - 3 * lambda * nx * (ny * ny) * nz * u[n][8] - 4 * mu * (nx * nx) * ny * nz * u[n][3] + 4 * mu * (nx * nx) * ny * nz * u[n][4] + 4 * mu * (nx * nx) * ny * nz * u[n][5] - 2 * mu * nx * ny * (nz * nz) * u[n][7] - 2 * mu * nx * (ny * ny) * nz * u[n][8]) / ((lambda + 2 * mu) * (sum_square * sum_square));
    //         v[n][1] = -(lambda * nx * (ny * ny * ny) * u[n][6] - 2 * mu * (ny * ny * ny * ny) * u[n][7] - lambda * (ny * ny * ny * ny) * u[n][7] + lambda * (nx * nx * nx) * ny * u[n][6] + 4 * lambda * nx * (nz * nz * nz) * u[n][3] + 2 * lambda * nx * (nz * nz * nz) * u[n][4] + 2 * lambda * (nx * nx * nx) * nz * u[n][4] + 4 * lambda * (nx * nx * nx) * nz * u[n][5] + lambda * ny * (nz * nz * nz) * u[n][8] + lambda * (ny * ny * ny) * nz * u[n][8] + 2 * mu * nx * (ny * ny * ny) * u[n][6] + 2 * mu * (nx * nx * nx) * ny * u[n][6] + 4 * mu * nx * (nz * nz * nz) * u[n][3] + 4 * mu * (nx * nx * nx) * nz * u[n][5] + 2 * mu * ny * (nz * nz * nz) * u[n][8] + 2 * mu * (ny * ny * ny) * nz * u[n][8] - lambda * (nx * nx) * (ny * ny) * u[n][7] - 4 * lambda * (nx * nx) * (nz * nz) * u[n][7] - lambda * (ny * ny) * (nz * nz) * u[n][7] - 2 * mu * (nx * nx) * (ny * ny) * u[n][7] - 4 * mu * (nx * nx) * (nz * nz) * u[n][7] - 2 * mu * (ny * ny) * (nz * nz) * u[n][7] + 4 * lambda * nx * (ny * ny) * nz * u[n][3] - 2 * lambda * nx * (ny * ny) * nz * u[n][4] + 4 * lambda * nx * (ny * ny) * nz * u[n][5] - 3 * lambda * nx * ny * (nz * nz) * u[n][6] - 3 * lambda * (nx * nx) * ny * nz * u[n][8] + 4 * mu * nx * (ny * ny) * nz * u[n][3] - 4 * mu * nx * (ny * ny) * nz * u[n][4] + 4 * mu * nx * (ny * ny) * nz * u[n][5] - 2 * mu * nx * ny * (nz * nz) * u[n][6] - 2 * mu * (nx * nx) * ny * nz * u[n][8]) / ((lambda + 2 * mu) * (sum_square * sum_square));
    //         v[n][2] = -(4 * lambda * nx * (ny * ny * ny) * u[n][3] - 2 * mu * (nz * nz * nz * nz) * u[n][8] - lambda * (nz * nz * nz * nz) * u[n][8] + 4 * lambda * (nx * nx * nx) * ny * u[n][4] + 2 * lambda * nx * (ny * ny * ny) * u[n][5] + 2 * lambda * (nx * nx * nx) * ny * u[n][5] + lambda * nx * (nz * nz * nz) * u[n][6] + lambda * (nx * nx * nx) * nz * u[n][6] + lambda * ny * (nz * nz * nz) * u[n][7] + lambda * (ny * ny * ny) * nz * u[n][7] + 4 * mu * nx * (ny * ny * ny) * u[n][3] + 4 * mu * (nx * nx * nx) * ny * u[n][4] + 2 * mu * nx * (nz * nz * nz) * u[n][6] + 2 * mu * (nx * nx * nx) * nz * u[n][6] + 2 * mu * ny * (nz * nz * nz) * u[n][7] + 2 * mu * (ny * ny * ny) * nz * u[n][7] - 4 * lambda * (nx * nx) * (ny * ny) * u[n][8] - lambda * (nx * nx) * (nz * nz) * u[n][8] - lambda * (ny * ny) * (nz * nz) * u[n][8] - 4 * mu * (nx * nx) * (ny * ny) * u[n][8] - 2 * mu * (nx * nx) * (nz * nz) * u[n][8] - 2 * mu * (ny * ny) * (nz * nz) * u[n][8] + 4 * lambda * nx * ny * (nz * nz) * u[n][3] + 4 * lambda * nx * ny * (nz * nz) * u[n][4] - 2 * lambda * nx * ny * (nz * nz) * u[n][5] - 3 * lambda * nx * (ny * ny) * nz * u[n][6] - 3 * lambda * (nx * nx) * ny * nz * u[n][7] + 4 * mu * nx * ny * (nz * nz) * u[n][3] + 4 * mu * nx * ny * (nz * nz) * u[n][4] - 4 * mu * nx * ny * (nz * nz) * u[n][5] - 2 * mu * nx * (ny * ny) * nz * u[n][6] - 2 * mu * (nx * nx) * ny * nz * u[n][7]) / ((lambda + 2 * mu) * (sum_square * sum_square));
    //         v[n][3] = ((nx * nx * nx) * u[n][2] * sqrt(mu * rho * sum_square) + (nz * nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) + nx * (ny * ny) * u[n][2] * sqrt(mu * rho * sum_square) - (nx * nx) * nz * u[n][0] * sqrt(mu * rho * sum_square) - nx * (nz * nz) * u[n][2] * sqrt(mu * rho * sum_square) + (ny * ny) * nz * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][7] + mu * (nz * nz * nz * nz) * rho * u[n][7] + mu * (nx * nx) * (ny * ny) * rho * u[n][7] - 2 * mu * (nx * nx) * (nz * nz) * rho * u[n][7] + mu * (ny * ny) * (nz * nz) * rho * u[n][7] - 2 * nx * ny * nz * u[n][1] * sqrt(mu * rho * sum_square) + mu * nx * (ny * ny * ny) * rho * u[n][6] + mu * (nx * nx * nx) * ny * rho * u[n][6] + 2 * mu * nx * (nz * nz * nz) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * nz * rho * u[n][3] - 2 * mu * nx * (nz * nz * nz) * rho * u[n][5] + 2 * mu * (nx * nx * nx) * nz * rho * u[n][5] + mu * ny * (nz * nz * nz) * rho * u[n][8] + mu * (ny * ny * ny) * nz * rho * u[n][8] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][3] - 4 * mu * nx * (ny * ny) * nz * rho * u[n][4] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][5] - 3 * mu * nx * ny * (nz * nz) * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][8]) / (2 * mu * rho * (sum_square * sum_square));
    //         v[n][4] = ((nx * nx * nx) * u[n][1] * sqrt(mu * rho * sum_square) + (ny * ny * ny) * u[n][0] * sqrt(mu * rho * sum_square) - (nx * nx) * ny * u[n][0] * sqrt(mu * rho * sum_square) - nx * (ny * ny) * u[n][1] * sqrt(mu * rho * sum_square) + nx * (nz * nz) * u[n][1] * sqrt(mu * rho * sum_square) + ny * (nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][8] + mu * (ny * ny * ny * ny) * rho * u[n][8] - 2 * mu * (nx * nx) * (ny * ny) * rho * u[n][8] + mu * (nx * nx) * (nz * nz) * rho * u[n][8] + mu * (ny * ny) * (nz * nz) * rho * u[n][8] - 2 * nx * ny * nz * u[n][2] * sqrt(mu * rho * sum_square) + 2 * mu * nx * (ny * ny * ny) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * ny * rho * u[n][3] - 2 * mu * nx * (ny * ny * ny) * rho * u[n][4] + 2 * mu * (nx * nx * nx) * ny * rho * u[n][4] + mu * nx * (nz * nz * nz) * rho * u[n][6] + mu * (nx * nx * nx) * nz * rho * u[n][6] + mu * ny * (nz * nz * nz) * rho * u[n][7] + mu * (ny * ny * ny) * nz * rho * u[n][7] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][3] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][4] - 4 * mu * nx * ny * (nz * nz) * rho * u[n][5] - 3 * mu * nx * (ny * ny) * nz * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][7]) / (2 * mu * rho * (sum_square * sum_square));
    //         v[n][5] = (rho * rho * (lambda + 2 * mu) * (lambda * (nx * nx * nx * nx) * ny * u[n][0] + lambda * nx * (ny * ny * ny * ny) * u[n][1] + 2 * mu * (nx * nx * nx * nx) * ny * u[n][0] + 2 * mu * nx * (ny * ny * ny * ny) * u[n][1] + lambda * (nx * nx) * (ny * ny * ny) * u[n][0] + lambda * (nx * nx * nx) * (ny * ny) * u[n][1] + 2 * mu * (nx * nx) * (ny * ny * ny) * u[n][0] + 2 * mu * (nx * nx * nx) * (ny * ny) * u[n][1] + lambda * nx * (ny * ny * ny) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx * nx) * ny * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * (ny * ny * ny) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx * nx) * ny * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx) * ny * (nz * nz) * u[n][0] + lambda * nx * (ny * ny) * (nz * nz) * u[n][1] + 2 * mu * (nx * nx) * ny * (nz * nz) * u[n][0] + 2 * mu * nx * (ny * ny) * (nz * nz) * u[n][1] + 2 * mu * (nx * nx) * (ny * ny) * u[n][8] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz * nz) * u[n][2] + lambda * nx * (ny * ny * ny) * nz * u[n][2] + lambda * (nx * nx * nx) * ny * nz * u[n][2] + 2 * mu * nx * ny * (nz * nz * nz) * u[n][2] + 2 * mu * nx * (ny * ny * ny) * nz * u[n][2] + 2 * mu * (nx * nx * nx) * ny * nz * u[n][2] + lambda * nx * ny * (nz * nz) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * nx * (ny * ny) * nz * u[n][6] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * (nx * nx) * ny * nz * u[n][7] * sqrt(rho * (lambda + 2 * mu) * sum_square))) / ((rho * (lambda + 2 * mu) * sum_square) * (rho * (lambda + 2 * mu) * sum_square) * sqrt(rho * (lambda + 2 * mu) * sum_square));
    //         v[n][6] = ((nx * nx) * nz * u[n][0] * sqrt(mu * rho * sum_square) - (nz * nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) - nx * (ny * ny) * u[n][2] * sqrt(mu * rho * sum_square) - (nx * nx * nx) * u[n][2] * sqrt(mu * rho * sum_square) + nx * (nz * nz) * u[n][2] * sqrt(mu * rho * sum_square) - (ny * ny) * nz * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][7] + mu * (nz * nz * nz * nz) * rho * u[n][7] + mu * (nx * nx) * (ny * ny) * rho * u[n][7] - 2 * mu * (nx * nx) * (nz * nz) * rho * u[n][7] + mu * (ny * ny) * (nz * nz) * rho * u[n][7] + 2 * nx * ny * nz * u[n][1] * sqrt(mu * rho * sum_square) + mu * nx * (ny * ny * ny) * rho * u[n][6] + mu * (nx * nx * nx) * ny * rho * u[n][6] + 2 * mu * nx * (nz * nz * nz) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * nz * rho * u[n][3] - 2 * mu * nx * (nz * nz * nz) * rho * u[n][5] + 2 * mu * (nx * nx * nx) * nz * rho * u[n][5] + mu * ny * (nz * nz * nz) * rho * u[n][8] + mu * (ny * ny * ny) * nz * rho * u[n][8] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][3] - 4 * mu * nx * (ny * ny) * nz * rho * u[n][4] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][5] - 3 * mu * nx * ny * (nz * nz) * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][8]) / (2 * mu * rho * (sum_square * sum_square));
    //         v[n][7] = ((nx * nx) * ny * u[n][0] * sqrt(mu * rho * sum_square) - (ny * ny * ny) * u[n][0] * sqrt(mu * rho * sum_square) - (nx * nx * nx) * u[n][1] * sqrt(mu * rho * sum_square) + nx * (ny * ny) * u[n][1] * sqrt(mu * rho * sum_square) - nx * (nz * nz) * u[n][1] * sqrt(mu * rho * sum_square) - ny * (nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][8] + mu * (ny * ny * ny * ny) * rho * u[n][8] - 2 * mu * (nx * nx) * (ny * ny) * rho * u[n][8] + mu * (nx * nx) * (nz * nz) * rho * u[n][8] + mu * (ny * ny) * (nz * nz) * rho * u[n][8] + 2 * nx * ny * nz * u[n][2] * sqrt(mu * rho * sum_square) + 2 * mu * nx * (ny * ny * ny) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * ny * rho * u[n][3] - 2 * mu * nx * (ny * ny * ny) * rho * u[n][4] + 2 * mu * (nx * nx * nx) * ny * rho * u[n][4] + mu * nx * (nz * nz * nz) * rho * u[n][6] + mu * (nx * nx * nx) * nz * rho * u[n][6] + mu * ny * (nz * nz * nz) * rho * u[n][7] + mu * (ny * ny * ny) * nz * rho * u[n][7] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][3] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][4] - 4 * mu * nx * ny * (nz * nz) * rho * u[n][5] - 3 * mu * nx * (ny * ny) * nz * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][7]) / (2 * mu * rho * (sum_square * sum_square));
    //         v[n][8] = -(rho * rho * (lambda + 2 * mu) * (lambda * (nx * nx * nx * nx) * ny * u[n][0] + lambda * nx * (ny * ny * ny * ny) * u[n][1] + 2 * mu * (nx * nx * nx * nx) * ny * u[n][0] + 2 * mu * nx * (ny * ny * ny * ny) * u[n][1] + lambda * (nx * nx) * (ny * ny * ny) * u[n][0] + lambda * (nx * nx * nx) * (ny * ny) * u[n][1] + 2 * mu * (nx * nx) * (ny * ny * ny) * u[n][0] + 2 * mu * (nx * nx * nx) * (ny * ny) * u[n][1] - lambda * nx * (ny * ny * ny) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * (nx * nx * nx) * ny * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * (ny * ny * ny) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * (nx * nx * nx) * ny * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx) * ny * (nz * nz) * u[n][0] + lambda * nx * (ny * ny) * (nz * nz) * u[n][1] + 2 * mu * (nx * nx) * ny * (nz * nz) * u[n][0] + 2 * mu * nx * (ny * ny) * (nz * nz) * u[n][1] - 2 * mu * (nx * nx) * (ny * ny) * u[n][8] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz * nz) * u[n][2] + lambda * nx * (ny * ny * ny) * nz * u[n][2] + lambda * (nx * nx * nx) * ny * nz * u[n][2] + 2 * mu * nx * ny * (nz * nz * nz) * u[n][2] + 2 * mu * nx * (ny * ny * ny) * nz * u[n][2] + 2 * mu * (nx * nx * nx) * ny * nz * u[n][2] - lambda * nx * ny * (nz * nz) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * ny * (nz * nz) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * nx * (ny * ny) * nz * u[n][6] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * (nx * nx) * ny * nz * u[n][7] * sqrt(rho * (lambda + 2 * mu) * sum_square))) / ((rho * (lambda + 2 * mu) * sum_square) * (rho * (lambda + 2 * mu) * sum_square) * sqrt(rho * (lambda + 2 * mu) * sum_square));
    //     }

    //     // Shock capture interpolation
    //     for (int n = 0; n < 9; n++)
    //     {
    //         v_ip12n[n] = WENO5_interpolation(v[0][n], v[1][n], v[2][n], v[3][n], v[4][n]);
    //         v_ip12p[n] = WENO5_interpolation(v[5][n], v[4][n], v[3][n], v[2][n], v[1][n]);
    //     }

    //     // Transform back to physical domain
    //     u_ip12n[0] = (v_ip12n[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) - (v_ip12n[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) + (ny * v_ip12n[4] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (ny * v_ip12n[7] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nz * v_ip12n[3] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (nz * v_ip12n[6] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz));
    //     u_ip12n[1] = (v_ip12n[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12n[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12n[4] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12n[7] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12n[3] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12n[6] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    //     u_ip12n[2] = (nz * v_ip12n[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (nz * v_ip12n[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (v_ip12n[3] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12n[6] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12n[4] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12n[7] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    //     u_ip12n[3] = (nx * v_ip12n[5]) / (2 * ny) + (nx * v_ip12n[8]) / (2 * ny) + (nx * ny * v_ip12n[4]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * ny * v_ip12n[7]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12n[3]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12n[6]) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (v_ip12n[2] * (2 * lambda * (ny * ny) - lambda * (nx * nx) + 2 * mu * (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12n[1] * (2 * lambda * (nz * nz) - lambda * (nx * nx) + 2 * mu * (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12n[0] * ((ny * ny) + (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu));
    //     u_ip12n[4] = (ny * v_ip12n[5]) / (2 * nx) + (ny * v_ip12n[8]) / (2 * nx) - ((ny * ny) * nz * v_ip12n[3]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - ((ny * ny) * nz * v_ip12n[6]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12n[4] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12n[7] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12n[2] * (2 * lambda * (nx * nx) - lambda * (ny * ny) + 2 * mu * (nx * nx))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12n[0] * (2 * lambda * (nz * nz) - lambda * (ny * ny) + 2 * mu * (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12n[1] * ((nx * nx) + (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu));
    //     u_ip12n[5] = ((nz * nz) * v_ip12n[5]) / (2 * nx * ny) + ((nz * nz) * v_ip12n[8]) / (2 * nx * ny) - (ny * (nz * nz) * v_ip12n[4]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * (nz * nz) * v_ip12n[7]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12n[3] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12n[6] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12n[1] * (2 * lambda * (nx * nx) - lambda * (nz * nz) + 2 * mu * (nx * nx))) / (2 * nx * nz * (3 * lambda + 2 * mu)) - (v_ip12n[0] * (2 * lambda * (ny * ny) - lambda * (nz * nz) + 2 * mu * (ny * ny))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12n[2] * ((nx * nx) + (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu));
    //     u_ip12n[6] = -((nx * nx * nx) * v_ip12n[0] - (ny * ny * ny) * v_ip12n[3] - (ny * ny * ny) * v_ip12n[6] - (nz * nz * nz) * v_ip12n[4] - (nz * nz * nz) * v_ip12n[5] - (nz * nz * nz) * v_ip12n[7] - (nz * nz * nz) * v_ip12n[8] - nx * (ny * ny) * v_ip12n[0] + (nx * nx) * ny * v_ip12n[3] + (nx * nx) * ny * v_ip12n[6] - nx * (nz * nz) * v_ip12n[0] + (nx * nx) * nz * v_ip12n[4] + (nx * nx) * nz * v_ip12n[5] + (nx * nx) * nz * v_ip12n[7] + (nx * nx) * nz * v_ip12n[8] + ny * (nz * nz) * v_ip12n[3] + (ny * ny) * nz * v_ip12n[4] - (ny * ny) * nz * v_ip12n[5] + ny * (nz * nz) * v_ip12n[6] + (ny * ny) * nz * v_ip12n[7] - (ny * ny) * nz * v_ip12n[8]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    //     u_ip12n[7] = v_ip12n[1] + v_ip12n[3] + v_ip12n[6] + (nz * v_ip12n[5]) / ny + (nz * v_ip12n[8]) / ny;
    //     u_ip12n[8] = v_ip12n[2] + v_ip12n[4] + v_ip12n[5] + v_ip12n[7] + v_ip12n[8];

    //     u_ip12p[0] = (v_ip12p[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) - (v_ip12p[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) + (ny * v_ip12p[4] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (ny * v_ip12p[7] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nz * v_ip12p[3] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (nz * v_ip12p[6] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz));
    //     u_ip12p[1] = (v_ip12p[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12p[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12p[4] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12p[7] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12p[3] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12p[6] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    //     u_ip12p[2] = (nz * v_ip12p[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (nz * v_ip12p[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (v_ip12p[3] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12p[6] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12p[4] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12p[7] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    //     u_ip12p[3] = (nx * v_ip12p[5]) / (2 * ny) + (nx * v_ip12p[8]) / (2 * ny) + (nx * ny * v_ip12p[4]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * ny * v_ip12p[7]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12p[3]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12p[6]) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (v_ip12p[2] * (2 * lambda * (ny * ny) - lambda * (nx * nx) + 2 * mu * (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12p[1] * (2 * lambda * (nz * nz) - lambda * (nx * nx) + 2 * mu * (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12p[0] * ((ny * ny) + (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu));
    //     u_ip12p[4] = (ny * v_ip12p[5]) / (2 * nx) + (ny * v_ip12p[8]) / (2 * nx) - ((ny * ny) * nz * v_ip12p[3]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - ((ny * ny) * nz * v_ip12p[6]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12p[4] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12p[7] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12p[2] * (2 * lambda * (nx * nx) - lambda * (ny * ny) + 2 * mu * (nx * nx))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12p[0] * (2 * lambda * (nz * nz) - lambda * (ny * ny) + 2 * mu * (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12p[1] * ((nx * nx) + (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu));
    //     u_ip12p[5] = ((nz * nz) * v_ip12p[5]) / (2 * nx * ny) + ((nz * nz) * v_ip12p[8]) / (2 * nx * ny) - (ny * (nz * nz) * v_ip12p[4]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * (nz * nz) * v_ip12p[7]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12p[3] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12p[6] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12p[1] * (2 * lambda * (nx * nx) - lambda * (nz * nz) + 2 * mu * (nx * nx))) / (2 * nx * nz * (3 * lambda + 2 * mu)) - (v_ip12p[0] * (2 * lambda * (ny * ny) - lambda * (nz * nz) + 2 * mu * (ny * ny))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12p[2] * ((nx * nx) + (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu));
    //     u_ip12p[6] = -((nx * nx * nx) * v_ip12p[0] - (ny * ny * ny) * v_ip12p[3] - (ny * ny * ny) * v_ip12p[6] - (nz * nz * nz) * v_ip12p[4] - (nz * nz * nz) * v_ip12p[5] - (nz * nz * nz) * v_ip12p[7] - (nz * nz * nz) * v_ip12p[8] - nx * (ny * ny) * v_ip12p[0] + (nx * nx) * ny * v_ip12p[3] + (nx * nx) * ny * v_ip12p[6] - nx * (nz * nz) * v_ip12p[0] + (nx * nx) * nz * v_ip12p[4] + (nx * nx) * nz * v_ip12p[5] + (nx * nx) * nz * v_ip12p[7] + (nx * nx) * nz * v_ip12p[8] + ny * (nz * nz) * v_ip12p[3] + (ny * ny) * nz * v_ip12p[4] - (ny * ny) * nz * v_ip12p[5] + ny * (nz * nz) * v_ip12p[6] + (ny * ny) * nz * v_ip12p[7] - (ny * ny) * nz * v_ip12p[8]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    //     u_ip12p[7] = v_ip12p[1] + v_ip12p[3] + v_ip12p[6] + (nz * v_ip12p[5]) / ny + (nz * v_ip12p[8]) / ny;
    //     u_ip12p[8] = v_ip12p[2] + v_ip12p[4] + v_ip12p[5] + v_ip12p[7] + v_ip12p[8];

    // #ifdef LF
    //     // Riemann solver: Lax-Friedrichs
    //     fu_ip12n[0] = -(nx * (lambda * u_ip12n[4] + lambda * u_ip12n[5] + u_ip12n[3] * (lambda + 2 * mu)) + mu * ny * u_ip12n[8] + mu * nz * u_ip12n[7]);
    //     fu_ip12n[1] = -(ny * (lambda * u_ip12n[3] + lambda * u_ip12n[5] + u_ip12n[4] * (lambda + 2 * mu)) + mu * nx * u_ip12n[8] + mu * nz * u_ip12n[6]);
    //     fu_ip12n[2] = -(nz * (lambda * u_ip12n[3] + lambda * u_ip12n[4] + u_ip12n[5] * (lambda + 2 * mu)) + mu * nx * u_ip12n[7] + mu * ny * u_ip12n[6]);
    //     fu_ip12n[3] = -((nx * u_ip12n[0]) * buoyancy);
    //     fu_ip12n[4] = -((ny * u_ip12n[1]) * buoyancy);
    //     fu_ip12n[5] = -((nz * u_ip12n[2]) * buoyancy);
    //     fu_ip12n[6] = -((ny * u_ip12n[2]) * buoyancy + (nz * u_ip12n[1]) * buoyancy);
    //     fu_ip12n[7] = -((nx * u_ip12n[2]) * buoyancy + (nz * u_ip12n[0]) * buoyancy);
    //     fu_ip12n[8] = -((nx * u_ip12n[1]) * buoyancy + (ny * u_ip12n[0]) * buoyancy);

    //     fu_ip12p[0] = -(nx * (lambda * u_ip12p[4] + lambda * u_ip12p[5] + u_ip12p[3] * (lambda + 2 * mu)) + mu * ny * u_ip12p[8] + mu * nz * u_ip12p[7]);
    //     fu_ip12p[1] = -(ny * (lambda * u_ip12p[3] + lambda * u_ip12p[5] + u_ip12p[4] * (lambda + 2 * mu)) + mu * nx * u_ip12p[8] + mu * nz * u_ip12p[6]);
    //     fu_ip12p[2] = -(nz * (lambda * u_ip12p[3] + lambda * u_ip12p[4] + u_ip12p[5] * (lambda + 2 * mu)) + mu * nx * u_ip12p[7] + mu * ny * u_ip12p[6]);
    //     fu_ip12p[3] = -((nx * u_ip12p[0]) * buoyancy);
    //     fu_ip12p[4] = -((ny * u_ip12p[1]) * buoyancy);
    //     fu_ip12p[5] = -((nz * u_ip12p[2]) * buoyancy);
    //     fu_ip12p[6] = -((ny * u_ip12p[2]) * buoyancy + (nz * u_ip12p[1]) * buoyancy);
    //     fu_ip12p[7] = -((nx * u_ip12p[2]) * buoyancy + (nz * u_ip12p[0]) * buoyancy);
    //     fu_ip12p[8] = -((nx * u_ip12p[1]) * buoyancy + (ny * u_ip12p[0]) * buoyancy);

    //     for (int n = 0; n < 9; n++)
    //     {
    //         Riemann_flux[idx * WSIZE + n] = 0.5f * (fu_ip12p[n] + fu_ip12n[n] - alpha * (u_ip12p[n] - u_ip12n[n]));
    //     }
    // #endif

    //     END_CALCULATE3D()

    CALCULATE3D(i, j, k, HALO, _nx, HALO, _ny, _nz - HALO, _nz)

#ifdef PML
    pml_beta_x = pml_beta.x[i];
    pml_beta_y = pml_beta.y[j];
    pml_beta_z = pml_beta.z[k];
#endif

    nx = CJM[idx * CJMSIZE + 6] + 1e-6;
    ny = CJM[idx * CJMSIZE + 7] + 1e-6;
    nz = CJM[idx * CJMSIZE + 8] + 1e-6;
    sum_square = nx * nx + ny * ny + nz * nz;
    jac = 1.0 / CJM[idx * CJMSIZE + 9];

    mu = CJM[idx * CJMSIZE + 10];
    lambda = CJM[idx * CJMSIZE + 11];
    buoyancy = CJM[idx * CJMSIZE + 12];
    rho = 1.0 / buoyancy;

    // Store original variables
    for (int n = 0; n < 6; n++)
    {
        u[n][0] = W[INDEX(i, j, k - 2 + n) * WSIZE + 0];
        u[n][1] = W[INDEX(i, j, k - 2 + n) * WSIZE + 1];
        u[n][2] = W[INDEX(i, j, k - 2 + n) * WSIZE + 2];
        u[n][3] = W[INDEX(i, j, k - 2 + n) * WSIZE + 3];
        u[n][4] = W[INDEX(i, j, k - 2 + n) * WSIZE + 4];
        u[n][5] = W[INDEX(i, j, k - 2 + n) * WSIZE + 5];
        u[n][6] = W[INDEX(i, j, k - 2 + n) * WSIZE + 6];
        u[n][7] = W[INDEX(i, j, k - 2 + n) * WSIZE + 7];
        u[n][8] = W[INDEX(i, j, k - 2 + n) * WSIZE + 8];
    }

    // Transform to characteristic domain
    for (int n = 0; n < 6; n++)
    {
        v[n][0] = -(lambda * nx * (ny * ny * ny) * u[n][7] - 2 * mu * (nx * nx * nx * nx) * u[n][6] - lambda * (nx * nx * nx * nx) * u[n][6] + lambda * (nx * nx * nx) * ny * u[n][7] + lambda * nx * (nz * nz * nz) * u[n][8] + lambda * (nx * nx * nx) * nz * u[n][8] + 2 * lambda * ny * (nz * nz * nz) * u[n][3] + 2 * lambda * (ny * ny * ny) * nz * u[n][3] + 4 * lambda * ny * (nz * nz * nz) * u[n][4] + 4 * lambda * (ny * ny * ny) * nz * u[n][5] + 2 * mu * nx * (ny * ny * ny) * u[n][7] + 2 * mu * (nx * nx * nx) * ny * u[n][7] + 2 * mu * nx * (nz * nz * nz) * u[n][8] + 2 * mu * (nx * nx * nx) * nz * u[n][8] + 4 * mu * ny * (nz * nz * nz) * u[n][4] + 4 * mu * (ny * ny * ny) * nz * u[n][5] - lambda * (nx * nx) * (ny * ny) * u[n][6] - lambda * (nx * nx) * (nz * nz) * u[n][6] - 4 * lambda * (ny * ny) * (nz * nz) * u[n][6] - 2 * mu * (nx * nx) * (ny * ny) * u[n][6] - 2 * mu * (nx * nx) * (nz * nz) * u[n][6] - 4 * mu * (ny * ny) * (nz * nz) * u[n][6] - 2 * lambda * (nx * nx) * ny * nz * u[n][3] + 4 * lambda * (nx * nx) * ny * nz * u[n][4] + 4 * lambda * (nx * nx) * ny * nz * u[n][5] - 3 * lambda * nx * ny * (nz * nz) * u[n][7] - 3 * lambda * nx * (ny * ny) * nz * u[n][8] - 4 * mu * (nx * nx) * ny * nz * u[n][3] + 4 * mu * (nx * nx) * ny * nz * u[n][4] + 4 * mu * (nx * nx) * ny * nz * u[n][5] - 2 * mu * nx * ny * (nz * nz) * u[n][7] - 2 * mu * nx * (ny * ny) * nz * u[n][8]) / ((lambda + 2 * mu) * (sum_square * sum_square));
        v[n][1] = -(lambda * nx * (ny * ny * ny) * u[n][6] - 2 * mu * (ny * ny * ny * ny) * u[n][7] - lambda * (ny * ny * ny * ny) * u[n][7] + lambda * (nx * nx * nx) * ny * u[n][6] + 4 * lambda * nx * (nz * nz * nz) * u[n][3] + 2 * lambda * nx * (nz * nz * nz) * u[n][4] + 2 * lambda * (nx * nx * nx) * nz * u[n][4] + 4 * lambda * (nx * nx * nx) * nz * u[n][5] + lambda * ny * (nz * nz * nz) * u[n][8] + lambda * (ny * ny * ny) * nz * u[n][8] + 2 * mu * nx * (ny * ny * ny) * u[n][6] + 2 * mu * (nx * nx * nx) * ny * u[n][6] + 4 * mu * nx * (nz * nz * nz) * u[n][3] + 4 * mu * (nx * nx * nx) * nz * u[n][5] + 2 * mu * ny * (nz * nz * nz) * u[n][8] + 2 * mu * (ny * ny * ny) * nz * u[n][8] - lambda * (nx * nx) * (ny * ny) * u[n][7] - 4 * lambda * (nx * nx) * (nz * nz) * u[n][7] - lambda * (ny * ny) * (nz * nz) * u[n][7] - 2 * mu * (nx * nx) * (ny * ny) * u[n][7] - 4 * mu * (nx * nx) * (nz * nz) * u[n][7] - 2 * mu * (ny * ny) * (nz * nz) * u[n][7] + 4 * lambda * nx * (ny * ny) * nz * u[n][3] - 2 * lambda * nx * (ny * ny) * nz * u[n][4] + 4 * lambda * nx * (ny * ny) * nz * u[n][5] - 3 * lambda * nx * ny * (nz * nz) * u[n][6] - 3 * lambda * (nx * nx) * ny * nz * u[n][8] + 4 * mu * nx * (ny * ny) * nz * u[n][3] - 4 * mu * nx * (ny * ny) * nz * u[n][4] + 4 * mu * nx * (ny * ny) * nz * u[n][5] - 2 * mu * nx * ny * (nz * nz) * u[n][6] - 2 * mu * (nx * nx) * ny * nz * u[n][8]) / ((lambda + 2 * mu) * (sum_square * sum_square));
        v[n][2] = -(4 * lambda * nx * (ny * ny * ny) * u[n][3] - 2 * mu * (nz * nz * nz * nz) * u[n][8] - lambda * (nz * nz * nz * nz) * u[n][8] + 4 * lambda * (nx * nx * nx) * ny * u[n][4] + 2 * lambda * nx * (ny * ny * ny) * u[n][5] + 2 * lambda * (nx * nx * nx) * ny * u[n][5] + lambda * nx * (nz * nz * nz) * u[n][6] + lambda * (nx * nx * nx) * nz * u[n][6] + lambda * ny * (nz * nz * nz) * u[n][7] + lambda * (ny * ny * ny) * nz * u[n][7] + 4 * mu * nx * (ny * ny * ny) * u[n][3] + 4 * mu * (nx * nx * nx) * ny * u[n][4] + 2 * mu * nx * (nz * nz * nz) * u[n][6] + 2 * mu * (nx * nx * nx) * nz * u[n][6] + 2 * mu * ny * (nz * nz * nz) * u[n][7] + 2 * mu * (ny * ny * ny) * nz * u[n][7] - 4 * lambda * (nx * nx) * (ny * ny) * u[n][8] - lambda * (nx * nx) * (nz * nz) * u[n][8] - lambda * (ny * ny) * (nz * nz) * u[n][8] - 4 * mu * (nx * nx) * (ny * ny) * u[n][8] - 2 * mu * (nx * nx) * (nz * nz) * u[n][8] - 2 * mu * (ny * ny) * (nz * nz) * u[n][8] + 4 * lambda * nx * ny * (nz * nz) * u[n][3] + 4 * lambda * nx * ny * (nz * nz) * u[n][4] - 2 * lambda * nx * ny * (nz * nz) * u[n][5] - 3 * lambda * nx * (ny * ny) * nz * u[n][6] - 3 * lambda * (nx * nx) * ny * nz * u[n][7] + 4 * mu * nx * ny * (nz * nz) * u[n][3] + 4 * mu * nx * ny * (nz * nz) * u[n][4] - 4 * mu * nx * ny * (nz * nz) * u[n][5] - 2 * mu * nx * (ny * ny) * nz * u[n][6] - 2 * mu * (nx * nx) * ny * nz * u[n][7]) / ((lambda + 2 * mu) * (sum_square * sum_square));
        v[n][3] = ((nx * nx * nx) * u[n][2] * sqrt(mu * rho * sum_square) + (nz * nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) + nx * (ny * ny) * u[n][2] * sqrt(mu * rho * sum_square) - (nx * nx) * nz * u[n][0] * sqrt(mu * rho * sum_square) - nx * (nz * nz) * u[n][2] * sqrt(mu * rho * sum_square) + (ny * ny) * nz * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][7] + mu * (nz * nz * nz * nz) * rho * u[n][7] + mu * (nx * nx) * (ny * ny) * rho * u[n][7] - 2 * mu * (nx * nx) * (nz * nz) * rho * u[n][7] + mu * (ny * ny) * (nz * nz) * rho * u[n][7] - 2 * nx * ny * nz * u[n][1] * sqrt(mu * rho * sum_square) + mu * nx * (ny * ny * ny) * rho * u[n][6] + mu * (nx * nx * nx) * ny * rho * u[n][6] + 2 * mu * nx * (nz * nz * nz) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * nz * rho * u[n][3] - 2 * mu * nx * (nz * nz * nz) * rho * u[n][5] + 2 * mu * (nx * nx * nx) * nz * rho * u[n][5] + mu * ny * (nz * nz * nz) * rho * u[n][8] + mu * (ny * ny * ny) * nz * rho * u[n][8] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][3] - 4 * mu * nx * (ny * ny) * nz * rho * u[n][4] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][5] - 3 * mu * nx * ny * (nz * nz) * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][8]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][4] = ((nx * nx * nx) * u[n][1] * sqrt(mu * rho * sum_square) + (ny * ny * ny) * u[n][0] * sqrt(mu * rho * sum_square) - (nx * nx) * ny * u[n][0] * sqrt(mu * rho * sum_square) - nx * (ny * ny) * u[n][1] * sqrt(mu * rho * sum_square) + nx * (nz * nz) * u[n][1] * sqrt(mu * rho * sum_square) + ny * (nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][8] + mu * (ny * ny * ny * ny) * rho * u[n][8] - 2 * mu * (nx * nx) * (ny * ny) * rho * u[n][8] + mu * (nx * nx) * (nz * nz) * rho * u[n][8] + mu * (ny * ny) * (nz * nz) * rho * u[n][8] - 2 * nx * ny * nz * u[n][2] * sqrt(mu * rho * sum_square) + 2 * mu * nx * (ny * ny * ny) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * ny * rho * u[n][3] - 2 * mu * nx * (ny * ny * ny) * rho * u[n][4] + 2 * mu * (nx * nx * nx) * ny * rho * u[n][4] + mu * nx * (nz * nz * nz) * rho * u[n][6] + mu * (nx * nx * nx) * nz * rho * u[n][6] + mu * ny * (nz * nz * nz) * rho * u[n][7] + mu * (ny * ny * ny) * nz * rho * u[n][7] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][3] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][4] - 4 * mu * nx * ny * (nz * nz) * rho * u[n][5] - 3 * mu * nx * (ny * ny) * nz * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][7]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][5] = (rho * rho * (lambda + 2 * mu) * (lambda * (nx * nx * nx * nx) * ny * u[n][0] + lambda * nx * (ny * ny * ny * ny) * u[n][1] + 2 * mu * (nx * nx * nx * nx) * ny * u[n][0] + 2 * mu * nx * (ny * ny * ny * ny) * u[n][1] + lambda * (nx * nx) * (ny * ny * ny) * u[n][0] + lambda * (nx * nx * nx) * (ny * ny) * u[n][1] + 2 * mu * (nx * nx) * (ny * ny * ny) * u[n][0] + 2 * mu * (nx * nx * nx) * (ny * ny) * u[n][1] + lambda * nx * (ny * ny * ny) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx * nx) * ny * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * (ny * ny * ny) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx * nx) * ny * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx) * ny * (nz * nz) * u[n][0] + lambda * nx * (ny * ny) * (nz * nz) * u[n][1] + 2 * mu * (nx * nx) * ny * (nz * nz) * u[n][0] + 2 * mu * nx * (ny * ny) * (nz * nz) * u[n][1] + 2 * mu * (nx * nx) * (ny * ny) * u[n][8] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz * nz) * u[n][2] + lambda * nx * (ny * ny * ny) * nz * u[n][2] + lambda * (nx * nx * nx) * ny * nz * u[n][2] + 2 * mu * nx * ny * (nz * nz * nz) * u[n][2] + 2 * mu * nx * (ny * ny * ny) * nz * u[n][2] + 2 * mu * (nx * nx * nx) * ny * nz * u[n][2] + lambda * nx * ny * (nz * nz) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * nx * (ny * ny) * nz * u[n][6] * sqrt(rho * (lambda + 2 * mu) * sum_square) + 2 * mu * (nx * nx) * ny * nz * u[n][7] * sqrt(rho * (lambda + 2 * mu) * sum_square))) / ((rho * (lambda + 2 * mu) * sum_square) * (rho * (lambda + 2 * mu) * sum_square) * sqrt(rho * (lambda + 2 * mu) * sum_square));
        v[n][6] = ((nx * nx) * nz * u[n][0] * sqrt(mu * rho * sum_square) - (nz * nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) - nx * (ny * ny) * u[n][2] * sqrt(mu * rho * sum_square) - (nx * nx * nx) * u[n][2] * sqrt(mu * rho * sum_square) + nx * (nz * nz) * u[n][2] * sqrt(mu * rho * sum_square) - (ny * ny) * nz * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][7] + mu * (nz * nz * nz * nz) * rho * u[n][7] + mu * (nx * nx) * (ny * ny) * rho * u[n][7] - 2 * mu * (nx * nx) * (nz * nz) * rho * u[n][7] + mu * (ny * ny) * (nz * nz) * rho * u[n][7] + 2 * nx * ny * nz * u[n][1] * sqrt(mu * rho * sum_square) + mu * nx * (ny * ny * ny) * rho * u[n][6] + mu * (nx * nx * nx) * ny * rho * u[n][6] + 2 * mu * nx * (nz * nz * nz) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * nz * rho * u[n][3] - 2 * mu * nx * (nz * nz * nz) * rho * u[n][5] + 2 * mu * (nx * nx * nx) * nz * rho * u[n][5] + mu * ny * (nz * nz * nz) * rho * u[n][8] + mu * (ny * ny * ny) * nz * rho * u[n][8] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][3] - 4 * mu * nx * (ny * ny) * nz * rho * u[n][4] + 2 * mu * nx * (ny * ny) * nz * rho * u[n][5] - 3 * mu * nx * ny * (nz * nz) * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][8]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][7] = ((nx * nx) * ny * u[n][0] * sqrt(mu * rho * sum_square) - (ny * ny * ny) * u[n][0] * sqrt(mu * rho * sum_square) - (nx * nx * nx) * u[n][1] * sqrt(mu * rho * sum_square) + nx * (ny * ny) * u[n][1] * sqrt(mu * rho * sum_square) - nx * (nz * nz) * u[n][1] * sqrt(mu * rho * sum_square) - ny * (nz * nz) * u[n][0] * sqrt(mu * rho * sum_square) + mu * (nx * nx * nx * nx) * rho * u[n][8] + mu * (ny * ny * ny * ny) * rho * u[n][8] - 2 * mu * (nx * nx) * (ny * ny) * rho * u[n][8] + mu * (nx * nx) * (nz * nz) * rho * u[n][8] + mu * (ny * ny) * (nz * nz) * rho * u[n][8] + 2 * nx * ny * nz * u[n][2] * sqrt(mu * rho * sum_square) + 2 * mu * nx * (ny * ny * ny) * rho * u[n][3] - 2 * mu * (nx * nx * nx) * ny * rho * u[n][3] - 2 * mu * nx * (ny * ny * ny) * rho * u[n][4] + 2 * mu * (nx * nx * nx) * ny * rho * u[n][4] + mu * nx * (nz * nz * nz) * rho * u[n][6] + mu * (nx * nx * nx) * nz * rho * u[n][6] + mu * ny * (nz * nz * nz) * rho * u[n][7] + mu * (ny * ny * ny) * nz * rho * u[n][7] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][3] + 2 * mu * nx * ny * (nz * nz) * rho * u[n][4] - 4 * mu * nx * ny * (nz * nz) * rho * u[n][5] - 3 * mu * nx * (ny * ny) * nz * rho * u[n][6] - 3 * mu * (nx * nx) * ny * nz * rho * u[n][7]) / (2 * mu * rho * (sum_square * sum_square));
        v[n][8] = -(rho * rho * (lambda + 2 * mu) * (lambda * (nx * nx * nx * nx) * ny * u[n][0] + lambda * nx * (ny * ny * ny * ny) * u[n][1] + 2 * mu * (nx * nx * nx * nx) * ny * u[n][0] + 2 * mu * nx * (ny * ny * ny * ny) * u[n][1] + lambda * (nx * nx) * (ny * ny * ny) * u[n][0] + lambda * (nx * nx * nx) * (ny * ny) * u[n][1] + 2 * mu * (nx * nx) * (ny * ny * ny) * u[n][0] + 2 * mu * (nx * nx * nx) * (ny * ny) * u[n][1] - lambda * nx * (ny * ny * ny) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * (nx * nx * nx) * ny * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * (ny * ny * ny) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * (nx * nx * nx) * ny * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * (nx * nx * nx) * ny * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * nx * (ny * ny * ny) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * (nx * nx) * ny * (nz * nz) * u[n][0] + lambda * nx * (ny * ny) * (nz * nz) * u[n][1] + 2 * mu * (nx * nx) * ny * (nz * nz) * u[n][0] + 2 * mu * nx * (ny * ny) * (nz * nz) * u[n][1] - 2 * mu * (nx * nx) * (ny * ny) * u[n][8] * sqrt(rho * (lambda + 2 * mu) * sum_square) + lambda * nx * ny * (nz * nz * nz) * u[n][2] + lambda * nx * (ny * ny * ny) * nz * u[n][2] + lambda * (nx * nx * nx) * ny * nz * u[n][2] + 2 * mu * nx * ny * (nz * nz * nz) * u[n][2] + 2 * mu * nx * (ny * ny * ny) * nz * u[n][2] + 2 * mu * (nx * nx * nx) * ny * nz * u[n][2] - lambda * nx * ny * (nz * nz) * u[n][3] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * ny * (nz * nz) * u[n][4] * sqrt(rho * (lambda + 2 * mu) * sum_square) - lambda * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * nx * ny * (nz * nz) * u[n][5] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * nx * (ny * ny) * nz * u[n][6] * sqrt(rho * (lambda + 2 * mu) * sum_square) - 2 * mu * (nx * nx) * ny * nz * u[n][7] * sqrt(rho * (lambda + 2 * mu) * sum_square))) / ((rho * (lambda + 2 * mu) * sum_square) * (rho * (lambda + 2 * mu) * sum_square) * sqrt(rho * (lambda + 2 * mu) * sum_square));
    }

    // Shock capture interpolation
    for (int n = 0; n < 9; n++)
    {
        v_ip12n[n] = WENO5_interpolation(v[0][n], v[1][n], v[2][n], v[3][n], v[4][n]);
        v_ip12p[n] = WENO5_interpolation(v[5][n], v[4][n], v[3][n], v[2][n], v[1][n]);
    }

    // Transform back to physical domain
    u_ip12n[0] = (v_ip12n[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) - (v_ip12n[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) + (ny * v_ip12n[4] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (ny * v_ip12n[7] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nz * v_ip12n[3] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (nz * v_ip12n[6] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz));
    u_ip12n[1] = (v_ip12n[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12n[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12n[4] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12n[7] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12n[3] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12n[6] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12n[2] = (nz * v_ip12n[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (nz * v_ip12n[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (v_ip12n[3] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12n[6] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12n[4] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12n[7] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12n[3] = (nx * v_ip12n[5]) / (2 * ny) + (nx * v_ip12n[8]) / (2 * ny) + (nx * ny * v_ip12n[4]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * ny * v_ip12n[7]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12n[3]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12n[6]) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (v_ip12n[2] * (2 * lambda * (ny * ny) - lambda * (nx * nx) + 2 * mu * (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12n[1] * (2 * lambda * (nz * nz) - lambda * (nx * nx) + 2 * mu * (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12n[0] * ((ny * ny) + (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu));
    u_ip12n[4] = (ny * v_ip12n[5]) / (2 * nx) + (ny * v_ip12n[8]) / (2 * nx) - ((ny * ny) * nz * v_ip12n[3]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - ((ny * ny) * nz * v_ip12n[6]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12n[4] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12n[7] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12n[2] * (2 * lambda * (nx * nx) - lambda * (ny * ny) + 2 * mu * (nx * nx))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12n[0] * (2 * lambda * (nz * nz) - lambda * (ny * ny) + 2 * mu * (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12n[1] * ((nx * nx) + (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu));
    u_ip12n[5] = ((nz * nz) * v_ip12n[5]) / (2 * nx * ny) + ((nz * nz) * v_ip12n[8]) / (2 * nx * ny) - (ny * (nz * nz) * v_ip12n[4]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * (nz * nz) * v_ip12n[7]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12n[3] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12n[6] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12n[1] * (2 * lambda * (nx * nx) - lambda * (nz * nz) + 2 * mu * (nx * nx))) / (2 * nx * nz * (3 * lambda + 2 * mu)) - (v_ip12n[0] * (2 * lambda * (ny * ny) - lambda * (nz * nz) + 2 * mu * (ny * ny))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12n[2] * ((nx * nx) + (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu));
    u_ip12n[6] = -((nx * nx * nx) * v_ip12n[0] - (ny * ny * ny) * v_ip12n[3] - (ny * ny * ny) * v_ip12n[6] - (nz * nz * nz) * v_ip12n[4] - (nz * nz * nz) * v_ip12n[5] - (nz * nz * nz) * v_ip12n[7] - (nz * nz * nz) * v_ip12n[8] - nx * (ny * ny) * v_ip12n[0] + (nx * nx) * ny * v_ip12n[3] + (nx * nx) * ny * v_ip12n[6] - nx * (nz * nz) * v_ip12n[0] + (nx * nx) * nz * v_ip12n[4] + (nx * nx) * nz * v_ip12n[5] + (nx * nx) * nz * v_ip12n[7] + (nx * nx) * nz * v_ip12n[8] + ny * (nz * nz) * v_ip12n[3] + (ny * ny) * nz * v_ip12n[4] - (ny * ny) * nz * v_ip12n[5] + ny * (nz * nz) * v_ip12n[6] + (ny * ny) * nz * v_ip12n[7] - (ny * ny) * nz * v_ip12n[8]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12n[7] = v_ip12n[1] + v_ip12n[3] + v_ip12n[6] + (nz * v_ip12n[5]) / ny + (nz * v_ip12n[8]) / ny;
    u_ip12n[8] = v_ip12n[2] + v_ip12n[4] + v_ip12n[5] + v_ip12n[7] + v_ip12n[8];

    u_ip12p[0] = (v_ip12p[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) - (v_ip12p[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * ny) + (ny * v_ip12p[4] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (ny * v_ip12p[7] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nz * v_ip12p[3] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (nz * v_ip12p[6] * sqrt(mu * rho * sum_square)) / (-(nx * nx) + (ny * ny) + (nz * nz));
    u_ip12p[1] = (v_ip12p[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12p[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx) - (v_ip12p[4] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12p[7] * ((nx * nx) - (nz * nz)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12p[3] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12p[6] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12p[2] = (nz * v_ip12p[5] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (nz * v_ip12p[8] * sqrt(rho * (lambda + 2 * mu) * sum_square)) / (2 * nx * ny) - (v_ip12p[3] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (v_ip12p[6] * ((nx * nx) - (ny * ny)) * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * nz * v_ip12p[4] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) + (ny * nz * v_ip12p[7] * sqrt(mu * rho * sum_square)) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12p[3] = (nx * v_ip12p[5]) / (2 * ny) + (nx * v_ip12p[8]) / (2 * ny) + (nx * ny * v_ip12p[4]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * ny * v_ip12p[7]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12p[3]) / (-(nx * nx) + (ny * ny) + (nz * nz)) + (nx * nz * v_ip12p[6]) / (-(nx * nx) + (ny * ny) + (nz * nz)) - (v_ip12p[2] * (2 * lambda * (ny * ny) - lambda * (nx * nx) + 2 * mu * (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12p[1] * (2 * lambda * (nz * nz) - lambda * (nx * nx) + 2 * mu * (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12p[0] * ((ny * ny) + (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu));
    u_ip12p[4] = (ny * v_ip12p[5]) / (2 * nx) + (ny * v_ip12p[8]) / (2 * nx) - ((ny * ny) * nz * v_ip12p[3]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - ((ny * ny) * nz * v_ip12p[6]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12p[4] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * v_ip12p[7] * ((nx * nx) - (nz * nz))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12p[2] * (2 * lambda * (nx * nx) - lambda * (ny * ny) + 2 * mu * (nx * nx))) / (2 * nx * ny * (3 * lambda + 2 * mu)) - (v_ip12p[0] * (2 * lambda * (nz * nz) - lambda * (ny * ny) + 2 * mu * (nz * nz))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12p[1] * ((nx * nx) + (nz * nz))) / (2 * nx * nz * (3 * lambda + 2 * mu));
    u_ip12p[5] = ((nz * nz) * v_ip12p[5]) / (2 * nx * ny) + ((nz * nz) * v_ip12p[8]) / (2 * nx * ny) - (ny * (nz * nz) * v_ip12p[4]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (ny * (nz * nz) * v_ip12p[7]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12p[3] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (nz * v_ip12p[6] * ((nx * nx) - (ny * ny))) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz))) - (v_ip12p[1] * (2 * lambda * (nx * nx) - lambda * (nz * nz) + 2 * mu * (nx * nx))) / (2 * nx * nz * (3 * lambda + 2 * mu)) - (v_ip12p[0] * (2 * lambda * (ny * ny) - lambda * (nz * nz) + 2 * mu * (ny * ny))) / (2 * ny * nz * (3 * lambda + 2 * mu)) + (lambda * v_ip12p[2] * ((nx * nx) + (ny * ny))) / (2 * nx * ny * (3 * lambda + 2 * mu));
    u_ip12p[6] = -((nx * nx * nx) * v_ip12p[0] - (ny * ny * ny) * v_ip12p[3] - (ny * ny * ny) * v_ip12p[6] - (nz * nz * nz) * v_ip12p[4] - (nz * nz * nz) * v_ip12p[5] - (nz * nz * nz) * v_ip12p[7] - (nz * nz * nz) * v_ip12p[8] - nx * (ny * ny) * v_ip12p[0] + (nx * nx) * ny * v_ip12p[3] + (nx * nx) * ny * v_ip12p[6] - nx * (nz * nz) * v_ip12p[0] + (nx * nx) * nz * v_ip12p[4] + (nx * nx) * nz * v_ip12p[5] + (nx * nx) * nz * v_ip12p[7] + (nx * nx) * nz * v_ip12p[8] + ny * (nz * nz) * v_ip12p[3] + (ny * ny) * nz * v_ip12p[4] - (ny * ny) * nz * v_ip12p[5] + ny * (nz * nz) * v_ip12p[6] + (ny * ny) * nz * v_ip12p[7] - (ny * ny) * nz * v_ip12p[8]) / (nx * (-(nx * nx) + (ny * ny) + (nz * nz)));
    u_ip12p[7] = v_ip12p[1] + v_ip12p[3] + v_ip12p[6] + (nz * v_ip12p[5]) / ny + (nz * v_ip12p[8]) / ny;
    u_ip12p[8] = v_ip12p[2] + v_ip12p[4] + v_ip12p[5] + v_ip12p[7] + v_ip12p[8];

#ifdef LF
    // Riemann solver: Lax-Friedrichs
    fu_ip12n[0] = -(nx * (lambda * u_ip12n[4] + lambda * u_ip12n[5] + u_ip12n[3] * (lambda + 2 * mu)) + mu * ny * u_ip12n[8] + mu * nz * u_ip12n[7]);
    fu_ip12n[1] = -(ny * (lambda * u_ip12n[3] + lambda * u_ip12n[5] + u_ip12n[4] * (lambda + 2 * mu)) + mu * nx * u_ip12n[8] + mu * nz * u_ip12n[6]);
    fu_ip12n[2] = -(nz * (lambda * u_ip12n[3] + lambda * u_ip12n[4] + u_ip12n[5] * (lambda + 2 * mu)) + mu * nx * u_ip12n[7] + mu * ny * u_ip12n[6]);
    fu_ip12n[3] = -((nx * u_ip12n[0]) * buoyancy);
    fu_ip12n[4] = -((ny * u_ip12n[1]) * buoyancy);
    fu_ip12n[5] = -((nz * u_ip12n[2]) * buoyancy);
    fu_ip12n[6] = -((ny * u_ip12n[2]) * buoyancy + (nz * u_ip12n[1]) * buoyancy);
    fu_ip12n[7] = -((nx * u_ip12n[2]) * buoyancy + (nz * u_ip12n[0]) * buoyancy);
    fu_ip12n[8] = -((nx * u_ip12n[1]) * buoyancy + (ny * u_ip12n[0]) * buoyancy);

    fu_ip12p[0] = -(nx * (lambda * u_ip12p[4] + lambda * u_ip12p[5] + u_ip12p[3] * (lambda + 2 * mu)) + mu * ny * u_ip12p[8] + mu * nz * u_ip12p[7]);
    fu_ip12p[1] = -(ny * (lambda * u_ip12p[3] + lambda * u_ip12p[5] + u_ip12p[4] * (lambda + 2 * mu)) + mu * nx * u_ip12p[8] + mu * nz * u_ip12p[6]);
    fu_ip12p[2] = -(nz * (lambda * u_ip12p[3] + lambda * u_ip12p[4] + u_ip12p[5] * (lambda + 2 * mu)) + mu * nx * u_ip12p[7] + mu * ny * u_ip12p[6]);
    fu_ip12p[3] = -((nx * u_ip12p[0]) * buoyancy);
    fu_ip12p[4] = -((ny * u_ip12p[1]) * buoyancy);
    fu_ip12p[5] = -((nz * u_ip12p[2]) * buoyancy);
    fu_ip12p[6] = -((ny * u_ip12p[2]) * buoyancy + (nz * u_ip12p[1]) * buoyancy);
    fu_ip12p[7] = -((nx * u_ip12p[2]) * buoyancy + (nz * u_ip12p[0]) * buoyancy);
    fu_ip12p[8] = -((nx * u_ip12p[1]) * buoyancy + (ny * u_ip12p[0]) * buoyancy);

    for (int n = 0; n < 9; n++)
    {
        Riemann_flux[idx * WSIZE + n] = 0.5f * (fu_ip12p[n] + fu_ip12n[n] - alpha * (u_ip12p[n] - u_ip12n[n]));
    }
#endif

    END_CALCULATE3D()
}

__GLOBAL__
void cal_du_x(FLOAT *Riemann_flux, FLOAT *h_W, FLOAT *CJM, FLOAT *u,
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

    if (k >= _nz - 3 && k < _nz)
    {
        float hu[7][9];
        // ! Unable to satisfy the stencil of high-order terms, 3 layers near the free surface need to be recalculated
        for (int n = 0; n < 7; n++)
        {
            hu[n][0] = -(CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 0] * (CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 11] * u[INDEX(i - 3 + n, j, k) * WSIZE + 4] + CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 11] * u[INDEX(i - 3 + n, j, k) * WSIZE + 5] + u[INDEX(i - 3 + n, j, k) * WSIZE + 3] * (CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 11] + 2 * CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 10])) + CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 10] * CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 1] * u[INDEX(i - 3 + n, j, k) * WSIZE + 8] + CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 10] * CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 2] * u[INDEX(i - 3 + n, j, k) * WSIZE + 7]);
            hu[n][1] = -(CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 1] * (CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 11] * u[INDEX(i - 3 + n, j, k) * WSIZE + 3] + CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 11] * u[INDEX(i - 3 + n, j, k) * WSIZE + 5] + u[INDEX(i - 3 + n, j, k) * WSIZE + 4] * (CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 11] + 2 * CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 10])) + CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 10] * CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 0] * u[INDEX(i - 3 + n, j, k) * WSIZE + 8] + CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 10] * CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 2] * u[INDEX(i - 3 + n, j, k) * WSIZE + 6]);
            hu[n][2] = -(CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 2] * (CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 11] * u[INDEX(i - 3 + n, j, k) * WSIZE + 3] + CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 11] * u[INDEX(i - 3 + n, j, k) * WSIZE + 4] + u[INDEX(i - 3 + n, j, k) * WSIZE + 5] * (CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 11] + 2 * CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 10])) + CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 10] * CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 0] * u[INDEX(i - 3 + n, j, k) * WSIZE + 7] + CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 10] * CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 1] * u[INDEX(i - 3 + n, j, k) * WSIZE + 6]);
            hu[n][3] = -((CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 0] * u[INDEX(i - 3 + n, j, k) * WSIZE + 0]) * CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 12]);
            hu[n][4] = -((CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 1] * u[INDEX(i - 3 + n, j, k) * WSIZE + 1]) * CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 12]);
            hu[n][5] = -((CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 2] * u[INDEX(i - 3 + n, j, k) * WSIZE + 2]) * CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 12]);
            hu[n][6] = -((CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 1] * u[INDEX(i - 3 + n, j, k) * WSIZE + 2]) * CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 12] + (CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 2] * u[INDEX(i - 3 + n, j, k) * WSIZE + 1]) * CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 12]);
            hu[n][7] = -((CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 0] * u[INDEX(i - 3 + n, j, k) * WSIZE + 2]) * CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 12] + (CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 2] * u[INDEX(i - 3 + n, j, k) * WSIZE + 0]) * CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 12]);
            hu[n][8] = -((CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 0] * u[INDEX(i - 3 + n, j, k) * WSIZE + 1]) * CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 12] + (CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 1] * u[INDEX(i - 3 + n, j, k) * WSIZE + 0]) * CJM[INDEX(i - 3 + n, j, k) * CJMSIZE + 12]);
        }

        for (int n = 0; n < 9; n++)
        {
            h_W[idx * WSIZE + n] = DT * rDH * jac * (-((Riemann_flux[idx * WSIZE + n] - 1.0f / 24 * order2_approximation_f(hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n], hu[6][n]) + 7.0f / 5760 * order4_approximation_f(hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n], hu[6][n])) - (Riemann_flux[idx_n1 * WSIZE + n] - 1.0f / 24 * order2_approximation_f(hu[0][n], hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n]) + 7.0f / 5760 * order4_approximation_f(hu[0][n], hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n]))));
        }
    }
    else
    {
        for (int n = 0; n < 9; n++)
        {
            h_W[idx * WSIZE + n] = DT * rDH * jac * (-((Riemann_flux[idx * WSIZE + n] - 1.0f / 24 * order2_approximation(Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n], Riemann_flux[idx_p2 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n], Riemann_flux[idx_p2 * WSIZE + n])) - (Riemann_flux[idx_n1 * WSIZE + n] - 1.0f / 24 * order2_approximation(Riemann_flux[idx_n3 * WSIZE + n], Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Riemann_flux[idx_n3 * WSIZE + n], Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n]))));
        }
    }
    END_CALCULATE3D()
}

__GLOBAL__
void cal_du_y(FLOAT *Riemann_flux, FLOAT *h_W, FLOAT *CJM, FLOAT *u,
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

    if (k >= _nz - 3 && k < _nz)
    {
        float hu[7][9];
        // ! Unable to satisfy the stencil of high-order terms, 3 layers near the free surface need to be recalculated
        for (int n = 0; n < 7; n++)
        {
            hu[n][0] = -(CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 3] * (CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 11] * u[INDEX(i, j - 3 + n, k) * WSIZE + 4] + CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 11] * u[INDEX(i, j - 3 + n, k) * WSIZE + 5] + u[INDEX(i, j - 3 + n, k) * WSIZE + 3] * (CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 11] + 2 * CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 10])) + CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 10] * CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 4] * u[INDEX(i, j - 3 + n, k) * WSIZE + 8] + CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 10] * CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 5] * u[INDEX(i, j - 3 + n, k) * WSIZE + 7]);
            hu[n][1] = -(CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 4] * (CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 11] * u[INDEX(i, j - 3 + n, k) * WSIZE + 3] + CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 11] * u[INDEX(i, j - 3 + n, k) * WSIZE + 5] + u[INDEX(i, j - 3 + n, k) * WSIZE + 4] * (CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 11] + 2 * CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 10])) + CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 10] * CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 3] * u[INDEX(i, j - 3 + n, k) * WSIZE + 8] + CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 10] * CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 5] * u[INDEX(i, j - 3 + n, k) * WSIZE + 6]);
            hu[n][2] = -(CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 5] * (CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 11] * u[INDEX(i, j - 3 + n, k) * WSIZE + 3] + CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 11] * u[INDEX(i, j - 3 + n, k) * WSIZE + 4] + u[INDEX(i, j - 3 + n, k) * WSIZE + 5] * (CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 11] + 2 * CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 10])) + CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 10] * CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 3] * u[INDEX(i, j - 3 + n, k) * WSIZE + 7] + CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 10] * CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 4] * u[INDEX(i, j - 3 + n, k) * WSIZE + 6]);
            hu[n][3] = -((CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 3] * u[INDEX(i, j - 3 + n, k) * WSIZE + 0]) * CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 12]);
            hu[n][4] = -((CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 4] * u[INDEX(i, j - 3 + n, k) * WSIZE + 1]) * CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 12]);
            hu[n][5] = -((CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 5] * u[INDEX(i, j - 3 + n, k) * WSIZE + 2]) * CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 12]);
            hu[n][6] = -((CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 4] * u[INDEX(i, j - 3 + n, k) * WSIZE + 2]) * CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 12] + (CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 5] * u[INDEX(i, j - 3 + n, k) * WSIZE + 1]) * CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 12]);
            hu[n][7] = -((CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 3] * u[INDEX(i, j - 3 + n, k) * WSIZE + 2]) * CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 12] + (CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 5] * u[INDEX(i, j - 3 + n, k) * WSIZE + 0]) * CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 12]);
            hu[n][8] = -((CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 3] * u[INDEX(i, j - 3 + n, k) * WSIZE + 1]) * CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 12] + (CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 4] * u[INDEX(i, j - 3 + n, k) * WSIZE + 0]) * CJM[INDEX(i, j - 3 + n, k) * CJMSIZE + 12]);
        }

        for (int n = 0; n < 9; n++)
        {
            h_W[idx * WSIZE + n] += DT * rDH * jac * (-((Riemann_flux[idx * WSIZE + n] - 1.0f / 24 * order2_approximation_f(hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n], hu[6][n]) + 7.0f / 5760 * order4_approximation_f(hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n], hu[6][n])) - (Riemann_flux[idx_n1 * WSIZE + n] - 1.0f / 24 * order2_approximation_f(hu[0][n], hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n]) + 7.0f / 5760 * order4_approximation_f(hu[0][n], hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n]))));
        }
    }
    else
    {
        for (int n = 0; n < 9; n++)
        {
            h_W[idx * WSIZE + n] += DT * rDH * jac * (-((Riemann_flux[idx * WSIZE + n] - 1.0f / 24 * order2_approximation(Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n], Riemann_flux[idx_p2 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n], Riemann_flux[idx_p2 * WSIZE + n])) - (Riemann_flux[idx_n1 * WSIZE + n] - 1.0f / 24 * order2_approximation(Riemann_flux[idx_n3 * WSIZE + n], Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Riemann_flux[idx_n3 * WSIZE + n], Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n]))));
        }
    }
    END_CALCULATE3D()
}

__GLOBAL__
void cal_du_z(FLOAT *Riemann_flux, FLOAT *h_W, FLOAT *CJM, FLOAT *u,
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

    if (k >= _nz - 3 && k < _nz)
    {
        float hu[7][9];
        // ! Unable to satisfy the stencil of high-order terms, 3 layers near the free surface need to be recalculated
        for (int n = 0; n < 7; n++)
        {
            hu[n][0] = -(CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 6] * (CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 11] * u[INDEX(i, j, k - 3 + n) * WSIZE + 4] + CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 11] * u[INDEX(i, j, k - 3 + n) * WSIZE + 5] + u[INDEX(i, j, k - 3 + n) * WSIZE + 3] * (CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 11] + 2 * CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 10])) + CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 10] * CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 7] * u[INDEX(i, j, k - 3 + n) * WSIZE + 8] + CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 10] * CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 8] * u[INDEX(i, j, k - 3 + n) * WSIZE + 7]);
            hu[n][1] = -(CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 7] * (CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 11] * u[INDEX(i, j, k - 3 + n) * WSIZE + 3] + CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 11] * u[INDEX(i, j, k - 3 + n) * WSIZE + 5] + u[INDEX(i, j, k - 3 + n) * WSIZE + 4] * (CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 11] + 2 * CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 10])) + CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 10] * CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 6] * u[INDEX(i, j, k - 3 + n) * WSIZE + 8] + CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 10] * CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 8] * u[INDEX(i, j, k - 3 + n) * WSIZE + 6]);
            hu[n][2] = -(CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 8] * (CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 11] * u[INDEX(i, j, k - 3 + n) * WSIZE + 3] + CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 11] * u[INDEX(i, j, k - 3 + n) * WSIZE + 4] + u[INDEX(i, j, k - 3 + n) * WSIZE + 5] * (CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 11] + 2 * CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 10])) + CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 10] * CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 6] * u[INDEX(i, j, k - 3 + n) * WSIZE + 7] + CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 10] * CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 7] * u[INDEX(i, j, k - 3 + n) * WSIZE + 6]);
            hu[n][3] = -((CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 6] * u[INDEX(i, j, k - 3 + n) * WSIZE + 0]) * CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 12]);
            hu[n][4] = -((CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 7] * u[INDEX(i, j, k - 3 + n) * WSIZE + 1]) * CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 12]);
            hu[n][5] = -((CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 8] * u[INDEX(i, j, k - 3 + n) * WSIZE + 2]) * CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 12]);
            hu[n][6] = -((CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 7] * u[INDEX(i, j, k - 3 + n) * WSIZE + 2]) * CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 12] + (CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 8] * u[INDEX(i, j, k - 3 + n) * WSIZE + 1]) * CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 12]);
            hu[n][7] = -((CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 6] * u[INDEX(i, j, k - 3 + n) * WSIZE + 2]) * CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 12] + (CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 8] * u[INDEX(i, j, k - 3 + n) * WSIZE + 0]) * CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 12]);
            hu[n][8] = -((CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 6] * u[INDEX(i, j, k - 3 + n) * WSIZE + 1]) * CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 12] + (CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 7] * u[INDEX(i, j, k - 3 + n) * WSIZE + 0]) * CJM[INDEX(i, j, k - 3 + n) * CJMSIZE + 12]);
        }

        for (int n = 0; n < 9; n++)
        {
            h_W[idx * WSIZE + n] += DT * rDH * jac * (-((Riemann_flux[idx * WSIZE + n] - 1.0f / 24 * order2_approximation_f(hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n], hu[6][n]) + 7.0f / 5760 * order4_approximation_f(hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n], hu[6][n])) - (Riemann_flux[idx_n1 * WSIZE + n] - 1.0f / 24 * order2_approximation_f(hu[0][n], hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n]) + 7.0f / 5760 * order4_approximation_f(hu[0][n], hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n]))));
        }
    }
    else
    {
        for (int n = 0; n < 9; n++)
        {
            h_W[idx * WSIZE + n] += DT * rDH * jac * (-((Riemann_flux[idx * WSIZE + n] - 1.0f / 24 * order2_approximation(Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n], Riemann_flux[idx_p2 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n], Riemann_flux[idx_p2 * WSIZE + n])) - (Riemann_flux[idx_n1 * WSIZE + n] - 1.0f / 24 * order2_approximation(Riemann_flux[idx_n3 * WSIZE + n], Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Riemann_flux[idx_n3 * WSIZE + n], Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n]))));
        }
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

    //     wave_deriv_alternative_flux_FD_char_x<<<blocks, threads>>>(wave.Riemann_flux, wave.h_W, wave.W, CJM,
    // #ifdef PML
    //                                                                pml_beta,
    // #endif // PML
    //                                                                _nx_, _ny_, _nz_, rDH, DT
    // #ifdef LF
    //                                                                ,
    //                                                                vp_max_for_SCFDM
    // #endif // LF
    //     );

    cal_du_x<<<blocks, threads>>>(wave.Riemann_flux, wave.h_W, CJM, wave.W,
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
    //     wave_deriv_alternative_flux_FD_char_y<<<blocks, threads>>>(wave.Riemann_flux, wave.h_W, wave.W, CJM,
    // #ifdef PML
    //                                                                pml_beta,
    // #endif // PML
    //                                                                _nx_, _ny_, _nz_, rDH, DT
    // #ifdef LF
    //                                                                ,
    //                                                                vp_max_for_SCFDM
    // #endif // LF
    //     );

    cal_du_y<<<blocks, threads>>>(wave.Riemann_flux, wave.h_W, CJM, wave.W,
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
    //     wave_deriv_alternative_flux_FD_char_z<<<blocks, threads>>>(wave.Riemann_flux, wave.h_W, wave.W, CJM,
    // #ifdef PML
    //                                                                pml_beta,
    // #endif // PML
    //                                                                _nx_, _ny_, _nz_, rDH, DT
    // #ifdef LF
    //                                                                ,
    //                                                                vp_max_for_SCFDM
    // #endif // LF
    //     );

    cal_du_z<<<blocks, threads>>>(wave.Riemann_flux, wave.h_W, CJM, wave.W,
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