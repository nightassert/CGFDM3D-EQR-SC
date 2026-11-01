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
#include "header.h"

#define order2_approximation(u1, u2, u3, u4, u5) (1.0f / 12 * (-u1 + 16 * u2 - 30 * u3 + 16 * u4 - u5))
#define order4_approximation(u1, u2, u3, u4, u5) ((u1 - 4 * u2 + 6 * u3 - 4 * u4 + u5))

#define order2_approximation_f(u1, u2, u3, u4, u5, u6) (1.0f / 48 * (-5 * u1 + 39 * u2 - 34 * u3 - 34 * u4 + 39 * u5 - 5 * u6))
#define order4_approximation_f(u1, u2, u3, u4, u5, u6) (1.0f / 2 * (u1 - 3 * u2 + 2 * u3 + 2 * u4 - 3 * u5 + u6))

extern float g_vp_max;

#ifdef GPU_CUDA
#define WENO5_INLINE __forceinline__
#else
#define WENO5_INLINE inline
#endif

// WENO5-JS interpolation scheme
__DEVICE__ WENO5_INLINE
float WENO5_JS(float u1, float u2, float u3, float u4, float u5)
{
    // smoothness indicators in sum-of-squares form: mathematically identical
    // to the expanded form, but beta >= 0 and free of catastrophic cancellation
    float d1 = u3 - 2.0f * u4 + u5;
    float d2 = 3.0f * u3 - 4.0f * u4 + u5;
    float e1 = u2 - 2.0f * u3 + u4;
    float e2 = u2 - u4;
    float f1 = u1 - 2.0f * u2 + u3;
    float f2 = u1 - 4.0f * u2 + 3.0f * u3;

    float WENO_beta1 = (13.0f / 12.0f) * d1 * d1 + 0.25f * d2 * d2;
    float WENO_beta2 = (13.0f / 12.0f) * e1 * e1 + 0.25f * e2 * e2;
    float WENO_beta3 = (13.0f / 12.0f) * f1 * f1 + 0.25f * f2 * f2;

    // float epsilon deliberately: mixing 1e-6 (double) makes these three
    // divisions run on the FP64 pipe, which is 1/64 rate on consumer GPUs
    WENO_beta1 = 0.3125f / ((WENO_beta1 + 1e-6f) * (WENO_beta1 + 1e-6f));
    WENO_beta2 = 0.625f / ((WENO_beta2 + 1e-6f) * (WENO_beta2 + 1e-6f));
    WENO_beta3 = 0.0625f / ((WENO_beta3 + 1e-6f) * (WENO_beta3 + 1e-6f));

    // nonlinear weights
    float sum_weights = WENO_beta1 + WENO_beta2 + WENO_beta3;

    // defensive fallback: only reachable if the weights overflow or the input is inf/NaN
    if (!(sum_weights > 0.0f) || !(sum_weights < 3.0e38f))
    {
        return 0.3125f * (0.375f * u3 + 0.75f * u4 - 0.125f * u5) + 0.625f * (-0.125f * u2 + 0.75f * u3 + 0.375f * u4) + 0.0625f * (0.375f * u1 - 1.25f * u2 + 1.875f * u3);
    }

    float total_weights = 1.0f / sum_weights;

    // WENO interpolation
    return (WENO_beta1 * total_weights) * (0.375f * u3 + 0.75f * u4 - 0.125f * u5) + (WENO_beta2 * total_weights) * (-0.125f * u2 + 0.75f * u3 + 0.375f * u4) + (WENO_beta3 * total_weights) * (0.375f * u1 - 1.25f * u2 + 1.875f * u3);
}

// WENO5-Z interpolation (Borges et al., 2008).  Same stencils and ideal
// weights as WENO5_JS; the nonlinear weights use the global smoothness
// indicator tau5 = |beta1 - beta3|, so the formal order is kept at smooth
// extrema where the JS weights degrade.  This lowers the dissipation on wave
// peaks without giving up the non-oscillatory behaviour.
__DEVICE__ WENO5_INLINE
float WENO5_Z(float u1, float u2, float u3, float u4, float u5)
{
    float d1 = u3 - 2.0f * u4 + u5;
    float d2 = 3.0f * u3 - 4.0f * u4 + u5;
    float e1 = u2 - 2.0f * u3 + u4;
    float e2 = u2 - u4;
    float f1 = u1 - 2.0f * u2 + u3;
    float f2 = u1 - 4.0f * u2 + 3.0f * u3;

    float WENO_beta1 = (13.0f / 12.0f) * d1 * d1 + 0.25f * d2 * d2;
    float WENO_beta2 = (13.0f / 12.0f) * e1 * e1 + 0.25f * e2 * e2;
    float WENO_beta3 = (13.0f / 12.0f) * f1 * f1 + 0.25f * f2 * f2;

    // Z weights: alpha_k = d_k * (1 + tau5 / (beta_k + eps))^2
    float tau5 = fabsf(WENO_beta1 - WENO_beta3);
    float t1 = 1.0f + tau5 / (WENO_beta1 + 1e-6f);
    float t2 = 1.0f + tau5 / (WENO_beta2 + 1e-6f);
    float t3 = 1.0f + tau5 / (WENO_beta3 + 1e-6f);

    float WENO_alpha1 = 0.3125f * t1 * t1;
    float WENO_alpha2 = 0.625f * t2 * t2;
    float WENO_alpha3 = 0.0625f * t3 * t3;

    float sum_weights = WENO_alpha1 + WENO_alpha2 + WENO_alpha3;

    // Defensive fallback, identical to WENO5_JS: only reachable if the weights
    // overflow or the input is inf/NaN.
    if (!(sum_weights > 0.0f) || !(sum_weights < 3.0e38f))
    {
        return 0.3125f * (0.375f * u3 + 0.75f * u4 - 0.125f * u5) + 0.625f * (-0.125f * u2 + 0.75f * u3 + 0.375f * u4) + 0.0625f * (0.375f * u1 - 1.25f * u2 + 1.875f * u3);
    }

    float total_weights = 1.0f / sum_weights;

    return (WENO_alpha1 * total_weights) * (0.375f * u3 + 0.75f * u4 - 0.125f * u5) + (WENO_alpha2 * total_weights) * (-0.125f * u2 + 0.75f * u3 + 0.375f * u4) + (WENO_alpha3 * total_weights) * (0.375f * u1 - 1.25f * u2 + 1.875f * u3);
}

// Vp of one cell from the material slots of CJM: mu (+10), lambda (+11),
// 1/rho (+12).  Used by the optional per-interface LF speed.
__DEVICE__ WENO5_INLINE
float weno_cell_vp(const FLOAT *CJM, long long idx)
{
    float mu = CJM[idx * CJMSIZE + 10];
    float lambda = CJM[idx * CJMSIZE + 11];
    return sqrtf((lambda + 2.0f * mu) * CJM[idx * CJMSIZE + 12]);
}

__GLOBAL__
void wave_deriv_alternative_flux_FD_x(FLOAT *Riemann_flux, FLOAT *h_W, FLOAT *W, FLOAT *CJM,
                                      int _nx_, int _ny_, int _nz_, float rDH, float DT, float alpha,
                                      int localAlpha, int wenoZ
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


    float mu;
    float lambda;
    float buoyancy;

    float nx;
    float ny;
    float nz;
    float jac;

    long long idx_n3, idx_n2, idx_n1, idx, idx_p1, idx_p2, idx_p3;

    float u_ip12n[9], u_ip12p[9];

    float fu_ip12n[9], fu_ip12p[9];

    // * X direction
    CALCULATE3D(i, j, k, HALO - 1, _nx, HALO, _ny, HALO, _nz)


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

    // LF dissipation speed at the i+1/2 interface.  The historical choice is
    // one global Vp_max; the local option takes the faster of the two cells
    // sharing the face, so slow shallow layers are not over-damped.  Cap at
    // the global value, which is the CFL-consistent bound used when DT was
    // chosen.
    float alpha_face = alpha;
    if (localAlpha)
    {
        float vp_i = sqrtf((lambda + 2.0f * mu) * buoyancy);
        alpha_face = fmaxf(vp_i, weno_cell_vp(CJM, idx_p1));
        if (alpha_face > alpha)
            alpha_face = alpha;
    }

    // Shock capture interpolation.  JS is the historical choice; Z keeps the
    // formal order at smooth extrema and is therefore less dissipative on
    // wave peaks.  The branch is uniform over the whole kernel launch.
    if (wenoZ)
    {
        for (int n = 0; n < 9; n++)
        {
            u_ip12n[n] = WENO5_Z(W[idx_n2 * WSIZE + n], W[idx_n1 * WSIZE + n], W[idx * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx_p2 * WSIZE + n]);
            u_ip12p[n] = WENO5_Z(W[idx_p3 * WSIZE + n], W[idx_p2 * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx * WSIZE + n], W[idx_n1 * WSIZE + n]);
        }
    }
    else
    {
        for (int n = 0; n < 9; n++)
        {
            u_ip12n[n] = WENO5_JS(W[idx_n2 * WSIZE + n], W[idx_n1 * WSIZE + n], W[idx * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx_p2 * WSIZE + n]);
            u_ip12p[n] = WENO5_JS(W[idx_p3 * WSIZE + n], W[idx_p2 * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx * WSIZE + n], W[idx_n1 * WSIZE + n]);
        }
    }

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
        Riemann_flux[idx * WSIZE + n] = 0.5f * (fu_ip12p[n] + fu_ip12n[n] - alpha_face * (u_ip12p[n] - u_ip12n[n]));
    }

    END_CALCULATE3D()
}

__GLOBAL__
void wave_deriv_alternative_flux_FD_y(FLOAT *Riemann_flux, FLOAT *h_W, FLOAT *W, FLOAT *CJM,
                                      int _nx_, int _ny_, int _nz_, float rDH, float DT, float alpha,
                                      int localAlpha, int wenoZ
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


    float mu;
    float lambda;
    float buoyancy;

    float nx;
    float ny;
    float nz;
    float jac;

    long long idx_n3, idx_n2, idx_n1, idx, idx_p1, idx_p2, idx_p3;

    float u_ip12n[9], u_ip12p[9];

    float fu_ip12n[9], fu_ip12p[9];

    // * Y direction
    CALCULATE3D(i, j, k, HALO, _nx, HALO - 1, _ny, HALO, _nz)


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

    // LF dissipation speed at the j+1/2 interface; see the x kernel.
    float alpha_face = alpha;
    if (localAlpha)
    {
        float vp_i = sqrtf((lambda + 2.0f * mu) * buoyancy);
        alpha_face = fmaxf(vp_i, weno_cell_vp(CJM, idx_p1));
        if (alpha_face > alpha)
            alpha_face = alpha;
    }

    // Shock capture interpolation.  JS is the historical choice; Z keeps the
    // formal order at smooth extrema and is therefore less dissipative on
    // wave peaks.  The branch is uniform over the whole kernel launch.
    if (wenoZ)
    {
        for (int n = 0; n < 9; n++)
        {
            u_ip12n[n] = WENO5_Z(W[idx_n2 * WSIZE + n], W[idx_n1 * WSIZE + n], W[idx * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx_p2 * WSIZE + n]);
            u_ip12p[n] = WENO5_Z(W[idx_p3 * WSIZE + n], W[idx_p2 * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx * WSIZE + n], W[idx_n1 * WSIZE + n]);
        }
    }
    else
    {
        for (int n = 0; n < 9; n++)
        {
            u_ip12n[n] = WENO5_JS(W[idx_n2 * WSIZE + n], W[idx_n1 * WSIZE + n], W[idx * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx_p2 * WSIZE + n]);
            u_ip12p[n] = WENO5_JS(W[idx_p3 * WSIZE + n], W[idx_p2 * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx * WSIZE + n], W[idx_n1 * WSIZE + n]);
        }
    }

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
        Riemann_flux[idx * WSIZE + n] = 0.5f * (fu_ip12p[n] + fu_ip12n[n] - alpha_face * (u_ip12p[n] - u_ip12n[n]));
    }
    END_CALCULATE3D()
}

__GLOBAL__
void wave_deriv_alternative_flux_FD_z(FLOAT *Riemann_flux, FLOAT *h_W, FLOAT *W, FLOAT *CJM,
                                      int _nx_, int _ny_, int _nz_, float rDH, float DT, float alpha,
                                      int localAlpha, int wenoZ
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


    float mu;
    float lambda;
    float buoyancy;

    float nx;
    float ny;
    float nz;
    float jac;

    long long idx_n3, idx_n2, idx_n1, idx, idx_p1, idx_p2, idx_p3;

    float u_ip12n[9], u_ip12p[9];

    float fu_ip12n[9], fu_ip12p[9];

    // * Z direction
    CALCULATE3D(i, j, k, HALO, _nx, HALO, _ny, HALO - 1, _nz)


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

    // LF dissipation speed at the k+1/2 interface; see the x kernel.
    float alpha_face = alpha;
    if (localAlpha)
    {
        float vp_i = sqrtf((lambda + 2.0f * mu) * buoyancy);
        alpha_face = fmaxf(vp_i, weno_cell_vp(CJM, idx_p1));
        if (alpha_face > alpha)
            alpha_face = alpha;
    }

    // Shock capture interpolation.  JS is the historical choice; Z keeps the
    // formal order at smooth extrema and is therefore less dissipative on
    // wave peaks.  The branch is uniform over the whole kernel launch.
    if (wenoZ)
    {
        for (int n = 0; n < 9; n++)
        {
            u_ip12n[n] = WENO5_Z(W[idx_n2 * WSIZE + n], W[idx_n1 * WSIZE + n], W[idx * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx_p2 * WSIZE + n]);
            u_ip12p[n] = WENO5_Z(W[idx_p3 * WSIZE + n], W[idx_p2 * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx * WSIZE + n], W[idx_n1 * WSIZE + n]);
        }
    }
    else
    {
        for (int n = 0; n < 9; n++)
        {
            u_ip12n[n] = WENO5_JS(W[idx_n2 * WSIZE + n], W[idx_n1 * WSIZE + n], W[idx * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx_p2 * WSIZE + n]);
            u_ip12p[n] = WENO5_JS(W[idx_p3 * WSIZE + n], W[idx_p2 * WSIZE + n], W[idx_p1 * WSIZE + n], W[idx * WSIZE + n], W[idx_n1 * WSIZE + n]);
        }
    }

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
        Riemann_flux[idx * WSIZE + n] = 0.5f * (fu_ip12p[n] + fu_ip12n[n] - alpha_face * (u_ip12p[n] - u_ip12n[n]));
    }
    END_CALCULATE3D()
}

__GLOBAL__
void cal_du_x(FLOAT *Riemann_flux, FLOAT *h_W, FLOAT *CJM, FLOAT *u,
              int _nx_, int _ny_, int _nz_, float rDH, int FB1, int FB2, int FB3, float DT,
              float *pml_src_x1, float *pml_src_x2, int nPML)
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

    // WENO C-PML: inside an x-PML slab the x-directional contribution is
    // stashed into the h_Aux slot of that surface instead of being written
    // to h_W; wenoPmlDeriv reads it back and adds the C-PML form.
    float *pml_src = NULL;
    int nx = _nx_ - 2 * HALO;
    int ny = _ny_ - 2 * HALO;
    int nz = _nz_ - 2 * HALO;
    if (NULL != pml_src_x1 && i < HALO + nPML)
        pml_src = pml_src_x1 + Index3D(i - HALO, j - HALO, k - HALO, nPML, ny, nz) * WSIZE;
    else if (NULL != pml_src_x2 && i >= _nx - nPML)
        pml_src = pml_src_x2 + Index3D(i - HALO - (nx - nPML), j - HALO, k - HALO, nPML, ny, nz) * WSIZE;

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
            float du = DT * rDH * jac * (-((Riemann_flux[idx * WSIZE + n] - 1.0f / 24 * order2_approximation_f(hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n], hu[6][n]) + 7.0f / 5760 * order4_approximation_f(hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n], hu[6][n])) - (Riemann_flux[idx_n1 * WSIZE + n] - 1.0f / 24 * order2_approximation_f(hu[0][n], hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n]) + 7.0f / 5760 * order4_approximation_f(hu[0][n], hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n]))));
            if (NULL != pml_src)
            {
                pml_src[n] = du;
                h_W[idx * WSIZE + n] = 0.0f; // the C-PML form is added by wenoPmlDeriv
            }
            else
                h_W[idx * WSIZE + n] = du;
        }
    }
    else
    {
        for (int n = 0; n < 9; n++)
        {
            float du = DT * rDH * jac * (-((Riemann_flux[idx * WSIZE + n] - 1.0f / 24 * order2_approximation(Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n], Riemann_flux[idx_p2 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n], Riemann_flux[idx_p2 * WSIZE + n])) - (Riemann_flux[idx_n1 * WSIZE + n] - 1.0f / 24 * order2_approximation(Riemann_flux[idx_n3 * WSIZE + n], Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Riemann_flux[idx_n3 * WSIZE + n], Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n]))));
            if (NULL != pml_src)
            {
                pml_src[n] = du;
                h_W[idx * WSIZE + n] = 0.0f; // the C-PML form is added by wenoPmlDeriv
            }
            else
                h_W[idx * WSIZE + n] = du;
        }
    }
    END_CALCULATE3D()
}

__GLOBAL__
void cal_du_y(FLOAT *Riemann_flux, FLOAT *h_W, FLOAT *CJM, FLOAT *u,
              int _nx_, int _ny_, int _nz_, float rDH, int FB1, int FB2, int FB3, float DT,
              float *pml_src_y1, float *pml_src_y2, int nPML)
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

    // WENO C-PML: inside a y-PML slab the y-directional contribution is
    // stashed into the h_Aux slot of that surface instead of being
    // accumulated into h_W; wenoPmlDeriv reads it back and applies the
    // C-PML correction.
    float *pml_src = NULL;
    int nx = _nx_ - 2 * HALO;
    int ny = _ny_ - 2 * HALO;
    int nz = _nz_ - 2 * HALO;
    if (NULL != pml_src_y1 && j < HALO + nPML)
        pml_src = pml_src_y1 + Index3D(i - HALO, j - HALO, k - HALO, nx, nPML, nz) * WSIZE;
    else if (NULL != pml_src_y2 && j >= _ny - nPML)
        pml_src = pml_src_y2 + Index3D(i - HALO, j - HALO - (ny - nPML), k - HALO, nx, nPML, nz) * WSIZE;

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
            float du = DT * rDH * jac * (-((Riemann_flux[idx * WSIZE + n] - 1.0f / 24 * order2_approximation_f(hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n], hu[6][n]) + 7.0f / 5760 * order4_approximation_f(hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n], hu[6][n])) - (Riemann_flux[idx_n1 * WSIZE + n] - 1.0f / 24 * order2_approximation_f(hu[0][n], hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n]) + 7.0f / 5760 * order4_approximation_f(hu[0][n], hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n]))));
            if (NULL != pml_src)
                pml_src[n] = du;
            else
                h_W[idx * WSIZE + n] += du;
        }
    }
    else
    {
        for (int n = 0; n < 9; n++)
        {
            float du = DT * rDH * jac * (-((Riemann_flux[idx * WSIZE + n] - 1.0f / 24 * order2_approximation(Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n], Riemann_flux[idx_p2 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n], Riemann_flux[idx_p2 * WSIZE + n])) - (Riemann_flux[idx_n1 * WSIZE + n] - 1.0f / 24 * order2_approximation(Riemann_flux[idx_n3 * WSIZE + n], Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Riemann_flux[idx_n3 * WSIZE + n], Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n]))));
            if (NULL != pml_src)
                pml_src[n] = du;
            else
                h_W[idx * WSIZE + n] += du;
        }
    }
    END_CALCULATE3D()
}

__GLOBAL__
void cal_du_z(FLOAT *Riemann_flux, FLOAT *h_W, FLOAT *CJM, FLOAT *u,
              int _nx_, int _ny_, int _nz_, float rDH, int FB1, int FB2, int FB3, float DT,
              float *pml_src_z1, float *pml_src_z2, int nPML)
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

    // WENO C-PML: inside a z-PML slab the z-directional contribution is
    // stashed into the h_Aux slot of that surface instead of being
    // accumulated into h_W; wenoPmlDeriv reads it back and applies the
    // C-PML correction.
    float *pml_src = NULL;
    int nx = _nx_ - 2 * HALO;
    int ny = _ny_ - 2 * HALO;
    int nz = _nz_ - 2 * HALO;
    if (NULL != pml_src_z1 && k < HALO + nPML)
        pml_src = pml_src_z1 + Index3D(i - HALO, j - HALO, k - HALO, nx, ny, nPML) * WSIZE;
    else if (NULL != pml_src_z2 && k >= _nz - nPML)
        pml_src = pml_src_z2 + Index3D(i - HALO, j - HALO, k - HALO - (nz - nPML), nx, ny, nPML) * WSIZE;

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
            float du = DT * rDH * jac * (-((Riemann_flux[idx * WSIZE + n] - 1.0f / 24 * order2_approximation_f(hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n], hu[6][n]) + 7.0f / 5760 * order4_approximation_f(hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n], hu[6][n])) - (Riemann_flux[idx_n1 * WSIZE + n] - 1.0f / 24 * order2_approximation_f(hu[0][n], hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n]) + 7.0f / 5760 * order4_approximation_f(hu[0][n], hu[1][n], hu[2][n], hu[3][n], hu[4][n], hu[5][n]))));
            if (NULL != pml_src)
                pml_src[n] = du;
            else
                h_W[idx * WSIZE + n] += du;
        }
    }
    else
    {
        for (int n = 0; n < 9; n++)
        {
            float du = DT * rDH * jac * (-((Riemann_flux[idx * WSIZE + n] - 1.0f / 24 * order2_approximation(Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n], Riemann_flux[idx_p2 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n], Riemann_flux[idx_p2 * WSIZE + n])) - (Riemann_flux[idx_n1 * WSIZE + n] - 1.0f / 24 * order2_approximation(Riemann_flux[idx_n3 * WSIZE + n], Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n]) + 7.0f / 5760 * order4_approximation(Riemann_flux[idx_n3 * WSIZE + n], Riemann_flux[idx_n2 * WSIZE + n], Riemann_flux[idx_n1 * WSIZE + n], Riemann_flux[idx * WSIZE + n], Riemann_flux[idx_p1 * WSIZE + n]))));
            if (NULL != pml_src)
                pml_src[n] = du;
            else
                h_W[idx * WSIZE + n] += du;
        }
    }
    END_CALCULATE3D()
}

void waveDeriv_alternative_flux_FD(GRID grid, WAVE wave, FLOAT *CJM, AUX6 Aux6, MPI_BORDER border,
								   int FB1, int FB2, int FB3, float DT, MPI_COORD thisMPICoord, PARAMS params)
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	float rDH = grid.rDH;

	// WENO-Z keeps the same stencils and ideal weights as WENO-JS; only the
	// nonlinear weights differ, so the whole scheme sits in this one flag.
	int wenoZ = (params.wenoReconstruction == WENO_RECONSTRUCTION_Z);

	// WENO C-PML scratch: the h_Aux slot of each active surface doubles as
	// the buffer in which cal_du_x/y/z stash their directional contribution
	// inside the PML slabs (see weno_pml.cpp).  NULL disables the stash and
	// the contribution is accumulated into h_W as usual.
	float *pml_src_x1 = NULL;
	float *pml_src_x2 = NULL;
	float *pml_src_y1 = NULL;
	float *pml_src_y2 = NULL;
	float *pml_src_z1 = NULL;
	float *pml_src_z2 = NULL;

	if (params.nPML > 0)
	{
		if (border.isx1)
			pml_src_x1 = Aux6.Aux1x.h_Aux;
		if (border.isy1)
			pml_src_y1 = Aux6.Aux1y.h_Aux;
		if (border.isz1)
			pml_src_z1 = Aux6.Aux1z.h_Aux;

		if (border.isx2)
			pml_src_x2 = Aux6.Aux2x.h_Aux;
		if (border.isy2)
			pml_src_y2 = Aux6.Aux2y.h_Aux;
#ifndef FREE_SURFACE
		if (border.isz2)
			pml_src_z2 = Aux6.Aux2z.h_Aux;
#endif
	}

#ifdef GPU_CUDA
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

	wave_deriv_alternative_flux_FD_x<<<blocks, threads>>>(wave.t_W, wave.h_W, wave.W, CJM,
														  _nx_, _ny_, _nz_, rDH, DT, g_vp_max, params.wenoLocalAlpha, wenoZ);
	CHECK(cudaGetLastError());
	cal_du_x<<<blocks, threads>>>(wave.t_W, wave.h_W, CJM, wave.W,
								  _nx_, _ny_, _nz_, rDH, FB1, FB2, FB3, DT,
								  pml_src_x1, pml_src_x2, grid.nPML);
	CHECK(cudaGetLastError());

	wave_deriv_alternative_flux_FD_y<<<blocks, threads>>>(wave.t_W, wave.h_W, wave.W, CJM,
														  _nx_, _ny_, _nz_, rDH, DT, g_vp_max, params.wenoLocalAlpha, wenoZ);
	CHECK(cudaGetLastError());
	cal_du_y<<<blocks, threads>>>(wave.t_W, wave.h_W, CJM, wave.W,
								  _nx_, _ny_, _nz_, rDH, FB1, FB2, FB3, DT,
								  pml_src_y1, pml_src_y2, grid.nPML);
	CHECK(cudaGetLastError());

	wave_deriv_alternative_flux_FD_z<<<blocks, threads>>>(wave.t_W, wave.h_W, wave.W, CJM,
														  _nx_, _ny_, _nz_, rDH, DT, g_vp_max, params.wenoLocalAlpha, wenoZ);
	CHECK(cudaGetLastError());
	cal_du_z<<<blocks, threads>>>(wave.t_W, wave.h_W, CJM, wave.W,
								  _nx_, _ny_, _nz_, rDH, FB1, FB2, FB3, DT,
								  pml_src_z1, pml_src_z2, grid.nPML);
	CHECK(cudaGetLastError());
#else
	wave_deriv_alternative_flux_FD_x(wave.t_W, wave.h_W, wave.W, CJM,
									 _nx_, _ny_, _nz_, rDH, DT, g_vp_max, params.wenoLocalAlpha, wenoZ);
	cal_du_x(wave.t_W, wave.h_W, CJM, wave.W,
			 _nx_, _ny_, _nz_, rDH, FB1, FB2, FB3, DT,
			 pml_src_x1, pml_src_x2, grid.nPML);

	wave_deriv_alternative_flux_FD_y(wave.t_W, wave.h_W, wave.W, CJM,
									 _nx_, _ny_, _nz_, rDH, DT, g_vp_max, params.wenoLocalAlpha, wenoZ);
	cal_du_y(wave.t_W, wave.h_W, CJM, wave.W,
			 _nx_, _ny_, _nz_, rDH, FB1, FB2, FB3, DT,
			 pml_src_y1, pml_src_y2, grid.nPML);

	wave_deriv_alternative_flux_FD_z(wave.t_W, wave.h_W, wave.W, CJM,
									 _nx_, _ny_, _nz_, rDH, DT, g_vp_max, params.wenoLocalAlpha, wenoZ);
	cal_du_z(wave.t_W, wave.h_W, CJM, wave.W,
			 _nx_, _ny_, _nz_, rDH, FB1, FB2, FB3, DT,
			 pml_src_z1, pml_src_z2, grid.nPML);
#endif
}

