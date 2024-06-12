/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: solve_displacement.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2021-11-03
*   Discription: Solve Displacement
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

__GLOBAL__
void SolveDisplacementKernel(FLOAT *W, FLOAT *Dis, FLOAT *CJM, int _nx_, int _ny_, int _nz_, FLOAT dt)
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

    float vx, vy, vz;
    long long index;

    CALCULATE3D(i, j, k, 0, _nx_, 0, _ny_, 0, _nz_)
    index = INDEX(i, j, k);

#ifdef SCFDM
    float buoyancy = CJM[index * CJMSIZE + 12];
    vx = W[index * WSIZE + 0] * buoyancy;
    vy = W[index * WSIZE + 1] * buoyancy;
    vz = W[index * WSIZE + 2] * buoyancy;

#else
    vx = W[index * WSIZE + 0];
    vy = W[index * WSIZE + 1];
    vz = W[index * WSIZE + 2];
#endif

    Dis[index * 3 + 0] += vx * dt;
    Dis[index * 3 + 1] += vy * dt;
    Dis[index * 3 + 2] += vz * dt;

    END_CALCULATE3D()
}

void SolveDisplacement(GRID grid, FLOAT *W, FLOAT *Dis, FLOAT *CJM, FLOAT dt)
{
    dim3 threads(32, 4, 4);
    dim3 blocks;
    blocks.x = (grid._nx_ + threads.x - 1) / threads.x;
    blocks.y = (grid._ny_ + threads.y - 1) / threads.y;
    blocks.z = (grid._nz_ + threads.z - 1) / threads.z;

    SolveDisplacementKernel<<<blocks, threads>>>(W, Dis, CJM, grid._nx_, grid._ny_, grid._nz_, dt);
}