/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: exp_decay_absorb.cpp
*   Author: Tianhong Xu, 12231218@mail.sustech.edu.cn
*   Created Time: 2023-11-16
*   Discription: Exponential decay absorbing boundary layers
*
*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*
=================================================================*/
#ifdef SCFDM
#include "header.h"

#ifdef EXP_DECAY

__GLOBAL__
void exp_decay_layers(FLOAT *W, int _nx_, int _ny_, int _nz_, int _NX_, int _NY_, int _NZ_, int frontNX, int frontNY, int frontNZ, int n_exp_decay_layers, FLOAT exp_decay_alpha)
{
#ifdef GPU_CUDA
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
#else
    // int i = HALO;
    // int j = HALO;
    // int k = _nz - HALO;
#endif

    long long index;
    int p; // Distance from point to the inner boundary
    int NX_front, NY_front, NZ_front;
    float G;

    CALCULATE3D(i, j, k, HALO, _nx_ - HALO, HALO, _ny_ - HALO, HALO, _nz_ - HALO)
    index = INDEX(i, j, k);
    NX_front = i - HALO + frontNX;
    NY_front = j - HALO + frontNY;
    NZ_front = k - HALO + frontNZ;

    if (NX_front <= n_exp_decay_layers)
    {
        p = n_exp_decay_layers - NX_front;
        G = exp(-(exp_decay_alpha * (p)) * (exp_decay_alpha * (p)));
        for (int n = 0; n < 9; n++)
        {
            W[index * WSIZE + n] *= G;
        }
    }

    if (NX_front >= _NX_ - 2 * HALO - n_exp_decay_layers)
    {
        p = n_exp_decay_layers - (_NX_ - 2 * HALO - NX_front);
        G = exp(-(exp_decay_alpha * (p)) * (exp_decay_alpha * (p)));
        for (int n = 0; n < 9; n++)
        {
            W[index * WSIZE + n] *= G;
        }
    }

    if (NY_front <= n_exp_decay_layers)
    {
        p = n_exp_decay_layers - NY_front;
        G = exp(-(exp_decay_alpha * (p)) * (exp_decay_alpha * (p)));
        for (int n = 0; n < 9; n++)
        {
            W[index * WSIZE + n] *= G;
        }
    }

    if (NY_front >= _NY_ - 2 * HALO - n_exp_decay_layers)
    {
        p = n_exp_decay_layers - (_NY_ - 2 * HALO - NY_front);
        G = exp(-(exp_decay_alpha * (p)) * (exp_decay_alpha * (p)));
        for (int n = 0; n < 9; n++)
        {
            W[index * WSIZE + n] *= G;
        }
    }

    if (NZ_front <= n_exp_decay_layers)
    {
        p = n_exp_decay_layers - NZ_front;
        G = exp(-(exp_decay_alpha * (p)) * (exp_decay_alpha * (p)));
        for (int n = 0; n < 9; n++)
        {
            W[index * WSIZE + n] *= G;
        }
    }

#ifndef FREE_SURFACE
    if (NZ_front >= _NZ_ - 2 * HALO - n_exp_decay_layers)
    {
        p = n_exp_decay_layers - (_NZ_ - 2 * HALO - NZ_front);
        G = exp(-(exp_decay_alpha * (p)) * (exp_decay_alpha * (p)));
        for (int n = 0; n < 9; n++)
        {
            W[index * WSIZE + n] *= G;
        }
    }
#endif

    END_CALCULATE3D()
}

void expDecayLayers(
    GRID grid, WAVE wave)
{

    int n_exp_decay_layers = 20;
    float exp_decay_alpha = 0.015;

    int _nx_ = grid._nx_;
    int _ny_ = grid._ny_;
    int _nz_ = grid._nz_;

    int _NX_ = grid._NX_;
    int _NY_ = grid._NY_;
    int _NZ_ = grid._NZ_;

    int frontNX = grid.frontNX;
    int frontNY = grid.frontNY;
    int frontNZ = grid.frontNZ;

    FLOAT *W = wave.W;

#ifdef GPU_CUDA
    dim3 threads(32, 4, 1);
    dim3 blocks;

    blocks.x = (_nx_ + threads.x - 1) / threads.x;
    blocks.y = (_ny_ + threads.y - 1) / threads.y;
    blocks.z = (_nz_ + threads.z - 1) / threads.z;

    exp_decay_layers<<<blocks, threads>>>(W, _nx_, _ny_, _nz_, _NX_, _NY_, _NZ_, frontNX, frontNY, frontNZ, n_exp_decay_layers, exp_decay_alpha);

#else

    exp_decay_layers(W, _nx_, _ny_, _nz_, _NX_, _NY_, _NZ_, frontNX, frontNY, frontNZ, n_exp_decay_layers, exp_decay_alpha);

#endif // GPU_CUDA
}
#endif // EXP_DECAY
#endif