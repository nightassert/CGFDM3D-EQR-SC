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
    vx = W[index * WSIZE + 6] * buoyancy;
    vy = W[index * WSIZE + 7] * buoyancy;
    vz = W[index * WSIZE + 8] * buoyancy;

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