#include "header.h"
typedef void (*WAVE_RK_FUNC_FLOAT)(FLOAT *h_W, FLOAT *W, FLOAT *t_W, FLOAT *m_W, long long num);
__GLOBAL__
void wave_tvd_rk0(FLOAT *h_W, FLOAT *W, FLOAT *t_W, FLOAT *m_W, long long WStride)
{
    // printf( "wave_rk0\n" );
#ifdef GPU_CUDA
    long long i = threadIdx.x + blockIdx.x * blockDim.x;
#else
    long long i = 0;
#endif
    float h_w, w, t_w, m_w;
    CALCULATE1D(i, 0, WStride)
    h_w = (float)h_W[i];
    m_w = (float)W[i];

    w = m_w + h_w;

    m_W[i] = m_w;
    W[i] = w;
    END_CALCULATE1D()
}

__GLOBAL__
void wave_tvd_rk1(FLOAT *h_W, FLOAT *W, FLOAT *t_W, FLOAT *m_W, long long WStride)
{
    // printf( "wave_rk1\n" );
#ifdef GPU_CUDA
    long long i = threadIdx.x + blockIdx.x * blockDim.x;
#else
    long long i = 0;
#endif
    float h_w, w, t_w, m_w;
    CALCULATE1D(i, 0, WStride)
    h_w = (float)h_W[i];
    m_w = (float)m_W[i];
    w = (float)W[i];

    w = 0.75 * m_w + 0.25 * w + 0.25 * h_w;

    W[i] = w;
    END_CALCULATE1D()
}

__GLOBAL__
void wave_tvd_rk2(FLOAT *h_W, FLOAT *W, FLOAT *t_W, FLOAT *m_W, long long WStride)
{
    // printf( "wave_rk2\n" );
#ifdef GPU_CUDA
    long long i = threadIdx.x + blockIdx.x * blockDim.x;
#else
    long long i = 0;
#endif
    float h_w, w, t_w, m_w;
    CALCULATE1D(i, 0, WStride)
    h_w = (float)h_W[i];
    m_w = (float)m_W[i];
    w = (float)W[i];

    w = (1.0 / 3.0) * m_w + (2.0 / 3.0) * w + (2.0 / 3.0) * h_w;

    W[i] = w;
    END_CALCULATE1D()
}

void waveRk_tvd(GRID grid, int irk, WAVE wave)
{
    WAVE_RK_FUNC_FLOAT tvd_wave_rk[3] = {wave_tvd_rk0, wave_tvd_rk1, wave_tvd_rk2};
    long long num = grid._nx_ * grid._ny_ * grid._nz_ * WSIZE;

#ifdef GPU_CUDA
    dim3 threads(1024, 1, 1);
    dim3 blocks;
    blocks.x = (num + threads.x - 1) / threads.x;
    blocks.y = 1;
    blocks.z = 1;
    tvd_wave_rk[irk]<<<blocks, threads>>>(wave.h_W, wave.W, wave.t_W, wave.m_W, num);
    // wave_rk<<< blocks, threads >>>( h_W, W, t_W, m_W, num, DT );
    CHECK(cudaDeviceSynchronize());
#else
    wave_rk[irk](wave.h_W, wave.W, wave.t_W, wave.m_W, num);
#endif
}
