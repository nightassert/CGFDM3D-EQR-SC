/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: macro.h
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2021-11-03
*   Discription: Define macros
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

#ifndef __MACRO__
#define __MACRO__

#define FLOAT float

#ifdef GPU_CUDA

#undef FLOAT

#ifdef FLOAT16
#define FLOAT __half //__nv_bfloat16//__half
#else
#define FLOAT float
#endif

#endif

#define POW2(x) ((x) * (x))
#define GAUSS_FUN(t, a, t0) (exp(-POW2(((t) - (t0)) / (a))) / (a * 1.772453850905516))

#define HALO 3
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

// #define WAVESIZE 9 //9 Wave components: Vx Vy Vz Txx Tyy Tzz Txy Txz Tyz
// #define COORDSIZE 3 //3 coordinate components: coordX coordY coordZ
#define CONTRASIZE 9 // contravariant components
#define CONSIZE 9 // contravariant components

#define MEDIUMSIZE 3 // 3 medium components: Vs Vp rho ( lam mu bouyancy )
#define MOMSIZE 6

#define CSIZE 3
#define MSIZE 3
#define CJMSIZE 13
#define WSIZE 9
#define MATSIZE 9

#ifdef SOLVE_PGA
#define PGVSIZE 4
#else
#define PGVSIZE 2
#endif

#define PI 3.141592657f
// #define PI 3.1415926535898
#define RADIAN2DEGREE (180.0 / PI)
#define DEGREE2RADIAN (PI / 180.0)

// #define Cv 5.0e3
// #define Cs 1.0e-6
// #define Crho 1e6//~= Cv/Cs/rho
#define Cv 1.0
#define Cs 1.0
#define Crho 1.0 //~= Cv/Cs/rho

#define NO_SOURCE_SMOOTH

#ifdef GPU_CUDA

#define Malloc cudaMalloc
#define Memset cudaMemset
#define Free cudaFree
#define Memcpy cudaMemcpy

#define __DEVICE__ __device__
#define __GLOBAL__ __global__

#define CHECK(call)                                          \
  {                                                          \
    const cudaError_t error = call;                          \
    if (error != cudaSuccess)                                \
    {                                                        \
      fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
      fprintf(stderr, "code: %d, reason: %s\n",              \
              error, cudaGetErrorString(error));             \
    }                                                        \
  }

#else
int Malloc(void **mem, long long size);

#define Memset memset
#define Free free
#define Memcpy memcpy
#define __DEVICE__
#define __GLOBAL__

#define CHECK(call) call

#endif

#define THREE 3

// forward difference coefficient
#define af_1 (-0.30874f)
#define af0 (-0.6326f)
#define af1 (1.2330f)
#define af2 (-0.3334f)
#define af3 (0.04168f)
// backward difference coefficient
#define ab_1 (0.30874f)
#define ab0 (0.6326f)
#define ab1 (-1.2330f)
#define ab2 (0.3334f)
#define ab3 (-0.04168f)

#define alpha1 0.0f
#define alpha2 0.5f
#define alpha3 0.5f
#define alpha4 1.0f

// #define beta1 1.0f//0.16666667f
#define beta1 0.16666667f
#define beta2 0.33333333f
#define beta3 0.33333333f
#define beta4 0.16666667f

#define Cf1 (-1.16666667f)
#define Cf2 (1.33333333f)
#define Cf3 (-0.16666667f)

#define Cb1 (1.16666667f)
#define Cb2 (-1.33333333f)
#define Cb3 (0.16666667f)

// #define Cf1 ( - 7.0 / 6.0 )
// #define Cf2 (   4.0 / 3.0 )
// #define Cf3 ( - 1.0 / 6.0 )

// #define Cb1 (   7.0 / 6.0 )
// #define Cb2 ( - 4.0 / 3.0 )
// #define Cb3 (   1.0 / 6.0 )

// grid: x, y, z

#include "axis.h"

#endif //__MACRO__
