/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: freeSurface.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2019-04-11
*   Discription:

*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*
=================================================================*/

#ifndef HEADER_H
#define HEADER_H
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

#ifdef GPU_CUDA
#include <cublas_v2.h>
#include <cuda_fp16.h>
// #include <cuda_bf16.h>
#endif

#include <fstream>
#include <iostream>
#include <iomanip>
using namespace std;
#include <list>
#include <map>
#include <vector>

#if __GNUC__
#include <sys/stat.h>
#include <sys/types.h>
#elif _MSC_VER
#include <windows.h>
#include <direct.h>
#endif

#include <proj.h>
#include "macro.h"
#include "struct.h"
#include "cJSON.h"

#include "functions.h"

#endif // !HEADER_H
