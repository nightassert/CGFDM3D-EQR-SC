/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: cpu_Malloc.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2021-11-02
*   Discription: Malloc memory on CPU

*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*
=================================================================*/

#ifndef GPU_CUDA

#include <stdio.h>
#include <stdlib.h>
int Malloc(void **mem, long long size)
{
	*mem = malloc(size);
	if (*mem == NULL)
	{
		printf("can not malloc, Error: %s:%d\n", __FILE__, __LINE__);
	}
	return 0;
}

#endif
