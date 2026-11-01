/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: main.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2021-11-01
*   Discription: Main function
*
*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*
=================================================================*/

#include "header.h"

#ifdef GPU_CUDA
// 在 MPI_Init 之前为本进程建立 CUDA context。
// 原因：OpenMPI 5 的 accelerator(cuda) 组件在 MPI_Init 阶段用 cuCtxGetCurrent
// 探测当前线程的 CUDA context。CGFDM3D 原本在 init_MPI(MPI_Init) 之后才在
// init_gpu 里 cudaSetDevice 建 context，导致探测失败、CUDA IPC 直传被禁用，
// halo 的 device 缓冲区回退到 smcuda 碎块中转路径（大量 cuMemcpyAsync+sync，
// 严重拖慢多卡通信）。在 MPI_Init 前预建 context 即可恢复 IPC 直传。
static void establish_cuda_context_before_mpi(void)
{
	// 优先用本地 rank 预绑定本进程将使用的 GPU（与 params.json 的 gpu_nodes
	// 顺序一致）；取不到时退化为在 device 0 上建 context，仅为让 MPI 探测成功。
	int dev = 0;
	const char *lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK"); // OpenMPI
	if (NULL == lr)
		lr = getenv("PMI_LOCAL_RANK"); // 部分启动器
	if (NULL == lr)
		lr = getenv("MPI_LOCALRANKID"); // Intel MPI 兼容
	if (NULL != lr)
	{
		int devCount = 0;
		if (cudaGetDeviceCount(&devCount) == cudaSuccess && devCount > 0)
			dev = atoi(lr) % devCount;
	}
	cudaSetDevice(dev); // 建立该设备 context
	cudaFree(0);		// 强制 runtime 真正初始化 context
}
#endif // GPU_CUDA

int main(int argc, char **argv)
{

#ifdef GPU_CUDA
	establish_cuda_context_before_mpi();
#endif

	PARAMS params = {};
	GRID grid = {0};

	MPI_COORD thisMPICoord = {0};
	MPI_NEIGHBOR mpiNeighbor = {0};
	MPI_Comm comm_cart;

	getParams(&params);
	init_MPI(&argc, &argv, params, &comm_cart, &thisMPICoord, &mpiNeighbor);
	init_grid(params, &grid, thisMPICoord);

	createDir(params);

#ifdef GPU_CUDA
	init_gpu(grid.PX, grid.PY, grid.PZ);
#endif

	MPI_Barrier(comm_cart);

	run(comm_cart, thisMPICoord, mpiNeighbor, grid, params);

	MPI_Barrier(comm_cart);
	MPI_Comm_free(&comm_cart);
	MPI_Finalize();
	return 0;
}
