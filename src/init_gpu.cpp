/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: init_gpu.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2021-10-30
*   Discription: Initialize GPU
*
*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*
=================================================================*/

#include "header.h"

// Select the CUDA device for this MPI rank.  The optional "gpu_nodes"
// block in params.json maps MPI ranks to devices by host name; when it is
// absent (a workstation with one GPU, for example) the rank simply picks
// devices round-robin.
void init_gpu(int PX, int PY, int PZ)
{
	char jsonFile[1024] = {0};
	strcpy(jsonFile, "params.json");
	FILE *fp = fopen(jsonFile, "r");

	if (NULL == fp)
	{
		printf("There is not %s file!\n", jsonFile);
		MPI_Abort(MPI_COMM_WORLD, 100);
	}

	fseek(fp, 0, SEEK_END);
	int len = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	char *jsonStr = (char *)malloc(len + 1);
	if (NULL == jsonStr)
	{
		printf("Can't allocate json string memory\n");
		fclose(fp);
		MPI_Abort(MPI_COMM_WORLD, 1001);
	}

	fread(jsonStr, sizeof(char), len, fp);
	jsonStr[len] = '\0';
	fclose(fp);

	cJSON *object = cJSON_Parse(jsonStr);
	if (NULL == object)
	{
		printf("Can't parse json file!\n");
		free(jsonStr);
		MPI_Abort(MPI_COMM_WORLD, 1001);
	}

	cJSON *objArray = cJSON_GetObjectItem(object, "gpu_nodes");
	int nodeCnt = objArray ? cJSON_GetArraySize(objArray) : 0;

	int thisRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &thisRank);

	char thisMPINodeName[256] = {0};
	int nameLens = 0;
	MPI_Get_processor_name(thisMPINodeName, &nameLens);

	int deviceSet = 0;
	int frontGPNCnt = 0;

	for (int i = 0; i < nodeCnt; i++)
	{
		cJSON *nodeObj = cJSON_GetArrayItem(objArray, i);
		int nodeGPUCnt = cJSON_GetArraySize(nodeObj);

		if (0 == strcmp(nodeObj->string, thisMPINodeName))
		{
			int thisNodeRankID = thisRank - frontGPNCnt;
			for (int j = 0; j < nodeGPUCnt; j++)
			{
				cJSON *nodeItem = cJSON_GetArrayItem(nodeObj, j);
				if (thisNodeRankID == j)
				{
#ifdef GPU_CUDA
					CHECK(cudaSetDevice(nodeItem->valueint));
#endif
					deviceSet = 1;
				}
			}
		}

		frontGPNCnt += nodeGPUCnt;
	}

#ifdef GPU_CUDA
	if (!deviceSet)
	{
		int deviceCount = 0;
		CHECK(cudaGetDeviceCount(&deviceCount));
		int device = (deviceCount > 0) ? (thisRank % deviceCount) : 0;
		CHECK(cudaSetDevice(device));
	}
#endif

	if (nodeCnt > 0 && frontGPNCnt != PX * PY * PZ && 0 == thisRank)
	{
		printf("Warning: gpu_nodes lists %d GPUs but the run uses %d MPI ranks. "
			   "Falling back to round-robin device selection where needed.\n",
			   frontGPNCnt, PX * PY * PZ);
	}

	cJSON_Delete(object);
	free(jsonStr);

	MPI_Barrier(MPI_COMM_WORLD);
}
