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

void init_gpu(int PX, int PY, int PZ)
{
	char jsonFile[1024] = {0};
	strcpy(jsonFile, "params.json");
	FILE *fp;
	fp = fopen(jsonFile, "r");

	if (NULL == fp)
	{
		printf("There is not %s file!\n", jsonFile);
		MPI_Abort(MPI_COMM_WORLD, 100); // exit( 1 );
										// exit( 1 );
	}

	fseek(fp, 0, SEEK_END);
	int len = ftell(fp);

	fseek(fp, 0, SEEK_SET);

	char *jsonStr = (char *)malloc(len * sizeof(char));

	if (NULL == jsonStr)
	{
		printf("Can't allocate json string memory\n");
	}

	fread(jsonStr, sizeof(char), len, fp);

	// printf( "%s\n", jsonStr );
	cJSON *object;
	cJSON *objArray;

	object = cJSON_Parse(jsonStr);
	if (NULL == object)
	{
		printf("Can't parse json file!\n");
		// exit( 1 );
		MPI_Abort(MPI_COMM_WORLD, 1001); // exit( 1 );
		return;
	}

	fclose(fp);

	int nodeCnt = 0;

	if (objArray = cJSON_GetObjectItem(object, "gpu_nodes"))
	{
		nodeCnt = cJSON_GetArraySize(objArray);
		//	printf( "nodeCnt = %d\n", nodeCnt );
	}

	int i, j;
	cJSON *nodeObj, *nodeItem;

	int nameLens;
	char thisMPINodeName[256];

	MPI_Get_processor_name(thisMPINodeName, &nameLens);

	// printf( "this mpi node name is %s\n", thisMPINodeName  );

	//	printf( "node: %s\n", thisMPINodeName );

	int nodeGPUCnt = 0;

	int frontGPNCnt = 0;
	int thisRank;
	int thisNodeRankID;
	MPI_Comm_rank(MPI_COMM_WORLD, &thisRank);

	// printf( "==================================" );
	for (i = 0; i < nodeCnt; i++)
	{
		nodeObj = cJSON_GetArrayItem(objArray, i);

		nodeGPUCnt = cJSON_GetArraySize(nodeObj);
		if (0 == strcmp(nodeObj->string, thisMPINodeName))
		{

			thisNodeRankID = thisRank - frontGPNCnt;

			for (j = 0; j < nodeGPUCnt; j++)
			{
				nodeItem = cJSON_GetArrayItem(nodeObj, j);
				if (thisNodeRankID == j)
				{
					// printf( "%s[%d] is available!\n", nodeObj->string, nodeItem->valueint );
#ifdef GPU_CUDA
					CHECK(cudaSetDevice(nodeItem->valueint));
#endif
				}
			}

			// break;
		}

		frontGPNCnt += nodeGPUCnt;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (frontGPNCnt != PX * PY * PZ)
	{
		printf("The GPU numbers can't match the MPI numbers\n");
		MPI_Abort(MPI_COMM_WORLD, 1002); // exit( 1 );
										 // exit( 1  );
										 // MPI_Finalize( );
	}
}
