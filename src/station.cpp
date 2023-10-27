/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name:station.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time:2021-11-22
*   Discription:
*
================================================================*/
#include "header.h"
void allocStation(STATION *station, STATION *cpu_station, int stationNum, int NT)
{
	int sizeIdx = sizeof(int) * stationNum * CSIZE;

	int *pIndex = NULL;

	CHECK(Malloc((void **)&pIndex, sizeIdx));
	if (pIndex == NULL)
	{
		printf("can't allocate station XYZ memory!\n");
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}
	CHECK(Memset(pIndex, 0, sizeIdx));
	station->XYZ = pIndex;

#ifdef GPU_CUDA
	int *cpu_pIndex = NULL;
	cpu_pIndex = (int *)malloc(sizeIdx);
	if (cpu_pIndex == NULL)
	{
		printf("can't allocate station XYZ memory!\n");
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}
	memset(cpu_pIndex, 0, sizeIdx);
	cpu_station->XYZ = cpu_pIndex;
#else
	cpu_station->XYZ = station->XYZ;
#endif

	long long sizeWave = sizeof(float) * NT * stationNum * WSIZE;
	float *pWave = NULL;

	CHECK(Malloc((void **)&pWave, sizeWave));
	if (pWave == NULL)
	{
		printf("can't allocate station wave memory!\n");
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}
	CHECK(Memset(pWave, 0, sizeWave));
	station->wave = pWave;

#ifdef GPU_CUDA
	float *cpu_pWave = NULL;

	cpu_pWave = (float *)malloc(sizeWave);
	if (cpu_pWave == NULL)
	{
		printf("can't allocate station wave memory!\n");
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}
	memset(cpu_pWave, 0, sizeWave);
	cpu_station->wave = cpu_pWave;
#else
	cpu_station->wave = station->wave;
#endif
}

void freeStation(STATION station, STATION cpu_station)
{
	Free(station.XYZ);
	Free(station.wave);
#ifdef GPU_CUDA
	free(cpu_station.XYZ);
	free(cpu_station.wave);
#endif
}

int readStationIndex(GRID grid)
{
	char jsonFile[1024] = {0};
	strcpy(jsonFile, "station.json");
	FILE *fp;
	fp = fopen(jsonFile, "r");

	if (NULL == fp)
	{
		printf("There is not %s file!\n", jsonFile);
		return 0;
	}

	fseek(fp, 0, SEEK_END);
	int len = ftell(fp);

	fseek(fp, 0, SEEK_SET);

	char *jsonStr = (char *)malloc(len * sizeof(char));

	if (NULL == jsonStr)
	{
		printf("Can't allocate json string memory\n");
		return 0;
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
		return 0;
	}

	fclose(fp);

	int stationCnt = 0;

	if (objArray = cJSON_GetObjectItem(object, "station(point)"))
	{
		stationCnt = cJSON_GetArraySize(objArray);
	}

	cJSON *stationObj, *stationItem;

	int i, j, stationNum;
	int X, Y, Z, thisX, thisY, thisZ;
	int frontNX = grid.frontNX;
	int frontNY = grid.frontNY;
	int frontNZ = grid.frontNZ;

	int _nx = grid._nx;
	int _ny = grid._ny;
	int _nz = grid._nz;

	stationNum = 0;
	for (i = 0; i < stationCnt; i++)
	{
		stationObj = cJSON_GetArrayItem(objArray, i);

		int a = cJSON_GetArraySize(stationObj);
		if (a != 3)
		{
			printf("In file %s, the coodinate index don't equal to 3. However, it equals to %d\n", jsonFile, a);
			return 0;
		}

		stationItem = cJSON_GetArrayItem(stationObj, 0);
		X = stationItem->valueint;
		thisX = X - frontNX + HALO;

		stationItem = cJSON_GetArrayItem(stationObj, 1);
		Y = stationItem->valueint;
		thisY = Y - frontNY + HALO;

		stationItem = cJSON_GetArrayItem(stationObj, 2);
		Z = stationItem->valueint;
		thisZ = Z - frontNZ + HALO;

		if (thisX >= HALO && thisX < _nx &&
			thisY >= HALO && thisY < _ny &&
			thisZ >= HALO && thisZ < _nz)
		{
			stationNum++;
		}
	}
	// printf( "stationNum = %d\n", stationNum );

	return stationNum;
}

void initStationIndex(GRID grid, STATION station)
{
	char jsonFile[1024] = {0};
	strcpy(jsonFile, "station.json");
	FILE *fp;
	fp = fopen(jsonFile, "r");

	if (NULL == fp)
	{
		printf("There is not %s file!\n", jsonFile);
		return;
	}

	fseek(fp, 0, SEEK_END);
	int len = ftell(fp);

	fseek(fp, 0, SEEK_SET);

	char *jsonStr = (char *)malloc(len * sizeof(char));

	if (NULL == jsonStr)
	{
		printf("Can't allocate json string memory\n");
		return;
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
		return;
	}

	fclose(fp);

	int stationCnt = 0;

	if (objArray = cJSON_GetObjectItem(object, "station(point)"))
	{
		stationCnt = cJSON_GetArraySize(objArray);
		// printf( "stationCnt = %d\n", stationCnt );
	}

	cJSON *stationObj, *stationItem;
	int i, j;
	int thisX, thisY, thisZ;

	int frontNX = grid.frontNX;
	int frontNY = grid.frontNY;
	int frontNZ = grid.frontNZ;

	int _nx = grid._nx;
	int _ny = grid._ny;
	int _nz = grid._nz;

	int X, Y, Z;

	// printf( "X = %p\n", station.X  );
	// printf( "Y = %p\n", station.Y  );
	// printf( "Z = %p\n", station.Z  );

	int stationIdx = 0;
	for (i = 0; i < stationCnt; i++)
	{
		stationObj = cJSON_GetArrayItem(objArray, i);

		int a = cJSON_GetArraySize(stationObj);
		if (a != 3)
		{
			printf("In file %s, the coodinate index don't equal to 3. However, it equals to %d\n", jsonFile, a);
			return;
		}

		stationItem = cJSON_GetArrayItem(stationObj, 0);
		X = stationItem->valueint;
		thisX = X - frontNX + HALO;

		stationItem = cJSON_GetArrayItem(stationObj, 1);
		Y = stationItem->valueint;
		thisY = Y - frontNY + HALO;

		stationItem = cJSON_GetArrayItem(stationObj, 2);
		Z = stationItem->valueint;
		thisZ = Z - frontNZ + HALO;

		if (thisX >= HALO && thisX < _nx &&
			thisY >= HALO && thisY < _ny &&
			thisZ >= HALO && thisZ < _nz)
		{
			station.XYZ[stationIdx * CSIZE + 0] = thisX;
			station.XYZ[stationIdx * CSIZE + 1] = thisY;
			station.XYZ[stationIdx * CSIZE + 2] = thisZ;

			stationIdx++;
		}
	}
}

void stationCPU2GPU(STATION station, STATION station_cpu, int stationNum)
{
	int size = sizeof(int) * stationNum * CSIZE;
#ifdef GPU_CUDA
	CHECK(Memcpy(station.XYZ, station_cpu.XYZ, size, cudaMemcpyHostToDevice));
#endif
}

__GLOBAL__
void storage_station(int stationNum, STATION station, FLOAT *W, int _nx_, int _ny_, int _nz_, int NT, int it
#ifdef SCFDM
					 ,
					 FLOAT *CJM
#endif
)
{

#ifdef GPU_CUDA
	int i = threadIdx.x + blockIdx.x * blockDim.x;
#else
	int i = 0;
#endif
	long long index = 0, pos = 0;
	int X, Y, Z;

	float C1 = 1.0 / Cv;
	float C2 = 1.0 / Cs;

#ifdef SCFDM
	float mu, lambda, buoyancy;
	float u_conserv[9]; // Exx, Eyy, Ezz, Exy, Eyz, Exz, Px, Py, Pz
	float u_phy[9];		// Txx, Tyy, Tzz, Txy, Txz, Tyz, Vx, Vy, Vz
#endif

	CALCULATE1D(i, 0, stationNum)
	X = station.XYZ[i * CSIZE + 0];
	Y = station.XYZ[i * CSIZE + 1];
	Z = station.XYZ[i * CSIZE + 2];

	index = INDEX(X, Y, Z);
	pos = it + i * NT;

#ifdef SCFDM
	mu = CJM[index * CJMSIZE + 10];
	lambda = CJM[index * CJMSIZE + 11];
	buoyancy = CJM[index * CJMSIZE + 12];
	buoyancy *= Crho;

	for (int n = 0; n < 9; n++)
	{
		u_conserv[n] = W[index * WSIZE + n];
	}

	u_phy[0] = lambda * u_conserv[1] + lambda * u_conserv[2] + u_conserv[0] * (lambda + 2 * mu);
	u_phy[1] = lambda * u_conserv[0] + lambda * u_conserv[2] + u_conserv[1] * (lambda + 2 * mu);
	u_phy[2] = lambda * u_conserv[0] + lambda * u_conserv[1] + u_conserv[2] * (lambda + 2 * mu);
	u_phy[3] = 2 * mu * u_conserv[3];
	u_phy[4] = 2 * mu * u_conserv[5];
	u_phy[5] = 2 * mu * u_conserv[4];
	u_phy[6] = u_conserv[6] * buoyancy;
	u_phy[7] = u_conserv[7] * buoyancy;
	u_phy[8] = u_conserv[8] * buoyancy;

	station.wave[pos * WSIZE + 0] = (float)u_phy[0];
	station.wave[pos * WSIZE + 1] = (float)u_phy[1];
	station.wave[pos * WSIZE + 2] = (float)u_phy[2];
	station.wave[pos * WSIZE + 3] = (float)u_phy[3];
	station.wave[pos * WSIZE + 4] = (float)u_phy[4];
	station.wave[pos * WSIZE + 5] = (float)u_phy[5];
	station.wave[pos * WSIZE + 6] = (float)u_phy[6];
	station.wave[pos * WSIZE + 7] = (float)u_phy[7];
	station.wave[pos * WSIZE + 8] = (float)u_phy[8];
#else

	station.wave[pos * WSIZE + 0] = (float)W[index * WSIZE + 0] * C1;
	station.wave[pos * WSIZE + 1] = (float)W[index * WSIZE + 1] * C1;
	station.wave[pos * WSIZE + 2] = (float)W[index * WSIZE + 2] * C1;
	station.wave[pos * WSIZE + 3] = (float)W[index * WSIZE + 3] * C2;
	station.wave[pos * WSIZE + 4] = (float)W[index * WSIZE + 4] * C2;
	station.wave[pos * WSIZE + 5] = (float)W[index * WSIZE + 5] * C2;
	station.wave[pos * WSIZE + 6] = (float)W[index * WSIZE + 6] * C2;
	station.wave[pos * WSIZE + 7] = (float)W[index * WSIZE + 7] * C2;
	station.wave[pos * WSIZE + 8] = (float)W[index * WSIZE + 8] * C2;
#endif

	END_CALCULATE1D()
}

void storageStation(GRID grid, int NT, int stationNum, STATION station, FLOAT *W, int it
#ifdef SCFDM
					,
					FLOAT *CJM
#endif
)
{
	long long num = stationNum;

	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

#ifdef GPU_CUDA
	dim3 threads(32, 1, 1);
	dim3 blocks;
	blocks.x = (num + threads.x - 1) / threads.x;
	blocks.y = 1;
	blocks.z = 1;
	storage_station<<<blocks, threads>>>(stationNum, station, W, _nx_, _ny_, _nz_, NT, it
#ifdef SCFDM
										 ,
										 CJM
#endif
	);
	CHECK(cudaDeviceSynchronize());
#else
	storage_station(stationNum, station, W, _nx_, _ny_, _nz_, NT, it);
#endif
}

void stationGPU2CPU(STATION station, STATION station_cpu, int stationNum, int NT)
{
	long long sizeWave = sizeof(float) * NT * stationNum * WSIZE;

#ifdef GPU_CUDA
	CHECK(Memcpy(station_cpu.wave, station.wave, sizeWave, cudaMemcpyDeviceToHost));
#endif
}

void stationWrite(PARAMS params, GRID grid, MPI_COORD thisMPICoord, STATION station, int NT, int stationNum)
{
	FILE *fp;
	char fileName[256];
	sprintf(fileName, "%s/station_mpi_%d_%d_%d.bin", params.OUT, thisMPICoord.X, thisMPICoord.Y, thisMPICoord.Z);

	int i = 0;

	for (i = 0; i < stationNum; i++)
	{
		station.XYZ[i * CSIZE + 0] = grid.frontNX + station.XYZ[i * CSIZE + 0] - HALO;
		station.XYZ[i * CSIZE + 1] = grid.frontNY + station.XYZ[i * CSIZE + 1] - HALO;
		station.XYZ[i * CSIZE + 2] = grid.frontNZ + station.XYZ[i * CSIZE + 2] - HALO;
	}

	fp = fopen(fileName, "wb");

	fwrite(station.XYZ, sizeof(int), stationNum * CSIZE, fp);
	fwrite(station.wave, sizeof(float), NT * stationNum * WSIZE, fp);

	fclose(fp);
}
