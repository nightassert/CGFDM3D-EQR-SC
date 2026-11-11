/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: volume_io.cpp
*   Discription: Optional 3D volume output of the velocity field.
*
*   The output is configured by "volume.json" in the run directory.  When the
*   file is absent (or "enabled" is 0) nothing is written, so the default
*   behaviour of the program is unchanged.  When it is present, every rank
*   writes its intersection of the requested block to its own NetCDF-4 file
*   output/volume_mpi_X_Y_Z.nc with the dimensions (time, z, y, x) and the
*   variables vx, vy, vz.  Local grid indices are stored in the integer
*   variables I, J and K.
*
*   volume.json:
*   {
*       "enabled": 1,          // optional, default 1
*       "it_skip": 10,         // write every N-th time step, default 1
*       "xStart": 0, "xEnd": 800,
*       "yStart": 0, "yEnd": 600,
*       "zStart": 300, "zEnd": 400   // K = NZ-1 is the free surface
*   }
*
*   All ranges are inclusive, in global grid indices (same convention as
*   station.json).  Missing range keys default to the whole grid.
*
*   The CGFDM branch stores velocity in W directly; the WENO branch stores
*   conservative momentum rho*vx, so the conversion v = W * buoyancy
*   (buoyancy = CJM[...,12] * Crho) is applied here, in the same way as in
*   storage_station() and wave_conserv2phy().
*
*   The volume output needs the NetCDF C library.  Build with NETCDF=ON
*   (default) and point NETCDF_HOME at the installation if it is not in
*   /usr.  With NETCDF=OFF the program still builds and simply ignores
*   volume.json.
=================================================================*/

#include "header.h"
#include <string.h>

#define VOLUME_JSON_FILE "volume.json"

#ifdef USE_NETCDF

#include <netcdf.h>

#define NC_CHECK(call)                                                                 \
	{                                                                                  \
		int ncStatus = (call);                                                         \
		if (ncStatus != NC_NOERR)                                                      \
		{                                                                              \
			fprintf(stderr, "NetCDF error %s:%d: %s\n", __FILE__, __LINE__,            \
					nc_strerror(ncStatus));                                            \
			fflush(stderr);                                                            \
			MPI_Abort(MPI_COMM_WORLD, 10003);                                          \
		}                                                                              \
	}

static char *readJsonText(const char *fileName)
{
	FILE *fp = fopen(fileName, "r");
	if (NULL == fp)
		return NULL;

	fseek(fp, 0, SEEK_END);
	long len = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	char *text = (char *)malloc(len + 1);
	if (NULL == text)
	{
		fclose(fp);
		return NULL;
	}

	size_t nread = fread(text, sizeof(char), len, fp);
	text[nread] = '\0';
	fclose(fp);

	return text;
}

static int jsonInt(cJSON *object, const char *key, int defaultValue)
{
	cJSON *item = cJSON_GetObjectItem(object, key);
	if (NULL == item)
		return defaultValue;

	int type = item->type & 255;
	if (cJSON_Number == type)
		return item->valueint;
	if (cJSON_True == type)
		return 1;
	if (cJSON_False == type)
		return 0;

	return defaultValue;
}

static void formatBytes(double bytes, char *buf, int size)
{
	static const char *unit[] = {"B", "KB", "MB", "GB", "TB", "PB"};
	int i = 0;
	while (bytes >= 1024.0 && i < 5)
	{
		bytes /= 1024.0;
		i++;
	}
	snprintf(buf, size, "%.2f %s", bytes, unit[i]);
}

void initVolumeOutput(PARAMS params, GRID grid, MPI_COORD thisMPICoord, VOLUME_OUT *volume)
{
	int thisRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &thisRank);

	memset(volume, 0, sizeof(VOLUME_OUT));

	char *text = readJsonText(VOLUME_JSON_FILE);
	if (NULL == text)
	{
		if (0 == thisRank)
			printf("volume.json not found: 3D volume output disabled.\n");
		return;
	}

	cJSON *object = cJSON_Parse(text);
	free(text);
	if (NULL == object)
	{
		fprintf(stderr, "rank %d: can't parse %s\n", thisRank, VOLUME_JSON_FILE);
		fflush(stderr);
		MPI_Abort(MPI_COMM_WORLD, 10004);
	}

	int enabled = jsonInt(object, "enabled", jsonInt(object, "enable", 1));
	if (!enabled)
	{
		if (0 == thisRank)
			printf("volume.json: \"enabled\" is 0, 3D volume output disabled.\n");
		cJSON_Delete(object);
		return;
	}

	volume->itSkip = jsonInt(object, "it_skip", 1);
	if (volume->itSkip < 1)
	{
		if (0 == thisRank)
			printf("volume.json: it_skip = %d is invalid, using 1.\n", volume->itSkip);
		volume->itSkip = 1;
	}

	int xStart = jsonInt(object, "xStart", 0);
	int xEnd = jsonInt(object, "xEnd", grid.NX - 1);
	int yStart = jsonInt(object, "yStart", 0);
	int yEnd = jsonInt(object, "yEnd", grid.NY - 1);
	int zStart = jsonInt(object, "zStart", 0);
	int zEnd = jsonInt(object, "zEnd", grid.NZ - 1);
	cJSON_Delete(object);

	// Keep the block inside the grid; report a correction instead of aborting.
	if (xStart > xEnd)
	{
		int tmp = xStart;
		xStart = xEnd;
		xEnd = tmp;
	}
	if (yStart > yEnd)
	{
		int tmp = yStart;
		yStart = yEnd;
		yEnd = tmp;
	}
	if (zStart > zEnd)
	{
		int tmp = zStart;
		zStart = zEnd;
		zEnd = tmp;
	}

	int clampX1 = MAX(xStart, 0), clampX2 = MIN(xEnd, grid.NX - 1);
	int clampY1 = MAX(yStart, 0), clampY2 = MIN(yEnd, grid.NY - 1);
	int clampZ1 = MAX(zStart, 0), clampZ2 = MIN(zEnd, grid.NZ - 1);

	if (xStart != clampX1 || xEnd != clampX2 ||
		yStart != clampY1 || yEnd != clampY2 ||
		zStart != clampZ1 || zEnd != clampZ2)
	{
		if (0 == thisRank)
			printf("volume.json: block %d..%d / %d..%d / %d..%d is outside the grid, "
				   "clamped to %d..%d / %d..%d / %d..%d.\n",
				   xStart, xEnd, yStart, yEnd, zStart, zEnd,
				   clampX1, clampX2, clampY1, clampY2, clampZ1, clampZ2);
	}

	volume->xStart = clampX1;
	volume->xEnd = clampX2;
	volume->yStart = clampY1;
	volume->yEnd = clampY2;
	volume->zStart = clampZ1;
	volume->zEnd = clampZ2;

	if (volume->xStart > volume->xEnd ||
		volume->yStart > volume->yEnd ||
		volume->zStart > volume->zEnd)
	{
		if (0 == thisRank)
			printf("volume.json: the block does not intersect the grid, "
				   "3D volume output disabled.\n");
		return;
	}

	// Local intersection owned by this rank.
	int iHiGlobal = grid.frontNX + grid.nx - 1;
	int jHiGlobal = grid.frontNY + grid.ny - 1;
	int kHiGlobal = grid.frontNZ + grid.nz - 1;

	volume->iLo = MAX(volume->xStart, grid.frontNX);
	volume->jLo = MAX(volume->yStart, grid.frontNY);
	volume->kLo = MAX(volume->zStart, grid.frontNZ);

	int iHi = MIN(volume->xEnd, iHiGlobal);
	int jHi = MIN(volume->yEnd, jHiGlobal);
	int kHi = MIN(volume->zEnd, kHiGlobal);

	if (volume->iLo > iHi || volume->jLo > jHi || volume->kLo > kHi)
		return; // this rank owns no point of the block

	volume->nx = iHi - volume->iLo + 1;
	volume->ny = jHi - volume->jLo + 1;
	volume->nz = kHi - volume->kLo + 1;

	if (0 == thisRank)
	{
		long long nPoints = (long long)(volume->xEnd - volume->xStart + 1) *
							(volume->yEnd - volume->yStart + 1) *
							(volume->zEnd - volume->zStart + 1);
		int NT = (int)(params.TMAX / params.DT);
		long long frames = (long long)(NT - 1) / volume->itSkip + 1;
		double perFrame = (double)nPoints * 3.0 * sizeof(float);
		double total = perFrame * (double)frames;
		char perFrameStr[32], totalStr[32];

		formatBytes(perFrame, perFrameStr, sizeof(perFrameStr));
		formatBytes(total, totalStr, sizeof(totalStr));

		printf("=============================================\n");
		printf("3D volume output: ON (%s)\n", VOLUME_JSON_FILE);
		printf("  block    : x %d..%d, y %d..%d, z %d..%d (%lld points)\n",
			   volume->xStart, volume->xEnd, volume->yStart, volume->yEnd,
			   volume->zStart, volume->zEnd, nPoints);
		printf("  it_skip  : %d\n", volume->itSkip);
		printf("  frames   : %lld of %d steps\n", frames, NT);
		printf("  per frame: %s (float32, vx + vy + vz)\n", perFrameStr);
		printf("  total    : %s\n", totalStr);
		printf("  files    : %s/volume_mpi_X_Y_Z.nc, one per MPI rank\n", params.OUT);
		printf("  note     : each rank writes only its local part of the block; "
			   "K = NZ-1 is the free surface\n");
		printf("=============================================\n");
		fflush(stdout);
	}

	char fileName[512];
	snprintf(fileName, sizeof(fileName), "%s/volume_mpi_%d_%d_%d.nc",
			 params.OUT, thisMPICoord.X, thisMPICoord.Y, thisMPICoord.Z);

	int ncStatus = nc_create(fileName, NC_CLOBBER | NC_NETCDF4, &volume->ncid);
	if (ncStatus != NC_NOERR)
	{
		printf("rank %d: NetCDF-4 creation failed (%s), falling back to classic NetCDF "
			   "(note: classic files are limited to 2 GB).\n",
			   thisRank, nc_strerror(ncStatus));
		NC_CHECK(nc_create(fileName, NC_CLOBBER, &volume->ncid));
	}

	NC_CHECK(nc_def_dim(volume->ncid, "time", NC_UNLIMITED, &volume->timeDimID));
	NC_CHECK(nc_def_dim(volume->ncid, "z", (size_t)volume->nz, &volume->zDimID));
	NC_CHECK(nc_def_dim(volume->ncid, "y", (size_t)volume->ny, &volume->yDimID));
	NC_CHECK(nc_def_dim(volume->ncid, "x", (size_t)volume->nx, &volume->xDimID));

	int dims4[4] = {volume->timeDimID, volume->zDimID, volume->yDimID, volume->xDimID};
	NC_CHECK(nc_def_var(volume->ncid, "vx", NC_FLOAT, 4, dims4, &volume->vxVarID));
	NC_CHECK(nc_def_var(volume->ncid, "vy", NC_FLOAT, 4, dims4, &volume->vyVarID));
	NC_CHECK(nc_def_var(volume->ncid, "vz", NC_FLOAT, 4, dims4, &volume->vzVarID));

	int dimX[1] = {volume->xDimID};
	int dimY[1] = {volume->yDimID};
	int dimZ[1] = {volume->zDimID};
	NC_CHECK(nc_def_var(volume->ncid, "I", NC_INT, 1, dimX, &volume->iVarID));
	NC_CHECK(nc_def_var(volume->ncid, "J", NC_INT, 1, dimY, &volume->jVarID));
	NC_CHECK(nc_def_var(volume->ncid, "K", NC_INT, 1, dimZ, &volume->kVarID));
	NC_CHECK(nc_def_var(volume->ncid, "time", NC_FLOAT, 1, &volume->timeDimID, &volume->timeVarID));

	const char *velocityUnits = "m/s";
	NC_CHECK(nc_put_att_text(volume->ncid, volume->vxVarID, "units",
							 strlen(velocityUnits), velocityUnits));
	NC_CHECK(nc_put_att_text(volume->ncid, volume->vyVarID, "units",
							 strlen(velocityUnits), velocityUnits));
	NC_CHECK(nc_put_att_text(volume->ncid, volume->vzVarID, "units",
							 strlen(velocityUnits), velocityUnits));
	const char *xLong = "global grid index in X";
	const char *yLong = "global grid index in Y";
	const char *zLong = "global grid index in Z (NZ-1 is the free surface)";
	NC_CHECK(nc_put_att_text(volume->ncid, volume->iVarID, "long_name",
							 strlen(xLong), xLong));
	NC_CHECK(nc_put_att_text(volume->ncid, volume->jVarID, "long_name",
							 strlen(yLong), yLong));
	NC_CHECK(nc_put_att_text(volume->ncid, volume->kVarID, "long_name",
							 strlen(zLong), zLong));
	const char *timeUnits = "s";
	NC_CHECK(nc_put_att_text(volume->ncid, volume->timeVarID, "units",
							 strlen(timeUnits), timeUnits));

	const char *title = "CGFDM3D-EQR-SC 3D velocity volume";
	NC_CHECK(nc_put_att_text(volume->ncid, NC_GLOBAL, "title", strlen(title), title));
	const char *schemeName = (params.scheme == SCHEME_WENO) ? "WENO" : "CGFDM";
	NC_CHECK(nc_put_att_text(volume->ncid, NC_GLOBAL, "scheme",
							 strlen(schemeName), schemeName));
	NC_CHECK(nc_put_att_int(volume->ncid, NC_GLOBAL, "global_NX", NC_INT, 1, &grid.NX));
	NC_CHECK(nc_put_att_int(volume->ncid, NC_GLOBAL, "global_NY", NC_INT, 1, &grid.NY));
	NC_CHECK(nc_put_att_int(volume->ncid, NC_GLOBAL, "global_NZ", NC_INT, 1, &grid.NZ));
	NC_CHECK(nc_put_att_int(volume->ncid, NC_GLOBAL, "global_xStart", NC_INT, 1, &volume->xStart));
	NC_CHECK(nc_put_att_int(volume->ncid, NC_GLOBAL, "global_xEnd", NC_INT, 1, &volume->xEnd));
	NC_CHECK(nc_put_att_int(volume->ncid, NC_GLOBAL, "global_yStart", NC_INT, 1, &volume->yStart));
	NC_CHECK(nc_put_att_int(volume->ncid, NC_GLOBAL, "global_yEnd", NC_INT, 1, &volume->yEnd));
	NC_CHECK(nc_put_att_int(volume->ncid, NC_GLOBAL, "global_zStart", NC_INT, 1, &volume->zStart));
	NC_CHECK(nc_put_att_int(volume->ncid, NC_GLOBAL, "global_zEnd", NC_INT, 1, &volume->zEnd));
	NC_CHECK(nc_put_att_int(volume->ncid, NC_GLOBAL, "it_skip", NC_INT, 1, &volume->itSkip));
	NC_CHECK(nc_put_att_int(volume->ncid, NC_GLOBAL, "centerX", NC_INT, 1, &grid.originalX));
	NC_CHECK(nc_put_att_int(volume->ncid, NC_GLOBAL, "centerY", NC_INT, 1, &grid.originalY));

	double dh = (double)grid.DH;
	NC_CHECK(nc_put_att_double(volume->ncid, NC_GLOBAL, "DH", NC_DOUBLE, 1, &dh));

	double depthKm = (double)params.Depth;
	NC_CHECK(nc_put_att_double(volume->ncid, NC_GLOBAL, "Depth_km", NC_DOUBLE, 1, &depthKm));

	NC_CHECK(nc_enddef(volume->ncid));

	// Global grid indices of the local block, written once.
	int maxLen = MAX(volume->nx, MAX(volume->ny, volume->nz));
	int *indexBuf = (int *)malloc(sizeof(int) * maxLen);
	if (NULL == indexBuf)
	{
		fprintf(stderr, "rank %d: can't allocate volume index buffer\n", thisRank);
		fflush(stderr);
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}
	for (int i = 0; i < volume->nx; i++)
		indexBuf[i] = volume->iLo + i;
	NC_CHECK(nc_put_var_int(volume->ncid, volume->iVarID, indexBuf));
	for (int j = 0; j < volume->ny; j++)
		indexBuf[j] = volume->jLo + j;
	NC_CHECK(nc_put_var_int(volume->ncid, volume->jVarID, indexBuf));
	for (int k = 0; k < volume->nz; k++)
		indexBuf[k] = volume->kLo + k;
	NC_CHECK(nc_put_var_int(volume->ncid, volume->kVarID, indexBuf));
	free(indexBuf);

	long long nPoints = (long long)volume->nx * volume->ny * volume->nz;
	volume->hData = (float *)malloc(sizeof(float) * 3 * nPoints);
	if (NULL == volume->hData)
	{
		fprintf(stderr, "rank %d: can't allocate volume host buffer\n", thisRank);
		fflush(stderr);
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}
	memset(volume->hData, 0, sizeof(float) * 3 * nPoints);

#ifdef GPU_CUDA
	CHECK(Malloc((void **)&volume->dData, sizeof(float) * 3 * nPoints));
#endif

	volume->enabled = 1;
}

__GLOBAL__
void pack_volume_velocity(FLOAT *W, FLOAT *CJM, float *out,
						  int nb_x, int nb_y, int nb_z,
						  int _nx_, int _ny_, int _nz_,
						  int iOff, int jOff, int kOff, int isWENO)
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

	long long num = (long long)nb_x * nb_y * nb_z;

	CALCULATE3D(i, j, k, 0, nb_x, 0, nb_y, 0, nb_z)
	long long index = INDEX(i + iOff, j + jOff, k + kOff);
	long long pos = ((long long)k * nb_y + j) * nb_x + i;

	// CGFDM: W[0..2] are velocities (scaled by 1/Cv, Cv == 1 by default).
	// WENO: W[0..2] are conservative momenta rho*v, so multiply by buoyancy.
	float c = 1.0f / (float)Cv;
	if (isWENO)
		c = CJM[index * CJMSIZE + 12] * Crho;

	out[pos] = (float)W[index * WSIZE + 0] * c;
	out[pos + num] = (float)W[index * WSIZE + 1] * c;
	out[pos + 2 * num] = (float)W[index * WSIZE + 2] * c;
	END_CALCULATE3D()
}

void volumeWrite(GRID grid, PARAMS params, FLOAT *W, FLOAT *CJM, VOLUME_OUT *volume, int it)
{
	if (!volume->enabled)
		return;

	long long nPoints = (long long)volume->nx * volume->ny * volume->nz;
	int isWENO = (params.scheme == SCHEME_WENO);

	int iOff = volume->iLo - grid.frontNX + HALO;
	int jOff = volume->jLo - grid.frontNY + HALO;
	int kOff = volume->kLo - grid.frontNZ + HALO;

#ifdef GPU_CUDA
	dim3 threads(32, 4, 4);
	dim3 blocks;
	blocks.x = (volume->nx + threads.x - 1) / threads.x;
	blocks.y = (volume->ny + threads.y - 1) / threads.y;
	blocks.z = (volume->nz + threads.z - 1) / threads.z;

	pack_volume_velocity<<<blocks, threads>>>(W, CJM, volume->dData,
											  volume->nx, volume->ny, volume->nz,
											  grid._nx_, grid._ny_, grid._nz_,
											  iOff, jOff, kOff, isWENO);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaMemcpy(volume->hData, volume->dData, sizeof(float) * 3 * nPoints,
					 cudaMemcpyDeviceToHost));
#else
	pack_volume_velocity(W, CJM, volume->hData,
						 volume->nx, volume->ny, volume->nz,
						 grid._nx_, grid._ny_, grid._nz_,
						 iOff, jOff, kOff, isWENO);
#endif

	size_t start[4] = {(size_t)(it / volume->itSkip), 0, 0, 0};
	size_t count[4] = {1, (size_t)volume->nz, (size_t)volume->ny, (size_t)volume->nx};

	NC_CHECK(nc_put_vara_float(volume->ncid, volume->vxVarID, start, count, volume->hData));
	NC_CHECK(nc_put_vara_float(volume->ncid, volume->vyVarID, start, count, volume->hData + nPoints));
	NC_CHECK(nc_put_vara_float(volume->ncid, volume->vzVarID, start, count, volume->hData + 2 * nPoints));

	float time = (float)(it * params.DT);
	size_t timeStart = (size_t)(it / volume->itSkip);
	size_t timeCount = 1;
	NC_CHECK(nc_put_vara_float(volume->ncid, volume->timeVarID, &timeStart, &timeCount, &time));
}

void freeVolumeOutput(VOLUME_OUT *volume)
{
	if (volume->enabled)
		NC_CHECK(nc_close(volume->ncid));

	free(volume->hData);
#ifdef GPU_CUDA
	Free(volume->dData);
#endif

	memset(volume, 0, sizeof(VOLUME_OUT));
}

#else // !USE_NETCDF

void initVolumeOutput(PARAMS params, GRID grid, MPI_COORD thisMPICoord, VOLUME_OUT *volume)
{
	int thisRank;
	(void)params;
	(void)grid;
	(void)thisMPICoord;
	MPI_Comm_rank(MPI_COMM_WORLD, &thisRank);

	memset(volume, 0, sizeof(VOLUME_OUT));

	FILE *fp = fopen(VOLUME_JSON_FILE, "r");
	if (NULL != fp)
	{
		fclose(fp);
		if (0 == thisRank)
			printf("volume.json found, but this build has no NetCDF support. "
				   "Rebuild with NETCDF=ON to use 3D volume output.\n");
	}
}

void volumeWrite(GRID grid, PARAMS params, FLOAT *W, FLOAT *CJM, VOLUME_OUT *volume, int it)
{
	(void)grid;
	(void)params;
	(void)W;
	(void)CJM;
	(void)it;
	volume->enabled = 0;
}

void freeVolumeOutput(VOLUME_OUT *volume)
{
	memset(volume, 0, sizeof(VOLUME_OUT));
}

#endif // USE_NETCDF
