/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name:terrain.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time:2021-09-05
*   Discription:
*
================================================================*/
#include "header.h"

#define degreePerBlcokX 5
#define degreePerBlcokY 5

#define pointsPerBlockX 6000
#define pointsPerBlockY 6000
#define extendPoint 2

#define errorNegativeElevation -32768
#define errorPositiveElevation 32768
#define ERROR_ELEVATION

void outputTotalTerrain(PARAMS params, MPI_COORD thisMPICoord, GRID grid, float *totalTerrain, int totalPointX, int totalPointY)
{
	if (thisMPICoord.X == 0 && thisMPICoord.Y == 0 && thisMPICoord.Z == grid.PZ - 1)
	{
		char totalTerrainFileName[256];
		sprintf(totalTerrainFileName, "%s/totalTerrain.bin", params.OUT);
		FILE *totalTerrainFile = fopen(totalTerrainFileName, "wb");

		fwrite(totalTerrain, sizeof(float), totalPointX * totalPointY, totalTerrainFile);

		fclose(totalTerrainFile);
	}
}

void outputLonLat(PARAMS params, MPI_COORD thisMPICoord, GRID grid, LONLAT LonLat)
{
	int i, j, index;
	int halo = grid.halo;

	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	int _nx = grid._nx;
	int _ny = grid._ny;
	int _nz = grid._nz;

	int num = _nx_ * _ny_;

	FILE *lonFile, *latFile;
	char lonFileName[256], latFileName[256];
	sprintf(lonFileName, "%s/lon_mpi_%d_%d_%d.bin", params.OUT, thisMPICoord.X, thisMPICoord.Y, thisMPICoord.Z);
	sprintf(latFileName, "%s/lat_mpi_%d_%d_%d.bin", params.OUT, thisMPICoord.X, thisMPICoord.Y, thisMPICoord.Z);
	lonFile = fopen(lonFileName, "wb");
	latFile = fopen(latFileName, "wb");

	if (!(lonFile && latFile))
	{
		cout << "Can't creat Longitude Latitude File!" << endl;
	}

	float lon, lat;

	FOR_LOOP2D(i, j, halo, _nx, halo, _ny)
	index = Index2D(i, j, _nx_, _ny_);
	lon = LonLat.lon[index];
	lat = LonLat.lat[index];
	fwrite(&(lon), sizeof(float), 1, lonFile);
	fwrite(&(lat), sizeof(float), 1, latFile);
	END_LOOP2D()

	// fwrite( LonLat.lon, sizeof( float ), num, lonFile );
	// fwrite( LonLat.lat, sizeof( float ), num, latFile );

	fclose(lonFile);
	fclose(latFile);
}

void outputTerrain(PARAMS params, MPI_COORD thisMPICoord, GRID grid, float *terrain)
{
	int i, j, index;
	int halo = grid.halo;

	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	int _nx = grid._nx;
	int _ny = grid._ny;
	int _nz = grid._nz;

	int num = _nx_ * _ny_;

	FILE *terrainFile;
	char terrainFileName[256];
	sprintf(terrainFileName, "%s/terrain_mpi_%d_%d_%d.bin", params.OUT, thisMPICoord.X, thisMPICoord.Y, thisMPICoord.Z);
	terrainFile = fopen(terrainFileName, "wb");

	if (!terrainFile)
	{
		cout << "Can't creat Terrain File!" << endl;
	}

	FOR_LOOP2D(i, j, halo, _nx, halo, _ny)
	index = Index2D(i, j, _nx_, _ny_);
	fwrite(terrain + index, sizeof(float), 1, terrainFile);
	END_LOOP2D()
	// fwrite( terrain, sizeof( float ), num, terrainFile );

	fclose(terrainFile);
}

void calculateCutLonLat(GRID grid, PJ *P, PJ_DIRECTION PD, PJ_COORD pj_LonLat[4])
{

	int _originalX = grid._originalX;
	int _originalY = grid._originalY;

	int _NX_ = grid._NX_;
	int _NY_ = grid._NY_;

	double leftDis = -(_originalX + extendPoint) * grid.DH;
	double rightDis = ((_NX_ - _originalX) + extendPoint) * grid.DH;

	double downDis = -(_originalY + extendPoint) * grid.DH;
	double upDis = ((_NY_ - _originalY) + extendPoint) * grid.DH;

	PJ_XY leftDownXY, leftUpXY, rightDown, rightUpXY;

	leftDownXY.x = leftDis;
	leftDownXY.y = downDis;
	leftUpXY.x = leftDis;
	leftUpXY.y = upDis;

	rightDown.x = rightDis;
	rightDown.y = downDis;
	rightUpXY.x = rightDis;
	rightUpXY.y = upDis;

	PJ_XY xy[4] = {leftDownXY, leftUpXY, rightDown, rightUpXY};

	int i = 0;
	for (i = 0; i < 4; i++)
		pj_LonLat[i].xy = xy[i];

	proj_trans_array(P, PD, 4, pj_LonLat);

	// for ( i = 0; i < 4; i ++ )
	//	printf( "longitude: %lf, latitude: %lf\n", pj_LonLat[i].lp.lam * RADIAN2DEGREE, pj_LonLat[i].lp.phi * RADIAN2DEGREE );
}

void calculateCutLonLatLoc(GRID grid, PJ *P, PJ_DIRECTION PD, PJ_COORD pj_LonLat[4])
{

	int I, J;
	int frontNX = grid.frontNX;
	int frontNY = grid.frontNY;
	int originalX = grid.originalX;
	int originalY = grid.originalY;
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;

	float DH = grid.DH;

	I = frontNX;
	double leftDis = (I - HALO) * DH - originalX * DH;
	I = frontNX + _nx_;
	double rightDis = (I - HALO) * DH - originalX * DH;

	J = frontNY;
	double downDis = (J - HALO) * DH - originalY * DH;
	J = frontNY + _ny_;
	double upDis = (J - HALO) * DH - originalY * DH;

	// printf( "frontNX = %d, frontNY = %d\n", frontNX, frontNY  );

	PJ_XY leftDownXY, leftUpXY, rightDown, rightUpXY;

	leftDownXY.x = leftDis;
	leftDownXY.y = downDis;
	leftUpXY.x = leftDis;
	leftUpXY.y = upDis;

	rightDown.x = rightDis;
	rightDown.y = downDis;
	rightUpXY.x = rightDis;
	rightUpXY.y = upDis;

	PJ_XY xy[4] = {leftDownXY, leftUpXY, rightDown, rightUpXY};

	int i = 0;
	for (i = 0; i < 4; i++)
		pj_LonLat[i].xy = xy[i];

	proj_trans_array(P, PD, 4, pj_LonLat);
}

// Eastern Hemisphere
void readSRTM90(PARAMS params, float *totalTerrain)
{
	int lon, lat;

	int blockX = params.blockX;
	int blockY = params.blockY;

	int lonStart = params.lonStart;
	int latStart = params.latStart;

	int lonEnd = lonStart + (blockX - 1) * degreePerBlcokX;
	int latEnd = latStart + (blockY - 1) * degreePerBlcokY;

	char srtm90FileName[512];

	int totalPointX = (pointsPerBlockX - 1) * blockX + 1;
	int totalPointY = (pointsPerBlockY - 1) * blockY + 1;

	float *oneBlockTerrain = (float *)malloc(pointsPerBlockX * pointsPerBlockY * sizeof(float));
	memset(oneBlockTerrain, 0, pointsPerBlockX * pointsPerBlockY * sizeof(float));

	int i = 0, j = 0;

	// #ifdef LON_FAST

	for (lat = latStart; lat <= latEnd; lat += 5)
	{

		i = 0;
		for (lon = lonStart; lon <= lonEnd; lon += 5)
		{

			memset(srtm90FileName, 0, sizeof(char) * 512);
			sprintf(srtm90FileName, "%s/srtm_%dN%dE.bin", params.TerrainDir, lat, lon);
			FILE *fp = fopen(srtm90FileName, "rb");
			if (NULL == fp)
			{
				printf("There is no such file %s\n", srtm90FileName);
				exit(1);
			}
			int startI = i * (pointsPerBlockX - 1);
			int startJ = j * (pointsPerBlockY - 1);

			int endX = (i + 1) * (pointsPerBlockX - 1) + 1;
			int endY = (j + 1) * (pointsPerBlockY - 1) + 1;
			int ii, jj;
			long long pos, index;
			// int pos = ( i + j * blockX ) * pointsPerBlockX * pointsPerBlockY;
			// fread( &totalTerrain[pos], sizeof( float ), pointsPerBlockX * pointsPerBlockY,fp);
			fread(oneBlockTerrain, sizeof(float), pointsPerBlockX * pointsPerBlockY, fp);
			// for ( jj = 0; jj < ( pointsPerBlockY - 1 ); jj ++  )
			//{
			//	for ( ii = 0; ii < ( pointsPerBlockX - 1 ); ii ++ )
			//	{
			//		index = ii + jj * pointsPerBlockX;
			//		pos = ( startI + ii ) + ( startJ + jj ) * totalPointX;
			//		totalTerrain[pos] = oneBlockTerrain[index];
			//	}
			// }

			FOR_LOOP2D(ii, jj, 0, pointsPerBlockX - 1, 0, pointsPerBlockY - 1)
			index = Index2D(ii, jj, pointsPerBlockX, pointsPerBlockY);
			// index = ii + jj * pointsPerBlockX;// Index2D( ii, jj, pointsPerBlockX, pointsPerBlockY );
			// pos = ( startI + ii ) + ( startJ + jj ) * totalPointX;
			pos = Index2D(startI + ii, startJ + jj, totalPointX, totalPointY);
			totalTerrain[pos] = oneBlockTerrain[index];
			END_LOOP2D()

			// for ( int j = 0; j < pointsPerBlockY; j ++  )
			//{
			//	printf( "\n"  );
			//	for ( int i = 0; i < pointsPerBlockX; i ++ )
			//	{
			//		printf( "%e ", totalTerrain[i + j * pointsPerBlockX]  );
			//	}
			// }
			fclose(fp);
			i++;
		}
		j++;
	}

	free(oneBlockTerrain);
}

void resetTotalTerrain(PARAMS params, float *totalTerrain)
{
	int totalPointX = (pointsPerBlockX - 1) * params.blockX + 1;
	int totalPointY = (pointsPerBlockY - 1) * params.blockY + 1;
	int i = 0, j = 0;
	long long index;

	FOR_LOOP2D(i, j, 0, totalPointX, 0, totalPointY)
	index = Index2D(i, j, totalPointX, totalPointY);
	if (totalTerrain[index] < errorNegativeElevation + 1.0 || totalTerrain[index] > errorPositiveElevation - 1.0)
	{
		totalTerrain[index] = 0.0;
	}
	END_LOOP2D()
}

void callibrateTotalTerrain(PARAMS params, GRID grid, float *totalTerrain, float lon_0, float lat_0)
{

	int totalPointX = (pointsPerBlockX - 1) * params.blockX + 1;
	int totalPointY = (pointsPerBlockY - 1) * params.blockY + 1;
	int i = 0, j = 0;
	long long index, index1, index2;

	float *gradTotalTerrain = (float *)malloc(sizeof(float) * totalPointX * totalPointY);
	float *newTotalTerrain = (float *)malloc(sizeof(float) * totalPointX * totalPointY);
	float gradX = 0.0, gradY = 0.0;

	long long cnt = 0;

	float a, b, c, d;

	memcpy(newTotalTerrain, totalTerrain, sizeof(float) * totalPointX * totalPointY);

	PJ_CONTEXT *C;
	PJ *P;

	C = proj_context_create();

	char projStr[256]; //""
	sprintf(projStr, "+proj=aeqd +lon_0=%lf +lat_0=%lf +x_0=0.0 +y_0=0.0 +ellps=WGS84", lon_0, lat_0);

	// printf( projStr  );
	// printf( "\n"  );
	P = proj_create(C, projStr);
	if (NULL == P)
	{
		printf("Failed to create projection\n");
	}

	PJ_COORD pj_LonLat[4];
	double Lon[4], Lat[4];

	calculateCutLonLatLoc(grid, P, PJ_INV, pj_LonLat);

	proj_destroy(P);
	proj_context_destroy(C);

	float maxLon = -1000.0, maxLat = -1000.0;
	float minLon = 1000.0, minLat = 1000.0;
	for (i = 0; i < 4; i++)
	{
		Lon[i] = pj_LonLat[i].lp.lam * RADIAN2DEGREE;
		Lat[i] = pj_LonLat[i].lp.phi * RADIAN2DEGREE;
		if (maxLon <= Lon[i])
			maxLon = Lon[i];
		if (maxLat <= Lat[i])
			maxLat = Lat[i];
		if (minLon >= Lon[i])
			minLon = Lon[i];
		if (minLat >= Lat[i])
			minLat = Lat[i];
		// printf( "longitude: %lf, latitude: %lf\n", Lon[i], Lat[i] );
	}

	double lonStart = params.lonStart;
	double latStart = params.latStart;
	// printf( "lonStart = %f, latStart = %f\n", lonStart, latStart );

	double deltaLon = double(degreePerBlcokX * params.blockX) / double(totalPointX);
	double deltaLat = double(degreePerBlcokY * params.blockY) / double(totalPointY);
	// printf( "deltaLon = %f, deltaLat = %f\n", deltaLon, deltaLat );

	int expandPoint = 10;
	// printf( "maxLon = %f, minLon = %f\n", maxLon, minLon );
	// printf( "maxLat = %f, minLat = %f\n", maxLat, minLat );
	int ILonStart = int((minLon - lonStart) / deltaLon) - expandPoint;
	int JLatStart = int((minLat - latStart) / deltaLat) - expandPoint;

	int ILonEnd = int((maxLon - lonStart) / deltaLon) + expandPoint;
	int JLatEnd = int((maxLat - latStart) / deltaLat) + expandPoint;

	// printf( "ILonStart = %d, ILonEnd = %d, JLatStart = %d, JLatEnd = %d\n", ILonStart, ILonEnd, JLatStart, JLatEnd  );

	FOR_LOOP2D(i, j, ILonStart, ILonEnd, JLatStart, JLatEnd)
	index = Index2D(i, j, totalPointX, totalPointY);

	index1 = Index2D(i - 1, j, totalPointX, totalPointY);
	index2 = Index2D(i + 1, j, totalPointX, totalPointY);
	a = totalTerrain[index2];
	b = totalTerrain[index1];
	gradX = a - b;

	index1 = Index2D(i, j - 1, totalPointX, totalPointY);
	index2 = Index2D(i, j + 1, totalPointX, totalPointY);
	c = totalTerrain[index2];
	d = totalTerrain[index1];
	gradY = c - d;

	gradTotalTerrain[index] = sqrt(gradX * gradX + gradY * gradY);

	if (gradTotalTerrain[index] > 100.0)
	{
		newTotalTerrain[index] = (a + b + c + d) * 0.25;
	}

	// newTotalTerrain[index] = 0.0;

	END_LOOP2D()

	int thisRank;
	MPI_Comm_rank(MPI_COMM_WORLD, &thisRank);

	memcpy(totalTerrain, newTotalTerrain, sizeof(float) * totalPointX * totalPointY);

#ifdef Terrain_Smooth
	double sum = 0;
	long long ii, jj;
	int nGauss = 5;

	int n = 0;
	double *Dg = (double *)malloc(sizeof(double) * (2 * nGauss + 1));

	double ra = nGauss * 0.5;
	for (n = 0; n < (2 * nGauss + 1); n++)
	{
		Dg[n] = GAUSS_FUN(n - nGauss, ra, 0.0);
	}
	long long pos = 0;
	FOR_LOOP2D(i, j, ILonStart, ILonEnd, JLatStart, JLatEnd)
	index = Index2D(i, j, totalPointX, totalPointY);
	sum = 0.0;
	FOR_LOOP2D(ii, jj, i - nGauss, i + nGauss + 1, j - nGauss, j + nGauss + 1)
	// double amp = Dg[ii + nGauss-i] * Dg[jj + nGauss-j] / 0.998750;
	double amp = Dg[ii + nGauss - i] * Dg[jj + nGauss - j] / 0.996768;
	// printf( "D1 = %f, D2 = %f\n", Dg[ii + nGauss-i] * Dg[jj + nGauss-j]  );
	pos = Index2D(ii, jj, totalPointX, totalPointY);
	sum += amp * totalTerrain[pos];
	END_LOOP2D()
	// printf( "sum = %f\n", sum );
	newTotalTerrain[index] = sum; // totalTerrain[pos];
	END_LOOP2D()
	memcpy(totalTerrain, newTotalTerrain, sizeof(float) * totalPointX * totalPointY);
#endif

	free(gradTotalTerrain);
	free(newTotalTerrain);
}

void cart2LonLat(GRID grid, PJ *P, PJ_DIRECTION PD, float *coord, LONLAT LonLat)
{
	PJ_COORD *pj_coord;

	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	int _nz = grid._nz;

	long long num = _nx_ * _ny_;

	int i = 0, j = 0;
	long long index = 0, pos = 0;
	int k = _nz - 1;

	pj_coord = (PJ_COORD *)malloc(sizeof(PJ_COORD) * num);

	FOR_LOOP2D(i, j, 0, _nx_, 0, _ny_)
	index = INDEX(i, j, k) * CSIZE; // i + j * _nx_ + k *_nx_ * _ny_;
	pos = Index2D(i, j, _nx_, _ny_);
	pj_coord[pos].xy.x = coord[index + 0];
	pj_coord[pos].xy.y = coord[index + 1];

	END_LOOP2D()
	// cout << "========================" << endl;

	proj_trans_array(P, PD, num, pj_coord);

	FOR_LOOP2D(i, j, 0, _nx_, 0, _ny_)
	pos = Index2D(i, j, _nx_, _ny_);
	LonLat.lon[pos] = pj_coord[pos].lp.lam * RADIAN2DEGREE;
	LonLat.lat[pos] = pj_coord[pos].lp.phi * RADIAN2DEGREE;

	// cout << "Lon: " << LonLat.lon[pos] << "Lat:" << LonLat.lat[pos] << endl;
	END_LOOP2D()

	free(pj_coord);
}

void projTrans(double lon_0, double lat_0, GRID grid, float *coord, LONLAT LonLat, float lonMin, float lonMax, float latMin, float latMax)
{

	// #include "/public/software/proj-8.1.0/include/proj.h"
	PJ_CONTEXT *C;
	PJ *P;

	C = proj_context_create();

	char projStr[256]; //""
	sprintf(projStr, "+proj=aeqd +lon_0=%lf +lat_0=%lf +x_0=0.0 +y_0=0.0 +ellps=WGS84", lon_0, lat_0);

	// printf( projStr  );
	// printf( "\n"  );
	P = proj_create(C, projStr);
	if (NULL == P)
	{
		printf("Failed to create projection\n");
	}

	float lon[4] = {0};
	float lat[4] = {0};

	PJ_COORD pj_LonLat[4];

	calculateCutLonLat(grid, P, PJ_INV, pj_LonLat);

	// printf( "LonStart = %f, LonEnd = %f, LatStart = %f, LatEnd = %f\n", lonMin, lonMax, latMin, latMax );

	int i = 0;
	for (i = 0; i < 4; i++)
	{
		lon[i] = pj_LonLat[i].lp.lam * RADIAN2DEGREE;
		lat[i] = pj_LonLat[i].lp.phi * RADIAN2DEGREE;

		// printf( "Lon = %f, Lat = %f\n", lon[i], lat[i] );
		if (lon[i] > lonMax || lon[i] < lonMin || lat[i] > latMax || lat[i] < latMin)
		{
			printf("The longitude or latitude range of calculation are extends the Terrain or Medium lon lat Range! Please, check the lon or lat of Terrain data and Medium data!\n");
			MPI_Abort(MPI_COMM_WORLD, 1001);
		}
	}

	cart2LonLat(grid, P, PJ_INV, coord, LonLat);

	proj_destroy(P);
	proj_context_destroy(C);
}

/*
 *	C----------D
 *	|		   |
 *	|		   |
 *	|		   |
 *	|		   |
 *	A----------B
 */

double bilinear(double x, double y, double x1, double x2, double y1, double y2, double f11, double f12, double f21, double f22)
{
	// return ( f11 + f21 + f12 + f22 ) * 0.25;//(f11*(x2 - x)*(y2 - y) + f21*(x - x1)*(y2 - y) + f12*(x2 - x)*(y - y1) + f22*(x - x1)*(y - y1));///((x2 - x1)*(y2 - y1));
	return ((f11 * (x2 - x) * (y2 - y) + f21 * (x - x1) * (y2 - y) + f12 * (x2 - x) * (y - y1) + f22 * (x - x1) * (y - y1)) / ((x2 - x1) * (y2 - y1)));
}
double bilinearInterp(double x, double y, double x1, double y1, double x2, double y2, double A, double B, double C, double D)
{
	double AB = (x2 - x) / (x2 - x1) * A + (x - x1) / (x2 - x1) * B;
	double CD = (x2 - x) / (x2 - x1) * C + (x - x1) / (x2 - x1) * D;
	double R = (y2 - y) / (y2 - y1) * AB + (y - y1) / (y2 - y1) * CD;
	// R = ( A + B + C + D ) * 0.25;

	return R;
}

double interp2d(double x[2], double y[2], double z[4], double x_, double y_)
{
	int i, j;
	double Li = 0.0;
	double Lx[2], Ly[2];

	for (i = 0; i < 2; i++)
	{
		Lx[i] = 1;
		for (j = 0; j < 2; j++)
		{
			if (i == j)
				continue;
			Lx[i] = Lx[i] * (x_ - x[j]) / (x[i] - x[j]);
		}
	}

	for (i = 0; i < 2; i++)
	{
		Ly[i] = 1;
		for (j = 0; j < 2; j++)
		{
			if (i == j)
				continue;
			Ly[i] = Ly[i] * (y_ - y[j]) / (y[i] - y[j]);
		}
	}

	for (j = 0; j < 2; j++)
	{
		for (i = 0; i < 2; i++)
		{
			Li = Li + Lx[i] * Ly[j] * z[i * 2 + j];
		}
	}

	return Li;
}

void terrainInterp(PARAMS params, GRID grid, float *terrain, float *totalTerrain, LONLAT LonLat, float lon_0, float lat_0)
{
	int totalPointX = (pointsPerBlockX - 1) * params.blockX + 1;
	int totalPointY = (pointsPerBlockY - 1) * params.blockY + 1;

	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;

	int i = 0, j = 0;
	long long index = 0, idx = 0;

	double lonStart = params.lonStart;
	double latStart = params.latStart;

	double deltaLon = double(degreePerBlcokX * params.blockX) / double(totalPointX);
	double deltaLat = double(degreePerBlcokY * params.blockY) / double(totalPointY);

	double x_;
	double y_;
	double x1;
	double y1;
	double x2;
	double y2;

	double x[2] = {0.0};
	double y[2] = {0.0};
	double z[4] = {0.0};

	int I = 0, J = 0, pos = 0;

	int nGauss = 5;

	float DH = grid.DH;

	int frontNX = grid.frontNX;
	int frontNY = grid.frontNY;

	int originalX = grid.originalX;
	int originalY = grid.originalY;

	int e_nx = _nx_ + 2 * nGauss;
	int e_ny = _ny_ + 2 * nGauss;
	long long e_num = e_nx * e_ny;

	PJ_CONTEXT *C;
	PJ *P;

	C = proj_context_create();
	if (NULL == C)
	{
		printf("Failed to context C\n");
	}

	char projStr[256]; //""
	sprintf(projStr, "+proj=aeqd +lon_0=%lf +lat_0=%lf +x_0=0.0 +y_0=0.0 +ellps=WGS84", lon_0, lat_0);

	// printf( projStr  );
	// printf( "\n"  );
	P = proj_create(C, projStr);
	if (NULL == P)
	{
		printf("Failed to create projection\n");
	}

	PJ_COORD *pj_coord;
	pj_coord = (PJ_COORD *)malloc(sizeof(PJ_COORD) * e_num);
	if (NULL == pj_coord)
	{
		printf("can not malloc PJ_COORD\n");
	}

	float *exTerrain = (float *)malloc(sizeof(float) * e_nx * e_ny);

	FOR_LOOP2D(i, j, -nGauss, _nx_ + nGauss, -nGauss, _ny_ + nGauss)
	index = Index2D(i + nGauss, j + nGauss, e_nx, e_ny);
	I = frontNX + i;
	J = frontNY + j;
	pj_coord[index].xy.x = (I - HALO) * DH - originalX * DH;
	pj_coord[index].xy.y = (J - HALO) * DH - originalY * DH;
	END_LOOP2D()

	proj_trans_array(P, PJ_INV, e_num, pj_coord);

	FOR_LOOP2D(i, j, 0, e_nx, 0, e_ny)
	index = Index2D(i, j, e_nx, e_ny);

	x_ = pj_coord[index].lp.lam * RADIAN2DEGREE;
	y_ = pj_coord[index].lp.phi * RADIAN2DEGREE;

	I = int((x_ - lonStart) / deltaLon);
	J = int((y_ - latStart) / deltaLat);

	x1 = I * deltaLon + lonStart;
	y1 = J * deltaLat + latStart;

	// printf( "lon = %f, lat = %f\n", x_, y_ );

	x2 = x1 + deltaLon;
	y2 = y1 + deltaLat;

	x[0] = x1;
	x[1] = x2;
	y[0] = y1;
	y[1] = y2;

	pos = Index2D(I, J, totalPointX, totalPointY);
	// f11 = totalTerrain[pos]; //A
	z[0] = totalTerrain[pos]; // A

	pos = Index2D(I + 1, J, totalPointX, totalPointY);
	// f12 = totalTerrain[pos]; //B
	z[2] = totalTerrain[pos]; // B

	pos = Index2D(I + 1, J + 1, totalPointX, totalPointY);
	// f21 = totalTerrain[pos]; //B
	z[1] = totalTerrain[pos]; // B

	pos = Index2D(I + 1, J + 1, totalPointX, totalPointY);
	// f22 = totalTerrain[pos]; //D
	z[3] = totalTerrain[pos]; // D

	exTerrain[index] = interp2d(x, y, z, x_, y_);

	if ((i - nGauss) >= 0 && (i - nGauss) < _nx_ && (j - nGauss) >= 0 && (j - nGauss) < _ny_)
	{
		index = Index2D(i, j, e_nx, e_ny);
		idx = Index2D(i - nGauss, j - nGauss, _nx_, _ny_);
		terrain[idx] = exTerrain[index];
	}

	END_LOOP2D()

	free(pj_coord);
	proj_destroy(P);
	proj_context_destroy(C);

#ifdef Terrain_Smooth
	double sum = 0;
	long long ii, jj;

	int n = 0;
	double *Dg = (double *)malloc(sizeof(double) * (2 * nGauss + 1));

	double ra = nGauss * 0.5;
	for (n = 0; n < (2 * nGauss + 1); n++)
	{
		Dg[n] = GAUSS_FUN(n - nGauss, ra, 0.0);
	}

	FOR_LOOP2D(i, j, 0, e_nx, 0, e_ny)
	if ((i - nGauss) >= 0 && (i - nGauss) < _nx_ && (j - nGauss) >= 0 && (j - nGauss) < _ny_)
	{
		sum = 0;
		FOR_LOOP2D(ii, jj, i - nGauss, i + nGauss + 1, j - nGauss, j + nGauss + 1)
		//	double amp = Dg[ii + nGauss-i] * Dg[jj + nGauss-j] / 0.998750;
		double amp = Dg[ii + nGauss - i] * Dg[jj + nGauss - j] / 0.996768;
		pos = Index2D(ii, jj, e_nx, e_ny);
		sum += amp * exTerrain[pos];
		END_LOOP2D()

		idx = Index2D(i - nGauss, j - nGauss, _nx_, _ny_);
		terrain[idx] = sum;
	}

	END_LOOP2D()
#endif

	free(exTerrain);
}

void preprocessTerrain(PARAMS params, MPI_Comm comm_cart, MPI_COORD thisMPICoord, GRID grid, float *coord, float *terrain)
{
	double lon_0 = params.centerLongitude;
	double lat_0 = params.centerLatitude;

	int halo = grid.halo;

	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	int _nx = grid._nx;
	int _ny = grid._ny;
	int _nz = grid._nz;

	int i = 0, j = 0, k = 0;
	long long index = 0;

	// if ( 1 == params.SRTM90 && thisMPICoord.Z == grid.PZ - 1 )
	if (1 == params.SRTM90)
	{
		// printf( "======X = %d, Y = %d, Z = %d========\n", thisMPICoord.X, thisMPICoord.Y, thisMPICoord.Z  );
		float *totalTerrain;
		int totalPointX = (pointsPerBlockX - 1) * params.blockX + 1;
		int totalPointY = (pointsPerBlockY - 1) * params.blockY + 1;

		totalTerrain = (float *)malloc(totalPointX * totalPointY * sizeof(float));
		memset(totalTerrain, 0, totalPointX * totalPointY * sizeof(float));

		readSRTM90(params, totalTerrain);
#ifdef ERROR_ELEVATION
		resetTotalTerrain(params, totalTerrain);
#endif
		callibrateTotalTerrain(params, grid, totalTerrain, lon_0, lat_0);

		LONLAT LonLat;
		LonLat.lon = (double *)malloc(sizeof(double) * _nx_ * _ny_);
		LonLat.lat = (double *)malloc(sizeof(double) * _nx_ * _ny_);
		memset(LonLat.lon, 0, sizeof(double) * _nx_ * _ny_);
		memset(LonLat.lat, 0, sizeof(double) * _nx_ * _ny_);

		int LonStart = params.lonStart;
		int LatStart = params.latStart;

		int LonEnd = LonStart + params.blockX * degreePerBlcokX;
		int LatEnd = LatStart + params.blockY * degreePerBlcokY;

		// printf( "LonStart = %d, LonEnd = %d, LatStart = %d, LatEnd = %d\n", LonStart, LonEnd, LatStart, LatEnd );
		projTrans(lon_0, lat_0, grid, coord, LonLat, (float)LonStart, (float)LonEnd, (float)LatStart, (float)LatEnd);

		terrainInterp(params, grid, terrain, totalTerrain, LonLat, lon_0, lat_0);
		//		terrainInterp( params, grid, terrain, totalTerrain, LonLat );
		outputTotalTerrain(params, thisMPICoord, grid, totalTerrain, totalPointX, totalPointY);

		free(totalTerrain);

		if (thisMPICoord.Z == grid.PZ - 1)
		{
			if (0 == thisMPICoord.X && 0 == thisMPICoord.Y)
				printf("ouput \"projection data\" including longitude, latitude and terrain on the gound of the calculation area.\n");
			outputLonLat(params, thisMPICoord, grid, LonLat);
			outputTerrain(params, thisMPICoord, grid, terrain);
		}
		free(LonLat.lon);
		free(LonLat.lat);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	double Depth = params.Depth * 1e3;
	int NZ = grid.NZ;
	int pos = 0;

	int K = 0;
	int frontNZ = grid.frontNZ;
	double DZ = 0.0;
	int nx = grid.nx;
	int ny = grid.ny;
	int nz = grid.nz;

	FOR_LOOP3D(i, j, k, 0, _nx_, 0, _ny_, 0, _nz_)
	index = INDEX(i, j, k) * CSIZE;
	pos = Index2D(i, j, _nx_, _ny_);
	K = frontNZ + k - halo;
	DZ = double(terrain[pos] + abs(Depth)) / double(NZ - 1);
	coord[index + 2] = -abs(Depth) + DZ * K;
	// printf( "coord = %f\n", coord[index+2]  );
	END_LOOP3D()
}
