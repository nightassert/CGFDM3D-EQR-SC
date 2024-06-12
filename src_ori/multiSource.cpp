/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: multisource.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2021-09-14
*   Discription: Multisource
*
*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*
=================================================================*/

#include "header.h"

// bug: line 596: long long ii, indexII, indexI;

void readSourceInfo(SOURCE_FILE_INPUT *src_in, char *sourceFileName)
{
	FILE *sourceFile = fopen(sourceFileName, "rb");

	if (sourceFile == NULL)
	{
		printf("can't open source file!"
			   "Please check if there is a source named %s!\n",
			   sourceFileName);
		MPI_Abort(MPI_COMM_WORLD, 1001);
	}

#ifndef SOURCE_NPTS_LONG_LONG
	int nnnnnn;
	fread(&(nnnnnn), sizeof(int), 1, sourceFile);
	src_in->npts = nnnnnn;
#else
	fread(&(src_in->npts), sizeof(long long), 1, sourceFile);
#endif
	fread(&(src_in->nt), sizeof(int), 1, sourceFile);
	fread(&(src_in->dt), sizeof(float), 1, sourceFile);

	fclose(sourceFile);
}
void allocSourceLocation(SOURCE_FILE_INPUT *src_in)
{
	long long npts = src_in->npts;

	long long size = sizeof(float) * npts * CSIZE;

	float *pCoord = NULL;
	pCoord = (float *)malloc(size);
	memset(pCoord, 0, size);

	src_in->lon = pCoord + 0 * npts;
	src_in->lat = pCoord + 1 * npts;
	src_in->coordZ = pCoord + 2 * npts;
}

void freeSourceLocation(SOURCE_FILE_INPUT src_in)
{
	free(src_in.lon);
}

void readSourceLocation(SOURCE_FILE_INPUT src_in, char *sourceFileName)
{
	FILE *sourceFile = fopen(sourceFileName, "rb");

	if (NULL == sourceFile)
		cout << sourceFileName << ", is source file" << endl;
	fseek(sourceFile, sizeof(int) + sizeof(int) + sizeof(float), SEEK_CUR);

	long long npts = src_in.npts;
	int nt = src_in.nt;

	long long i = 0;
	for (i = 0; i < npts; i++)
	{
		fread(&(src_in.lon[i]), 1, sizeof(float), sourceFile);
		fread(&(src_in.lat[i]), 1, sizeof(float), sourceFile);
		fread(&(src_in.coordZ[i]), 1, sizeof(float), sourceFile);

		fseek(sourceFile, 3 * sizeof(float) + 2 * nt * sizeof(float), SEEK_CUR);
	}

	// int nnnn = 1000;

	// cout << "lon[" << nnnn  <<"] = " << src_in.lon[nnnn] << endl;
	// cout << "npts = " << npts << ", nt = " << nt << ", dt = " << src_in.dt << endl;
	fclose(sourceFile);
}

void LonLat2cart(PJ *P, PJ_DIRECTION PD, SOURCE_FILE_INPUT src_in)
{
	long long npts = src_in.npts;

	PJ_COORD *pj_coord;

	pj_coord = (PJ_COORD *)malloc(sizeof(PJ_COORD) * src_in.npts);

	long long i = 0;
	for (i = 0; i < npts; i++)
	{
		pj_coord[i].lp.lam = src_in.lon[i] * DEGREE2RADIAN;
		pj_coord[i].lp.phi = src_in.lat[i] * DEGREE2RADIAN;
	}

	proj_trans_array(P, PD, npts, pj_coord);

	for (i = 0; i < npts; i++)
	{
		src_in.lon[i] = pj_coord[i].xy.x;
		src_in.lat[i] = pj_coord[i].xy.y;
	}

	free(pj_coord);
}

void projTrans(PARAMS params, SOURCE_FILE_INPUT src_in)
{
	float lon_0 = params.centerLongitude;
	float lat_0 = params.centerLatitude;

	long long i = 0;
	float maxLon = -180.0, minLon = 180.0, maxLat = -90.0, minLat = 90.0;
	for (i = 0; i < src_in.npts; i++)
	{
		if (maxLon < src_in.lon[i])
		{
			maxLon = src_in.lon[i];
		}
		if (minLon > src_in.lon[i])
		{
			minLon = src_in.lon[i];
		}

		if (maxLat < src_in.lat[i])
		{
			maxLat = src_in.lat[i];
		}
		if (minLat > src_in.lat[i])
		{
			minLat = src_in.lat[i];
		}
	}

	// cout << "maxLon = " << maxLon << ", minLon = " << minLon << ", maxLat = " << maxLat << ", minLat = " << minLat << endl;

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

	LonLat2cart(P, PJ_FWD, src_in);
	proj_destroy(P);
	proj_context_destroy(C);
}

long long srcCoord2Index(GRID grid, float Depth, float *coord, float *cpu_terrain, SOURCE_FILE_INPUT src_in,
						 map<long long, long long> &point_index) // map<int, POINT_INDEX> pos_pointIndex )
{
	long long i = 0;
	long long npts = src_in.npts;
	int srcX, srcY, srcZ;

	float DH = grid.DH;
	double DZ = 0.0;
	double terrain;

	float *x = src_in.lon;
	float *y = src_in.lat;
	float *z = src_in.coordZ;

	int originalX = grid.originalX;
	int originalY = grid.originalY;

	int frontNX = grid.frontNX;
	int frontNY = grid.frontNY;
	int frontNZ = grid.frontNZ;

	int halo = grid.halo;
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	int _nx = grid._nx;
	int _ny = grid._ny;
	int _nz = grid._nz;

	int NZ = grid.NZ;

	long long pointNum = 0;

	// cout << "x = " << x[npts-1] << ", y = " << y[npts-1] << ", z = " << z[npts-1] << endl;
	float maxCoordZ = -Depth;
	float minCoordZ = Depth;
	for (int jj = 0; jj < npts; jj++)
	{
		if (maxCoordZ < z[jj])
			maxCoordZ = z[jj];

		if (minCoordZ > z[jj])
			minCoordZ = z[jj];
	}

	for (int jj = 0; jj < npts; jj++)
	{
		z[jj] = z[jj] - maxCoordZ;
	}
	// cout << "maxCoordZ = " << maxCoordZ << ", minCoordZ = " << minCoordZ  << endl;

	int extendNZ = 0;

	// cout << npts << endl;
	int rank;

	int k = 0;

	long long index1 = 0, index2 = 0, index = 0, pos = 0;
	double dz1 = 0.0, dz2 = 0.0;
	for (i = 0; i < npts; i++)
	{
		srcX = int(x[i] / DH + 0.5) + originalX - frontNX + halo;
		srcY = int(y[i] / DH + 0.5) + originalY - frontNY + halo;
#define SourceTerrain

#ifdef NO_SOURCE_SMOOTH
		if (srcX >= halo && srcX < _nx && srcY >= halo && srcY < _ny)
		{
#ifdef SourceTerrain
			pos = Index2D(srcX, srcY, _nx_, _ny_);
			terrain = cpu_terrain[pos]; //
			DZ = coord[INDEX(srcX, srcY, 4) * CSIZE + 2] - coord[INDEX(srcX, srcY, 3) * CSIZE + 2];
			terrain = DZ * double(NZ - 1) - abs(Depth);
			z[i] = terrain + z[i];
#endif // SourceTerrain
			for (k = halo; k < _nz; k++)
			{
				index1 = INDEX(srcX, srcY, k - 1);
				index2 = INDEX(srcX, srcY, k + 1);
				index = INDEX(srcX, srcY, k);

				dz1 = (coord[index * CSIZE + 2] - coord[index1 * CSIZE + 2]) * 0.5;
				dz2 = (coord[index2 * CSIZE + 2] - coord[index * CSIZE + 2]) * 0.5;

				if (coord[index * CSIZE + 2] - dz1 <= z[i] && z[i] < dz2 + coord[index * CSIZE + 2])
				{
					srcZ = k;
					point_index[i] = INDEX(srcX, srcY, srcZ);
					pointNum++;
				}
			}
		}
#else

		if (srcX >= 0 && srcX < _nx_ && srcY >= 0 && srcY < _ny_)
		{
#ifdef SourceTerrain
			pos = Index2D(srcX, srcY, _nx_, _ny_);
			terrain = cpu_terrain[pos]; // DZ * double( NZ - 1 ) - abs( Depth );
			z[i] = terrain + z[i];
#endif // SourceTerrain
			for (k = 0; k < _nz_; k++)
			{
				if (k - 1 == -1)
				{
					index = INDEX(srcX, srcY, 0);
					index2 = INDEX(srcX, srcY, 1);
					dz2 = (coord[index2 * CSIZE + 2] - coord[index * CSIZE + 2]) * 0.5;
					dz1 = dz2;
				}
				if (k + 1 == _nz_)
				{
					index1 = INDEX(srcX, srcY, _nz_ - 2);
					index = INDEX(srcX, srcY, _nz_ - 1);
					dz1 = (coord[index * CSIZE + 2] - coord[index1 * CSIZE + 2]) * 0.5;
					dz2 = dz1;
				}
				if (k - 1 != -1 && k + 1 != _nz_)
				{
					index1 = INDEX(srcX, srcY, k - 1);
					index = INDEX(srcX, srcY, k);
					index2 = INDEX(srcX, srcY, k + 1);
					dz1 = (coord[index * CSIZE + 2] - coord[index1 * CSIZE + 2]) * 0.5;
					dz2 = (coord[index2 * CSIZE + 2] - coord[index * CSIZE + 2]) * 0.5;
				}

				if (coord[index * CSIZE + 2] - dz1 <= z[i] && z[i] < dz2 + coord[index * CSIZE + 2])
				{
					srcZ = k;
					point_index[i] = INDEX(srcX, srcY, srcZ);
					pointNum++;
				}
			}
		}
#endif // NO_SOURCE_SMOOTH
	}

	return pointNum;
}

void allocSourceParams(SOURCE_FILE_INPUT *src_in, long long pointNum)
{
	if (0 == pointNum)
		return;
	int nt = src_in->nt;

	float *pTmp = NULL;
	long long num = pointNum * (3 + 2 * nt);
	long long size = num * sizeof(float);

	pTmp = (float *)malloc(size);
	memset(pTmp, 0, size);

	src_in->area = pTmp + 0 * pointNum;
	src_in->strike = pTmp + 1 * pointNum;
	src_in->dip = pTmp + 2 * pointNum;

	pTmp = pTmp + 3 * pointNum;

	src_in->rake = pTmp + 0 * pointNum;
	src_in->rate = pTmp + nt * pointNum;
}
void freeSourceParams(SOURCE_FILE_INPUT src_in, long long pointNum)
{
	if (0 == pointNum)
		return;
	free(src_in.area);
}

void readSourceParams(SOURCE_FILE_INPUT src_in, char *sourceFileName, map<long long, long long> &point_index)
{

	int size = point_index.size();
	if (0 == size)
	{
		return;
	}
	FILE *sourceFile = fopen(sourceFileName, "rb");

	if (NULL == sourceFile)
		cout << sourceFileName << ", is source file" << endl;

	int npts = src_in.npts;
	int nt = src_in.nt;
	long long i = 0;
	long long nptsIndex;

	int headSize = sizeof(int) + sizeof(int) + sizeof(int); // npts nt dt
	long long byteSize = 6 * sizeof(float) +				// lon lat coordZ area strike dip
						 2 * sizeof(float) * nt;

	for (map<long long, long long>::iterator it = point_index.begin(); it != point_index.end(); it++)
	{
		nptsIndex = it->first;
		// cout << nptsIndex << endl;
		fseek(sourceFile, headSize + byteSize * nptsIndex + 3 * sizeof(float), SEEK_SET);
		//
		fread(src_in.area + i, sizeof(float), 1, sourceFile);
		fread(src_in.strike + i, sizeof(float), 1, sourceFile);
		fread(src_in.dip + i, sizeof(float), 1, sourceFile);

		fread(src_in.rake + i * nt, sizeof(float), nt, sourceFile);
		fread(src_in.rate + i * nt, sizeof(float), nt, sourceFile);
		i++;
		// if ( i == 10000 )
		//		cout << "s = " << src_in.strike[i] << ", d = " << src_in.dip[i] << ", a = " << src_in.area[i] << ", r = " << src_in.rake[i] << endl;
	}

	// cout << "size = " << size << endl;
	// for ( i = 0; i < size; i ++ )
	//{
	//	cout << "dip = " << src_in.area[i] << endl;
	// }

	// int nnnn = 1000;
	// cout <<  size   << endl;

	// cout << "lon[" << nnnn  <<"] = " << src_in.lon[nnnn] << endl;
	// cout << "npts = " << npts << ", nt = " << nt << ", dt = " << src_in.dt << endl;
	fclose(sourceFile);
}

void allocMomentRate(float **momentRate, long long pointNum, int nt)
{
	if (0 == pointNum)
	{
		return;
	}
	long long num = pointNum * nt;
	long long size = sizeof(float) * num * MOMSIZE;

	CHECK(Malloc((void **)momentRate, size));
	CHECK(Memset(*momentRate, 0, size));
	if (*momentRate == NULL)
	{
		printf("can't allocate momentRate memory!\n");
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}
	CHECK(Memset(*momentRate, 0, size));
}

void allocMomentRate_cpu(float **cpu_momentRate, long long pointNum, int nt)
{
	if (0 == pointNum)
	{
		return;
	}
	long long num = pointNum * nt;
	long long size = sizeof(float) * num * MOMSIZE;
	*cpu_momentRate = (float *)malloc(size);
	memset(*cpu_momentRate, 0, size);

	if (*cpu_momentRate == NULL)
	{
		printf("can't allocate momentRate memory!\n");
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}
	memset(*cpu_momentRate, 0, size);
}

void freeMomentRate(float *momentRate, long long pointNum)
{
	if (0 == pointNum)
		return;
	Free(momentRate);
}

void freeMomentRate_cpu(float *cpu_momentRate, long long pointNum)
{
	if (0 == pointNum)
		return;
	free(cpu_momentRate);
}

void solveMomentRate(PARAMS params, SOURCE_FILE_INPUT src_in, float *momentRate, long long pointNum)
{
	if (0 == pointNum)
		return;

	float s, d, r, a, rt;
	float *strike = src_in.strike, *dip = src_in.dip, *area = src_in.area;
	float *rake = src_in.rake, *rate = src_in.rate;

	long long p = 0, index = 0;

	int it = 0, nt = src_in.nt;

	float M11 = 0.0f;
	float M22 = 0.0f;
	float M33 = 0.0f;
	float M12 = 0.0f;
	float M13 = 0.0f;
	float M23 = 0.0f;

	// cout << "pointNum = " << pointNum << endl;
	for (p = 0; p < pointNum; p++)
	{
		s = strike[p] * DEGREE2RADIAN;
		d = dip[p] * DEGREE2RADIAN;
		a = area[p];
		// if ( p == 1000 )
		//{
		//	cout << "s = " << s << ", d = " << d << ", a = " << a << endl;
		// }

		for (it = 0; it < nt; it++)
		{
			index = it + p * nt;

			if (params.degree2radian == 0)
				r = rake[index];
			else
				r = rake[index] * DEGREE2RADIAN;
			rt = rate[index];

			/*
			M11 = -(sin(d) * cos(r) * sin(s * 2.0) + sin(d * 2.0) * sin(r) * sin(s) * sin(s));
			M22 = sin(d) * cos(r) * sin(s * 2.0) - sin(d * 2.0) * sin(r) * cos(s) * cos(s);
			M33 = -(M11+M22);
			M12 = sin(d) * cos(r) * cos(s * 2.0) + 0.5 * sin(d * 2.0) * sin(r) * sin(s * 2.0);
			M13 = -(cos(d) * cos(r) * cos(s) + cos(d * 2.0) * sin(r) * sin(s));
			M23 = -(cos(d) * cos(r) * sin(s) - cos(d * 2.0) * sin(r) * cos(s));

			Mxx[index] =  M22 * a * rt;
			Myy[index] =  M11 * a * rt;
			Mzz[index] =  M33 * a * rt;
			Mxy[index] =  M12 * a * rt;
			Mxz[index] = -M23 * a * rt;
			Myz[index] = -M13 * a * rt;
			*/

			M11 = -(sin(d) * cos(r) * sin(2.0 * s) + sin(2.0 * d) * sin(r) * sin(s) * sin(s));
			M22 = sin(d) * cos(r) * sin(2.0 * s) - sin(2.0 * d) * sin(r) * cos(s) * cos(s);
			M33 = -(M11 + M22);
			M12 = sin(d) * cos(r) * cos(2.0 * s) + 0.5 * sin(2.0 * d) * sin(r) * sin(2.0 * s);
			M13 = -(cos(d) * cos(r) * cos(s) + cos(2.0 * d) * sin(r) * sin(s));
			M23 = -(cos(d) * cos(r) * sin(s) - cos(2.0 * d) * sin(r) * cos(s));

			momentRate[index * MOMSIZE + 0] = M22 * a * rt;
			momentRate[index * MOMSIZE + 1] = M11 * a * rt;
			momentRate[index * MOMSIZE + 2] = M33 * a * rt;
			momentRate[index * MOMSIZE + 3] = M12 * a * rt;
			momentRate[index * MOMSIZE + 4] = -M23 * a * rt;
			momentRate[index * MOMSIZE + 5] = -M13 * a * rt;

			/*
			if ( p == 10000 && it == 500 )
			{
				cout << "s = " << s << ", d = " << d << ", a = " << a << ", r = " << r << endl;
				printf( "M0 = %.10f\n",   M22 * a * rt );
				printf( "M1 = %.10f\n",   M11 * a * rt );
				printf( "M2 = %.10f\n",   M33 * a * rt );
				printf( "M3 = %.10f\n",   M12 * a * rt );
				printf( "M4 = %.10f\n", - M23 * a * rt );
				printf( "M5 = %.10f\n", - M13 * a * rt );
			}
			*/
		}
	}
}

void dealDuplicateIndex(float *momentRate,
						map<long long, long long> &point_index,
						map<long long, long long> &index_point,
						SOURCE_FILE_INPUT src_in)
{

	long long point_num = point_index.size();
	// cout << "point_num = " << point_num  <<  endl;
	if (0 == point_num)
	{
		return;
	}
	long long pnt = 0, idx = 0;

	// map < long long, long long > index_point;

	int i = 0;
	// cout << "1111111111111111111111111111111111111111111"<< endl;
	for (map<long long, long long>::iterator it = point_index.begin(); it != point_index.end(); it++)
	{
		idx = it->second;
		index_point[idx] = i; // It means that the hashmap index_point stores the max pnt since pnt is in a high order.
		i++;
	}

	// cout << "2222222222222222222222222222222222222222222"<< endl;

	int t = 0;
	int nt = src_in.nt;
	long long ii, indexII, indexI;

	i = 0;
	for (map<long long, long long>::iterator it = point_index.begin(); it != point_index.end(); it++)
	{
		idx = it->second;
		ii = index_point[idx];
		if (ii > i)
		{
			for (t = 0; t < nt; t++)
			{
				indexI = t + i * nt;
				indexII = t + ii * nt;

				momentRate[indexII * MOMSIZE + 0] += momentRate[indexI * MOMSIZE + 0];
				momentRate[indexII * MOMSIZE + 1] += momentRate[indexI * MOMSIZE + 1];
				momentRate[indexII * MOMSIZE + 2] += momentRate[indexI * MOMSIZE + 2];
				momentRate[indexII * MOMSIZE + 3] += momentRate[indexI * MOMSIZE + 3];
				momentRate[indexII * MOMSIZE + 4] += momentRate[indexI * MOMSIZE + 4];
				momentRate[indexII * MOMSIZE + 5] += momentRate[indexI * MOMSIZE + 5];
			}
		}
		i++;
	}

	// cout << "point_index size = " << point_index.size( )  << ", index_point size = " << index_point.size()  << endl;
}

void outputSourceData(PARAMS params, SOURCE_FILE_INPUT src_in,
					  map<long long, long long> &index_point,
					  float *momentRate, MPI_COORD thisMPICoord)
{

	int size = index_point.size();
	if (0 == size)
	{
		return;
	}
	char fileName[256];
	sprintf(fileName, "%s/source_mpi_%d_%d_%d.bin", params.OUT, thisMPICoord.X, thisMPICoord.Y, thisMPICoord.Z);

	FILE *file = fopen(fileName, "wb");

	if (NULL == file)
		printf("The file %s can not be opened\n", fileName);

	int npts = src_in.npts;
	int nt = src_in.nt;
	float dt = src_in.dt;

	fwrite(&size, sizeof(int), 1, file);
	fwrite(&nt, sizeof(int), 1, file);
	fwrite(&dt, sizeof(float), 1, file);

	long long index = 0;
	long long ii = 0;
	long long pos = 0;

	for (map<long long, long long>::iterator it = index_point.begin(); it != index_point.end(); it++)
	{
		index = it->first;
		ii = it->second;
		pos = ii * nt;
		fwrite(&index, sizeof(long long), 1, file);
		fwrite(&(momentRate[pos * MOMSIZE + 0]), sizeof(float), nt, file);
		fwrite(&(momentRate[pos * MOMSIZE + 1]), sizeof(float), nt, file);
		fwrite(&(momentRate[pos * MOMSIZE + 2]), sizeof(float), nt, file);
		fwrite(&(momentRate[pos * MOMSIZE + 3]), sizeof(float), nt, file);
		fwrite(&(momentRate[pos * MOMSIZE + 4]), sizeof(float), nt, file);
		fwrite(&(momentRate[pos * MOMSIZE + 5]), sizeof(float), nt, file);
	}

	fclose(file);
}

void verifyLocation(PARAMS params, SOURCE_FILE_INPUT src_in)
{
	FILE *file[3];

	char fileNameX[256], fileNameY[256], fileNameZ[256];

	sprintf(fileNameX, "%s/source_coord_X.bin", params.OUT);
	sprintf(fileNameY, "%s/source_coord_Y.bin", params.OUT);
	sprintf(fileNameZ, "%s/source_coord_Z.bin", params.OUT);

	file[0] = fopen(fileNameX, "wb");
	file[1] = fopen(fileNameY, "wb");
	file[2] = fopen(fileNameZ, "wb");

	long long npts = src_in.npts;

	float *x = src_in.lon;
	float *y = src_in.lat;
	float *z = src_in.coordZ;

	fwrite(x, sizeof(float), npts, file[0]);
	fwrite(y, sizeof(float), npts, file[1]);
	fwrite(z, sizeof(float), npts, file[2]);

	fclose(file[0]);
	fclose(file[1]);
	fclose(file[2]);
}

void allocSrcIndex(long long **srcIndex, long long npts)
{
	if (npts == 0)
		return;

	long long size = sizeof(long long) * npts;

	CHECK(Malloc((void **)srcIndex, size));
	CHECK(Memset(*srcIndex, 0, size));
}

void freeSrcIndex(long long *srcIndex, long long npts)
{
	if (npts == 0)
		return;
	Free(srcIndex);
}

void allocSrcIndex_cpu(long long **srcIndex, long long npts)
{
	if (npts == 0)
		return;

	long long size = sizeof(long long) * npts;

	*srcIndex = (long long *)malloc(size);
	memset(*srcIndex, 0, size);
}

void freeSrcIndex_cpu(long long *srcIndex, long long npts)
{
	if (npts == 0)
		return;
	free(srcIndex);
}

void changeStorageOrder(SOURCE_FILE_INPUT src_in,
						map<long long, long long> &index_point,
						long long *srcIndex,
						float *momentRateOld,
						float *momentRateNew)
{
	int nt = src_in.nt;
	int npts = index_point.size();
	if (0 == npts)
	{
		return;
	}

	int t = 0;
	long long ii = 0;
	long long pos0 = 0, pos1 = 0;

	long long J = 0;
	for (map<long long, long long>::iterator it = index_point.begin(); it != index_point.end(); it++)
	{
		srcIndex[J] = it->first;
		ii = it->second;
		for (t = 0; t < nt; t++)
		{
			pos0 = ii * nt + t;
			pos1 = J + t * npts;
			momentRateNew[pos1 * MOMSIZE + 0] = momentRateOld[pos0 * MOMSIZE + 0];
			momentRateNew[pos1 * MOMSIZE + 1] = momentRateOld[pos0 * MOMSIZE + 1];
			momentRateNew[pos1 * MOMSIZE + 2] = momentRateOld[pos0 * MOMSIZE + 2];
			momentRateNew[pos1 * MOMSIZE + 3] = momentRateOld[pos0 * MOMSIZE + 3];
			momentRateNew[pos1 * MOMSIZE + 4] = momentRateOld[pos0 * MOMSIZE + 4];
			momentRateNew[pos1 * MOMSIZE + 5] = momentRateOld[pos0 * MOMSIZE + 5];
			/*
			if ( J == 10000 && t == 500 )
			{
				printf( "momentRateNew = %f\n", momentRateNew[pos1*MOMSIZE+0] );
				printf( "momentRateNew = %f\n", momentRateNew[pos1*MOMSIZE+1] );
				printf( "momentRateNew = %f\n", momentRateNew[pos1*MOMSIZE+2] );
				printf( "momentRateNew = %f\n", momentRateNew[pos1*MOMSIZE+3] );
				printf( "momentRateNew = %f\n", momentRateNew[pos1*MOMSIZE+4] );
				printf( "momentRateNew = %f\n", momentRateNew[pos1*MOMSIZE+5] );
			}
			*/
		}
		J++;
	}
}

void allocMomentRateSlice(float **momentRateSlice, long long npts)
{
	if (0 == npts)
		return;

	long long num = npts;
	long long size = sizeof(float) * num * MOMSIZE;

	CHECK(Malloc((void **)momentRateSlice, size));
	CHECK(Memset(*momentRateSlice, 0, size));

	if (*momentRateSlice == NULL)
	{
		printf("can't allocate momentRateSlice  memory!\n");
		MPI_Abort(MPI_COMM_WORLD, 10001);
	}
}

void freeMomentRateSlice(float *momentRateSlice, long long npts)
{
	if (0 == npts)
		return;
	Free(momentRateSlice);
}

// This function includes allocating srcIndex and momentRate memory.
void init_MultiSource(PARAMS params, GRID grid, MPI_COORD thisMPICoord, float *coord, float *terrain, long long **srcIndex, float **momentRate, float **momentRateSlice, SOURCE_FILE_INPUT *ret_src_in)
{
	int thisRank;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Comm_rank(MPI_COMM_WORLD, &thisRank);
	if (thisRank == 0)
		printf("Processing fault(Source data precessing) data...\n");
	MPI_Barrier(MPI_COMM_WORLD);

	SOURCE_FILE_INPUT src_in;
	char sourceFileName[256];
	sprintf(sourceFileName, "%s/%s", params.sourceDir, params.sourceFile);

	// cout << sourceFileName << ", is source file" << endl;
	// cout << "==================================" << endl;

	readSourceInfo(&src_in, sourceFileName);
	allocSourceLocation(&src_in); // A1
	readSourceLocation(src_in, sourceFileName);

	projTrans(params, src_in);

	// allocSrcIndex( srcIndex, src_in.npts );

	float Depth = params.Depth * 1000;

	map<long long, long long> point_index;

	map<long long, long long> index_point;

	long long pointNum = srcCoord2Index(grid, Depth, coord, terrain, src_in, point_index);

	if (0 == thisMPICoord.X && 0 == thisMPICoord.Y && 0 == thisMPICoord.Z)
	{
		verifyLocation(params, src_in);
	}

	freeSourceLocation(src_in); // F1
	MPI_Barrier(MPI_COMM_WORLD);

	allocSourceParams(&src_in, pointNum); // A2
	readSourceParams(src_in, sourceFileName, point_index);
	/**/
	float *momentRateOld;
	allocMomentRate_cpu(&momentRateOld, pointNum, src_in.nt); // A3

	solveMomentRate(params, src_in, momentRateOld, pointNum);
	freeSourceParams(src_in, pointNum); // F2

	dealDuplicateIndex(momentRateOld, point_index, index_point, src_in);
	point_index.clear();
	outputSourceData(params, src_in, index_point, momentRateOld, thisMPICoord);
	MPI_Barrier(MPI_COMM_WORLD);

	long long npts = index_point.size();
	int nt = src_in.nt;
	float *momentRateNew;
	allocMomentRate_cpu(&momentRateNew, npts, nt); // A4

	long long *srcIndexNew;
	allocSrcIndex_cpu(&srcIndexNew, npts); // A5
	changeStorageOrder(src_in, index_point, srcIndexNew, momentRateOld, momentRateNew);
	freeMomentRate_cpu(momentRateOld, pointNum); // F3

	index_point.clear();
	allocMomentRateSlice(momentRateSlice, npts);
#ifdef GPU_CUDA
	allocMomentRate(momentRate, npts, nt);
	allocSrcIndex(srcIndex, npts);
	long long mr_size = npts * nt * sizeof(float) * MOMSIZE;
	CHECK(cudaMemcpy(*momentRate, momentRateNew, mr_size, cudaMemcpyHostToDevice));

	long long si_size = npts * sizeof(long long);
	CHECK(cudaMemcpy(*srcIndex, srcIndexNew, si_size, cudaMemcpyHostToDevice));

	freeMomentRate_cpu(momentRateNew, npts); // F4
	freeSrcIndex_cpu(srcIndexNew, npts);	 // F5

#else
	*momentRate = momentRateNew;
	*srcIndex = srcIndexNew;
#endif
	ret_src_in->nt = src_in.nt;
	ret_src_in->dt = src_in.dt;
	ret_src_in->npts = npts;
	// MPI_Barrier( MPI_COMM_WORLD );
	// cout << "npts = "<< npts << endl;
	// cout << "1111111111111111111111111111111111111111111"<< endl;
}

void finish_MultiSource(long long *srcIndex, float *momentRate, float *momentRateSlice, long long npts)
{
	freeSrcIndex(srcIndex, npts);
	freeMomentRate(momentRate, npts);
	freeMomentRateSlice(momentRateSlice, npts);
}
