/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: struct.h
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2021-11-03
*   Discription: Free Surface Boundary Condition
*
*	Update: Tianhong Xu, 12231218@mail.sustech.edu.cn
*   Update Time: 2023-11-16
*   Update Content: Add SCFDM
*
*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*
=================================================================*/

#ifndef __STRUCT__
#define __STRUCT__
typedef struct PARAMS
{

	double TMAX;

	double DT;
	double DH;

	int NX;
	int NY;
	int NZ;

	int PX;
	int PY;
	int PZ;

	int centerX;
	int centerY;

	double centerLatitude;
	double centerLongitude;

	int sourceX;
	int sourceY;
	int sourceZ;

	int IT_SKIP;

	int sliceX;
	int sliceY;
	int sliceZ;
	int sliceFreeSurf;

	int nPML;

	int gauss_hill;
	int useTerrain;
	int useMedium;
	int useMultiSource;
	int useSingleSource;
	float rickerfc;

	int ShenModel;
	int Crust_1Medel;
	int LayeredModel;

	char LayeredFileName[256];

	int itSlice;
	int itStep;
	char waveOutput[64];
	char sliceName[64];
	int itStart;
	int itEnd;
	int igpu;
	char OUT[256];

	char TerrainDir[256];

	int SRTM90;

	int lonStart;
	int latStart;
	int blockX;
	int blockY;

	float Depth;

	float MLonStart;
	float MLatStart;
	float MLonEnd;
	float MLatEnd;

	float MLonStep;
	float MLatStep;

	float CrustLonStep;
	float CrustLatStep;

	float MVeticalStep;

	char MediumDir[256];
	char crustDir[256];

	char sourceFile[256];
	char sourceDir[256];

	int degree2radian;
} PARAMS;

typedef struct COORD
{
	float *x;
	float *y;
	float *z;
} COORDINATE, COORD;

typedef struct LONLAT
{
	double *lon;
	double *lat;
	double *depth;
} LONLAT;

typedef struct GRID
{

	int PX;
	int PY;
	int PZ;

	int _NX_;
	int _NY_;
	int _NZ_;

	int _NX;
	int _NY;
	int _NZ;

	int NX;
	int NY;
	int NZ;

	int _nx_;
	int _ny_;
	int _nz_;

	int _nx;
	int _ny;
	int _nz;

	int nx;
	int ny;
	int nz;

	int frontNX;
	int frontNY;
	int frontNZ;

	int _frontNX;
	int _frontNY;
	int _frontNZ;

	int originalX;
	int originalY;

	int _originalX;
	int _originalY;
	// int originalZ;

	int halo;

	int nPML;

	float DH;
	float rDH;

} GRID;

typedef struct MPI_NEIGHBOR
{
	int X1; // left
	int X2; // right

	int Y1; // front
	int Y2; // back

	int Z1; // down
	int Z2; // up

} MPI_NEIGHBOR;

typedef struct NCFILE
{
	int ncID;

	int ntDimID;
	int nzDimID;
	int nyDimID;
	int nxDimID;

	int VxVarID;
	int VyVarID;
	int VzVarID;

	int coordXVarID;
	int coordYVarID;
	int coordZVarID;

	int lonVarID;
	int latVarID;

} NCFILE;

typedef struct SOURCE_FILE_INPUT
{

	long long npts; // source point number
	int nt;			// number of source time sequences of every point
	float dt;		// time sample interval

	float *lon;
	float *lat;
	float *coordZ;

	float *area;
	float *strike;
	float *dip;

	float *rake;
	float *rate;

} SOURCE_FILE_INPUT;

typedef struct POINT_INDEX
{
	int X;
	int Y;
	int Z;

} POINT_INDEX;

typedef struct WAVE
{
	FLOAT *h_W;
	FLOAT *W;
	FLOAT *t_W;
	FLOAT *m_W;
	// ! For alternative flux finite difference by Tianhong Xu
	FLOAT *Fu, *Gu, *Hu;
	FLOAT *E;
	FLOAT *fu_ip12x, *fu_ip12y, *fu_ip12z;
} WAVE;

typedef struct STATION
{
	int *XYZ;
	float *wave;
} STATION;

typedef struct AUXILIARY
{
	float *h_Aux;
	float *Aux;
	float *t_Aux;
	float *m_Aux;
} AUXILIARY, AUX;

typedef struct AUX6SURF
{
	AUX Aux1x;
	AUX Aux1y;
	AUX Aux1z;

	AUX Aux2x;
	AUX Aux2y;
	AUX Aux2z;
} AUX6SURF, AUX6;

/*
typedef struct AUXILIARY
{
	float * Vx;
	float * Vy;
	float * Vz;
	float * Txx;
	float * Tyy;
	float * Tzz;
	float * Txy;
	float * Txz;
	float * Tyz;
}AUXILIARY, AUX;
*/

typedef struct PML_ALPHA
{
	float *x;
	float *y;
	float *z;
} PML_ALPHA;

typedef struct PML_BETA
{
	float *x;
	float *y;
	float *z;
} PML_BETA;

typedef struct PML_D
{
	float *x;
	float *y;
	float *z;
} PML_D;

typedef struct MPI_BORDER
{
	int isx1;
	int isx2;
	int isy1;
	int isy2;
	int isz1;
	int isz2;

} MPI_BORDER;

typedef struct CONTRAVARIANT
{
	float *xi_x;
	float *xi_y;
	float *xi_z;
	float *et_x;
	float *et_y;
	float *et_z;
	float *zt_x;
	float *zt_y;
	float *zt_z;
} CONTRAVARIANT;

typedef struct CONTRAVARIANT_FLOAT
{
	FLOAT *xi_x;
	FLOAT *xi_y;
	FLOAT *xi_z;
	FLOAT *et_x;
	FLOAT *et_y;
	FLOAT *et_z;
	FLOAT *zt_x;
	FLOAT *zt_y;
	FLOAT *zt_z;
} CONTRAVARIANT_FLOAT;

typedef struct Mat3x3
{
	float *M11;
	float *M12;
	float *M13;
	float *M21;
	float *M22;
	float *M23;
	float *M31;
	float *M32;
	float *M33;
} Mat3x3;

typedef struct Mat_rDZ
{
	float *_rDZ_DX;
	float *_rDZ_DY;
} Mat_rDZ;

typedef struct SLICE
{
	int X;
	int Y;
	int Z;
} SLICE;

typedef struct SLICE_DATA
{
	float *x;
	float *y;
	float *z;

} SLICE_DATA;

typedef struct SOURCE
{
	int X;
	int Y;
	int Z;
} SOURCE;

typedef struct SOURCE_INDEX
{
	int *X;
	int *Y;
	int *Z;
} SOURCE_INDEX;

typedef struct MEDIUM
{
	float *mu;
	float *lambda;
	float *buoyancy;
} MEDIUM;

typedef struct MEDIUM_FLOAT
{
	FLOAT *mu;
	FLOAT *lambda;
	FLOAT *buoyancy;
} MEDIUM_FLOAT;

typedef struct SEND_RECV_DATA
{
	float *thisXSend1;
	float *thisXRecv1;
	float *thisYSend1;
	float *thisYRecv1;
	float *thisZSend1;
	float *thisZRecv1;

	float *thisXSend2;
	float *thisXRecv2;
	float *thisYSend2;
	float *thisYRecv2;
	float *thisZSend2;
	float *thisZRecv2;

} SEND_RECV_DATA;

typedef struct SEND_RECV_DATA_FLOAT
{
	FLOAT *thisXSend1;
	FLOAT *thisXRecv1;
	FLOAT *thisYSend1;
	FLOAT *thisYRecv1;
	FLOAT *thisZSend1;
	FLOAT *thisZRecv1;

	FLOAT *thisXSend2;
	FLOAT *thisXRecv2;
	FLOAT *thisYSend2;
	FLOAT *thisYRecv2;
	FLOAT *thisZSend2;
	FLOAT *thisZRecv2;

} SEND_RECV_DATA_FLOAT;

typedef struct WSLICE
{
	float *sliceX;
	float *sliceY;
	float *sliceZ;
} WSLICE;

typedef struct MPI_COORDINATE
{
	int X;
	int Y;
	int Z;
} MPI_COORDINATE, MPI_COORD;

typedef struct DELTA_H_RANGE
{
	float *DT_min;
	float *DT_max;
} DELTA_H_RANGE;

typedef struct POINT_OR_VECTOR
{
	double x;
	double y;
	double z;
} POINT_OR_VECTOR;

typedef struct SOURCE_INFO
{
	int npts;
	int nt;
	float dt;
} SOURCE_INFO;

typedef struct MOMENT_RATE
{
	float *Mxx;
	float *Myy;
	float *Mzz;
	float *Mxy;
	float *Mxz;
	float *Myz;

} MOMENT_RATE;

typedef struct PGV
{
	float *pgvh;
	float *pgv;
} PGV;

#endif //__STRUCT__
