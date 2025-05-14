/*================================================================
*   ESS, Southern University of Science and Technology
*
*   File Name: data_io.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time: 2021-11-06
*   Discription: Data I/O
*
*	Update: Tianhong Xu, 12231218@mail.sustech.edu.cn
*   Update Time: 2023-11-16
*   Update Content: Add SCFDM
*
*	Update: Tianhong Xu, 12231218@mail.sustech.edu.cn
*   Update Time: 2024-06-11
*   Update Content: Modify the equations to Wenqiang Zhang (2023)
*
*   Reference:
*      1. Wang, W., Zhang, Z., Zhang, W., Yu, H., Liu, Q., Zhang, W., & Chen, X. (2022). CGFDM3D‐EQR: A platform for rapid response to earthquake disasters in 3D complex media. Seismological Research Letters, 93(4), 2320-2334. https://doi.org/https://doi.org/10.1785/0220210172
*      2. Xu, T., & Zhang, Z. (2024). Numerical simulation of 3D seismic wave based on alternative flux finite-difference WENO scheme. Geophysical Journal International, 238(1), 496-512. https://doi.org/https://doi.org/10.1093/gji/ggae167
*      3. Zhang, W., Liu, Y., & Chen, X. (2023). A Mixed‐Flux‐Based Nodal Discontinuous Galerkin Method for 3D Dynamic Rupture Modeling. Journal of Geophysical Research: Solid Earth, e2022JB025817. 
*
=================================================================*/

#include "header.h"

void locateSlice(PARAMS params, GRID grid, SLICE *slice)
{
	int sliceX = params.sliceX;
	int sliceY = params.sliceY;
	int sliceZ = params.sliceZ;

	int frontNX = grid.frontNX;
	int frontNY = grid.frontNY;
	int frontNZ = grid.frontNZ;

	slice->X = sliceX - frontNX + HALO;
	slice->Y = sliceY - frontNY + HALO;
	slice->Z = sliceZ - frontNZ + HALO;

	// printf( "slice.X = %d, slice.Y = %d, slice.Z = %d\n", slice->X, slice->Y, slice->Z );
}

void locateFreeSurfSlice(GRID grid, SLICE *slice)
{
	int sliceX = -1;
	int sliceY = -1;
	int sliceZ = grid.NZ - 1;

	int frontNX = grid.frontNX;
	int frontNY = grid.frontNY;
	int frontNZ = grid.frontNZ;

	slice->X = sliceX - frontNX + HALO;
	slice->Y = sliceY - frontNY + HALO;
	slice->Z = sliceZ - frontNZ + HALO;

	// printf( "slice.X = %d, slice.Y = %d, slice.Z = %d\n", slice->X, slice->Y, slice->Z );
}

void allocDataout(GRID grid, char XYZ, float **dataout)
{
	int nx = grid.nx;
	int ny = grid.ny;
	int nz = grid.nz;

	long long num = 0;

	switch (XYZ)
	{
	case 'X':
		num = ny * nz;
		break;
	case 'Y':
		num = nx * nz;
		break;
	case 'Z':
		num = nx * ny;
		break;
	}

	float *pData = NULL;
	long long size = sizeof(float) * num;

	CHECK(Malloc((void **)&pData, size));
	CHECK(Memset(pData, 0, size));

	*dataout = pData;
}

void freeDataout(float *dataout)
{

	Free(dataout);
}

__GLOBAL__
void pack_iodata_x(void *datain, int VAR, int VARSIZE, float *dataout, int nx, int ny, int nz, int I, int CVS, int FP_TYPE)
{

	// printf( "datain = %p\n", datain  );
	int _nx_ = nx + 2 * HALO;
	int _ny_ = ny + 2 * HALO;
	int _nz_ = nz + 2 * HALO;

	double c = 1.0;

	float *datain1;
	FLOAT *datain2;

	if (1 == FP_TYPE)
		datain1 = (float *)datain;
	if (2 == FP_TYPE)
		datain2 = (FLOAT *)datain;

#ifdef GPU_CUDA
	int j0 = threadIdx.x + blockIdx.x * blockDim.x;
	int k0 = threadIdx.y + blockIdx.y * blockDim.y;
#else
	int j0 = 0;
	int k0 = 0;
#endif

	int i = I;
	int j = j0 + HALO;
	int k = k0 + HALO;

	long long index, pos;

	if (CVS == 1)
		c = 1.0 / Cv;
	if (CVS == 2)
		c = 1.0 / Cs;

	CALCULATE2D(j0, k0, 0, ny, 0, nz)
	j = j0 + HALO;
	k = k0 + HALO;
	index = INDEX(i, j, k) * VARSIZE + VAR;
	pos = Index2D(j0, k0, ny, nz);
	if (1 == FP_TYPE)
		dataout[pos] = datain1[index] * c;
	if (2 == FP_TYPE)
		dataout[pos] = (float)datain2[index] * c;
	// if ( 1 == FP_TYPE )
	// printf( "1:dataout = %f\n", dataout[pos] );
	// printf( "FP_TYPE = %d\n", FP_TYPE );
	END_CALCULATE2D()
}

__GLOBAL__
void pack_iodata_y(void *datain, int VAR, int VARSIZE, float *dataout, int nx, int ny, int nz, int J, int CVS, int FP_TYPE)
{
	int _nx_ = nx + 2 * HALO;
	int _ny_ = ny + 2 * HALO;
	int _nz_ = nz + 2 * HALO;

	double c = 1.0;

	float *datain1;
	FLOAT *datain2;

	if (1 == FP_TYPE)
		datain1 = (float *)datain;
	if (2 == FP_TYPE)
		datain2 = (FLOAT *)datain;

#ifdef GPU_CUDA
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int k0 = threadIdx.y + blockIdx.y * blockDim.y;
#else
	int i0 = 0;
	int k0 = 0;
#endif

	int i = i0 + HALO;
	int j = J;
	int k = k0 + HALO;

	long long index, pos;

	if (CVS == 1)
		c = 1.0 / Cv;
	if (CVS == 2)
		c = 1.0 / Cs;

	CALCULATE2D(i0, k0, 0, nx, 0, nz)
	i = i0 + HALO;
	k = k0 + HALO;
	index = INDEX(i, j, k) * VARSIZE + VAR;
	pos = Index2D(i0, k0, nx, nz);
	if (1 == FP_TYPE)
		dataout[pos] = datain1[index] * c;
	if (2 == FP_TYPE)
		dataout[pos] = (float)datain2[index] * c;
	// printf( "2:dataout = %f\n", dataout[pos] );
	END_CALCULATE2D()
}

__GLOBAL__
void pack_iodata_z(void *datain, int VAR, int VARSIZE, float *dataout, int nx, int ny, int nz, int K, int CVS, int FP_TYPE)
{
	int _nx_ = nx + 2 * HALO;
	int _ny_ = ny + 2 * HALO;
	int _nz_ = nz + 2 * HALO;

	double c = 1.0;

	float *datain1;
	FLOAT *datain2;

	if (1 == FP_TYPE)
		datain1 = (float *)datain;
	if (2 == FP_TYPE)
		datain2 = (FLOAT *)datain;

#ifdef GPU_CUDA
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
#else
	int i0 = 0;
	int j0 = 0;
#endif

	int i = i0 + HALO;
	int j = j0 + HALO;
	int k = K;

	long long index, pos;

	if (CVS == 1)
		c = 1.0 / Cv;
	if (CVS == 2)
		c = 1.0 / Cs;

	CALCULATE2D(i0, j0, 0, nx, 0, ny)
	i = i0 + HALO;
	j = j0 + HALO;
	index = INDEX(i, j, k) * VARSIZE + VAR;
	pos = Index2D(i0, j0, nx, ny);
	if (1 == FP_TYPE)
		dataout[pos] = datain1[index] * c;
	if (2 == FP_TYPE)
		dataout[pos] = (float)datain2[index] * c;
	// printf( "3:dataout = %f\n", dataout[pos] );
	END_CALCULATE2D()
}

void allocDataout_cpu(GRID grid, char XYZ, float **dataout)
{
	int nx = grid.nx;
	int ny = grid.ny;
	int nz = grid.nz;

	long long num = 0;

	switch (XYZ)
	{
	case 'X':
		num = ny * nz;
		break;
	case 'Y':
		num = nx * nz;
		break;
	case 'Z':
		num = nx * ny;
		break;
	}

	long long size = sizeof(float) * num;

	*dataout = (float *)malloc(size);
	Memset(*dataout, 0, size);
}

void freeDataout_cpu(float *dataout)
{
	Free(dataout);
}

void allocSliceData(GRID grid, SLICE slice, SLICE_DATA *sliceData, SLICE_DATA *sliceDataCpu)
{
	int _nx = grid._nx;
	int _ny = grid._ny;
	int _nz = grid._nz;

	if (slice.X >= HALO && slice.X < _nx)
		allocDataout(grid, 'X', &(sliceData->x));
	if (slice.Y >= HALO && slice.Y < _ny)
		allocDataout(grid, 'Y', &(sliceData->y));
	if (slice.Z >= HALO && slice.Z < _nz)
		allocDataout(grid, 'Z', &(sliceData->z));

#ifdef GPU_CUDA
	if (slice.X >= HALO && slice.X < _nx)
		allocDataout_cpu(grid, 'X', &(sliceDataCpu->x));
	if (slice.Y >= HALO && slice.Y < _ny)
		allocDataout_cpu(grid, 'Y', &(sliceDataCpu->y));
	if (slice.Z >= HALO && slice.Z < _nz)
		allocDataout_cpu(grid, 'Z', &(sliceDataCpu->z));
#else
	if (slice.X >= HALO && slice.X < _nx)
		sliceDataCpu->x = sliceData->x;
	if (slice.Y >= HALO && slice.Y < _ny)
		sliceDataCpu->y = sliceData->y;
	if (slice.Z >= HALO && slice.Z < _nz)
		sliceDataCpu->z = sliceData->z;
#endif
}

void freeSliceData(GRID grid, SLICE slice, SLICE_DATA sliceData, SLICE_DATA sliceDataCpu)
{
	int _nx = grid._nx;
	int _ny = grid._ny;
	int _nz = grid._nz;

	if (slice.X >= HALO && slice.X < _nx)
		freeDataout(sliceData.x);
	if (slice.Y >= HALO && slice.Y < _ny)
		freeDataout(sliceData.y);
	if (slice.Z >= HALO && slice.Z < _nz)
		freeDataout(sliceData.z);

#ifdef GPU_CUDA
	if (slice.X >= HALO && slice.X < _nx)
		freeDataout_cpu(sliceDataCpu.x);
	if (slice.Y >= HALO && slice.Y < _ny)
		freeDataout_cpu(sliceDataCpu.y);
	if (slice.Z >= HALO && slice.Z < _nz)
		freeDataout_cpu(sliceDataCpu.z);
#endif
}

void data2D_output_bin(GRID grid, SLICE slice,
					   MPI_COORD thisMPICoord,
					   void *datain, int VAR, int VARSIZE, SLICE_DATA sliceData, SLICE_DATA sliceDataCpu,
					   const char *name, int CVS, int FP_TYPE)
{

	int _nx = grid._nx;
	int _ny = grid._ny;
	int _nz = grid._nz;

	int nx = grid.nx;
	int ny = grid.ny;
	int nz = grid.nz;

	if (slice.X >= HALO && slice.X < _nx)
	{

#ifdef GPU_CUDA
		dim3 threads(32, 16, 1);
		dim3 blocks;
		blocks.x = (ny + threads.x - 1) / threads.x;
		blocks.y = (nz + threads.y - 1) / threads.y;
		blocks.z = 1;
		pack_iodata_x<<<blocks, threads>>>(datain, VAR, VARSIZE, sliceData.x, nx, ny, nz, slice.X, CVS, FP_TYPE);
		long long size = sizeof(float) * ny * nz;
		CHECK(cudaMemcpy(sliceDataCpu.x, sliceData.x, size, cudaMemcpyDeviceToHost));

#else
		pack_iodata_x(datain, VAR, VARSIZE, sliceData.x, nx, ny, nz, slice.X, CVS, FP_TYPE);

		sliceDataCpu.x = sliceData.x;

#endif

		FILE *fp;
		char fileName[256];
		sprintf(fileName, "%s_X_mpi_%d_%d_%d.bin", name, thisMPICoord.X, thisMPICoord.Y, thisMPICoord.Z);

		fp = fopen(fileName, "wb");

		fwrite(sliceDataCpu.x, sizeof(float), ny * nz, fp);

		fclose(fp);
	}

	if (slice.Y >= HALO && slice.Y < _ny)
	{

#ifdef GPU_CUDA
		dim3 threads(32, 16, 1);
		dim3 blocks;
		blocks.x = (nx + threads.x - 1) / threads.x;
		blocks.y = (nz + threads.y - 1) / threads.y;
		blocks.z = 1;
		pack_iodata_y<<<blocks, threads>>>(datain, VAR, VARSIZE, sliceData.y, nx, ny, nz, slice.Y, CVS, FP_TYPE);
		long long size = sizeof(float) * nx * nz;
		CHECK(cudaMemcpy(sliceDataCpu.y, sliceData.y, size, cudaMemcpyDeviceToHost));
#else
		pack_iodata_y(datain, VAR, VARSIZE, sliceData.y, nx, ny, nz, slice.Y, CVS, FP_TYPE);
		sliceDataCpu.y = sliceData.y;
#endif

		FILE *fp;
		char fileName[256];
		sprintf(fileName, "%s_Y_mpi_%d_%d_%d.bin", name, thisMPICoord.X, thisMPICoord.Y, thisMPICoord.Z);

		fp = fopen(fileName, "wb");

		fwrite(sliceDataCpu.y, sizeof(float), nx * nz, fp);

		fclose(fp);
	}

	if (slice.Z >= HALO && slice.Z < _nz)
	{

#ifdef GPU_CUDA
		dim3 threads(32, 16, 1);
		dim3 blocks;
		blocks.x = (nx + threads.x - 1) / threads.x;
		blocks.y = (ny + threads.y - 1) / threads.y;
		blocks.z = 1;
		pack_iodata_z<<<blocks, threads>>>(datain, VAR, VARSIZE, sliceData.z, nx, ny, nz, slice.Z, CVS, FP_TYPE);
		long long size = sizeof(float) * nx * ny;
		CHECK(cudaMemcpy(sliceDataCpu.z, sliceData.z, size, cudaMemcpyDeviceToHost));
#else
		pack_iodata_z(datain, VAR, VARSIZE, sliceData.z, nx, ny, nz, slice.Z, CVS, FP_TYPE);
		sliceDataCpu.z = sliceData.z;
#endif

		FILE *fp;
		char fileName[256];
		sprintf(fileName, "%s_Z_mpi_%d_%d_%d.bin", name, thisMPICoord.X, thisMPICoord.Y, thisMPICoord.Z);

		fp = fopen(fileName, "wb");

		fwrite(sliceDataCpu.z, sizeof(float), nx * ny, fp);

		fclose(fp);
	}
}

#ifdef SCFDM
__GLOBAL__
void wave_conserv2phy(FLOAT *wave_conserv, FLOAT *wave_phy, FLOAT *CJM, int _nx_, int _ny_, int _nz_)
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

	float u_conserv[9], u_phy[9];
	long long index;
	float mu, lambda, buoyancy;

	CALCULATE3D(i, j, k, 0, _nx_, 0, _ny_, 0, _nz_)
	index = INDEX(i, j, k);

	mu = CJM[index * CJMSIZE + 10];
	lambda = CJM[index * CJMSIZE + 11];
	buoyancy = CJM[index * CJMSIZE + 12];
	buoyancy *= Crho;

	for (int n = 0; n < 9; n++)
	{
		u_conserv[n] = wave_conserv[index * WSIZE + n];
	}

	// Calculate physical variables
	u_phy[0] = lambda * u_conserv[4] + lambda * u_conserv[5] + u_conserv[3] * (lambda + 2 * mu);
	u_phy[1] = lambda * u_conserv[3] + lambda * u_conserv[5] + u_conserv[4] * (lambda + 2 * mu);
	u_phy[2] = lambda * u_conserv[3] + lambda * u_conserv[4] + u_conserv[5] * (lambda + 2 * mu);
	u_phy[3] = mu * u_conserv[8];
	u_phy[4] = mu * u_conserv[7];
	u_phy[5] = mu * u_conserv[6];
	u_phy[6] = u_conserv[0] * buoyancy;
	u_phy[7] = u_conserv[1] * buoyancy;
	u_phy[8] = u_conserv[2] * buoyancy;

	for (int n = 0; n < 9; n++)
	{
		wave_phy[index * WSIZE + n] = u_phy[n];
	}

	END_CALCULATE3D()
}
#endif

void data2D_XYZ_out(MPI_COORD thisMPICoord, PARAMS params, GRID grid, FLOAT *wave, SLICE slice, SLICE_DATA sliceData, SLICE_DATA sliceDataCpu, char var, int it
#ifdef SCFDM
					,
					FLOAT *CJM
#endif
)
{

	// printf( "wave = %p\n", wave );

#ifdef SCFDM
	FLOAT *wave_phy; // Txx, Tyy, Tzz, Txy, Txz, Tyz, Vx, Vy, Vz
	CHECK(Malloc((void **)&wave_phy, sizeof(FLOAT) * WSIZE * grid._nx_ * grid._ny_ * grid._nz_));

	dim3 threads(32, 4, 4);
	dim3 blocks;
	blocks.x = (grid._nx_ + threads.x - 1) / threads.x;
	blocks.y = (grid._ny_ + threads.y - 1) / threads.y;
	blocks.z = (grid._nz_ + threads.z - 1) / threads.z;

	wave_conserv2phy<<<blocks, threads>>>(wave, wave_phy, CJM, grid._nx_, grid._ny_, grid._nz_);
#endif

	int FP_TYPE = 2;
	switch (var)
	{
	case 'V':
	{

		char VxFileName[128], VyFileName[128], VzFileName[128];

		sprintf(VxFileName, "%s/Vx_%d", params.OUT, it);
		sprintf(VyFileName, "%s/Vy_%d", params.OUT, it);
		sprintf(VzFileName, "%s/Vz_%d", params.OUT, it);
#ifdef SCFDM
		// ! For alternative flux finite difference by Tianhong Xu
		data2D_output_bin(grid, slice, thisMPICoord, wave_phy, 6 /*x*/, WSIZE, sliceData, sliceDataCpu, VxFileName, 1, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, wave_phy, 7 /*x*/, WSIZE, sliceData, sliceDataCpu, VyFileName, 1, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, wave_phy, 8 /*x*/, WSIZE, sliceData, sliceDataCpu, VzFileName, 1, FP_TYPE);
#else
		data2D_output_bin(grid, slice, thisMPICoord, wave, 0 /*x*/, WSIZE, sliceData, sliceDataCpu, VxFileName, 1, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, wave, 1 /*x*/, WSIZE, sliceData, sliceDataCpu, VyFileName, 1, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, wave, 2 /*x*/, WSIZE, sliceData, sliceDataCpu, VzFileName, 1, FP_TYPE);
#endif
	}
	break;
	case 'T':
	{

		char TxxFileName[128], TyyFileName[128], TzzFileName[128];
		char TxyFileName[128], TxzFileName[128], TyzFileName[128];

		sprintf(TxxFileName, "%s/Txx_%d", params.OUT, it);
		sprintf(TyyFileName, "%s/Tyy_%d", params.OUT, it);
		sprintf(TzzFileName, "%s/Tzz_%d", params.OUT, it);

		sprintf(TxyFileName, "%s/Txy_%d", params.OUT, it);
		sprintf(TxzFileName, "%s/Txz_%d", params.OUT, it);
		sprintf(TyzFileName, "%s/Tyz_%d", params.OUT, it);

#ifdef SCFDM
		data2D_output_bin(grid, slice, thisMPICoord, wave_phy, 0 /*x*/, WSIZE, sliceData, sliceDataCpu, TxxFileName, 2, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, wave_phy, 1 /*x*/, WSIZE, sliceData, sliceDataCpu, TyyFileName, 2, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, wave_phy, 2 /*x*/, WSIZE, sliceData, sliceDataCpu, TzzFileName, 2, FP_TYPE);

		data2D_output_bin(grid, slice, thisMPICoord, wave_phy, 3 /*x*/, WSIZE, sliceData, sliceDataCpu, TxyFileName, 2, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, wave_phy, 4 /*x*/, WSIZE, sliceData, sliceDataCpu, TxzFileName, 2, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, wave_phy, 5 /*x*/, WSIZE, sliceData, sliceDataCpu, TyzFileName, 2, FP_TYPE);
#else
		data2D_output_bin(grid, slice, thisMPICoord, wave, 3 /*x*/, WSIZE, sliceData, sliceDataCpu, TxxFileName, 2, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, wave, 4 /*x*/, WSIZE, sliceData, sliceDataCpu, TyyFileName, 2, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, wave, 5 /*x*/, WSIZE, sliceData, sliceDataCpu, TzzFileName, 2, FP_TYPE);

		data2D_output_bin(grid, slice, thisMPICoord, wave, 6 /*x*/, WSIZE, sliceData, sliceDataCpu, TxyFileName, 2, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, wave, 7 /*x*/, WSIZE, sliceData, sliceDataCpu, TxzFileName, 2, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, wave, 8 /*x*/, WSIZE, sliceData, sliceDataCpu, TyzFileName, 2, FP_TYPE);
#endif
	}
	break;

	case 'F':
	{
		char VxFileName[128], VyFileName[128], VzFileName[128];
		sprintf(VxFileName, "%s/FreeSurfVx_%d", params.OUT, it);
		sprintf(VyFileName, "%s/FreeSurfVy_%d", params.OUT, it);
		sprintf(VzFileName, "%s/FreeSurfVz_%d", params.OUT, it);

#ifdef SCFDM
		data2D_output_bin(grid, slice, thisMPICoord, wave_phy, 6 /*x*/, WSIZE, sliceData, sliceDataCpu, VxFileName, 1, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, wave_phy, 7 /*x*/, WSIZE, sliceData, sliceDataCpu, VyFileName, 1, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, wave_phy, 8 /*x*/, WSIZE, sliceData, sliceDataCpu, VzFileName, 1, FP_TYPE);
#else
		data2D_output_bin(grid, slice, thisMPICoord, wave, 0 /*x*/, WSIZE, sliceData, sliceDataCpu, VxFileName, 1, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, wave, 1 /*x*/, WSIZE, sliceData, sliceDataCpu, VyFileName, 1, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, wave, 2 /*x*/, WSIZE, sliceData, sliceDataCpu, VzFileName, 1, FP_TYPE);
#endif
	}
	}

#ifdef SCFDM
	Free(wave_phy);
#endif
}

void data2D_Model_out(MPI_COORD thisMPICoord, PARAMS params, GRID grid, float *coord, float *medium, SLICE slice, SLICE_DATA sliceData, SLICE_DATA sliceDataCpu)
{
	int FP_TYPE = 1;

	char XName[256], YName[256], ZName[256];

	{
		sprintf(XName, "%s/coordX", params.OUT);
		sprintf(YName, "%s/coordY", params.OUT);
		sprintf(ZName, "%s/coordZ", params.OUT);

		data2D_output_bin(grid, slice, thisMPICoord, coord, 0 /*x*/, CSIZE, sliceData, sliceDataCpu, XName, 0, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, coord, 1 /*y*/, CSIZE, sliceData, sliceDataCpu, YName, 0, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, coord, 2 /*z*/, CSIZE, sliceData, sliceDataCpu, ZName, 0, FP_TYPE);

		memset(XName, 0, 256);
		memset(YName, 0, 256);
		memset(ZName, 0, 256);
	}
	{
		sprintf(XName, "%s/Vs", params.OUT);
		sprintf(YName, "%s/Vp", params.OUT);
		sprintf(ZName, "%s/rho", params.OUT);

		data2D_output_bin(grid, slice, thisMPICoord, medium, 0 /*Vs */, MSIZE, sliceData, sliceDataCpu, XName, 0, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, medium, 1 /*Vp */, MSIZE, sliceData, sliceDataCpu, YName, 0, FP_TYPE);
		data2D_output_bin(grid, slice, thisMPICoord, medium, 2 /*Rho*/, MSIZE, sliceData, sliceDataCpu, ZName, 0, FP_TYPE);
	}
}

#ifdef SOLVE_DISPLACEMENT
void data2D_XYZ_out_Dis(MPI_COORD thisMPICoord, PARAMS params, GRID grid, SLICE slice, SLICE_DATA sliceData, SLICE_DATA sliceDataCpu, int it, FLOAT *Dis)
{

	int FP_TYPE = 2;

	char UxFileName[128], UyFileName[128], UzFileName[128];
	sprintf(UxFileName, "%s/FreeSurfUx_%d", params.OUT, it);
	sprintf(UyFileName, "%s/FreeSurfUy_%d", params.OUT, it);
	sprintf(UzFileName, "%s/FreeSurfUz_%d", params.OUT, it);

	data2D_output_bin(grid, slice, thisMPICoord, Dis, 0 /*x*/, 3, sliceData, sliceDataCpu, UxFileName, 1, FP_TYPE);
	data2D_output_bin(grid, slice, thisMPICoord, Dis, 1 /*x*/, 3, sliceData, sliceDataCpu, UyFileName, 1, FP_TYPE);
	data2D_output_bin(grid, slice, thisMPICoord, Dis, 2 /*x*/, 3, sliceData, sliceDataCpu, UzFileName, 1, FP_TYPE);
}
#endif