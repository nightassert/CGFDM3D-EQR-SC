/*================================================================
*   ESS, Southern University of Science and Technology
*   
*   File Name:MPI_send_recv.cpp
*   Author: Wenqiang Wang, 11849528@mail.sustech.edu.cn
*   Created Time:2021-11-17
*   Discription:
*
================================================================*/
#include "header.h"

typedef void (*PACK_UNPACK_FUNC)( float * con, float * thisSend, int xStartHalo, int X0, int Y0, int Z0, int X, int Y, int Z );


__GLOBAL__
void packX( float * con, float * thisSend, 
	int xStartHalo, int X0, int Y0, int Z0, int X, int Y, int Z )
{
#ifdef GPU_CUDA
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;
#else
	int i0 = 0;
	int j0 = 0;
	int k0 = 0;
#endif

	long long index;
	long long pos;

	int i = 0;
	int j = 0;
	int k = 0;
	
	//printf( "X = %d, Y = %d, Z = %d; X0 = %d, Y0 = %d, Z0 = %d\n", X, Y, Z, X0, Y0, Z0 );
	CALCULATE3D( i0, j0, k0, 0, X0, 0, Y0, 0, Z0 )
		i = i0 + xStartHalo;
		j = j0;
		k = k0;
		index = Index3D( i, j, k, X, Y, Z );
		pos = Index3D( i0, j0, k0, X0, Y0, Z0 );
		thisSend[pos] = con[index];
    END_CALCULATE3D( )
}

__GLOBAL__
void unpackX( float * con, float * thisRecv,  
	int xStartHalo, int X0, int Y0, int Z0, int X, int Y, int Z )
{
#ifdef GPU_CUDA
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;
#else
	int i0 = 0;
	int j0 = 0;
	int k0 = 0;
#endif

	long long index;
	long long pos;

	int i = 0;
	int j = 0;
	int k = 0;
	
	CALCULATE3D( i0, j0, k0, 0, X0, 0, Y0, 0, Z0 )
		i = i0 + xStartHalo;
		j = j0;
		k = k0;
		index = Index3D( i, j, k, X, Y, Z );
		pos = Index3D( i0, j0, k0, X0, Y0, Z0 );
		con[index] = thisRecv[pos];
    END_CALCULATE3D( )
}

void PackUnpackX( float * con, float * thisSendRecv, 
	int xStartHalo, int VARSIZE, int _nx_, int _ny_, int _nz_, PACK_UNPACK_FUNC pack_unpack_func )
{

#ifdef XFAST
	int X0 = HALO * VARSIZE;
	int Y0 = _ny_;
	int Z0 = _nz_;
	int X  = _nx_ * VARSIZE;
	int Y  = Y0;
	int Z  = Z0;
	xStartHalo = xStartHalo * VARSIZE;
#endif
#ifdef ZFAST
	int X0 = HALO;
	int Y0 = _ny_;
	int Z0 = _nz_* VARSIZE;
	int X  = _nx_;
	int Y  = Y0;
	int Z  = Z0;
#endif

#ifdef GPU_CUDA
	dim3 threads( 4, 8, 16);
	dim3 blocks;
	blocks.x = ( X0 + threads.x - 1 ) / threads.x;
	blocks.y = ( Y0 + threads.y - 1 ) / threads.y;
	blocks.z = ( Z0 + threads.z - 1 ) / threads.z;
	pack_unpack_func<<< blocks, threads >>>
	( con, thisSendRecv, xStartHalo, X0, Y0, Z0, X, Y, Z );
	CHECK( cudaDeviceSynchronize( ) );
#else
	pack_unpack_func
	( con, thisSendRecv, xStartHalo, X0, Y0, Z0, X, Y, Z );
#endif
}


__GLOBAL__
void packY( float * con, float * thisSend, 
	int yStartHalo, int X0, int Y0, int Z0, int X, int Y, int Z )
{
	//printf("pack_MPI_y\n");
#ifdef GPU_CUDA
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;
#else
	int i0 = 0;
	int j0 = 0;
	int k0 = 0;
#endif

	long long index;
	long long pos;

	int i = 0;
	int j = 0;
	int k = 0;

	CALCULATE3D( i0, j0, k0, 0, X0, 0, Y0, 0, Z0 )
		i = i0;
		j = j0 + yStartHalo;
		k = k0;
		index = Index3D( i, j, k, X, Y, Z );
		pos = Index3D( i0, j0, k0, X0, Y0, Z0 );
		thisSend[pos] = con[index];
    END_CALCULATE3D( )

}

__GLOBAL__
void unpackY( float * con, float * thisRecv, 
	int yStartHalo, int X0, int Y0, int Z0, int X, int Y, int Z )
{
	//printf("unpack_MPI_y\n");
#ifdef GPU_CUDA
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;
#else
	int i0 = 0;
	int j0 = 0;
	int k0 = 0;
#endif

	long long index;
	long long pos;

	int i = 0;
	int j = 0;
	int k = 0;

	CALCULATE3D( i0, j0, k0, 0, X0, 0, Y0, 0, Z0 )
		i = i0;
		j = j0 + yStartHalo;
		k = k0;
		index = Index3D( i, j, k, X, Y, Z );
		pos = Index3D( i0, j0, k0, X0, Y0, Z0 );
		con[index] = thisRecv[pos];
	END_CALCULATE3D( )

}
void PackUnpackY( float * con, float * thisSendRecv, 
	int yStartHalo, int VARSIZE, int _nx_, int _ny_, int _nz_, PACK_UNPACK_FUNC pack_unpack_func )
{
#ifdef XFAST
	int X0 = _nx_ * VARSIZE;
	int Y0 = HALO;
	int Z0 = _nz_;
	int X  = X0;
	int Y  = _ny_;
	int Z  = Z0;
#endif
#ifdef ZFAST
	int X0 = _nx_;
	int Y0 = HALO;
	int Z0 = _nz_* VARSIZE;
	int X  = X0;
	int Y  = _ny_;
	int Z  = Z0;
#endif

#ifdef GPU_CUDA
	dim3 threads( 8, 4, 16);
	dim3 blocks;
	blocks.x = ( X0 + threads.x - 1 ) / threads.x;
	blocks.y = ( Y0 + threads.y - 1 ) / threads.y;
	blocks.z = ( Z0 + threads.z - 1 ) / threads.z;
	pack_unpack_func<<< blocks, threads >>>
	( con, thisSendRecv, yStartHalo, X0, Y0, Z0, X, Y, Z );
	CHECK( cudaDeviceSynchronize( ) );
#else
	pack_unpack_func
	( con, thisSendRecv, yStartHalo, X0, Y0, Z0, X, Y, Z );
#endif
}


__GLOBAL__
void packZ( float * con, float * thisSend, 
	int zStartHalo, int X0, int Y0, int Z0, int X, int Y, int Z )
{
	//printf("pack_MPI_y\n");
#ifdef GPU_CUDA
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;
#else
	int i0 = 0;
	int j0 = 0;
	int k0 = 0;
#endif

	long long index;
	long long pos;

	int i = 0;
	int j = 0;
	int k = 0;

	
	CALCULATE3D( i0, j0, k0, 0, X0, 0, Y0, 0, Z0 )
		i = i0;
		j = j0;
		k = k0 + zStartHalo;
		index = Index3D( i, j, k, X, Y, Z );
		pos = Index3D( i0, j0, k0, X0, Y0, Z0 );
		thisSend[pos] = con[index];
	END_CALCULATE3D( )

}

__GLOBAL__
void unpackZ( float * con, float * thisRecv, 
	int zStartHalo, int X0, int Y0, int Z0, int X, int Y, int Z )
{
	//printf("unpack_MPI_y\n");
#ifdef GPU_CUDA
	int i0 = threadIdx.x + blockIdx.x * blockDim.x;
	int j0 = threadIdx.y + blockIdx.y * blockDim.y;
	int k0 = threadIdx.z + blockIdx.z * blockDim.z;
#else
	int i0 = 0;
	int j0 = 0;
	int k0 = 0;
#endif

	long long index;
	long long pos;

	int i = 0;
	int j = 0;
	int k = 0;

	CALCULATE3D( i0, j0, k0, 0, X0, 0, Y0, 0, Z0 )
		i = i0;
		j = j0;
		k = k0 + zStartHalo;
		index = Index3D( i, j, k, X, Y, Z );
		pos = Index3D( i0, j0, k0, X0, Y0, Z0 );
		con[index] = thisRecv[pos];
	END_CALCULATE3D( )

}
void PackUnpackZ( float * con, float * thisSendRecv, 
	int zStartHalo, int VARSIZE, int _nx_, int _ny_, int _nz_, PACK_UNPACK_FUNC pack_unpack_func )
{
#ifdef XFAST
	int X0 = _nx_ * VARSIZE;
	int Y0 = _ny_;
	int Z0 = HALO;
	int X  = X0;
	int Y  = Y0;
	int Z  = _nz_;
#endif
#ifdef ZFAST
	int X0 = _nx_;
	int Y0 = _ny_;
	int Z0 = HALO * VARSIZE;
	int X  = X0;
	int Y  = Y0;
	int Z  = _nz_ * VARSIZE;
	zStartHalo = zStartHalo * VARSIZE;
#endif


	//printf( "X = %d, Y = %d, Z = %d; X0 = %d, Y0 = %d, Z0 = %d\n", X, Y, Z, X0, Y0, Z0 );
#ifdef GPU_CUDA
	dim3 threads( 16, 8, 4);
	dim3 blocks;
	blocks.x = ( X0 + threads.x - 1 ) / threads.x;
	blocks.y = ( Y0 + threads.y - 1 ) / threads.y;
	blocks.z = ( Z0 + threads.z - 1 ) / threads.z;
	pack_unpack_func<<< blocks, threads >>>
	( con, thisSendRecv, zStartHalo, X0, Y0, Z0, X, Y, Z );
	CHECK( cudaDeviceSynchronize( ) );
#else
	pack_unpack_func
	( con, thisSendRecv, zStartHalo, X0, Y0, Z0, X, Y, Z );
#endif
}


void alloc_send_recv( GRID grid, float ** send, float ** recv, int VARSIZE, char XYZ )
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	long long num = 0;
	
	switch( XYZ )
	{
		case 'X':
			num = _ny_ * _nz_* HALO * VARSIZE;
			break;       
		case 'Y':         
			num = _nx_ * _nz_* HALO * VARSIZE;
			break;       
		case 'Z':         
			num = _nx_ * _ny_* HALO * VARSIZE;
			break;
	}
		
	float * pSend = NULL;
	float * pRecv = NULL;
	long long size = sizeof( float ) * num;

	CHECK( Malloc( ( void ** )&pSend, size ) );
	CHECK( Memset(  pSend, 0, size ) ); 
	
	CHECK( Malloc( ( void ** )&pRecv, size ) );
	CHECK( Memset(  pRecv, 0, size ) ); 

	*send = pSend;
	*recv = pRecv;

}


void allocSendRecv( GRID grid, MPI_NEIGHBOR mpiNeighbor, SEND_RECV_DATA * sr, int VARSIZE )
{
	memset( sr, 0, sizeof( SEND_RECV_DATA ) );
	
	/*if ( mpiNeighbor.X1 > 0 )*/	alloc_send_recv( grid, &( sr->thisXSend1 ), &( sr->thisXRecv1 ), VARSIZE, 'X' );
	/*if ( mpiNeighbor.Y1 > 0 )*/	alloc_send_recv( grid, &( sr->thisYSend1 ), &( sr->thisYRecv1 ), VARSIZE, 'Y' );
	/*if ( mpiNeighbor.Z1 > 0 )*/	alloc_send_recv( grid, &( sr->thisZSend1 ), &( sr->thisZRecv1 ), VARSIZE, 'Z' ); 

	/*if ( mpiNeighbor.X2 > 0 )*/	alloc_send_recv( grid, &( sr->thisXSend2 ), &( sr->thisXRecv2 ), VARSIZE, 'X' );
	/*if ( mpiNeighbor.Y2 > 0 )*/	alloc_send_recv( grid, &( sr->thisYSend2 ), &( sr->thisYRecv2 ), VARSIZE, 'Y' );
	/*if ( mpiNeighbor.Z2 > 0 )*/	alloc_send_recv( grid, &( sr->thisZSend2 ), &( sr->thisZRecv2 ), VARSIZE, 'Z' );
	
}

void freeSendRecv( MPI_NEIGHBOR mpiNeighbor, SEND_RECV_DATA sr )
{
	/*if ( mpiNeighbor.X1 > 0 )*/	{ Free( sr.thisXSend1);	Free( sr.thisXRecv1 ); };
	/*if ( mpiNeighbor.Y1 > 0 )*/	{ Free( sr.thisYSend1);	Free( sr.thisYRecv1 ); };
	/*if ( mpiNeighbor.Z1 > 0 )*/	{ Free( sr.thisZSend1);	Free( sr.thisZRecv1 ); }; 
                                 
	/*if ( mpiNeighbor.X2 > 0 )*/	{ Free( sr.thisXSend2);	Free( sr.thisXRecv2 ); };
	/*if ( mpiNeighbor.Y2 > 0 )*/	{ Free( sr.thisYSend2);	Free( sr.thisYRecv2 ); };
	/*if ( mpiNeighbor.Z2 > 0 )*/	{ Free( sr.thisZSend2);	Free( sr.thisZRecv2 ); };
	
}


void mpiSendRecv( MPI_Comm comm_cart, MPI_NEIGHBOR mpiNeighbor, GRID grid, float * con, SEND_RECV_DATA sr, int VARSIZE )
{
	int _nx_ = grid._nx_;
	int _ny_ = grid._ny_;
	int _nz_ = grid._nz_;

	int nx = grid.nx;
	int ny = grid.ny;
	int nz = grid.nz;

	int _nx = grid._nx;
	int _ny = grid._ny;
	int _nz = grid._nz;

	long long num = 0;

	float * thisXSend1 = sr.thisXSend1;
	float * thisXRecv1 = sr.thisXRecv1;
	float * thisYSend1 = sr.thisYSend1;
	float * thisYRecv1 = sr.thisYRecv1;
	float * thisZSend1 = sr.thisZSend1;
	float * thisZRecv1 = sr.thisZRecv1;
                                 
	float * thisXSend2 = sr.thisXSend2;
	float * thisXRecv2 = sr.thisXRecv2;
	float * thisYSend2 = sr.thisYSend2;
	float * thisYRecv2 = sr.thisYRecv2;
	float * thisZSend2 = sr.thisZSend2;
	float * thisZRecv2 = sr.thisZRecv2;

	int xStartHalo, yStartHalo, zStartHalo;

	MPI_Status stat;

//x direction data exchange
	xStartHalo = nx;
	if ( mpiNeighbor.X2 >= 0 ) PackUnpackX( con, thisXSend2, xStartHalo, VARSIZE, _nx_, _ny_, _nz_,   packX );

	num = HALO * _ny_ * _nz_ * VARSIZE * sizeof( float );
	MPI_Sendrecv( sr.thisXSend2, num, MPI_CHAR, mpiNeighbor.X2, 101,
				  sr.thisXRecv1, num, MPI_CHAR, mpiNeighbor.X1, 101,
				  comm_cart, &stat );

	xStartHalo = 0;
	if ( mpiNeighbor.X1 >= 0 ) PackUnpackX( con, thisXRecv1, xStartHalo, VARSIZE, _nx_, _ny_, _nz_, unpackX );

	
	xStartHalo = HALO;
	if ( mpiNeighbor.X1 >= 0 ) PackUnpackX( con, thisXSend1, xStartHalo, VARSIZE, _nx_, _ny_, _nz_,   packX );

	num = HALO * _ny_ * _nz_ * VARSIZE * sizeof( float );
	MPI_Sendrecv( sr.thisXSend1, num, MPI_CHAR, mpiNeighbor.X1, 102,
				  sr.thisXRecv2, num, MPI_CHAR, mpiNeighbor.X2, 102,
				  comm_cart, &stat );

	xStartHalo = _nx;
	if ( mpiNeighbor.X2 >= 0 ) PackUnpackX( con, thisXRecv2, xStartHalo, VARSIZE, _nx_, _ny_, _nz_, unpackX );

//y direction data exchange
	yStartHalo = ny;
	if ( mpiNeighbor.Y2 >= 0 ) PackUnpackY( con, thisYSend2, yStartHalo, VARSIZE, _nx_, _ny_, _nz_,   packY );

	num = HALO * _nx_ * _nz_ * VARSIZE * sizeof( float );
	MPI_Sendrecv( sr.thisYSend2, num, MPI_CHAR, mpiNeighbor.Y2, 103,
				  sr.thisYRecv1, num, MPI_CHAR, mpiNeighbor.Y1, 103,
				  comm_cart, &stat );

	yStartHalo = 0;
	if ( mpiNeighbor.Y1 >= 0 ) PackUnpackY( con, thisYRecv1, yStartHalo, VARSIZE, _nx_, _ny_, _nz_, unpackY );

	
	yStartHalo = HALO;
	if ( mpiNeighbor.Y1 >= 0 ) PackUnpackY( con, thisYSend1, yStartHalo, VARSIZE, _nx_, _ny_, _nz_,   packY );

	num = HALO * _nx_ * _nz_ * VARSIZE * sizeof( float );
	MPI_Sendrecv( sr.thisYSend1, num, MPI_CHAR, mpiNeighbor.Y1, 104,
				  sr.thisYRecv2, num, MPI_CHAR, mpiNeighbor.Y2, 104,
				  comm_cart, &stat );

	yStartHalo = _ny;
	if ( mpiNeighbor.Y2 >= 0 ) PackUnpackY( con, thisYRecv2, yStartHalo, VARSIZE, _nx_, _ny_, _nz_, unpackY );

//z direction data exchange
	zStartHalo = nz;
	if ( mpiNeighbor.Z2 >= 0 ) PackUnpackZ( con, thisZSend2, zStartHalo, VARSIZE, _nx_, _ny_, _nz_,   packZ );

	num = HALO * _nx_ * _ny_ * VARSIZE * sizeof( float );
	MPI_Sendrecv( sr.thisZSend2, num, MPI_CHAR, mpiNeighbor.Z2, 105,
				  sr.thisZRecv1, num, MPI_CHAR, mpiNeighbor.Z1, 105,
				  comm_cart, &stat );

	zStartHalo = 0;
	if ( mpiNeighbor.Z1 >= 0 ) PackUnpackZ( con, thisZRecv1, zStartHalo, VARSIZE, _nx_, _ny_, _nz_, unpackZ );

	
	zStartHalo = HALO;
	if ( mpiNeighbor.Z1 >= 0 ) PackUnpackZ( con, thisZSend1, zStartHalo, VARSIZE, _nx_, _ny_, _nz_,   packZ );

	num = HALO * _nx_ * _ny_ * VARSIZE * sizeof( float );
	MPI_Sendrecv( sr.thisZSend1, num, MPI_CHAR, mpiNeighbor.Z1, 106,
				  sr.thisZRecv2, num, MPI_CHAR, mpiNeighbor.Z2, 106,
				  comm_cart, &stat );

	zStartHalo = _nz;
	if ( mpiNeighbor.Z2 >= 0 ) PackUnpackZ( con, thisZRecv2, zStartHalo, VARSIZE, _nx_, _ny_, _nz_, unpackZ );

/*
*/
}

