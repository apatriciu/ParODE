/*
Kernel for computing the matrix x vector + vector in the downsweep stage of the scan
The local size is ColumnsTileSize x RowsTileSize and the local memory vectors localVect should have at least ColumnsTileSize x RowsTileSize elements
There is one block created for each result vector. 
global_size(0) = ColumnsTileSize * number of resultant vectors.
global_size(1) = RowsTileSize
The total number of elements in the vector of matrices should be a multiple of 2*nElementStride.
MatrixSize is the size of the matrix; there is no restriction on the size of the matrix.
ColumnsTileSize should be a power of two
*/

#include <FTypeDef.h>

__kernel void MatrixTimesVectorPlusVectorKernelScanDownsweep(	__global fType* VectMSource, 
															__global fType* VectVSource,
															__global fType* VectVDest, 
															int nElementStride, 
															int MatrixSize,
															__local fType* localVect){
	// this is a matrix vector operation. 
	// 2D block and grid
	// the first dimension of the workgroup goes along line
	// the second dimension of the workgroup allows processing more than one line at a time
	// one workgroup for each matrix x vector + vector operation
	int SourceAIndex = ((get_global_id(0) / get_local_size(0)) + 1) * 2 * nElementStride - 1; // this is also the destination
	int SourceBIndex = SourceAIndex - nElementStride ;

	int tIndex = get_local_id(0) + get_local_id(1) * get_local_size(0);
	for(int rowStart = 0; rowStart < MatrixSize; rowStart += get_local_size(1)){
		localVect[tIndex] = 0.0;
		for(int columnStart = 0; columnStart < MatrixSize; columnStart += get_local_size(0))
			localVect[tIndex] += ( columnStart + get_local_id(0) < MatrixSize && 
								   rowStart +  get_local_id(1) < MatrixSize) ? 
								   VectMSource[ SourceBIndex * MatrixSize * MatrixSize + 
												(rowStart + get_local_id(1)) * MatrixSize + 
												columnStart + get_local_id(0)] * 
								   VectVSource[SourceAIndex * MatrixSize + columnStart + get_local_id(0)] : 0.0;
		barrier( CLK_LOCAL_MEM_FENCE );
		// reduction
		for(unsigned int stride = (get_local_size(0) >> 1); stride > 0; stride >>= 1){
			if(get_local_id(0) < stride)
				localVect[tIndex] += localVect[tIndex + stride];
			barrier( CLK_LOCAL_MEM_FENCE );
		}

		if( (get_local_id(0) == 0) && (rowStart + get_local_id(1) < MatrixSize))
			VectVDest[SourceAIndex * MatrixSize + rowStart + get_local_id(1)] = localVect[get_local_id(1) * get_local_size(0)] + 
																			 VectVSource[SourceBIndex * MatrixSize + rowStart + get_local_id(1)];
	}
}
