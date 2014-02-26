/*
Kernel for computing the product between two matrices in the downsweep stage of the scan
The local size is TileSize x TileSize and the local memory vectors TileSourceA and TileSourceB have the TileSize x TileSize elements
There is one block created for each result matrix. 
global_size(0) = TileSize * number of resultant matrices.
global_size(1) = TileSize
The total number of elements in the vector of matrices should be a multiple of 2*nElementStride.
MatrixSize is the size of the matrix; there is no restriction on the size of the matrix.
*/

#include <FTypeDef.h>

__kernel void MatrixMatrixMultiplicationKernelScanUpsweep(	__global fType* VectMSource, // vector with source matrices
															__global fType* VectMDest, // vector with destination matrices
															int nElementStride, // matrix stride within the vector of matrices
															int MatrixSize,
															__local fType* TileSourceA,
															__local fType* TileSourceB){
	// special matrix-matrix multiplication to be used for PODES 
	// the matrices are stored in a vector like structure
	// there is one workgroup per multiplication
	// this should compute matrix with high index multiplied with matrix with low index
	// the result should go into the high index
	int SourceAIndex = ((get_global_id(0) / get_local_size(0)) + 1) * 2 * nElementStride - 1; // this is also the destination
	int SourceBIndex = SourceAIndex - nElementStride ;

	int LocalRow = get_local_id(1);
	int LocalColumn = get_local_id(0);
	int TileSize = get_local_size(0);

	int nTiles = (MatrixSize - 1) / TileSize + 1;
	for(int rowTileStart = 0; rowTileStart < MatrixSize; rowTileStart += TileSize){
		int ResRow = rowTileStart + LocalRow;
		for(int columnTileStart = 0; columnTileStart < MatrixSize; columnTileStart += TileSize){
			int ResColumn = columnTileStart + LocalColumn;
			fType ResValue = 0.0;
			for(int ii = 0; ii < nTiles; ii++){
				// load the two tiles
				TileSourceA[LocalRow * TileSize + LocalColumn] = (ii * TileSize + LocalColumn < MatrixSize && ResRow < MatrixSize) ?
														VectMSource[ SourceAIndex * MatrixSize * MatrixSize + 
																	ResRow * MatrixSize + 
																	ii * TileSize + LocalColumn] : 0.0;
				TileSourceB[LocalRow * TileSize + LocalColumn] = (ii * TileSize + LocalRow < MatrixSize && ResColumn < MatrixSize) ? 
														VectMSource[ SourceBIndex * MatrixSize * MatrixSize +
																	(ii * TileSize + LocalRow) * MatrixSize +
																	ResColumn] : 0.0;
				barrier( CLK_LOCAL_MEM_FENCE );
				for(int jj = 0; jj < get_local_size(0); jj++)
					ResValue += TileSourceA[LocalRow * TileSize + jj] * TileSourceB[jj * TileSize + LocalColumn];
				barrier( CLK_LOCAL_MEM_FENCE );
			}
			if(ResRow < MatrixSize && ResColumn < MatrixSize)
				VectMDest[SourceAIndex * MatrixSize * MatrixSize + ResRow * MatrixSize + ResColumn] = ResValue;
		}
	}
}
