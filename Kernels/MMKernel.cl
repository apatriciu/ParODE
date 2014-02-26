#include <FTypeDef.h>

__kernel void MatrixMatrixMultiplicationKernel(__global fType* VectMSource, 
											   __global fType* VectMDest, 
												int SourceAIndex, 
												int SourceBIndex, 
												int DestIndex, 
												int MatrixSize,
												__local fType* TileSourceA,
												__local fType* TileSourceB){
	// special matrix-matrix multiplication to be used for PODES 
	// the matrices are stored in a vector like structure
	int ResColumn = get_global_id(0);
	int ResRow = get_global_id(1);

	int LocalRow = get_local_id(1);
	int LocalColumn = get_local_id(0);
	int TileSize = get_local_size(0);

	int nTiles = (MatrixSize - 1) / TileSize + 1;
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
		VectMDest[DestIndex * MatrixSize * MatrixSize + ResRow * MatrixSize + ResColumn] = ResValue;
}
