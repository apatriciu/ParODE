#include <FTypeDef.h>

__kernel void MatrixTimesVectorPlusVectorKernel(__global fType* VectMSource, 
												__global fType* VectVSource,
											    __global fType* VectVDest, 
												int SourceAIndex, 
												int SourceBIndex, 
												int DestIndex, 
												int MatrixSize,
												__local fType* localVect){
	// this is a matrix vector operation. 
	// we can implement using one block per line.
	// 1D block and grid
	// use a block size
	int resRow = get_global_id(0) / get_local_size(0);
	int tIndex = get_local_id(0);

	int LocalMemorySize = get_local_size(0);
	int nTiles = (MatrixSize - 1) / LocalMemorySize + 1;
	localVect[tIndex] = 0.0;
	for(int ii = 0; ii < nTiles; ii++)
		localVect[tIndex] += ( ii * LocalMemorySize + tIndex < MatrixSize ) ? VectMSource[ SourceBIndex * MatrixSize * MatrixSize + 
																						resRow * MatrixSize + 
																						ii * LocalMemorySize + tIndex] * 
																		   VectVSource[SourceAIndex * MatrixSize + 
																					ii * LocalMemorySize + tIndex] : 0.0;
	barrier( CLK_LOCAL_MEM_FENCE );
	// reduction
	for(unsigned int stride = (LocalMemorySize >> 1); stride > 0; stride >>= 1){
		if(tIndex < stride)
			localVect[tIndex] += localVect[tIndex + stride];
		barrier( CLK_LOCAL_MEM_FENCE );
	}

	if(tIndex == 0)
		VectVDest[DestIndex * MatrixSize + resRow] = localVect[0] + VectVSource[SourceBIndex * MatrixSize + resRow];
}
