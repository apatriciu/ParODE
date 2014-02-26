#include <FTypeDef.h>
// Multiplies BBar with uBar and stores the result in vectDest
// the grid has two dimensions; there is one workgroup per item
// one row of threads computes the inner product between one line of BBar and u
// multiple rows allow the computation of several result elements in the same time
// the size of the workgroup is the same as the MVV one
// the number of workgroups shall be the nuber of items per device
// we have to keep track of u inputs
__kernel void BTimesUAdamsMoulton(
								__global fType* BBar,
								__global fType* vectU,
								__global fType* vectDest,
								unsigned int nBRows,
								unsigned int nBCols,
								unsigned int nInputs,
								__local fType* localBuffer ){
	unsigned long indexElement = get_global_id(0) / get_local_size(0);
	unsigned long indexStartU = indexElement * nInputs;
	unsigned int tIndex = get_local_id(1) * get_local_size(0) + get_local_id(0);
	for(int rowStart = 0; rowStart < nBRows; rowStart += get_local_size(1)){
		localBuffer[tIndex] = 0.0;
		for(int columnStart = 0; columnStart < nBCols; columnStart += get_local_size(0))
			localBuffer[tIndex] += 
					( rowStart + get_local_id(1) < nBRows && columnStart + get_local_id(0) < nBCols )? 
							BBar[(rowStart + get_local_id(1)) * nBCols + 
									columnStart + get_local_id(0)] * 
									vectU[indexStartU + columnStart + get_local_id(0)] : 
							0.0;
		// reduction along the rows of localBuffer
		barrier( CLK_LOCAL_MEM_FENCE );
		for(unsigned int stride = (get_local_size(0) >> 1); stride > 0; stride >>= 1){
			if(get_local_id(0) < stride)
				localBuffer[tIndex] += localBuffer[tIndex + stride];
			barrier( CLK_LOCAL_MEM_FENCE );
		}
		// write the result in the global memory
		if( (get_local_id(0) == 0) && (rowStart + get_local_id(1) < nBRows))
			vectDest[ (4 * indexElement + 3) * nBRows + rowStart + get_local_id(1)] = localBuffer[get_local_id(1) * get_local_size(0)];
	}
}
