#include <FTypeDef.h>
// Implements the operation
// y = C x + D u for a batch
// the grid has two dimensions; there is one workgroup per item
// one row of threads computes the inner product between one line of C respecively D and x respectively u
// multiple rows of threads allow the computation of several result elements in the same time
// the size of the workgroup is the same as the MVV one
// the number of workgroups shall be the nuber of items per device
// nStrideX is the stride within vector X 
// nStrideX is the stride within vector U
// DFlag is 0 if D is 0

__kernel void SystemOutputKernel(__global fType* C, 
								 __global fType* D,
								 __global fType* vectX,
								 __global fType* vectU,
								 __global fType* vectY,
								 int nStrideX,
								 int nOffsetX,
								 int nStrideU,
								 int nOffsetU,
								 int nSystemSize,
								 int nInputSize,
								 int nOutputSize,
								 int bZeroC,
								 int bZeroD,
								 __local fType* localVect){
	unsigned long indexElement = get_global_id(0) / get_local_size(0);
	unsigned long indexStartX = indexElement * nStrideX * nSystemSize + 
								nOffsetX * nSystemSize;
	unsigned long indexStartU = indexElement * nStrideU * nInputSize + 
								nOffsetU * nInputSize;
	unsigned int tIndex = get_local_id(1) * get_local_size(0) + get_local_id(0);
	for(int rowStart = 0; rowStart < nOutputSize; rowStart += get_local_size(1)){
		localVect[tIndex] = 0.0;
		if(bZeroC != 1)
			for(int columnStart = 0; columnStart < nSystemSize; columnStart += get_local_size(0))
				localVect[tIndex] += 
						( rowStart + get_local_id(1) < nOutputSize && columnStart + get_local_id(0) < nSystemSize ) ? 
								C[(rowStart + get_local_id(1)) * nSystemSize + columnStart + get_local_id(0)] * 
														   vectX[indexStartX + columnStart + get_local_id(0)] : 
								0.0;
		if(bZeroD != 1) // add also the Du products; there will be no divergence as all threads will have the same DFlag value
			for(int columnStart = 0; columnStart < nInputSize; columnStart += get_local_size(0))
				localVect[tIndex] += 
					(rowStart + get_local_id(1) < nOutputSize && columnStart + get_local_id(0) < nInputSize) ?
							D[(rowStart + get_local_id(1)) * nInputSize + columnStart + get_local_id(0)] * 
													vectU[indexStartU + columnStart + get_local_id(0)] : 0.0;
		// reduction along the rows of localBuffer
		barrier( CLK_LOCAL_MEM_FENCE );
		for(unsigned int stride = (get_local_size(0) >> 1); stride > 0; stride >>= 1){
			if(get_local_id(0) < stride)
				localVect[tIndex] += localVect[tIndex + stride];
			barrier( CLK_LOCAL_MEM_FENCE );
		}
		// write the result in the global memory
		if( (get_local_id(0) == 0) && (rowStart + get_local_id(1) < nOutputSize))
			vectY[ indexElement * nOutputSize + rowStart + get_local_id(1)] = localVect[get_local_id(1) * get_local_size(0)];
	}
}
