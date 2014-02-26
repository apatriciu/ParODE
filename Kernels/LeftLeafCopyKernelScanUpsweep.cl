#include <FTypeDef.h>

__kernel void LeftLeafCopyKernelScanUpsweep(__global fType* VectVSource,
											__global fType* VectVDest, 
											int nElementStride, 
											int nElements){
	// copy the root of the tree to the  left son
	// nElement stride the  stride within the vectors of elements 
	// 1D block and grid
	// copy elements in chunks of get_local_size(0) until we are done
	int SourceIndex = ((get_global_id(0) / get_local_size(0)) + 1) * 2 * nElementStride - 1; 
	int DestinationIndex = SourceIndex - nElementStride ; // this is also the destination
	int GlobalDestinationVectBaseIndex = DestinationIndex * nElements;

	for(int StartLocalIndex = 0; StartLocalIndex < nElements; StartLocalIndex += get_local_size(0)){
		int LocalIndex = StartLocalIndex + get_local_id(0);
		if(LocalIndex < nElements)
			VectVDest[GlobalDestinationVectBaseIndex + LocalIndex] = VectVSource[GlobalDestinationVectBaseIndex + LocalIndex];
	}
}
