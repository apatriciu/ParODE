#include <FTypeDef.h>
// Distribute A over the vector of matrices
// the grid has one dimension; there is one workgroup per matrix
// the size of the workgroup shall be the minimum between the maximum workgroup dimension and the number of elements in the matrix
// the number of workgroups shall be the nuber of items per device
__kernel void DistributeA(__global fType* ABar,
						 __global fType* vectMatrixDest,
						 int VectorSize){
	unsigned long indexStartElement = (get_global_id(0) / get_local_size(0)) * VectorSize;
	for(unsigned TileStart = 0; TileStart < VectorSize; TileStart += get_local_size(0))
		if(TileStart + get_local_id(0) < VectorSize)
			vectMatrixDest[indexStartElement + TileStart + get_local_id(0)] = ABar[TileStart + get_local_id(0)];
}
