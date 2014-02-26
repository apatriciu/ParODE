#ifndef __DEBUG_UTILITIES_PARODE__
#define __DEBUG_UTILITIES_PARODE__

#include <ParODETypeDefs.h>

// Debug Functions
void PrintMatrices(fType* vectMatrices, int MatrixSize,
					 	unsigned int nElements);
void PrintVectors(fType* vectVectors, int MatrixSize,
						unsigned int nElements);
void PrintMatrix(fType* pMatrix, int nRows, int nCols);
#endif
