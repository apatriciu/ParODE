#include <DebugUtilities.h>
#include <iostream>

// Debug Functions
void PrintMatrices(fType* vectMatrices, int MatrixSize, unsigned int nElements){
	for(int ii = 0; ii < nElements; ii++){
		std::cout << "Matrix " << ii << std::endl;
		for(int row = 0; row < MatrixSize; row++){
			for(int col = 0; col < MatrixSize; col++)
				std::cout << vectMatrices[ii * MatrixSize * MatrixSize + row * MatrixSize + col] << " ";
			std::cout << std::endl;
		}
	}
}

void PrintVectors(fType* vectVectors, int MatrixSize, unsigned int nElements){
	for(int ii = 0; ii < nElements; ii++){
		std::cout << "Vector " << ii << " : ";
		for(int row = 0; row < MatrixSize; row++)
			std::cout << vectVectors[ii * MatrixSize + row] << " ";
		std::cout << std::endl;
	}
}

void PrintMatrix(fType* pMatrix, int nRows, int nCols){
	for(int row = 0; row < nRows; row++){
		for(int col = 0; col < nCols; col++)
			std::cout << pMatrix[row * nCols + col] << " ";
		std::cout << std::endl;
	}
}
