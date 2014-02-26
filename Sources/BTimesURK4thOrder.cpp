/*
 * BTimesURK4thOrder.cpp
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#include <BTimesURK4thOrder.h>
#include <cassert>

const char* BTimesURK4thOrder::pszProgramFileName = "BTimesURK4thOrder.cl";
const char* BTimesURK4thOrder::pszKernelName = "BTimesURK4thOrder";
const char* BTimesURK4thOrder::pszProgramCode = "";

BTimesURK4thOrder::BTimesURK4thOrder(){
	_strKernelFileName = string(pszProgramFileName);
	_strKernelName = string(pszKernelName);
	_strProgramCode = string(pszProgramCode);
}

BTimesURK4thOrder::~BTimesURK4thOrder() {
}

void BTimesURK4thOrder::Launch(
		  const vector<size_t>& WorkgroupSize,
		  const vector<size_t>& GlobalSize,
		  int queueIndex,
		  OpenCLMemBuffer* BBuffer,
		  OpenCLMemBuffer* UBuffer,
		  OpenCLMemBuffer* DestVectBuffer,
		  int nMatrixSize,
		  int nInputs){
	assert(WorkgroupSize.size() == GlobalSize.size());
	assert(WorkgroupSize.size() == 2);
	assert(GlobalSize[1] == WorkgroupSize[1]);

	// set parameters
	_pKernel->SetParameter(0, BBuffer);
	_pKernel->SetParameter(1, UBuffer);
	_pKernel->SetParameter(2, DestVectBuffer);
	_pKernel->SetParameter(3, sizeof(unsigned int), &nMatrixSize);
	_pKernel->SetParameter(4, sizeof(unsigned int), &nInputs);
	_pKernel->SetSharedMemParameter(5, WorkgroupSize[0] * WorkgroupSize[1] * sizeof(fType));
	// launch the grid on queue 1
	_pKernel->Execute(_pKernel->GetContext()->GetQueue(queueIndex),
						GlobalSize, WorkgroupSize);
}
