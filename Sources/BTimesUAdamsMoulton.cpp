/*
 * BTimesUAdamsMoulton.cpp
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#include <BTimesUAdamsMoulton.h>
#include <cassert>

const char* BTimesUAdamsMoulton::pszProgramFileName = "BTimesUAdamsMoulton.cl";
const char* BTimesUAdamsMoulton::pszKernelName = "BTimesUAdamsMoulton";
const char* BTimesUAdamsMoulton::pszProgramCode = "";

BTimesUAdamsMoulton::BTimesUAdamsMoulton(){
	_strKernelFileName = string(pszProgramFileName);
	_strKernelName = string(pszKernelName);
	_strProgramCode = string(pszProgramCode);
}

BTimesUAdamsMoulton::~BTimesUAdamsMoulton() {
}

void BTimesUAdamsMoulton::Launch(
		  const vector<size_t>& WorkgroupSize,
		  const vector<size_t>& GlobalSize,
		  int queueIndex,
		  OpenCLMemBuffer* BBuffer,
		  OpenCLMemBuffer* UBuffer,
		  OpenCLMemBuffer* DestVectBuffer,
		  int nBRows,
		  int nBCols,
		  int nInputs){
	assert(WorkgroupSize.size() == GlobalSize.size());
	assert(WorkgroupSize.size() == 2);
	assert(GlobalSize[1] == WorkgroupSize[1]);

	// set parameters
	_pKernel->SetParameter(0, BBuffer);
	_pKernel->SetParameter(1, UBuffer);
	_pKernel->SetParameter(2, DestVectBuffer);
	_pKernel->SetParameter(3, sizeof(unsigned int), &nBRows);
	_pKernel->SetParameter(4, sizeof(unsigned int), &nBCols);
	_pKernel->SetParameter(5, sizeof(unsigned int), &nInputs);
	_pKernel->SetSharedMemParameter(6, WorkgroupSize[0] * WorkgroupSize[1] * sizeof(fType));
	// launch the grid on queue 1
	_pKernel->Execute(_pKernel->GetContext()->GetQueue(queueIndex),
						GlobalSize, WorkgroupSize);
}
