/*
 * DistributeA.cpp
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#include <DistributeA.h>
#include <cassert>

const char* DistributeA::pszProgramFileName = "DistributeA.cl";
const char* DistributeA::pszKernelName = "DistributeA";
const char* DistributeA::pszProgramCode = "";

DistributeA::DistributeA(){
	_strKernelFileName = string(pszProgramFileName);
	_strKernelName = string(pszKernelName);
	_strProgramCode = string(pszProgramCode);
}

DistributeA::~DistributeA() {
}

void DistributeA::Launch(
		  const vector<size_t>& WorkgroupSize,
		  const vector<size_t>& GlobalSize,
		  int queueIndex,
		  OpenCLMemBuffer* SourceA,
		  OpenCLMemBuffer* DestMatricesVector,
		  int nElements){
	assert(WorkgroupSize.size() == GlobalSize.size());
	assert(WorkgroupSize.size() == 1);
	_pKernel->SetParameter(0, SourceA);
	_pKernel->SetParameter(1, DestMatricesVector);
	_pKernel->SetParameter(2, sizeof(int), (void *)&nElements);
	// launch the grid
	_pKernel->Execute(_pKernel->GetContext()->GetQueue(queueIndex),
			GlobalSize, WorkgroupSize);
}
