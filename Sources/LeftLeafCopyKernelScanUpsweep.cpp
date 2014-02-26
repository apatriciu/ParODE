/*
 * LeftLeafCopyKernelScanUpsweep.cpp
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#include <LeftLeafCopyKernelScanUpsweep.h>
#include <cassert>

const char* LeftLeafCopyKernelScanUpsweep::pszProgramFileName = "LeftLeafCopyKernelScanUpsweep.cl";
const char* LeftLeafCopyKernelScanUpsweep::pszKernelName = "LeftLeafCopyKernelScanUpsweep";
const char* LeftLeafCopyKernelScanUpsweep::pszProgramCode = "";

LeftLeafCopyKernelScanUpsweep::LeftLeafCopyKernelScanUpsweep(){
	_strKernelFileName = string(pszProgramFileName);
	_strKernelName = string(pszKernelName);
	_strProgramCode = string(pszProgramCode);
}

LeftLeafCopyKernelScanUpsweep::~LeftLeafCopyKernelScanUpsweep() {
}

void LeftLeafCopyKernelScanUpsweep::Launch(
		  const vector<size_t>& WorkgroupSize,
		  const vector<size_t>& GlobalSize,
		  int queueIndex,
		  OpenCLMemBuffer* SourceBuffer,
		  OpenCLMemBuffer* DestBuffer,
		  int d,
		  int nElems){
	assert(WorkgroupSize.size() == GlobalSize.size());
	assert(WorkgroupSize.size() == 1);
	// set parameters for copy kernel for the matrix component
	_pKernel->SetParameter(0, SourceBuffer);
	_pKernel->SetParameter(1, DestBuffer);
	_pKernel->SetParameter(2, sizeof(int), (void*)(&d));
	_pKernel->SetParameter(3, sizeof(int), (void*)(&nElems));
	_pKernel->Execute(_pKernel->GetContext()->GetQueue(queueIndex), GlobalSize, WorkgroupSize);
}
