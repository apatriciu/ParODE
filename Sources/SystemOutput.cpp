/*
 * SystemOutput.cpp
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#include <SystemOutput.h>
#include <cassert>

const char* SystemOutput::pszProgramFileName = "SystemOutput.cl";
const char* SystemOutput::pszKernelName = "SystemOutputKernel";
const char* SystemOutput::pszProgramCode = "";

SystemOutput::SystemOutput(){
	_strKernelFileName = string(pszProgramFileName);
	_strKernelName = string(pszKernelName);
	_strProgramCode = string(pszProgramCode);
}

SystemOutput::~SystemOutput() {
}

void SystemOutput::Launch(
		  const vector<size_t>& WorkgroupSize,
		  const vector<size_t>& GlobalSize,
		  int queueIndex,
		  OpenCLMemBuffer* CBuffer,
		  OpenCLMemBuffer* DBuffer,
		  OpenCLMemBuffer* XBuffer,
		  OpenCLMemBuffer* UBuffer,
		  OpenCLMemBuffer* YBuffer,
		  int nStateStride,
		  int nStateOffset,
		  int nInputStride,
		  int nInputOffset,
		  int nSystemSize,
		  int nInputs,
		  int nOutputs,
		  int flagZeroC,
		  int flagZeroD){
	// check the inputs
	assert(WorkgroupSize.size() == GlobalSize.size());
	assert(WorkgroupSize.size() == 2);
	assert(WorkgroupSize[1] == GlobalSize[1]);
	// compute the output Y
	// set parameters for SystemOutput Kernel
	_pKernel->SetParameter(0, CBuffer);
	_pKernel->SetParameter(1, DBuffer);
	_pKernel->SetParameter(2, XBuffer);
	_pKernel->SetParameter(3, UBuffer);
	_pKernel->SetParameter(4, YBuffer);
	_pKernel->SetParameter(5, sizeof(int), (void*)(&nStateStride));
	_pKernel->SetParameter(6, sizeof(int), (void*)(&nStateOffset));
	_pKernel->SetParameter(7, sizeof(int), (void*)(&nInputStride));
	_pKernel->SetParameter(8, sizeof(int), (void*)(&nInputOffset));
	_pKernel->SetParameter(9, sizeof(int), (void*)(&nSystemSize));
	_pKernel->SetParameter(10, sizeof(int), (void*)(&nInputs));
	_pKernel->SetParameter(11, sizeof(int), (void*)(&nOutputs));
	_pKernel->SetParameter(12, sizeof(int), (void*)(&flagZeroC));
	_pKernel->SetParameter(13, sizeof(int), (void*)(&flagZeroD));
	_pKernel->SetSharedMemParameter(14, WorkgroupSize[0] *
			  	  	  	  	  WorkgroupSize[1] * sizeof(fType));
	_pKernel->Execute(_pKernel->GetContext()->GetQueue(queueIndex),
			GlobalSize,
			WorkgroupSize);
}
