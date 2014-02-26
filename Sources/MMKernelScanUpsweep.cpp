/*
 * MMKernelScanUpsweep.cpp
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#include <MMKernelScanUpsweep.h>
#include <cassert>

const char* MMKernelScanUpsweep::pszProgramFileName = "MMKernelScanUpsweep.cl";
const char* MMKernelScanUpsweep::pszKernelName = "MatrixMatrixMultiplicationKernelScanUpsweep";
const char* MMKernelScanUpsweep::pszProgramCode = "";

MMKernelScanUpsweep::MMKernelScanUpsweep(){
	_strKernelFileName = string(pszProgramFileName);
	_strKernelName = string(pszKernelName);
	_strProgramCode = string(pszProgramCode);
}

MMKernelScanUpsweep::~MMKernelScanUpsweep() {
}

void MMKernelScanUpsweep::Launch(
		  const vector<size_t>& WorkgroupSize,
		  const vector<size_t>& GlobalSize,
		  int queueIndex,
		  OpenCLMemBuffer* SourceMatricesVector,
		  OpenCLMemBuffer* DestMatricesVector,
		  int d,
		  int nMatrixSize){
	assert(WorkgroupSize.size() == GlobalSize.size());
	assert(WorkgroupSize.size() == 2);
	assert(WorkgroupSize[0] == WorkgroupSize[1]);
	assert(GlobalSize[1] == WorkgroupSize[1]);
	_pKernel->SetParameter(0, SourceMatricesVector);
	_pKernel->SetParameter(1, DestMatricesVector);
	_pKernel->SetParameter(2, sizeof(int), (void*)(&d));
	_pKernel->SetParameter(3, sizeof(int), (void*)(&nMatrixSize));
	_pKernel->SetSharedMemParameter(4, WorkgroupSize[0] *
							 WorkgroupSize[1] * sizeof(fType));
	_pKernel->SetSharedMemParameter(5, WorkgroupSize[0] *
			 	 	 	 	 WorkgroupSize[1] * sizeof(fType));
	// call the MM kernel for upsweep
	_pKernel->Execute(_pKernel->GetContext()->GetQueue(queueIndex),
			GlobalSize,
			WorkgroupSize);
}
