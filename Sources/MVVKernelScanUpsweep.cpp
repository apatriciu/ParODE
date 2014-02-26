/*
 * MMKernelScanUpsweep.cpp
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#include <MVVKernelScanUpsweep.h>
#include <cassert>

const char* MVVKernelScanUpsweep::pszProgramFileName = "MVVKernelScanUpsweep.cl";
const char* MVVKernelScanUpsweep::pszKernelName = "MatrixTimesVectorPlusVectorKernelScanUpsweep";
const char* MVVKernelScanUpsweep::pszProgramCode = "";

MVVKernelScanUpsweep::MVVKernelScanUpsweep(){
	_strKernelFileName = string(pszProgramFileName);
	_strKernelName = string(pszKernelName);
	_strProgramCode = string(pszProgramCode);
}

MVVKernelScanUpsweep::~MVVKernelScanUpsweep() {
}

void MVVKernelScanUpsweep::Launch(
		  const vector<size_t>& WorkgroupSize,
		  const vector<size_t>& GlobalSize,
		  int queueIndex,
		  OpenCLMemBuffer* SourceMatricesBuffer,
		  OpenCLMemBuffer* SourceVectorsBuffer,
		  OpenCLMemBuffer* DestVectorsBuffer,
		  int d,
		  int nMatrixSize){
	assert(WorkgroupSize.size() == GlobalSize.size());
	assert(WorkgroupSize.size() == 2);
	assert(GlobalSize[1] == WorkgroupSize[1]);
	// set parameters for MVV upsweep
	_pKernel->SetParameter(0, SourceMatricesBuffer);
	_pKernel->SetParameter(1, SourceVectorsBuffer);
	_pKernel->SetParameter(2, DestVectorsBuffer);
	_pKernel->SetParameter(3, sizeof(int), (void*)(&d));
	_pKernel->SetParameter(4, sizeof(int), (void*)(&nMatrixSize));
	_pKernel->SetSharedMemParameter(5, WorkgroupSize[0] *
							 WorkgroupSize[1] * sizeof(fType));
	// call the MVV kernel for upsweep; queue 1
	_pKernel->Execute(_pKernel->GetContext()->GetQueue(queueIndex), GlobalSize, WorkgroupSize);
}
