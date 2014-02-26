/*
 * MMKernelScanUpsweep.h
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#ifndef MVVKERNELSCANUPSWEEP_H_
#define MVVKERNELSCANUPSWEEP_H_

#include <KernelWrapper.h>

class MVVKernelScanUpsweep: public KernelWrapper {
public:
	virtual ~MVVKernelScanUpsweep();
	void Launch(const vector<size_t>& WorkgroupSize,
			  const vector<size_t>& GlobalSize,
			  int queueIndex,
			  OpenCLMemBuffer* SourceMatricesBuffer,
			  OpenCLMemBuffer* SourceVectorsBuffer,
			  OpenCLMemBuffer* DestVectorsBuffer,
			  int d,
			  int nMatrixSize);
	MVVKernelScanUpsweep();
protected:
	static const char* pszProgramFileName;
	static const char* pszKernelName;
	static const char* pszProgramCode;
};

#endif /* MVVKERNELSCANUPSWEEP_H_ */
