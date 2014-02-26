/*
 * MMKernelScanUpsweep.h
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#ifndef MMKERNELSCANUPSWEEP_H_
#define MMKERNELSCANUPSWEEP_H_

#include <KernelWrapper.h>

class MMKernelScanUpsweep: public KernelWrapper {
public:
	virtual ~MMKernelScanUpsweep();
	void Launch(const vector<size_t>& WorkgroupSize,
			  const vector<size_t>& GlobalSize,
			  int queueIndex,
			  OpenCLMemBuffer* SourceMatricesVector,
			  OpenCLMemBuffer* DestMatricesVector,
			  int d,
			  int nMatrixSize);
	MMKernelScanUpsweep();
protected:
	static const char* pszProgramFileName;
	static const char* pszKernelName;
	static const char* pszProgramCode;
};

#endif /* MMKERNELSCANUPSWEEP_H_ */
