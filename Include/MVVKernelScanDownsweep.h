/*
 * MVVKernelScanDownsweep.h
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#ifndef MVVKernelScanDownsweep_H_
#define MVVKernelScanDownsweep_H_

#include <KernelWrapper.h>

class MVVKernelScanDownsweep: public KernelWrapper {
public:
	virtual ~MVVKernelScanDownsweep();
	void Launch(const vector<size_t>& WorkgroupSize,
			  const vector<size_t>& GlobalSize,
			  int queueIndex,
			  OpenCLMemBuffer* SourceMatricesBuffer,
			  OpenCLMemBuffer* SourceVectorsBuffer,
			  OpenCLMemBuffer* DestVectorsBuffer,
			  int d,
			  int nMatrixSize);
	MVVKernelScanDownsweep();
protected:
	static const char* pszProgramFileName;
	static const char* pszKernelName;
	static const char* pszProgramCode;
};

#endif /* MVVKernelScanDownsweep_H_ */
