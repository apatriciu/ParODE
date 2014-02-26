/*
 * MMKernelScanDownsweep.h
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#ifndef MMKernelScanDownsweep_H_
#define MMKernelScanDownsweep_H_

#include <KernelWrapper.h>

class MMKernelScanDownsweep: public KernelWrapper {
public:
	virtual ~MMKernelScanDownsweep();
	void Launch(const vector<size_t>& WorkgroupSize,
			  const vector<size_t>& GlobalSize,
			  int queueIndex,
			  OpenCLMemBuffer* SourceMatricesVector,
			  OpenCLMemBuffer* DestMatricesVector,
			  int d,
			  int nMatrixSize);
	MMKernelScanDownsweep();
protected:
	static const char* pszProgramFileName;
	static const char* pszKernelName;
	static const char* pszProgramCode;
};

#endif /* MMKernelScanDownsweep_H_ */
