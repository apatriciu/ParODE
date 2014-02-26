/*
 * RootCopyKernelScanDownsweep.h
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#ifndef RootCopyKernelScanDownsweep_H_
#define RootCopyKernelScanDownsweep_H_

#include <KernelWrapper.h>

class RootCopyKernelScanDownsweep: public KernelWrapper {
public:
	virtual ~RootCopyKernelScanDownsweep();
	void Launch(
			  const vector<size_t>& WorkgroupSize,
			  const vector<size_t>& GlobalSize,
			  int queueIndex,
			  OpenCLMemBuffer* SourceBuffer,
			  OpenCLMemBuffer* DestBuffer,
			  int d,
			  int nElems);
	RootCopyKernelScanDownsweep();
protected:
	static const char* pszProgramFileName;
	static const char* pszKernelName;
	static const char* pszProgramCode;
};

#endif /* RootCopyKernelScanDownsweep_H_ */
