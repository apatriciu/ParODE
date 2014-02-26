/*
 * LeftLeafCopyKernelScanUpsweep.h
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#ifndef LeftLeafCopyKernelScanUpsweep_H_
#define LeftLeafCopyKernelScanUpsweep_H_

#include <KernelWrapper.h>

class LeftLeafCopyKernelScanUpsweep: public KernelWrapper {
public:
	virtual ~LeftLeafCopyKernelScanUpsweep();
	void Launch(
			  const vector<size_t>& WorkgroupSize,
			  const vector<size_t>& GlobalSize,
			  int queueIndex,
			  OpenCLMemBuffer* SourceBuffer,
			  OpenCLMemBuffer* DestBuffer,
			  int d,
			  int nElems);
	LeftLeafCopyKernelScanUpsweep();
protected:
	static const char* pszProgramFileName;
	static const char* pszKernelName;
	static const char* pszProgramCode;
};

#endif /* LeftLeafCopyKernelScanUpsweep_H_ */
