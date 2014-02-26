/*
 * SystemOutput.h
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#ifndef SystemOutput_H_
#define SystemOutput_H_

#include <KernelWrapper.h>

class SystemOutput: public KernelWrapper {
public:
	virtual ~SystemOutput();
	void Launch(
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
			  int flagZeroD);
	SystemOutput();
protected:
	static const char* pszProgramFileName;
	static const char* pszKernelName;
	static const char* pszProgramCode;
};

#endif /* SystemOutput_H_ */
