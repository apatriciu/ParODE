/*
 * BTimesUAdamsMoulton.h
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#ifndef BTimesUAdamsMoulton_H_
#define BTimesUAdamsMoulton_H_

#include <KernelWrapper.h>

class BTimesUAdamsMoulton: public KernelWrapper {
public:
	virtual ~BTimesUAdamsMoulton();
	void Launch(
			  const vector<size_t>& WorkgroupSize,
			  const vector<size_t>& GlobalSize,
			  int queueIndex,
			  OpenCLMemBuffer* BBuffer,
			  OpenCLMemBuffer* UBuffer,
			  OpenCLMemBuffer* DestVectBuffer,
			  int nBRows,
			  int nBCols,
			  int nInputs);
	BTimesUAdamsMoulton();
protected:
	static const char* pszProgramFileName;
	static const char* pszKernelName;
	static const char* pszProgramCode;
};

#endif /* BTimesUAdamsMoulton_H_ */
