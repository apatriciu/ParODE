/*
 * BTimesURK4thOrder.h
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#ifndef BTimesURK4thOrder_H_
#define BTimesURK4thOrder_H_

#include <KernelWrapper.h>

class BTimesURK4thOrder: public KernelWrapper {
public:
	virtual ~BTimesURK4thOrder();
	void Launch(
			  const vector<size_t>& WorkgroupSize,
			  const vector<size_t>& GlobalSize,
			  int queueIndex,
			  OpenCLMemBuffer* BBuffer,
			  OpenCLMemBuffer* UBuffer,
			  OpenCLMemBuffer* DestVectBuffer,
			  int nMatrixSize,
			  int nInputs);
	BTimesURK4thOrder();
protected:
	static const char* pszProgramFileName;
	static const char* pszKernelName;
	static const char* pszProgramCode;
};

#endif /* BTimesURK4thOrder_H_ */
