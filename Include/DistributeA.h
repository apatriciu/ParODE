/*
 * DistributeA.h
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#ifndef DistributeA_H_
#define DistributeA_H_

#include <KernelWrapper.h>

class DistributeA: public KernelWrapper {
public:
	virtual ~DistributeA();
	void Launch(
			  const vector<size_t>& WorkgroupSize,
			  const vector<size_t>& GlobalSize,
			  int queueIndex,
			  OpenCLMemBuffer* SourceA,
			  OpenCLMemBuffer* DestMatricesVector,
			  int nElements);
	DistributeA();
protected:
	static const char* pszProgramFileName;
	static const char* pszKernelName;
	static const char* pszProgramCode;
};

#endif /* DistributeA_H_ */
