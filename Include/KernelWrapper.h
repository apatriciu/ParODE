/*
 * KernelWrapper.h
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#ifndef KERNELWRAPPER_H_
#define KERNELWRAPPER_H_

#include <GPUManagement.h>
#include <string>
#include <vector>
#include <ParODETypeDefs.h>

using namespace std;

class KernelWrapper{
public:
	virtual ~KernelWrapper();
	template<class KernelType>
	static KernelWrapper* CreateOn(OpenCLDeviceAndContext* pDC,
									string strKernelFolder = string("./"),
									string strIncludeFolder = string(),
									bool fromInternalString = false){
		KernelWrapper* pKW = new KernelType;
		string KernelProgramFile;
		KernelProgramFile = strKernelFolder + pKW->_strKernelFileName;
		bool bCreated = pDC->CreateProgram(	KernelProgramFile,
											(pKW->_strKernelName),
											strIncludeFolder,
											(pKW->_pKernel));
		if(!bCreated)
			return NULL;
		return pKW;
	}
	// this is just a casting operator that allows us to
	// launch the grid in an elegant way
	template<class KernelTypeCall>
	KernelTypeCall* GetKernel(){
		return dynamic_cast<KernelTypeCall*>(this);
	};
	// casting to OpenCLKernel
	operator OpenCLKernel* (){
		return _pKernel;
	};
	// nothing to do here;
	// this will have to be implemented by a child class
	void Launch(const vector<size_t>& WorkgroupSize,
				  const vector<size_t>& GlobalSize,
				  int queueIndex){};
protected:
	KernelWrapper();
	OpenCLKernel* 	_pKernel;
	string 			_strKernelName;
	string 			_strKernelFileName;
	string 			_strProgramCode;
};

#endif /* KERNELWRAPPER_H_ */
