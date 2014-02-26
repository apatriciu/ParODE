#ifndef _GPUODESOLVER_H
#define _GPUODESOLVER_H

#include "ParODE.h"

#include <string>
using namespace std;
#include <GPUManagement.h>
#include <KernelWrapper.h>

class GPUODESolver {
  public:
     GPUODESolver(GPUManagement* GPUM,
    		 vector<KernelWrapper*>& vectGPUP,
    		 vector<OpenCLKernel*>& vectUKernels);

    //Simulates the ODE.
    //u func provides the input function. There should be some strict typing used in the definition of function u.
    
	virtual ErrorCode SimulateODE(const matrixf& A, const matrixf& B, const matrixf& C, const matrixf& D, 
								double tStart, double tEnd, double tStep, 
								const vectorf& x0, vectorf & tVect, matrixf & xVect) = 0;
	virtual ErrorCode SimulateODE(const matrixf& A, const matrixf& B, const matrixf& C, const matrixf& D,
			double tStart, double tEnd, double tStep,
			const vectorf& x0, fType tVect[], fType xVect[], int &nSteps);
	virtual  ~GPUODESolver(){
		_vectGPUKernels.clear();
		_vectUKernels.clear();
	};
protected:
	// predefined kernels
	GPUManagement* _pGPUM;
	vector<KernelWrapper*> _vectGPUKernels;
	// kernel 0 is matrix matrix multiplication upsweep for device 0
	// kernel 1 is matrix times vector plus vector upsweep for device 0
	// kernel 2 left leaf copy kernel scan upsweep for device 0
	// kernel 3 is matrix matrix multiplication downsweep for device 0
	// kernel 4 is matrix times vector plus vector downsweep for device 0
	// kernel 5 is the copy operation used in the downsweep stage for device 0
	// kernel 6 is DistributeA for device 0
	// kernel 7 is BTimesURK4thOrder for device 0
	// kernel 8 is C x + D u kernel for device 0
	// kernel 9 is BTimesUAdamsMoulton for device 0
	// etc for device 1
	unsigned int			_nKernels; // how many kernels we have for one device
	vector<OpenCLKernel*>	_vectUKernels; // kernels for computing u; one kernel per device
};
#endif
