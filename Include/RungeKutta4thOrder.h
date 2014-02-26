#ifndef _RUNGEKUTTA4THORDER_H
#define _RUNGEKUTTA4THORDER_H

#include <vector>
#include "GPUFixedStepIterative.h"

class RungeKutta4thOrder : public GPUODESolverFixedStepIterative {
private:
	fType* _ABarHost;
	fType* _BBarHost;
	vector<OpenCLMemBuffer*>	_vectABarBuffers; // buffers for storing a copy of ABar; one per device
	vector<OpenCLMemBuffer*>	_vectBBarBuffers; // buffers for storing a copy of BBar; one per device
private:
	void DeleteHostObjects();
	void DeleteGPUObjects();
	void DeleteVector(std::vector<OpenCLMemBuffer*>& vect);
protected:
    virtual void InitializeSolverData(const matrixf& A, const matrixf& B,
										double& tStart, double tEnd, double tStep, const vectorf& x0);

	virtual void InitializeBatchData(OpenCLMemBuffer* pBufferMatrixSource,
									 OpenCLMemBuffer* pBufferVectorSource,
									 int nDeviceIndex,
									 double tStart, double tStep, int nSteps);

    virtual void CleanSolverData();

  public:
    virtual  ~RungeKutta4thOrder();
    RungeKutta4thOrder(GPUManagement* GPUM,
    		vector<KernelWrapper*>& vectGPUP,
    		vector<OpenCLKernel*>& vectU);
};
#endif
