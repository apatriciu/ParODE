#ifndef _ADAMSMOULTON_H
#define _ADAMSMOULTON_H

#include <GPUFixedStepIterative.h>

class AdamsMoulton : public GPUODESolverFixedStepIterative {
private:
	fType* _ABarHost;
	fType* _BBarHost;
	vector<OpenCLMemBuffer*>	_vectABarBuffers; // buffers for storing a copy of ABar; one per device
	vector<OpenCLMemBuffer*>	_vectBBarBuffers; // buffers for storing a copy of BBar; one per device
	// objects for computing the first 4 iterations
	typedef vector<fType> 	state_type_init;
	static void system_function(const state_type_init & x, state_type_init & dxdt, const double t);
	static vector<vectorf>	_UInit;
	static matrixf 		   	_AInit;
	static matrixf 		   	_BInit;
	static double			_DeltaTInit;
private:
	void ComputeInitialState(const matrixf& A,
							 const matrixf& B,
							 const vectorf& x0,
							 double fDeltaT,
							 double tStart,
							 vectorf& x0Scan);
	void DeleteHostObjects();
	void DeleteGPUObjects();
	void DeleteVector(std::vector<OpenCLMemBuffer*>& vect);
	void ComputeNStepsPerDevice();
	void CreateInputBuffers();
	void AllocateMatrixBuffers(const matrixf& A, const matrixf& B, double tStep);
protected:
    virtual void InitializeSolverData(const matrixf& A, const matrixf& B,
										double& tStart, double tEnd,
										double tStep, const vectorf& x0);

	virtual void InitializeBatchData(OpenCLMemBuffer* pBufferMatrixSource,
									 OpenCLMemBuffer* pBufferVectorSource,
									 int nDeviceIndex,
									 double tStart, double tStep, int nSteps);

    virtual void CleanSolverData();
    virtual void GetInitialStatesAndInputs(matrixf& xVectInit,
			 	 	 	 	 	 	 	   matrixf& uVectInit);

  public:
    virtual  ~AdamsMoulton();

     AdamsMoulton(	GPUManagement* GPUM,
    		 	 	vector<KernelWrapper*>& vectGPUP,
    		 	 	vector<OpenCLKernel*>& vectU);

};
#endif
