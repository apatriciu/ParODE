#ifndef _GPUFIXEDSTEPITERATIVE_H
#define _GPUFIXEDSTEPITERATIVE_H


#include <ParODE.h>

#include <string>
using namespace std;

#include <GPUODESolver.h>

//Implements a generic parallel solver. We use a scan operation to solve the recurrence. The generic algorithm is implemented here, whereas the vector initialization implementation is deffered to subclasses.
class GPUODESolverFixedStepIterative : public GPUODESolver {
protected:
	// the total number of simulation steps in the output vector
	unsigned long _nTotalStepsSimulation;
	// the total number of steps in the simulation
	// this is different from _nTotalStepsSimulation for multi-step methods
	unsigned long _nSteps;
	// the total number of steps per device in one batch
	vector<unsigned long> _nStepsPerDevice;
	// original system size
	unsigned int _nSystemSize;
	// scan matrix size
	unsigned int _nMatrixSize;
	// number of inputs
	unsigned int _nInputs;
	// number of outputs
	unsigned int _nOutputSize;
	int				_bZeroC; // true if C == 0 
	int				_bZeroD; // true if D == 0 
	// groups size; this is a function of the shared memory
	size_t _nLocalSizeMM;
	size_t _nLocalSizeMVVColumn;
	size_t _nLocalSizeMVVRow;
	// buffers for vectors; will hold  one batch of vectors for each device
	// one is source and the other one is destination
	// InitializeSolverData is responsible for the alocation of the buffer on each device
	vector<OpenCLMemBuffer*> _BufferVectors[2]; 
	// buffer for matrices; will hold one batch of matrices for each device
	// one is source and the other one is destination
	// InitializeSolverData is responsible for the alocation of the buffer on each device
	vector<OpenCLMemBuffer*> _BufferMatrices[2];
	// GPUBuffers for system output C matrix; one per device
	vector<OpenCLMemBuffer*> _BufferC;
	// GPUBuffers for system output D matrix; one per device
	vector<OpenCLMemBuffer*> _BufferD;
	// GPUBuffers for system output y; one per device
	vector<OpenCLMemBuffer*>	_BufferYVectors;
	vector<OpenCLMemBuffer*>	_vectUBuffers; // buffers for storing u one per device;
	// input stride; this is used by the output kernels
	int _nInputStride;
	int _nInputOffset;
	// state stride; this is used by the output kernels
	int _nStateStride;
	int _nStateOffset;
	bool _bStrideSet;
	// initial state and matrix
	// these may not be exacly the _X0 provided to the solver
	vectorf _X0;
	matrixf _M0;
	bool _bInitialConditionsSet;
protected:
	void SetStrides(int nInputStride, int nStateStride,
					int nInputOffset = 0, int nStateOffset = 0){
		_bStrideSet = true;
		_nInputStride = nInputStride;
		_nStateStride = nStateStride;
		_nInputOffset = nInputOffset;
		_nStateOffset = nStateOffset;
	};
	void SetInitialConditions(vectorf& X0){
		_X0 = X0;
		// set _M0 to identity
		_M0.resize(_nMatrixSize, _nMatrixSize);
		for(int row = 0; row < _nMatrixSize; row++)
			for(int col = 0; col < _nMatrixSize; col++)
				_M0(row, col) = row == col ? 1.0 : 0.0;
		_bInitialConditionsSet = true;
	};
	void SetLocalSizeAttributes(size_t nLocalSizeMM, size_t nLocalSizeMVVColumn, size_t nLocalSizeMVVRow){
		_nLocalSizeMM = nLocalSizeMM;
		_nLocalSizeMVVColumn = nLocalSizeMVVColumn;
		_nLocalSizeMVVRow = nLocalSizeMVVRow;
	};
	void SetNStepsPerDevice(vector<unsigned long>& nStepsPerDevice){
		_nStepsPerDevice = nStepsPerDevice;
	};
	void SetNSteps(unsigned long nSteps){_nSteps = nSteps;};
	void SetBufferVector(vector<OpenCLMemBuffer*> BufferVectors[], vector<OpenCLMemBuffer*> BufferMatrices[]){
		_BufferVectors[0] = BufferVectors[0];
		_BufferVectors[1] = BufferVectors[1];
		_BufferMatrices[0] = BufferMatrices[0];
		_BufferMatrices[1] = BufferMatrices[1];
	};
	void BuildOutputObjects(const matrixf& C, const matrixf& D);
	void DeleteGPUFSIObjects();
  public:
	virtual  ~GPUODESolverFixedStepIterative(){
	};
	GPUODESolverFixedStepIterative(GPUManagement* GPUM,
										vector<KernelWrapper*>& vectGPUP,
										vector<OpenCLKernel*>& vectUKernels);

    //Simulates the ODE.
    //u func provides the input function. There should be some strict typing used in the definition of function u.
	virtual ErrorCode SimulateODE(const matrixf& A, const matrixf& B, const matrixf& C, const matrixf& D, 
									double tStart, double tEnd, double tStep, 
									const vectorf& x0, vectorf & tVect, matrixf & xVect);
	// input kernel specification
	// the kernel will be launch in a workfront with two dimensions
	// the first dimension provides all the inputs for one time stamp
	// the second dimension provides the vectors for each time stamp
	// the kernel should have the following params
	// - tStart the start time for the current batch
	// - tStep time step
	// - vectResult buffer for results - this should have at least cl_global_size(0) * cl_global_size(1) elements
  protected:
    virtual void InitializeSolverData(	const matrixf& A, const matrixf& B,
										double& tStart, double tEnd, double tStep,
										const vectorf& x0) = 0;
	virtual void InitializeBatchData(OpenCLMemBuffer* pBufferMatrixSource,
									 OpenCLMemBuffer* pBufferVectorSource,
									 int nDeviceIndex,
									 double tStart, double tStep, int nSteps) = 0;
    virtual void CleanSolverData() = 0;
    virtual void GetInitialStatesAndInputs(matrixf& xVectInit,
    									   matrixf& uVectInit);
};
#endif
