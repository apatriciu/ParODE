#ifndef _PARODE_H
#define _PARODE_H

//Source File for the main library interface.
#include <string>
#include <vector>
using namespace std;
// boost library files
#include <FTypeDef.h>
#include <GPUManagement.h>
#include <KernelWrapper.h>
#include <MMKernelScanUpsweep.h>
#include <MVVKernelScanUpsweep.h>
#include <LeftLeafCopyKernelScanUpsweep.h>
#include <MMKernelScanDownsweep.h>
#include <MVVKernelScanDownsweep.h>
#include <RootCopyKernelScanDownsweep.h>
#include <DistributeA.h>
#include <BTimesURK4thOrder.h>
#include <SystemOutput.h>
#include <BTimesUAdamsMoulton.h>
#include <ParODECppInterface.h>

class GPUODESolver;

//Implements the  interface functions for the ParODE solver.
//Declare and initialize the static member _Instance
class ParODE {
private:
	enum{
		NOT_INITIALIZED = 0, // GPU objects not created
		PARTIAL_INITIALIZATION, // GPU device objects created; no queues or kernels yet
		FULL_INITIALIZATION // ready to go
	} _InitializationState;
public:
    ErrorCode GetAvailableGPUs(vector<unsigned int>& vectIds);
    ErrorCode GetDeviceName(unsigned int device, string& deviceName);
    ErrorCode InitializeSelectedGPUs(const vector<unsigned int>& selectedDevices,
    								 string& strKernelsFolder,
    								 string& strIncludeFolder);
    //Simulates the ODE.
    //u func provides the input function. There should be some strict typing used in the definition of function u.
    
    ErrorCode SimulateODE(	const matrixf& A, const matrixf& B,
    						const matrixf& C, const matrixf& D,
    						int uIndex,
    						double tStart, double tEnd, double tStep,
    						const vectorf& x0,
    						SolverType solver,
    						vectorf& tVect, matrixf& xVect);

    ErrorCode SimulateODE(	const matrixf& A, const matrixf& B,
    						const matrixf& C, const matrixf& D,
    						int uIndex,
    						double tStart, double tEnd, double tStep,
    						const vectorf& x0,
    						SolverType solver,
    						fType tVect[], fType xVect[],
    						int &nSteps);

    //Initializes the GPU computations; 
    //- Identifies all GPUs in the system; Retrieve the parameters of each GPU; We use only the GPUs that have shared memory.
    //- compiles the kernels for all the GPUs;
    //- if everything is OK it returns ParODE_OK; otherwise an error code is returned;
    ErrorCode InitializeGPU(string& strKernelsFolder,
			 	 	 	 	 string& strIncludeFolder);

    void CreateDefaultKernels();

    //Return a description of the error.
    void GetErrorDescription(ErrorCode ErrNo, string& strErrDescription);

	// create a kernel for the input
	int RegisterUKernel(const string& strUProgram);

	// timing functions
	ErrorCode StartTimer();
	ErrorCode StopTimer(float &fTime);

#ifdef __TESTING_KERNELS__
	OpenCLKernel* GetKernel(int KIndex){
		return (OpenCLKernel*)(_vectGPUKernels[KIndex]);
	};
	GPUManagement* GetGPUManagement(){return _pGPUManagement;};
	vector<KernelWrapper*>& GetKernelsvectorReference(){return _vectGPUKernels;};
	vector<OpenCLKernel*>& GetUKernelsVectorReference(int index){return _vectKernelsInputs[index];};
#endif

  private:
    static ParODE* _Instance;
  protected:
	ParODE();
	virtual ~ParODE();
	// error control functions
	void Clean(); // delete all the kernels (if any) in _vectGPUKernels _vectOpenCLKernelsInputs

  public:
    //Implements the Singleton pattern
    static ParODE* Instance();
    static void Delete();

  private:
    GPUODESolver* CreateSolver(SolverType solver, int uIndex);
    GPUManagement * _pGPUManagement;
	vector<KernelWrapper*> _vectGPUKernels;
	// kernel 0 is matrix matrix multiplication upsweep for device 0
	// kernel 1 is matrix times vector plus vector upsweep for device 0
	// kernel 2 is left leaf copy operation upsweep
	// kernel 3 is matrix matrix multiplication downsweep for device 0
	// kernel 4 is matrix times vector plus vector downsweep for device 0
	// kernel 5 is the copy operation used in the downsweep stage for device 0
	// kernel 6 is the distributeA kernel
	// kernel 7 is the B times u kernel RK4
	// kernel 8 is the output kernel C x + D u
	// kernel 9 is the B x u kernel Adams Moulton
	string _strKernelFolderName;
	string _strIncludeFolder;
	// 
	vector< vector<OpenCLKernel*> >	_vectKernelsInputs;
};
#endif
