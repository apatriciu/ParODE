#include <cassert>
#include <GPUODESolver.h>
#include <GPUManagement.h>
#include <KernelWrapper.h>
#include <string>
#include <stdexcept>

GPUODESolver::GPUODESolver(
		GPUManagement* GPUM,
		vector<KernelWrapper*>& vectGPUP,
		vector<OpenCLKernel*>& vectUKernels):
	_pGPUM(GPUM)
{
	_vectGPUKernels = vectGPUP;
	_vectUKernels = vectUKernels;
	_nKernels = vectGPUP.size() / GPUM->GetNumberOfDevices();
	assert((vectGPUP.size() % GPUM->GetNumberOfDevices()) == 0);
	assert(vectUKernels.size() == GPUM->GetNumberOfDevices());
	if((vectGPUP.size() % GPUM->GetNumberOfDevices()) != 0){
		std::string strParam("2 and 3");
		throw std::invalid_argument(strParam);
	}
}

ErrorCode GPUODESolver::SimulateODE(const matrixf& A, const matrixf& B, const matrixf& C, const matrixf& D,
		double tStart, double tEnd, double tStep,
		const vectorf& x0, fType tVect[], fType xVect[], int &nSteps){
	matrixf xVectLocal;
	vectorf tVectLocal;
	ErrorCode retVal = SimulateODE(A, B, C, D, tStart, tEnd, tStep,
								x0, tVectLocal, xVectLocal);
	if(retVal != ParODE_OK){
		nSteps = 0;
		return retVal;
	}
	nSteps = tVectLocal.size();
	int nOutputs = xVectLocal.size1();
	for(int col = 0; col < nSteps; col++){
		for(int row = 0; row < nOutputs; row++)
			xVect[col * nOutputs + row] = xVectLocal(row, col);
		tVect[col] = tVectLocal[col];
	}
	return ParODE_OK;
}
