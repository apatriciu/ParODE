/*
 * ParODECInterface.h
 *
 *  Created on: 2013-11-26
 *      Author: alexandru
 */

#ifndef PARODECINTERFACE_H_
#define PARODECINTERFACE_H_

#include <FTypeDef.h>

#ifdef C_INCLUDE
#define C_EXPORT extern
#else
#define C_EXPORT
#endif

#ifndef C_INCLUDE
extern "C"
{
#endif
	//Returns a description of the error.
	// input - ErrNo
	// output - strErrDescription[] description of the  error;
	// the user is resposible with the memory space management
	C_EXPORT void GetErrorDescription(int *ErrNo, char strErrDescription[]);
	// Initialize all GPUs on the system
	// output - nErr[0] - error code (0 - Success)
	C_EXPORT void InitializeAllGPUs(char* pszKernelFolder,
			 char* pszIncludeFolder,
			 int nErr[]);
	// Look for all the available GPUs on the system
	// The user is resposible with the memory allocation and deallocation
	// outputs
	// 	nGPUs[0] - the number of elements in the array GPUIds
	//	GPUIds[] - array with the available GPUs
	// 	nErr[0] - error code (0 - Success)
	C_EXPORT void GetAvailableGPUs( int nGPUs[], int GPUIds[], int nErr[]);
	// Reads the device name
	// The user is responsible with the memory allocation and deallocation 
	// input - *DeviceId - the id of teh device to be querried
	// outputs:
	// 	strDeviceName - name of the device
	//	nErr[0] - error code (0 - Success)
	C_EXPORT void GetDeviceName( int *DeviceId, char strDeviceName[], int nErr[]);
	// Initialize selected GPUs on the system; These should be a subset of
	// the list returned by GetAvailableGPUs
	// inputs:
	// 	nGPUs[0] - number of GPUs in GPUIds
	//	GPUIds[] - GPUs ids
	// outputs:
	//	nErr[0] - error code (0 - Success)
	C_EXPORT void InitializeSelectedGPUs( int *nGPUs, int GPUIds[],
				char* pszKernelFolder,
				char* pszIncludeFolder, int nErr[]);

		// interface for plain C calls
		// the caller should take care of memory allocation and release for A, B, C, D
		// the caller should take care of memory release for tVect and xVect
	C_EXPORT void SimulateODE(	fType A[], // row major A
						fType B[], // row major B
						fType C[], // row major C
						fType D[], // row major D
						int *nInputs,
						int *nStates,
						int *nOutputs,
						int *ZeroB,
						int *ZeroC,
						int *ZeroD,
						int *uIndex,
						double *tStart, double *tEnd, double *tStep,
						fType x0[],
						int *solver,
						// Outputs
						fType tVect[],
						fType xVect[],
						int *nSteps,
						int nErr[]);
	// Register an input for simulation
	// input - uFunc - string with the input function. 
	// 	This is a C function that will be executed on the device.
	// outputs
	//	uIndex[0] - the index of the input; this will be used on the simulate call
	// 	nErr[0] - error code (0 - Success)
	C_EXPORT void RegisterInput(char uFunc[], int uIndex[], int nErr[]);

	//Release all GPU datastructures.
	C_EXPORT void CloseGPUC(); // this should be void CloseGPUC(void) to avoid some warnings

	// timing functions
	// Starts the GPU timer
	// output - nErr[0] - error code (0 - Success)
	C_EXPORT void StartTimer(int nErr[]);
	// Returns the  time passes from the previous call to StartTimer
	// output - 
	// 	fTime[0] - time in ms passed from the previous call to  StartTimer
	// 	nErr[0] - error code (0 - Success)
	C_EXPORT void StopTimer(float fTime[], int nErr[]);
#ifndef C_INCLUDE
}
#endif

#endif /* PARODECINTERFACE_H_ */
