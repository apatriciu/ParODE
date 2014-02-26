/*
 * ParODECppInterface.h
 *
 *  Created on: 2013-11-26
 *      Author: alexandru
 */

#ifndef PARODECPPINTERFACE_H_
#define PARODECPPINTERFACE_H_
#include <FTypeDef.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
typedef boost::numeric::ublas::matrix<fType> matrixf;
typedef boost::numeric::ublas::vector<fType> vectorf;
#include <string>
#include <vector>
#include <ParODEEnums.h>

//
//Return a description of the error.

ErrorCode InitializeAllGPUs(std::string& strKernelFolder,
							std::string& strIncludeFolder);

ErrorCode GetAvailableGPUs(std::vector<unsigned int>& GPUIds);

ErrorCode GetDeviceName( unsigned int DeviceId, std::string& strDeviceName);

ErrorCode InitializeSelectedGPUs(const std::vector<unsigned int>& GPUIds,
								std::string& strKernelFolder,
								std::string& strIncludeFolder);

//Release all GPU datastructures.
void CloseGPU();

// timing functions
ErrorCode StartTimer();
ErrorCode StopTimer(float &fTime);

//Return a description of the error.
void GetErrorDescription(ErrorCode ErrNo, std::string& strErrDescription);

//Simulates the ODE.
//u func provides the input function. There should be some strict typing used in the definition of function u.
ErrorCode SimulateODE(	const matrixf & A,
						const matrixf & B,
						const matrixf & C,
						const matrixf & D,
						int uIndex,
						double tStart, double tEnd, double tStep,
						const vectorf& x0,
						SolverType solver,
						vectorf& tVect,
						matrixf& xVect);

// register an input kernel
int RegisterInput(const std::string& uFunc);

#endif /* PARODECPPINTERFACE_H_ */
