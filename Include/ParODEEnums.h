/*
 * ParODEEnums.h
 *
 *  Created on: 2013-11-27
 *      Author: alexandru
 */

#ifndef PARODEENUMS_H_
#define PARODEENUMS_H_

enum SolverType {
  RUNGE_KUTTA = 0,
  ADAMS_BASHFORTH_MOULTON = 1
};

//Error Codes that may be returned by the solver functions.
enum ErrorCode {
  ParODE_OK = 0,
  ParODE_NoGPU = -1,
  ParODE_NotInitialized = -2,
  ParODE_CouldNotCreateGPUKernels = -3,
  ParODE_NotEnoughResources = -4,
  ParODE_InvalidDeviceId = -5,
  ParODE_InvalidSystem = -6,
  ParODE_InvalidSolverType = -7
};

#endif /* PARODEENUMS_H_ */
