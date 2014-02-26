#include <GPUFixedStepIterative.h>
#include <GPUManagement.h>
#include <GPUODESolver.h>
#include <KernelWrapper.h>
#include <cstring>
#include <ParODEException.h>

GPUODESolverFixedStepIterative::GPUODESolverFixedStepIterative(
		GPUManagement* GPUM,
		vector<KernelWrapper*>& vectGPUP,
		vector<OpenCLKernel*>& vectUKernels):
	GPUODESolver(GPUM, vectGPUP, vectUKernels),
	_nMatrixSize(0),
	_bInitialConditionsSet(false),
	_nSteps(0),
	_bZeroC(1),
	_bZeroD(1){
}

void GPUODESolverFixedStepIterative::DeleteGPUFSIObjects(){
	// delete GPU Objects
	// delete C matrix buffers
	for(int ii = 0; ii < _BufferC.size(); ii++){
		delete _BufferC[ii];
		_BufferC[ii] = NULL;
	}
	_BufferC.clear();
	// delete D matrix buffers
	for(int ii = 0; ii < _BufferD.size(); ii++){
		delete _BufferD[ii];
		_BufferD[ii] = NULL;
	}
	_BufferD.clear();
	// delete Y vector buffers
	for(int ii = 0; ii < _BufferYVectors.size(); ii++){
		delete _BufferYVectors[ii];
		_BufferYVectors[ii] = NULL;
	}
	_BufferYVectors.clear();
	// delete vectors buffers
	for(int ii = 0; ii < 2; ii++){
		for(int jj = 0; jj < _BufferVectors[ii].size(); jj++){
			delete _BufferVectors[ii][jj];
			_BufferVectors[ii][jj] = NULL;
		}
		_BufferVectors[ii].clear();
	}
	// delete matrices buffers
	for(int ii = 0; ii < 2; ii++){
		for(int jj = 0; jj < _BufferMatrices[ii].size(); jj++){
			delete _BufferMatrices[ii][jj];
			_BufferMatrices[ii][jj] = NULL;
		}
		_BufferMatrices[ii].clear();
	}
}

void GPUODESolverFixedStepIterative::BuildOutputObjects(
		const matrixf& C,
		const matrixf& D){
	// Set Specific Output Values
	_bZeroD = D.size1() == 0;
	_bZeroC = C.size1() == 0;
	if(C.size1() != 0 || D.size1() != 0){
		// some consistency checks
		if(C.size1() != 0)
			assert( C.size2() == _nSystemSize );
		if(D.size1() != 0)
			assert( D.size2() == _nInputs );
		if(C.size1() != 0 && D.size1() != 0)
			assert(C.size1() == D.size1());
		int nDevices = _nStepsPerDevice.size();
		// set output size
		_nOutputSize = max(C.size1(), D.size1());
		try{
			// allocate GPUBuffers
			for(int nDI = 0; nDI < nDevices; nDI++){
				OpenCLMemBuffer* pBuffer;
				// create C buffer
				size_t szBuffer = C.size1() * C.size2() * sizeof(fType);
				if(szBuffer == 0) szBuffer = 1;
				_pGPUM->GetDeviceAndContext(nDI)->CreateMemBuffer(szBuffer, pBuffer);
				// copy the value of the C to GPU
				if(szBuffer != 0){
					fType* pHostBuffer;
					pHostBuffer = new fType[C.size1() * C.size2()];
					for(int row = 0; row < C.size1(); row++)
						for(int col = 0; col < C.size2(); col++)
							pHostBuffer[row * C.size2() + col] = C(row, col);
					pBuffer->MemWrite(pHostBuffer, pBuffer->GetContext()->GetQueue(0));
					delete [] pHostBuffer;
				}
				_BufferC.push_back(pBuffer);
				// create D buffer
				szBuffer = D.size1() * D.size2() * sizeof(fType);
				if(szBuffer == 0) szBuffer = 1;
				_pGPUM->GetDeviceAndContext(nDI)->CreateMemBuffer(szBuffer, pBuffer);
				// copy the value of the D to GPU
				if(szBuffer != 0){
					fType* pHostBuffer;
					pHostBuffer = new fType[D.size1() * D.size2()];
					for(int row = 0; row < D.size1(); row++)
						for(int col = 0; col < D.size2(); col++)
							pHostBuffer[row * D.size2() + col] = D(row, col);
					pBuffer->MemWrite(pHostBuffer, pBuffer->GetContext()->GetQueue(0));
					delete [] pHostBuffer;
				}
				_BufferD.push_back(pBuffer);
				// create Y buffers
				szBuffer = _nOutputSize * _nStepsPerDevice[nDI] * sizeof(fType);
				_pGPUM->GetDeviceAndContext(nDI)->CreateMemBuffer(szBuffer, pBuffer);
				_BufferYVectors.push_back(pBuffer);
			}
		}
		catch(std::exception& except){
			DeleteGPUFSIObjects();
			std::cerr << except.what();
			throw ParODEException();
			return;
		}
	}
	else
		_nOutputSize = _nSystemSize;
	return;
}

void GPUODESolverFixedStepIterative::GetInitialStatesAndInputs(matrixf& xVectInit,
    					       matrixf& uVectInit){
}

ErrorCode GPUODESolverFixedStepIterative::SimulateODE(	const matrixf& A, const matrixf& B, const matrixf& C, const matrixf& D, 
														double tStart, double tEnd, double tStep, 
														const vectorf& x0, vectorf & tVect, matrixf & xVect){
	try{
		InitializeSolverData( A, B, tStart, tEnd, tStep, x0);
		BuildOutputObjects(C, D);
	}
	catch(std::exception& except){
		return ParODE_NotEnoughResources;
	}
	tVect.resize(_nTotalStepsSimulation);
	for(int ii = 0; ii < _nTotalStepsSimulation - _nSteps; ii++)
		tVect[ii] = tStart - (_nTotalStepsSimulation - _nSteps - ii) * tStep;
	tVect[ _nTotalStepsSimulation - _nSteps ] = tStart;
	for(int ii = 1; ii < _nSteps; ii++)
		tVect[(_nTotalStepsSimulation - _nSteps) + ii] = tVect[ (_nTotalStepsSimulation - _nSteps) + ii - 1] + tStep;
	xVect.resize(_nOutputSize, _nTotalStepsSimulation);
	matrixf xVectInit;
	try{
		if(_nTotalStepsSimulation != _nSteps){
			matrixf xVectInit;
			matrixf uVectInit;
			GetInitialStatesAndInputs(xVectInit, uVectInit);
			// compute outVectInit
			if(_bZeroC == 1 && _bZeroD == 1){
				// copy the state in the output
				for(int ii = 0; ii < _nTotalStepsSimulation - _nSteps; ii++)
					for(int jj = 0; jj < _nOutputSize; jj++)
						xVect(jj, ii) = xVectInit(jj, ii);
			} else{
				for(int ii = 0; ii < _nTotalStepsSimulation - _nSteps; ii++)
					for(int jj = 0; jj < _nOutputSize; jj++){
						fType val = 0.0;
						if(!_bZeroC)
							for(int kk = 0; kk < _nSystemSize; kk++)
								val += C(jj, kk) * xVectInit(kk, ii);
						if(!_bZeroD)
							for(int kk = 0; kk < _nInputs; kk++)
								val += D(jj, kk) * uVectInit(kk, ii);
						xVect(jj, ii) = val;
					}
			}
		}
	}
	catch(exception &e){
		std::cerr << "Exception caught : " << e.what() << std::endl;
		return ParODE_InvalidSolverType;
	}
	// compute the total number of steps per batch
	unsigned long NSB = 0; // number of steps per batch
	unsigned long maxStepsPerDevice = 0;
	unsigned int nDevices = _pGPUM->GetNumberOfDevices();
	assert(_nStepsPerDevice.size() == nDevices);
	for(int ii = 0; ii < _nStepsPerDevice.size(); ii++){
		NSB += _nStepsPerDevice[ii];
		if(maxStepsPerDevice < _nStepsPerDevice[ii])
			maxStepsPerDevice = _nStepsPerDevice[ii];
	}
	int nB;
	// accumulators for computing the global scan
	fType* accMatrix = new fType[_nMatrixSize * _nMatrixSize];
	fType* accVector = new fType[_nMatrixSize];
	// initialize the accumulator vector with the initial value X0
	// initialize the accumulator matrix with identity
	for(int ii = 0;  ii < _nMatrixSize; ii++){
		accVector[ii] = _X0(ii);
		for(int jj = 0; jj < _nMatrixSize; jj++)
			accMatrix[ii * _nMatrixSize + jj] = _M0(ii, jj);
	}
	long globalResultTIndex(_nTotalStepsSimulation - _nSteps);
	// global simulation time
	try{
		for(nB = 0; nB < (_nSteps - 1) / NSB + 1; nB++){
			// test if  this  the last iteration
			if( nB == (_nSteps - 1) / NSB ){
				long nStepsLeft = _nSteps - nB * NSB;
				// distribute the steps left
				int ii;
				for(ii = 0; ii < _nStepsPerDevice.size(); ii++){
					if(nStepsLeft <= _nStepsPerDevice[ii])
						break;
					nStepsLeft -= _nStepsPerDevice[ii];
				}
				assert(ii < _nStepsPerDevice.size());
				// re-adjust to a power of two
				int nStepsii(1);
				while(nStepsii < nStepsLeft)
					nStepsii <<= 1;
				_nStepsPerDevice[ii++] = nStepsii;
				for(;ii < _nStepsPerDevice.size();  ii++) _nStepsPerDevice[ii] = 0;
			}
			unsigned int nDevices = _nStepsPerDevice.size();
			vector<int> nSource(nDevices, 0), nDest(nDevices, 1);
			double tStartBatch = tStart + nB * NSB * tStep;
			vector<int> nElementStart( _nStepsPerDevice.size(), 0 );
			for(int ii = 1; ii < nElementStart.size(); ii++)
				nElementStart[ii] = nElementStart[ii-1] + _nStepsPerDevice[ii-1];
			for(int nDI = 0; nDI < nDevices; nDI++)
				InitializeBatchData(_BufferMatrices[nSource[nDI]][nDI],
									_BufferVectors[nSource[nDI]][nDI],
									nDI,
									tStartBatch + nElementStart[nDI] * tStep,
									tStep,
									_nStepsPerDevice[nDI]);
			// up-sweep
			for(int d = 1; d  < maxStepsPerDevice; d <<= 1 ){
				for(int nDI = 0; nDI < nDevices; nDI++)
					if(d  < _nStepsPerDevice[nDI]){
						// set parameters for MM upsweep
						vector<size_t> szLocal(2), szGlobal(2);
						szLocal[0] = szLocal[1] = _nLocalSizeMM;
						szGlobal[0] = szLocal[0] * _nStepsPerDevice[nDI] / (2 * d);
						szGlobal[1] = szLocal[1];
						_vectGPUKernels[nDI * _nKernels]->GetKernel<MMKernelScanUpsweep>()->Launch(
													 szLocal, szGlobal, 0,
													_BufferMatrices[nSource[nDI]][nDI],
													_BufferMatrices[nDest[nDI]][nDI],
													d,
													_nMatrixSize);
						// set parameters for MVV upsweep
						szLocal[0] = _nLocalSizeMVVColumn;
						szLocal[1] = _nLocalSizeMVVRow;
						szGlobal[0] = szLocal[0] * _nStepsPerDevice[nDI] / (2 * d);
						szGlobal[1] = szLocal[1];
						_vectGPUKernels[nDI * _nKernels + 1]->GetKernel<MVVKernelScanUpsweep>()->Launch(
													szLocal, szGlobal, 1,
													_BufferMatrices[nSource[nDI]][nDI],
													_BufferVectors[nSource[nDI]][nDI],
													_BufferVectors[nDest[nDI]][nDI],
													d,
													_nMatrixSize);
						// copy the left leaf
						// set parameters for copy kernel for the matrix component
						int nMSZ = _nMatrixSize * _nMatrixSize;
						vector<size_t> szLocalCopy(1), szGlobalCopy(1);
						szLocalCopy[0] = min(nMSZ, (int)(_pGPUM->GetDeviceAndContext(nDI)->MaxWorkgroupSize()));
						szGlobalCopy[0] = szLocalCopy[0] * _nStepsPerDevice[nDI] / (2 * d);
						_vectGPUKernels[nDI * _nKernels + 2]->GetKernel<LeftLeafCopyKernelScanUpsweep>()->Launch(
								szLocalCopy, szGlobalCopy, 0,
								_BufferMatrices[nSource[nDI]][nDI],
								_BufferMatrices[nDest[nDI]][nDI],
								d,
								nMSZ);
						// set parameters for copy kernel for the vector component
						szLocalCopy[0] = _nLocalSizeMM;
						szGlobalCopy[0] = szLocalCopy[0] * _nStepsPerDevice[nDI] / (2 * d);
						_vectGPUKernels[nDI * _nKernels + 2]->GetKernel<LeftLeafCopyKernelScanUpsweep>()->Launch(
								szLocalCopy, szGlobalCopy, 1,
								_BufferVectors[nSource[nDI]][nDI],
								_BufferVectors[nDest[nDI]][nDI],
								d,
								_nMatrixSize);
					}
				for(int nDI = 0; nDI < _nStepsPerDevice.size(); nDI++)
					if(d  < _nStepsPerDevice[nDI]){
						// synchronize both queues for device nDI
						_pGPUM->GetDeviceAndContext(nDI)->GetQueue(0)->Synchronize();
						_pGPUM->GetDeviceAndContext(nDI)->GetQueue(1)->Synchronize();
						// change source with destination
						nSource[nDI] = nDest[nDI];
						nDest[nDI] = (nSource[nDI] + 1) % 2;
					}
			}
			// copy the reduction data from the devices to the host memory
			// temporary buffer for copying data
			fType* bufferMatrix = new fType[_nMatrixSize * _nMatrixSize];
			fType* bufferVector = new fType[_nMatrixSize];
			fType* tempMatrix = new fType[_nMatrixSize * _nMatrixSize];
			fType* tempVector = new fType[_nMatrixSize];
			for(int nDI = 0; nDI < _nStepsPerDevice.size(); nDI++){
				// copy the reduction result
				_BufferMatrices[nSource[nDI]][nDI]->MemRead(	(void*)bufferMatrix, _BufferMatrices[nSource[nDI]][nDI]->GetContext()->GetQueue(0),
														_nMatrixSize * _nMatrixSize * (_nStepsPerDevice[nDI] - 1) * sizeof(fType), _nMatrixSize * _nMatrixSize * sizeof(fType));
				_BufferVectors[nSource[nDI]][nDI]->MemRead(	(void*)bufferVector, _BufferVectors[nSource[nDI]][nDI]->GetContext()->GetQueue(0),
														_nMatrixSize * (_nStepsPerDevice[nDI] - 1) * sizeof(fType), _nMatrixSize * sizeof(fType));
				// wait for the memcopy to complete
				_BufferMatrices[nSource[nDI]][nDI]->GetContext()->GetQueue(0)->Synchronize();
				// save the accumulator into the temporary buffers
				memcpy(tempMatrix, accMatrix, _nMatrixSize * _nMatrixSize * sizeof(fType));
				memcpy(tempVector, accVector, _nMatrixSize * sizeof(fType));
				// compute the new accumulator matrix
				for(int row = 0; row < _nMatrixSize; row++)
					for(int column = 0; column < _nMatrixSize; column++){
						fType resProd = 0.0;
						for(int kk = 0; kk < _nMatrixSize; kk++)
							resProd += bufferMatrix[row * _nMatrixSize + kk] * tempMatrix[kk * _nMatrixSize + column];
						accMatrix[row * _nMatrixSize + column] = resProd;
					}
				// compute the new accumulator vector
				for(int row = 0; row < _nMatrixSize; row++){
					fType resVal(0.0);
					for(int column = 0; column < _nMatrixSize; column++)
						resVal += bufferMatrix[row * _nMatrixSize + column] * tempVector[column];
					accVector[row] = resVal + bufferVector[row];
					}
				// copy the old accumulators in the reduction vectors GPU buffers
				_BufferMatrices[nSource[nDI]][nDI]->MemWrite(tempMatrix, _BufferMatrices[nSource[nDI]][nDI]->GetContext()->GetQueue(0),
					_nMatrixSize * _nMatrixSize * (_nStepsPerDevice[nDI] - 1) * sizeof(fType), _nMatrixSize * _nMatrixSize * sizeof(fType));
				_BufferVectors[nSource[nDI]][nDI]->MemWrite(tempVector, _BufferVectors[nSource[nDI]][nDI]->GetContext()->GetQueue(0),
					_nMatrixSize * (_nStepsPerDevice[nDI] - 1) * sizeof(fType), _nMatrixSize * sizeof(fType));
				// wait for the memcpy to finish
				_BufferVectors[nSource[nDI]][nDI]->GetContext()->GetQueue(0)->Synchronize();
			}
			delete [] bufferMatrix;
			delete [] bufferVector;
			delete [] tempMatrix;
			delete [] tempVector;
			// downsweep stage
			for(int d = maxStepsPerDevice / 2; d  > 0; d >>= 1 ){
				for(int nDI = 0; nDI < nDevices; nDI++)
					if(d  < _nStepsPerDevice[nDI]){
						// set parameters for MM kernel downsweep
						vector<size_t> szLocal(2), szGlobal(2);
						szLocal[0] = szLocal[1] = _nLocalSizeMM;
						szGlobal[0] = szLocal[0] * _nStepsPerDevice[nDI] / (2 * d);
						szGlobal[1] = szLocal[1];
						_vectGPUKernels[nDI * _nKernels + 3]->GetKernel<MMKernelScanDownsweep>()->Launch(
								szLocal, szGlobal, 0,
								_BufferMatrices[nSource[nDI]][nDI],
								_BufferMatrices[nDest[nDI]][nDI],
								d,
								_nMatrixSize);
						// call the MVV kernel for downsweep; queue 1
						szLocal[0] = _nLocalSizeMVVColumn;
						szLocal[1] = _nLocalSizeMVVRow;
						szGlobal[0] = szLocal[0] * _nStepsPerDevice[nDI] / (2 * d);
						szGlobal[1] = szLocal[1];
						_vectGPUKernels[nDI * _nKernels + 4]->GetKernel<MVVKernelScanDownsweep>()->Launch(
								szLocal, szGlobal, 1,
								_BufferMatrices[nSource[nDI]][nDI],
								_BufferVectors[nSource[nDI]][nDI],
								_BufferVectors[nDest[nDI]][nDI],
								d,
								_nMatrixSize);
						// set parameters Matrix copy downsweep
						// call the copy kernel for downsweep; queue 0 (copy the matrix component)
						vector<size_t> szLocalCopy(1), szGlobalCopy(1);
						int nMSZ = _nMatrixSize * _nMatrixSize;
						szLocalCopy[0] = min(nMSZ, (int)(_pGPUM->GetDeviceAndContext(nDI)->MaxWorkgroupSize()));
						szGlobalCopy[0] = szLocalCopy[0] * _nStepsPerDevice[nDI] / (2 * d);
						_vectGPUKernels[nDI * _nKernels + 5]->GetKernel<RootCopyKernelScanDownsweep>()->Launch(
							szLocalCopy, szGlobalCopy, 0,
							_BufferMatrices[nSource[nDI]][nDI],
							_BufferMatrices[nDest[nDI]][nDI],
							d,
							nMSZ);
						// set parameters Vector copy downsweep
						// call the copy kernel for downsweep; queue 1 (copy the vector component)
						szLocalCopy[0] = _nLocalSizeMM;
						szGlobalCopy[0] = szLocalCopy[0] * _nStepsPerDevice[nDI] / (2 * d);
						_vectGPUKernels[nDI * _nKernels + 5]->GetKernel<RootCopyKernelScanDownsweep>()->Launch(
							szLocalCopy, szGlobalCopy, 1,
							_BufferVectors[nSource[nDI]][nDI],
							_BufferVectors[nDest[nDI]][nDI],
							d,
							_nMatrixSize);
					}
				// synchronize all threads
				for(int nDI = 0; nDI < _nStepsPerDevice.size(); nDI++)
					if(d  < _nStepsPerDevice[nDI]){
						// synchronize both queues for device nDI
						_pGPUM->GetDeviceAndContext(nDI)->GetQueue(0)->Synchronize();
						_pGPUM->GetDeviceAndContext(nDI)->GetQueue(1)->Synchronize();
						// change source with destination
						nSource[nDI] = nDest[nDI];
						nDest[nDI] = (nSource[nDI] + 1) % 2;
					}
			}
			// the result is in nSource[nDI] for each device
			// copy the result from the device memory to the result vector
			for(int nDI = 0; nDI < nDevices; nDI++)
				if(_nStepsPerDevice[nDI] > 0){
					fType* buffer;
					int bufferStride;
					if(_bZeroC == 1 && _bZeroD == 1){
						buffer = new fType[  _nStepsPerDevice[nDI] * _nMatrixSize];
						_BufferVectors[nSource[nDI]][nDI]->MemRead(	(void*)buffer, _BufferVectors[nSource[nDI]][nDI]->GetContext()->GetQueue(0),
																	0, _nStepsPerDevice[nDI] * _nMatrixSize * sizeof(fType));
						bufferStride = _nMatrixSize;
					}
					else{
						// compute the output Y
						// set parameters for SystemOutput Kernel
						vector<size_t> szLocalOutputKernel(2), szGlobalOutputKernel(2);
						szLocalOutputKernel[0] = _nLocalSizeMVVColumn;
						szLocalOutputKernel[1] = _nLocalSizeMVVRow;
						szGlobalOutputKernel[0] = _nStepsPerDevice[nDI] * _nLocalSizeMVVColumn; // this should actually be nElementsLocal - 1
						szGlobalOutputKernel[1] = szLocalOutputKernel[1];
						_vectGPUKernels[nDI * _nKernels + 8]->GetKernel<SystemOutput>()->Launch(
							szLocalOutputKernel, szGlobalOutputKernel, 0,
							_BufferC[nDI],
							_BufferD[nDI],
							_BufferVectors[nSource[nDI]][nDI],
							_vectUBuffers[nDI],
							_BufferYVectors[nDI],
							_nStateStride,
							_nStateOffset,
							_nInputStride,
							_nInputOffset,
							_nSystemSize,
							_nInputs,
							_nOutputSize,
							_bZeroC,
							_bZeroD);
						_pGPUM->GetDeviceAndContext(nDI)->GetQueue(0)->Synchronize();
						// copy the result from the GPU memory to the CPU memory
						buffer = new fType[  _nStepsPerDevice[nDI] * _nOutputSize];
						_BufferYVectors[nDI]->MemRead((void*)buffer, _BufferYVectors[nDI]->GetContext()->GetQueue(0),
														0, _nStepsPerDevice[nDI] * _nOutputSize * sizeof(fType));
						bufferStride = _nOutputSize;
					}
					for(int ii = 0; ii < _nStepsPerDevice[nDI]; ii++)
						if(globalResultTIndex + ii < xVect.size2())
							for(int jj = 0; jj < _nOutputSize; jj++)
								xVect(jj, globalResultTIndex +  ii) =
										buffer[(ii + 1)  * bufferStride - _nOutputSize + jj];
					// we assume that for state vectors that include more than one state
					// the current 'original' state is stored at the end of the vector
					delete [] buffer;
					globalResultTIndex += _nStepsPerDevice[nDI];
				}
		}
	}
	catch(exception& except){
		delete [] accMatrix;
		delete [] accVector;
		DeleteGPUFSIObjects();
		CleanSolverData();
		return ParODE_NotEnoughResources;
	}
	delete [] accMatrix;
	delete [] accVector;
	CleanSolverData();
	DeleteGPUFSIObjects();
	return ParODE_OK;
}
