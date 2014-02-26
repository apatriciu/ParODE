
#include <AdamsMoulton.h>
#include <GPUManagement.h>
#include <fstream>
#include <DebugUtilities.h>
#include <ParODEException.h>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/odeint.hpp>

vector<vectorf> AdamsMoulton::_UInit;
matrixf 		AdamsMoulton::_AInit;
matrixf 		AdamsMoulton::_BInit;
double 			AdamsMoulton::_DeltaTInit;

void AdamsMoulton::system_function(const state_type_init & x, state_type_init & dxdt,
									const double t){
	dxdt.resize(x.size(), 0.0);
	int tIndex = (int)(floor(t / _DeltaTInit + 0.5));
	for(int row = 0; row < x.size(); row++){
		fType fVal = 0.0;
		for(int col = 0; col < _AInit.size2(); col++)
			fVal += _AInit(row, col) * x[col];
		for(int col = 0; col < _BInit.size2(); col++)
			fVal += _BInit(row, col) * _UInit[tIndex][col];
		dxdt[row] = fVal;
	}
}

void AdamsMoulton::ComputeInitialState(const matrixf& A,
						 const matrixf& B,
						 const vectorf& x0,
						 double fDeltaT,
						 double tStart,
						 vectorf& x0Scan){
	// preconditions
	// B.size2() == _nInputs
	assert(B.size2() == _nInputs);

	_DeltaTInit = fDeltaT / 2.0;
	_AInit = A;
	_BInit = B;

	boost::numeric::odeint::runge_kutta4< state_type_init > rk4;

	state_type_init x(x0.size(), 0.0);

	x0Scan.resize(4 * x0.size());
	for(int row = 0; row < x0.size(); row++)
		x0Scan[row] = x[row] = x0[row];

	// fill the u vector
	// allocate a buffer for u
	int nElementsLocal = 4 * 2 + 1;
	size_t nFloats = B.size2() * nElementsLocal;

	OpenCLMemBuffer* pBuffer;
	try{
		_pGPUM->GetDeviceAndContext(0)->CreateMemBuffer(nFloats * sizeof(fType), pBuffer);
		// compute u
		// the group size(0) is equal with the number of inputs in the system
		// the group size(1) should be adjusted such that we have more than 64 work items per group
		// the global size(0) is a multiple of local size(0) such that all elements are generated
		// the global size(1) is equal with local size(1)
		fType tStartInternal = tStart;
		fType tStepInternal = fDeltaT / 2.0;
		_vectUKernels[0]->SetParameter(0, pBuffer);
		_vectUKernels[0]->SetParameter(1, sizeof(fType), (void*)&tStartInternal);
		_vectUKernels[0]->SetParameter(2, sizeof(fType), (void*)&tStepInternal);
		_vectUKernels[0]->SetParameter(3, sizeof(int), (void*)&nElementsLocal);
		// launch the threads
		vector<size_t> globalDimGenU(2), localDimGenU(2);
		localDimGenU[0] = _nInputs;
		for( localDimGenU[1] = 1; localDimGenU[1] < nElementsLocal && localDimGenU[1] * localDimGenU[0] < 32; localDimGenU[1]++);
		globalDimGenU[1] = localDimGenU[1];
		int nBlocks = (nElementsLocal - 1) / localDimGenU[1] + 1;
		globalDimGenU[0] = nBlocks * localDimGenU[0];
		_vectUKernels[0]->Execute(_vectUKernels[0]->GetContext()->GetQueue(0), globalDimGenU, localDimGenU);
		// synchronize queue 1
		_pGPUM->GetDeviceAndContext(0)->GetQueue(1)->Synchronize();
		// copy back the result
		fType* uVect = new fType[nFloats];
		pBuffer->MemRead(uVect, pBuffer->GetContext()->GetQueue(0));
		_UInit.resize(nElementsLocal);
		for(int ii = 0; ii < nElementsLocal; ii++){
			_UInit[ii].resize(B.size2());
			for(int row = 0; row < B.size2(); row++)
				_UInit[ii][row] = uVect[ii * B.size2() + row];
		}
		delete [] uVect;
		delete pBuffer;
	}
	catch(exception& e){
		std::cerr << "Exception caught : " << e.what();
		throw ParODEException();
	}
	// call the rk4 stepper
	double tDisc(0.0);
	for(int ii = 1; ii < 4; ii++){
		rk4.do_step(system_function, x, tDisc, fDeltaT);
		for(int row = 0; row < x.size(); row++)
			x0Scan[ii * x.size() + row] = x[row];
		tDisc += fDeltaT;
	}
}

void AdamsMoulton::GetInitialStatesAndInputs(matrixf& xVectInit,
											 matrixf& uVectInit){
	xVectInit.resize(_nSystemSize, 3);
	uVectInit.resize(_nInputs, 3);
	for(int ii = 0; ii < 3; ii++){
		for(int jj = 0; jj < _nSystemSize; jj++)
			xVectInit(jj, ii) = _X0[ii * _nSystemSize + jj];
		for(int jj = 0; jj < _nInputs; jj++)
			uVectInit(jj, ii) = _UInit[ii * 2][jj];
	}
}

void AdamsMoulton::ComputeNStepsPerDevice(){
	// find the device with the least local memory available
	unsigned long minLocalMemory(0);
	size_t	minWorkgroupSize(0);
	for(int ii = 0; ii < _pGPUM->GetNumberOfDevices(); ii++){
		if(minLocalMemory > _pGPUM->GetDeviceAndContext(ii)->LocalMemorySize() ||
			minLocalMemory == 0)
			minLocalMemory = _pGPUM->GetDeviceAndContext(ii)->LocalMemorySize();
		if( minWorkgroupSize < _pGPUM->GetDeviceAndContext(ii)->MaxWorkgroupSize() ||
			minWorkgroupSize == 0)
			minWorkgroupSize = _pGPUM->GetDeviceAndContext(ii)->MaxWorkgroupSize();
	}
	if(minLocalMemory == 0 || minWorkgroupSize == 0)
		throw ParODEException();

	// compute the number of time steps per device
	size_t tileSizeMM;
	size_t nLocalSizeMVVColumn;
	size_t nLocalSizeMVVRow;
	// set the tile size for M x M such that we fill one wavefront (as much as possible).
	// however if the matrices are smaller than 8 x 8 it will be impossible to fill one wavefront.
	// we may need to redesign the M x M kernel such that it can compute multiple matrix products.
	tileSizeMM = (_nMatrixSize < 8) ? 4 : ((_nMatrixSize < 16) ? 8 : 16);

	size_t localMemoryMM = 2 * tileSizeMM * tileSizeMM * sizeof(fType);
	// allow this to take up to 2/3 of the local memory
	if(localMemoryMM > 0.66 * minLocalMemory) tileSizeMM >>= 1;

	// nLocalSizeMVVColumn should be a power of 2; it will be the smallest power of 2 that is greater than 3 * nInputs
	for(nLocalSizeMVVColumn = 1; nLocalSizeMVVColumn < _nMatrixSize; nLocalSizeMVVColumn <<= 1);
	nLocalSizeMVVRow = min((minWorkgroupSize / nLocalSizeMVVColumn), (size_t)_nMatrixSize);
	size_t localMemoryMVV = nLocalSizeMVVColumn * nLocalSizeMVVRow * sizeof(fType);
	if(localMemoryMVV > 0.3 * minLocalMemory) nLocalSizeMVVRow >>= 1;
	SetLocalSizeAttributes( tileSizeMM, nLocalSizeMVVColumn, nLocalSizeMVVRow);

	vector<unsigned long> vectElementsPerDevice;
	// the number of elements per device is a function of the global memory available on the device
	// on the other hand we need to have a balanced load between the devices
	unsigned long GlobalMemoryPerStep(2 * (_nMatrixSize * _nMatrixSize + _nMatrixSize) * sizeof(fType));
	unsigned long GlobalMemoryMatrixPerStep(_nMatrixSize * _nMatrixSize * sizeof(fType));
	int nDevices = _pGPUM->GetNumberOfDevices();
	vector<unsigned long> maxElementsPerDevice(nDevices);
	vector<unsigned int> ComputingUnitsPerDevice(nDevices);
	for(int ii = 0; ii < nDevices; ii++){
		unsigned long globalMemoryDevice = (unsigned long)(0.8 * _pGPUM->GetDeviceAndContext(ii)->GlobalMemorySize()),
					  maxAllocationUnit = (unsigned long)(0.8 * _pGPUM->GetDeviceAndContext(ii)->MaximumMemoryAllocationSize());
		maxElementsPerDevice[ii] = min( (unsigned long)(globalMemoryDevice / GlobalMemoryPerStep),
										(unsigned long)(maxAllocationUnit / GlobalMemoryMatrixPerStep));
		ComputingUnitsPerDevice[ii] = _pGPUM->GetDeviceAndContext(ii)->ComputeUnits();
	}
	// compute the maximum number of ComputeUnits
	unsigned int maxComputeUnitsPerDevice = 0;
	unsigned int indexMaxCUDevice = 0;
	for(int ii = 0; ii < nDevices; ii++)
		if(maxComputeUnitsPerDevice < ComputingUnitsPerDevice[ii]){
			maxComputeUnitsPerDevice = ComputingUnitsPerDevice[ii];
			indexMaxCUDevice = ii;
		}
	// compute the elements / computeunits ratio for the most powerfull device
	double ratioECU = maxElementsPerDevice[indexMaxCUDevice] / maxComputeUnitsPerDevice;
	// adjust the maximum loads such that
	// - we have a power of 2 elements per device
	// - the load is balanced the ratio elements to compute units for each device is approximately equal with ratioECU
	for(int ii = 0; ii < nDevices; ii++){
		int nElementsPerDevice = 1;
		while(	nElementsPerDevice < maxElementsPerDevice[ii] &&
				(double)nElementsPerDevice / (double)ComputingUnitsPerDevice[ii] < ratioECU)
				nElementsPerDevice <<= 1;
		vectElementsPerDevice.push_back(nElementsPerDevice >> 1);
	}
	// if the sum of all elements per device is larger than the simulation steps then adjust the number of max elements per device
	// such that the sum of all elements per device is approximately equal with the simulation steps
	unsigned long sumElements = 0;
	for(int ii = 0; ii < vectElementsPerDevice.size(); ii++)
		sumElements += vectElementsPerDevice[ii];
	if(sumElements > _nSteps){
		while(sumElements >= _nSteps){
			sumElements = 0;
			for(int ii = 0; ii < vectElementsPerDevice.size(); ii++){
				vectElementsPerDevice[ii] >>= 1;
				sumElements += vectElementsPerDevice[ii];
			}
		}
		// go back one step such that we accomodate all steps in one batch
		for(int ii = 0; ii < vectElementsPerDevice.size(); ii++)
			vectElementsPerDevice[ii] <<= 1;
	}

	SetNStepsPerDevice( vectElementsPerDevice);
}

void AdamsMoulton::CreateInputBuffers(){
	try{
		for(int indexDevice = 0; indexDevice < _pGPUM->GetNumberOfDevices(); indexDevice++){
			size_t szBufferU = 	(_nStepsPerDevice[indexDevice] + 4)*
								_nInputs * sizeof(fType);
			OpenCLMemBuffer* pMemBuffer;
			_pGPUM->GetDeviceAndContext(indexDevice)->CreateMemBuffer(szBufferU, pMemBuffer);
			_vectUBuffers.push_back(pMemBuffer);
		}
	}
	catch(exception& e){
		std::cerr << "Exception caught : " << e.what() << std::endl;
		DeleteGPUObjects();
		DeleteHostObjects();
		for(int ii = 0; ii < _vectUBuffers.size(); ii++){
			delete _vectUBuffers[ii];
			_vectUBuffers[ii] = NULL;
		}
		_vectUBuffers.clear();
		throw ParODEException();
	}
}

void AdamsMoulton::CleanSolverData(){
	DeleteHostObjects();
	DeleteGPUObjects();
	return;
}

void AdamsMoulton::AllocateMatrixBuffers(const matrixf& A, const matrixf& B, double tStep){
	// build ABar;
	matrixf A_0(A.size1(), A.size2()),
			A_1(A.size1(), A.size2()),
			A_2(A.size1(), A.size2()),
			A_3(A.size1(), A.size2());
	matrixf B__1(B.size1(), B.size2()),
			B_0(B.size1(), B.size2()),
			B_1(B.size1(), B.size2()),
			B_2(B.size1(), B.size2()),
			B_3(B.size1(), B.size2());

	matrixf A2(A.size1(), A.size2());
	matrixf AB(B.size1(), B.size2());

	// compute matrix products
	A2 = prod(A, A);
	AB = prod(A, B);

	// compute submatrices
	A_0 = boost::numeric::ublas::identity_matrix<fType>(A.size1()) +
			(28.0 * tStep / 24.0) * A +
			(495.0 * tStep * tStep / 576.0) * A2;
	A_1 = (-5.0 * tStep / 24.0) * A + (-531.0 * tStep * tStep / 576.0) * A2;
	A_2 = (1.0 * tStep / 24.0) * A + (333.0 * tStep * tStep / 576.0) * A2;
	A_3 = (-81.0 * tStep * tStep / 576.0) * A2;

	B__1 = (9.0 * tStep / 24.0) * B;
	B_0 = (19.0 * tStep / 24.0) * B + (495.0 * tStep * tStep / 576.0) * AB;
	B_1 = (-5.0 * tStep / 24.0) * B + (-531.0 * tStep * tStep / 576.0) * AB;
	B_2 = (tStep / 24.0) * B + (333.0 * tStep * tStep / 576.0) * AB;
	B_3 = (-81.0 * tStep * tStep / 576.0) * AB;

	try{
		size_t nElementsABar = _nMatrixSize * _nMatrixSize;
		size_t nElementsBBar = _nSystemSize * 5 * _nInputs * sizeof(fType);
		// allocate host ABar
		_ABarHost = new fType[nElementsABar];
		memset(_ABarHost, 0, nElementsABar * sizeof(fType));
		// allocate host BBar
		_BBarHost = new fType[nElementsBBar];
		memset(_BBarHost, 0, nElementsBBar * sizeof(fType));
		// initialize ABar
		/*
		ABar = [0, I, 0, 0;
				0, 0, I, 0;
				0, 0, 0, I;
				A_3, A_2, A_1, A_0]
		*/
		for(int row = 0; row < _nSystemSize; row++){
			for(int col = 0; col < _nSystemSize; col++){
				_ABarHost[3 * _nSystemSize * _nMatrixSize + row * _nMatrixSize +
				          col] = A_3(row, col);
				_ABarHost[3 * _nSystemSize * _nMatrixSize + row * _nMatrixSize +
				          _nSystemSize + col] = A_2(row, col);
				_ABarHost[3 * _nSystemSize * _nMatrixSize + row * _nMatrixSize +
				          2 * _nSystemSize + col] = A_1(row, col);
				_ABarHost[3 * _nSystemSize * _nMatrixSize + row * _nMatrixSize +
				          3 * _nSystemSize + col] = A_0(row, col);
			}
			_ABarHost[row * _nMatrixSize +
			          _nSystemSize + row] = 1.0;
			_ABarHost[_nSystemSize * _nMatrixSize + row * _nMatrixSize +
			          2 * _nSystemSize + row] = 1.0;
			_ABarHost[2 * _nSystemSize * _nMatrixSize + row * _nMatrixSize +
			          3 * _nSystemSize + row] = 1.0;
		}
		// initialize BBar
		/*
		 * BBar = [B_3, B_2, B_1, B_0, B__1];
		 */
		for(int row = 0; row < _nSystemSize; row++)
			for(int col = 0; col < _nInputs; col++){
				_BBarHost[row * 5 * _nInputs + col] = B_3(row, col);
				_BBarHost[row * 5 * _nInputs + _nInputs + col] = B_2(row, col);
				_BBarHost[row * 5 * _nInputs + 2 * _nInputs + col] = B_1(row, col);
				_BBarHost[row * 5 * _nInputs + 3 * _nInputs + col] = B_0(row, col);
				_BBarHost[row * 5 * _nInputs + 4 * _nInputs + col] = B__1(row, col);
			}
		// allocate GPU buffers and copy from host to GPU
		for(int ii = 0; ii < _pGPUM->GetNumberOfDevices(); ii++){
			// allocate ABar buffer
			size_t memSize;
			OpenCLMemBuffer* pBuffer;
			memSize = nElementsABar * sizeof(fType);
			_pGPUM->GetDeviceAndContext(ii)->CreateMemBuffer(memSize, pBuffer);
			pBuffer->MemWrite(_ABarHost, pBuffer->GetContext()->GetQueue(0));
			_vectABarBuffers.push_back(pBuffer);
			// allocate BBar buffer
			memSize = nElementsBBar * sizeof(fType);
			_pGPUM->GetDeviceAndContext(ii)->CreateMemBuffer(memSize, pBuffer);
			pBuffer->MemWrite(_BBarHost, pBuffer->GetContext()->GetQueue(1));
			_vectBBarBuffers.push_back(pBuffer);
		}
	}
	catch(exception& e){
		std::cerr << "Exception caught : " << e.what() << std::endl;
		DeleteGPUObjects();
		DeleteHostObjects();
		throw ParODEException();
	}
}

void AdamsMoulton::InitializeSolverData(const matrixf& A, const matrixf& B,
		double& tStart, double tEnd, double tStep, const vectorf& x0){
	// sanity size checks
	if(A.size1() != A.size2() || A.size1() != B.size1() || A.size1() != x0.size())
		throw ParODEException();
	// initialize dimensions
	_nMatrixSize = 4 * A.size1();
	_nInputs = B.size2();
	_nSystemSize = A.size1();
	_nInputStride = 1;
	// set the initial conditions
	vectorf x0Scan;
	ComputeInitialState(A, B, x0, tStep, tStart, x0Scan);
	SetInitialConditions(x0Scan);
	// adjust tStart to skip the initialization steps
	tStart += 3 * tStep; // this will be the new t_0
	_nSteps = (unsigned long)( (tEnd - tStart)/tStep) + 1;
	_nTotalStepsSimulation = _nSteps + 3ul;
	ComputeNStepsPerDevice();
	AllocateMatrixBuffers(A, B, tStep);
	// allocate the buffers for holding matrices and vectors
	vector<OpenCLMemBuffer*> vBufferMatrices[2], vBufferVectors[2];
	try{
		for(int indexDevice = 0; indexDevice < _pGPUM->GetNumberOfDevices(); indexDevice++){
			size_t memSize;
			// create buffers for matrices and vectors
			for(int kk = 0; kk < 2; kk++){
				OpenCLMemBuffer *pBuffer;
				memSize = _nMatrixSize * _nMatrixSize * _nStepsPerDevice[indexDevice] *  sizeof(fType);
				_pGPUM->GetDeviceAndContext(indexDevice)->CreateMemBuffer(memSize, pBuffer);
				vBufferMatrices[kk].push_back(pBuffer);
				memSize = _nMatrixSize * _nStepsPerDevice[indexDevice] *  sizeof(fType);
				_pGPUM->GetDeviceAndContext(indexDevice)->CreateMemBuffer(memSize, pBuffer);
				vBufferVectors[kk].push_back(pBuffer);
			}
		}
	}
	catch(std::exception& except){
		for(int ii = 0; ii < 2; ii++){
			DeleteVector(vBufferMatrices[ii]);
			DeleteVector(vBufferVectors[ii]);
		}
		DeleteHostObjects();
		DeleteGPUObjects();
		std::cerr << "Exception caught : " << except.what() << std::endl;
		throw ParODEException();
	}
	// set the buffer objects
	SetBufferVector(vBufferVectors, vBufferMatrices);
	for(int ii = 0; ii < 2; ii++){
		vBufferVectors[ii].clear();
		vBufferMatrices[ii].clear();
	}

	// allocate the buffers for inputs
	CreateInputBuffers();

	SetStrides(1, 4, 3, 3);

	return;
}

void AdamsMoulton::InitializeBatchData(OpenCLMemBuffer* pBufferMatrixSource,
								 OpenCLMemBuffer* pBufferVectorSource,
								 int nDeviceIndex,
								 double tStart, double tStep, int nSteps){
	// distribute ABar to pBufferMatrixSource
	// set the parameters
	int nElements = _nMatrixSize * _nMatrixSize;
	vector<size_t> global_size(1), local_size(1);
	local_size[0] = _nLocalSizeMM;
	global_size[0] = local_size[0] * nSteps;
	_vectGPUKernels[nDeviceIndex * _nKernels + 6]->GetKernel<DistributeA>()->Launch(
			local_size, global_size, 0,
			_vectABarBuffers[nDeviceIndex],
			pBufferMatrixSource,
			nElements);
	// clear the content of pBufferVectorSource
	pBufferVectorSource->MemFill(0x0, pBufferVectorSource->GetContext()->GetQueue(0));

	// compute u
	// the group size(0) is equal with the number of inputs in the system
	// the group size(1) should be adjusted such that we have more than 64 work items per group
	// the global size(0) is a multiple of local size(0) such that all elements are generated
	// the global size(1) is equal with local size(1)
	fType tStartInternal = tStart - 3 * tStep; // we have to generate t_{-3}, t_{-2}, t_{-1}, t_0, etc..
	fType tStepInternal = tStep;
	int nElementsLocal = nSteps + 4;
	_vectUKernels[nDeviceIndex]->SetParameter(0, _vectUBuffers[nDeviceIndex]);
	_vectUKernels[nDeviceIndex]->SetParameter(1, sizeof(fType), (void*)&tStartInternal);
	_vectUKernels[nDeviceIndex]->SetParameter(2, sizeof(fType), (void*)&tStepInternal);
	_vectUKernels[nDeviceIndex]->SetParameter(3, sizeof(int), (void*)&nElementsLocal);
	// launch the threads
	vector<size_t> globalDimGenU(2), localDimGenU(2);
	localDimGenU[0] = _nInputs;
	for( localDimGenU[1] = 1; localDimGenU[1] < nElementsLocal && localDimGenU[1] * localDimGenU[0] < 32; localDimGenU[1]++);
	globalDimGenU[1] = localDimGenU[1];
	int nBlocks = (nElementsLocal - 1) / localDimGenU[1] + 1;
	globalDimGenU[0] = nBlocks * localDimGenU[0];
	_vectUKernels[nDeviceIndex]->Execute(_vectUKernels[nDeviceIndex]->GetContext()->GetQueue(1), globalDimGenU, localDimGenU);
	// synchronize queues
	_pGPUM->GetDeviceAndContext(nDeviceIndex)->GetQueue(1)->Synchronize();
	_pGPUM->GetDeviceAndContext(nDeviceIndex)->GetQueue(0)->Synchronize();
	// multiply B with u vectors
	vector<size_t> szLocalBU(2), szGlobalBU(2);
	szLocalBU[0] = _nLocalSizeMVVColumn;
	szLocalBU[1] = _nLocalSizeMVVRow;
	szGlobalBU[0] = nSteps * _nLocalSizeMVVColumn; // this should actually be nElementsLocal - 1 ???
	szGlobalBU[1] = szLocalBU[1];
	// launch the grid on queue 1
	_vectGPUKernels[nDeviceIndex * _nKernels + 9]->GetKernel<BTimesUAdamsMoulton>()->Launch(
			szLocalBU, szGlobalBU, 1,
			_vectBBarBuffers[nDeviceIndex],
			_vectUBuffers[nDeviceIndex],
			pBufferVectorSource,
			_nSystemSize,
			5 * _nInputs,
			_nInputs);
	_pGPUM->GetDeviceAndContext(nDeviceIndex)->GetQueue(1)->Synchronize();
	return;
}

void AdamsMoulton::DeleteHostObjects(){
	if(_ABarHost != NULL){
		delete [] _ABarHost;
		_ABarHost = NULL;
	}
	if(_BBarHost != NULL){
		delete [] _BBarHost;
		_BBarHost = NULL;
	}
}

void AdamsMoulton::DeleteGPUObjects(){
	for(int ii = 0; ii < _vectUBuffers.size(); ii++){
		delete _vectUBuffers[ii];
		_vectUBuffers[ii] = NULL;
	}
	_vectUBuffers.clear();
	for(int ii = 0; ii < _vectABarBuffers.size(); ii++){
		delete _vectABarBuffers[ii];
		_vectABarBuffers[ii] = NULL;
	}
	_vectABarBuffers.clear();
	for(int ii = 0; ii < _vectBBarBuffers.size(); ii++){
		delete _vectBBarBuffers[ii];
		_vectBBarBuffers[ii] = NULL;
	}
	_vectBBarBuffers.clear();
}

void AdamsMoulton::DeleteVector(std::vector<OpenCLMemBuffer*>& vect){
	for(int ii = 0; ii < vect.size(); ii++){
		delete vect[ii];
		vect[ii] = NULL;
	}
	vect.clear();
}

AdamsMoulton::~AdamsMoulton(){
}

AdamsMoulton::AdamsMoulton(	GPUManagement* GPUM,
								vector<KernelWrapper*>& vectGPUP,
								vector<OpenCLKernel*>& vectU):
GPUODESolverFixedStepIterative(GPUM, vectGPUP, vectU),
_ABarHost(NULL),
_BBarHost(NULL){
}
