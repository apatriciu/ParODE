
#include "RungeKutta4thOrder.h"
#include "GPUManagement.h"
#include <fstream>
#include <DebugUtilities.h>
#include <ParODEException.h>

void RungeKutta4thOrder::DeleteVector(std::vector<OpenCLMemBuffer*>& vect){
	for(int ii = 0; ii < vect.size(); ii++)
		delete vect[ii];
	vect.clear();
}

void RungeKutta4thOrder::InitializeSolverData(const matrixf& A, const matrixf& B,
											  double& tStart, double tEnd, double tStep, const vectorf& x0){

	_nMatrixSize = A.size1();
	_nInputs = B.size2();
	_nSystemSize = _nMatrixSize;
	_nSteps = (unsigned long)( (tEnd - tStart)/tStep) + 1;
	_nTotalStepsSimulation = _nSteps;
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
	// allocate and intialize host memory matrices ABar and BBar
	matrixf ABarBoost(A.size1(), A.size2());
	matrixf B1Boost(B.size1(), B.size2());
	matrixf B2Boost(B.size1(), B.size2());
	matrixf B3Boost(B.size1(), B.size2());
	matrixf Id(A.size1(), A.size1(), 0.0);
	matrixf A2, A3, AB, A2B, A3B;
	A2 = prod(A, A);
	A3 = prod(A2, A);
	AB = prod(A, B);
	A2B = prod(A2, B);
	A3B = prod(A3, B);
	for(int ii = 0; ii < Id.size1(); ii++) Id(ii, ii) = 1.0;
	B1Boost = (1.0/6.0) * (tStep * B + 
			  tStep * tStep * AB + 
			  0.5 * tStep * tStep * tStep * A2B + 
			  0.25 * tStep * tStep * tStep * tStep * A3B);
	B2Boost = (1.0/6.0) * (4.0 * tStep * B + 2.0 * tStep * tStep * AB + 0.5 * tStep * tStep * tStep * A2B);
	B3Boost = (1.0/6.0) * tStep * B;
	ABarBoost = Id + tStep * A + 
				0.5 * tStep * tStep * A2 + 
				(1.0 / 6.0) * tStep * tStep * tStep * A3 + 
				(1.0 / 24.0) * tStep * tStep * tStep * tStep * prod(A, A3);
	_ABarHost = new fType[A.size1() * A.size2()];
	_BBarHost = new fType[B.size1() * 3 * B.size2()];

	// copy ABar and BBar to linear host memory
	for(int row = 0; row < ABarBoost.size1(); row++){
		for(int col = 0; col < ABarBoost.size2(); col++)
			_ABarHost[row * ABarBoost.size1() + col] = ABarBoost(row, col);
		for(int col = 0; col < B1Boost.size2(); col++){
			_BBarHost[row * 3 * B1Boost.size2() + col] = B1Boost(row, col);
			_BBarHost[row * 3 * B1Boost.size2() + B1Boost.size2() + col] = B2Boost(row, col);
			_BBarHost[row * 3 * B1Boost.size2() + 2 * B1Boost.size2() + col] = B3Boost(row, col);
		}
	}

	int szA = A.size1();
	vectorf x0L(x0);
	SetInitialConditions(x0L);
	size_t tileSizeMM;
	size_t nLocalSizeMVVColumn;
	size_t nLocalSizeMVVRow;
	tileSizeMM = (szA < 8) ? 4 : ((szA < 16) ? 8 : 16); // we should have at least 256 maximum workgroup size

	size_t localMemoryMM = 2 * tileSizeMM * tileSizeMM * sizeof(fType);
	// allow this to take up to 2/3 of the local memory
	if(localMemoryMM > 0.66 * minLocalMemory) tileSizeMM >>= 1;
	
	// nLocalSizeMVVColumn should be a power of 2; it will be the smallest power of 2 that is greater than 3 * nInputs
	for(nLocalSizeMVVColumn = 1; nLocalSizeMVVColumn < 3 * _nInputs; nLocalSizeMVVColumn <<= 1);
	nLocalSizeMVVRow = min((minWorkgroupSize / nLocalSizeMVVColumn), (size_t)szA);
	size_t localMemoryMVV = nLocalSizeMVVColumn * nLocalSizeMVVRow * sizeof(fType);
	if(localMemoryMVV > 0.3 * minLocalMemory) nLocalSizeMVVRow >>= 1;
	SetLocalSizeAttributes( tileSizeMM, nLocalSizeMVVColumn, nLocalSizeMVVRow);
		
	vector<unsigned long> vectElementsPerDevice;
	// the number of elements per device is a function of the global memory available on the device
	// on the other hand we need to have a balanced load between the devices
	unsigned long GlobalMemoryPerStep(2 * (szA * szA + szA) * sizeof(fType));
	unsigned long GlobalMemoryMatrixPerStep(szA * szA * sizeof(fType));
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
	// allocate the buffers for holding matrices and vectors
	bool bRetValue;
	vector<OpenCLMemBuffer*> vBufferMatrices[2], vBufferVectors[2];
	try{
		for(int indexDevice = 0; indexDevice < nDevices; indexDevice++){
			size_t memSize;
			// create buffers for matrices and vectors
			for(int kk = 0; kk < 2; kk++){
				OpenCLMemBuffer *pBuffer;
				memSize = szA * szA * vectElementsPerDevice[indexDevice] *  sizeof(fType);
				_pGPUM->GetDeviceAndContext(indexDevice)->CreateMemBuffer(memSize, pBuffer);
				vBufferMatrices[kk].push_back(pBuffer);
				memSize = szA * vectElementsPerDevice[indexDevice] *  sizeof(fType);
				_pGPUM->GetDeviceAndContext(indexDevice)->CreateMemBuffer(memSize, pBuffer);
				vBufferVectors[kk].push_back(pBuffer);
			}
		}
	}
	catch(std::exception& except){
		DeleteVector(vBufferMatrices[0]);
		DeleteVector(vBufferMatrices[1]);
		DeleteVector(vBufferVectors[0]);
		DeleteVector(vBufferVectors[1]);
		std::cerr << "Exception caught : " << except.what() << std::endl;
		throw ParODEException();
	}
	// set the buffer objects
	SetBufferVector(vBufferVectors, vBufferMatrices);
	for(int ii = 0; ii < 2; ii++){
		vBufferVectors[ii].clear();
		vBufferMatrices[ii].clear();
	}
	try{
		// create the OpenCL buffers for ABar, BBar, and u
		for(int ii = 0; ii < nDevices; ii++){
			OpenCLMemBuffer* pBuffer;
			size_t	szBuffer;
			// create the buffer for ABar
			szBuffer = ABarBoost.size1() * ABarBoost.size2() * sizeof(fType);
			_pGPUM->GetDeviceAndContext(ii)->CreateMemBuffer(szBuffer, pBuffer);
			_vectABarBuffers.push_back(pBuffer);
			// create the buffer for BBar
			szBuffer = 3 * B1Boost.size1() * B1Boost.size2() * sizeof(fType);
			_pGPUM->GetDeviceAndContext(ii)->CreateMemBuffer(szBuffer, pBuffer);
			_vectBBarBuffers.push_back(pBuffer);
			// create the buffers for u
			// we need to keep also the half steps
			szBuffer = (2 * _nStepsPerDevice[ii] + 1) * B1Boost.size2() * sizeof(fType);
			_pGPUM->GetDeviceAndContext(ii)->CreateMemBuffer(szBuffer, pBuffer);
			_vectUBuffers.push_back(pBuffer);
		}
		// copy ABar and BBar to each device
		for(int ii = 0; ii < nDevices; ii++){
			// copy ABar
			_vectABarBuffers[ii]->MemWrite(_ABarHost, _vectABarBuffers[ii]->GetContext()->GetQueue(0));
			// copy BBar
			_vectBBarBuffers[ii]->MemWrite(_BBarHost, _vectBBarBuffers[ii]->GetContext()->GetQueue(1));
		}
	}
	catch(std::exception& except){
		std::cerr << "Exception caught " << except.what() << std::endl;
		DeleteGPUObjects();
		throw ParODEException();
	}
	SetStrides(2, 1);
	return;
}

void RungeKutta4thOrder::DeleteGPUObjects(){

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

void RungeKutta4thOrder::InitializeBatchData(
						 OpenCLMemBuffer* pBufferMatrixSource, 
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
	// compute u
	// the group size(0) is equal with the number of inputs in the system
	// the group size(1) should be adjusted such that we have more than 64 work items per group
	// the global size(0) is a multiple of local size(0) such that all elements are generated
	// the global size(1) is equal with local size(1)
	fType tStartInternal = tStart;
	fType tStepInternal = tStep / 2.0;
	int nElementsLocal = nSteps * 2 + 1;
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
	// synchronize queue 1
	_pGPUM->GetDeviceAndContext(nDeviceIndex)->GetQueue(1)->Synchronize();
	// multiply B with u vectors
	vector<size_t> szLocalBU(2), szGlobalBU(2);
	szLocalBU[0] = _nLocalSizeMVVColumn;
	szLocalBU[1] = _nLocalSizeMVVRow;
	szGlobalBU[0] = nSteps * _nLocalSizeMVVColumn;
	szGlobalBU[1] = szLocalBU[1];
	// launch the grid on queue 1
	_vectGPUKernels[nDeviceIndex * _nKernels + 7]->GetKernel<BTimesURK4thOrder>()->Launch(
			szLocalBU, szGlobalBU, 1,
			_vectBBarBuffers[nDeviceIndex],
			_vectUBuffers[nDeviceIndex],
			pBufferVectorSource,
			_nMatrixSize,
			_nInputs);
	_pGPUM->GetDeviceAndContext(nDeviceIndex)->GetQueue(1)->Synchronize();
	return;
}

void RungeKutta4thOrder::CleanSolverData(){
	DeleteHostObjects();
	DeleteGPUObjects();
	return;
}

void RungeKutta4thOrder::DeleteHostObjects(){
	if(_ABarHost != NULL){
		delete [] _ABarHost;
		_ABarHost = NULL;
	}
	if(_BBarHost != NULL){
		delete [] _BBarHost;
		_BBarHost = NULL;
	}
}

RungeKutta4thOrder::~RungeKutta4thOrder(){
	DeleteHostObjects();
	DeleteGPUObjects();
}

RungeKutta4thOrder::RungeKutta4thOrder(GPUManagement* GPUM,
		vector<KernelWrapper*>& vectGPUP,
		vector<OpenCLKernel*>& vectUKernel):
GPUODESolverFixedStepIterative(GPUM, vectGPUP, vectUKernel),
_ABarHost(NULL),
_BBarHost(NULL){
}
