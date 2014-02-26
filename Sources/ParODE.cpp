//Source File for the main library interface.
#include <fstream>
#include <ParODE.h>
#include <GPUManagement.h>
#include <ParODEException.h>
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
#include <AdamsMoulton.h>
#include <RungeKutta4thOrder.h>
#include <cstring>
#include <ParODECInterface.h>
#include <ParODECppInterface.h>
#include <string>

extern "C" {
	//Return a description of the error.
	void GetErrorDescription(int *ErrNo, char strErrDescription[]){
		string strErr;
		ErrorCode nErr = (ErrorCode)(*ErrNo);
		ParODE::Instance()->GetErrorDescription(nErr, strErr);
		// we assume that there is enough space in the buffer to receive the
		// error description
		strcpy(strErrDescription, strErr.c_str());
		return;
	}

	void InitializeAllGPUs( char* pszKernelFolder,
			 char* pszIncludeFolder,
			 int nErr[]){
		string strKernelFolder(pszKernelFolder);
		string strIncludeFolder(pszIncludeFolder);
		nErr[0] = ParODE::Instance()->InitializeGPU(strKernelFolder,
													strIncludeFolder);
	}

	void GetAvailableGPUs( int nGPUs[], int GPUIds[], int nErr[]){
		ErrorCode retVal;
		vector<unsigned int> vectIds;
		retVal =  ParODE::Instance()->GetAvailableGPUs(vectIds);
		if(retVal != ParODE_OK){
			nGPUs[0] = 0;
			nErr[0] = retVal;
			return;
		}
		nGPUs[0] = vectIds.size();
		for(int ii = 0; ii < nGPUs[0]; ii++)
			GPUIds[ii] = vectIds[ii];
		nErr[0] = ParODE_OK;
		return;
	}

	void GetDeviceName( int *DeviceId, char strDeviceName[], int nErr[]){
		ErrorCode retVal;
		string deviceName;
		unsigned int devIDCL = *DeviceId;
		retVal = ParODE::Instance()->GetDeviceName(devIDCL, deviceName);
		if(retVal != ParODE_OK){
			strDeviceName[0] = 0x0;
			nErr[0] = retVal;
			return;
		}
		strcpy(strDeviceName, deviceName.c_str());
		nErr[0] = ParODE_OK;
		return;
	}

	void InitializeSelectedGPUs( int *nGPUs, int GPUIds[],
								 char* pszKernelFolder,
								 char* pszIncludeFolder,
								 int nErr[]){
		vector<unsigned int> selectedDevices;
		string strKernelFolder(pszKernelFolder);
		string strIncludeFolder(pszIncludeFolder);

		for(int ii = 0; ii < *nGPUs; ii++)
			selectedDevices.push_back(GPUIds[ii]);

		ErrorCode retVal = ParODE::Instance()->InitializeSelectedGPUs(selectedDevices,
				strKernelFolder, strIncludeFolder);
		nErr[0] = retVal;
		return;
	}

	// interface for plain C calls
	// the caller should take care of memory allocation and release for A, B, C, D
	// the caller should take care of memory allocation and release for tVect and xVect
	void SimulateODE(	fType A[], // row major A
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
						int nSteps[],
						int nErr[]){
		matrixf Ab, Bb, Cb, Db;

		if(*nStates == 0){
			nErr[0] = ParODE_InvalidSystem;
			return;
		}
		Ab.resize(*nStates, *nStates);
		for(int row = 0; row < *nStates; row++)
			for(int col = 0; col < *nStates; col++)
				Ab(row, col) = A[row * (*nStates) + col];
		if(*ZeroB != 1){
			Bb.resize(*nStates, *nInputs);
			for(int row = 0; row < *nStates; row++)
				for(int col = 0; col < *nInputs; col++)
					Bb(row, col) = B[row * (*nInputs) + col];
		}
		if(*ZeroC != 1){
			Cb.resize(*nOutputs, *nStates);
			for(int row = 0; row < *nOutputs; row++)
				for(int col = 0; col < *nStates; col++)
					Cb(row, col) = C[row * (*nStates) + col];
		}
		if(*ZeroD != 1){
			Db.resize(*nOutputs, *nInputs);
			for(int row = 0; row < *nOutputs; row++)
				for(int col = 0; col < *nInputs; col++)
					Db(row, col) = D[row * (*nInputs)+ col];
		}

		vectorf x0b(*nStates);
		for(int ii = 0; ii < *nStates; ii++)
			x0b[ii] = x0[ii];

		ErrorCode retVal = ParODE::Instance()->SimulateODE(Ab, Bb, Cb, Db,
				  *uIndex,
				  *tStart, *tEnd, *tStep, x0b, (SolverType)(*solver),
				  tVect, xVect, nSteps[0]);
		nErr[0] = retVal;
	}

	void RegisterInput(char uFunc[], int uIndex[], int nErr[]){
		string strUFunc(uFunc);
		uIndex[0] = ParODE::Instance()->RegisterUKernel(uFunc);
		nErr[0] = uIndex[0] >= 0 ? 0 : uIndex[0];
	}

	//Release all GPU datastructures.
	void CloseGPUC(){
		ParODE::Delete();
	}

	// timing functions
	void StartTimer(int nErr[]){
		nErr[0] =  ParODE::Instance()->StartTimer();
		return;
	}

	void StopTimer(float fTime[], int nErr[]){
		nErr[0] = ParODE::Instance()->StopTimer(fTime[0]);
	}
}

// cpp interface
//Return a description of the error.
void GetErrorDescription(ErrorCode ErrNo, std::string& strErrDescription)
{
  ParODE::Instance()->GetErrorDescription(ErrNo, strErrDescription);
  return;
}

//Initializes the GPU computations; 
//- Identifies all GPUs in the system; Retrieve the parameters of each GPU;
// We use only the GPUs that have shared memory.
//- compiles the kernels for all the GPUs;
//- if everything is OK it returns ParODE_OK; otherwise an error code is returned;
ErrorCode InitializeAllGPUs(std::string& strKernelFolder,
							std::string& strIncludeFolder)
{
  return ParODE::Instance()->InitializeGPU(strKernelFolder, strIncludeFolder);
}

ErrorCode GetAvailableGPUs(std::vector<unsigned int>& GPUIds){
	return ParODE::Instance()->GetAvailableGPUs(GPUIds);
}

ErrorCode GetDeviceName( unsigned int DeviceId, std::string& strDeviceName){
	return ParODE::Instance()->GetDeviceName(DeviceId, strDeviceName);
}

ErrorCode InitializeSelectedGPUs( const std::vector<unsigned int>& GPUIds,
									std::string& strKernelFolder,
									std::string& strIncludeFolder){
	return ParODE::Instance()->InitializeSelectedGPUs(GPUIds, strKernelFolder, strIncludeFolder);
}

//Simulates the ODE.
//u func provides the input function. There should be some strict typing used in the definition of function u.

ErrorCode SimulateODE(	const matrixf & A, const matrixf & B,
										const matrixf & C, const matrixf & D,
										int uIndex,
										double tStart, double tEnd, double tStep,
										const vectorf& x0,
										SolverType solver,
										vectorf& tVect, matrixf& xVect)
{
  return ParODE::Instance()->SimulateODE(A, B, C, D,
		  uIndex,
		  tStart, tEnd, tStep, x0, solver, tVect, xVect);
}

int RegisterInput(const std::string& uFunc){
	return ParODE::Instance()->RegisterUKernel(uFunc);
}

//Release all GPU datastructures.
void CloseGPU()
{
  ParODE::Delete();
}

// Timing functions
ErrorCode StartTimer(){
	return ParODE::Instance()->StartTimer();
}

ErrorCode StopTimer(float &fTime){
	return ParODE::Instance()->StopTimer(fTime);
}

ParODE::ParODE() : _InitializationState(NOT_INITIALIZED), _pGPUManagement(NULL){
}

ParODE::~ParODE(){
	Clean();
}

// timing functions
ErrorCode ParODE::StartTimer(){
	if(_InitializationState != FULL_INITIALIZATION)
		return ParODE_NotInitialized;
	try{
		_pGPUManagement->StartTimer();
	}
	catch(exception &e){
		std::cerr << "Exception Caught : " << e.what() << std::endl;
		return ParODE_NotEnoughResources;
	}
	return ParODE_OK;
}

ErrorCode ParODE::StopTimer(float &fTime){
	if(_InitializationState != FULL_INITIALIZATION)
		return ParODE_NotInitialized;
	try{
		fTime = _pGPUManagement->StopTimer();
	}
	catch(exception &e){
		std::cerr << "Exception Caught : " << e.what() << std::endl;
		return ParODE_NotEnoughResources;
	}
	return ParODE_OK;
}

//Simulates the ODE.
//u func provides the input function. There should be some strict typing used in the definition of function u.

GPUODESolver* ParODE::CreateSolver(SolverType solver, int uIndex){
	GPUODESolver* pSolver;
	switch(solver){
	case RUNGE_KUTTA:
	{
		pSolver = new RungeKutta4thOrder(
				_pGPUManagement,
				_vectGPUKernels,
				_vectKernelsInputs[uIndex]);
		break;
	}
	case ADAMS_BASHFORTH_MOULTON:
	{
		pSolver = new AdamsMoulton(
				_pGPUManagement,
				_vectGPUKernels,
				_vectKernelsInputs[uIndex]);
		break;
	}
	default :
		return NULL;
	}
	return pSolver;
}

ErrorCode ParODE::SimulateODE(	const matrixf& A, const matrixf& B,
		    					const matrixf& C, const matrixf& D,
		    					int uIndex,
		    					double tStart, double tEnd, double tStep,
		    					const vectorf& x0,
		    					SolverType solver,
		    					fType tVect[], fType xVect[],
		    					int &nSteps){
	GPUODESolver* pSolver = CreateSolver(solver, uIndex);
	if(pSolver == NULL)
		return ParODE_InvalidSolverType;
	ErrorCode retVal =
			pSolver->SimulateODE(A, B, C, D, tStart, tEnd, tStep, x0, tVect, xVect, nSteps);
	delete pSolver;
	return retVal;
}

ErrorCode ParODE::SimulateODE(	const matrixf& A, const matrixf& B,
		    					const matrixf& C, const matrixf& D,
		    					int uIndex,
		    					double tStart, double tEnd, double tStep,
		    					const vectorf& x0,
		    					SolverType solver,
		    					vectorf& tVect, matrixf& xVect) {
	GPUODESolver* pSolver = CreateSolver(solver, uIndex);
	if(pSolver == NULL)
		return ParODE_InvalidSolverType;
	ErrorCode retVal =
			pSolver->SimulateODE(A, B, C, D, tStart, tEnd, tStep, x0, tVect, xVect);
	delete pSolver;
	return retVal;
}

//Initializes the GPU computations; 
//- Identifies all GPUs in the system; Retrieve the parameters of each GPU; We use only the GPUs that have shared memory.
//- compiles the kernels for all the GPUs;
//- if everything is OK it returns ParODE_OK; otherwise an error code is returned;
ErrorCode ParODE::InitializeGPU(string& strKernelsFolder, string& strIncludeFolder) {
	Clean();
	_strKernelFolderName = strKernelsFolder;
	_strIncludeFolder = strIncludeFolder;
	try{
		// create the GPU management object
		_pGPUManagement = new GPUManagement();
		bool bResult = _pGPUManagement->Initialize(2);
		if(bResult == false) return ParODE_NoGPU;
		CreateDefaultKernels();
	}
	catch(exception& e){
		Clean();
		std::cerr << "Exception Caught : " << e.what() << std::endl;
		return ParODE_CouldNotCreateGPUKernels;
	}
	_InitializationState = FULL_INITIALIZATION;
	return ParODE_OK;
}

void ParODE::CreateDefaultKernels(){
	int nDevices = _pGPUManagement->GetNumberOfDevices();
	for(int ii = 0; ii < nDevices; ii++){
		KernelWrapper* pKW;
		pKW = KernelWrapper::CreateOn<MMKernelScanUpsweep>(_pGPUManagement->GetDeviceAndContext(ii),
															_strKernelFolderName,
															_strIncludeFolder);
		_vectGPUKernels.push_back(pKW);
		pKW = KernelWrapper::CreateOn<MVVKernelScanUpsweep>(_pGPUManagement->GetDeviceAndContext(ii),
															_strKernelFolderName,
															_strIncludeFolder);
		_vectGPUKernels.push_back(pKW);
		pKW = KernelWrapper::CreateOn<LeftLeafCopyKernelScanUpsweep>(_pGPUManagement->GetDeviceAndContext(ii),
															_strKernelFolderName,
															_strIncludeFolder);
		_vectGPUKernels.push_back(pKW);
		pKW = KernelWrapper::CreateOn<MMKernelScanDownsweep>(_pGPUManagement->GetDeviceAndContext(ii),
															_strKernelFolderName,
															_strIncludeFolder);
		_vectGPUKernels.push_back(pKW);
		pKW = KernelWrapper::CreateOn<MVVKernelScanDownsweep>(_pGPUManagement->GetDeviceAndContext(ii),
															_strKernelFolderName,
															_strIncludeFolder);
		_vectGPUKernels.push_back(pKW);
		pKW = KernelWrapper::CreateOn<RootCopyKernelScanDownsweep>(_pGPUManagement->GetDeviceAndContext(ii),
															_strKernelFolderName,
															_strIncludeFolder);
		_vectGPUKernels.push_back(pKW);
		pKW = KernelWrapper::CreateOn<DistributeA>(_pGPUManagement->GetDeviceAndContext(ii),
															_strKernelFolderName,
															_strIncludeFolder);
		_vectGPUKernels.push_back(pKW);
		pKW = KernelWrapper::CreateOn<BTimesURK4thOrder>(_pGPUManagement->GetDeviceAndContext(ii),
															_strKernelFolderName,
															_strIncludeFolder);
		_vectGPUKernels.push_back(pKW);
		pKW = KernelWrapper::CreateOn<SystemOutput>(_pGPUManagement->GetDeviceAndContext(ii),
															_strKernelFolderName,
															_strIncludeFolder);
		_vectGPUKernels.push_back(pKW);
		pKW = KernelWrapper::CreateOn<BTimesUAdamsMoulton>(_pGPUManagement->GetDeviceAndContext(ii),
															_strKernelFolderName,
															_strIncludeFolder);
		_vectGPUKernels.push_back(pKW);
		}
}

int ParODE::RegisterUKernel(const string& strUProgram){
	if(_InitializationState != FULL_INITIALIZATION)
		return ParODE_NotInitialized;
	vector<OpenCLKernel*> vectUKernels; // one per device

	string uProgramName = _strKernelFolderName + string("u.cl");
	string GPUDefines;
	string InternalUProgramCode;
	char* str;
	size_t szstr;
	// read the program file
	// Open file stream
	std::fstream f(uProgramName.c_str(), (std::fstream::in | std::fstream::binary));
	// Check if we have opened file stream
	if(!f.is_open()) return ParODE_CouldNotCreateGPUKernels;
	size_t  sizeFile;
	// Find the stream size
	f.seekg(0, std::fstream::end);
	szstr = sizeFile = (size_t)f.tellg();
	f.seekg(0, std::fstream::beg);

	str = new char[szstr + 1];
	if (!str) {
		f.close();
		return ParODE_CouldNotCreateGPUKernels;
		}
	// Read file
	f.read(str, sizeFile);
	f.close();
	str[szstr] = '\0';
	InternalUProgramCode = string(str);
	delete [] str;
	GPUDefines = string("#include <FTypeDef.h>\n");
	string UProgramCode = GPUDefines + strUProgram + InternalUProgramCode;

	// create the kernels
	try{
		for(int ii = 0; ii < _pGPUManagement->GetNumberOfDevices(); ii++){
			OpenCLKernel* pKernel;
			string kernelName("u");
			_pGPUManagement->GetDeviceAndContext(ii)->CreateProgram(
					UProgramCode, kernelName, _strIncludeFolder, pKernel, false);
			vectUKernels.push_back(pKernel);
		}
	}
	catch(std::exception& e){
		std::cerr << "Exception caught : "<< e.what() << std::endl;
		for(int jj = 0; jj < vectUKernels.size(); jj++){
			delete vectUKernels[jj];
			vectUKernels[jj] = NULL;
		}
		vectUKernels.clear();
		return ParODE_CouldNotCreateGPUKernels;
	}
	_vectKernelsInputs.push_back(vectUKernels);
	return _vectKernelsInputs.size() - 1;
}

void ParODE::Clean(){
	if(_vectGPUKernels.size() > 0){
		for(int ii = 0; ii < _vectGPUKernels.size(); ii++){
			delete _vectGPUKernels[ii];
			_vectGPUKernels[ii] = NULL;
		}
		_vectGPUKernels.clear();
	}
	if(_vectKernelsInputs.size() != 0){
		vector< vector<OpenCLKernel*> >::iterator itInputKernelsVectors;
		itInputKernelsVectors = _vectKernelsInputs.begin();
		while(itInputKernelsVectors != _vectKernelsInputs.end()){
			vector< OpenCLKernel* >::iterator itInputKernels = (*itInputKernelsVectors).begin();
			while( itInputKernels != (*itInputKernelsVectors).end() ){
				delete (*itInputKernels);
				(*itInputKernelsVectors).erase(itInputKernels);
			}
			_vectKernelsInputs.erase(itInputKernelsVectors);
		}
	}
	if(_pGPUManagement != NULL){
		delete _pGPUManagement;
		_pGPUManagement = NULL;
	}
	_InitializationState = NOT_INITIALIZED;
}

//Return a description of the error.
void ParODE::GetErrorDescription(ErrorCode ErrNo, string& strErrDescription) {
	switch(ErrNo){
	case ParODE_OK:
		strErrDescription = string("No Error");
		break;
	case ParODE_NoGPU:
		strErrDescription = string("Sorry, no GPU available. Please make sure that you have installed a GPU with shared memory and you have installed the proper OpenCL drivers installed.");
		break;
	case ParODE_NotInitialized:
		strErrDescription = string("ParODE not initialized. Please call InitializeGPU.");
		break;
	case ParODE_CouldNotCreateGPUKernels:
		strErrDescription = string("Could not create GPU kernels; not enough resources.");
		break;
	default:
		strErrDescription = string("Unknown Error");
		break;
	}
}

ErrorCode ParODE::GetAvailableGPUs(vector<unsigned int>& vectIds){
	Clean();
	// create the GPU management object
	_pGPUManagement = new GPUManagement();
	_pGPUManagement->GetGPUDevices(vectIds);
	_InitializationState = PARTIAL_INITIALIZATION;
	return ParODE_OK;
}

ErrorCode ParODE::GetDeviceName(unsigned int device, string& deviceName){
	if(_InitializationState == NOT_INITIALIZED)
		return ParODE_NotInitialized;
	deviceName = _pGPUManagement->GetDeviceName( device );
	if( deviceName.empty() )
		return ParODE_InvalidDeviceId;
	return ParODE_OK;
}

ErrorCode ParODE::InitializeSelectedGPUs(const vector<unsigned int>& selectedDevices,
		string& strKernelsFolder, string& strIncludeFolder){
	if(_InitializationState != PARTIAL_INITIALIZATION)
		return ParODE_NotInitialized;
	try{
		_strKernelFolderName = strKernelsFolder;
		_strIncludeFolder = strIncludeFolder;
		// create the GPU management object
		bool bResult = _pGPUManagement->Initialize(selectedDevices, 2);
		if(bResult == false) return ParODE_NoGPU;
		CreateDefaultKernels();
	}
	catch(exception& e){
		Clean();
		std::cerr << "Exception Caught : " << e.what() << std::endl;
		return ParODE_CouldNotCreateGPUKernels;
	}
	_InitializationState = FULL_INITIALIZATION;
	return ParODE_OK;
}

ParODE* ParODE::_Instance = NULL;

//Implements the Singleton pattern
ParODE* ParODE::Instance()
{
  if(_Instance == NULL){
  	_Instance = new ParODE;
  }
  return _Instance;
}

void ParODE::Delete()
{
  if(_Instance != NULL){
  	delete _Instance;
  	_Instance = NULL;
  }
}

