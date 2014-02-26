#include "GPUManagement.h"
#include <assert.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <stdexcept>

const GPUManagement::ErrorStructureType GPUManagement::_vectErrors[] = { // the list should end with a CL_SUCCESS
		{CL_INVALID_VALUE, "Invalid Value in OpenCL function call."},
		{CL_OUT_OF_HOST_MEMORY, "Can not allocate host resources; Out of memory."},
		{CL_INVALID_PLATFORM, "Platform not valid."},
		{CL_INVALID_DEVICE_TYPE, "Invalid device_type."},
		{CL_DEVICE_NOT_FOUND, "No OpenCL devices that matched device_type were found."},
		{CL_INVALID_DEVICE, "Device not valid."},
		{CL_INVALID_CONTEXT, "Invalid context."},
		{CL_INVALID_QUEUE_PROPERTIES, "Queue properties are not supported by the device."},
		{CL_INVALID_PROGRAM_EXECUTABLE, "There is no successfully built program executable available for device associated with command_queue."},
		{CL_INVALID_COMMAND_QUEUE, "Not a valid command-queue."},
		{CL_INVALID_KERNEL, "Invalid kernel object."},
		{CL_INVALID_CONTEXT, "Context associated with command_queue and kernel are not the same."},
		{CL_INVALID_KERNEL_ARGS, "Kernel argument values have not been specified."},
		{CL_INVALID_WORK_DIMENSION, "Invalid work_dim."},
		{CL_INVALID_GLOBAL_WORK_SIZE, "Invalid global work size."},
		{CL_INVALID_GLOBAL_OFFSET, "Invalid global offset."},
		{CL_INVALID_WORK_GROUP_SIZE, "Invalid workgroup size."},
		{CL_INVALID_WORK_ITEM_SIZE, "Invalid workitem size."},
		{CL_MISALIGNED_SUB_BUFFER_OFFSET, "Misaligned buffer object parameter."},
		{CL_INVALID_IMAGE_SIZE, "Invalid image  size."},
		{CL_IMAGE_FORMAT_NOT_SUPPORTED, "Image format not supported."},
		{CL_OUT_OF_RESOURCES, "Not enough device resources."},
		{CL_MEM_OBJECT_ALLOCATION_FAILURE, "Could not allocate memory objects."},
		{CL_INVALID_EVENT_WAIT_LIST, "Invalid event wait list."},
		{CL_INVALID_ARG_INDEX, "Invalid argument index."},
		{CL_INVALID_ARG_VALUE, "Invalid argument value."},
		{CL_INVALID_MEM_OBJECT, "Invalid memory object."},
		{CL_INVALID_SAMPLER, "Invalid sampler object."},
		{CL_INVALID_ARG_SIZE, "Invalid argument size"},
		{CL_INVALID_ARG_VALUE, "Invalid argument value"},
		{CL_SUCCESS, "Other Error"}};

void GPUManagement::GetGPUDevices(
		vector<unsigned int>& deviceIDs){
	// look for a platform that supports OpenCL
	// look for 4 platforms
	cl_platform_id platform_id_vector[MAXIMUM_NUMBER_OF_PLATFORMS];
	cl_uint nPlatforms;
	CheckError( clGetPlatformIDs(MAXIMUM_NUMBER_OF_PLATFORMS,
								platform_id_vector,
								&nPlatforms) );
	// loop through platforms and select the ones with GPU
	int nGPUPlatforms(0);

	char szPlatformParamValue[256];
	size_t ParamValueLength;

	cl_device_id vectDeviceID[MAX_DEVICES_PER_PLATFORM];
	cl_uint nDevices;

	for(unsigned int ii = 0; ii < nPlatforms; ii++){
		// print the platform name and OpenCL version
		CheckError( clGetPlatformInfo (platform_id_vector[ii],
					CL_PLATFORM_NAME,
					256,
					szPlatformParamValue,
					&ParamValueLength) );
		std::cout << "Platform #" << ii << " : " << szPlatformParamValue << std::endl;
		CheckError( clGetPlatformInfo (platform_id_vector[ii],
					CL_PLATFORM_VERSION,
					256,
					szPlatformParamValue,
					&ParamValueLength) );
		std::cout << "\tSupporting : " << szPlatformParamValue << std::endl;
		// check the profile
		CheckError( clGetPlatformInfo (platform_id_vector[ii],
					CL_PLATFORM_PROFILE,
					256,
					szPlatformParamValue,
					&ParamValueLength) );
		std::cout << "\tProfile : " << szPlatformParamValue << std::endl;
		if( strcmp(szPlatformParamValue, "FULL_PROFILE") ) continue;
		// query devices; select only GPU devices
		CheckError( clGetDeviceIDs (platform_id_vector[ii],
									CL_DEVICE_TYPE_GPU,
									MAX_DEVICES_PER_PLATFORM,
									vectDeviceID,
									&nDevices) );
		for(int jj = 0; jj < nDevices; jj++){
			OpenCLDeviceAndContext* newDevice;
			newDevice = new OpenCLDeviceAndContext(platform_id_vector[ii], vectDeviceID[jj]);
			if(newDevice->DeviceAvailable()){
				vectDevices.push_back(newDevice);
				deviceIDs.push_back(vectDevices.size() - 1);
			}
		}
	}
	return;
}

string GPUManagement::GetDeviceName(unsigned int devID){
	if(devID < vectDevices.size())
		return string( vectDevices[devID]->DeviceName() );
	return string();
}

//Initialize from a specific list of platforms and devices
// the vector with devices and queues
bool GPUManagement::Initialize(
			const vector<unsigned int>& deviceIDs,
			int nQueuesPerDevice){

	vector<cl_device_id> vectSelectedDevices;
	for(int ii = 0; ii < deviceIDs.size(); ii++)
		vectSelectedDevices.push_back( vectDevices[ deviceIDs[ii] ]->DeviceId() );

	vector<OpenCLDeviceAndContext*>::iterator itDevices;

	for(itDevices = vectDevices.begin(); itDevices != vectDevices.end();){
		vector<cl_device_id>::iterator itReqIds;
		for(itReqIds = vectSelectedDevices.begin();
				itReqIds != vectSelectedDevices.end(); itReqIds++)
			if(*itReqIds == (*itDevices)->DeviceId())
				break;
		if( itReqIds != vectSelectedDevices.end() ){
			// create the number of queues requested
			if( !(*itDevices)->CreateQueues(nQueuesPerDevice))
				return false;
			itDevices++;
		}
		else
			vectDevices.erase(itDevices);
	}
	return true;
}

//Initialize the vector with devices and queues
// use all available GPUs with local memory
bool GPUManagement::Initialize(int nQueuesPerDevice){
	// look for a platform that supports OpenCL
	// look for 4 platforms
	cl_platform_id platform_id_vector[MAXIMUM_NUMBER_OF_PLATFORMS];
	cl_uint nPlatforms;
	CheckError( clGetPlatformIDs(MAXIMUM_NUMBER_OF_PLATFORMS,
								platform_id_vector,
								&nPlatforms) );
	// loop through platforms and select the ones with GPU
	int nGPUPlatforms(0);

	char szPlatformParamValue[256];
	size_t ParamValueLength;

	cl_device_id vectDeviceID[MAX_DEVICES_PER_PLATFORM];
	cl_uint nDevices;

	for(unsigned int ii = 0; ii < nPlatforms; ii++){
		// print the platform name and OpenCL version
		CheckError( clGetPlatformInfo (platform_id_vector[ii],
					CL_PLATFORM_NAME,
					256,
					szPlatformParamValue,
					&ParamValueLength) );
		std::cout << "Platform #" << ii << " : " << szPlatformParamValue << std::endl;
		CheckError( clGetPlatformInfo (platform_id_vector[ii],
					CL_PLATFORM_VERSION,
					256,
					szPlatformParamValue,
					&ParamValueLength) );
		std::cout << "\tSupporting : " << szPlatformParamValue << std::endl;
		// check the profile
		CheckError( clGetPlatformInfo (platform_id_vector[ii],
					CL_PLATFORM_PROFILE,
					256,
					szPlatformParamValue,
					&ParamValueLength) );
		std::cout << "\tProfile : " << szPlatformParamValue << std::endl;
		if( strcmp(szPlatformParamValue, "FULL_PROFILE") ) continue;
		// query devices; select only GPU devices
		CheckError( clGetDeviceIDs (platform_id_vector[ii],
									CL_DEVICE_TYPE_GPU,
									MAX_DEVICES_PER_PLATFORM,
									vectDeviceID,
									&nDevices) );
		for(int jj = 0; jj < nDevices; jj++){
			// print the GPU devices
			OpenCLDeviceAndContext* newDevice;
			newDevice = new OpenCLDeviceAndContext(platform_id_vector[ii], vectDeviceID[jj]);
			if(newDevice->DeviceAvailable()){
				if(newDevice->CreateQueues(nQueuesPerDevice))
					vectDevices.push_back(newDevice);
			}
		}
	}
	return (vectDevices.size() > 0);
}

int GPUManagement::GetNumberOfDevices() {
  return vectDevices.size();
}

GPUManagement::GPUManagement() {
}

GPUManagement::~GPUManagement() {
	// clean everithing
	for(int ii = 0; ii < vectDevices.size(); ii++){
		delete vectDevices[ii];
		vectDevices[ii] = NULL;
	}
	vectDevices.clear();
}

OpenCLDeviceAndContext* GPUManagement::GetDeviceAndContext(int nIndex) {
  assert(nIndex >= 0 && nIndex < vectDevices.size());
  return( vectDevices[nIndex] );
}

cl_ulong GPUManagement::GetGPUTime(bool bStart){
	if(vectDevices.size() == 0) throw GPUException();
	if(vectDevices[0]->GetNoOfQueues() == 0) throw GPUException();
	size_t timing_memory_size(8 * sizeof(int));
	// create timing memory
	cl_int errcode_ret;
	cl_mem timing_memory = clCreateBuffer ( vectDevices[0]->ContextId(),
			CL_MEM_READ_WRITE, timing_memory_size, 0, &errcode_ret);
	GPUManagement::CheckError(errcode_ret);
	if(errcode_ret != CL_SUCCESS) throw GPUException();
	// create the timing event
	cl_event TimingEvent;
	// launch a dummy fill
	int hostBuffer[8];
	cl_int errCode = clEnqueueWriteBuffer( vectDevices[0]->GetQueue(0)->GetQueueId(),
			timing_memory,
			CL_TRUE,
			0,
			timing_memory_size,
			hostBuffer,
			0,
			NULL,
			&TimingEvent);
	GPUManagement::CheckError( errCode );
	if(errCode != CL_SUCCESS)
		throw GPUException();
	// make sure that the fill has finished
	errcode_ret = clWaitForEvents (1, &TimingEvent);
	GPUManagement::CheckError( errCode );
	if(errCode != CL_SUCCESS) throw GPUException();
	// retrieve the timing data
	cl_ulong clTime;
	cl_profiling_info profilingData = bStart ? CL_PROFILING_COMMAND_END : CL_PROFILING_COMMAND_START;
	errCode = clGetEventProfilingInfo(TimingEvent, CL_PROFILING_COMMAND_END,
						sizeof(cl_ulong), &clTime, NULL);
	GPUManagement::CheckError( errCode );
	if(errCode != CL_SUCCESS) throw GPUException();
	// cleaning
	clReleaseEvent(TimingEvent);
	clReleaseMemObject(timing_memory);
	return clTime;
}

void GPUManagement::StartTimer(){
	_ProfilingStartTime = GetGPUTime(true);
}

float GPUManagement::StopTimer(){
	cl_ulong EndTime = GetGPUTime(false);
	// the numbers are in nanoseconds
	// convert to milliseconds
	return ((((float)(EndTime - _ProfilingStartTime))/1000.0)/1000.0);
}

bool OpenCLDeviceAndContext::CreateMemBuffer(size_t bufferSize, OpenCLMemBuffer* & memBuffer) {
	memBuffer = new OpenCLMemBuffer(this, bufferSize);
	if(!memBuffer->CreatedOK()) {
		delete memBuffer;
		memBuffer = NULL;
		throw GPUException();
		return false;
	}
	return true;
}

bool OpenCLDeviceAndContext::CreateProgram(string& strProgramFile,
						string& strKernelName,
						string& strIncludeFolder,
						OpenCLKernel* &pProgram,
						bool bFileKernel) {
	pProgram = new OpenCLKernel(this,
			strProgramFile,
			strKernelName,
			strIncludeFolder,
			bFileKernel);
	if(!pProgram->CreatedOK()){
		delete pProgram;
		pProgram = NULL;
		throw GPUException();
		return false;
	}
	return true;
}

bool OpenCLDeviceAndContext::CreateQueues(int nQueues) {

	for(int jj = 0; jj  < nQueues; jj++){
		OpenCLExecutionQueue*  pQueue = new OpenCLExecutionQueue(_DeviceId, _ContextId);
		if(pQueue->CreatedOK())
			vectQueues.push_back(pQueue);
		else{
			// clean the queues vector
			for(int ii = 0; ii < vectQueues.size(); ii++){
				delete vectQueues[ii];
				vectQueues[ii] = NULL;
			}
			vectQueues.clear();
			throw GPUException();
		}
	}
	return (vectQueues.size() == nQueues);
}

int OpenCLDeviceAndContext::GetNoOfQueues() {
  return vectQueues.size();
}

OpenCLExecutionQueue* OpenCLDeviceAndContext::GetQueue(int nIndex) {
  assert( nIndex >= 0 && nIndex < vectQueues.size() );
  return vectQueues[nIndex];
}

OpenCLDeviceAndContext::OpenCLDeviceAndContext(cl_platform_id PlatformId, cl_device_id DeviceId):
_bDeviceAvailable(false),
_bHasDouble(false),
_bHasLocalMemory(false){
	_DeviceId = DeviceId;
	size_t  szRetValue;
	char szDeviceName[256];
	GPUManagement::CheckError( clGetDeviceInfo (_DeviceId,
										 CL_DEVICE_NAME,
										 256,
										 szDeviceName,
										 &szRetValue) );
	std::cout << "Device " << szDeviceName << std::endl;
	_strDeviceName = string(szDeviceName);
	GPUManagement::CheckError( clGetDeviceInfo (DeviceId,
										 CL_DEVICE_MAX_COMPUTE_UNITS,
										 sizeof(cl_uint),
										 &_nComputeUnits,
										 &szRetValue) );
	std::cout << "\tCompute Units : " << _nComputeUnits << std::endl;
	GPUManagement::CheckError( clGetDeviceInfo (DeviceId,
										 CL_DEVICE_MAX_WORK_GROUP_SIZE,
										 sizeof(size_t),
										 &_nMaxWorkGroupSize,
										 &szRetValue) );
	std::cout << "\tMaximum Workgroup Size : " << _nMaxWorkGroupSize << std::endl;
	GPUManagement::CheckError( clGetDeviceInfo (DeviceId,
								CL_DEVICE_MAX_MEM_ALLOC_SIZE,
								sizeof(cl_ulong),
								&_nMaxMemAllocationSize,
								&szRetValue) );
	std::cout << "\tMaximum Memory Allocation Size : " << _nMaxMemAllocationSize << std::endl;
	GPUManagement::CheckError( clGetDeviceInfo (DeviceId,
										 CL_DEVICE_GLOBAL_MEM_SIZE,
										 sizeof(cl_ulong),
										 &_nGlobalMemorySize,
										 &szRetValue) );
	std::cout << "\tGlobal Memory Size : " << _nGlobalMemorySize << std::endl;
	cl_device_local_mem_type LocalMemoryType;
	GPUManagement::CheckError( clGetDeviceInfo (DeviceId,
								CL_DEVICE_LOCAL_MEM_TYPE,
								sizeof(cl_device_local_mem_type),
								&LocalMemoryType,
								&szRetValue) );
	switch(LocalMemoryType){
		case CL_LOCAL : 
			std::cout << "\tLocal Memory Implemented on the Device\n";
			break;
		case CL_GLOBAL : 
			std::cout << "\tLocal Memory Implemented on Global Memory\n";
			break;
		case CL_NONE :
			std::cout << "\tNo Local Memory\n";
			break;
		}
	_bHasLocalMemory = (LocalMemoryType == CL_LOCAL) || ((LocalMemoryType == CL_GLOBAL));
	GPUManagement::CheckError( clGetDeviceInfo (DeviceId,
										 CL_DEVICE_LOCAL_MEM_SIZE,
										 sizeof(cl_ulong),
										 &_nLocalMemorySize,
										 &szRetValue) );
	std::cout << "\tLocal Memory Size : " << _nLocalMemorySize << std::endl;
// NVidia doesn't implement the CL_DEVICE_DOUBLE_FP_CONFIG device info query for now
// Hopefully that no one will try to run this programs on an old device
#ifdef CL_DEVICE_DOUBLE_FP_CONFIG
	GPUManagement::CheckError( clGetDeviceInfo (DeviceId,
										 CL_DEVICE_DOUBLE_FP_CONFIG,
										 sizeof(cl_device_fp_config),
										 &_DeviceDoubleConfig,
										 &szRetValue) );
	_bHasDouble = _DeviceDoubleConfig == (CL_FP_FMA |
								CL_FP_ROUND_TO_NEAREST |
								CL_FP_ROUND_TO_ZERO |
								CL_FP_ROUND_TO_INF |
								CL_FP_INF_NAN |
								CL_FP_DENORM);
#else
	_bHasDouble =	true;
#endif
	std::cout << "\tHas double : " << (_bHasDouble ? "True" : "False") << std::endl;
	GPUManagement::CheckError( clGetDeviceInfo (DeviceId,
										 CL_DEVICE_AVAILABLE,
										 sizeof(cl_bool),
										 &_bDeviceAvailable,
										 &szRetValue) );
	std::cout << "\tDevice Available : " << (_bDeviceAvailable ? "TRUE" : "FALSE") << std::endl;
	// check subdevices
	cl_uint nMaxSubdevices;
	GPUManagement::CheckError( clGetDeviceInfo (DeviceId,
												CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
												sizeof(cl_uint),
												&nMaxSubdevices,
												&szRetValue) );
	std::cout << "\t MaxSubdevices : " << nMaxSubdevices << std::endl;
	// create the context on the device
	cl_context_properties* properties;
	properties = new cl_context_properties[3];
	properties[0] = CL_CONTEXT_PLATFORM;
	properties[1] = (cl_context_properties)PlatformId;
	properties[2] = 0;
	cl_device_id* devices;
	devices = new cl_device_id[1];
	devices[0] = _DeviceId;
	cl_int errcode_ret;
	_ContextId = clCreateContext(properties,
									1,
									devices,
									NULL,
									NULL,
									&errcode_ret);
	GPUManagement::CheckError( errcode_ret );
	delete [] devices;
	delete [] properties;
}

//clean everything
OpenCLDeviceAndContext::~OpenCLDeviceAndContext(){
}

//Launch the jobs in the queue.
void OpenCLExecutionQueue::Flush() {
	GPUManagement::CheckError( clFlush(_CommandQueueId) );
}

//block the calling thread untill all the jobs on the queue are finished
void OpenCLExecutionQueue::Synchronize() {
	GPUManagement::CheckError( clFinish(_CommandQueueId) );
}

OpenCLExecutionQueue::OpenCLExecutionQueue(cl_device_id DeviceId, cl_context ContextId) {
	cl_command_queue_properties props = CL_QUEUE_PROFILING_ENABLE;
	cl_int errcode_ret;

	_CommandQueueId = clCreateCommandQueue (ContextId, DeviceId, props, &errcode_ret);
	GPUManagement::CheckError( errcode_ret );
	_bCreatedOK = (errcode_ret == CL_SUCCESS);
}

OpenCLExecutionQueue::~OpenCLExecutionQueue(){
	GPUManagement::CheckError( clReleaseCommandQueue (_CommandQueueId) );
}

bool OpenCLMemBuffer::MemWrite(const void* hostBuffer, OpenCLExecutionQueue* pQueue) {
	cl_int errcode;
	errcode = clEnqueueWriteBuffer( pQueue->GetQueueId(),
								_MemBufferId,
								CL_TRUE,
								0,
								_bufferSize,
								hostBuffer,
								0,
								NULL,
								NULL);
	GPUManagement::CheckError( errcode );
	if(errcode != CL_SUCCESS)
		throw GPUException();
	return (errcode == CL_SUCCESS);
}

bool OpenCLMemBuffer::MemWrite(const void* hostBuffer, OpenCLExecutionQueue* pQueue, size_t offset, size_t nBytes){
	if(offset + nBytes > _bufferSize){
		throw GPUException();
		return false;
	}
	cl_int errcode;
	errcode = clEnqueueWriteBuffer( pQueue->GetQueueId(),
								_MemBufferId,
								CL_TRUE,
								offset,
								nBytes,
								hostBuffer,
								0,
								NULL,
								NULL);
	GPUManagement::CheckError( errcode );
	if(errcode != CL_SUCCESS)
		throw GPUException();
	return (errcode == CL_SUCCESS);
}

bool OpenCLMemBuffer::MemRead(void* hostBuffer, OpenCLExecutionQueue* pQueue) {
	cl_int errcode = clEnqueueReadBuffer ( pQueue->GetQueueId(),
								_MemBufferId,
								CL_TRUE,
								0,
								_bufferSize,
								hostBuffer,
								0,
								NULL,
								NULL);
	GPUManagement::CheckError( errcode );
	if(errcode != CL_SUCCESS)
		throw GPUException();
	return (errcode == CL_SUCCESS);
}

bool OpenCLMemBuffer::MemRead(void* hostBuffer, OpenCLExecutionQueue* pQueue, size_t offset, size_t nBytes){
	if(offset + nBytes > _bufferSize){
		throw GPUException();
		return false;
	}
	cl_int errcode = clEnqueueReadBuffer ( pQueue->GetQueueId(),
								_MemBufferId,
								CL_TRUE,
								offset,
								nBytes,
								hostBuffer,
								0,
								NULL,
								NULL);
	GPUManagement::CheckError( errcode );
	if(errcode != CL_SUCCESS)
		throw GPUException();
	return (errcode == CL_SUCCESS);
}

void OpenCLMemBuffer::MemFill(int value, OpenCLExecutionQueue* pQueue){
	void* hostBuffer = malloc(_bufferSize);
	if(hostBuffer == NULL)
		throw GPUException();
	std::memset(hostBuffer, value, _bufferSize);
	cl_int errCode = clEnqueueWriteBuffer( pQueue->GetQueueId(),
								_MemBufferId,
								CL_TRUE,
								0,
								_bufferSize,
								hostBuffer,
								0,
								NULL,
								NULL);
	free(hostBuffer);
	GPUManagement::CheckError( errCode );
	if(errCode != CL_SUCCESS)
		throw GPUException();
}

void OpenCLMemBuffer::MemFill(void* pPattern, size_t szPattern, OpenCLExecutionQueue* pQueue){
	char* hostBuffer = (char*)malloc(_bufferSize);
	if(hostBuffer == NULL)
		throw GPUException();
	size_t offset;
	// fill the host buffer with copies of pPattern
	for(offset = 0; offset + szPattern <= _bufferSize; offset += szPattern)
		std::memcpy(&(hostBuffer[offset]), pPattern, szPattern);
	cl_int errCode = clEnqueueWriteBuffer( pQueue->GetQueueId(),
			_MemBufferId,
			CL_TRUE,
			0,
			_bufferSize,
			hostBuffer,
			0,
			NULL,
			NULL);
	free(hostBuffer);
	GPUManagement::CheckError( errCode );
	if(errCode != CL_SUCCESS)
		throw GPUException();
}

OpenCLMemBuffer::OpenCLMemBuffer(OpenCLDeviceAndContext * pDeviceAndContext, size_t memSize) {
	cl_int errcode_ret;
	cl_mem bufferID = clCreateBuffer ( pDeviceAndContext->ContextId(), CL_MEM_READ_WRITE, memSize, 0, &errcode_ret);
	GPUManagement::CheckError(errcode_ret);
	if(errcode_ret == CL_SUCCESS){
		_bufferSize = memSize;
		_MemBufferId = bufferID;
		_DeviceContext = pDeviceAndContext;
		_bCreatedOK = true;
		return;
	}
	_bCreatedOK = false;
	_bufferSize = 0;
	_MemBufferId = 0;
	throw GPUException();
}

OpenCLMemBuffer::~OpenCLMemBuffer(){
	if( _bCreatedOK )
		GPUManagement::CheckError( clReleaseMemObject(_MemBufferId) );
}

bool OpenCLKernel::Execute(OpenCLExecutionQueue* executionQueue, 
						   const std::vector<size_t> & szGrid, 
						   const std::vector<size_t> & szBlock) {
	if( (szGrid.size() != szBlock.size()) && szGrid.size() < 3 ){
		throw GPUException();
		return false;
	}

	int nDimensions = szGrid.size();
	size_t* gridSize = new size_t[nDimensions];
	size_t* blockSize = new size_t[nDimensions];

	for(int ii = 0; ii < nDimensions; ii++){
		gridSize[ii] = szGrid[ii];
		blockSize[ii] = szBlock[ii];
	}
	cl_int errcode = clEnqueueNDRangeKernel(
						executionQueue->GetQueueId(),
						_KernelId,
						nDimensions,
						NULL,
						gridSize,
						blockSize,
						0,
						NULL,
						NULL);
	GPUManagement::CheckError(errcode);
	if(errcode != CL_SUCCESS)
		throw GPUException();
	return (errcode == CL_SUCCESS);
}

OpenCLKernel::OpenCLKernel(OpenCLDeviceAndContext* pDevice,
						string& strProgramFile,
						string& strKernelName,
						string& strIncludeFolder,
						bool bFileKernel):
	_bCreatedOK(false)
{
	// initialize the include directories
	_vectOpenCLIncludeDirectories.push_back( string("./") );
	if(strIncludeFolder.size() != 0)
		_vectOpenCLIncludeDirectories.push_back( strIncludeFolder );
	// create the program object
	size_t      size;
    char*       str;
	cl_int errcode_ret;
	if(bFileKernel){
		// Open file stream
		std::fstream f(strProgramFile.c_str(), (std::fstream::in | std::fstream::binary));
		// Check if we have opened file stream
		if (f.is_open()) {
			size_t  sizeFile;
			// Find the stream size
			f.seekg(0, std::fstream::end);
			size = sizeFile = (size_t)f.tellg();
			f.seekg(0, std::fstream::beg);

			str = new char[size + 1];
			if (!str) {
				f.close();
				return;
			}

			// Read file
			f.read(str, sizeFile);
			f.close();
			str[size] = '\0';
		}
		else{
			throw std::invalid_argument(strProgramFile);
		}
	}
	else{
		size = strProgramFile.size();
		str = new char[size + 1];
		strcpy(str, strProgramFile.c_str());
	}

	size_t arr_sizes[1] = {size};
    _ProgramId = clCreateProgramWithSource(pDevice->ContextId(),
				1,
				(const char**)(&str),
				arr_sizes,
				&errcode_ret);
	GPUManagement::CheckError( errcode_ret );
	if(errcode_ret != CL_SUCCESS){
		delete [] str;
		return;
	}
	std::string flagsStr;
	for(int ii = 0; ii < _vectOpenCLIncludeDirectories.size(); ii++)
		flagsStr += string(" -I ") + _vectOpenCLIncludeDirectories[ii];
#ifdef _GPU_DEBUUGGING_
	flagsStr += string(" -g");
#endif
	cl_device_id devid = pDevice->DeviceId();
	errcode_ret = clBuildProgram(_ProgramId, 1, &devid, flagsStr.c_str(), NULL, NULL);
	GPUManagement::CheckError( errcode_ret );
	if(errcode_ret != CL_SUCCESS)
	{
		if(errcode_ret == CL_BUILD_PROGRAM_FAILURE)
		{
			cl_int logStatus;
			char *buildLog = NULL;
			size_t buildLogSize = 0;
			GPUManagement::CheckError( clGetProgramBuildInfo (
							_ProgramId, 
							devid, 
							CL_PROGRAM_BUILD_LOG, 
							buildLogSize, 
							buildLog, 
							&buildLogSize) );
			buildLog = (char*)malloc(buildLogSize);
			if(buildLog == NULL) std::cerr << "Failed to allocate host memory. (buildLog)";

			memset(buildLog, 0, buildLogSize);

			GPUManagement::CheckError( clGetProgramBuildInfo (
							_ProgramId, 
							devid, 
							CL_PROGRAM_BUILD_LOG, 
							buildLogSize, 
							buildLog, 
							NULL) );

			std::cout << " \n\t\t\tBUILD LOG\n";
			std::cout << " ************************************************\n";
			std::cout << buildLog << std::endl;
			std::cout << " ************************************************\n";
			free(buildLog);
		}
	}
    delete[] str;
	// create the kernel
	_KernelId = clCreateKernel(_ProgramId, strKernelName.c_str(), &errcode_ret);
	GPUManagement::CheckError( errcode_ret );
	if(errcode_ret != CL_SUCCESS){
		return;
	}
	errcode_ret = clGetKernelInfo (_KernelId, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), (void *)(&_NParams),NULL);
	GPUManagement::CheckError( errcode_ret );
	if(errcode_ret != CL_SUCCESS){
		return;
	}
	_bCreatedOK = true;
	_DeviceContext = pDevice;
	_KernelName = strKernelName;
	return;
}

OpenCLKernel::~OpenCLKernel(){
	if(_bCreatedOK){
		GPUManagement::CheckError( clReleaseKernel (_KernelId) );
		GPUManagement::CheckError( clReleaseProgram (_ProgramId) );
	}
}

bool OpenCLKernel::SetParameter(int nIndex, size_t nSize, const void* pParam) {
	if(nIndex >= _NParams || !_bCreatedOK){
		throw GPUException();
		return false;
	}
	cl_int errcode = clSetKernelArg(_KernelId, nIndex, nSize, pParam);
	GPUManagement::CheckError( errcode );
	if(errcode != CL_SUCCESS){ 
		throw GPUException();
		return false;
	}
	return true;
}

bool OpenCLKernel::SetParameter(int nIndex, OpenCLMemBuffer* pBuffer) {
	if(nIndex >= _NParams || !_bCreatedOK){
		char str[10];
		sprintf(str, "%d", nIndex);
		throw std::invalid_argument(str);
		return false;
	}
	cl_int errcode = clSetKernelArg(_KernelId, nIndex, sizeof(cl_mem), (void *)(&(pBuffer->BufferId())));
	GPUManagement::CheckError( errcode );
	if(errcode != CL_SUCCESS){
		throw GPUException();
		return false;
	}
	return true;
}

bool OpenCLKernel::SetSharedMemParameter(int nIndex, size_t nSize) {
	if(nIndex >= _NParams || !_bCreatedOK){
		char str[10];
		sprintf(str, "%d", nIndex);
		throw std::invalid_argument(str);
		return false;
	}

	cl_int errcode = clSetKernelArg(_KernelId, nIndex, nSize, NULL);
	GPUManagement::CheckError( errcode );
	if(errcode != CL_SUCCESS){
		throw GPUException();
		return false;
	}
	return true;
}
