#ifndef _GPUMANAGEMENT_H
#define _GPUMANAGEMENT_H

#include <CL/cl.h>

#include <string>
#include <vector>
#include <exception>
#include <iostream>
using namespace std;

// forward declarations
class OpenCLDeviceAndContext;
class OpenCLExecutionQueue;
class OpenCLMemBuffer;
class OpenCLKernel;

class GPUException : public exception{
	virtual const char* what() const throw(){
		return "GPU Error\n";
	}
};

class GPUManagement {
private:
	typedef struct {cl_int _errError;
			const char* _strErrorDescription;
	} ErrorStructureType;
	const static ErrorStructureType _vectErrors[];
private:
	const static cl_int MAXIMUM_NUMBER_OF_PLATFORMS = 4; // maximum number of platforms
	const static cl_uint MAX_DEVICES_PER_PLATFORM = 4; // maximum number of GPU devices per platform
	vector<OpenCLDeviceAndContext*> vectDevices;
// timing objects
	// we have a dummy memory transfer on queue 0 of 10 floats
	cl_ulong _ProfilingStartTime;
	cl_ulong GetGPUTime(bool bStart = true);
public:
    //Intializes the vector with devices and queues. 
	static inline void CheckError(cl_int nErrorCode){
		if(nErrorCode == CL_SUCCESS) return;
		int nErrIndex = 0;
		while(_vectErrors[nErrIndex]._errError != nErrorCode && _vectErrors[nErrIndex]._errError != CL_SUCCESS) nErrIndex++;
		std::cerr << _vectErrors[nErrIndex]._strErrorDescription << std::endl;
	};
    bool Initialize(int nQueuesPerDevice);
    int GetNumberOfDevices();
    GPUManagement();
    virtual ~GPUManagement();
    OpenCLDeviceAndContext* GetDeviceAndContext(int nIndex);
    void GetGPUDevices(
    		vector<unsigned int>& deviceIDs);
    string GetDeviceName(unsigned int devID);
    bool Initialize(const vector<unsigned int>& deviceIDs,
    				int nQueuesPerDevice);
    void StartTimer();
    float StopTimer();
};

//We assume that we will create only one execution context per device. 
class OpenCLDeviceAndContext {
  public:
    bool CreateMemBuffer(size_t bufferSize, OpenCLMemBuffer* & memBuffer);
    bool CreateProgram(string& strProgramFile, string& strKernelName,
    					string& strIncludeFolder,
    					OpenCLKernel* & pProgram, bool bFileKernel = true);
  private:
    vector<OpenCLExecutionQueue*> vectQueues;

  public:
    bool CreateQueues(int nQueues);
    int GetNoOfQueues();
    OpenCLExecutionQueue* GetQueue(int nIndex);
    OpenCLDeviceAndContext(cl_platform_id PlatformId, cl_device_id DeviceId);
	inline bool HasLocalMemory(){return _bHasLocalMemory;};
	inline unsigned long LocalMemorySize(){return _nLocalMemorySize;};
	inline unsigned long GlobalMemorySize(){return _nGlobalMemorySize;};
	inline unsigned long MaximumMemoryAllocationSize(){return _nMaxMemAllocationSize;};
	inline bool HasDouble(){return _bHasDouble;};
	inline bool DeviceAvailable(){return _bDeviceAvailable;}
	inline unsigned int ComputeUnits(){return _nComputeUnits;};
	inline size_t MaxWorkgroupSize(){return _nMaxWorkGroupSize;};
	inline string& DeviceName(){return _strDeviceName;};
	inline cl_device_id DeviceId(){return _DeviceId;};
    virtual ~OpenCLDeviceAndContext();

  private:
    cl_device_id	_DeviceId;
    cl_context		_ContextId;
	// data about the device
	cl_uint			_nComputeUnits;
	size_t			_nMaxWorkGroupSize;
	cl_ulong		_nMaxMemAllocationSize;
	cl_ulong		_nGlobalMemorySize;
	bool			_bHasLocalMemory;
	bool			_bHasDouble;
	cl_ulong		_nLocalMemorySize;
	cl_device_fp_config _DeviceDoubleConfig;
	bool			_bDeviceAvailable;
	string			_strDeviceName;
protected:
	inline cl_context ContextId(){return _ContextId;};
	friend class OpenCLMemBuffer;
	friend class OpenCLKernel;
	friend class GPUManagement;
};

class OpenCLExecutionQueue {
  public:
    //Launch the jobs in the queue.
    void Flush();

    //block the calling thread untill all the jobs in the queue are finished
    void Synchronize();

    OpenCLExecutionQueue(cl_device_id DeviceId, cl_context ContextId);
	inline bool CreatedOK(){return _bCreatedOK;};

    virtual ~OpenCLExecutionQueue();


  private:
    cl_command_queue	_CommandQueueId;
	bool				_bCreatedOK;
protected:
	inline cl_command_queue GetQueueId(){return _CommandQueueId;};
	friend class OpenCLMemBuffer;
	friend class OpenCLKernel;
	friend class GPUManagement;
};

class OpenCLMemBuffer {
  private:
    OpenCLDeviceAndContext * _DeviceContext;
	cl_mem _MemBufferId;
	size_t _bufferSize;
	bool _bCreatedOK;

  public:
    OpenCLMemBuffer(OpenCLDeviceAndContext * pDeviceAndContext, size_t memSize);
    virtual ~OpenCLMemBuffer();
    bool MemWrite(const void* hostBuffer, OpenCLExecutionQueue* pQueue);
	bool MemWrite(const void* hostBuffer, OpenCLExecutionQueue* pQueue, size_t offset, size_t nBytes);
    bool MemRead(void* hostBuffer, OpenCLExecutionQueue* pQueue);
    bool MemRead(void* hostBuffer, OpenCLExecutionQueue* pQueue, size_t offset, size_t nBytes);
    void MemFill(int value, OpenCLExecutionQueue* pQueue);
    void MemFill(void* pPattern, size_t szPattern, OpenCLExecutionQueue* pQueue);
	inline OpenCLDeviceAndContext* GetContext(){return _DeviceContext;}
	inline bool CreatedOK(){return _bCreatedOK;};

	template<class printTypeClass>
	void PrintAs(OpenCLExecutionQueue* pQueue, int nElemsPerLine = 0){

		size_t szElem = sizeof(printTypeClass);
		// make sure that the buffer is a multiple of printTypeClass elements
		if(_bufferSize % szElem != 0)
			cout << "Error, the buffer is not a multiple of requested elements\n";
		size_t nElems = _bufferSize / szElem;
		// allocate nElems
		printTypeClass* hostBuffer = new printTypeClass[nElems];
		cl_int errCode = clEnqueueReadBuffer ( pQueue->GetQueueId(),
									_MemBufferId,
									CL_TRUE,
									0,
									_bufferSize,
									hostBuffer,
									0,
									NULL,
									NULL);
		GPUManagement::CheckError( errCode );
		if(errCode != CL_SUCCESS)
			throw GPUException();
		// print the elements
		cout << "nElements = " << nElems << endl;
		for(int ii = 0; ii < nElems; ii++){
			cout << hostBuffer[ii] << ", ";
			if(nElemsPerLine != 0)
				if((ii + 1) % nElemsPerLine == 0) cout << endl;
		}
		cout << endl;
		delete [] hostBuffer;
	}

  protected:
  cl_mem& BufferId(){return _MemBufferId;};
  friend class OpenCLKernel;
};

class OpenCLKernel {
  private:
    cl_kernel	_KernelId;
	cl_program	_ProgramId;
    OpenCLDeviceAndContext * _DeviceContext;
	bool _bCreatedOK;
    string _KernelName;
    cl_uint _NParams;
	// Default OpenCL Include Directories 
	vector<string> _vectOpenCLIncludeDirectories;

  public:
    bool Execute(OpenCLExecutionQueue* executionQueue, 
				const std::vector<size_t> & szGrid, 
				const std::vector<size_t> & szBlock);
	// Create the OpenCL kernel object
	// pDevice - pointer to the device on which we will create the kernel
	// strProgramFile - if bFileKernel == true this is the name of the file for the kernel; 
	//					if bFileKernel == false this string contains all the program code
	// strKernelName - name of the kernel function
	// bFileKernel - specifies if strProgramFile is the name of a file or contains the kernel code
    OpenCLKernel( OpenCLDeviceAndContext* pDevice, string& strProgramFile,
    				string& strKernelName,
    				string& strIncludeFolder,
    				bool bFileKernel = true);
    virtual ~OpenCLKernel();
    bool SetParameter(int nIndex, size_t nSize, const void* pParam);
    bool SetParameter(int nIndex, OpenCLMemBuffer* pBuffer);
    bool SetSharedMemParameter(int nIndex, size_t nSize);
	inline bool CreatedOK(){return _bCreatedOK;};
	inline OpenCLDeviceAndContext* GetContext(){return _DeviceContext;};
};
#endif
