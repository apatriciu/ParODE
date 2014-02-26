// void GetErrorDescription(int *ErrNo, char strErrDescription[]);
function s = GetErrorDescription(ErrNo)
    s = call('GetErrorDescription', ErrNo, 1, 'i', 'out', [1,128], 2, 'c');
endfunction
// c = call('Function1', a, 1, 'r', n, 2, 'i', 'out', [m,n], 3, 'r')
// void InitializeAllGPUs(int nErr[]);
function nErr = InitializeAllGPUs(strParODEKFolder, strParODEKIncludeFolder)
    nErr = call('InitializeAllGPUs', strParODEKFolder, 1, 'c', ...
                                    strParODEKIncludeFolder, 2, 'c', ...
                                    'out', ...
                                    [1, 1], 3, 'i');
endfunction

// void GetAvailableGPUs( int nGPUs[], int GPUIds[], int nErr[]);
function [nGPUs, GPUIds, nErr] = GetAvailableGPUs()
    [nGPUs, GPUIds, nErr] = call('GetAvailableGPUs', 'out', [1, 1], 1, 'i', ...
                                                            [1, 4], 2, 'i', ...
                                                            [1, 1], 3, 'i');
endfunction

// void GetDeviceName( int *DeviceId, char strDeviceName[], int nErr[]);
function [strDeviceName, nErr] = GetDeviceName(DeviceID)
    [strDeviceName, nErr] = call('GetDeviceName', DeviceID, 1, 'i', ...
                                    'out', ...
                                    [1, 64], 2, 'c', ...
                                    [1, 1], 3, 'i');
endfunction

// void InitializeSelectedGPUs( int *nGPUs, int GPUIds[], int nErr[]);
function nErr = InitializeSelectedGPUs(nGPUs, GPUIDs, strParODEKFolder, strParODEKIncludeFolder)
    nErr = call('InitializeSelectedGPUs', nGPUs, 1, 'i', ...
                                        GPUIDs, 2, 'i', ...
                                        strParODEKFolder, 3, 'c', ...
                                        strParODEKIncludeFolder, 4, 'c', ...
                                        'out', ...
                                        [1, 1], 5, 'i');
endfunction

// void RegisterInput(char uFunc[], int uIndex[], int nErr[]);
function [uIndex, nErr] = RegisterInput(uFunc)
    [uIndex, nErr] = call('RegisterInput', uFunc, 1, 'c', ...
                        'out',... 
                        [1, 1], 2, 'i', ...
                        [1, 1], 3, 'i');
endfunction

// void CloseGPUC();
function CloseGPU()
    call('CloseGPUC');
endfunction

// timing functions
// void StartTimer(int nErr[]);
function nErr = StartTimer()
    nErr = call('StartTimer', 'out', [1, 1], 1, 'i');
endfunction

// void StopTimer(float fTime[], int nErr[]);
function [fTime, nErr] = StopTimer()
    [fTime, nErr] = call('StopTimer', 'out', [1, 1], 1, 'r', [1, 1], 2, 'i');
endfunction

// the caller should take care of memory allocation and release for A, B, C, D
// the caller should take care of memory release for tVect and xVect
// void SimulateODE(	fType A[], // row major A
//					fType B[], // row major B
//					fType C[], // row major C
//					fType D[], // row major D
//					int *nInputs,
//					int *nStates,
//					int *nOutputs,
//					int *ZeroB,
//					int *ZeroC,
//					int *ZeroD,
//					int *uIndex,
//					double *tStart, double *tEnd, double *tStep,
//					fType x0[],
//					int *solver,
//					// Outputs
//					fType tVect[],
//					fType xVect[],
//					int *nSteps,
//					int nErr[]);
function [tVect, xVect, nErr] = SimulateODE(A, B, C, D, ...
    uIndex, tStart, tStep, nSteps, x0, solver)
    // transpose the system matrices
    // SciLab stores column major and ParODE expects row major
    ALocal = A.';
    BLocal = B.';
    CLocal = C.';
    DLocal = D.';
    tEnd = tStart + (nSteps - 1) * tStep + 0.0001 * tStep;
    nStates = size(A, 1);
    nInputs = size(B, 2);
    ZeroB = 0;
    ZeroC = 0;
    ZeroD = 0;
    if(size(B, 1) == 0)
        ZeroB = 1;
    end
    if(size(C, 1) == 0)
        ZeroC = 1;
    end
    if(size(D, 1) == 0)
        ZeroD = 1;
    end
    if(size(C, 1) <> 0)
        nOutputs = size(C, 1);
    elseif(size(D, 1) <> 0)
        nOutputs = size(D, 1);
    else
        nOutputs = nStates;
    end
    [tVect, xVect, nStepsLocal, nErr] = call('SimulateODE', ALocal, 1, 'r', ...
                                             BLocal, 2, 'r',...
                                             CLocal, 3, 'r',...
                                             DLocal, 4, 'r',...
                                             nInputs, 5, 'i',...
                                             nStates, 6, 'i',...
                                             nOutputs, 7, 'i',...
                                             ZeroB, 8, 'i',...
                                             ZeroC, 9, 'i',...
                                             ZeroD, 10, 'i',...
                                             uIndex, 11, 'i',...
                                             tStart, 12, 'r',...
                                             tEnd, 13, 'r', ...
                                             tStep, 14, 'r',...
                                             x0, 15, 'r',...
                                             solver, 16, 'i',...
                                             'out', ...
                                             [1, nSteps], 17, 'r',...
                                             [nOutputs, nSteps], 18, 'r',...
                                             [1, 1], 19, 'i',...
                                             [1, 1], 20, 'i');
endfunction
