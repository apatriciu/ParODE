function TestQueryGPUDevices()
    // this is the folder in which you have the shared library ParODEShared
    strPathParODELibrary = '/home/alexandru/Projects/ParallelODESolver/Software/Binary/';
    // this is the folder in which you have the kernel files
    strParODEKFolder = '/home/alexandru/Projects/ParallelODESolver/Software/Binary/';
    // this is the folder in which you have any headers that you need for the kernels
    strParODEKIncludeFolder = '/home/alexandru/Projects/ParallelODESolver/Software/Binary';
    // load the  functions in the environment
    exec('./LoadParODELibrary.sci', -1);
    exec('./ParODEInterfaceStubs.sci', -1);
    exec('./UnloadParODE.sci', -1);
    // load library
    parODELibIndex = LoadParODELibrary(strPathParODELibrary);
    disp( parODELibIndex );
    // Get all GPUs
    [nGPUs, GPUIds, nErr] = GetAvailableGPUs();
    if nErr == 0 then
        for ii = 1:nGPUs
            disp( GPUIds(ii) );
            [strDeviceName, nErr] = GetDeviceName( GPUIds(ii) );
            if nErr == 0 then
                disp( strDeviceName );
            else
                disp( GetErrorDescription(nErr) );
            end
        end
    else
        disp( GetErrorDescription(nErr) );
    end
    nErr = InitializeSelectedGPUs(nGPUs, GPUIds, strParODEKFolder, strParODEKIncludeFolder);
    if nErr == 0 then
        disp( 'ParODE Initialized OK' );
    else
        disp( 'ParODE Initialization Error' );
    end
    // release the library
    CloseGPU();
    // unload library
    UnloadParODE(parODELibIndex);
endfunction
