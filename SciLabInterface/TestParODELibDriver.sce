function TestParODELibDriver()
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
    // initialize all GPUs
    nErr = InitializeAllGPUs(strParODEKFolder, strParODEKIncludeFolder);
    disp( nErr );
    // release the library
    CloseGPU();
    // unload library
    UnloadParODE(parODELibIndex);
endfunction
