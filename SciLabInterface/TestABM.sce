function TestABM()
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
    // register the input  kernel
    uFunc = 'fType UFunc(fType t, int nInputIndex){' + ...
                'fType fVal = nInputIndex == 0 ? t :' + ...
                'nInputIndex == 1 ? t :' + ...
                '0.0;' + ...
                'return fVal;' + ...
                '}';
    disp( uFunc );
    nErr = StartTimer();
    if nErr == 0 then
        disp( 'ParODE Start Timer OK' );
    else
        disp( 'ParODE Start Timer Error' );
    end
    [uIndex, nErr] = RegisterInput(uFunc);
    if nErr == 0 then
        disp( 'ParODE Register Input OK' );
    else
        disp( 'ParODE Register Input Error' );
    end
    [fTime, nErr] = StopTimer();
    if nErr == 0 then
        disp( 'ParODE Stop Timer OK' );
        disp(fTime);
    else
        disp( 'ParODE Stop Timer Error' );
    end
    A = [-2.0, 0.0; 0.0, -2.0];
    B = [1.0, 0.0; 0.0, 1.0];
    C = [1.0, 0.0; 0.0, 1.0];
    D = [1.0, 0.0; 0.0, 1.0];
    x0 = [1.0, -1.0];
    solver = 1;
    tStart = 0.0;
    tStep = 0.1;
    nSteps = 50;
    // call the simulation
    [tVect, xVect, nErr] = SimulateODE(A, B, C, D, ...
                            uIndex, tStart, tStep, nSteps, x0, solver);
    if nErr == 0 then
        disp('Test ABM OK');
        disp(tVect);
        disp(xVect);
    else
        disp('Test ABM failed');
    end

    // release the library
    CloseGPU();
    // unload library
    UnloadParODE(parODELibIndex);
endfunction
