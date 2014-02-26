import ParODE

print("Test GetAvailableGPUs");
(GPUList, nErr) =  ParODE.GetAvailableGPUs();
print("Available GPUs");
print( GPUList );
print( 'Get Device Name' );
(GPUName, nErr) = ParODE.GetDeviceName( GPUList[0] );
print( "GPU Name :" );
print( GPUName );
print( 'Initialize Selected GPUs' );
nErr = ParODE.InitializeSelectedGPUs(GPUList);
print( nErr );
print( 'Start Timer' );
print( ParODE.StartTimer() );
print( 'Test Get Error Description' ); 
nErr = -1;
print( ParODE.GetErrorDescription(nErr) );
print( 'Stop Timer' );
(fTimerValue, nErr) = ParODE.StopTimer();
print 'fTimerValue = {0}; nErr = {1}'.format(fTimerValue, nErr) ;
ParODE.CloseGPU();
print( 'Done!' );

