import ParODE

(GPUList, nErr) =  ParODE.GetAvailableGPUs();
print 'Available GPUs : {0};'.format( GPUList );
(GPUName, nErr) = ParODE.GetDeviceName( GPUList[0] );
print 'GPU Name : {0}'.format( GPUName );
nErr = ParODE.InitializeSelectedGPUs(GPUList);
A = [ [-2.0, 1.0], [0.5, -3.0] ];
B = [ [1.0, 0.0], [0.0, 1.0] ];
C = [ [1.0, 0.0], [0.0, 1.0] ];
D = [ [1.0, 0.0], [0.0, 1.0] ];

uFunc = ('fType UFunc(fType t, int nInputIndex){\n'
	'fType fVal = nInputIndex == 0 ? t :\n'
	'nInputIndex == 1 ? t :\n'
	'0.0;\n'
	'return fVal;\n'
	'}\n');
(uIndex, nErr) = ParODE.RegisterInput( uFunc );
print 'uIndex = {0}, nErr = {1}'.format(uIndex, nErr) ;
x0 = [-1.0, 1.0];
solver = 1;
tStart = 0.0;
tStep = 0.1;
nSteps = 50;
(tVect, xVect, nErr) = ParODE.SimulateODE(A, B, C, D, uIndex, tStart, tStep, nSteps, x0, solver);
print 'tVect = {0}'.format( tVect );
print 'xVect = {0}'.format( xVect );
print 'nErr = {0}'.format( nErr );
ParODE.CloseGPU();
print( 'Done!' );

