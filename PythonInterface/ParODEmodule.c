#include <Python.h>
#include <ParODECInterface.h>

static PyObject *ParODEError;

// void GetErrorDescription(int *ErrNo, char strErrDescription[]);
static PyObject*
ParODE_GetErrorDescription(PyObject *self, PyObject *args){
	// input : nErr
	int nErr;
	if(!PyArg_ParseTuple(args, "i", &nErr))
		return NULL;
	char strErr[256];
	GetErrorDescription(&nErr, strErr);
	return Py_BuildValue("s", strErr);
}

// void InitializeAllGPUs(int nErr[], char* pszKernelFolder, char* pszIncludeFolder,);
static PyObject*
ParODE_InitializeAllGPUs(PyObject *self, PyObject *args){
	int nErr = 0;
	char* strKernelFolder;
	int strSize = strlen(KERNEL_FOLDER);
	strKernelFolder = (char*)malloc( (strSize + 1) * sizeof(char) );
	if(strKernelFolder == NULL)
		return NULL;
	strcpy(strKernelFolder, KERNEL_FOLDER);
	strSize = strlen( KERNEL_INCLUDE_FOLDER );
	char* strIncludeFolder;
	strIncludeFolder = (char*)malloc((strSize + 1) * sizeof(char));
	if(strIncludeFolder == NULL){
		free(strKernelFolder);
		return NULL;
	}
	strcpy(strIncludeFolder, KERNEL_INCLUDE_FOLDER);
	// printf("KernelFolder : %s\n", strKernelFolder);
	// printf("Include Folder : %s\n", strIncludeFolder);
	InitializeAllGPUs(strKernelFolder, strIncludeFolder, &nErr);
	free(strKernelFolder);
	free(strIncludeFolder);
	return Py_BuildValue("i", nErr);
}

// void GetAvailableGPUs( int nGPUs[], int GPUIds[], int nErr[]);
static PyObject*
ParODE_GetAvailableGPUs(PyObject *self, PyObject *args){
  	int nGPUs = 0;
	int GPUIds[8];
	int nErr = 0;
	//  
	GetAvailableGPUs(&nGPUs, GPUIds, &nErr);
	PyObject *item;
	PyObject *retTuple;
	retTuple = PyTuple_New(2);
	if(retTuple == NULL){
		PyErr_SetString(ParODEError, "Not Enough Resources");
		return NULL;
	}
	PyObject* listGPUs = PyList_New(nGPUs);
	if(listGPUs == NULL){
		PyErr_SetString(ParODEError, "Not Enough Resources");
		Py_DECREF(retTuple);
		return NULL;
	}
	int ii;
	for(ii = 0; ii < nGPUs; ii++){
		item = Py_BuildValue("i", GPUIds[ii]);
		PyList_SET_ITEM(listGPUs, ii, item);
	}
	PyTuple_SET_ITEM(retTuple, 0, listGPUs);
	item = Py_BuildValue("i", nErr);
	PyTuple_SET_ITEM(retTuple, 1, item);
	return retTuple;
}

// void GetDeviceName( int *DeviceId, char strDeviceName[], int nErr[]);
static PyObject*
ParODE_GetDeviceName(PyObject *self, PyObject *args){
	int deviceID;
	if(!PyArg_ParseTuple(args, "i", &deviceID))
		return NULL;
	char deviceName[128];
	int nErr = 0;
	GetDeviceName(&deviceID, deviceName, &nErr);
	return Py_BuildValue("s i", deviceName, nErr);
}

// void InitializeSelectedGPUs( int *nGPUs, int GPUIds[], int nErr[]);
static PyObject*
ParODE_InitializeSelectedGPUs(PyObject *self, PyObject *args){
	int nErr = 0;
	PyObject* pListGPUs;
	if(!PyArg_ParseTuple(args, "O", &pListGPUs))
		return NULL;
	int nElements;
	int *GPUIds;
	if(!PyList_CheckExact(pListGPUs))
		return NULL;
	nElements = PyList_Size(pListGPUs);
	GPUIds = (int*)malloc(nElements * sizeof(int));
	int ii;
	for(ii = 0; ii < nElements; ii++){
		PyObject* pElement = PyList_GET_ITEM(pListGPUs, ii);
		long value = PyInt_AsLong(pElement);
		if(value == -1 && PyErr_Occurred()){
			free(GPUIds);
			return NULL;
		}
		GPUIds[ii] = (int)value;
	}
	char* strKernelFolder;
	int strSize = strlen(KERNEL_FOLDER);
	strKernelFolder = (char*)malloc( (strSize + 1) * sizeof(char) );
	if(strKernelFolder == NULL){
		free(GPUIds);
		return NULL;
	}
	strcpy(strKernelFolder, KERNEL_FOLDER);
	strSize = strlen( KERNEL_INCLUDE_FOLDER );
	char* strIncludeFolder;
	strIncludeFolder = (char*)malloc((strSize + 1) * sizeof(char));
	if(strIncludeFolder == NULL){
		free(GPUIds);
		free(strKernelFolder);
		return NULL;
	}
	strcpy(strIncludeFolder, KERNEL_INCLUDE_FOLDER);
	InitializeSelectedGPUs( &nElements, GPUIds, strKernelFolder, strIncludeFolder, &nErr );
	free(GPUIds);
	free(strKernelFolder);
	free(strIncludeFolder);
	return Py_BuildValue("i", nErr);
}

int DisassembleMatrix(PyObject *pMatrixList, float **pMatrix, int *rows, int *cols){
	// builds a C matrix from a list
	// assumes a row major representation
	// returns 0 on failure
	if( !PyList_CheckExact(pMatrixList) ) return(0);
	*rows = PyList_GET_SIZE(pMatrixList);
	if(*rows == 0){
		*cols = 0;
		*pMatrix = NULL;
		return(1);
	}
	int ii, jj;
	PyObject* rowList;
	PyObject* matrixElement;
	int nColumnsLocal;
	double dVal;
	*cols = -1;
	for(ii = 0; ii < *rows; ii++){
		rowList = PyList_GET_ITEM(pMatrixList, ii);
		if( !PyList_CheckExact(rowList) ) 
			if( *cols != -1 ){
				free(*pMatrix);
				return(0);
			}
		// get the number of columns
		nColumnsLocal = PyList_GET_SIZE(rowList);
		if(*cols == -1){
			*cols = nColumnsLocal;
			*pMatrix = (float*)malloc((*rows)*(*cols)*sizeof(float));
			if(*pMatrix == NULL)
				return(0);
		}
		else if(nColumnsLocal != *cols){
			// the matrix list does not have proper sizes
			free(*pMatrix);
			return(0);
		}
		// copy all list elements into the matrix
		for(jj = 0; jj < *cols; jj++){
			matrixElement = PyList_GET_ITEM(rowList, jj);
			dVal = PyFloat_AsDouble(matrixElement);
			if(dVal == -1.0 && PyErr_Occurred()){
				free(*pMatrix);
				return(0);
			}
			(*pMatrix)[ii * (*cols) + jj] = (float)(dVal);
		}
	}
	return(1);
}

void PrintMatrix(float* pM, int rows, int cols){
	// print a matrix stored in a vector as a row major representation
	if(pM == NULL) return; // nothing to do
	int ii, jj;
	for(ii = 0; ii < rows; ii++){
		for(jj = 0; jj < cols; jj++)
			printf("%f, ", pM[ii * cols + jj]);
		printf("\n"); 
	}
}

// interface for plain C calls
// the caller should take care of memory allocation and release for A, B, C, D
// the caller should take care of memory release for tVect and xVect
// void SimulateODE(	fType A[], // row major A
//						fType B[], // row major B
//						fType C[], // row major C
//						fType D[], // row major D
//						int *nInputs,
//						int *nStates,
//						int *nOutputs,
//						int *ZeroB,
//						int *ZeroC,
//						int *ZeroD,
//						int *uIndex,
//						double *tStart, double *tEnd, double *tStep,
//						fType x0[],
//						int *solver,
//						// Outputs
//						fType tVect[],
//						fType xVect[],
//						int *nSteps,
//						int nErr[]);
static PyObject*
ParODE_SimulateODE(PyObject *self, PyObject *args){
	PyObject* AList;
	float* A = NULL;
	PyObject* BList;
	float* B = NULL;
	PyObject* CList;
	float* C = NULL;
	PyObject* DList;
	float* D = NULL;
	int uIndex;
	double tStart;
	double tStep;
	int nSteps;
	PyObject* x0List;
	float* x0 = NULL;
	int solver;
	if(!PyArg_ParseTuple(args, "OOOOiddiOi", 
		&AList, &BList, &CList, &DList,
		&uIndex, &tStart, &tStep, &nSteps, &x0List, &solver))
		return NULL;
	int 	nStates = 0, 
		nInputs = 0, 
		nOutputs = 0;
	int 	ZeroB = 0, 
		ZeroC = 0, 
		ZeroD = 0;
	if(!PyList_CheckExact(AList) ||
		!PyList_CheckExact(BList) ||
		!PyList_CheckExact(CList) ||
		!PyList_CheckExact(DList) ||
		!PyList_CheckExact(x0List))
		return NULL;
	// convert A
	int nCols, nRows;
	if( !DisassembleMatrix(AList, &A, &nRows, &nCols) )
		goto ReturnErr;
	// A cannot be empty; A has to be square
	if(nCols == 0 || nRows == 0 || nCols != nRows)
		goto ReturnErr;
	nStates = nRows;
	// convert B
	if( !DisassembleMatrix(BList, &B, &nRows, &nCols) )
		goto ReturnErr;
	if( nRows == 0 || nCols == 0)
		ZeroB = 1;
	else if( nRows == nStates && nCols != 0)
		nInputs = nCols;
	else 
		goto ReturnErr;
	// Convert C
	if( !DisassembleMatrix(CList, &C, &nRows, &nCols) )
		goto ReturnErr;
	if( nRows == 0 || nCols == 0)
		ZeroC = 1;
	else if( nCols == nStates ) 
		nOutputs = nRows;
	else
		goto ReturnErr;
	// convert D
	if( !DisassembleMatrix(DList, &D, &nRows, &nCols) )
		goto ReturnErr;
	if( nRows == 0 || nCols == 0)
		ZeroD = 1;
	else{
		if( ZeroB ) nInputs = nCols;
		if( ZeroC ) nOutputs = nRows;
		if( nInputs != nCols || 
		    nOutputs != nRows)
			goto ReturnErr;
	}
	if(ZeroC && ZeroD) nOutputs = nStates;
	// convert x0
	if( !PyList_CheckExact(x0List) )
		goto ReturnErr;
	nRows = PyList_Size(x0List);
	if( nRows != nStates )
		goto ReturnErr;
	int ii;
	PyObject* element;
	double dElementValue;
	x0 = (float*)malloc(nStates * sizeof(float));
	for( ii = 0; ii < nStates; ii++){
		element = PyList_GetItem(x0List, ii);
		dElementValue = PyFloat_AsDouble(element);
		if( dElementValue == -1 && PyErr_Occurred() )
			goto ReturnErr;
		x0[ii] = (float)dElementValue;
	}
	// print input values
	printf("A : \n");
	PrintMatrix(A, nStates, nStates);
	printf("B : \n");
	PrintMatrix(B, nStates, nInputs);
	printf("C : \n");
	PrintMatrix(C, nOutputs, nStates);
	printf("D : \n");
	PrintMatrix(D, nOutputs, nInputs);
	printf("x0 : \n");
	for(ii = 0; ii < nStates; ii++)
		printf("%f, ", x0[ii]);
	printf("\n");
	printf("nStates = %d; nInputs = %d; nOutputs = %d\n", nStates, nInputs, nOutputs);
	printf("ZeroB = %d; ZeroC = %d; ZeroD = %d\n", ZeroB, ZeroC, ZeroD); 
	printf("uIndex = %d\n", uIndex);
	printf("tStart = %lf, tStep = %lf, nSteps = %d\n", tStart, tStep, nSteps);
	printf("Solver : %d\n", solver);
	
	float* tVect;
	float* xVect;
	int nStepsOut;
	double tEnd = tStart + (nSteps - 1) * tStep;
	int nErr = 0; 
	tVect = (float*)malloc(nSteps * sizeof(float));
	xVect = (float*)malloc(nSteps * nOutputs * sizeof(float));
	SimulateODE(A, // row major A
			B, // row major B
			C, // row major C
			D, // row major D
			&nInputs,
			&nStates,
			&nOutputs,
			&ZeroB,
			&ZeroC,
			&ZeroD,
			&uIndex,
			&tStart, &tEnd, &tStep,
			x0,
			&solver,
//			Outputs
			tVect,
			xVect,
			&nStepsOut,
			&nErr);
	int jj;
	// return the computed values
	PyObject* retTuple;
	retTuple = PyTuple_New(3);
	PyObject* tVectList;
	PyObject* xVectList;
	PyObject* xListItem;
	tVectList = PyList_New(nStepsOut);
	xVectList = PyList_New(nStepsOut);
	if(retTuple == NULL || tVectList == NULL || xVectList == NULL)
		goto ReturnErr;
	// build tVect and xVect
	for(ii = 0; ii < nStepsOut; ii++){
		element = PyFloat_FromDouble( tVect[ii] );
		if( element == NULL )
			goto ReturnErr1;
		PyList_SET_ITEM(tVectList, ii, element);
		xListItem = PyList_New(nOutputs);
		if( xListItem == NULL)
			goto ReturnErr1;
		// populate xListItem
		for(jj = 0; jj < nOutputs; jj++){
			element = PyFloat_FromDouble( xVect[ii * nOutputs + jj] );
			if(element == NULL){
				Py_DECREF(xListItem);
				goto ReturnErr1;
			}
			PyList_SET_ITEM(xListItem, jj, element);
		}
		PyList_SET_ITEM(xVectList, ii, xListItem);
	}
	PyTuple_SET_ITEM(retTuple, 0, tVectList);
	PyTuple_SET_ITEM(retTuple, 1, xVectList);
	element = Py_BuildValue("i", nErr);
	PyTuple_SET_ITEM(retTuple, 2, element);
	// cleanup
	free(tVect);
	free(xVect);
	if(A != NULL) free(A);
	if(B != NULL) free(B);
	if(C != NULL) free(C);
	if(D != NULL) free(D);
	if(x0 != NULL) free(x0);
	return retTuple;
	ReturnErr1:
		Py_DECREF(retTuple);
		Py_DECREF(tVectList);
		Py_DECREF(xVectList);
	ReturnErr:
 		// clean-up
		if(A != NULL) free(A);
		if(B != NULL) free(B);
		if(C != NULL) free(C);
		if(D != NULL) free(D);
		if(x0 != NULL) free(x0);
		return Py_None;
}

// void RegisterInput(char uFunc[], int uIndex[], int nErr[]);
static PyObject*
ParODE_RegisterInput(PyObject *self, PyObject *args){
	char *uFunc;
	if(!PyArg_ParseTuple(args, "s", &uFunc))
		return NULL;
	int uIndex = 0;
	int nErr = 0;
	RegisterInput(uFunc, &uIndex, &nErr);
	return Py_BuildValue("i i", uIndex, nErr);
}

//Release all GPU datastructures.
// void CloseGPUC();
static PyObject*
ParODE_CloseGPU(PyObject *self, PyObject *args){
	CloseGPUC();
	Py_INCREF(Py_None);
	return Py_None;
}

// timing functions
// void StartTimer(int nErr[]);
static PyObject*
ParODE_StartTimer(PyObject *self, PyObject *args){
	int nErr = 0;
	StartTimer(&nErr);
	return Py_BuildValue("i", nErr);
}

// void StopTimer(float fTime[], int nErr[])
static PyObject*
ParODE_StopTimer(PyObject *self, PyObject *args){
	float fTime = 0.0f;
	int nErr = 0;
	StopTimer(&fTime, &nErr);
	return Py_BuildValue("f i", fTime, nErr);
}

static PyMethodDef ParODEMethods[] = {
    {"GetErrorDescription", ParODE_GetErrorDescription, METH_VARARGS, "Get Error Description String."},
    {"InitializeAllGPUs", ParODE_InitializeAllGPUs, METH_VARARGS, "Initialize all available GPUs."},
    {"GetAvailableGPUs", ParODE_GetAvailableGPUs, METH_VARARGS, "Retrieve all the available GPUs."},
    {"GetDeviceName", ParODE_GetDeviceName, METH_VARARGS, "Get the device name."},
    {"InitializeSelectedGPUs", ParODE_InitializeSelectedGPUs, METH_VARARGS, "Initialize Selected GPUs."},
    {"SimulateODE", ParODE_SimulateODE, METH_VARARGS, "Simulate an LTI system."},
    {"RegisterInput", ParODE_RegisterInput, METH_VARARGS, "Register an input  kernel."},
    {"CloseGPU", ParODE_CloseGPU, METH_VARARGS, "Release GPU resources."},
    {"StartTimer", ParODE_StartTimer, METH_VARARGS, "Start GPU timer."},
    {"StopTimer", ParODE_StopTimer, METH_VARARGS, "Retrieve timer value."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initParODE(void)
{
    PyObject *m;
    m = Py_InitModule("ParODE", ParODEMethods);
    if (m == NULL)
        return;
    ParODEError = PyErr_NewException("ParODE.error", NULL, NULL);
    Py_INCREF(ParODEError);
    PyModule_AddObject(m, "error", ParODEError);
}

