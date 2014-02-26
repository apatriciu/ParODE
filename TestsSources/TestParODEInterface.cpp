#define __TESTING_KERNELS__
#include <ParODECppInterface.h>
#include <ParODE.h>
#include <string>
#include <iostream>
#include <vector>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/odeint.hpp>

using namespace std;

typedef vector<fType> state_type;

matrixf AGlobal;
matrixf BGlobal;

void uF(const double t, state_type& u){
	u.resize(BGlobal.size2(), 0.0);
	u[0] = (fType)(sin(t));
	u[1] = (fType)(cos(t));
	u[2] = (fType)(sin(t));
	u[3] = (fType)(cos(t));
	//u[2] = (fType)(0.0);
}

void sys( const state_type & x, state_type & dxdt, const double t)
{
	state_type u;
	uF(t, u);
	dxdt.resize(AGlobal.size1());
	for(int row = 0; row < AGlobal.size1(); row++){
		fType fVal(0.0);
		for(int col = 0; col < AGlobal.size2(); col++)
			fVal += AGlobal(row, col) * x[col];
		for(int col = 0; col < BGlobal.size2(); col++)
			fVal += BGlobal(row, col) * u[col];
		dxdt[row] = fVal;
	}
}

bool TestRK4(matrixf& A,
	matrixf& B, 
	matrixf& C, 
	matrixf& D, 
	int uIndex,
	double tStart, 
	double tStep,
	int nSystemSize,
	int nInputs,
	int nOutputs,
	int nSteps,
	vectorf& x0){

	double tEnd( tStart + (nSteps - 1) * tStep  );
	vectorf tVect;
	matrixf xVect;

	SimulateODE(A, B, C, D, uIndex, tStart, tEnd, tStep, x0,
			RUNGE_KUTTA, tVect, xVect);

//	std::cout << tVect << std::endl;
//	std::cout << xVect << std::endl;

	boost::numeric::odeint::runge_kutta4< state_type > rk4;
	state_type x(A.size1(), 0.0);
	for(int row = 0; row < x.size(); row++)
		x[row] = x0(row);
	double tDisc(tStart);
	double fErr(0);

	for(int ii = 1; ii < nSteps; ii++){
		state_type y;
		rk4.do_step(sys, x, tDisc, tStep);
		state_type u;
		tDisc += tStep;
		uF(tDisc, u);
		y.resize(nOutputs, 0.0);
		// compute the output
		if(C.size1() != 0 && C.size2() != 0)
			for(int row = 0; row < nOutputs; row++){
				fType val = 0.0;
				for(int col = 0; col < nSystemSize; col++)
					val += C(row, col) * x[col];
				y[row] = val;
			}
		if(D.size1() != 0 && D.size2() != 0)
			for(int row = 0; row < nOutputs; row++){
				fType val = 0.0;
				for(int col = 0; col < nInputs; col++)
					val += D(row, col) * u[col];
				y[row] += val;
			}
		if(C.size1() == 0 && C.size2() == 0 && D.size1() == 0 && D.size2() == 0)
			y = x;
//		// print the output
//		std::cout << tDisc << " - ";
//		for(int row = 0; row < y.size(); row++)
//			std::cout << y[row] << ", ";
//		std::cout << std::endl;
		for(int jj = 0; jj < y.size(); jj++)
			fErr += (xVect(jj, ii) - y[jj]) * (xVect(jj, ii) - y[jj]);
	}
	std::cout << "Err = " << fErr << std::endl;
	return (sqrt(fErr) < 0.1);
}

bool TestABM(matrixf& A,
		matrixf& B,
		matrixf& C,
		matrixf& D,
		int uIndex,
		double tStart,
		double tStep,
		int nSystemSize,
		int nInputs,
		int nOutputs,
		int nSteps,
		vectorf& x0){

	double tEnd( tStart + (nSteps - 1) * tStep  );
	vectorf tVect;
	matrixf xVect;

	SimulateODE(A, B, C, D, uIndex, tStart, tEnd, tStep, x0, ADAMS_BASHFORTH_MOULTON, tVect, xVect);

//	std::cout << tVect << std::endl;
//	std::cout << xVect << std::endl;

	/*
	 * we have to implement a classic adams bashford moulton scheme with 4 terms memory
	 * the C++ Boost implementation uses the predictor from Adams Bashford as the starting step
	 * in Adams Moulton method
	 */

	boost::numeric::odeint::runge_kutta4< state_type > rk4;
	state_type x(nSystemSize, 0.0);
	for(int row = 0; row < x.size(); row++)
		x[row] = x0(row);
	double tDisc(tStart);
	double fErr(0);
	state_type y(nOutputs);
	state_type u;
	state_type dx(nSystemSize);
	vector<vectorf> dx_b(4);

	for(int ii = 0; ii < 4; ii++)
		dx_b[ii].resize(nSystemSize);

	sys(x, dx, tDisc);
	for(int jj = 0; jj < nSystemSize; jj++)
		dx_b[2](jj) = dx[jj];
	rk4.do_step(sys, x, tDisc, tStep);
	tDisc += tStep;
	sys(x, dx, tDisc);
	for(int jj = 0; jj < nSystemSize; jj++)
		dx_b[1](jj) = dx[jj];
	rk4.do_step(sys, x, tDisc, tStep);
	tDisc += tStep;
	sys(x, dx, tDisc);
	for(int jj = 0; jj < nSystemSize; jj++)
		dx_b[0](jj) = dx[jj];
	rk4.do_step(sys, x, tDisc, tStep);

	tDisc += tStep;
	// compute the output
	uF(tDisc, u);
	if(C.size1() != 0 && C.size2() != 0)
		for(int row = 0; row < nOutputs; row++){
			fType val = 0.0;
			for(int col = 0; col < nSystemSize; col++)
				val += C(row, col) * x[col];
			y[row] = val;
		}
	if(D.size1() != 0 && D.size2() != 0)
		for(int row = 0; row < nOutputs; row++){
			fType val = 0.0;
			for(int col = 0; col < nInputs; col++)
				val += D(row, col) * u[col];
			y[row] += val;
		}
	if(C.size1() == 0 && C.size2() == 0 && D.size1() == 0 && D.size2() == 0)
		y = x;
//	// print the output
//	std::cout << tDisc << " - ";
//	for(int row = 0; row < y.size(); row++)
//		std::cout << y[row] << ", ";
//	std::cout << std::endl;
	y.resize(nOutputs, 0.0);
	// Adams Bashford Moulton stuff
	vectorf predictor, dxpredictor, xc;
	dxpredictor.resize(nSystemSize);
	double ABCoefs[] = {55.0/24.0, -59.0/24.0, 37.0/24.0, -9.0/24.0};
	double AMCoefs[] = {9.0/24.0, 19.0/24.0, -5.0/24.0, 1.0/24.0};
	xc.resize(nSystemSize);
	for(int ii = 0; ii < nSystemSize; ii++)
		xc(ii) = x[ii];
	for(int ii = 1; ii < nSteps - 4; ii++){
		// Adams Bashford predictor
		// shift derivatives
		dx_b[3] = dx_b[2];
		dx_b[2] = dx_b[1];
		dx_b[1] = dx_b[0];
		// update x
		for(int jj = 0; jj < nSystemSize; jj++)
			x[jj] = xc(jj);
		sys(x, dx, tDisc);
		// retrieve new dx
		for(int jj = 0; jj < nSystemSize; jj++)
			dx_b[0](jj) = dx[jj];
		predictor = xc + tStep * (ABCoefs[0] * dx_b[0] +
								  ABCoefs[1] * dx_b[1] +
								  ABCoefs[2] * dx_b[2] +
								  ABCoefs[3] * dx_b[3]);
		// predictor deriv1ative
		tDisc += tStep;
		// update x
		for(int jj = 0; jj < nSystemSize; jj++)
			x[jj] = predictor(jj);
		sys(x, dx, tDisc);
		for(int jj = 0; jj < nSystemSize; jj++)
			dxpredictor(jj) = dx[jj];

		// Adams Moulton corector step
		xc = xc + tStep * (	AMCoefs[0] * dxpredictor +
							AMCoefs[1] * dx_b[0] +
							AMCoefs[2] * dx_b[1] +
							AMCoefs[3] * dx_b[2]);
		uF(tDisc, u);
		y.resize(nOutputs, 0.0);
		// compute the output
		if(C.size1() != 0 && C.size2() != 0)
			for(int row = 0; row < nOutputs; row++){
				fType val = 0.0;
				for(int col = 0; col < nSystemSize; col++)
					val += C(row, col) * xc[col];
				y[row] = val;
			}
		if(D.size1() != 0 && D.size2() != 0)
			for(int row = 0; row < nOutputs; row++){
				fType val = 0.0;
				for(int col = 0; col < nInputs; col++)
					val += D(row, col) * u[col];
				y[row] += val;
			}
		if(C.size1() == 0 && C.size2() == 0 && D.size1() == 0 && D.size2() == 0){
			y.resize(xc.size());
			for(int jj = 0; jj < y.size(); jj++)
				y[jj]= xc(jj);
		}
		// print the output
//		std::cout << tDisc << " - ";
//		for(int row = 0; row < y.size(); row++)
//			std::cout << y[row] << ", ";
//		std::cout << std::endl;
		for(int jj = 0; jj < y.size(); jj++)
			fErr += (xVect(jj, ii + 3) - y[jj]) * (xVect(jj, ii + 3) - y[jj]);
	}
	std::cout << "Err = " << fErr << std::endl;
	return (sqrt(fErr) < 0.1);
}

int main(){
  
	ErrorCode retVal;
	std::vector<unsigned int> GPUIDs;
	retVal = GetAvailableGPUs(GPUIDs);
	if(retVal != ParODE_OK || GPUIDs.size() == 0)
		return 0;
	std::vector<unsigned int> DevicesToInitialize;
	for(int ii = 0; ii < GPUIDs.size(); ii++){
		std::string strDeviceName;
    	retVal = GetDeviceName( GPUIDs[ii], strDeviceName);
		if(retVal != ParODE_OK)
			return 0;
		std::cerr << "Device " << ii << " " << strDeviceName << std::endl;
		if( strDeviceName == string("GeForce GTX 480") )
			DevicesToInitialize.push_back(GPUIDs[ii]);
	}
	if( DevicesToInitialize.size() == 0 ){
		std::cout << "No devices selected; Exiting\n";
		::getchar();
		return 0;
	}
	string strKernelFolder("/home/patriciu/Projects/ParallelODESolver/Software/Cpp/Kernels/");
	string strKernelIncludeFolder("/home/patriciu/Projects/ParallelODESolver/Software/Cpp/Include");
    retVal = InitializeSelectedGPUs(DevicesToInitialize, strKernelFolder, strKernelIncludeFolder);
	if(retVal != ParODE_OK) return 0;
	// register the input
//	string uFunc(	"fType UFunc(fType t, int nInputIndex){\n"
//				"fType fVal = nInputIndex == 0 ? (fType)(sin(t)) :\n"
//				"nInputIndex == 1 ? (fType)(cos(t)) :\n"
//				"0.0;\n"
//				"return fVal;\n"
//				"}\n"
//		);
	string uFunc(	"fType UFunc(fType t, int nInputIndex){\n"
				"fType fVal = nInputIndex == 0 ? (fType)(sin(t)) :\n"
				"nInputIndex == 1 ? (fType)(cos(t)) :\n"
				"nInputIndex == 2 ? (fType)(sin(t)) :\n"
				"nInputIndex == 3 ? (fType)(cos(t)) :\n"
				"0.0;\n"
				"return fVal;\n"
				"}\n"
		);
	int indexKernel = RegisterInput(uFunc);
	if(indexKernel < 0){
		std::cout << "TestFailed : Could not create the input kernel\n";
		::getchar();
		return 0;
	}
	// build the system matrices
	int nSystemSize(20);
	int nInputs(4);
	int nOutputs(4);
	matrixf A(nSystemSize, nSystemSize, 0.0);
	matrixf B(nSystemSize, nInputs, 1.0);
	matrixf C(nOutputs, nSystemSize, 0.0);
	matrixf D(nOutputs, nInputs, 0.0);

	A = -10.0 * boost::numeric::ublas::identity_matrix<fType>(nSystemSize);
	B(0, 0) = 1.0;
	B(1, 1) = 1.0;
	for(int row = 0; row < A.size1(); row++)
		for(int col = 0; col < A.size2(); col++)
			A(row, col) += ((fType)0.01) * (fType)((fType)rand() / (fType)RAND_MAX);
	//		B = boost::numeric::ublas::identity_matrix<fType>(nInputs);
	for(int row = 0; row < B.size1(); row++)
		for(int col = 0; col < B.size2(); col++)
			B(row, col) = (fType)((fType)rand() / (fType)RAND_MAX);
	for(int row = 0; row < C.size1(); row++)
		for(int col = 0; col < C.size2(); col++)
			C(row, col) = (fType)((fType)rand() / (fType)RAND_MAX);
	for(int row = 0; row < D.size1(); row++)
		for(int col = 0; col < D.size2(); col++)
			D(row, col) = (fType)((fType)rand() / (fType)RAND_MAX);
//	D = boost::numeric::ublas::identity_matrix<fType>(nSystemSize);

	std::cout << "A = " << A << std::endl;
	std::cout << "B = " << B << std::endl;
	std::cout << "C = " << C << std::endl;
	std::cout << "D = " << D << std::endl;

	AGlobal = A;
	BGlobal = B;

	double tStart(0.0);
	double tStep(0.1);
	unsigned int nSteps = 1000;
	vectorf x0(nSystemSize, 1.0);

	if(TestRK4(A, B, C, D, indexKernel,
			tStart, tStep, nSystemSize, nInputs, nOutputs, nSteps, x0))
		std::cout << "Test RK Passed\n";
	else
		std::cout << "Test RK Failed\n";

	if(TestABM(A, B, C, D, indexKernel,
			tStart, tStep, nSystemSize, nInputs, nOutputs, nSteps, x0))
		std::cout << "Test ABM Passed\n";
	else
		std::cout << "Test ABM Failed\n";

	CloseGPU();

	::getchar();
	return 0;
}

