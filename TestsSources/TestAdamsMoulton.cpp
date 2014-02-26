#define __TESTING_KERNELS__
#include <ParODE.h>
#include <GPUManagement.h>
#include <AdamsMoulton.h>
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
	//std::cout << "-----------------\n" << t << std::endl;
	//std::cout << AGlobal;
	//std::cout << BGlobal;
	//std::cout << "x = " << x[0] << ", " << x[1] << std::endl;
	//std::cout << "u = " << u[0] << ", " << u[1] << std::endl;
	//std::cout << "dxdt = " << dxdt[0] << ", " << dxdt[1] << std::endl;
}

class TestAM : public AdamsMoulton{
public:
	TestAM(	GPUManagement* GPUM, vector<KernelWrapper*>& vectGPUP, vector<OpenCLKernel*>& vectUKernel):
		AdamsMoulton(GPUM, vectGPUP, vectUKernel){

	};
	~TestAM(){};
	bool TestAdamsMoulton(){
		int nSystemSize(2);
		int nInputs(2);
		int nOutputs(2);
		matrixf A(nSystemSize, nSystemSize, 0.0);
		matrixf B(nSystemSize, nInputs, 0.0);
		matrixf C(nOutputs, nSystemSize);
		matrixf D(nOutputs, nInputs);

		A = -5.0 * boost::numeric::ublas::identity_matrix<fType>(nSystemSize);
		for(int row = 0; row < A.size1(); row++)
			for(int col = 0; col < A.size2(); col++)
				A(row, col) += 2.0 * (fType)((fType)rand() / (fType)RAND_MAX);
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

		std::cout << "A = " << A << std::endl;
		std::cout << "B = " << B << std::endl;
		std::cout << "C = " << C << std::endl;
		std::cout << "D = " << D << std::endl;

		AGlobal = A;
		BGlobal = B;

		matrixf XRes;
		double tStart(0.0);
		double tStep(0.1);
		unsigned int nSteps = 50;
		double tEnd = (nSteps - 1) * tStep + tStart;

		vectorf x0(nSystemSize, 10.0);
		vectorf tVect;
		matrixf xVect;

		SimulateODE(A, B, C, D, tStart, tEnd, tStep, x0, tVect, xVect);

		std::cout << tVect << std::endl;
		std::cout << xVect << std::endl;

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
		// print the output
		std::cout << tDisc << " - ";
		for(int row = 0; row < y.size(); row++)
			std::cout << y[row] << ", ";
		std::cout << std::endl;
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
			std::cout << tDisc << " - ";
			for(int row = 0; row < y.size(); row++)
				std::cout << y[row] << ", ";
			std::cout << std::endl;
			for(int jj = 0; jj < y.size(); jj++)
				fErr += (xVect(jj, ii + 3) - y[jj]) * (xVect(jj, ii + 3) - y[jj]);
		}
		std::cout << "Err = " << fErr << std::endl;
		return (sqrt(fErr) < 0.1);
	};
};

int main(){
	string strKernelFolder("/home/patriciu/Projects/ParallelODESolver/Software/Cpp/Kernels/");
	string strKernelIncludeFolder("/home/patriciu/Projects/ParallelODESolver/Software/Cpp/Include");
	std::vector<unsigned int> vectIds;
	if( ParODE::Instance()->GetAvailableGPUs(vectIds) != ParODE_OK){
		std::cerr << "Test Failed; Could not initialize the GPU\n";
		::getchar();
		return 0;
	}
	std::vector<unsigned int> vectSelectedGPUs;
	std::cout << "We will use the following devices:\n";
	for(int ii = 0; ii < vectIds.size(); ii++){
		string strDeviceName;
		ParODE::Instance()->GetDeviceName(vectIds[ii], strDeviceName);
		if( strDeviceName == string("GeForce GTX 480") ){
			vectSelectedGPUs.push_back(vectIds[ii]);
			std::cout << "Device " << vectIds[ii] << " : " << strDeviceName << std::endl;
		}
	}
	if( vectSelectedGPUs.size() == 0 ){
		std::cout << "No devices selected; Exiting\n";
		::getchar();
		return 0;
	}
	if(ParODE::Instance()->InitializeSelectedGPUs(vectSelectedGPUs, strKernelFolder, strKernelIncludeFolder) != ParODE_OK){
		std::cerr << "Test Failed; Could not initialize the GPU\n";
		::getchar();
		return 0;
	}

//	string uFunc(	"fType UFunc(fType t, int nInputIndex){\n"
//					"fType fVal = nInputIndex == 0 ? 0.0 :\n"
//					"((nInputIndex == 1) ? 0.0 :\n"
//					"((nInputIndex == 2) ? 0.0 :\n"
//					"0.0));\n"
//					"return fVal;\n"
//					"}\n"
//				);
		string uFunc(	"fType UFunc(fType t, int nInputIndex){\n"
						"fType fVal = nInputIndex == 0 ? sin(t) :\n"
						"nInputIndex == 1 ? cos(t) :\n"
						"0.0;\n"
						"return fVal;\n"
						"}\n"
				);
	int indexKernel = ParODE::Instance()->RegisterUKernel(uFunc);
	if(indexKernel < 0){
		std::cout << "TestFailed : Could not create the input kernel\n";
		::getchar();
		return 0;
	}

	TestAM* pTestAM = new TestAM(	ParODE::Instance()->GetGPUManagement(),
									ParODE::Instance()->GetKernelsvectorReference(),
									ParODE::Instance()->GetUKernelsVectorReference(indexKernel) );

	if(pTestAM->TestAdamsMoulton())
		std::cout << "Test Succeeded\n";
	else
		std::cout << "Test Failed\n";

	delete pTestAM;

	exit_label:
	::getchar();

	return 0;
}
