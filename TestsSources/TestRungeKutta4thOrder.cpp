#define __TESTING_KERNELS__
#include <ParODE.h>
#include <GPUManagement.h>
#include <RungeKutta4thOrder.h>
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

class TestRK : public RungeKutta4thOrder{
public:
	TestRK(	GPUManagement* GPUM, vector<KernelWrapper*>& vectGPUP, vector<OpenCLKernel*>& vectUKernel):
		RungeKutta4thOrder(GPUM, vectGPUP, vectUKernel){

	};
	~TestRK(){};
	bool TestRungeKutta(){
		int nSystemSize(2);
		int nInputs(2);
		int nOutputs(2);
		matrixf A(nSystemSize, nSystemSize, 0.0);
		matrixf B(nSystemSize, nInputs, 0.0);
		matrixf C(nOutputs, nSystemSize, 0.0);
		matrixf D(nOutputs, nInputs, 0.0);

		A = -5.0 * boost::numeric::ublas::identity_matrix<fType>(nSystemSize);
		for(int row = 0; row < A.size1(); row++)
			for(int col = 0; col < A.size2(); col++)
				A(row, col) += (fType)((fType)rand() / (fType)RAND_MAX);
		 //B = boost::numeric::ublas::identity_matrix<fType>(2);
		for(int row = 0; row < B.size1(); row++)
			for(int col = 0; col < B.size2(); col++)
				B(row, col) = (fType)((fType)rand() / (fType)RAND_MAX);
//		C = boost::numeric::ublas::identity_matrix<fType>(nSystemSize);
		for(int row = 0; row < C.size1(); row++)
			for(int col = 0; col < C.size2(); col++)
				C(row, col) = (fType)((fType)rand() / (fType)RAND_MAX);
//		D = boost::numeric::ublas::identity_matrix<fType>(nSystemSize);
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

		boost::numeric::odeint::runge_kutta4< state_type > rk4;
		state_type x(nSystemSize, 0.0);
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
			// print the output
			std::cout << tDisc << " - ";
			for(int row = 0; row < y.size(); row++)
				std::cout << y[row] << ", ";
			std::cout << std::endl;
			for(int jj = 0; jj < y.size(); jj++)
				fErr += (xVect(jj, ii) - y[jj]) * (xVect(jj, ii) - y[jj]);
		}
		std::cout << "Err = " << fErr << std::endl;
		return (sqrt(fErr) < 0.1);
	};
};

int main(){
	string strKernelFolder("/home/alexandru/Projects/ParallelODESolver/Software/Cpp/Kernels/");
	string strKernelIncludeFolder("/home/alexandru/Projects/ParallelODESolver/Software/Cpp/Include");
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
		if( strDeviceName == string("BeaverCreek") ){
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

////	string uFunc(	"fType UFunc(fType t, int nInputIndex){\n"
////					"fType fVal = nInputIndex == 0 ? 0.0 :\n"
////					"((nInputIndex == 1) ? 0.0 :\n"
////					"((nInputIndex == 2) ? 0.0 :\n"
////					"0.0));\n"
////					"return fVal;\n"
////					"}\n"
////				);
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

	TestRK* pTestRK = new TestRK(	ParODE::Instance()->GetGPUManagement(),
									ParODE::Instance()->GetKernelsvectorReference(),
									ParODE::Instance()->GetUKernelsVectorReference(indexKernel) );

	if(pTestRK->TestRungeKutta())
		std::cout << "Test Succeeded\n";
	else
		std::cout << "Test Failed\n";

	exit_label:
	::getchar();

	return 0;
}
