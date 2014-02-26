/*
 * ParODEException.cpp
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#include <ParODEException.h>

ParODEException::ParODEException() throw(){
}

ParODEException::~ParODEException() throw(){
}

const char* ParODEException::what() const throw(){
	return "ParODE Exception!!!\n";
}
