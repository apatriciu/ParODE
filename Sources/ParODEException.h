/*
 * ParODEException.h
 *
 *  Created on: 2013-11-12
 *      Author: patriciu
 */

#ifndef ParODEException_H_
#define ParODEException_H_

#include <exception>

class ParODEException: public std::exception {
public:
	ParODEException() throw();
	virtual ~ParODEException();
	virtual const char* what() const throw();
};

#endif /* ParODEException_H_ */
