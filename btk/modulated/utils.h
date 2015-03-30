#ifndef _UTILS_H_
#define _UTILS_H_

//#include "debug_output.h"

#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_complex.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <string.h>		// memcpy
#include <assert.h>


/*******************************************************************/
/* Convert a gsl_vector (double) to a double-array of		   */
/* the same length. The space for the double-array		   */
/* is allocated on the heap with new and should be released	   */
/* with delete[].						   */
/* 								   */
/* Param: gsl_vector to convert					   */
/*******************************************************************/
double* dvector2darray(gsl_vector const * const);


/************************************************************/
/* calculates the convolution of two vectors                */
/* 							    */
/* Param: b  convolution vector of length |b|      	    */
/*        x  input vector of length |x|			    */
/* 							    */
/* Return: y = b*x of length |x|+|b|-1			    */
/*                 delay = |b| -1                           */
/************************************************************/
gsl_vector*         filter_FIR(gsl_vector* b,         gsl_vector* x);
gsl_vector_complex* filter_FIR(gsl_vector* b,         gsl_vector_complex* x);
gsl_vector_complex* filter_FIR(gsl_vector_complex* b, gsl_vector_complex* x);


/****************************************************************/
/* multiplies a real valued vector with a complex scalar        */
/* and returns the result				        */
/* 							        */
/* Param: vector      real valued vector		        */
/*        b           complex value			        */
/* 							        */
/* Return: y = b*vector					        */
/****************************************************************/
gsl_vector_complex* gsl_vector_mul_complex(gsl_vector const *const vector, gsl_complex const b);


/**********************************************************************/
/* adds two complex vectors and stores the result in the first	      */
/* one								      */
/* 								      */
/* Param: a     1st source and destination vector		      */
/*        b     2nd source vector				      */
/* 								      */
/* Result: a = a+b						      */
/**********************************************************************/
void gsl_vector_add(gsl_vector_complex* a, gsl_vector_complex const* const b);


/***************************************************************************/
/* given a gsl_vector_short this creates a complex vector of it and	   */
/* returns it								   */
/* 									   */
/* Param: a   short vector						   */
/* 									   */
/* Result: complex vector with contents of a				   */
/***************************************************************************/
gsl_vector_complex* gsl_vector_short2complex(gsl_vector_short const * const a);

void gsl_vector_complex_REAL(gsl_vector* dest, gsl_vector_complex const * const src);

#endif
