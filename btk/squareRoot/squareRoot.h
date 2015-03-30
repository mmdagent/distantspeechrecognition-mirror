//                              -*- C++ -*-
//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.squareRoot
//  Purpose: Updating real and complex Cholesky factorizations.
//  Author:  John McDonough
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or (at
// your option) any later version.
// 
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.


#ifndef _squareRoot_h_
#define _squareRoot_h_

#include <stdio.h>
#include <assert.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include "common/jexception.h"

void vectorMatrixProduct(const gsl_vector_complex* vec,
			 const gsl_matrix_complex* mat, gsl_matrix* D);

void makeConjugateSymmetric(gsl_matrix_complex* mat);

void choleskyForwardSub(const gsl_matrix* A, gsl_vector* x);

void choleskyBackSub(const gsl_matrix* A, gsl_vector* x);

void choleskyForwardSubComplex(const gsl_matrix_complex* lt,
			       const gsl_vector_complex* rhs,
			       gsl_vector_complex* lhs,
			       bool conjugate = false);

void choleskyBackSubComplex(const gsl_matrix_complex* lt,
			    const gsl_vector_complex* rhs,
			    gsl_vector_complex* lhs,
			    bool conjugate = false);

void rankOneUpdateCholeskyFactor(gsl_matrix_complex* A11,
				 const double alpha_m,
				 const gsl_vector_complex* c_m);

void propagateCovarSquareRootStep1(gsl_matrix* A12,
				   gsl_matrix* A22);

void propagateCovarSquareRootStep2a(gsl_matrix* A11,
				    gsl_matrix* A12,
				    gsl_matrix* A21,
				    gsl_matrix* A22);

void propagateCovarSquareRootStep2b(gsl_matrix* A22);

void propagateCovarSquareRoot(gsl_matrix* A11,
			      gsl_matrix* A12,
			      gsl_matrix* A21,
			      gsl_matrix* A22);

void propagateCovarSquareRootReal(gsl_matrix* A11,
				  gsl_matrix* A12,
				  gsl_matrix* A21,
				  gsl_matrix* A22, bool flag = false);

void sweepLowerTriangular(gsl_matrix* A, gsl_matrix* B);

void propagateCovarSquareRootStep1(gsl_matrix_complex* A12,
				   gsl_matrix_complex* A22);

void propagateCovarSquareRootStep2a(gsl_matrix_complex* A11,
				    gsl_matrix_complex* A12,
				    gsl_matrix_complex* A21,
				    gsl_matrix_complex* A22);

void propagateCovarSquareRootStep2b(gsl_matrix_complex* A22);

void propagateCovarSquareRoot(gsl_matrix_complex* A11,
			      gsl_matrix_complex* A12,
			      gsl_matrix_complex* A21,
			      gsl_matrix_complex* A22);

void propagateInfoSquareRoot(gsl_matrix_complex* sqrt_Pm_inv,
			     gsl_matrix_complex* A12,
			     gsl_vector_complex* a_21,
			     gsl_vector_complex* a_22, bool rankOneA12 = true);

void propagateInfoSquareRootStep2_RLS(gsl_matrix_complex* sqrt_Pm_inv,
				      gsl_vector_complex* a_12,
				      gsl_vector_complex* a_21,
				      gsl_complex a_22);

void propagateInfoSquareRoot_RLS(gsl_matrix_complex* sqrt_Pm_inv,
				 gsl_vector_complex* a_12,
				 gsl_vector_complex* a_21,
				 gsl_complex a_22);

void addDiagonalLoading(gsl_matrix_complex* sqrt_Pm_inv, int dim, double wght);

gsl_vector* choleskyGetDiagonal(gsl_vector* v, const gsl_matrix* m);
gsl_vector* getSquareDiagonal(gsl_vector* v, const gsl_matrix* m);

#endif
