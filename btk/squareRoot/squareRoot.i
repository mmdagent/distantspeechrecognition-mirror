//
//                       Beamforming Toolkit
//                              (btk)
//
//  Module:  btk.squareRoot
//  Purpose: Updating complex Cholesky factorizations.
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


%module(package="btk") squareRoot

%{
#include "stream/stream.h"
#include "stream/pyStream.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "squareRoot.h"
%}

%init {
  // NumPy needs to set up callback functions
  import_array();
}

typedef int size_t;

%include gsl/gsl_types.h
%include jexception.i
%include complex.i
%include matrix.i
%include vector.i

void vectorMatrixProduct(const gsl_vector_complex* vec,
			 const gsl_matrix_complex* mat, gsl_matrix* D);

void makeConjugateSymmetric(gsl_matrix_complex* mat);

void choleskyForwardSub(const gsl_matrix* A, gsl_vector* x);

void choleskyBackSub(const gsl_matrix* A, gsl_vector* x);

// given 'sqrt_D', 'alpha_m' and 'c_m', calculate 'sqrt_Dm'
// through a series of Givens rotations
void rankOneUpdateCholeskyFactor(gsl_matrix_complex* sqrt_D,
                                 const double alpha_m,
                                 const gsl_vector_complex* c_m);

// perform forward substitution for a lower triangular matrix 'lt'
void choleskyForwardSubComplex(const gsl_matrix_complex* lt,
			       const gsl_vector_complex* rhs,
			       gsl_vector_complex* lhs,
			       bool conjugate = 0);

// perform back substitution for a lower triangular matrix 'lt'
void choleskyBackSubComplex(const gsl_matrix_complex* lt,
                            const gsl_vector_complex* rhs,
                            gsl_vector_complex* lhs,
                            bool conjugate = 0);

void propagateCovarSquareRootReal(gsl_matrix* sqrt_D,
				  gsl_matrix* A_12,
				  gsl_matrix* Gm_sqrt_Dm,
				  gsl_matrix* sqrt_Pm, bool flag = false);

void sweepLowerTriangular(gsl_matrix* A, gsl_matrix* B);

void propagateCovarSquareRootStep1(gsl_matrix_complex* A_12,
                                   gsl_matrix_complex* sqrt_Pm);

void propagateCovarSquareRootStep2a(gsl_matrix_complex* sqrt_Dm,
                                    gsl_matrix_complex* A_12,
                                    gsl_matrix_complex* Gm_sqrt_Dm,
                                    gsl_matrix_complex* sqrt_Pm);

void propagateCovarSquareRootStep2b(gsl_matrix_complex* sqrt_Pm);

void propagateCovarSquareRoot(gsl_matrix_complex* sqrt_D,
			      gsl_matrix_complex* A_12,
			      gsl_matrix_complex* Gm_sqrt_Dm,
			      gsl_matrix_complex* sqrt_Pm);

void propagateInfoSquareRoot(gsl_matrix_complex* sqrt_Pm_inv,
			     gsl_matrix_complex* A_12,
			     gsl_vector_complex* a_21,
			     gsl_vector_complex* a_22,
                             bool rankOneA12 = 1);

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
