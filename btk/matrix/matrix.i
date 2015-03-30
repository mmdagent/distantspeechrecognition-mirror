//                              -*- C++ -*-
//
//                               Millenium
//                  Automatic Speech Recognition System
//                                 (asr)
//
//  Module:  btk.matrix
//  Purpose: Wrapper for GSL matrix objects.
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

%module(package="btk") matrix

%{
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_matrix_float.h>
#include "matrix/gslmatrix.h"
%}

#ifdef AUTODOC
%section "Matrix", before
#endif

%include typedefs.i
%include jexception.i

#ifndef INLINE_DECL
#define INLINE_DECL extern inline
#endif

%include <gsl/gsl_matrix_double.h>
%include <gsl/gsl_matrix_float.h>

%extend gsl_matrix {
  gsl_matrix(unsigned m, unsigned n) {
    return gsl_matrix_alloc(m, n);
  }

  ~gsl_matrix() {
    gsl_matrix_free(self);
  }

  unsigned nrows() const {
    return self->size1;
  }

  unsigned ncols() const {
    return self->size2;
  }

  float __getitem__(int m, int n) {
    return gsl_matrix_get(self, m, n);
  }

  void __setitem__(float item, int n, int m) {
    gsl_matrix_set(self, m, n, item);
  }
}

%extend gsl_matrix_float {
  gsl_matrix_float(unsigned m, unsigned n) {
    return gsl_matrix_float_alloc(m, n);
  }

  ~gsl_matrix_float() {
      // gsl_matrix_float_free(self);
  }

  unsigned nrows() const {
    return self->size1;
  }

  unsigned ncols() const {
    return self->size2;
  }

  float __getitem__(int m, int n) {
    return gsl_matrix_float_get(self, m, n);
  }

  void __setitem__(float item, int n, int m) {
    gsl_matrix_float_set(self, m, n, item);
  }
}

gsl_matrix_float* gsl_matrix_float_load(gsl_matrix_float* m, const char* filename, bool old = false);

gsl_matrix_float* gsl_matrix_float_resize(gsl_matrix_float* m, size_t size1, size_t size2);

