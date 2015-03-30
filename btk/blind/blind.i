//
//                       Beamforming Toolkit
//                              (btk)
//
//  Module:  btk.modulated
//  Purpose: Cosine modulated analysis and synthesis filter banks.
//  Author:  John McDonough.
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


%module(package="btk") blind

%{
#include "stream/stream.h"
#include "stream/pyStream.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "blind/blind.h"
%}

%init {
  // NumPy needs to set up callback functions
  import_array();
}

typedef int size_t;

%include gsl/gsl_types.h
%include error.i
%include jexception.i
%include complex.i
%include matrix.i
%include vector.i

%rename(__str__) *::print;
%rename(__getitem__) *::operator[](int);
%ignore *::nextSampleBlock(const double* smp);

%import stream/stream.i

void blind();

void gradient(const gsl_matrix* Rss, const gsl_matrix* Ryy, gsl_matrix* W);

%rename(__str__) print;
%ignore *::print();
