//                              -*- C++ -*-
//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.maximumLikelihood
//  Purpose: Estimation of maximum likelihood beamforming parameters.
//  Author:  John McDonough and Ulrich Klee
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


%module(package="btk") maximumLikelihood

%{
#include "maximumLikelihood/maximumLikelihood.h"
#include "stream/stream.h"
#include "stream/pyStream.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
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

