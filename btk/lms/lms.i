//                              -*- C++ -*-
//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.lms
//  Purpose: Implementation of LMS algorithms.
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


%module(package="btk") lms

%{
#include "stream/stream.h"
#include "stream/pyStream.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "lms/lms.h"
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

%rename(__str__) *::print;
%rename(__getitem__) *::operator[](int);
%ignore *::nextSampleBlock(const double* smp);

%pythoncode %{
import btk
from btk import stream
oldimport = """
%}
%import stream/stream.i
%pythoncode %{
"""
%}

// ----- definition for class `FastBlockLMSFeature' -----
// 
%ignore FastBlockLMSFeature;
class FastBlockLMSFeature : public VectorFloatFeatureStream {
public:
  FastBlockLMSFeature(VectorFloatFeatureStreamPtr& desired, VectorFloatFeatureStreamPtr& samp, float alpha = 0.0001, float gamma = 0.98,
		      const String& nm = "Fast Block LMS Feature");

  const gsl_vector_float* next() const;
};

class FastBlockLMSFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    FastBlockLMSFeaturePtr(VectorFloatFeatureStreamPtr& desired, VectorFloatFeatureStreamPtr& samp, float alpha = 0.0001, float gamma = 0.98,
			   const String& nm = "Fast Block LMS Feature") {
      return new FastBlockLMSFeaturePtr(new FastBlockLMSFeature(desired, samp, alpha, gamma, nm));
    }

    FastBlockLMSFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  FastBlockLMSFeature* operator->();
};


%rename(__str__) print;
%ignore *::print();
