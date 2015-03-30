//                              -*- C++ -*-
//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.convolution
//  Purpose: Block convolution realization of an LTI system with the FFT.
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


%module(package="btk") convolution

%{
#include "stream/stream.h"
#include "stream/pyStream.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "convolution/convolution.h"
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

// ----- definition for class `OverlapAdd' -----
// 
%ignore OverlapAdd;
class OverlapAdd : public VectorFloatFeatureStream {
 public:
  OverlapAdd(VectorFloatFeatureStreamPtr& samp, const gsl_vector* impulseResponse, unsigned fftLen = 0,
	     const String& nm = "Overlap Add");

  ~OverlapAdd();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();
};

class OverlapAddPtr : public VectorFloatFeatureStreamPtr {
public:
  %extend {
    OverlapAddPtr(VectorFloatFeatureStreamPtr& samp, const gsl_vector* impulseResponse, unsigned fftLen = 0,
		  const String& nm = "Overlap Add") {
      return new OverlapAddPtr(new OverlapAdd(samp, impulseResponse, fftLen, nm));
    }
    
    OverlapAddPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  OverlapAdd* operator->();
};


// ----- definition for class `OverlapSave' -----
// 
%ignore OverlapSave;
class OverlapSave : public VectorFloatFeatureStream {
 public:
  OverlapSave(VectorFloatFeatureStreamPtr& samp,
	      const gsl_vector* impulseResponse, const String& nm = "Overlap Save");

  ~OverlapSave();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();
};

class OverlapSavePtr : public VectorFloatFeatureStreamPtr {
public:
  %extend {
    OverlapSavePtr(VectorFloatFeatureStreamPtr& samp,
		   const gsl_vector* impulseResponse, const String& nm = "Overlap Save") {
      return new OverlapSavePtr(new OverlapSave(samp, impulseResponse, nm));
    }
    
    OverlapSavePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  OverlapSave* operator->();
};


%rename(__str__) print;
%ignore *::print();
