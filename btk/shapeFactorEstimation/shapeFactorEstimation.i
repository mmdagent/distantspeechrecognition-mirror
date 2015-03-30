//                              -*- C++ -*-
//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.modulated
//  Purpose: Estimation of shape factors for the generalized Gaussian pdf.
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


%module(package="btk") shapeFactorEstimation

%{
#include "stream/stream.h"
#include "stream/pyStream.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "shapeFactorEstimation/shapeFactorEstimation.h"
%}

%init {
  // NumPy needs to set up callback functions
  import_array();
}

typedef int size_t;

%include gsl/gsl_types.h

%include typedefs.i
%include jexception.i
%include complex.i
%include matrix.i
%include vector.i

%rename(__str__) *::print;
%rename(__getitem__) *::operator[](int);

%pythoncode %{
oldimport = """
%}
%pythoncode %{
"""
%}

// ----- definition for class `ShapeFactorFeatures' -----
//
%ignore ShapeFactorFeatures;
class ShapeFactorFeatures {
 public:
  ShapeFactorFeatures(unsigned classesN, unsigned maxFrames = 500);
  ~ShapeFactorFeatures();

  void insert(unsigned idx, const gsl_vector_complex* samps, const gsl_vector* vars);
  void write(const String& fileName) const;
  void read(const String& fileName);
  unsigned featuresN(unsigned classX) const;
  void add(const ShapeFactorFeatures& fromFeatures);
  unsigned classN() const;
  void clear();

  double logLhood(unsigned classX, unsigned subbandX, double f) const;
};

class ShapeFactorFeaturesPtr {
 public:
  %extend {
    ShapeFactorFeaturesPtr(unsigned classesN, unsigned maxFrames = 500) {
      return new ShapeFactorFeaturesPtr(new ShapeFactorFeatures(classesN, maxFrames));
    }
  }

  ShapeFactorFeatures* operator->();
};

// ----- definition for class `ShapeFactors' -----
//
%ignore ShapeFactors;
class ShapeFactors {
 public:
  ShapeFactors(const ShapeFactorFeaturesPtr& features = NULL);

  void estimate(unsigned classX);
  void read(const String& fileName);
  void write(const String& fileName) const;
  void writeParam(const String& fileName, unsigned classX, unsigned subX) const;
  void clear();
};

class ShapeFactorsPtr {
 public:
  %extend {
    ShapeFactorsPtr(const ShapeFactorFeaturesPtr& features = NULL) {
      return new ShapeFactorsPtr(new ShapeFactors(features));
    }
  }

  ShapeFactors* operator->();
};

%rename(__str__) print;
%ignore *::print();
