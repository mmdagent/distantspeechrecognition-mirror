//                           -*- C++ -*-
//
//                       Beamforming Toolkit
//                              (btk)
//
//  Module:  btk.TDEstimator
//  Purpose: 
//  Author:  
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

%module(package="btk") TDEstimator

%{
#include "TDEstimator/CCTDE.h"
#include "stream/stream.h"
#include "feature/feature.h"
#include <numpy/arrayobject.h>
#include "stream/pyStream.h"
#include "modulated/prototypeDesign.h"
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

%pythoncode %{
import btk
from btk import stream
oldimport = """
%}
%import stream/stream.i
%pythoncode %{
"""
%}

// ----- definition for class `CCTDE' -----
// 
%ignore CCTDE;
class CCTDE : public VectorFeatureStream {
 public:
  CCTDE( SampleFeaturePtr& samp1, SampleFeaturePtr& samp2, unsigned nHeldMaxCC=1, int freqLowerLimit=-1, int freqUpperLimit=-1, const String& nm= "CCTDE" );
  ~CCTDE();
  
  void setTargetFrequencyRange( int freqLowerLimit, int freqUpperLimit );
  void  allsamples( int fftLen=-1 );
  virtual const gsl_vector* next(int frameX = -5);
  virtual void  reset();

  const unsigned *getSampleDelays(){return (const unsigned *)_sampleDelays;}
  const double *getCCValues(){return (const double *)_ccValues;}
};


class CCTDEPtr : public VectorFeatureStreamPtr {
 public:
  %extend {
    CCTDE( SampleFeaturePtr& samp1, SampleFeaturePtr& samp2, bool isRTProcessing=false, unsigned nHeldMaxCC=1, int freqLowerLimit=-1, int freqUpperLimit=-1, const String& nm= "CCTDE" ){
      return new CCTDEPtr(new CCTDE(SampleFeaturePtr& samp1, SampleFeaturePtr& samp2, bool isRTProcessing, unsigned nHeldMaxCC, int freqLowerLimit, int freqUpperLimit, const String& nm));
    }

    CCTDEPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  CCTDE* operator->();
};

