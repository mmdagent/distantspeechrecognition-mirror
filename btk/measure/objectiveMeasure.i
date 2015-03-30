//                           -*- C++ -*-
//
//                       Beamforming Toolkit
//                              (btk)
//
//  Module:  btk.beamforming
//  Purpose: Beamforming in the subband domain.
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

%module(package="btk") beamformer
%{
#include "measure/objectiveMeasure.h"
#include <numpy/arrayobject.h>
#include "stream/pyStream.h"
%}

%init {
  // NumPy needs to set up callback functions
  import_array();
}

//typedef int size_t;

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

%import objectiveMeasure.h

// ----- definition for class `SNR' -----
//
%ignore SNR;
class SNR {
 public:
  float getSNR( const String& fn1, const String& fn2, int normalizationOption, int chX=1, int samplerate=16000, int cfrom=-1, int to=-1 );
  float getSNR2( gsl_vector_float *original, gsl_vector_float *enhanced, int normalizationOption);
};

class SNRPtr {
 public:
  %extend {
    SNRPtr() {
      return new SNRPtr(new SNR );
    }
  }

  SNR* operator->();
};

// ----- definition for class `ItakuraSaitoMeasurePS' -----
//
%ignore ItakuraSaitoMeasurePS;
class ItakuraSaitoMeasurePS {
 public:
  ItakuraSaitoMeasurePS( unsigned fftLen,  unsigned r = 1, unsigned windowType = 1, const String& nm = "ItakuraSaitoMeasurePS" );
  float getDistance( const String& fn1, const String& fn2, int chX=1, int samplerate=16000, int bframe =0, int eframe = -1 );
  int frameShiftLength();
};

class ItakuraSaitoMeasurePSPtr {
 public:
  %extend {
    ItakuraSaitoMeasurePSPtr( unsigned fftLen,  unsigned r = 1, unsigned windowType = 1, const String& nm = "ItakuraSaitoMeasurePS" ) {
      return new ItakuraSaitoMeasurePSPtr(new ItakuraSaitoMeasurePS(fftLen,  r, windowType, nm ) );
    }
  }

  ItakuraSaitoMeasurePS* operator->();
};
