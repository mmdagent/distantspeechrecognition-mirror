//                              -*- C++ -*-
//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.convolution
//  Purpose: Block convolution realization of an LTI system with the FFT.
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


#ifndef _convolution_h_
#define _convolution_h_

#include <stdio.h>
#include <assert.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include "common/jexception.h"

#include "stream/stream.h"


// ----- definition for class `OverlapAdd' -----
//
class OverlapAdd : public VectorFloatFeatureStream {
 public:
  OverlapAdd(VectorFloatFeatureStreamPtr& samp,
	     const gsl_vector* impulseResponse = NULL, unsigned fftLen = 0,
	     const String& nm = "OverlapAdd");

  ~OverlapAdd();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

 private:
  void _setImpulseResponse(const gsl_vector* impulseResponse);
  unsigned _checkFFTLen(unsigned sectionLen, unsigned irLen, unsigned fftLen);
  void _halfComplexUnpack(gsl_vector_complex* tgt, const double* src);

  const VectorFloatFeatureStreamPtr			_samp;
  const unsigned					_L;
  const unsigned					_P;
  const unsigned					_N;
  const unsigned					_N2;

  double*						_section;
  gsl_vector_complex*					_frequencyResponse;
  gsl_vector_float*					_buffer;
};

typedef Inherit<OverlapAdd, VectorFloatFeatureStreamPtr> OverlapAddPtr;


// ----- definition for class `OverlapSave' -----
//
class OverlapSave : public VectorFloatFeatureStream {
 public:
  OverlapSave(VectorFloatFeatureStreamPtr& samp,
	      const gsl_vector* impulseResponse = NULL, const String& nm = "OverlapSave");

  ~OverlapSave();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

  void update(const gsl_vector_complex* delta);

 private:
  void _setImpulseResponse(const gsl_vector* impulseResponse);
  unsigned _checkOutputSize(unsigned irLen, unsigned sampLen);
  unsigned _checkL(unsigned irLen, unsigned sampLen);
  void _halfComplexUnpack(gsl_vector_complex* tgt, const double* src);

  const VectorFloatFeatureStreamPtr			_samp;
  const unsigned					_L;
  const unsigned					_L2;
  const unsigned					_P;

  double*						_section;
  gsl_vector_complex*					_frequencyResponse;
};

typedef Inherit<OverlapSave, VectorFloatFeatureStreamPtr> OverlapSavePtr;


#endif // _convolution_h_

