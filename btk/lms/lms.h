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

#ifndef _lms_h_
#define _lms_h_

#include "convolution/convolution.h"


// ----- definition for class `FastBlockLMSFeature' -----
//
class FastBlockLMSFeature : public VectorFloatFeatureStream {
 public:
  FastBlockLMSFeature(VectorFloatFeatureStreamPtr& desired, VectorFloatFeatureStreamPtr& samp, float alpha, float gamma,
		      const String& nm = "Fast Block LMS Feature");
  virtual ~FastBlockLMSFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { _samp->reset(); VectorFloatFeatureStream::reset(); }

  void update();

 private:
  void _halfComplexPack(double*  tgt, const gsl_vector_complex* src);
  void _halfComplexUnpack(gsl_vector_complex* tgt, const double* src);

  VectorFloatFeatureStreamPtr			_desired;
  VectorFloatFeatureStreamPtr			_samp;
  OverlapSavePtr				_overlapSave;

  const unsigned				_N;
  const unsigned				_M;

  float						_alpha;
  float						_gamma;

  double*					_e;
  double*					_u;
  double*					_phi;

  gsl_vector_complex*				_U;
  gsl_vector_complex*				_E;
  gsl_vector_complex*				_Phi;

  gsl_vector*					_D;
};

typedef Inherit<FastBlockLMSFeature, VectorFloatFeatureStreamPtr> FastBlockLMSFeaturePtr;


#endif
