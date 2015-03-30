//                              -*- C++ -*-
//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.sad
//  Purpose: Voice activity detection.
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


#ifndef _sadFeature_h_
#define _sadFeature_h_

#include "stream/stream.h"
#include "common/mlist.h"


// ----- definition for class `BrightnessFeature' -----
//
class BrightnessFeature : public VectorFloatFeatureStream {
 public:
  BrightnessFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, const String& nm = "Brightness");
  virtual ~BrightnessFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { _src->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			_src;
  bool						_weight;
  float						_sampleRate;
  float						_max;
  float						_med;
  float						_df;
  float*					_frs;
};

typedef Inherit<BrightnessFeature, VectorFloatFeatureStreamPtr> BrightnessFeaturePtr;


// ----- definition for class `EnergyDiffusionFeature' -----
//
class EnergyDiffusionFeature : public VectorFloatFeatureStream {
 public:
  EnergyDiffusionFeature(const VectorFloatFeatureStreamPtr& src, const String& nm = "EnergyDiffusion");
  virtual ~EnergyDiffusionFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { _src->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			_src;
};

typedef Inherit<EnergyDiffusionFeature, VectorFloatFeatureStreamPtr> EnergyDiffusionFeaturePtr;


// ----- definition for class `BandEnergyRatioFeature' -----
//
class BandEnergyRatioFeature : public VectorFloatFeatureStream {
 public:
  BandEnergyRatioFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, float threshF = 0.0, const String& nm = "Band Energy Ratio");
  virtual ~BandEnergyRatioFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { _src->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			_src;
  float						_sampleRate;
  float						_max;
  float						_df;
  float						_threshF;
  int						_threshX;
};

typedef Inherit<BandEnergyRatioFeature, VectorFloatFeatureStreamPtr> BandEnergyRatioFeaturePtr;


// ----- definition for class `NormalizedFluxFeature' -----
//
class NormalizedFluxFeature : public VectorFloatFeatureStream {
 public:
  NormalizedFluxFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, float threshF = 0.0, const String& nm = "Band Energy Ratio");
  virtual ~NormalizedFluxFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { _src->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			_src;
  float*					_win0;
  float*					_win1;
};

typedef Inherit<NormalizedFluxFeature, VectorFloatFeatureStreamPtr> NormalizedFluxFeaturePtr;


// ----- definition for class `NegativeEntropyFeature' -----
//
class NegativeEntropyFeature : public VectorFloatFeatureStream {
 public:
  NegativeEntropyFeature(const VectorFloatFeatureStreamPtr& src, const String& nm = "Negative Entropy");
  virtual ~NegativeEntropyFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { _src->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			_src;
  float*					_win;
};

typedef Inherit<NegativeEntropyFeature, VectorFloatFeatureStreamPtr> NegativeEntropyFeaturePtr;


// ----- definition for class `SignificantSubbandsFeature' -----
//
class SignificantSubbandsFeature : public VectorFloatFeatureStream {
 public:
  SignificantSubbandsFeature(const VectorFloatFeatureStreamPtr& src, float thresh = 0.0, const String& nm = "Band Energy Ratio");
  virtual ~SignificantSubbandsFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { _src->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			_src;
  float						_thresh;
  float*					_win;
};

typedef Inherit<SignificantSubbandsFeature, VectorFloatFeatureStreamPtr> SignificantSubbandsFeaturePtr;


// ----- definition for class `NormalizedBandwidthFeature' -----
//
class NormalizedBandwidthFeature : public VectorFloatFeatureStream {
 public:
  NormalizedBandwidthFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, float thresh = 0.0, const String& nm = "Band Energy Ratio");
  virtual ~NormalizedBandwidthFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { _src->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			_src;
  float						_thresh;
  float						_sampleRate;
  float						_df;
  float*					_frs;
  float*					_win;
};

typedef Inherit<NormalizedBandwidthFeature, VectorFloatFeatureStreamPtr> NormalizedBandwidthFeaturePtr;

#endif

