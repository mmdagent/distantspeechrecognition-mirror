//                              -*- C++ -*-
//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.postfilter
//  Purpose: 
//  Author:  ABC
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

#ifndef _binauralprocessing_
#define _binauralprocessing_

#include <stdio.h>
#include <assert.h>
#include <float.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_fft_complex.h>
#include <common/refcount.h>
#include "common/jexception.h"

#include "stream/stream.h"
#include "postfilter/spectralsubtraction.h"
#include "beamformer/spectralinfoarray.h"
#include "beamformer/beamformer.h"

class BinaryMaskFilter : public VectorComplexFeatureStream {
public:
  BinaryMaskFilter( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR,
		       unsigned M, float threshold, float alpha, 
		       float dEta = 0.01, const String& nm = "BinaryMaskFilter" );
  ~BinaryMaskFilter();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  void setThreshold( float threshold ){ _threshold = threshold;}
  void setThresholds( const gsl_vector *thresholds );
  double getThreshold(){return _threshold;}
  gsl_vector *getThresholds(){return _thresholdAtFreq;}

protected:
  unsigned          _chanX;  /* want to extract the index of a channel */
  gsl_vector_float *_prevMu; /* binary mask at a previous frame */
  float             _alpha;  /* forgetting factor */
  float             _dEta;        /* flooring value */
  float             _threshold;   /* threshold for the ITD */
  gsl_vector        *_thresholdAtFreq;
  VectorComplexFeatureStreamPtr _srcL; /* left channel */
  VectorComplexFeatureStreamPtr _srcR; /* right channel */
};

typedef Inherit<BinaryMaskFilter, VectorComplexFeatureStreamPtr> BinaryMaskFilterPtr;

/**
   @class Implementation of binary masking based on C. Kim's Interspeech2010 paper 
   @brief binary mask two inputs based of the threshold of the interaural time delay (ITD)
   @usage
 */
class KimBinaryMaskFilter : public BinaryMaskFilter {
public:
  KimBinaryMaskFilter( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR,
		       unsigned M, float threshold, float alpha, 
		       float dEta = 0.01, float dPowerCoeff = 1/15, const String& nm = "KimBinaryMaskFilter" );
  ~KimBinaryMaskFilter();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();

  virtual const gsl_vector_complex* masking1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R, float threshold );

protected:
  float             _dPowerCoeff; /* power law non-linearity */
};

typedef Inherit<KimBinaryMaskFilter, BinaryMaskFilterPtr> KimBinaryMaskFilterPtr;

/**
   @class Implementation of estimating the threshold for C. Kim's ITD-based binary masking
   @brief binary mask two inputs based of the threshold of the interaural time delay (ITD)
   @usage
 */
class KimITDThresholdEstimator : public KimBinaryMaskFilter {
public:
  KimITDThresholdEstimator( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, float minThreshold = 0, float maxThreshold = 0, float width = 0.02, float minFreq= -1, float maxFreq=-1, int sampleRate=-1, float dEta = 0.01, float dPowerCoeff = 1/15, const String& nm = "KimITDThresholdEstimator" );
  ~KimITDThresholdEstimator();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void   reset();
  virtual double calcThreshold();
  const gsl_vector* getCostFunction();
 
protected:
  virtual void accumStats1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R );

protected:
  /* for restricting the search space */
  float _minThreshold;
  float _maxThreshold;
  float _width;
  unsigned _minFbinX;
  unsigned _maxFbinX;
  /* work space */
  double *_costFunctionValues;
  double *_sigma_T;
  double *_sigma_I;
  double *_mean_T;
  double *_mean_I;
  unsigned int _nCand;
  unsigned int _nSamples;
  gsl_vector *_buffer; /* for returning values from the python */
  bool _isCostFunctionComputed;
};

typedef Inherit<KimITDThresholdEstimator, KimBinaryMaskFilterPtr> KimITDThresholdEstimatorPtr;

/**
	@class
 */
class IIDBinaryMaskFilter : public BinaryMaskFilter {
public:
	IIDBinaryMaskFilter( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR,
						unsigned M, float threshold, float alpha, 
						float dEta = 0.01, const String& nm = "IIDBinaryMaskFilter" );
	~IIDBinaryMaskFilter();
	virtual const gsl_vector_complex* next(int frameX = -5);
	virtual void reset();
	
	virtual const gsl_vector_complex* masking1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R, float threshold );
};

typedef Inherit<IIDBinaryMaskFilter, BinaryMaskFilterPtr> IIDBinaryMaskFilterPtr;

/**
 @class binary masking based on a difference between magnitudes of two beamformers' outputs.
 @brief set zero to beamformer's output with the smaller magnitude over frequency bins
 */
class IIDThresholdEstimator : public KimITDThresholdEstimator {
public:
	IIDThresholdEstimator( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, 
							 float minThreshold = 0, float maxThreshold = 0, float width = 0.02,
							 float minFreq= -1, float maxFreq=-1, int sampleRate=-1, float dEta = 0.01, float dPowerCoeff = 0.5, const String& nm = "IIDThresholdEstimator" );
	~IIDThresholdEstimator();
	virtual const gsl_vector_complex* next(int frameX = -5);
	virtual void reset();
	virtual double calcThreshold();

protected:
	virtual void accumStats1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R );
private:
        double *_Y4_T;
        double *_Y4_I;
        double _beta;
};

typedef Inherit<IIDThresholdEstimator, KimITDThresholdEstimatorPtr> IIDThresholdEstimatorPtr;

/**
 @class binary masking based on a difference between magnitudes of two beamformers' outputs at each frequency bin.
 @brief set zero to beamformer's output with the smaller magnitude at each frequency bin
 */
class FDIIDThresholdEstimator : public BinaryMaskFilter {
public:
  FDIIDThresholdEstimator( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, 
			   float minThreshold = 0, float maxThreshold = 0, float width = 1000, 
			   float dEta = 0.01, float dPowerCoeff = 1/15, const String& nm = "FDIIDThresholdEstimator" );
  ~FDIIDThresholdEstimator();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void   reset();
  virtual double calcThreshold();
  const gsl_vector* getCostFunction( unsigned freqX );
 
protected:
  virtual void accumStats1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R );

protected:
  /* for restricting the search space */
  float _minThreshold;
  float _maxThreshold;
  float _width;
  float _dPowerCoeff;

  /* work space */
  double **_costFunctionValues;
  double **_Y4;
  double **_sigma;
  double **_mean;
  double _beta;
  unsigned int _nCand;
  unsigned int _nSamples;
  gsl_vector *_buffer; /* for returning values from the python */
  bool _isCostFunctionComputed;
};

typedef Inherit<FDIIDThresholdEstimator, BinaryMaskFilterPtr> FDIIDThresholdEstimatorPtr;

#endif
