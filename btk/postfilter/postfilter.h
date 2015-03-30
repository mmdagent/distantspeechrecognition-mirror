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

/**
   @file postfilter.h

   @brief implementation of post-filters for a microphone array. 

   The following post-filters are implemented:
   [1] Zelinski post-filter
   [2] APAB post-filter
   [3] McCowan's post-filter
   [4] Lefkimmiatis's post-filter

   The correspondig references are:
   [1] C.Claude Marro et al. "Analysis of noise reduction and dereverberation techniques based on microphone arrays with postfiltering", IEEE Trans. ASP, vol. 6, pp 240-259, May 1998.
   [2] M.Brandstein, "Microphone Arrays", Springer, ISBN 3-540-41953-5, pp.39-60.
   [3] Iain A. Mccowan et al., "Microphone array post-filter based on noise field coherence", IEEE Trans. SAP, vol. 11, pp. Nov. 709--716, 2003.
   [4] Stamatios Lefkimmiatis et al., "A generalized estimation approach for linear and nonlinear microphone array post-filters",  Speech Communication, 2007.
*/
#ifndef _postfilter_
#define _postfilter_

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
#include "postfilter/binauralprocessing.h"
#include "beamformer/spectralinfoarray.h"
#include "beamformer/beamformer.h"

typedef enum {
  TYPE_ZELINSKI1_REAL = 0x01,
  TYPE_ZELINSKI1_ABS  = 0x02,
  TYPE_APAB = 0x04,
  TYPE_ZELINSKI2 = 0x08,
  NO_USE_POST_FILTER = 0x00
} PostfilterType;

void ZelinskiFilter(gsl_vector_complex **arrayManifold,
		    SnapShotArrayPtr     snapShotArray, 
		    bool halfBandShift, 
		    gsl_vector_complex *beamformedSignal,
		    gsl_vector_complex **prevCSDs, 
		    gsl_vector_complex *pfweights,
		    double alpha, int Ropt );

void ApabFilter( gsl_vector_complex **arrayManifold,
		 SnapShotArrayPtr     snapShotArray, 
		 int fftLen, int nChan, bool halfBandShift,
		 gsl_vector_complex *beamformedSignal,
		 int channelX );

/**
   @class Zelinski post-filtering

   @brief filter beamformer's outputs under the assumption that noise signals between sensors are uncorrelated. 
   @usage
   1. construct an object,
   2. set the array snapshot and array manifold vectors with 
      either setBeamformer() or setSnapShotArray() and setArrayManifold(), 
   3. process data at each frame by caling next().
*/
class ZelinskiPostFilter: public VectorComplexFeatureStream {
public:
  ZelinskiPostFilter( VectorComplexFeatureStreamPtr &output, unsigned fftLen, double alpha=0.6, int type=2, int minFrames=0, const String& nm = "ZelinskPostFilter" );
  ~ZelinskiPostFilter();
  
  void setBeamformer( SubbandDSPtr &beamformer );
  void setSnapShotArray( SnapShotArrayPtr &snapShotArray );
  void setArrayManifoldVector( unsigned fbinX, gsl_vector_complex *arrayManifoldVector, bool halfBandShift, unsigned NC = 1 );
  const gsl_vector_complex* getPostFilterWeights()
  {
    if( NULL == _bfWeightsPtr ){
      return NULL;
    }
    return(_bfWeightsPtr->wp1());
  }

  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();

protected:
  PostfilterType _type; /* the type of the Zelinski-postfilters */
  double _alpha; /* forgetting factor */
  VectorComplexFeatureStreamPtr _samp; /* output of the beamformer */
  int _minFrames;
  SubbandDSPtr _beamformerPtr; /* */
  bool _hasBeamformerPtr; /* true if _beamformerPtr is set with setBeamformer() */
  beamformerWeights* _bfWeightsPtr;
  SnapShotArrayPtr _snapShotArray; /* multi-channel input */
  unsigned                                       _fftLen;
};

typedef Inherit<ZelinskiPostFilter, VectorComplexFeatureStreamPtr> ZelinskiPostFilterPtr;

/**
   @class McCowan post-filtering

   @brief process the beamformer's outputs with McCowan's post-filtering
   @usage
   1. construct an object,
   2. compute the noise coherence matrix through setDiffuseNoiseModel( micPositions, ssampleRate)
   3. set the array snapshot and array manifold vectors with 
      either setBeamformer() or setSnapShotArray() and setArrayManifold(), 
   4. process data at each frame by caling next().
*/
class McCowanPostFilter: public ZelinskiPostFilter {
public:
  McCowanPostFilter( VectorComplexFeatureStreamPtr &output, unsigned fftLen, double alpha=0.6, int type=2, int minFrames=0, float threshold=0.99, const String& nm = "McCowanPostFilter" );
  ~McCowanPostFilter();

  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  /* micPositions[][x,y,z] */

  const gsl_matrix_complex *getNoiseSpatialSpectralMatrix( unsigned fbinX );
  bool setNoiseSpatialSpectralMatrix( unsigned fbinX, gsl_matrix_complex* Rnn );
  bool setDiffuseNoiseModel( const gsl_matrix* micPositions, double sampleRate, double sspeed = 343740.0 );
  void setAllLevelsOfDiagonalLoading( float diagonalWeight );
  void setLevelOfDiagonalLoading( unsigned fbinX, float diagonalWeight );
  void divideAllNonDiagonalElements( float myu );
  void divideNonDiagonalElements( unsigned fbinX, float myu );

protected:
  double estimateAverageOfCleanSignalPSD( unsigned fbinX, gsl_vector_complex* currCSDf );
  virtual void PostFiltering();

protected:
  gsl_matrix_complex**                           _R; /* Noise spatial spectral matrices */
  float*                                         _diagonalWeights;
  float                                          _thresholdOfRij; /* to avoid the indeterminate solution*/
  gsl_vector_complex*                            _timeAlignedSignalf; /* workspace */
  bool                                           _isInvRcomputed; 
};

typedef Inherit<McCowanPostFilter, ZelinskiPostFilterPtr> McCowanPostFilterPtr;

/**
   @class Lefkimmiatis post-filtering
   @brief compute a Winer filter under the the diffuse noise field assumption
   @usage
   1. construct an object,
   2. compute the noise coherence matrix through setDiffuseNoiseModel( micPositions, ssampleRate)
   3. set the array snapshot and array manifold vectors with 
      either setBeamformer() or setSnapShotArray() and setArrayManifold(), 
   4. process data at each frame by caling next().
*/
class LefkimmiatisPostFilter: public McCowanPostFilter {
public:
  LefkimmiatisPostFilter( VectorComplexFeatureStreamPtr &output, unsigned fftLen, double minSV=1.0E-8, unsigned fbinX1=0, double alpha=0.6, int type=2, int minFrames=0, float threshold=0.99, const String& nm = "LefkimmiatisPostFilte" );
  ~LefkimmiatisPostFilter();

  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  /* micPositions[][x,y,z] */
  void calcInverseNoiseSpatialSpectralMatrix();

protected:
  double estimateAverageOfNoiseSignalPSD( unsigned fbinX, gsl_vector_complex* currCSDf );
  virtual void PostFiltering();
  
private:
  gsl_complex calcLambda( unsigned fbinX );
  
private:
  gsl_matrix_complex** _invR;
  gsl_vector_complex*  _tmpH;
  double              _minSV;
  unsigned            _fbinX1;
};

typedef Inherit<LefkimmiatisPostFilter, McCowanPostFilterPtr> LefkimmiatisPostFilterPtr;

/**
   @class high pass filter
*/
class highPassFilter: public VectorComplexFeatureStream {
public:
  highPassFilter( VectorComplexFeatureStreamPtr &output, float cutOffFreq, int sampleRate, const String& nm = "highPassFilter" );
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
private:
  unsigned _cutOfffreqBin;
  float    _cutOffFreq;
  VectorComplexFeatureStreamPtr _src;
};

typedef Inherit<highPassFilter, VectorComplexFeatureStreamPtr> highPassFilterPtr;

#endif

