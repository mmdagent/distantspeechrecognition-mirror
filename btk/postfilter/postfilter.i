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

%module(package="btk") postfilter

%{
#include "stream/stream.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "postfilter/postfilter.h"
#include "postfilter/spectralsubtraction.h"
#include "postfilter/binauralprocessing.h"
#include "stream/pyStream.h"
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


// ----- definition for class 'ZelinskiPostFilter' -----
// 
%ignore ZelinskiPostFilter;
class ZelinskiPostFilter: public VectorComplexFeatureStream {
public:
  ZelinskiPostFilter( VectorComplexFeatureStreamPtr &output, unsigned fftLen, double alpha=0.6, int type=2, in minFrames=0, const String& nm = "ZelinskPostFilter" );
  ~ZelinskiPostFilter();
  void setBeamformer( SubbandDSPtr &beamformer );
  void setSnapShotArray( SnapShotArrayPtr &snapShotArray );
  void setArrayManifoldVector( unsigned fbinX, gsl_vector_complex *arrayManifold, bool halfBandShift, unsigned NC = 1 );
  const gsl_vector_complex* getPostFilterWeights();

  virtual const gsl_vector_complex* next(int frameX = -5);

  virtual void reset();
};

class ZelinskiPostFilterPtr: public VectorComplexFeatureStreamPtr {
public:
  %extend {
    ZelinskiPostFilterPtr( VectorComplexFeatureStreamPtr &output, unsigned M, double alpha=0.6, int type=2, unsigned minFrames=0, const String& nm = "ZelinskPostFilter" ){
      return new ZelinskiPostFilterPtr(new ZelinskiPostFilter( output, M, alpha, type, minFrames, nm));
    }

    ZelinskiPostFilterPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  ZelinskiPostFilter* operator->();
};

// ----- definition for class 'McCowanPostFilter' -----
// 
%ignore McCowanPostFilter;
class McCowanPostFilter: public ZelinskiPostFilter {
public:
  McCowanPostFilter( VectorComplexFeatureStreamPtr &output, unsigned fftLen, double alpha=0.6, int type=2, in minFrames=0, float threshold=0.99, const String& nm = "McCowanPostFilterPtr" );
  ~McCowanPostFilter();

  virtual const gsl_vector_complex* next(int frameX = -5);

  virtual void reset();

  const gsl_matrix_complex *getNoiseSpatialSpectralMatrix( unsigned fbinX );
  bool setNoiseSpatialSpectralMatrix( unsigned fbinX, gsl_matrix_complex* Rnn );
  bool setDiffuseNoiseModel( const gsl_matrix* micPositions, double sampleRate, double sspeed = 343740.0 );
  void setAllLevelsOfDiagonalLoading( float diagonalWeight );
  void setLevelOfDiagonalLoading( unsigned fbinX, float diagonalWeight );
  void divideAllNonDiagonalElements( float myu );
  void divideNonDiagonalElements( unsigned fbinX, float myu );
};

class McCowanPostFilterPtr: public ZelinskiPostFilterPtr {
public:
  %extend {
    McCowanPostFilterPtr( VectorComplexFeatureStreamPtr &output, unsigned fftLen, double alpha=0.6, int type=2, unsigned minFrames=0, float threshold=0.99, const String& nm = "McCowanPostFilterPtr" ){
      return new McCowanPostFilterPtr(new McCowanPostFilter( output, fftLen, alpha, type, minFrames, threshold, nm));
    }
    
    McCowanPostFilterPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  McCowanPostFilter* operator->();
};

// ----- definition for class 'LefkimmiatisPostFilter' -----
// 
%ignore LefkimmiatisPostFilter;
class LefkimmiatisPostFilter: public McCowanPostFilter  {
public:
  LefkimmiatisPostFilter( VectorComplexFeatureStreamPtr &output, unsigned fftLen, double minSV=1.0E-8, unsigned fbinX1=0, double alpha=0.6, int type=2, in minFrames=0, float threshold=0.99, const String& nm = "LefkimmiatisPostFilterPtr" );
  ~LefkimmiatisPostFilter();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  void calcInverseNoiseSpatialSpectralMatrix();
};

class LefkimmiatisPostFilterPtr: public McCowanPostFilterPtr {
public:
  %extend {
    LefkimmiatisPostFilterPtr( VectorComplexFeatureStreamPtr &output, unsigned fftLen, double minSV=1.0E-8, unsigned fbinX1=0, double alpha=0.6, int type=2, unsigned minFrames=0, float threshold=0.99, const String& nm = "LefkimmiatisPostFilterPtr" ){
      return new LefkimmiatisPostFilterPtr(new LefkimmiatisPostFilter( output, fftLen, minSV, fbinX1, alpha, type, minFrames, threshold, nm));
    }
    
    LefkimmiatisPostFilterPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  LefkimmiatisPostFilter* operator->();
};

// ----- definition for class 'SpectralSubtractor' -----
// 
%ignore SpectralSubtractor;

class SpectralSubtractor : public VectorComplexFeatureStream {
 public:

  SpectralSubtractor(unsigned fftLen, bool halfBandShift = false, float ft=1.0, float flooringV=0.001, const String& nm = "SpectralSubtractor");
  ~SpectralSubtractor();
  void setChannel(VectorComplexFeatureStreamPtr& chan, double alpha=-1);
  
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  void setNoiseOverEstimationFactor(float ft);
  void startTraining();
  void stopTraining();
  void clearNoiseSamples();
  void clear();
  void startNoiseSubtraction();
  void stopNoiseSubtraction();
  bool readNoiseFile( const String& fn, unsigned idx=0 );
  bool writeNoiseFile( const String& fn, unsigned idx=0 );
};

class SpectralSubtractorPtr : public VectorComplexFeatureStreamPtr {
 public:
  %extend {
    SpectralSubtractorPtr(unsigned fftLen, bool halfBandShift = false, float ft=1.0, float flooringV=0.001, const String& nm = "SpectralSubtractor"){      
      return new SpectralSubtractorPtr(new SpectralSubtractor( fftLen, halfBandShift, ft, flooringV, nm));
    }

    SpectralSubtractorPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SpectralSubtractor* operator->();
};

// ----- definition for class 'WienerFilter' -----
// 
%ignore WienerFilter;
class WienerFilter : public VectorComplexFeatureStream {
 public:

  WienerFilter(VectorComplexFeatureStreamPtr &targetSignal, VectorComplexFeatureStreamPtr &noiseSignal, bool halfBandShift=false, float alpha=0.0, float flooringV=0.001, double beta=1.0, const String& nm = "WienerFilter");
  ~WienerFilter();
  
  void    setNoiseAmplificationFactor( double beta );
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  void    startUpdatingNoisePSD();
  void    stopUpdatingNoisePSD();
};

class WienerFilterPtr : public VectorComplexFeatureStreamPtr {
 public:
  %extend {
    WienerFilterPtr(VectorComplexFeatureStreamPtr &targetSignal, VectorComplexFeatureStreamPtr &noiseSignal, bool halfBandShift=false, float alpha=0.0, float flooringV=0.001, double beta=1.0, const String& nm = "WienerFilter"){
      return new WienerFilterPtr(new WienerFilter( targetSignal, noiseSignal, halfBandShift, alpha, flooringV, beta, nm));
    }

    WienerFilterPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  WienerFilter* operator->();
};

// ----- definition for class 'highPassFilter' -----
// 
%ignore highPassFilter;

class highPassFilter : public VectorComplexFeatureStream {
 public:

  highPassFilter( VectorComplexFeatureStreamPtr &output, float cutOffFreq = 150, int sampleRate, const String& nm = "highPassFilter" );
  ~highPassFilter();

  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
};

class highPassFilterPtr : public VectorComplexFeatureStreamPtr {
 public:
  %extend {
    highPassFilterPtr( VectorComplexFeatureStreamPtr &output, float cutOffFreq, int sampleRate, const String& nm = "highPassFilter" ){      
      return new highPassFilterPtr(new highPassFilter( output, cutOffFreq, sampleRate, nm));
    }

    highPassFilterPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  highPassFilter* operator->();
};

// ----- definition for class 'BinaryMaskFilter' -----
// 
%ignore BinaryMaskFilter;
class BinaryMaskFilter : public VectorComplexFeatureStream {
public:
  BinaryMaskFilter( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR,
		       unsigned M, float threshold, float alpha, 
		       float dEta = 0.01, const String& nm = "BinaryMaskFilter" );
  ~BinaryMaskFilter();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  void setThreshold( float threshold );
  void setThresholds( const gsl_vector *thresholds );
  double getThreshold();
  gsl_vector *getThresholds();
};

class BinaryMaskFilterPtr : public VectorComplexFeatureStreamPtr {
public:
  %extend {
    BinaryMaskFilterPtr( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, float threshold, float alpha, float dEta = 0.01, const String& nm = "BinaryMaskFilter" ){
      return new BinaryMaskFilterPtr(new BinaryMaskFilter(chanX, srcL, srcR, M, threshold, alpha, dEta, nm ));
    }
    BinaryMaskFilterPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  BinaryMaskFilter* operator->();
};

// ----- definition for class 'KimBinaryMaskFilter' -----
// 
%ignore KimBinaryMaskFilter;

class KimBinaryMaskFilter : public BinaryMaskFilter {
  public:
  KimBinaryMaskFilter( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M,float threshold, float alpha, float dEta = 0.01, float dPowerCoeff = 1/15, const String& nm = "KimBinaryMaskFilter" );
  ~KimBinaryMaskFilter();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  virtual const gsl_vector_complex* masking1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R, float threshold );
};

class KimBinaryMaskFilterPtr : public BinaryMaskFilterPtr {
public:
  %extend {
    KimBinaryMaskFilterPtr( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, float threshold, float alpha, float dEta = 0.01, float dPowerCoeff = 1/15, const String& nm = "KimBinaryMaskFilter" ){
      return new KimBinaryMaskFilterPtr(new KimBinaryMaskFilter( chanX, srcL, srcR, M, threshold, alpha, dEta,dPowerCoeff, nm ));
    }

    KimBinaryMaskFilterPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }
  
  KimBinaryMaskFilter* operator->();
};

// ----- definition for class 'KimITDThresholdEstimator' -----
// 
%ignore KimITDThresholdEstimator;

class KimITDThresholdEstimator : public KimBinaryMaskFilter {
public:
  KimITDThresholdEstimator( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, 
			    float minThreshold = 0, float maxThreshold = 0, float width = 0.02,
			    float minFreq= -1, float maxFreq=-1, int sampleRate=-1, float dEta = 0.01, float dPowerCoeff = 1/15, const String& nm = "KimITDThresholdEstimator" );
  ~KimITDThresholdEstimator();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  virtual double calcThreshold();
  const gsl_vector* getCostFunction();
};

class KimITDThresholdEstimatorPtr : public KimBinaryMaskFilterPtr {
public:
  %extend {
    KimITDThresholdEstimatorPtr( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, 
				 float minThreshold = 0, float maxThreshold = 0, float width = 0.02,
				 float minFreq= -1, float maxFreq=-1, int sampleRate=-1, float dEta = 0.01, float dPowerCoeff = 1/15, const String& nm = "KimITDThresholdEstimator" ){
      return new KimITDThresholdEstimatorPtr( new KimITDThresholdEstimator( srcL, srcR,M, minThreshold, maxThreshold, width, minFreq, maxFreq, sampleRate, dEta, dPowerCoeff, nm ) );
    }
    KimITDThresholdEstimatorPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }
  
  KimITDThresholdEstimator* operator->(); 
};

// ----- definition for class 'IIDBinaryMaskFilter' -----
// 
%ignore IIDBinaryMaskFilter;

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

class IIDBinaryMaskFilterPtr : public BinaryMaskFilterPtr {
public:
  %extend {
    IIDBinaryMaskFilterPtr( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR,
			    unsigned M, float threshold, float alpha, 
			    float dEta = 0.01, const String& nm = "IIDBinaryMaskFilter" ){
      return new IIDBinaryMaskFilterPtr( new IIDBinaryMaskFilter( chanX, srcL, srcR, M, threshold, alpha, dEta, nm ) );
    }
    IIDBinaryMaskFilterPtr __iter__() {
      (*self)->reset();  return *self;
    }	  
  }
  
  IIDBinaryMaskFilter* operator->();  	
};


// ----- definition for class 'IIDThresholdEstimator' -----
// 
%ignore IIDThresholdEstimator;

class IIDThresholdEstimator : public KimITDThresholdEstimator {
public:
  IIDThresholdEstimator( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, 
			 float minThreshold = 0, float maxThreshold = 0, float width = 0.02,
			 float minFreq= -1, float maxFreq=-1, int sampleRate=-1, float dEta = 0.01, float dPowerCoeff = 1/15, const String& nm = "IIDThresholdEstimator" );
  ~IIDThresholdEstimator();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  virtual double calcThreshold();
};

class IIDThresholdEstimatorPtr : public KimITDThresholdEstimatorPtr {
public:
  %extend {
    IIDThresholdEstimatorPtr( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, 
			      float minThreshold = 0, float maxThreshold = 0, float width = 0.02,
			      float minFreq= -1, float maxFreq=-1, int sampleRate=-1, float dEta = 0.01, float dPowerCoeff = 1/15, const String& nm = "IIDThresholdEstimator" ){
      return new IIDThresholdEstimatorPtr( new IIDThresholdEstimator( srcL, srcR, M, minThreshold, maxThreshold, width, minFreq, maxFreq, sampleRate, dEta, dPowerCoeff, nm ) );
    }
    
    IIDThresholdEstimatorPtr __iter__() {
      (*self)->reset();  return *self;
    }	  
  }
  
  IIDThresholdEstimator* operator->();  	
};

// ----- definition for class 'FDIIDThresholdEstimator' -----
// 
%ignore FDIIDThresholdEstimator;

class FDIIDThresholdEstimator : public BinaryMaskFilter {
public:
  FDIIDThresholdEstimator( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, float minThreshold = 0, float maxThreshold = 0, float width = 1000, float dEta = 0.01, float dPowerCoeff = 1/15, const String& nm = "FDIIDThresholdEstimator" );
  ~FDIIDThresholdEstimator();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void   reset();
  virtual double calcThreshold();
  const gsl_vector* getCostFunction( unsigned freqX );
};

class FDIIDThresholdEstimatorPtr : public BinaryMaskFilterPtr {
public:
  %extend {
    FDIIDThresholdEstimatorPtr( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, float minThreshold = 0, float maxThreshold = 0, float width = 1000, float dEta = 0.01, float dPowerCoeff = 1/15, const String& nm = "FDIIDThresholdEstimator" ){
      return new FDIIDThresholdEstimatorPtr( new FDIIDThresholdEstimator( srcL, srcR, M, minThreshold, maxThreshold, width, dEta, dPowerCoeff, nm) );
    }
    
    FDIIDThresholdEstimatorPtr __iter__() {
      (*self)->reset();  return *self;
    }	  
  }

  FDIIDThresholdEstimator* operator->(); 
};
