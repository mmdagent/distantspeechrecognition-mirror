#ifndef _spectralsubtraction_
#define _spectralsubtraction_

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
//#include "stream/pyStream.h"
#include "modulated/modulated.h"

class PSDEstimator {

 public:
  PSDEstimator(unsigned fftLen2);
  ~PSDEstimator();

  bool readEstimates( const String& fn );
  bool writeEstimates( const String& fn );

  const gsl_vector* getEstimate(){
    return (const gsl_vector*)_estimates;
  }

 protected:
  gsl_vector*_estimates; /* estimated noise PSD */
};

typedef refcount_ptr<PSDEstimator> PSDEstimatorPtr;

/**
   @class estimate the noise PSD for spectral subtraction (SS)
   1. construct an object 
   2. add a noise sample
   3. get noise estimates
 */
class averagePSDEstimator : public PSDEstimator {
 public:
  /** 
      @brief A constructor for SS
      @param unsigned fftLen2[in] the half of the FFT point
      @param double alpha[in] the forgetting factor for recursive averaging. 
                              If this is negative, the average is used
                              as the noise PSD estimate.
   */
  averagePSDEstimator(unsigned fftLen2, double alpha = -1.0 );
  ~averagePSDEstimator();

  void clearSamples();
  const gsl_vector* average();
  bool addSample( const gsl_vector_complex *sample );
  void clear();

 protected:
  double _alpha;
  bool _isSampleAdded;
  list<gsl_vector *> _sampleL;
};

typedef Inherit<averagePSDEstimator, PSDEstimatorPtr> averagePSDEstimatorPtr;

class SpectralSubtractor : public VectorComplexFeatureStream {
public:
  SpectralSubtractor(unsigned fftLen, bool halfBandShift = false, float ft=1.0, float flooringV=0.001, const String& nm = "SpectralSubtractor");
  ~SpectralSubtractor();
	 
  void setNoiseOverEstimationFactor(float ft){
    _ft = ft;
  }
  
  void setChannel(VectorComplexFeatureStreamPtr& chan, double alpha=-1 );
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  
  void startTraining(){
    _isTrainingStarted = true;
  }
  
  void stopTraining(){
    _isTrainingStarted = false;
    for(_NoisePSDIterator itr = _noisePSDList.begin(); itr != _noisePSDList.end(); itr++)
      (*itr)->average();
  }
  
  void clearNoiseSamples(){
    for(_NoisePSDIterator itr = _noisePSDList.begin(); itr != _noisePSDList.end(); itr++)
      (*itr)->clearSamples();
  }

  void clear(){
    for(_NoisePSDIterator itr = _noisePSDList.begin(); itr != _noisePSDList.end(); itr++)
      (*itr)->clear();
  }
	
  void startNoiseSubtraction(){
    _startNoiseSubtraction = true;
  }
  
  void stopNoiseSubtraction(){
    _startNoiseSubtraction = false;
  }
  
  bool readNoiseFile( const String& fn, unsigned idx=0 ){
    _isTrainingStarted = false;
    return _noisePSDList.at(idx)->readEstimates( fn );
  }

  bool writeNoiseFile( const String& fn, unsigned idx=0 ){
    return _noisePSDList.at(idx)->writeEstimates( fn );
  }

 protected:
  typedef list<VectorComplexFeatureStreamPtr>	_ChannelList;
  typedef _ChannelList::iterator			_ChannelIterator;
  typedef vector<averagePSDEstimatorPtr>		_NoisePSDList;
  typedef _NoisePSDList::iterator	        	_NoisePSDIterator;
  
  _ChannelList		_channelList;
  _NoisePSDList         _noisePSDList;
  unsigned		_fftLen;
  unsigned		_fftLen2;
  bool			_halfBandShift;
  bool                  _isTrainingStarted;
  unsigned              _totalTrainingSampleN;
  float			_ft;
  float			_flooringV;
  bool                  _startNoiseSubtraction;
};

typedef Inherit<SpectralSubtractor, VectorComplexFeatureStreamPtr> SpectralSubtractorPtr;

// ----- definition for class 'WienerFilter' -----
// 
class WienerFilter : public VectorComplexFeatureStream {
 public:
  WienerFilter( VectorComplexFeatureStreamPtr &targetSignal, VectorComplexFeatureStreamPtr &noiseSignal, bool halfBandShift=false, float alpha=0.0, float flooringV=0.001, double beta=1.0, const String& nm = "WienerFilter");
  ~WienerFilter();
  void    setNoiseAmplificationFactor( double beta ){ _beta = beta; }
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  void    startUpdatingNoisePSD(){ _updateNoisePSD = true; }
  void    stopUpdatingNoisePSD(){ _updateNoisePSD = false; }

 private:
  gsl_vector *_prevPSDs;
  gsl_vector *_prevPSDn;
  VectorComplexFeatureStreamPtr _targetSignal;
  VectorComplexFeatureStreamPtr _noiseSignal;
  unsigned		_fftLen;
  unsigned		_fftLen2;
  float			_alpha; // forgetting factor
  float			_flooringV;
  bool			_halfBandShift;
  float                 _beta; // amplification coefficient for a noise signal
  bool                  _updateNoisePSD;
};

typedef Inherit<WienerFilter, VectorComplexFeatureStreamPtr> WienerFilterPtr;

#endif
