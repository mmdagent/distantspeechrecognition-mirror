//                              -*- C++ -*-
//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.sad
//  Purpose: Voice activity detection.
//  Author:  ABC and John McDonough
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

#ifndef _sad_h_
#define _sad_h_

#include <stdio.h>
#include <assert.h>
#define _LOG_SAD_ 
#ifdef _LOG_SAD_ 
#include <stdarg.h>
#endif /* _LOG_SAD_ */

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_fit.h>

#include "common/jexception.h"
#include "stream/stream.h"
#include "feature/feature.h"
#include "sad/neural_spnsp_incl.h"


// ----- definition for abstract base class `NeuralNetVAD' -----
//
class NeuralNetVAD {
public:
  NeuralNetVAD(VectorFloatFeatureStreamPtr& cep,
				  unsigned context = 4, unsigned hiddenUnitsN = 1000, unsigned outputUnitsN = 2, float threshold = 0.1, const String& neuralNetFile = "");
  ~NeuralNetVAD();

  bool next(int frameX = -5);

  void reset();

  void read(const String& neuralNetFile);

private:
  void _shiftDown();
  void _increment() { _frameX++; }
  void _updateBuffer(int frameX);

  const VectorFloatFeatureStreamPtr			_cep;
  const int						FrameResetX;
  const unsigned					_cepLen;
  bool							_isSpeech;
  int							_frameX;
  unsigned						_framesPadded;
  unsigned						_context;
  unsigned						_hiddenUnitsN;
  unsigned						_outputUnitsN;
  float							_threshold;
  MLP*							_mlp;
  float**						_frame;
};

typedef refcount_ptr<NeuralNetVAD> NeuralNetVADPtr;


// ----- definition for abstract base class `VAD' -----
//
class VAD {
public:
  ~VAD();

  virtual bool next(int frameX = -5) = 0;

  virtual void reset() { _frameX = FrameResetX; }

  virtual void nextSpeaker() = 0;

  const gsl_vector_complex* frame() const { return _frame; }

 protected:
  VAD(VectorComplexFeatureStreamPtr& samp);

  void _increment() { _frameX++; }

  const VectorComplexFeatureStreamPtr			_samp;
  const int						FrameResetX;
  const unsigned					_fftLen;
  bool							_isSpeech;
  int							_frameX;
  gsl_vector_complex*					_frame;
};

typedef refcount_ptr<VAD> VADPtr;


// ----- definition for class `EnergyVAD' -----
//
class SimpleEnergyVAD : public VAD {
 public:
  SimpleEnergyVAD(VectorComplexFeatureStreamPtr& samp,
			       double threshold, double gamma = 0.995);
  ~SimpleEnergyVAD();

  virtual bool next(int frameX = -5);

  virtual void reset();

  virtual void nextSpeaker();

 private:
  const double						_threshold;
  const double						_gamma;
  double						_spectralEnergy;
};

typedef Inherit<SimpleEnergyVAD, VADPtr> SimpleEnergyVADPtr;


// ----- definition for class `SimpleLikelihoodRatioVAD' -----
//
class SimpleLikelihoodRatioVAD : public VAD {
 public:
  SimpleLikelihoodRatioVAD(VectorComplexFeatureStreamPtr& samp,
					double threshold, double alpha);
  ~SimpleLikelihoodRatioVAD();

  virtual bool next(int frameX = -5);

  virtual void reset();

  virtual void nextSpeaker();

  void setVariance(const gsl_vector* variance);

 private:
  double _calcAk(double vk, double gammak, double Rk);

  bool							_varianceSet;
  gsl_vector*						_noiseVariance;
  gsl_vector*						_prevAk;
  gsl_vector_complex*					_prevFrame;
  double						_threshold;
  double						_alpha;
};

typedef Inherit<SimpleLikelihoodRatioVAD, VADPtr> SimpleLikelihoodRatioVADPtr;

// ----- definition for class `EnergyVADFeature' -----
//
class EnergyVADFeature : public VectorFloatFeatureStream {
 public:
  EnergyVADFeature(const VectorFloatFeatureStreamPtr& source, double threshold = 0.5, unsigned bufferLength = 30, unsigned energiesN = 200, const String& nm = "Energy VAD");
  virtual ~EnergyVADFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

  void nextSpeaker();

 private:
  static int _comparator(const void* elem1, const void* elem2);
  virtual bool _aboveThreshold(const gsl_vector_float* vector);

  VectorFloatFeatureStreamPtr			_source;
  bool						_recognizing;

  gsl_vector_float**				_buffer;
  const unsigned				_bufferLength;
  int						_bufferIndex;
  unsigned					_bufferedN;
  
  unsigned					_aboveThresholdN;
  unsigned					_belowThresholdN;

  unsigned					_energiesN;
  double*					_energies;
  double*					_sortedEnergies;
  const unsigned				_medianIndex;
};

typedef Inherit<EnergyVADFeature, VectorFloatFeatureStreamPtr> EnergyVADFeaturePtr;


// ----- definition for abstract base class `VADMetric' -----
//
class VADMetric :  public Countable {
public:
  virtual ~VADMetric() {
#ifdef  _LOG_SAD_ 
    closeLogFile();
#endif /* _LOG_SAD_ */
  }

  virtual double next(int frameX = -5) = 0;

  virtual void reset() = 0;

  virtual void nextSpeaker() = 0;

  double score(){ return _curScore;}

protected:
  double                                        _curScore;

#ifdef  _LOG_SAD_ 
public:
  bool openLogFile( const String & logfilename );
  int  writeLog( const char *format, ... );
  void closeLogFile();
  void initScore(){ _score=0.0; _scoreX=0; }
  void setScore( double score){ _score=score; _scoreX++; }
  double getAverageScore(){ if(_scoreX==0){return 0;} return(_score/_scoreX); }
  
private:
  FILE *_logfp;
  double _score;
  unsigned _scoreX;
public:
  int   _frameX;
#endif /* _LOG_SAD_ */
protected:
  VADMetric() {
#ifdef  _LOG_SAD_ 
    _logfp = NULL;
#endif /* _LOG_SAD_ */
  }
};

#ifdef  _LOG_SAD_ 
bool VADMetric::openLogFile( const String & logfilename )
{
  if( NULL != _logfp ){
    printf("closing the previous log file\n");
    fclose(_logfp);
  }
  _logfp = fopen( logfilename.c_str(), "w" );
}

int VADMetric::writeLog( const char *format, ... )
{
  if( NULL != _logfp ){
    int ret;
    va_list args;

    va_start(args, format);
    ret = vfprintf( _logfp, format, args);
    va_end (args);
    
    return ret;
  }
  return 0;
}

void VADMetric::closeLogFile()
{
  if( NULL != _logfp )
    fclose(_logfp);
}
#endif /* _LOG_SAD_ */

typedef refcountable_ptr<VADMetric> VADMetricPtr;


// ----- definition for class `EnergyVADMetric' -----
//
class EnergyVADMetric : public VADMetric {
public:
  EnergyVADMetric(const VectorFloatFeatureStreamPtr& source, double initialEnergy = 5.0e+07, double threshold = 0.5, unsigned headN = 4,
		  unsigned tailN = 10, unsigned energiesN = 200, const String& nm = "Energy VAD Metric");
  
  ~EnergyVADMetric();

  virtual double next(int frameX = -5);

  virtual void reset();

  virtual void nextSpeaker();

  double energyPercentile(double percentile = 50.0) const;

private:
  static int _comparator(const void* elem1, const void* elem2);
  virtual bool _aboveThreshold(const gsl_vector_float* vector);

  VectorFloatFeatureStreamPtr			_source;
  const double					_initialEnergy;

  const unsigned				_headN;
  const unsigned				_tailN;
  bool						_recognizing;

  unsigned					_aboveThresholdN;
  unsigned					_belowThresholdN;

  unsigned					_energiesN;
  double*					_energies;
  double*					_sortedEnergies;
  const unsigned				_medianIndex;
};

typedef Inherit<EnergyVADMetric, VADMetricPtr> EnergyVADMetricPtr;

// ----- definition for class `MultiChannelVADMetric' -----
//
template <typename ChannelType>
class MultiChannelVADMetric : public VADMetric {
 public:
  MultiChannelVADMetric(unsigned fftLen,
			double sampleRate, double lowCutoff, double highCutoff, const String& nm);
  ~MultiChannelVADMetric();
  void setChannel(ChannelType& chan);

protected:
  unsigned _setLowX(double lowCutoff) const;
  unsigned _setHighX(double highCutoff) const;
  unsigned _setBinN() const;
  
  typedef list<ChannelType>      	_ChannelList;
  //typedef list<ChannelType>::iterator	_ChannelIterator;
  _ChannelList				_channelList;

  const unsigned			_fftLen;
  const unsigned			_fftLen2;
  const double				_sampleRate;
  const unsigned			_lowX;
  const unsigned			_highX;
  const unsigned			_binN;
  FILE                                 *_logfp;
};

template<> MultiChannelVADMetric<VectorFloatFeatureStreamPtr>::MultiChannelVADMetric(unsigned fftLen, double sampleRate, double lowCutoff, double highCutoff, const String& nm);
template<> MultiChannelVADMetric<VectorComplexFeatureStreamPtr>::MultiChannelVADMetric(unsigned fftLen, double sampleRate, double lowCutoff, double highCutoff, const String& nm);
template<> MultiChannelVADMetric<VectorFloatFeatureStreamPtr>::~MultiChannelVADMetric();
template<> MultiChannelVADMetric<VectorComplexFeatureStreamPtr>::~MultiChannelVADMetric();
template<> void MultiChannelVADMetric<VectorFloatFeatureStreamPtr>::setChannel(VectorFloatFeatureStreamPtr& chan);
template<> void MultiChannelVADMetric<VectorComplexFeatureStreamPtr>::setChannel(VectorComplexFeatureStreamPtr& chan);

typedef MultiChannelVADMetric<VectorFloatFeatureStreamPtr>   FloatMultiChannelVADMetric;
typedef MultiChannelVADMetric<VectorComplexFeatureStreamPtr> ComplexMultiChannelVADMetric;
typedef refcountable_ptr<FloatMultiChannelVADMetric>         FloatMultiChannelVADMetricPtr;
typedef refcountable_ptr<ComplexMultiChannelVADMetric>       ComplexMultiChannelVADMetricPtr;

// ----- definition for class `PowerSpectrumVADMetric' -----
//
/**
   @brief detect voice activity based on the energy comparison
   @usage
   1. construct the object with PowerSpectrumVADMetric().
   2. set the channel data with setChannel().
   3. call next().
   @note the first channel is associated with the target speaker.
 */
class PowerSpectrumVADMetric : public FloatMultiChannelVADMetric {
public:
  PowerSpectrumVADMetric(unsigned fftLen,
			 double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
			 const String& nm = "Power Spectrum VAD Metric");
  PowerSpectrumVADMetric(VectorFloatFeatureStreamPtr& source1, VectorFloatFeatureStreamPtr& source2,
			 double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
			 const String& nm = "Power Spectrum VAD Metric");

  ~PowerSpectrumVADMetric();
  //void setChannel(VectorFloatFeatureStreamPtr& chan);
  virtual double next(int frameX = -5);
  virtual void reset();
  virtual void nextSpeaker();
  gsl_vector *getMetrics(){ return _powerList;}
  void setE0( double E0 ){ _E0 = E0;}
  void clearChannel();

protected:
  typedef _ChannelList::iterator		_ChannelIterator;
  gsl_vector                                   *_powerList;
  double                                        _E0;
};

typedef Inherit<PowerSpectrumVADMetric, FloatMultiChannelVADMetricPtr> PowerSpectrumVADMetricPtr;

// ----- definition for class `NormalizedEnergyMetric' -----
//

/**
   @class

   @usage
   @note
*/

class NormalizedEnergyMetric : public  PowerSpectrumVADMetric {
  static double _initialEnergy;
public:
  NormalizedEnergyMetric(unsigned fftLen,
		double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
		const String& nm = "TSPS VAD Metric");

  ~NormalizedEnergyMetric();
  
  virtual double next(int frameX = -5);

  virtual void reset();
};

typedef Inherit<NormalizedEnergyMetric, PowerSpectrumVADMetricPtr> NormalizedEnergyMetricPtr;

// ----- definition for class `CCCVADMetric' -----
//

/**
   @class compute cross-correlation coefficients (CCC) as a function of time delays,
          and average up the n-best values.
   @usage
   @note
*/

class CCCVADMetric : public ComplexMultiChannelVADMetric {
public:
  CCCVADMetric(unsigned fftLen, unsigned nCand,
	       double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
	       const String& nm = "CCC VAD Metric");

  ~CCCVADMetric();  
  void setNCand(unsigned nCand);
  void setThreshold(double threshold){ _threshold = threshold;}
  virtual double next(int frameX = -5);
  virtual void reset();
  virtual void nextSpeaker();
  gsl_vector *getMetrics(){ return _ccList;}
  void clearChannel();

protected:
  typedef _ChannelList::iterator		_ChannelIterator;
  unsigned                                      _nCand;
  gsl_vector                                   *_ccList;
  gsl_vector_int                               *_sampleDelays;
  double                                        _threshold;
  double                                       *_packCrossSpectrum;
};

typedef Inherit<CCCVADMetric, ComplexMultiChannelVADMetricPtr> CCCVADMetricPtr;

// ----- definition for class `TSPSVADMetric' -----
//

/**
   @class

   @usage
   @note
*/

class TSPSVADMetric : public  PowerSpectrumVADMetric {
  static double _initialEnergy;
public:
  TSPSVADMetric(unsigned fftLen,
		double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
		const String& nm = "TSPS VAD Metric");

  ~TSPSVADMetric();
  
  virtual double next(int frameX = -5);

  virtual void reset();
};

typedef Inherit<TSPSVADMetric,PowerSpectrumVADMetricPtr> TSPSVADMetricPtr;

// ----- definition for class `NegentropyVADMetric' -----
//
class NegentropyVADMetric : public VADMetric {

protected:
  class _ComplexGeneralizedGaussian : public Countable {
  public:
    _ComplexGeneralizedGaussian(double shapeFactor = 2.0);
    double logLhood(gsl_complex X, double scaleFactor) const;
    double shapeFactor() const { return _shapeFactor; }
    double Bc() const { return _Bc; }
    double normalization() const { return _normalization; }

  protected:
    virtual double _calcBc() const;
    virtual double _calcNormalization() const;

    const double				_shapeFactor;
    /* const */ double				_Bc;
    /* const */ double				_normalization;
  };

  typedef refcountable_ptr<_ComplexGeneralizedGaussian> _ComplexGeneralizedGaussianPtr;

  typedef list<_ComplexGeneralizedGaussianPtr>	_GaussianList;
  typedef _GaussianList::iterator		_GaussianListIterator;
  typedef _GaussianList::const_iterator		_GaussianListConstIterator;

public:
  NegentropyVADMetric(const VectorComplexFeatureStreamPtr& source, const VectorFloatFeatureStreamPtr& spectralEstimator,
		      const String& shapeFactorFileName = "", double threshold = 0.5,
		      double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
		      const String& nm = "Negentropy VAD Metric");

  ~NegentropyVADMetric();

  virtual double next(int frameX = -5);

  virtual void reset();

  virtual void nextSpeaker();

  double calcNegentropy(int frameX);

protected:
  virtual bool _aboveThreshold(int frameX);
  unsigned _setLowX(double lowCutoff) const;
  unsigned _setHighX(double highCutoff) const;
  unsigned _setBinN() const;

  VectorComplexFeatureStreamPtr			_source;
  VectorFloatFeatureStreamPtr			_spectralEstimator;

  _GaussianList					_generalizedGaussians;
  _ComplexGeneralizedGaussianPtr		_gaussian;

  const double					_threshold;

  const unsigned				_fftLen;
  const unsigned				_fftLen2;

  const double					_sampleRate;
  const unsigned				_lowX;
  const unsigned				_highX;
  const unsigned				_binN;
};

typedef Inherit<NegentropyVADMetric, VADMetricPtr> NegentropyVADMetricPtr;


// ----- definition for class `MutualInformationVADMetric' -----
//
class MutualInformationVADMetric : public NegentropyVADMetric {

protected:
  class _JointComplexGeneralizedGaussian : public NegentropyVADMetric::_ComplexGeneralizedGaussian {
  public:
    _JointComplexGeneralizedGaussian(const NegentropyVADMetric::_ComplexGeneralizedGaussianPtr& ggaussian);
    ~_JointComplexGeneralizedGaussian();

    double logLhood(gsl_complex X1, gsl_complex X2, double scaleFactor1, double scaleFactor2, gsl_complex rho12) const;

  private:
    static const double      _sqrtTwo;
    static const gsl_complex _complexOne;
    static const gsl_complex _complexZero;

    virtual double _calcBc() const;
    virtual double _calcNormalization() const;

    double _lngammaRatio(double f) const;
    double _lngammaRatioJoint(double f) const;
    double _match(double f) const;

    double _matchScoreMarginal(double f) const;
    double _matchScoreJoint(double fJ) const;

    static const double				_tolerance;

    gsl_vector_complex*				_X;
    gsl_vector_complex*				_scratch;
    gsl_matrix_complex*				_SigmaXinverse;
  };

  typedef refcountable_ptr<_JointComplexGeneralizedGaussian> _JointComplexGeneralizedGaussianPtr;

  typedef list<_JointComplexGeneralizedGaussianPtr>	_JointGaussianList;
  typedef _JointGaussianList::iterator			_JointGaussianListIterator;
  typedef _JointGaussianList::const_iterator		_JointGaussianListConstIterator;

  typedef vector<gsl_complex>				_CrossCorrelationVector;

public:
  MutualInformationVADMetric(const VectorComplexFeatureStreamPtr& source1, const VectorComplexFeatureStreamPtr& source2,
			     const VectorFloatFeatureStreamPtr& spectralEstimator1, const VectorFloatFeatureStreamPtr& spectralEstimator2,
			     const String& shapeFactorFileName = "", double twiddle = -1.0, double threshold = 1.3, double beta = 0.95,
			     double sampleRate = 16000.0, double lowCutoff = 187.0, double highCutoff = 1000.0,
			     const String& nm = "Mutual Information VAD Metric");

  ~MutualInformationVADMetric();

  virtual double next(int frameX = -5);

  virtual void reset();

  virtual void nextSpeaker();

  double calcMutualInformation(int frameX);

protected:
  static const double				_epsilon;

  virtual bool _aboveThreshold(int frameX);
  double _calcFixedThreshold();
  double _calcTotalThreshold() const;
  void _initializePdfs();

  VectorComplexFeatureStreamPtr			_source2;
  VectorFloatFeatureStreamPtr			_spectralEstimator2;

  _JointGaussianList				_jointGeneralizedGaussians;
  _CrossCorrelationVector			_crossCorrelations;
  const double					_fixedThreshold;
  const double					_twiddle;
  const double					_threshold;
  const double					_beta;
};

typedef Inherit<MutualInformationVADMetric, NegentropyVADMetricPtr> MutualInformationVADMetricPtr;


// ----- definition for class `LikelihoodRatioVADMetric' -----
//
class LikelihoodRatioVADMetric : public NegentropyVADMetric {

public:
  LikelihoodRatioVADMetric(const VectorComplexFeatureStreamPtr& source1, const VectorComplexFeatureStreamPtr& source2,
			   const VectorFloatFeatureStreamPtr& spectralEstimator1, const VectorFloatFeatureStreamPtr& spectralEstimator2,
			   const String& shapeFactorFileName = "", double threshold = 0.0,
			   double sampleRate = 16000.0, double lowCutoff = 187.0, double highCutoff = 1000.0,
			   const String& nm = "Likelihood VAD Metric");

  ~LikelihoodRatioVADMetric();

  virtual double next(int frameX = -5);

  virtual void reset();

  virtual void nextSpeaker();

  double calcLikelihoodRatio(int frameX);

private:
  const VectorComplexFeatureStreamPtr			_source2;
  const VectorFloatFeatureStreamPtr			_spectralEstimator2;
};

typedef Inherit<LikelihoodRatioVADMetric, NegentropyVADMetricPtr> LikelihoodRatioVADMetricPtr;


// ----- definition for class `LowFullBandEnergyRatioVADMetric' -----
//
class LowFullBandEnergyRatioVADMetric : public VADMetric {
public:
  LowFullBandEnergyRatioVADMetric(const VectorFloatFeatureStreamPtr& source, const gsl_vector* lowpass, double threshold = 0.5, const String& nm = "Low- Full-Band Energy Ratio VAD Metric");

  ~LowFullBandEnergyRatioVADMetric();

  virtual double next(int frameX = -5);

  virtual void reset();

  virtual void nextSpeaker();

private:
  virtual bool _aboveThreshold(int frameX);
  void _calcAutoCorrelationVector(int frameX);
  void _calcCovarianceMatrix();
  double _calcLowerBandEnergy();

  VectorFloatFeatureStreamPtr			_source;
  const unsigned				_lagsN;
  gsl_vector*					_lowpass;
  gsl_vector*					_scratch;
  double*					_autocorrelation;
  gsl_matrix*					_covariance;
};

typedef Inherit<LowFullBandEnergyRatioVADMetric, VADMetricPtr> LowFullBandEnergyRatioVADMetricPtr;


// ----- definition for class `HangoverVADFeature' -----
//
class HangoverVADFeature : public VectorFloatFeatureStream {
protected:
  typedef pair<VADMetricPtr, double>		_MetricPair;
  typedef vector<_MetricPair>			_MetricList;
  typedef _MetricList::iterator			_MetricListIterator;
  typedef _MetricList::const_iterator		_MetricListConstIterator;

  static const unsigned EnergyVADMetricX		= 0;
  static const unsigned MutualInformationVADMetricX	= 1;
  static const unsigned LikelihoodRatioVADMetricX	= 2;

 public:
  HangoverVADFeature(const VectorFloatFeatureStreamPtr& source, const VADMetricPtr& metric, double threshold = 0.5,
		     unsigned headN = 4, unsigned tailN = 10, const String& nm = "Hangover VAD Feature");
  virtual ~HangoverVADFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

  virtual void nextSpeaker();

  int prefixN() const { return _prefixN - _headN; }

 protected:
  static int _comparator(const void* elem1, const void* elem2);
  virtual bool _aboveThreshold(int frameX);

  VectorFloatFeatureStreamPtr			_source;
  bool						_recognizing;

  gsl_vector_float**				_buffer;
  const unsigned				_headN;
  const unsigned				_tailN;
  int						_bufferIndex;
  unsigned					_bufferedN;

  unsigned					_aboveThresholdN;
  unsigned					_belowThresholdN;
  unsigned					_prefixN;

  _MetricList					_metricList;
};

typedef Inherit<HangoverVADFeature, VectorFloatFeatureStreamPtr> HangoverVADFeaturePtr;


// ----- definition for class `HangoverMIVADFeature' -----
//
class HangoverMIVADFeature : public HangoverVADFeature {
  static const unsigned EnergyVADMetricX		= 0;
  static const unsigned MutualInformationVADMetricX	= 1;
  static const unsigned PowerVADMetricX			= 2;

 public:
  HangoverMIVADFeature(const VectorFloatFeatureStreamPtr& source,
		       const VADMetricPtr& energyMetric, const VADMetricPtr& mutualInformationMetric, const VADMetricPtr& powerMetric,
		       double energythreshold = 0.5, double mutualInformationThreshold = 0.5, double powerThreshold = 0.5,
		       unsigned headN = 4, unsigned tailN = 10, const String& nm = "Hangover VAD Feature");

  int decisionMetric() const { return _decisionMetric; }

protected:
  virtual bool _aboveThreshold(int frameX);

  int						_decisionMetric;
};

typedef Inherit<HangoverMIVADFeature, HangoverVADFeaturePtr> HangoverMIVADFeaturePtr;


// ----- definition for class `HangoverMultiStageVADFeature' -----
//
class HangoverMultiStageVADFeature : public HangoverVADFeature {
 public:
  HangoverMultiStageVADFeature(const VectorFloatFeatureStreamPtr& source,
			       const VADMetricPtr& energyMetric, double energythreshold = 0.5, 
			       unsigned headN = 4, unsigned tailN = 10, const String& nm = "HangoverMultiStageVADFeature");
  ~HangoverMultiStageVADFeature();
  int decisionMetric() const { return _decisionMetric; }
  void setMetric( const VADMetricPtr& metricPtr, double threshold ){
    _metricList.push_back( _MetricPair( metricPtr, threshold ) );
  }

#ifdef _LOG_SAD_ 
  void initScores();
  gsl_vector *getScores();
#endif /* _LOG_SAD_ */

protected:
  virtual bool _aboveThreshold(int frameX);
  int	       _decisionMetric;
#ifdef _LOG_SAD_ 
  gsl_vector *_scores;
#endif /* _LOG_SAD_ */
};

typedef Inherit<HangoverMultiStageVADFeature, HangoverVADFeaturePtr> HangoverMultiStageVADFeaturePtr;

#endif
