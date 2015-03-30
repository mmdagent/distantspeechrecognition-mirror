//                              -*- C++ -*-
//
//                       Beamforming Toolkit
//                              (btk)
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

%module(package="btk") sad

#define _LOG_SAD_ 

%{
#include "stream/stream.h"
#include "stream/pyStream.h"
#include "feature/feature.h"
#include <numpy/arrayobject.h>
#include "sad/sad.h"
#include "sad/sadFeature.h"
#include "sad/ica.h"
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

%rename(__str__) *::print;
%rename(__getitem__) *::operator[](int);
%ignore *::nextSampleBlock(const double* smp);

%pythoncode %{
import btk
from btk import stream
oldimport = """
%}
%import stream/stream.i
%pythoncode %{
"""
%}


// ----- definition for class `NeuralNetVAD' -----
//
class NeuralNetVAD {
public:
  NeuralNetVAD(VectorFloatFeatureStreamPtr& cep,
				  unsigned context = 4, unsigned hiddenUnitsN = 1000, unsigned outputUnitsN = 2, float threshold = 0.1,
				  const String& neuralNetFile = "");
  ~NeuralNetVAD();

  bool next(int frameX = -5) = 0;

  void reset();

  void read(const String& neuralNetFile);
};

class NeuralNetVADPtr {
 public:
  %extend {
    NeuralNetVADPtr(VectorFloatFeatureStreamPtr& cep,
				       unsigned context = 4, unsigned hiddenUnitsN = 1000, unsigned outputUnitsN = 2, float threshold = 0.1,
				       const String& neuralNetFile = "") {
      return new NeuralNetVADPtr(new NeuralNetVAD(cep, context, hiddenUnitsN, outputUnitsN, threshold, neuralNetFile));
    }

    NeuralNetVADPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NeuralNetVAD* operator->();
};


// ----- definition for class `VAD' -----
//
%ignore VAD;
class VAD {
public:
  VAD(VectorFloatFeatureStreamPtr& samp)
    : _samp(samp) { }

  virtual bool next() = 0;

  virtual void reset() = 0;

  virtual void nextSpeaker() = 0;

  const gsl_vector_complex* frame() const;
};

class VADPtr {
 public:
  %extend {
    VADPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  VAD* operator->();
};


// ----- definition for class `SimpleEnergyVAD' -----
//
%ignore SimpleEnergyVAD;
class SimpleEnergyVAD : public VAD {
public:
  SimpleEnergyVAD(VectorComplexFeatureStreamPtr& samp,
		  double threshold, double gamma = 0.995);
  ~SimpleEnergyVAD();

  bool next();

  virtual void nextSpeaker();

  virtual void reset();
};

class SimpleEnergyVADPtr : public VADPtr {
 public:
  %extend {
    SimpleEnergyVADPtr(VectorComplexFeatureStreamPtr& samp,
		       double threshold, double gamma = 0.98) {
      return new SimpleEnergyVADPtr(new SimpleEnergyVAD(samp, threshold, gamma));
    }

    SimpleEnergyVADPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SimpleEnergyVAD* operator->();
};


// ----- definition for class `SimpleLikelihoodRatioVAD' -----
//
%ignore SimpleLikelihoodRatioVAD;
class SimpleLikelihoodRatioVAD : public VAD {
public:
  SimpleLikelihoodRatioVAD(VectorComplexFeatureStreamPtr& samp,
					double threshold = 0.0, double alpha = 0.99);
  ~SimpleLikelihoodRatioVAD();

  bool next();

  virtual void reset();

  virtual void nextSpeaker();

  void setVariance(const gsl_vector* variance);
};

class SimpleLikelihoodRatioVADPtr : public VADPtr {
 public:
  %extend {
    SimpleLikelihoodRatioVADPtr(VectorComplexFeatureStreamPtr& samp,
                                             double threshold = 0.0, double alpha = 0.99) {
	return new SimpleLikelihoodRatioVADPtr(new SimpleLikelihoodRatioVAD(samp, threshold, alpha));
    }

    SimpleLikelihoodRatioVADPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SimpleLikelihoodRatioVAD* operator->();
};


// ----- definition for class `EnergyVADFeature' -----
// 
%ignore EnergyVADFeature;
class EnergyVADFeature : public VectorFloatFeatureStream {
public:
  EnergyVADFeature(const VectorFloatFeatureStreamPtr& source, double threshold = 0.5, unsigned bufferLength = 30, unsigned energiesN = 200, const String& nm = "Energy VAD");

  virtual const gsl_vector_float* next() const;

  void reset();

  void nextSpeaker();
};

class EnergyVADFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    EnergyVADFeaturePtr(const VectorFloatFeatureStreamPtr& source, double threshold = 0.5, unsigned bufferLength = 40, unsigned energiesN = 200, const String& nm = "Hamming") {
      return new EnergyVADFeaturePtr(new EnergyVADFeature(source, threshold, bufferLength, energiesN, nm));
    }

    EnergyVADFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  EnergyVADFeature* operator->();
};


// ----- definition for abstract base class `VADMetric' -----
//
%ignore VADMetric;
class VADMetric :  public Countable {
public:
  VADMetric(const VectorFloatFeatureStreamPtr& source);

  ~VADMetric();

  virtual double next(int frameX = -5) = 0;

  virtual void reset() = 0;

  virtual void nextSpeaker() = 0;

  double score();

#ifdef  _LOG_SAD_ 
  bool openLogFile( const String & logfilename );
  int  writeLog( const char *format, ... );
  void closeLogFile();
  void initScore();
  void setScore( double score);
  double getAverageScore();
#endif /* _LOG_SAD_ */
};

class VADMetricPtr {
 public:
  %extend {
    VADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  VADMetric* operator->();
};

// ----- definition for class `EnergyVADMetric' -----
//
%ignore EnergyVADMetric;
class EnergyVADMetric : public VADMetric {
public:
  EnergyVADMetric(const VectorFloatFeatureStreamPtr& source, double initialEnergy = 5.0e+07, double threshold = 0.5, unsigned headN = 4,
		  unsigned tailN = 10, unsigned energiesN = 200, const String& nm = "Energy VAD Metric");

  ~EnergyVADMetric();

  virtual double next(int frameX = -5);

  virtual void reset();

  virtual void nextSpeaker();

  double energyPercentile(double percentile = 50.0) const;
};

class EnergyVADMetricPtr : public VADMetricPtr {
 public:
  %extend {
    EnergyVADMetricPtr(const VectorFloatFeatureStreamPtr& source, double initialEnergy = 5.0e+07, double threshold = 0.5, unsigned headN = 4,
		       unsigned tailN = 10, unsigned energiesN = 200, const String& nm = "Energy VAD Metric")
    {
      return new EnergyVADMetricPtr(new EnergyVADMetric(source, initialEnergy, threshold, headN, tailN, energiesN, nm));
    }

    EnergyVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  EnergyVADMetric* operator->();
};


// ----- definition for class `MultiChannelVADMetric' -----
//
%ignore FloatMultiChannelVADMetric;
class FloatMultiChannelVADMetric : public VADMetric {
public:
  FloatMultiChannelVADMetric();
  ~FloatMultiChannelVADMetric();
  void setChannel(VectorFloatFeatureStreamPtr& chan);
};

class FloatMultiChannelVADMetricPtr : public VADMetricPtr {
 public:
  %extend {
    FloatMultiChannelVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  FloatMultiChannelVADMetricPtr* operator->();
};

// ----- definition for class `MultiChannelVADMetric' -----
//
%ignore ComplexMultiChannelVADMetric;
class ComplexMultiChannelVADMetric : public VADMetric {
public:
  ComplexMultiChannelVADMetric();
  ~ComplexMultiChannelVADMetric();
  void setChannel(VectorComplexFeatureStreamPtr& chan);
};

class ComplexMultiChannelVADMetricPtr : public VADMetricPtr {
 public:
  %extend {
    ComplexMultiChannelVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  ComplexMultiChannelVADMetricPtr* operator->();
};

// ----- definition for class `PowerSpectrumVADMetric' -----
//
%ignore PowerSpectrumVADMetric;
class PowerSpectrumVADMetric : public FloatMultiChannelVADMetric {
public:
  PowerSpectrumVADMetric(unsigned fftLen,
			 double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
			 const String& nm = "Power Spectrum VAD Metric");
  /*PowerSpectrumVADMetric(const VectorFloatFeatureStreamPtr& source1, const VectorFloatFeatureStreamPtr& source2,
			 double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
			 const String& nm = "Power Spectrum VAD Metric");*/
  ~PowerSpectrumVADMetric();
  virtual double next(int frameX = -5);
  virtual void reset();
  virtual void nextSpeaker();
  gsl_vector *getMetrics();
  void setE0( double E0 );
  void clearChannel();
};

class PowerSpectrumVADMetricPtr : public FloatMultiChannelVADMetricPtr {
 public:
  %extend {
    PowerSpectrumVADMetricPtr(unsigned fftLen,
			      double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
			      const String& nm = "Power Spectrum VAD Metric")
    {
      return new PowerSpectrumVADMetricPtr(new PowerSpectrumVADMetric(fftLen, sampleRate, lowCutoff, highCutoff, nm));
    }

    PowerSpectrumVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  PowerSpectrumVADMetric* operator->();
};

// ----- definition for class `NormalizedEnergyMetric' -----
//
%ignore NormalizedEnergyMetric;
class NormalizedEnergyMetric : public PowerSpectrumVADMetric {
public:
  NormalizedEnergyMetric(unsigned fftLen,
		double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
		const String& nm = "NormalizedEnergyMetric");
  ~NormalizedEnergyMetric();
  virtual double next(int frameX = -5);
  virtual void reset();
  void setE0( double E0 );
};

class NormalizedEnergyMetricPtr : public PowerSpectrumVADMetricPtr {
 public:
  %extend {
    NormalizedEnergyMetricPtr(unsigned fftLen,
		     double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
		     const String& nm = "NormalizedEnergyMetric")
    {
      return new NormalizedEnergyMetricPtr(new NormalizedEnergyMetric(fftLen, sampleRate, lowCutoff, highCutoff, nm));
    }

    NormalizedEnergyMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NormalizedEnergyMetric* operator->();
};

// ----- definition for class `CCCVADMetric' -----
//
%ignore CCCVADMetric;
class CCCVADMetric : public ComplexMultiChannelVADMetric {
public:
  CCCVADMetric(unsigned fftLen, unsigned nCand,
	       double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
	       const String& nm = "CCC VAD Metric");
  ~CCCVADMetric();
  void setNCand(unsigned nCand);
  void setThreshold(double threshold);
  virtual double next(int frameX = -5);
  virtual void reset();
  virtual void nextSpeaker();
  gsl_vector *getMetrics();
  void clearChannel();
};
  
class CCCVADMetricPtr : public ComplexMultiChannelVADMetricPtr {
 public:
  %extend {
    CCCVADMetricPtr(unsigned fftLen, unsigned nCand,
		    double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
		    const String& nm = "CCC VAD Metric")
      {
	return new CCCVADMetricPtr(new CCCVADMetric(fftLen, nCand, sampleRate, lowCutoff, highCutoff, nm));
      }
    
    CCCVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  CCCVADMetric* operator->();
};

// ----- definition for class `TSPSVADMetric' -----
//
%ignore TSPSVADMetric;
class TSPSVADMetric : public PowerSpectrumVADMetric {
public:
  TSPSVADMetric(unsigned fftLen,
		double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
		const String& nm = "TSPS VAD Metric");

  ~TSPSVADMetric();
  virtual double next(int frameX = -5);
  virtual void reset();
  void setE0( double E0 );
};

class TSPSVADMetricPtr : public PowerSpectrumVADMetricPtr {
 public:
  %extend {
    TSPSVADMetricPtr(unsigned fftLen,
		     double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
		     const String& nm = "TSPS VAD Metric")
    {
      return new TSPSVADMetricPtr(new TSPSVADMetric(fftLen, sampleRate, lowCutoff, highCutoff, nm));
    }

    TSPSVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  TSPSVADMetric* operator->();
};

// ----- definition for class `NegentropyVADMetric' -----
//
%ignore NegentropyVADMetric;
class NegentropyVADMetric : public VADMetric {
public:
  NegentropyVADMetric(const VectorComplexFeatureStreamPtr& source, const VectorFloatFeatureStreamPtr& spectralEstimator,
		      const String& shapeFactorFileName = "", double threshold = 0.5,
		      double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
		      const String& nm = "Negentropy VAD Metric");

  ~NegentropyVADMetric();

  double calcNegentropy(int frameX);

  virtual double next(int frameX = -5);

  virtual void reset();

  virtual void nextSpeaker();
};

class NegentropyVADMetricPtr : public VADMetricPtr {
 public:
  %extend {
    NegentropyVADMetricPtr(const VectorComplexFeatureStreamPtr& source, const VectorFloatFeatureStreamPtr& spectralEstimator,
			   const String& shapeFactorFileName = "", double threshold = 0.5,
			   double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
			   const String& nm = "Negentropy VAD Metric")
    {
      return new NegentropyVADMetricPtr(new NegentropyVADMetric(source, spectralEstimator, shapeFactorFileName, threshold,
								sampleRate, lowCutoff, highCutoff, nm));
    }

    NegentropyVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NegentropyVADMetric* operator->();
};


// ----- definition for class `MutualInformationVADMetric' -----
//
%ignore MutualInformationVADMetric;
class MutualInformationVADMetric : public NegentropyVADMetric {
public:
  MutualInformationVADMetric(const VectorComplexFeatureStreamPtr& source1, const VectorComplexFeatureStreamPtr& source2,
			     const VectorFloatFeatureStreamPtr& spectralEstimator1, const VectorFloatFeatureStreamPtr& spectralEstimator2,
			     const String& shapeFactorFileName = "", double twiddle = -1.0, double threshold = 1.3, double beta = 0.95,
			     double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
			     const String& nm = "Mutual Information VAD Metric");

  ~MutualInformationVADMetric();

  virtual double next(int frameX = -5);

  virtual void reset();

  virtual void nextSpeaker();

  double calcMutualInformation(int frameX);
};

class MutualInformationVADMetricPtr : public NegentropyVADMetricPtr {
 public:
  %extend {
    MutualInformationVADMetricPtr(const VectorComplexFeatureStreamPtr& source1, const VectorComplexFeatureStreamPtr& source2,
				  const VectorFloatFeatureStreamPtr& spectralEstimator1, const VectorFloatFeatureStreamPtr& spectralEstimator2,
				  const String& shapeFactorFileName = "", double twiddle = -1.0, double threshold = 1.3, double beta = 0.95,
				  double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
				  const String& nm = "Mutual Information VAD Metric")
    {
      return new MutualInformationVADMetricPtr(new MutualInformationVADMetric(source1, source2, spectralEstimator1, spectralEstimator2,
									      shapeFactorFileName, twiddle, threshold, beta,
									      sampleRate, lowCutoff, highCutoff, nm));
    }

    MutualInformationVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  MutualInformationVADMetric* operator->();
};


// ----- definition for class `LikelihoodRatioVADMetric' -----
//
%ignore LikelihoodRatioVADMetric;
class LikelihoodRatioVADMetric : public NegentropyVADMetric {
public:
  LikelihoodRatioVADMetric(const VectorComplexFeatureStreamPtr& source1, const VectorComplexFeatureStreamPtr& source2,
			   const VectorFloatFeatureStreamPtr& spectralEstimator1, const VectorFloatFeatureStreamPtr& spectralEstimator2,
			   const String& shapeFactorFileName = "", double threshold = 1.0,
			   double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
			   const String& nm = "Mutual Information VAD Metric");

  ~LikelihoodRatioVADMetric();

  virtual double next(int frameX = -5);

  virtual void reset();

  virtual void nextSpeaker();

  double calcLikelihoodRatio(int frameX);
};

class LikelihoodRatioVADMetricPtr : public NegentropyVADMetricPtr {
 public:
  %extend {
    LikelihoodRatioVADMetricPtr(const VectorComplexFeatureStreamPtr& source1, const VectorComplexFeatureStreamPtr& source2,
				const VectorFloatFeatureStreamPtr& spectralEstimator1, const VectorFloatFeatureStreamPtr& spectralEstimator2,
				const String& shapeFactorFileName = "", double threshold = 0.0,
				double sampleRate = 16000.0, double lowCutoff = -1.0, double highCutoff = -1.0,
				const String& nm = "Mutual Information VAD Metric")
    {
      return new LikelihoodRatioVADMetricPtr(new LikelihoodRatioVADMetric(source1, source2, spectralEstimator1, spectralEstimator2,
									  shapeFactorFileName, threshold, sampleRate, lowCutoff, highCutoff, nm));
    }

    LikelihoodRatioVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  LikelihoodRatioVADMetric* operator->();
};


// ----- definition for class `LowFullBandEnergyRatioVADMetric' -----
//
class LowFullBandEnergyRatioVADMetric : public VADMetric {
public:
  LowFullBandEnergyRatioVADMetric(const VectorFloatFeatureStreamPtr& source, const gsl_vector* lowpass, double threshold = 0.5, const String& nm = "Low- Full-Band Energy Ratio VAD Metric");

  ~LowFullBandEnergyRatioVADMetric();

  virtual double next(int frameX = -5);

  virtual void reset();

  virtual void nextSpeaker();
};

class LowFullBandEnergyRatioVADMetricPtr : public VADMetricPtr {
 public:
  %extend {
    LowFullBandEnergyRatioVADMetricPtr(const VectorFloatFeatureStreamPtr& source, const gsl_vector* lowpass, double threshold = 0.5, const String& nm = "Low- Full-Band Energy Ratio VAD Metric")
    {
      return new LowFullBandEnergyRatioVADMetricPtr(new LowFullBandEnergyRatioVADMetric(source, lowpass, threshold, nm));
    }

    LowFullBandEnergyRatioVADMetricPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  LowFullBandEnergyRatioVADMetric* operator->();
};


// ----- definition for class `HangoverVADFeature' -----
// 
%ignore HangoverVADFeature;
class HangoverVADFeature : public VectorFloatFeatureStream {
public:
  HangoverVADFeature(const VectorFloatFeatureStreamPtr& source, const VADMetricPtr& metric, double threshold = 0.5,
		     unsigned headN = 4, unsigned tailN = 10, const String& nm = "Hangover VAD Feature");

  virtual const gsl_vector_float* next() const;

  void reset();

  void nextSpeaker();

  int prefixN() const;
};

class HangoverVADFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    HangoverVADFeaturePtr(const VectorFloatFeatureStreamPtr& source, const VADMetricPtr& metric, double threshold = 0.5,
			  unsigned headN = 4, unsigned tailN = 10, const String& nm = "Hangover VAD Feature")
    {
      return new HangoverVADFeaturePtr(new HangoverVADFeature(source, metric, threshold, headN, tailN, nm));
    }

    HangoverVADFeaturePtr __iter__()
    {
      (*self)->reset();  return *self;
    }
  }

  HangoverVADFeature* operator->();
};


// ----- definition for class `HangoverMIVADFeature' -----
//
%ignore HangoverMIVADFeature;
class HangoverMIVADFeature : public HangoverVADFeature {
public:
  HangoverMIVADFeature(const VectorFloatFeatureStreamPtr& source,
		       const VADMetricPtr& energyMetric, const VADMetricPtr& mutualInformationMetric, const VADMetricPtr& powerMetric,
		       double energyThreshold = 0.5, double mutualInformationThreshold = 0.5, double powerThreshold = 0.5,
		       unsigned headN = 4, unsigned tailN = 10, const String& nm = "Hangover MIVAD Feature");

  int decisionMetric() const;
};

class HangoverMIVADFeaturePtr : public HangoverVADFeaturePtr {
 public:
  %extend {
    HangoverMIVADFeaturePtr(const VectorFloatFeatureStreamPtr& source,
			    const VADMetricPtr& energyMetric, const VADMetricPtr& mutualInformationMetric, const VADMetricPtr& powerMetric,
			    double energyThreshold = 0.5, double mutualInformationThreshold = 0.5, double powerThreshold = 0.5,
			    unsigned headN = 4, unsigned tailN = 10, const String& nm = "Hangover MIVAD Feature")
    {
      return new HangoverMIVADFeaturePtr(new HangoverMIVADFeature(source, energyMetric, mutualInformationMetric, powerMetric,
								  energyThreshold, mutualInformationThreshold, powerThreshold,
								  headN, tailN, nm));
    }

    HangoverMIVADFeaturePtr __iter__()
    {
      (*self)->reset();  return *self;
    }
  }

  HangoverMIVADFeature* operator->();
};


// ----- definition for class `HangoverMultiStageVADFeature' -----
//
%ignore HangoverMultiStageVADFeature;
class HangoverMultiStageVADFeature : public HangoverVADFeature {
public:
  HangoverMultiStageVADFeature(const VectorFloatFeatureStreamPtr& source,
			       const VADMetricPtr& energyMetric, double energyThreshold = 0.5, 
			       unsigned headN = 4, unsigned tailN = 10, const String& nm = "HangoverMultiStageVADFeature");
  
  int decisionMetric() const;
  void setMetric( const VADMetricPtr& metricPtr, double threshold );

#ifdef _LOG_SAD_ 
  void initScores();
  gsl_vector *getScores();
#endif /* _LOG_SAD_ */
};

class HangoverMultiStageVADFeaturePtr : public HangoverVADFeaturePtr {
 public:
  %extend {
    HangoverMultiStageVADFeaturePtr(const VectorFloatFeatureStreamPtr& source,
				    const VADMetricPtr& energyMetric, double energyThreshold = 0.5, 
				    unsigned headN = 4, unsigned tailN = 10, const String& nm = "HangoverMultiStageVADFeature")
    {
      return new HangoverMultiStageVADFeaturePtr(new HangoverMultiStageVADFeature(source, energyMetric, energyThreshold, headN, tailN, nm));
    }

    HangoverMultiStageVADFeaturePtr __iter__()
    {
      (*self)->reset();  return *self;
    }
  }

  HangoverMultiStageVADFeature* operator->();
};


// ----- definition for class `BrightnessFeature' -----
// 
%ignore BrightnessFeature;
class BrightnessFeature : public VectorFloatFeatureStream {
public:
  BrightnessFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, const String& nm = "Brightness");

  const gsl_vector_float* next() const;
};

class BrightnessFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    BrightnessFeaturePtr(const VectorFloatFeatureStreamPtr& src, float sampleRate, const String& nm = "Brightness") {
      return new BrightnessFeaturePtr(new BrightnessFeature(src, sampleRate, nm));
    }

    BrightnessFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  BrightnessFeature* operator->();
};


// ----- definition for class `EnergyDiffusionFeature' -----
// 
%ignore EnergyDiffusionFeature;
class EnergyDiffusionFeature : public VectorFloatFeatureStream {
public:
  EnergyDiffusionFeature(const VectorFloatFeatureStreamPtr& src, const String& nm = "Energy Diffusion");

  const gsl_vector_float* next() const;
};

class EnergyDiffusionFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    EnergyDiffusionFeaturePtr(const VectorFloatFeatureStreamPtr& src, const String& nm = "Energy Diffusion") {
      return new EnergyDiffusionFeaturePtr(new EnergyDiffusionFeature(src, nm));
    }

    EnergyDiffusionFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  EnergyDiffusionFeature* operator->();
};


// ----- definition for class `BandEnergyRatioFeature' -----
// 
%ignore BandEnergyRatioFeature;
class BandEnergyRatioFeature : public VectorFloatFeatureStream {
public:
  BandEnergyRatioFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, float threshF = 0.0, const String& nm = "Band Energy Ratio");

  const gsl_vector_float* next() const;
};

class BandEnergyRatioFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    BandEnergyRatioFeaturePtr(const VectorFloatFeatureStreamPtr& src, float sampleRate, float threshF = 0.0, const String& nm = "Band Energy Ratio") {
      return new BandEnergyRatioFeaturePtr(new BandEnergyRatioFeature(src, sampleRate, threshF, nm));
    }

    BandEnergyRatioFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  BandEnergyRatioFeature* operator->();
};


// ----- definition for class `NormalizedFluxFeature' -----
// 
%ignore NormalizedFluxFeature;
class NormalizedFluxFeature : public VectorFloatFeatureStream {
public:
  NormalizedFluxFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, float threshF = 0.0, const String& nm = "Band Energy Ratio");

  const gsl_vector_float* next() const;
};

class NormalizedFluxFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    NormalizedFluxFeaturePtr(const VectorFloatFeatureStreamPtr& src, float sampleRate, float threshF = 0.0, const String& nm = "Band Energy Ratio") {
      return new NormalizedFluxFeaturePtr(new NormalizedFluxFeature(src, sampleRate, threshF, nm));
    }

    NormalizedFluxFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NormalizedFluxFeature* operator->();
};


// ----- definition for class `NegativeEntropyFeature' -----
// 
%ignore NegativeEntropyFeature;
class NegativeEntropyFeature : public VectorFloatFeatureStream {
public:
  NegativeEntropyFeature(const VectorFloatFeatureStreamPtr& src, const String& nm = "Negative Entropy");

  const gsl_vector_float* next() const;
};

class NegativeEntropyFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    NegativeEntropyFeaturePtr(const VectorFloatFeatureStreamPtr& src, const String& nm = "Negative Entropy") {
      return new NegativeEntropyFeaturePtr(new NegativeEntropyFeature(src, nm));
    }

    NegativeEntropyFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NegativeEntropyFeature* operator->();
};


// ----- definition for class `SignificantSubbandsFeature' -----
// 
%ignore SignificantSubbandsFeature;
class SignificantSubbandsFeature : public VectorFloatFeatureStream {
public:
  SignificantSubbandsFeature(const VectorFloatFeatureStreamPtr& src, float thresh = 0.0, const String& nm = "Significant Subbands");

  const gsl_vector_float* next() const;
};

class SignificantSubbandsFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    SignificantSubbandsFeaturePtr(const VectorFloatFeatureStreamPtr& src, float thresh = 0.0, const String& nm = "Significant Subbands") {
      return new SignificantSubbandsFeaturePtr(new SignificantSubbandsFeature(src, thresh, nm));
    }

    SignificantSubbandsFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SignificantSubbandsFeature* operator->();
};


// ----- definition for class `NormalizedBandwidthFeature' -----
// 
%ignore NormalizedBandwidthFeature;
class NormalizedBandwidthFeature : public VectorFloatFeatureStream {
public:
  NormalizedBandwidthFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, float thresh = 0.0, const String& nm = "Band Energy Ratio");

  const gsl_vector_float* next() const;
};

class NormalizedBandwidthFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    NormalizedBandwidthFeaturePtr(const VectorFloatFeatureStreamPtr& src, float sampleRate, float thresh = 0.0, const String& nm = "Band Energy Ratio") {
      return new NormalizedBandwidthFeaturePtr(new NormalizedBandwidthFeature(src, sampleRate, thresh, nm));
    }

    NormalizedBandwidthFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NormalizedBandwidthFeature* operator->();
};


// ----- definition for class `PCA' -----
// 
%ignore PCA;
class PCA {
public:
  PCA(unsigned dimN);
  ~PCA();

  void pca_svd(gsl_matrix* input, gsl_matrix* basis, gsl_vector* eigenVal, gsl_vector* whiten);
  void pca_eigen(gsl_matrix* input, gsl_matrix* basis, gsl_vector* eigenVal, gsl_vector* whiten);
};

class PCAPtr {
 public:
  %extend {
    PCAPtr(unsigned dimN) {
      return new PCAPtr(new PCA(dimN));
    }
  }

  PCA* operator->();
};


// ----- definition for class `FastICA' -----
// 
%ignore FastICA;
class FastICA {
public:
  FastICA(unsigned dimN, unsigned maxIterN);
  ~FastICA();

  void deflation(gsl_matrix* data, gsl_matrix* B, gsl_matrix* A, gsl_matrix* W, gsl_matrix* M,
		 gsl_matrix* neg, double eps, int maxIterN);
};

class FastICAPtr {
 public:
  %extend {
    FastICAPtr(unsigned dimN, unsigned maxIterN) {
      return new FastICAPtr(new FastICA(dimN, maxIterN));
    }
  }

  FastICA* operator->();
};
