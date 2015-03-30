//                              -*- C++ -*-
//
//                            Speech Front End for Beamforming
//                                  (btk)
//
//  Module:  btk.feature
//  Purpose: Speech recognition front end.
//  Author:  John McDonough and ABC
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


#ifndef _feature_h_
#define _feature_h_

#include <math.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_vector_complex.h>
#include "matrix/gslmatrix.h"
#include "stream/stream.h"
#include "common/mlist.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netdb.h>

#include <pthread.h>

#include "btk.h"

#ifdef HAVE_LIBFFTW3
#include <fftw3.h>
#endif /* #ifdef HAVE_LIBFFTW3 */

#include "feature/spectralestimator.h"
#include "feature/audioIO.h"

/* sflib and sndfile both define sf_perror so we put them into seperate namespaces */
namespace sndfile {
#include <sndfile.h>
}

void halfComplexUnpack(gsl_vector_complex* tgt, const double* src);

void halfComplexPack(double* tgt, const gsl_vector_complex* src, unsigned size = 0);


/**
* \defgroup AudioFeature Audio Feature Hierarchy
* This hierarchy of classes provides the capability for extracting audio
* features for use in automatic speech recognition.
*/
/*@{*/

/**
* \defgroup FileFeature File Feature.
*/
/*@{*/

// ----- definition for class `FileFeature' -----
//
class FileFeature : public VectorFloatFeatureStream {
 public:
  FileFeature(unsigned sz, const String& nm = "File") :
    VectorFloatFeatureStream(sz, nm), _feature(NULL) { }
 
  virtual ~FileFeature() {
    if (_feature != NULL)
      gsl_matrix_float_free(_feature);
  }

  FileFeature& operator=(const FileFeature& f);

  virtual const gsl_vector_float* next(int frameX = -5);

  unsigned size() const {
    if (_feature->size2 == 0) throw j_error("Matrix not loaded yet.");

    return (unsigned) _feature->size2;
  }

  void bload(const String& fileName, bool old = false);

  void copy(gsl_matrix_float* matrix);

 private:
  gsl_matrix_float*				_feature;
};

typedef Inherit<FileFeature, VectorFloatFeatureStreamPtr> FileFeaturePtr;

/*@}*/

/**
* \defgroup ConversionbitToShort Conversion bit Short
*/
/*@{*/

// ----- definition of 'Conversion24bit2Short' -----
//
class Conversion24bit2Short : public VectorShortFeatureStream {
 public:
  Conversion24bit2Short(VectorCharFeatureStreamPtr& src,
			const String& nm = "Conversion from 24 bit integer to Short") :
    VectorShortFeatureStream(src->size()/3, nm), _src(src) {};
  virtual void reset() { _src->reset(); VectorShortFeatureStream::reset(); }
  
  virtual const gsl_vector_short* next(int frameX = -5);
 private:
  VectorCharFeatureStreamPtr _src;
};

typedef Inherit<Conversion24bit2Short, VectorShortFeatureStreamPtr> Conversion24bit2ShortPtr;

/*@}*/

/**
* \defgroup Conversion24bit2Float Conversion 24 bit 2 Float
*/
/*@{*/

// ----- definition of Conversion24bit2Float -----
//
class Conversion24bit2Float : public VectorFloatFeatureStream {
 public:
  Conversion24bit2Float(VectorCharFeatureStreamPtr& src,
			const String& nm = "Conversion from 24 bit integer to Float") :
    VectorFloatFeatureStream(src->size()/3, nm), _src(src) {};
  virtual void reset() { _src->reset(); VectorFloatFeatureStream::reset(); }
  
  virtual const gsl_vector_float* next(int frameX = -5);
 private:
  VectorCharFeatureStreamPtr _src;
};

typedef Inherit<Conversion24bit2Float, VectorFloatFeatureStreamPtr> Conversion24bit2FloatPtr;


/**
* \defgroup SampleFeature Sample Feature
*/
/*@{*/

// ----- definition for class `SampleFeature' -----
//
class SampleFeature;
typedef Inherit<SampleFeature, VectorFloatFeatureStreamPtr> SampleFeaturePtr;
class SampleFeature : public VectorFloatFeatureStream {
 public:
  SampleFeature(const String& fn = "", unsigned blockLen = 320,
		unsigned shiftLen = 160, bool padZeros = false, const String& nm = "Sample");
  virtual ~SampleFeature();

  unsigned read(const String& fn, int format = 0, int samplerate = 16000,
		int chX = 1, int chN = 1, int cfrom = 0, int to = -1, int outsamplerate = -1, float norm = 0.0);

  void write(const String& fn, int format = sndfile::SF_FORMAT_WAV|sndfile::SF_FORMAT_PCM_16, int sampleRate = -1);

  void cut(unsigned cfrom, unsigned cto);

  void copySamples(SampleFeaturePtr& src, unsigned cfrom, unsigned to);

  unsigned samplesN() const { return _ttlSamples; }

  int getSampleRate() const { return _sampleRate; }

  int getChanN() const { return _nChan; }

  void randomize(int startX, int endX, double sigma2);

  virtual const gsl_vector_float* next(int frameX = -5);

  const gsl_vector_float* data();

  const gsl_vector* dataDouble();
  
  virtual void reset() { _cur = 0; VectorFloatFeatureStream::reset(); _endOfSamples = false; }

  void exit(){ reset(); throw jiterator_error("end of samples!");}

  void zeroMean();

  void addWhiteNoise( float snr );

  void setSamples(const gsl_vector* samples, unsigned sampleRate);
  
protected:
  float*					_samples;
  float                                         _norm;
  unsigned					_ttlSamples;
  const unsigned				_shiftLen;
  int						_sampleRate;
  int                                           _nChan;
  int                                           _format;
  unsigned					_cur;
  bool						_padZeros;
  bool						_endOfSamples;
  gsl_vector_float*                             _cpSamplesF;
  gsl_vector*                                   _cpSamplesD;

  const gsl_rng_type*				_T;
  gsl_rng*					_r;

private:
  SampleFeature(const SampleFeature& s);
  SampleFeature& operator=(const SampleFeature& s);
};


// ----- definition for class `SampleFeatureRunon' -----
//
class SampleFeatureRunon : public SampleFeature {
 public:
  SampleFeatureRunon(const String& fn = "", unsigned blockLen = 320,
		     unsigned shiftLen = 160, bool padZeros = false, const String& nm = "Sample") :
    SampleFeature(fn, blockLen, shiftLen, padZeros, nm) { }
    
  virtual void reset() { VectorFloatFeatureStream::reset(); }

  virtual int frameX() const { return (_cur / _shiftLen) - 1; }

  virtual int frameN() const { return (_ttlSamples / _shiftLen) - 1; }
};

typedef Inherit<SampleFeatureRunon, SampleFeaturePtr> SampleFeatureRunonPtr;

/*@}*/

/**
* \defgroup IterativeSampleFeature Iterative Sample Feature for the single channel data
*/
/*@{*/

// ----- definition for class `IterativeSingleChannelSampleFeature' -----
//
class IterativeSingleChannelSampleFeature : public VectorFloatFeatureStream {
 public:
 public:
  IterativeSingleChannelSampleFeature(unsigned blockLen = 320, const String& nm = "IterativeSingleChannelSampleFeature");
  virtual ~IterativeSingleChannelSampleFeature();

  void read(const String& fileName, int format = 0, int samplerate = 44100, int cfrom = 0, int cto = -1 );

  unsigned samplesN() const { return _ttlSamples; }

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

 private:

  float*				_samples;
  sndfile::SNDFILE*			_sndfile;
  sndfile::SF_INFO			_sfinfo;
  unsigned				_interval;
  unsigned				_blockN;
  unsigned				_sampleN;
  unsigned				_ttlSamples;

  const unsigned			_blockLen;
  unsigned				_cur;
  bool					_last;
  int					_cto;
};

typedef Inherit<IterativeSingleChannelSampleFeature, VectorFloatFeatureStreamPtr> IterativeSingleChannelSampleFeaturePtr;

/*@}*/

/**
* \defgroup IterativeSampleFeature Iterative Sample Feature
*/
/*@{*/

// ----- definition for class `IterativeSampleFeature' -----
//
class IterativeSampleFeature : public VectorFloatFeatureStream {
 public:
  IterativeSampleFeature(unsigned chX, unsigned blockLen = 320, unsigned firstChanX = 0, const String& nm = "Iterative Sample");
  virtual ~IterativeSampleFeature();

  void read(const String& fileName, int format = 0, int samplerate = 44100, int chN = 1, int cfrom = 0, int cto = -1 );

  unsigned samplesN() const { return _ttlSamples; }

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

  void changeFirstChannelID( unsigned firstChanX ){ _firstChanX = firstChanX; }

 private:
  IterativeSampleFeature(const IterativeSampleFeature& s);
  IterativeSampleFeature& operator=(const IterativeSampleFeature& s);

  static float*					_allSamples;
  static sndfile::SNDFILE*			_sndfile;
  static sndfile::SF_INFO			_sfinfo;
  static unsigned				_interval;
  static unsigned				_blockN;
  static unsigned				_sampleN;
  static unsigned				_allSampleN;
  static unsigned				_ttlSamples;

  const unsigned				_blockLen;
  const unsigned				_chanX;
  unsigned				        _firstChanX;
  unsigned					_cur;
  bool						_last;
  int						_cto;
};

typedef Inherit<IterativeSampleFeature, VectorFloatFeatureStreamPtr> IterativeSampleFeaturePtr;

/*@}*/

/**
* \defgroup BlockSizeConversionFeature Block Size Conversion Feature
*/
/*@{*/

// ----- definition for class `BlockSizeConversionFeature' -----
//
class BlockSizeConversionFeature : public VectorFloatFeatureStream {
 public:
  BlockSizeConversionFeature(const VectorFloatFeatureStreamPtr& src,
			     unsigned blockLen = 320,
			     unsigned shiftLen = 160, const String& nm = "Block Size Conversion");

  virtual void reset() { _curIn = _curOut = 0; _srcFrameX = -1; _src->reset();  VectorFloatFeatureStream::reset(); }

  virtual const gsl_vector_float* next(int frameX = -5);

 private:
  void _inputLonger();
  void _outputLonger();

  VectorFloatFeatureStreamPtr			_src;
  const unsigned				_inputLen;
  const unsigned				_blockLen;
  const unsigned				_shiftLen;
  const unsigned				_overlapLen;
  int                                           _srcFrameX;
  unsigned					_curIn;
  unsigned					_curOut;
  const gsl_vector_float*			_srcFeat;
};

typedef Inherit<BlockSizeConversionFeature, VectorFloatFeatureStreamPtr> BlockSizeConversionFeaturePtr;

/*@}*/

/**
* \defgroup BlockSizeConversionFeatureShort Block Size Conversion Feature Short
*/
/*@{*/

// ----- definition for class `BlockSizeConversionFeatureShort' -----
//
class BlockSizeConversionFeatureShort : public VectorShortFeatureStream {
 public:
  BlockSizeConversionFeatureShort(VectorShortFeatureStreamPtr& src,
				  unsigned blockLen = 320,
				  unsigned shiftLen = 160, const String& nm = "Block Size Conversion");

  virtual void reset() { _curIn = _curOut = 0;  _src->reset();  VectorShortFeatureStream::reset(); }

  virtual const gsl_vector_short* next(int frameX = -5);

 private:
  void _inputLonger();
  void _outputLonger();

  VectorShortFeatureStreamPtr			_src;
  const unsigned				_inputLen;
  const unsigned				_blockLen;
  const unsigned				_shiftLen;
  const unsigned				_overlapLen;
  unsigned					_curIn;
  unsigned					_curOut;
  const gsl_vector_short*			_srcFeat;
};

typedef Inherit<BlockSizeConversionFeatureShort, VectorShortFeatureStreamPtr> BlockSizeConversionFeatureShortPtr;

/*@}*/


#ifdef SMARTFLOW

namespace sflib {
#include "sflib.h"
}

/**
* \defgroup SmartFlowFeature Smart Flow Feature
*/
/*@{*/

// ----- definition for class `SmartFlowFeature' -----
//
class SmartFlowFeature : public VectorShortFeatureStream {
 public:
  SmartFlowFeature(sflib::sf_flow_sync* sfflow, unsigned blockLen = 320,
		   unsigned shiftLen = 160, const String& nm = "SmartFloatFeature") :
    VectorShortFeatureStream(blockLen, nm),
    _sfflow(sfflow), _blockLen(blockLen), _shiftLen(shiftLen) { }

  virtual ~SmartFlowFeature() { }

  virtual const gsl_vector_short* next(int frameX = -5);

  virtual void reset() { VectorShortFeatureStream::reset(); }

 private:
  sflib::sf_flow_sync* 				_sfflow;
  struct timespec				_tsin0;
  sflib::sf_counter				_pos0;
  const void* 					_in0;

  const unsigned				_blockLen;
  const unsigned				_shiftLen;
};

typedef Inherit<SmartFlowFeature, VectorShortFeatureStreamPtr> SmartFlowFeaturePtr;

#endif

/*@}*/

/**
* \defgroup PreemphasisFeature Preemphasis Feature
*/
/*@{*/

// ----- definition for class `PreemphasisFeature' -----
//
class PreemphasisFeature : public VectorFloatFeatureStream {
 public:
  PreemphasisFeature(const VectorFloatFeatureStreamPtr& samp, double mu = 0.95, const String& nm = "Preemphasis");
  virtual ~PreemphasisFeature() { }

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { _samp->reset(); VectorFloatFeatureStream::reset(); _prior = 0.0; }

  void nextSpeaker();

 private:
  VectorFloatFeatureStreamPtr			_samp;
  float						_prior;
  const double					_mu;
};

typedef Inherit<PreemphasisFeature, VectorFloatFeatureStreamPtr> PreemphasisFeaturePtr;

/*@}*/

/**
* \defgroup HammingFeatureShort Hamming Feature Short
*/
/*@{*/

// ----- definition for class `HammingFeatureShort' -----
//
class HammingFeatureShort : public VectorFloatFeatureStream {
 public:
  HammingFeatureShort(const VectorShortFeatureStreamPtr& samp, const String& nm = "HammingShort");
  virtual ~HammingFeatureShort() { delete[] _window; }

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { _samp->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorShortFeatureStreamPtr			_samp;
  unsigned					_windowLen;
  double*					_window;
};

typedef Inherit<HammingFeatureShort, VectorFloatFeatureStreamPtr> HammingFeatureShortPtr;

/*@}*/

/**
* \defgroup HammingFeature Hamming Feature
*/
/*@{*/

// ----- definition for class `HammingFeature' -----
//
class HammingFeature : public VectorFloatFeatureStream {
 public:
  HammingFeature(const VectorFloatFeatureStreamPtr& samp, const String& nm = "Hamming");
  virtual ~HammingFeature() { delete[] _window; }

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { _samp->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			_samp;
  unsigned					_windowLen;
  double*					_window;
};

typedef Inherit<HammingFeature, VectorFloatFeatureStreamPtr> HammingFeaturePtr;

/*@}*/

/**
* \defgroup FFTFeature FFT Feature
*/
/*@{*/

// ----- definition for class `FFTFeature' -----
//
class FFTFeature : public VectorComplexFeatureStream {
 public:
  FFTFeature(const VectorFloatFeatureStreamPtr& samp, unsigned fftLen, const String& nm = "FFT");
  virtual ~FFTFeature();

  virtual const gsl_vector_complex* next(int frameX = -5);

  virtual void reset() { _samp->reset(); VectorComplexFeatureStream::reset(); }

  unsigned fftLen()    const { return _fftLen;    }
  unsigned windowLen() const { return _windowLen; }

  unsigned nBlocks()     const { return 4; }
  unsigned subSampRate() const { return 2; }

 private:
  VectorFloatFeatureStreamPtr			_samp;
  unsigned					_fftLen;
  unsigned					_windowLen;
  double*					_samples;

#ifdef HAVE_LIBFFTW3
  fftw_plan					_fftwPlan;
  fftw_complex*					_output;
#endif
};

typedef Inherit<FFTFeature, VectorComplexFeatureStreamPtr> FFTFeaturePtr;

/**
* \defgroup SpectralPowerFeature Spectral Power Feature
*/
/*@{*/

// ----- definition for class `SpectralPowerFeature' -----
//
class SpectralPowerFloatFeature : public VectorFloatFeatureStream {
 public:
  SpectralPowerFloatFeature(const VectorComplexFeatureStreamPtr& fft, unsigned powN = 0, const String& nm = "PowerFloat");

  virtual ~SpectralPowerFloatFeature() { }

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { _fft->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorComplexFeatureStreamPtr			_fft;
};

typedef Inherit<SpectralPowerFloatFeature, VectorFloatFeatureStreamPtr> SpectralPowerFloatFeaturePtr;

/*@}*/


/**
* \defgroup SpectralPowerFeature Spectral Power Feature
*/
/*@{*/

// ----- definition for class `SpectralPowerFeature' -----
//
class SpectralPowerFeature : public VectorFeatureStream {
 public:
  SpectralPowerFeature(const VectorComplexFeatureStreamPtr& fft, unsigned powN = 0, const String& nm = "Power");

  virtual ~SpectralPowerFeature() { }

  virtual const gsl_vector* next(int frameX = -5);

  virtual void reset() { _fft->reset(); VectorFeatureStream::reset(); }

 private:
  VectorComplexFeatureStreamPtr			_fft;
};

typedef Inherit<SpectralPowerFeature, VectorFeatureStreamPtr> SpectralPowerFeaturePtr;

/*@}*/

/**
* \defgroup SignalPowerFeature Signal Power Feature
*/
/*@{*/

// ----- definition for class `SignalPowerFeature' -----
//
static const int ADCRANGE = 65536;

class SignalPowerFeature : public VectorFloatFeatureStream {
 public:
  SignalPowerFeature(const VectorFloatFeatureStreamPtr& samp, const String& nm = "Signal Power") :
    VectorFloatFeatureStream(/* size= */ 1, nm), _samp(samp), _range(float(ADCRANGE) * float(ADCRANGE) / 4.0) { }

  virtual ~SignalPowerFeature() { }

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { _samp->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			_samp;
  double					_range;
};

typedef Inherit<SignalPowerFeature, VectorFloatFeatureStreamPtr> SignalPowerFeaturePtr;

/*@}*/

/**
* \defgroup ALogFeature A-Log Feature
*/
/*@{*/

// ----- definition for class `ALogFeature' -----
//
class ALogFeature : public VectorFloatFeatureStream {
 public:
  ALogFeature(const VectorFloatFeatureStreamPtr& samp, double m = 1.0, double a = 4.0,
	      bool runon = false, const String& nm = "ALog Power") :
    VectorFloatFeatureStream(/* size= */ 1, nm), _samp(samp), _m(m), _a(a),
    _min(HUGE), _max(-HUGE), _minMaxFound(false), _runon(runon) { }

  virtual ~ALogFeature() { }

  virtual const gsl_vector_float* next(int frameX = -5);

  void nextSpeaker() { _min = HUGE; _max = -HUGE; _minMaxFound = false; }

  virtual void reset();

 private:
  void _findMinMax(const gsl_vector_float* block);

  VectorFloatFeatureStreamPtr			_samp;
  double					_m;
  double					_a;
  double					_min;
  double					_max;
  bool						_minMaxFound;
  bool						_runon;
};

typedef Inherit<ALogFeature, VectorFloatFeatureStreamPtr> ALogFeaturePtr;

/*@}*/

/**
* \defgroup NormalizeFeature Normalize Feature
*/
/*@{*/

// ----- definition for class `NormalizeFeature' -----
//
class NormalizeFeature : public VectorFloatFeatureStream {
 public:
  NormalizeFeature(const VectorFloatFeatureStreamPtr& samp, double min = 0.0, double max = 1.0,
		   bool runon = false, const String& nm = "Normalize");

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

  void nextSpeaker() { _xmin = HUGE; _xmax = -HUGE; _minMaxFound = false; }

 private:
  void _findMinMax(const gsl_vector_float* block);

  VectorFloatFeatureStreamPtr			_samp;
  double					_min;
  double					_max;
  double					_range;

  double					_xmin;
  double					_xmax;
  bool						_minMaxFound;
  bool						_runon;
};

typedef Inherit<NormalizeFeature, VectorFloatFeatureStreamPtr> NormalizeFeaturePtr;

/*@}*/

/**
* \defgroup ThresholdFeature Threshold Feature
*/
/*@{*/

// ----- definition for class `ThresholdFeature' -----
//
class ThresholdFeature : public VectorFloatFeatureStream {
 public:
  ThresholdFeature(const VectorFloatFeatureStreamPtr& samp, double value = 0.0, double thresh = 1.0,
		   const String& mode = "upper", const String& nm = "Threshold");

  virtual ~ThresholdFeature() { }

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { _samp->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			_samp;
  double					_value;
  double					_thresh;
  int						_compare;
};

typedef Inherit<ThresholdFeature, VectorFloatFeatureStreamPtr> ThresholdFeaturePtr;

/*@}*/

/**
* \defgroup SpectralResamplingFeature Spectral Resampling Feature
*/
/*@{*/

// ----- definition for class `SpectralResamplingFeature' -----
//
class SpectralResamplingFeature : public VectorFeatureStream {
  static const double SampleRatio;
 public:
  SpectralResamplingFeature(const VectorFeatureStreamPtr& src, double ratio = SampleRatio, unsigned len = 0,
			    const String& nm = "Resampling");

  virtual ~SpectralResamplingFeature();

  virtual const gsl_vector* next(int frameX = -5);

  virtual void reset() { _src->reset(); VectorFeatureStream::reset(); }

 private:
  VectorFeatureStreamPtr			_src;
  const double					_ratio;
  const unsigned				_len;
};

typedef Inherit<SpectralResamplingFeature, VectorFeatureStreamPtr> SpectralResamplingFeaturePtr;

/*@}*/

/**
* \defgroup SamplerateConversionFeature Samplerate Conversion Feature
*/
/*@{*/

// ----- definition for class `SamplerateConversionFeature' -----
//
#ifdef SRCONV

#include <samplerate.h>

class SamplerateConversionFeature : public VectorFloatFeatureStream {
 public:
  // ratio : Equal to input_sample_rate / output_sample_rate.
  SamplerateConversionFeature(const VectorFloatFeatureStreamPtr& src, unsigned sourcerate = 22050, unsigned destrate = 16000,
			      unsigned len = 0, const String& method = "fastest", const String& nm = "SamplerateConversion");
  virtual ~SamplerateConversionFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

 private:
  VectorFloatFeatureStreamPtr			_src;
  SRC_STATE*                                    _state;
  SRC_DATA                                      _data;
  int                                           _error;

  unsigned					_dataInSamplesN;
  unsigned					_dataOutStartX;
  unsigned					_dataOutSamplesN;
};

typedef Inherit<SamplerateConversionFeature, VectorFloatFeatureStreamPtr> SamplerateConversionFeaturePtr;

#endif

/*@}*/

/**
* \defgroup VTLNFeature VTLN Feature
*/
/*@{*/

// ----- definition for class `VTLNFeature' -----
//
// -------------------------------------------
// Piecewise linear: Y = X/Ratio  for X < edge
// -------------------------------------------
class VTLNFeature : public VectorFeatureStream {
 public:
  VTLNFeature(const VectorFeatureStreamPtr& pow,
	      unsigned coeffN = 0, double ratio = 1.0, double edge = 1.0, int version = 1,
	      const String& nm = "VTLN");
  virtual ~VTLNFeature() {  }

  virtual const gsl_vector* next(int frameX = -5);

  virtual void reset() { _pow->reset(); VectorFeatureStream::reset(); }

  // specify the warp factor
  void warp(double w) { _ratio = w; }

  void matrix(gsl_matrix* mat) const;

private:
  virtual const gsl_vector* nextFF(int frameX);
  virtual const gsl_vector* nextOrg(int frameX);

 private:
  VectorFeatureStreamPtr _pow;
  double	         _ratio;
  const double		 _edge;
  const int              _version;
  gsl_vector*            _auxV;
};

typedef Inherit<VTLNFeature, VectorFeatureStreamPtr> VTLNFeaturePtr;

/*@}*/

/**
* \defgroup MelFeature Mel Feature
*/
/*@{*/

// ----- definition for class 'MelFeature' -----
//
class MelFeature : public VectorFeatureStream {
  class _SparseMatrix {
  public:
    _SparseMatrix(unsigned m, unsigned n, unsigned version);
    ~_SparseMatrix();
    
    int _version; // _SparseMatrix version number, 1:Org, 2:Friedich's changes

    void melScale(int powN,  float rate, float low, float up, int filterN);
    void melScaleOrg(int powN,  float rate, float low, float up, int filterN);
    void melScaleFF(int powN,  float rate, float low, float up, int filterN);
    gsl_vector* fmatrixBMulotOrg( gsl_vector* C, const gsl_vector* A) const;
    gsl_vector* fmatrixBMulotFF( gsl_vector* C, const gsl_vector* A) const;
    gsl_vector* fmatrixBMulot( gsl_vector* C, const gsl_vector* A) const;
    void readBuffer(const String& fb);

    void matrix(gsl_matrix* mat) const;

  private:
    void _alloc(unsigned m, unsigned n);
    void _dealloc();

    float _mel(float hz);
    float _hertz(float m);

    float**					_data;
    unsigned					_m;
    unsigned					_n;
    unsigned*					_offset;	// offset
    unsigned*					_coefN;		// number of coefficients
    float					_rate;		// sampling rate in Hz
  };

 public:
  MelFeature(const VectorFeatureStreamPtr& mag, int powN = 0,
	     float rate = 16000.0, float low = 0.0, float up = 0.0,
	     unsigned filterN = 30, unsigned version = 1, const String& nm = "MelFFT");

  virtual ~MelFeature();

  virtual const gsl_vector* next(int frameX = -5);

  virtual void reset() { _mag->reset(); VectorFeatureStream::reset(); }

  void read(const String& fileName);

  void matrix(gsl_matrix* mat) const { _mel.matrix(mat); }

 private:
  // gsl_vector* _fmatrixBMulot(gsl_vector* C, const gsl_vector* A, FBMatrix* B) const;

  const unsigned				_nmel;
  const unsigned				_powN;
  VectorFeatureStreamPtr			_mag;
  _SparseMatrix					_mel;
};

typedef Inherit<MelFeature, VectorFeatureStreamPtr> MelFeaturePtr;


// ----- definition for class 'SphinxMelFeature' -----
//
class SphinxMelFeature : public VectorFeatureStream {
  class _Boundary {
  public:
    _Boundary(unsigned min_k, unsigned max_k)
      : _min_k(min_k), _max_k(max_k) { }

    unsigned					_min_k;
    unsigned					_max_k;
  };
  typedef vector<_Boundary>			_Boundaries;

 public:
  SphinxMelFeature(const VectorFeatureStreamPtr& mag, unsigned fftN = 512, unsigned powerN = 0,
		   float sampleRate = 16000.0, float lowerF = 0.0, float upperF = 0.0,
		   unsigned filterN = 30, const String& nm = "Sphinx Mel Filter Bank");

  virtual ~SphinxMelFeature();

  virtual const gsl_vector* next(int frameX = -5);

  virtual void reset() { _mag->reset(); VectorFeatureStream::reset(); }

  void read(const String& fileName);

 private:
  static double _melFrequency(double frequency);
  static double _melInverseFrequency(double frequency);

  const unsigned				_fftN;
  const unsigned				_filterN;
  const unsigned				_powerN;
  const double					_sampleRate;
  VectorFeatureStreamPtr			_mag;
  gsl_matrix*					_filters;
  _Boundaries					_boundaries;
};

typedef Inherit<SphinxMelFeature, VectorFeatureStreamPtr> SphinxMelFeaturePtr;

/*@}*/

/**
* \defgroup LogFeature Log Feature
*/
/*@{*/

// ----- definition for class `LogFeature' -----
//
class LogFeature : public VectorFloatFeatureStream {
 public:
  LogFeature(const VectorFeatureStreamPtr& mel, double m = 1.0, double a = 1.0,
	     bool sphinxFlooring = false, const String& nm = "LogMel");
  virtual ~LogFeature() { }

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { _mel->reset(); VectorFloatFeatureStream::reset(); }

 private:
  const unsigned				_nmel;
  VectorFeatureStreamPtr			_mel;
  const double					_m;
  const double					_a;
  const bool					_SphinxFlooring;
};

typedef Inherit<LogFeature, VectorFloatFeatureStreamPtr> LogFeaturePtr;

// ----- definition for class `FloatToDoubleConversionFeature' -----
//
class FloatToDoubleConversionFeature : public VectorFeatureStream {
 public:
  FloatToDoubleConversionFeature(const VectorFloatFeatureStreamPtr& src, const String& nm = "Float to Double Conversion") : VectorFeatureStream(src->size(), nm), _src(src) {};

  virtual ~FloatToDoubleConversionFeature() { }

  virtual const gsl_vector* next(int frameX = -5);

  virtual void reset() { _src->reset(); VectorFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr     _src;

};

typedef Inherit<FloatToDoubleConversionFeature, VectorFeatureStreamPtr> FloatToDoubleConversionFeaturePtr;

/*@}*/

/**
* \defgroup CepstralFeature Cepstral Feature
*/
/*@{*/

// ----- definition for class `CepstralFeature' -----
//
// type:
//   0 = 
//   1 = Type 2 DCT
//   2 = Sphinx Legacy
class CepstralFeature : public VectorFloatFeatureStream {
 public:
  CepstralFeature(const VectorFloatFeatureStreamPtr& mel, unsigned ncep = 13,
		  int type = 1, const String& nm = "Cepstral");

  virtual ~CepstralFeature() { gsl_matrix_float_free(_cos); }

  virtual const gsl_vector_float* next(int frameX = -5);

  gsl_matrix* matrix() const;

  virtual void reset() { _mel->reset(); VectorFloatFeatureStream::reset(); }
  
 private:
  void _sphinxLegacy();

  const unsigned				_ncep;
  gsl_matrix_float*				_cos;
  VectorFloatFeatureStreamPtr			_mel;
};

typedef Inherit<CepstralFeature, VectorFloatFeatureStreamPtr> CepstralFeaturePtr;

/*@}*/

/**
* \defgroup MeanSubtractionFeature Mean Subtraction Feature
*/
/*@{*/

// ----- definition for class `MeanSubtractionFeature' -----
//
class MeanSubtractionFeature : public VectorFloatFeatureStream {
 public:
  MeanSubtractionFeature(const VectorFloatFeatureStreamPtr& src, const VectorFloatFeatureStreamPtr& weight = NULL,
			 double devNormFactor = 0.0, bool runon = false, const String& nm = "Mean Subtraction");

  virtual ~MeanSubtractionFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

  void nextSpeaker();

  const gsl_vector_float* mean() const { return _mean; }

  void write(const String& fileName, bool variance = false) const;

 private:
  const gsl_vector_float* _nextRunon(int frameX);
  const gsl_vector_float* _nextBatch(int frameX);
  void _calcMeanVariance();
  void _normalize(const gsl_vector_float* srcVec);

  static const float				_varianceFloor;
  static const float				_beforeWgt;
  static const float				_afterWgt;
  static const unsigned				_framesN2Change;

  VectorFloatFeatureStreamPtr			_src;
  VectorFloatFeatureStreamPtr			_wgt;

  gsl_vector_float*				_mean;
  gsl_vector_float*				_var;
  const double					_devNormFactor;
  unsigned					_framesN;

  bool						_runon;
  bool						_meanVarianceFound;
};

typedef Inherit<MeanSubtractionFeature, VectorFloatFeatureStreamPtr> MeanSubtractionFeaturePtr;

/*@}*/

/**
* \defgroup FileMeanSubtractionFeature File Mean Subtraction Feature
*/
/*@{*/

// ----- definition for class `FileMeanSubtractionFeature' -----
//
class FileMeanSubtractionFeature : public VectorFloatFeatureStream {
 public:
  FileMeanSubtractionFeature(const VectorFloatFeatureStreamPtr& src,
			     double devNormFactor = 0.0, const String& nm = "File Mean Subtraction");

  virtual ~FileMeanSubtractionFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

  void read(const String& fileName, bool variance = false);

 private:
  static const float				_varianceFloor;

  VectorFloatFeatureStreamPtr			_src;
  gsl_vector_float*				_mean;
  gsl_vector_float*				_variance;
  const double					_devNormFactor;
};

typedef Inherit<FileMeanSubtractionFeature, VectorFloatFeatureStreamPtr> FileMeanSubtractionFeaturePtr;

/*@}*/

/**
* \defgroup AdjacentFeature Adjacent Feature
*/
/*@{*/

// ----- definition for class `AdjacentFeature' -----
//
class AdjacentFeature : public VectorFloatFeatureStream {
 public:
  AdjacentFeature(const VectorFloatFeatureStreamPtr& single, unsigned delta = 5,
		  const String& nm = "Adjacent");

  virtual ~AdjacentFeature() { }

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

 private:
  void _bufferNextFrame(int frameX);

  const unsigned				_delta;
  VectorFloatFeatureStreamPtr			_single;
  const unsigned				_singleSize;
  const unsigned				_plen;
  unsigned					_framesPadded;
};

typedef Inherit<AdjacentFeature, VectorFloatFeatureStreamPtr> AdjacentFeaturePtr;

/*@}*/

/**
* \defgroup LinearTransformFeature Linear Transform Feature
*/
/*@{*/

// ----- definition for class `LinearTransformFeature' -----
//
class LinearTransformFeature : public VectorFloatFeatureStream {
 public:
#if 0
  LinearTransformFeature(const VectorFloatFeatureStreamPtr& src,
			 gsl_matrix_float* mat = NULL, unsigned sz = 0, const String& nm = "Transform");
#else
  LinearTransformFeature(const VectorFloatFeatureStreamPtr& src, unsigned sz = 0, const String& nm = "Transform");
#endif

  virtual ~LinearTransformFeature() { gsl_matrix_float_free(_trans); }

  virtual const gsl_vector_float* next(int frameX = -5);

  gsl_matrix_float* matrix() const;

  void load(const String& fileName, bool old = false);

  void identity();

  virtual void reset() { _src->reset(); VectorFloatFeatureStream::reset(); }

 protected:
  VectorFloatFeatureStreamPtr			_src;
  gsl_matrix_float*				_trans;
};

typedef Inherit<LinearTransformFeature, VectorFloatFeatureStreamPtr> LinearTransformFeaturePtr;

/*@}*/

/**
* \defgroup StorageFeature Storage Feature
*/
/*@{*/

// ----- definition for class `StorageFeature' -----
//
class StorageFeature : public VectorFloatFeatureStream {
  typedef vector<gsl_vector_float*>	_StorageVector;
  static const int MaxFrames;
 public:
  StorageFeature(const VectorFloatFeatureStreamPtr& src, const String& nm = "Storage");

  virtual ~StorageFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { _src->reset(); VectorFloatFeatureStream::reset(); }

  void write(const String& fileName, bool plainText = false) const;
  void read(const String& fileName);
  int evaluate();

 private:
  VectorFloatFeatureStreamPtr			_src;
  _StorageVector				_frames;
};

typedef Inherit<StorageFeature, VectorFloatFeatureStreamPtr> StorageFeaturePtr;

/**
* \defgroup StaticStorageFeature Static Storage Feature
*/
/*@{*/

// ----- definition for class `StaticStorageFeature' -----
//
class StaticStorageFeature : public VectorFloatFeatureStream {
  typedef vector<gsl_vector_float*> _StorageVector;
  static const int MaxFrames;
 public:
  StaticStorageFeature(unsigned dim, const String& nm = "Static Storage");

  virtual ~StaticStorageFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { VectorFloatFeatureStream::reset(); }

  //void write(const String& fileName) const;
  void read(const String& fileName);
  int evaluate();
  unsigned currentNFrames() const { return _frameX; };

 private:
  //VectorFloatFeatureStreamPtr     _src;
  _StorageVector        _frames;
  int                  _nFrames;
};

typedef Inherit<StaticStorageFeature, VectorFloatFeatureStreamPtr> StaticStorageFeaturePtr;

/*@}*/

/**
* \defgroup CircularStorageFeature Circular Storage Feature
*/
/*@{*/

// ----- definition for class `CircularStorageFeature' -----
//
class CircularStorageFeature : public VectorFloatFeatureStream {
  typedef vector<gsl_vector_float*>	_StorageVector;
  static const int MaxFrames;
 public:
  CircularStorageFeature(const VectorFloatFeatureStreamPtr& src, unsigned framesN = 3,
			 const String& nm = "Circular Storage");

  virtual ~CircularStorageFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

 private:
  unsigned _getIndex(int diff) const;

  VectorFloatFeatureStreamPtr			_src;
  const unsigned				_framesN;
  _StorageVector				_frames;
  unsigned					_pointerX;
};

typedef Inherit<CircularStorageFeature, VectorFloatFeatureStreamPtr> CircularStorageFeaturePtr;

/*@}*/

/**
* \defgroup FilterFeature Filter Feature
*/
/*@{*/

// ----- definition for class `FilterFeature' -----
//
class FilterFeature : public VectorFloatFeatureStream {
  class _Buffer {
  public:
    _Buffer(unsigned len, unsigned nsamp)
      : _len(len), _nsamp(nsamp), _offset(int((_nsamp - 1) / 2)),
      _zero(0), _samples(new gsl_vector_float*[_nsamp])
    {
      assert (_nsamp % 2 == 1);
      for (unsigned i = 0; i < _nsamp; i++)
	_samples[i] = gsl_vector_float_calloc(_len);
    }
    ~_Buffer()
    {
      for (unsigned i = 0; i < _nsamp; i++)
	gsl_vector_float_free(_samples[i]);
      delete[] _samples;
    }

    const gsl_vector_float* sample(unsigned timeX) const {
      return _samples[_index(timeX)];
    }

    const double sample(int timeX, unsigned binX) const {
      unsigned idx = _index(timeX);
      const gsl_vector_float* vec = _samples[idx];
      return gsl_vector_float_get(vec, binX);
    }

    void nextSample(const gsl_vector_float* s = NULL) {
      _zero = (_zero + 1) % _nsamp;
      gsl_vector_float* nextBlock = _samples[(_zero + _offset) % _nsamp];

      if (s == NULL) {

	gsl_vector_float_set_zero(nextBlock);

      } else {

	assert( s->size == _len );
	gsl_vector_float_memcpy(nextBlock, s);

      }
    }

    void zero() {
      for (unsigned i = 0; i < _nsamp; i++)
	gsl_vector_float_set_zero(_samples[i]);
      _zero = _nsamp - _offset;
    }

    void print() const {
      for (int i = -_offset; i <= _offset; i++)
	printf("        %4d", i);
      printf("\n     --------------------------------------------------------------------------------\n");
      for (unsigned l = 0; l < _len; l++) {
	for (int i = -_offset; i <= _offset; i++)
	  printf("  %10.4f", sample(i, l));
	printf("\n");
      }
    }

  private:
    unsigned _index(int idx) const {
      assert ( abs(idx) <= _offset);
      unsigned ret = (_zero + _nsamp + idx) % _nsamp;
      return ret;
    }

    const unsigned				_len;
    const unsigned				_nsamp;
    int						_offset;
    unsigned					_zero;		// index of most recent sample
    gsl_vector_float**				_samples;
  };

 public:
  FilterFeature(const VectorFloatFeatureStreamPtr& src, gsl_vector* coeffA,
		const String& nm = "Filter");
  virtual ~FilterFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

  void printBuffer() const { _buffer.print(); }

 private:
  void _bufferNextFrame(int frameX);

  VectorFloatFeatureStreamPtr			_src;
  unsigned					_lenA;
  gsl_vector*					_coeffA;
  int						_offset;
  _Buffer					_buffer;
  unsigned					_framesPadded;
};

typedef Inherit<FilterFeature, VectorFloatFeatureStreamPtr> FilterFeaturePtr;

/*@}*/

/**
* \defgroup MergeFeature Merge Feature
*/
/*@{*/

// ----- definition for class `MergeFeature' -----
//
class MergeFeature : public VectorFloatFeatureStream {
  typedef list<VectorFloatFeatureStreamPtr>	_FeatureList;
  typedef _FeatureList::iterator		_FeatureListIterator;
  typedef _FeatureList::const_iterator		_FeatureListConstIterator;
 public:
  MergeFeature(VectorFloatFeatureStreamPtr& stat,
	       VectorFloatFeatureStreamPtr& delta,
	       VectorFloatFeatureStreamPtr& deltaDelta,
	       const String& nm = "Merge");

  virtual ~MergeFeature() { }

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

 private:
  _FeatureList					_flist;
};

typedef Inherit<MergeFeature, VectorFloatFeatureStreamPtr> MergeFeaturePtr;

/**
* \defgroup MultiModalFeature 
*/
/*@{*/

// ----- definition for class `MultiModalFeature MergeFeature' -----
//
class MultiModalFeature : public VectorFloatFeatureStream {
  typedef list<VectorFloatFeatureStreamPtr>	_FeatureList;
  typedef _FeatureList::iterator		_FeatureListIterator;
  typedef _FeatureList::const_iterator		_FeatureListConstIterator;
 public:
  MultiModalFeature(unsigned nModality, unsigned totalVecSize, const String& nm = "Multi");

  ~MultiModalFeature();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

  void addModalFeature( VectorFloatFeatureStreamPtr &feature, unsigned samplePeriodinNanoSec=1 );
  
 private:
  unsigned     *_samplePeriods;  /* sample period (in nano sec.) */
  unsigned     _minSamplePeriod;
  unsigned     _nModality;
  unsigned     _currVecSize;
  _FeatureList _flist;
};

typedef Inherit<MultiModalFeature, VectorFloatFeatureStreamPtr> MultiModalFeaturePtr;

/*@}*/

/**
* \defgroup FeatureSet Feature Set
*/
/*@{*/

// ----- definition for class `FeatureSet' -----
//
class FeatureSet {
  typedef List <VectorFloatFeatureStreamPtr>	_List;
 public:
  FeatureSet(const String& nm = "FeatureSet") :
    _name(nm), _list(nm) { }

  const String& name() const { return _name; }

  void add(VectorFloatFeatureStreamPtr& feat) { _list.add(feat->name(), feat); }
  VectorFloatFeatureStreamPtr& feature(const String& nm) { return _list[nm]; }

 private:
  const String		_name;
  _List			_list;
};

typedef refcount_ptr<FeatureSet>	FeatureSetPtr;

/*@}*/

void writeGSLMatrix(const String& fileName, const gsl_matrix* mat);


#ifdef JACK
#include <vector>
#include <jack/jack.h>
#include <jack/ringbuffer.h>

typedef struct {
  jack_port_t *port;
  jack_ringbuffer_t *buffer;
  unsigned buffersize;
  unsigned overrun;
  bool can_process;
} jack_channel_t;

/**
* \defgroup Jack Jack Object
*/
/*@{*/

class Jack {
 public:
  Jack(const String& nm);
  ~Jack();
  jack_channel_t* addPort(unsigned buffersize, const String& connection, const String& nm);
  void start(void) { can_capture = true; };
  unsigned getSampleRate() { return (unsigned)jack_get_sample_rate(client); }

 private:
  int process_callback (jack_nframes_t nframes);

  static int _process_callback(jack_nframes_t nframes, void *arg) {
    return static_cast<Jack *> (arg)->process_callback (nframes);
  }

  void shutdown_callback (void);
  static void _shutdown_callback(void *arg) {
    static_cast<Jack *> (arg)->shutdown_callback();
  }

  jack_client_t*				client;
  volatile bool					can_capture;
  volatile bool					can_process;
  vector<jack_channel_t*>			channel;
};

typedef refcount_ptr<Jack>			JackPtr;

/*@}*/

/**
* \defgroup JackFeature Jack Feature
*/
/*@{*/

// ----- definition for class `JackFeature' -----
//
class JackFeature;
typedef Inherit<JackFeature, VectorFloatFeatureStreamPtr> JackFeaturePtr;
class JackFeature : public VectorFloatFeatureStream {
 public:
  JackFeature(JackPtr& jack, unsigned blockLen, unsigned buffersize,
	      const String& connection, const String& nm);

  virtual ~JackFeature() { }

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { VectorFloatFeatureStream::reset(); }

 private:
  JackPtr					_jack;
  jack_channel_t*				channel;
};

#endif


/*@}*/

/**
* \defgroup ZeroCrossingRateHammingFeature Zero Crossing Rate Hamming Feature
*/
/*@{*/

// ----- definition for class `ZeroCrossingRateHammingFeature' -----
//
class ZeroCrossingRateHammingFeature : public VectorFloatFeatureStream {
 public:
  ZeroCrossingRateHammingFeature(const VectorFloatFeatureStreamPtr& samp, const String& nm = "Zero Crossing Rate Hamming");
  virtual ~ZeroCrossingRateHammingFeature() { delete[] _window; }

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { _samp->reset(); VectorFloatFeatureStream::reset(); }

 private:
  VectorFloatFeatureStreamPtr			_samp;
  unsigned					_windowLen;
  double*					_window;
};

typedef Inherit<ZeroCrossingRateHammingFeature, VectorFloatFeatureStreamPtr> ZeroCrossingRateHammingFeaturePtr;

/*@}*/

/**
* \defgroup YINPitchFeature YIN Pitch Feature
*/
/*@{*/

// ----- definition for class `YINPitchFeature' -----
//
class YINPitchFeature : public VectorFloatFeatureStream {
 public:
  YINPitchFeature(const VectorFloatFeatureStreamPtr& samp, unsigned samplerate = 16000, float threshold = 0.5, const String& nm = "YIN Pitch");
  virtual ~YINPitchFeature() { }

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { _samp->reset(); VectorFloatFeatureStream::reset(); }

 private:
  float	_getPitch(const gsl_vector_float *input, gsl_vector_float *yin, float tol);
  VectorFloatFeatureStreamPtr			_samp;
  unsigned _sr;  
  float	_tr;
};

typedef Inherit<YINPitchFeature, VectorFloatFeatureStreamPtr> YINPitchFeaturePtr;

/*@}*/

/**
* \defgroup SpikeFilter Spike Filter
*/
/*@{*/

// ----- definition for class `SpikeFilter' -----
//
class SpikeFilter : public VectorFloatFeatureStream {
 public:
  SpikeFilter(VectorFloatFeatureStreamPtr& src, unsigned tapN = 3, const String& nm = "Spike Filter");
  virtual ~SpikeFilter();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset() { _src->reset(); VectorFloatFeatureStream::reset(); }
  
private:
  VectorFloatFeatureStreamPtr		_src;
  const unsigned			_adcN;
  const unsigned			_queueN;
  float*				_queue;
  const unsigned			_windowN;
  float*				_window;
};

typedef Inherit<SpikeFilter, VectorFloatFeatureStreamPtr> SpikeFilterPtr;

/*@}*/

/**
* \defgroup SpikeFilter2 Spike Filter 2
*/
/*@{*/

// ----- definition for class `SpikeFilter2' -----
//
class SpikeFilter2 : public VectorFloatFeatureStream {
 public:
  SpikeFilter2(VectorFloatFeatureStreamPtr& src, 
	       unsigned width = 3, float maxslope = 7000.0, float startslope = 100.0, float thresh = 15.0, float alpha = 0.2, unsigned verbose = 1,
	       const String& nm = "Spike Filter 2");

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

  unsigned spikesN() const { return _count; }

private:
  VectorFloatFeatureStreamPtr		_src;
  const unsigned 			_adcN;
  const unsigned			_width;
  const float				_maxslope;
  const float				_startslope;
  const float				_thresh;
  float 				_alpha;
  float 				_beta;
  float					_meanslope;
  unsigned				_count;
  const unsigned			_verbose;
};

typedef Inherit<SpikeFilter2, VectorFloatFeatureStreamPtr> SpikeFilter2Ptr;

/*@}*/

namespace sndfile {
#include <sndfile.h>

/**
* \defgroup SoundFile Sound File
*/
/*@{*/

// ----- definition for class `SoundFile' -----
// 
class SoundFile {
 public:
  SoundFile(const String& fn,
	    int mode = sndfile::SFM_RDWR,
	    int format = sndfile::SF_FORMAT_WAV|sndfile::SF_FORMAT_PCM_16,
	    int samplerate = 16000,
	    int channels = 1,
	    bool normalize = false);
  ~SoundFile() { sf_close(_sndfile); }
  sf_count_t frames() const { return _sfinfo.frames; }
  int samplerate() const { return _sfinfo.samplerate; }
  int channels() const { return _sfinfo.channels; }
  int format() const { return _sfinfo.format; }
  int sections() const { return _sfinfo.sections; }
  int seekable() const { return _sfinfo.seekable; }
  sf_count_t readf(float *ptr, sf_count_t frames)
    { return sf_readf_float(_sndfile, ptr, frames); }
  sf_count_t writef(float *ptr, sf_count_t frames)
    { return sf_writef_float(_sndfile, ptr, frames); }
  sf_count_t read(float *ptr, sf_count_t items)
    { return sf_read_float(_sndfile, ptr, items); }
  sf_count_t write(float *ptr, sf_count_t items)
    { return sf_write_float(_sndfile, ptr, items); }
  sf_count_t seek(sf_count_t frames, int whence = SEEK_SET)
    { return sf_seek(_sndfile, frames, whence); }
 private:
  SNDFILE* _sndfile;
  SF_INFO _sfinfo;
};
}
typedef refcount_ptr<sndfile::SoundFile>	SoundFilePtr;

/*@}*/

/**
* \defgroup DirectSampleFeature Direct Sample Feature
*/
/*@{*/

// ----- definition for class `DirectSampleFeature' -----
// 
class DirectSampleFeature;
typedef Inherit<DirectSampleFeature, VectorFloatFeatureStreamPtr> DirectSampleFeaturePtr;
class DirectSampleFeature : public VectorFloatFeatureStream {
 public:
  DirectSampleFeature(const SoundFilePtr &sndfile,
		      unsigned blockLen = 320,
		      unsigned start = 0,
		      unsigned end = (unsigned)-1,
		      const String& nm = "DirectSample");
  virtual ~DirectSampleFeature() {}
  virtual const gsl_vector_float* next(int frameX = -5);
  int sampleRate() const { return _sndfile->samplerate(); }
  int channels() const { return _sndfile->channels(); }
  void setRegion(unsigned start = 0, unsigned end = (unsigned)-1) {
    _start = start;
    _end = end;
  }
  virtual void reset() {
    _sndfile->seek(_start, SEEK_SET);
    _cur = 0;
    VectorFloatFeatureStream::reset();
  }  
 private:
  SoundFilePtr _sndfile;
  unsigned _blockLen;
  unsigned _start;
  unsigned _end;
  unsigned _cur;
};

/*@}*/

/**
* \defgroup DirectSampleOutputFeature Direct Sample Output Feature
*/
/*@{*/

// ----- definition for class `DirectSampleOutputFeature' -----
// 
class DirectSampleOutputFeature;
typedef Inherit<DirectSampleOutputFeature, VectorFloatFeatureStreamPtr> DirectSampleOutputFeaturePtr;
class DirectSampleOutputFeature : public VectorFloatFeatureStream {
 public:
  DirectSampleOutputFeature(const VectorFloatFeatureStreamPtr& src,
			    const SoundFilePtr &sndfile,
			    const String& nm = "DirectSampleOutput");
  virtual ~DirectSampleOutputFeature() {}
  virtual const gsl_vector_float* next(int frameX = -5);
  virtual void reset() { _src->reset(); _sndfile->seek(0, SEEK_SET); VectorFloatFeatureStream::reset(); }  
 private:
  VectorFloatFeatureStreamPtr _src;
  SoundFilePtr _sndfile;
  unsigned _blockLen;
};

/*@}*/

/**
* \defgroup ChannelExtractionFeature Channel Extraction Feature
*/
/*@{*/

// ----- definition for class `ChannelExtractionFeature' -----
// 
class ChannelExtractionFeature;
typedef Inherit<ChannelExtractionFeature, VectorFloatFeatureStreamPtr> ChannelExtractionFeaturePtr;
class ChannelExtractionFeature : public VectorFloatFeatureStream {
 public:
  ChannelExtractionFeature(const VectorFloatFeatureStreamPtr& src,
			   unsigned chX = 0,
			   unsigned chN = 1,
			   const String& nm = "ChannelExtraction")
    : VectorFloatFeatureStream(src->size()/chN, nm), _src(src), _chX(chX), _chN(chN)
    {
      assert(chX < chN); 
      assert((src->size() % chN) == 0);
    }
  virtual ~ChannelExtractionFeature() {}
  virtual const gsl_vector_float* next(int frameX = -5);
  virtual void reset() { _src->reset(); VectorFloatFeatureStream::reset(); }  
 private:
  VectorFloatFeatureStreamPtr _src;
  unsigned _chX;
  unsigned _chN;
};

/*@}*/

/**
* \defgroup SignalInterferenceFeature Signal Interference Feature
*/
/*@{*/

// ----- definition for class SignalInterferenceFeature -----
//
class SignalInterferenceFeature;
typedef Inherit<SignalInterferenceFeature, VectorFloatFeatureStreamPtr> SignalInterferenceFeaturePtr;
class SignalInterferenceFeature : public VectorFloatFeatureStream{
public:
  SignalInterferenceFeature(VectorFloatFeatureStreamPtr& signal, VectorFloatFeatureStreamPtr& interference,
			    double dBInterference = 0.0, unsigned blockLen = 512, const String& nm = "Signal Interference");

  virtual ~SignalInterferenceFeature() {};
  virtual const gsl_vector_float* next(int frameX = -5);
  virtual void reset() { _signal->reset(); _interference->reset(); VectorFloatFeatureStream::reset(); }  

private:
  VectorFloatFeatureStreamPtr   	_signal;
  VectorFloatFeatureStreamPtr		_interference;
  const double				_level;
};

/*@}*/

/**
* \defgroup AmplificationFeature Amplification Feature
*/
/*@{*/

// ----- definition for class `AmplificationFeature' -----
// 
class AmplificationFeature;
typedef Inherit<AmplificationFeature, VectorFloatFeatureStreamPtr> AmplificationFeaturePtr;
class AmplificationFeature : public VectorFloatFeatureStream {
 public:
  AmplificationFeature(const VectorFloatFeatureStreamPtr& src,
			   double amplify = 1.0,
			   const String& nm = "Amplification")
    : VectorFloatFeatureStream(src->size(), nm), _src(src), _amplify(amplify)
    {}
  virtual ~AmplificationFeature() {}
  virtual const gsl_vector_float* next(int frameX = -5);
  virtual void reset() { _src->reset(); VectorFloatFeatureStream::reset(); }  
 private:
  VectorFloatFeatureStreamPtr _src;
  double _amplify;
};

// ----- definition for class `WriteSoundFile' -----
//
class WriteSoundFile {
public:
  WriteSoundFile(const String& fn, int sampleRate, int nChan=1, int format = sndfile::SF_FORMAT_WAV|sndfile::SF_FORMAT_PCM_32);
  ~WriteSoundFile();
  int write( gsl_vector *vector );
  int writeInt( gsl_vector *vector );
  int writeShort( gsl_vector *vector );
  int writeFloat( gsl_vector *vector );

private:
  sndfile::SNDFILE* _sndfile;
  sndfile::SF_INFO _sfinfo;
};

typedef refcount_ptr<WriteSoundFile> WriteSoundFilePtr;

// ----- definition for class `WriteHTKFeatureFile' -----
// 
class WriteHTKFeatureFile {
public:
  WriteHTKFeatureFile( const String& outputfile,
		       int nSamples, 
		       int sampPeriod=160 /* 16msec*/, 
		       short sampSize=2, 
		       short parmKind=9,
		       bool isBigEndian=false, /* Linux, Windows & Mac are Little endian OSes */
		       const String& nm = "WriteHTKFeatureFile");
  ~WriteHTKFeatureFile();
  void write( gsl_vector *vector );
private:
  unsigned _bufSize;
  float *_buf;
  FILE  *_fp;
  bool _isBigEndian;
};

typedef refcount_ptr<WriteHTKFeatureFile> WriteHTKFeatureFilePtr;

// ----- definition for class `HTKFeature' -----
// 
class HTKFeature : public VectorFeatureStream {
 public:
  HTKFeature(const String& inputfile="", int vecSize = 256, 
	     bool isBigEndian=false, /* Linux, Windows & Mac are Little endian OSes */	     
	     const String& nm = "HTKFeature");
  ~HTKFeature();
  virtual const gsl_vector* next(int frameX = -5);
  virtual void reset() { VectorFeatureStream::reset(); }  
  bool read( const String& inputfile, bool isBigEndian=false );
  int   samplesN(){ return _nSamples; }
  int   samplePeriod(){ return _sampPeriod; }
  short sampleSize(){ return _sampSize; }
  short parmKind(){ return _parmKind; }

private:
  float *_buf;
  FILE  *_fp;
  bool _isBigEndian;
  int  _nSamples;
  int  _sampPeriod;
  short _sampSize;
  short _parmKind;
};

typedef Inherit<HTKFeature, VectorFeatureStreamPtr> HTKFeaturePtr;

#endif
