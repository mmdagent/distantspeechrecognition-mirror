//                              -*- C++ -*-
//
//                                Nemesis
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.modulated
//  Purpose: Cosine modulated analysis and synthesis filter banks.
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


#ifndef _modulated_h_
#define _modulated_h_

#ifdef BTK_MEMDEBUG
#include "memcheck/memleakdetector.h"
#endif

#include <stdio.h>
#include <assert.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include "common/jexception.h"

#include "stream/stream.h"
//#include "btk.h"

#ifdef HAVE_LIBFFTW3
#include <fftw3.h>
#endif

inline int powi(int x, int p)
{
  if(p == 0) return 1;
  if(x == 0 && p > 0) return 0;
  if(p < 0) {assert(x == 1 || x == -1); return (-p % 2) ? x : 1;}
	
  int r = 1;
  for(;;) {
    if(p & 1) r *= x;
    if((p >>= 1) == 0)	return r;
    x *= x;
  }
}

/**
* \defgroup FilterBanks Filter Banks
* This hierarchy of classes provides the capability to divide
* signal into 'M' subbands and then resynthesize the original time-domain signal.
*/
/*@{*/

// ----- definition for class `BaseFilterBank' -----
//
class BaseFilterBank {
 public:
  virtual ~BaseFilterBank();

  virtual void reset() = 0;

 protected:
  class _RealBuffer {
  public:
    /*
      @brief Construct a circular buffer to keep samples periodically.
             It keeps nsamp arrays which is completely updated with the period 'nsamp'.
             Each array holds actual values of the samples.
      @param unsigned len [in] The size of each array
      @param unsigned nsamp [in] The period of the circular buffer
    */
    _RealBuffer(unsigned len, unsigned nsamp)
      : _len(len), _nsamp(nsamp), _zero(_nsamp - 1), _samples(new gsl_vector*[_nsamp])
    {
      for (unsigned i = 0; i < _nsamp; i++)
	_samples[i] = gsl_vector_calloc(_len);
    }
    ~_RealBuffer()
    {
      for (unsigned i = 0; i < _nsamp; i++)
	gsl_vector_free(_samples[i]);
      delete[] _samples;
    }

    const double sample(unsigned timeX, unsigned binX) const {
      unsigned idx = _index(timeX);
      const gsl_vector* vec = _samples[idx];
      return gsl_vector_get(vec, binX);
    }

    void nextSample(const gsl_vector* s = NULL, bool reverse = false) {
      _zero = (_zero + 1) % _nsamp;

      gsl_vector* nextBlock = _samples[_zero];

      if (s == NULL) {
	gsl_vector_set_zero(nextBlock);
      } else {
	if (s->size != _len)
	  throw jdimension_error("'_RealBuffer': Sizes do not match (%d vs. %d)", s->size, _len);
	assert( s->size == _len );
	if (reverse)
	  for (unsigned i = 0; i < _len; i++)
	    gsl_vector_set(nextBlock, i, gsl_vector_get(s, _len - i - 1));
	else
	  gsl_vector_memcpy(nextBlock, s);
      }
    }

    void nextSample(const gsl_vector_float* s) {
      _zero = (_zero + 1) % _nsamp;

      gsl_vector* nextBlock = _samples[_zero];

      assert( s->size == _len );
      for (unsigned i = 0; i < _len; i++)
	gsl_vector_set(nextBlock, i, gsl_vector_float_get(s, i));
    }

    void nextSample(const gsl_vector_short* s) {
      _zero = (_zero + 1) % _nsamp;

      gsl_vector* nextBlock = _samples[_zero];

      assert( s->size == _len );
      for (unsigned i = 0; i < _len; i++)
	gsl_vector_set(nextBlock, i, gsl_vector_short_get(s, i));
    }

    void zero() {
      for (unsigned i = 0; i < _nsamp; i++)
	gsl_vector_set_zero(_samples[i]);
      _zero = _nsamp - 1;
    }

  private:
    unsigned _index(unsigned idx) const {
      assert(idx < _nsamp);
      unsigned ret = (_zero + _nsamp - idx) % _nsamp;
      return ret;
    }

    const unsigned				_len;
    const unsigned				_nsamp;
    unsigned					_zero;		// index of most recent sample
    gsl_vector**				_samples;
  };

  BaseFilterBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0, bool synthesis = false);

  const unsigned				_M;
  const unsigned				_Mx2;
  const unsigned				_m;
  const unsigned				_mx2;
  const unsigned				_r;
  const unsigned				_R;
  const unsigned				_Rx2;
  const unsigned				_D;
};


// ----- definition for class `NormalFFTAnalysisBank' -----
//
/**
   @class do FFT on time discrete samples multiplied with a window.
*/
class NormalFFTAnalysisBank
  : protected BaseFilterBank, public VectorComplexFeatureStream {
 public:
  NormalFFTAnalysisBank(VectorFloatFeatureStreamPtr& samp,
			unsigned fftLen,  unsigned r = 1, unsigned windowType = 1, 
			const String& nm = "NormalFFTAnalysisBank");
  ~NormalFFTAnalysisBank();

  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  unsigned fftLen()	 const { return _N; }
  
protected:
  void _updateBuf();
  virtual void _updateBuffer(int frameX);
  
#ifdef HAVE_LIBFFTW3
  fftw_plan                          _fftwPlan;
#endif
  const VectorFloatFeatureStreamPtr  _samp;
  double*			     _output;
  int                                _winType; // 1 = hamming, 2 = hann window
  unsigned                           _N;       // FFT length
  const unsigned	             _processingDelay;
  unsigned			     _framesPadded;
  const gsl_vector*		     _prototype;
  _RealBuffer			     _buffer;
  gsl_vector*			     _convert;
  _RealBuffer		             _gsi;
};

typedef Inherit<NormalFFTAnalysisBank, VectorComplexFeatureStreamPtr> NormalFFTAnalysisBankPtr;

/**
* \defgroup OversampledFilterBank Oversampled Filter Bank
*/
/*@{*/


// ----- definition for class `OverSampledDFTFilterBank' -----
//
class OverSampledDFTFilterBank : public BaseFilterBank {
public:
  ~OverSampledDFTFilterBank();

  virtual void reset();

  protected:
  OverSampledDFTFilterBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0, bool synthesis = false, unsigned delayCompensationType=0, int gainFactor=1 );

  double polyphase(unsigned m, unsigned n) const {
    // assert(m < _Mx2);  assert(n < _m);  assert( m + _Mx2 * n < _prototype->size);
    return gsl_vector_get(_prototype, m + _M * n);
  }

  unsigned				_laN; /*>! the number of look-ahead */
  const unsigned			_N;
  unsigned				_processingDelay;
  const gsl_vector*			_prototype;
  _RealBuffer				_buffer;
  gsl_vector*				_convert;
  _RealBuffer				_gsi;
  const int                             _gainFactor;
};

/*@}*/

/**
* \defgroup PerfectReconstructionFilterBank Perfect Reconstruction Filter Bank
*/
/*@{*/


// ----- definition for class `PerfectReconstructionFilterBank' -----
//
class PerfectReconstructionFilterBank : public BaseFilterBank {
public:
  ~PerfectReconstructionFilterBank();

  virtual void reset();

  protected:
  PerfectReconstructionFilterBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0, bool synthesis = false);

  double polyphase(unsigned m, unsigned n) const {
    // assert(m < _Mx2);  assert(n < _m);  assert( m + _Mx2 * n < _prototype->size);
    return gsl_vector_get(_prototype, m + _Mx2 * n);
  }

  const unsigned				_N;
  const unsigned				_processingDelay;

  const gsl_vector*				_prototype;
  _RealBuffer					_buffer;
  gsl_vector*					_convert;
  gsl_vector_complex*				_w;
  _RealBuffer					_gsi;
};

/*@}*/

/**
* \addtogroup OversampledFilterBank
*/
/*@{*/

// ----- definition for class `OverSampledDFTAnalysisBank' -----
//
class OverSampledDFTAnalysisBank
: protected OverSampledDFTFilterBank, public VectorComplexFeatureStream {
  
 public:
  OverSampledDFTAnalysisBank(VectorFloatFeatureStreamPtr& samp,
			     gsl_vector* prototype, unsigned M, unsigned m, unsigned r, unsigned delayCompensationType =0,
			     const String& nm = "OverSampledDFTAnalysisBank");
  ~OverSampledDFTAnalysisBank();

  virtual const gsl_vector_complex* next(int frameX = -5);

  virtual void reset();

  unsigned fftLen()	 const { return _M; }
  unsigned nBlocks()	 const { return 4; }
  unsigned subSampRate() const { return 2; }
  bool isEnd(){return _endOfSamples;}

  OverSampledDFTFilterBank::polyphase;

 protected:
  void _updateBuf();
  virtual void _updateBuffer(int frameX);

#ifdef HAVE_LIBFFTW3
  fftw_plan					_fftwPlan;
#endif
  double*					_polyphaseOutput;
  unsigned					_framesPadded;
  const VectorFloatFeatureStreamPtr		_samp;
};

typedef Inherit<OverSampledDFTAnalysisBank, VectorComplexFeatureStreamPtr> OverSampledDFTAnalysisBankPtr;


// ----- definition for class `OverSampledDFTSynthesisBank' -----
//
class OverSampledDFTSynthesisBank
: private OverSampledDFTFilterBank, public VectorFloatFeatureStream {
 public:
  OverSampledDFTSynthesisBank(VectorComplexFeatureStreamPtr& samp,
			      gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0, 
			      unsigned delayCompensationType = 0, int gainFactor=1,
			      const String& nm = "OverSampledDFTSynthesisBank");

  OverSampledDFTSynthesisBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0, 
			      unsigned delayCompensationType = 0, int gainFactor=1, 
			      const String& nm = "OverSampledDFTSynthesisBank");

  ~OverSampledDFTSynthesisBank();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

  OverSampledDFTFilterBank::polyphase;

  const gsl_vector_float* nextFrame(int frameX);
  void inputSourceVector(const gsl_vector_complex* block);
  void doNotUseStreamFeature( bool flag=true ){
    _noStreamFeature = flag;
  }
  
 private:
  void _updateBuffer(int frameX);
  void _updateBuffer(const gsl_vector_complex* block);

  bool						_unprimed, _noStreamFeature;
  const VectorComplexFeatureStreamPtr		_samp;
#ifdef HAVE_LIBFFTW3
  fftw_plan					_fftwPlan;
#endif
  double*					_polyphaseInput;
};

typedef Inherit<OverSampledDFTSynthesisBank, VectorFloatFeatureStreamPtr> OverSampledDFTSynthesisBankPtr;

/*@}*/

/**
* \addtogroup PerfectReconstructionFilterBank
*/
/*@{*/

// ----- definition for class `PerfectReconstructionFFTAnalysisBank' -----
//
class PerfectReconstructionFFTAnalysisBank
: protected PerfectReconstructionFilterBank, public VectorComplexFeatureStream {
 public:
  PerfectReconstructionFFTAnalysisBank(VectorFloatFeatureStreamPtr& samp,
				       gsl_vector* prototype, unsigned M, unsigned m, unsigned r,
				       const String& nm = "PerfectReconstructionFFTAnalysisBank");
  ~PerfectReconstructionFFTAnalysisBank();

  virtual const gsl_vector_complex* next(int frameX = -5);

  virtual void reset();

  unsigned fftLen()	 const { return _Mx2; }
  unsigned nBlocks()	 const { return 4; }
  unsigned subSampRate() const { return 2; }

  PerfectReconstructionFilterBank::polyphase;

 protected:
  void _updateBuf();
  virtual void _updateBuffer(int frameX);

#ifdef HAVE_LIBFFTW3
  fftw_plan					_fftwPlan;
#endif
  double*					_polyphaseOutput;
  unsigned					_framesPadded;

  const VectorFloatFeatureStreamPtr		_samp;
};

typedef Inherit<PerfectReconstructionFFTAnalysisBank, VectorComplexFeatureStreamPtr> PerfectReconstructionFFTAnalysisBankPtr;


// ----- definition for class `PerfectReconstructionFFTSynthesisBank' -----
//
class PerfectReconstructionFFTSynthesisBank
: private PerfectReconstructionFilterBank, public VectorFloatFeatureStream {
 public:
  PerfectReconstructionFFTSynthesisBank(VectorComplexFeatureStreamPtr& samp,
					gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0,
					const String& nm = "PerfectReconstructionFFTSynthesisBank");

  PerfectReconstructionFFTSynthesisBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0,
					const String& nm = "PerfectReconstructionFFTSynthesisBank");

  ~PerfectReconstructionFFTSynthesisBank();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

  PerfectReconstructionFilterBank::polyphase;

 private:
  void _updateBuffer(int frameX);
  void _updateBuffer(const gsl_vector_complex* block);

  const VectorComplexFeatureStreamPtr		_samp;
#ifdef HAVE_LIBFFTW3
  fftw_plan					_fftwPlan;
#endif
  double*					_polyphaseInput;
};

typedef Inherit<PerfectReconstructionFFTSynthesisBank, VectorFloatFeatureStreamPtr> PerfectReconstructionFFTSynthesisBankPtr;

// ----- definition for class `DelayFeature' -----
//
class DelayFeature : public VectorComplexFeatureStream {  
 public:
  DelayFeature( const VectorComplexFeatureStreamPtr& samp, double delayT=0.0, const String& nm = "DelayFeature");
  ~DelayFeature();

  void setDelayTime( double delayT ){ _delayT = delayT; }
  virtual const gsl_vector_complex* next(int frameX = -5);

  virtual void reset();

private:
  double                                        _delayT;
  const VectorComplexFeatureStreamPtr		_samp;
};

typedef Inherit<DelayFeature, VectorComplexFeatureStreamPtr> DelayFeaturePtr;

gsl_vector* getWindow( unsigned winType, unsigned winLen );

void writeGSLFormat(const String& fileName, const gsl_vector* prototype);

/*@}*/

/*@}*/


#endif

