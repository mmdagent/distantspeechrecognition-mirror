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

#ifdef BTK_MEMDEBUG
#include "memcheck/memleakdetector.h"
#endif

#include <math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_fft_complex.h>

#include "common/jpython_error.h"
#include "modulated/modulated.h"

#ifdef HAVE_LIBFFTW3
#include <fftw3.h>
#endif

// unpack the 'complex_packed_array' into 'gsl_vector_complex'
//
static void complexUnpackedArrayPack(const gsl_vector_complex* src, double* tgt)
{
  unsigned len = src->size;

  for (unsigned m = 0; m < len; m++) {
    gsl_complex val = gsl_vector_complex_get(src, m);
    tgt[2*m]   = GSL_REAL(val);
    tgt[2*m+1] = GSL_IMAG(val);
  }
}

// unpack the 'complex_packed_array' into 'gsl_vector_complex'
//
static void complexPackedArrayUnpack(gsl_vector_complex* tgt, const double* src)
{
  unsigned len = tgt->size;

  for (unsigned m = 0; m < len; m++)
    gsl_vector_complex_set(tgt, m, gsl_complex_rect(src[2*m], src[2*m+1]));
}

/**
   @brief calculate a window.

   @param unsigned winType[in] a flag which indicates the returned window 
                               0 -> a rectangle window
                               1 -> Hamming window
                               2 -> Hanning window
   @param unsigned winLen[in] the length of a window
   @return a window 
 */
gsl_vector* getWindow( unsigned winType, unsigned winLen )
{
  gsl_vector* win = gsl_vector_calloc(winLen);
  
  switch( winType ){
  case 0:
    /* rectangle window */
    for (unsigned i = 0; i < winLen; i++)
      gsl_vector_set( win, i , 1.0 ); 
    break;
  case 2:
    /* Hanning window */
    for (unsigned i = 0; i < winLen; i++) {    
      gsl_vector_set( win, i , 0.5 * ( 1 - cos( (2.0*M_PI*i)/(double)(winLen-1) ) ) ); 
    }
    break;
  default:// Hamming window
    double temp = 2. * M_PI / (double)(winLen - 1);
    for ( unsigned i = 0 ; i < winLen; i++ )
      gsl_vector_set( win, i , 0.54 - 0.46 * cos( temp * i ) );
      //gsl_vector_set( win, i , 0.53836 - 0.46164 * cos( temp * i ) );
    break;
  }

  return win;
}

// ----- methods for class `BaseFilterBank' -----
//
BaseFilterBank::
BaseFilterBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r, bool synthesis)
  : _M(M), _Mx2(2*_M), _m(m), _mx2(2*_m),
    _r(r), _R(1 << _r), _Rx2(2 *_R), _D(_M / _R) { }

BaseFilterBank::~BaseFilterBank()
{
}

// ----- methods for class `OverSampledDFTAnalysisBank' -----
//

/**
   @brief construct an objects to transform samples by FFT.

   @param VectorFloatFeatureStreamPtr& samp[in/out]
   @param unsigned M[in] the length of FFT
   @param unsigned r[in] a decimation factor which decides a frame shift size.
   @param unsigned windowType[in]
*/
NormalFFTAnalysisBank::
NormalFFTAnalysisBank(VectorFloatFeatureStreamPtr& samp,
		      unsigned M, unsigned r, unsigned windowType,
		      const String& nm )
  : BaseFilterBank(NULL, M, 1, r, /*synthesis*/ false ), _N(_M), _processingDelay(_mx2 - 1),
    VectorComplexFeatureStream(_M, nm),
    _samp(samp),
    _framesPadded(0),
    _winType(windowType),
    _buffer(_M, /* m=1 */ 1 * _R),
    _convert(gsl_vector_calloc(_M)), _gsi( /* synthesis==false */ _D, _R)
{
  if (_samp->size() != _D)
    throw jdimension_error("Input block length (%d) != _D (%d)\n", _samp->size(), _D);

  //printf("'FFTAnalysisBank' Feature Input Size  = %d\n", _samp->size());
  
  _prototype  = (const gsl_vector *)getWindow( _winType, _N );

#ifdef HAVE_LIBFFTW3
  _output = static_cast<double*>(fftw_malloc(sizeof(fftw_complex) * _N * 2));
#else
  _output = new double[2 * _N];
#endif

#ifdef HAVE_LIBFFTW3
  fftw_init_threads();
  fftw_plan_with_nthreads(2);
  _fftwPlan = fftw_plan_dft_1d(_N, 
			       (double (*)[2])_output, 
			       (double (*)[2])_output, 
			       FFTW_FORWARD, 
			       FFTW_MEASURE);
#endif
  reset();
}

NormalFFTAnalysisBank::~NormalFFTAnalysisBank()
{  
  gsl_vector_free((gsl_vector*) _prototype);
  gsl_vector_free(_convert);
  
#ifdef HAVE_LIBFFTW3
  fftw_destroy_plan(_fftwPlan);
  fftw_free(_output);
#else
  delete [] _output;
#endif
}

void NormalFFTAnalysisBank::reset()
{
  _buffer.zero();  _gsi.zero();
  _samp->reset();  VectorComplexFeatureStream::reset();
  _framesPadded = 0;
}

void NormalFFTAnalysisBank::_updateBuf()
{
  for (unsigned sampX = 0; sampX < _R; sampX++)
    for (unsigned dimX = 0; dimX < _D; dimX++)
      gsl_vector_set(_convert, dimX + sampX * _D, _gsi.sample(_R - sampX - 1, dimX));
  _buffer.nextSample(_convert, /* reverse= */ true);
}

void NormalFFTAnalysisBank::_updateBuffer(int frameX)
{
  if (_framesPadded == 0) {				// normal processing

    try {
      /*
      if (_frameX == FrameResetX) {
	for (unsigned i = 0; i < _Rx2; i++) {
	  const gsl_vector_float* block = _samp->next(i);
	  _gsi.nextSample(block);
	}
      }
      */
      const gsl_vector_float* block = _samp->next(frameX /* + _Rx2 */);
      _gsi.nextSample(block);
      _updateBuf();
    } catch  (exception& e) {
      _gsi.nextSample();
      _updateBuf();

      // printf("Padding frame %d.\n", _framesPadded);

      _framesPadded++;
    }

  } else if (_framesPadded < _processingDelay) {	// pad with zeros

    _gsi.nextSample();
    _updateBuf();

    // printf("Padding frame %d.\n", _framesPadded);

    _framesPadded++;

  } else {						// end of utterance

    throw jiterator_error("end of samples!");

  }
}

const gsl_vector_complex* NormalFFTAnalysisBank::next(int frameX)
{
  if (frameX == _frameX) return _vector;
  
  _updateBuffer(frameX);

  // calculate outputs of polyphase filters
  for (unsigned m = 0; m < _M; m++) {
    double win_i = gsl_vector_get( _prototype, m );

    /*
      printf("  m = %4d \n", m);
      printf("    Got polyphase %g\n", win_i);
      printf("    Got sample %g\n", _buffer.sample( 0, _M - m -1));
    */

    _output[2*m]   = win_i * _buffer.sample(0, _M - m - 1 );
    _output[2*m+1] = 0.0;
  }

#ifdef HAVE_LIBFFTW3
  fftw_execute(_fftwPlan);
#else
  gsl_fft_complex_radix2_forward(_output, /* stride= */ 1, _N);
#endif

  complexPackedArrayUnpack(_vector, _output);

  _increment();
  return _vector;
}


// ----- methods for class `OverSampledDFTFilterBank' -----
//
OverSampledDFTFilterBank::
OverSampledDFTFilterBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r, bool synthesis, unsigned delayCompensationType, int gainFactor )
  : BaseFilterBank(prototype, M, m, r, synthesis), _N(_M * m),
    _prototype(gsl_vector_calloc(_N)), _buffer(_M, m * _R),
    _convert(gsl_vector_calloc(_M)), _gsi((synthesis ? _M : _D), _R),
    _gainFactor(gainFactor)
{
  if (prototype->size != _N)
    throw jconsistency_error("Prototype sizes do not match (%d vs. %d).",
			     prototype->size, _N);

  //printf("D = %d\n", _D);

  gsl_vector* pr = (gsl_vector*) _prototype;
  gsl_vector_memcpy(pr, prototype);

  _laN = 0; // indicates how many frames should be skipped.
  switch ( delayCompensationType ) {
    // de Haan's filter bank or Nyquist(M) filter bank
  case 1 : // compensate delays in the synthesis filter bank
    _processingDelay = m * _R - 1 ; // m * 2^r - 1 ;
    break;
  case 2 : // compensate delays in the analythesis and synthesis filter banks
    if( synthesis == true )
      _processingDelay = m * _R / 2 ;
    else{
      _processingDelay = m * _R - 1;
      _laN = _m * _R  / 2 - 1;
    }
    break;
    // undefined filter bank
  default :
    _processingDelay = _mx2 - 1;
    break;
  }

  // set the buffers to zero
  reset();
}

OverSampledDFTFilterBank::~OverSampledDFTFilterBank()
{
  gsl_vector_free((gsl_vector*) _prototype);
  gsl_vector_free(_convert);
}

void OverSampledDFTFilterBank::reset()
{
  _buffer.zero();  _gsi.zero();
}


// ----- methods for class `PerfectReconstructionFilterBank' -----
//
PerfectReconstructionFilterBank::
PerfectReconstructionFilterBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r, bool synthesis)
  : BaseFilterBank(prototype, M, m, r, synthesis), _N(_Mx2 * m), _processingDelay(_mx2 - 1),
    _prototype(gsl_vector_calloc(_N)), _buffer(_Mx2, m * (_r + 2)),
    _convert(gsl_vector_calloc(_Mx2)), _w(gsl_vector_complex_calloc(_Mx2)),
    _gsi((synthesis ? _Mx2 : _D), _Rx2)
{
  if (prototype->size != _N)
    throw jconsistency_error("Prototype sizes do not match (%d vs. %d).",
			     prototype->size, _N);

  //printf("D = %d\n", _D);

  gsl_vector* pr = (gsl_vector*) _prototype;
  gsl_vector_memcpy(pr, prototype);

  // set the buffers to zero
  reset();
}

PerfectReconstructionFilterBank::~PerfectReconstructionFilterBank()
{
  gsl_vector_free((gsl_vector*) _prototype);
  gsl_vector_free(_convert);
  gsl_vector_complex_free(_w);
}

void PerfectReconstructionFilterBank::reset()
{
  _buffer.zero();  _gsi.zero();
}


// ----- methods for class `OverSampledDFTAnalysisBank' -----
//
/*
  @brief construct an object to calculate subbands with analysis filter banks (FBs)
  @param VectorFloatFeatureStreamPtr& samp [in] an object to keep wave data
  @param gsl_vector* prototype [in] filter coefficients of a prototype of an analysis filter 
  @param unsigned M [in] the number of subbands
  @param unsigned m [in] fliter length factor ( the filter length == m * M )
  @param unsigned r [in] decimation factor
  @param unsigned delayCompensationType [in] 1 : delays are compensated in the synthesis FBs only. 2 : delays are compensated in the both FBs.
*/
OverSampledDFTAnalysisBank::
OverSampledDFTAnalysisBank(VectorFloatFeatureStreamPtr& samp,
			   gsl_vector* prototype, unsigned M, unsigned m, unsigned r, unsigned delayCompensationType, const String& nm)
  : OverSampledDFTFilterBank(prototype, M, m, r, /*sythesis=*/ false, delayCompensationType ),
    VectorComplexFeatureStream(_M, nm),
    _samp(samp),
#ifdef HAVE_LIBFFTW3
    _polyphaseOutput(static_cast<double*>(fftw_malloc(sizeof(fftw_complex) * _Mx2))),
#else
    _polyphaseOutput(new double[2 * _M]),
#endif
    _framesPadded(0)
{
  if (_samp->size() != _D)
    throw jdimension_error("Input block length (%d) != _D (%d)\n", _samp->size(), _D);

  //printf("'FFTAnalysisBank' Feature Input Size  = %d\n", _samp->size());
  //printf("'FFTAnalysisBank' Feature Output Size = %d\n", size());

#ifdef HAVE_LIBFFTW3
  fftw_init_threads();
  fftw_plan_with_nthreads(2);
  _fftwPlan = fftw_plan_dft_1d(_M, 
			       (double (*)[2])_polyphaseOutput, 
			       (double (*)[2])_polyphaseOutput, 
			       FFTW_BACKWARD, 
			       FFTW_MEASURE);
#endif
}

OverSampledDFTAnalysisBank::~OverSampledDFTAnalysisBank()
{
#ifdef HAVE_LIBFFTW3
  fftw_destroy_plan(_fftwPlan);
  fftw_free(_polyphaseOutput);
#else
  delete[] _polyphaseOutput;
#endif
}

void OverSampledDFTAnalysisBank::_updateBuf()
{
  /* note
     _gsi has _samples[R][D], M samples.
     Then the data of _gsi are given to _convert which has _sample[mR][M].
  */
  for (unsigned sampX = 0; sampX < _R; sampX++)
    for (unsigned dimX = 0; dimX < _D; dimX++)
      gsl_vector_set(_convert, dimX + sampX * _D, _gsi.sample(_R - sampX - 1, dimX));
  _buffer.nextSample(_convert, /* reverse= */ true);
}

const gsl_vector_complex* OverSampledDFTAnalysisBank::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  _updateBuffer(frameX);

  // calculate outputs of polyphase filters
  for (unsigned m = 0; m < _M; m++) {
    double sum  = 0.0;
    for (unsigned k = 0; k < _m; k++) {

      /*
      printf("  m = %4d : k = %4d.\n", m, k);
      printf("    Got polyphase %g\n", polyphase(m, k));
      printf("    Got sample %g\n", _buffer.sample(_R * k, m));
      */

      sum  += polyphase(m, k) * _buffer.sample(_R * k, m);
    }

    _polyphaseOutput[2*m]   = sum;
    _polyphaseOutput[2*m+1] = 0.0;
  }

#ifdef HAVE_LIBFFTW3
  fftw_execute(_fftwPlan);
#else
  gsl_fft_complex_radix2_backward(_polyphaseOutput, /* stride= */ 1, _M);
#endif

  complexPackedArrayUnpack(_vector, _polyphaseOutput);

  if( _gainFactor > 0 )
    for(unsigned m = 0; m < _M; m++) {
      gsl_vector_complex_set( _vector, m, 
			      gsl_complex_mul_real( gsl_vector_complex_get( _vector, m ),  _gainFactor ) );
}

  _increment();
  return _vector;
}

void OverSampledDFTAnalysisBank::reset()
{
  _samp->reset();  OverSampledDFTFilterBank::reset();  VectorComplexFeatureStream::reset();
  _buffer.zero();
  _framesPadded = 0;
}

void OverSampledDFTAnalysisBank::_updateBuffer(int frameX)
{
  if( true == _endOfSamples ){
    fprintf(stderr,"end of samples!\n");
    throw jiterator_error("end of samples!");
  }
  if( _laN >0 && _frameX == FrameResetX ){
    // skip samples for compensating the processing delays
    for (unsigned itnX = 0; itnX < _laN; itnX++){
      const gsl_vector_float* block = _samp->next(itnX/* + _Rx2 */);
      _gsi.nextSample(block);
      _updateBuf();
    }
  }
  if (_framesPadded == 0) {				// normal processing

    try {
      /*
      if (_frameX == FrameResetX) {
	for (unsigned i = 0; i < _Rx2; i++) {
	  const gsl_vector_float* block = _samp->next(i);
	  _gsi.nextSample(block);
	}
      }
      */
      const gsl_vector_float* block;
      if( frameX >= 0 )
	block = _samp->next(frameX + _laN );
      else // just take the next frame
	block = _samp->next(frameX );
      _gsi.nextSample(block);
      _updateBuf();
    } catch  (exception& e) {
      // it happens if the number of prcessing frames exceeds the data length.
      _gsi.nextSample();
      _updateBuf();

      // printf("Padding frame %d.\n", _framesPadded);

      _framesPadded++;
    }

  } else if (_framesPadded < _processingDelay) {	// pad with zeros

    _gsi.nextSample();
    _updateBuf();

    //fprintf(stderr,"Padding frame %d %d.\n", _framesPadded, _frameX );

    _framesPadded++;

  } else {						// end of utterance
    _endOfSamples = true;
    throw jiterator_error("end of samples!");
  }
}


// ----- methods for class `OverSampledDFTSynthesisBank' -----
//
OverSampledDFTSynthesisBank::
OverSampledDFTSynthesisBank(VectorComplexFeatureStreamPtr& samp,
			    gsl_vector* prototype, unsigned M, unsigned m, unsigned r, 
			    unsigned delayCompensationType, int gainFactor, 
			    const String& nm)
  : OverSampledDFTFilterBank(prototype, M, m, r, /*synthesis=*/ true, delayCompensationType, gainFactor ),
    VectorFloatFeatureStream(_D, nm),_samp(samp),_noStreamFeature(false),
#ifdef HAVE_LIBOverSampledDFTW3
    _polyphaseInput(static_cast<double*>(fftw_malloc(sizeof(fftw_complex) * _Mx2)))
#else
    _polyphaseInput(new double[2 * _M])
#endif
{
  //printf("'OverSampledDFTSynthesisBank' Feature Input Size  = %d\n", _samp->size());
  //printf("'OverSampledDFTSynthesisBank' Feature Output Size = %d\n", size());

#ifdef HAVE_LIBOverSampledDFTW3
  fftw_init_threads();
  fftw_plan_with_nthreads(2);
  _fftwPlan = fftw_plan_dft_1d(_M,
			       (double (*)[2])_polyphaseInput,
			       (double (*)[2])_polyphaseInput,
			       OverSampledDFTW_FORWARD,
			       OverSampledDFTW_MEASURE);
#endif
}

OverSampledDFTSynthesisBank::
OverSampledDFTSynthesisBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r, 
			    unsigned delayCompensationType, int gainFactor, 
			    const String& nm)
  : OverSampledDFTFilterBank(prototype, M, m, r, /*synthesis=*/ true, delayCompensationType, gainFactor ),
    VectorFloatFeatureStream(_D, nm),_unprimed(true),_noStreamFeature(true),
#ifdef HAVE_LIBOverSampledDFTW3
    _polyphaseInput(static_cast<double*>(fftw_malloc(sizeof(fftw_complex) * _M)))
#else
    _polyphaseInput(new double[2 * _M])
#endif
{
#ifdef HAVE_LIBOverSampledDFTW3
  fftw_init_threads();
  fftw_plan_with_nthreads(2);
  _fftwPlan = fftw_plan_dft_1d(_M, 
			       (double (*)[2])_polyphaseInput, 
			       (double (*)[2])_polyphaseInput, 
			       OverSampledDFTW_FORWARD, 
			       OverSampledDFTW_MEASURE);
#endif
}

OverSampledDFTSynthesisBank::~OverSampledDFTSynthesisBank()
{
#ifdef HAVE_LIBOverSampledDFTW3
  fftw_destroy_plan(_fftwPlan); 
  fftw_free(_polyphaseInput);
#else
  delete[] _polyphaseInput;
#endif
}

void OverSampledDFTSynthesisBank::inputSourceVector(const gsl_vector_complex* block)
{
  _updateBuffer(block);
}

void OverSampledDFTSynthesisBank::_updateBuffer(int frameX)
{
  // get next frame and perform forward OverSampledDFT
  if( false == _noStreamFeature ){
    const gsl_vector_complex* block = _samp->next(frameX);
    _updateBuffer(block);
  }
}

void OverSampledDFTSynthesisBank::_updateBuffer(const gsl_vector_complex* block)
{
  // get next frame and perform forward OverSampledDFT
  complexUnpackedArrayPack(block, _polyphaseInput);

#ifdef HAVE_LIBOverSampledDFTW3
  fftw_execute(_fftwPlan);
#else
  gsl_fft_complex_radix2_forward(_polyphaseInput, /* stride= */ 1, _M);
#endif

  for (unsigned m = 0; m < _M; m++)
    gsl_vector_set(_convert, m, _polyphaseInput[2*m]);

  // update buffer
  _buffer.nextSample(_convert);
}

const gsl_vector_float* OverSampledDFTSynthesisBank::nextFrame(int frameX)
{
  /*
  if (_unprimed) {
    for (unsigned itnX = 0; itnX < _processingDelay; itnX++) {
      _updateBuffer(itnX);
    }
    _unprimed = false;
  }
  */
  return next(frameX);
}

const gsl_vector_float* OverSampledDFTSynthesisBank::next(int frameX)
{
  if (frameX == _frameX + _processingDelay) return _vector;

  // "prime" the buffer
  if (_frameX == FrameResetX) {
    for (unsigned itnX = 0; itnX < _processingDelay; itnX++)
      _updateBuffer(itnX);
  }

  if ( frameX >= 0 && frameX - 1 != _frameX )
    ("The output might not be continuous %s: %d != %d\n",name().c_str(), frameX - 1, _frameX);

  if( frameX >= 0 )
    _updateBuffer( frameX + 1 + _processingDelay);
  else
    _updateBuffer(_frameX + 1 + _processingDelay);
  _increment();

  // calculate outputs of polyphase filters
  for (unsigned m = 0; m < _M; m++) {
    double sum  = 0.0;
    for (unsigned k = 0; k < _m; k++)
      sum  += polyphase(_M - m - 1, k) * _buffer.sample(_R * k, m);
    gsl_vector_set(_convert, m, sum);
  }
  _gsi.nextSample(_convert);

  // synthesize final output of filterbank
  gsl_vector_float_set_zero(_vector);
  for (unsigned sampX = 0; sampX < _R; sampX++)
    for (unsigned d = 0; d < _D; d++)
      gsl_vector_float_set(_vector, _D - d - 1, gsl_vector_float_get(_vector, _D - d - 1) + _gsi.sample(_R - sampX - 1, d + sampX * _D) );

  if( _gainFactor > 0 )
    gsl_vector_float_scale( _vector, (float)_gainFactor );

  return _vector;
}

void OverSampledDFTSynthesisBank::reset()
{
  if( false == _noStreamFeature ){
    _samp->reset(); 
  }
  OverSampledDFTFilterBank::reset();  
  VectorFloatFeatureStream::reset();
  _buffer.zero();
}

void writeGSLFormat(const String& fileName, const gsl_vector* prototype)
{
  //printf("Writing length %d prototype to %s\n", prototype->size, fileName.c_str());

  FILE* fp = fileOpen(fileName, "w");
  gsl_vector_fwrite (fp, prototype);
  fileClose(fileName, fp);
}


// ----- methods for class `PerfectReconstructionFFTAnalysisBank' -----
//
PerfectReconstructionFFTAnalysisBank::
PerfectReconstructionFFTAnalysisBank(VectorFloatFeatureStreamPtr& samp, gsl_vector* prototype,
				     unsigned M, unsigned m, unsigned r, const String& nm)
  : PerfectReconstructionFilterBank(prototype, M, m, r, /*sythesis=*/ false),
    VectorComplexFeatureStream(_Mx2, nm),
#ifdef HAVE_LIBFFTW3
    _polyphaseOutput(static_cast<double*>(fftw_malloc(sizeof(fftw_complex) * _Mx2))),
#else
    _polyphaseOutput(new double[2 * _Mx2]),
#endif
    _framesPadded(0), _samp(samp)
{
  // initialize 'w_k'
  gsl_complex W_M = gsl_complex_polar(/* rho= */ 1.0, /* theta= */ - M_PI / (2.0 * _M));
  gsl_complex val = gsl_complex_rect(1.0, 0.0);
  
  for (unsigned k = 0; k < _Mx2; k++) {
    gsl_vector_complex_set(_w, k, val);
    val = gsl_complex_mul(val, W_M);
  }
#ifdef HAVE_LIBFFTW3
  fftw_init_threads();
  fftw_plan_with_nthreads(2);
  _fftwPlan = fftw_plan_dft_1d(_Mx2, 
			       (double (*)[2])_polyphaseOutput, 
			       (double (*)[2])_polyphaseOutput, 
			       FFTW_BACKWARD, 
			       FFTW_MEASURE);
#endif
}

PerfectReconstructionFFTAnalysisBank::~PerfectReconstructionFFTAnalysisBank()
{
#ifdef HAVE_LIBFFTW3
  fftw_destroy_plan(_fftwPlan);
  fftw_free(_polyphaseOutput);
#else
  delete[] _polyphaseOutput;
#endif
}

void PerfectReconstructionFFTAnalysisBank::_updateBuf()
{
  for (unsigned sampX = 0; sampX < _Rx2; sampX++)
    for (unsigned dimX = 0; dimX < _D; dimX++)
      gsl_vector_set(_convert, dimX + sampX * _D, _gsi.sample(_Rx2 - sampX - 1, dimX));
  _buffer.nextSample(_convert, /* reverse= */ true);
}

const gsl_vector_complex* PerfectReconstructionFFTAnalysisBank::next(int frameX)
{
  if (frameX == _frameX) return _vector;
  
  _updateBuffer(frameX);

  // calculate outputs of polyphase filters
  for (unsigned m = 0; m < _Mx2; m++) {
    double sum  = 0.0;
    int    flip = 1;

    for (unsigned k = 0; k < _m; k++) {
      sum  += flip * polyphase(m, k) * _buffer.sample((_r + 2) * k, m);
      flip *= -1;
    }

    gsl_complex output      = gsl_complex_mul_real(gsl_vector_complex_get(_w, m), sum);
    _polyphaseOutput[2*m]   = GSL_REAL(output);
    _polyphaseOutput[2*m+1] = GSL_IMAG(output);
  }

#ifdef HAVE_LIBFFTW3
  fftw_execute(_fftwPlan);
  // scale output vector
  for(int i=0; i<_Mx2*2; i++) {
    _polyphaseOutput[i] = _polyphaseOutput[i] / _Mx2;
  }
#else
  gsl_fft_complex_radix2_inverse(_polyphaseOutput, /* stride= */ 1, _Mx2);
#endif

  complexPackedArrayUnpack(_vector, _polyphaseOutput);

  _increment();
  return _vector;
}

void PerfectReconstructionFFTAnalysisBank::reset()
{
  PerfectReconstructionFilterBank::reset();  VectorComplexFeatureStream::reset();
  _buffer.zero();    _samp->reset();
  _framesPadded = 0;
}

void PerfectReconstructionFFTAnalysisBank::_updateBuffer(int frameX)
{
  if (_framesPadded == 0) {				// normal processing

    try {
      /*
      if (_frameX == FrameResetX) {
	for (unsigned i = 0; i < _Rx2; i++) {
	  const gsl_vector_float* block = _samp->next(i);
	  _gsi.nextSample(block);
	}
      }
      */
      const gsl_vector_float* block = _samp->next(frameX /* + _Rx2 */);
      _gsi.nextSample(block);
      _updateBuf();
    } catch  (exception& e) {
      _gsi.nextSample();
      _updateBuf();

      // printf("Padding frame %d.\n", _framesPadded);

      _framesPadded++;
    }

  } else if (_framesPadded < _processingDelay) {	// pad with zeros

    _gsi.nextSample();
    _updateBuf();

    // printf("Padding frame %d.\n", _framesPadded);

    _framesPadded++;

  } else {						// end of utterance

    throw jiterator_error("end of samples!");

  }
}


// ----- methods for class `PerfectReconstructionFFTSynthesisBank' -----
//
PerfectReconstructionFFTSynthesisBank::
PerfectReconstructionFFTSynthesisBank(VectorComplexFeatureStreamPtr& samp,
				      gsl_vector* prototype, unsigned M, unsigned m, unsigned r,
				      const String& nm)
  : PerfectReconstructionFilterBank(prototype, M, m, r, /*synthesis=*/ true),
    VectorFloatFeatureStream(_D, nm),
    _samp(samp),
#ifdef HAVE_LIBFFTW3
    _polyphaseInput(static_cast<double*>(fftw_malloc(sizeof(fftw_complex) * _Mx2)))
#else
    _polyphaseInput(new double[2 * _Mx2])
#endif
{
  //printf("'PerfectReconstructionFFTSynthesisBank' Feature Input Size  = %d\n", _samp->size());
  //printf("'PerfectReconstructionFFTSynthesisBank' Feature Output Size = %d\n", size());

  // initialize 'w_k'
  gsl_complex W_M = gsl_complex_polar(/* rho= */ 1.0, /* theta= */ M_PI / (2.0 * _M));
  gsl_complex val = gsl_complex_rect(1.0, 0.0);
  
  for (unsigned k = 0; k < _Mx2; k++) {
    gsl_vector_complex_set(_w, k, val);
    val = gsl_complex_mul(val, W_M);
  }
#ifdef HAVE_LIBFFTW3
  fftw_init_threads();
  fftw_plan_with_nthreads(2);
  _fftwPlan = fftw_plan_dft_1d(_Mx2,
			       (double (*)[2])_polyphaseInput,
			       (double (*)[2])_polyphaseInput,
			       FFTW_FORWARD,
			       FFTW_MEASURE);
#endif
}

PerfectReconstructionFFTSynthesisBank::
PerfectReconstructionFFTSynthesisBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r,
		 const String& nm)
  : PerfectReconstructionFilterBank(prototype, M, m, r, /*synthesis=*/ true),
    VectorFloatFeatureStream(_D, nm),
#ifdef HAVE_LIBFFTW3
    _polyphaseInput(static_cast<double*>(fftw_malloc(sizeof(fftw_complex) * _Mx2)))
#else
    _polyphaseInput(new double[2 * _Mx2])
#endif
{
  // initialize 'w_k'
  gsl_complex W_M = gsl_complex_polar(/* rho= */ 1.0, /* theta= */ M_PI / (2.0 * _M));
  gsl_complex val = gsl_complex_rect(1.0, 0.0);
  
  for (unsigned k = 0; k < _Mx2; k++) {
    gsl_vector_complex_set(_w, k, val);
    val = gsl_complex_mul(val, W_M);
  }
#ifdef HAVE_LIBFFTW3
  fftw_init_threads();
  fftw_plan_with_nthreads(2);
  _fftwPlan = fftw_plan_dft_1d(_Mx2, 
			       (double (*)[2])_polyphaseInput, 
			       (double (*)[2])_polyphaseInput, 
			       FFTW_FORWARD, 
			       FFTW_MEASURE);
#endif
}

PerfectReconstructionFFTSynthesisBank::~PerfectReconstructionFFTSynthesisBank()
{
#ifdef HAVE_LIBFFTW3
  fftw_destroy_plan(_fftwPlan); 
  fftw_free(_polyphaseInput);
#else
  delete[] _polyphaseInput;
#endif
}

void PerfectReconstructionFFTSynthesisBank::_updateBuffer(int frameX)
{
  // get next frame and perform forward FFT
  const gsl_vector_complex* block = _samp->next(frameX);
  _updateBuffer(block);
}

void PerfectReconstructionFFTSynthesisBank::_updateBuffer(const gsl_vector_complex* block)
{
  // get next frame and perform forward FFT
  complexUnpackedArrayPack(block, _polyphaseInput);

#ifdef HAVE_LIBFFTW3
  fftw_execute(_fftwPlan);
#else
  gsl_fft_complex_radix2_forward(_polyphaseInput, /* stride= */ 1, _Mx2);
#endif

  // apply 'w' factors
  for (unsigned m = 0; m < _Mx2; m++) {
    gsl_complex val = gsl_complex_rect(_polyphaseInput[2*m], _polyphaseInput[2*m+1]);
    gsl_vector_set(_convert, m, GSL_REAL(gsl_complex_mul(val, gsl_vector_complex_get(_w, m))));
  }

  // update buffer
  _buffer.nextSample(_convert);
}

const gsl_vector_float* PerfectReconstructionFFTSynthesisBank::next(int frameX)
{
  if (frameX == _frameX + _processingDelay) return _vector;

  // "prime" the buffer
  if (_frameX == FrameResetX) {
    for (unsigned itnX = 0; itnX < _processingDelay; itnX++)
      _updateBuffer(itnX);
  }

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
		       name().c_str(), frameX - 1, _frameX);

  _updateBuffer(_frameX + 1 + _processingDelay);
  _increment();

  // calculate outputs of polyphase filters
  for (unsigned m = 0; m < _Mx2; m++) {
    double sum  = 0.0;
    int    flip = (_m % 2 == 1) ? 1 : -1;

    for (unsigned k = 0; k < _m; k++) {
      sum  += flip * polyphase(m, _m - k - 1) * _buffer.sample((_r + 2) * k, m);
      flip *= -1;
    }
    gsl_vector_set(_convert, m, sum);
  }
  _gsi.nextSample(_convert);

  // synthesize final output of filterbank
  gsl_vector_float_set_zero(_vector);
  for (unsigned sampX = 0; sampX < _Rx2; sampX++)
    for (unsigned d = 0; d < _D; d++)
      gsl_vector_float_set(_vector, _D - d - 1, gsl_vector_float_get(_vector, _D - d - 1) + _gsi.sample(_Rx2 - sampX - 1, d + sampX * _D) / _R);

  return _vector;
}

void PerfectReconstructionFFTSynthesisBank::reset()
{
  _samp->reset();  PerfectReconstructionFilterBank::reset();  VectorFloatFeatureStream::reset();
  _buffer.zero();
}

// ----- definition for class `DelayFeature' -----
//
DelayFeature::DelayFeature(const VectorComplexFeatureStreamPtr& samp, double delayT, const String& nm )
  : VectorComplexFeatureStream( samp->size(), nm), _samp(samp), _delayT(delayT)
{
}

DelayFeature::~DelayFeature()
{}

void DelayFeature::reset()
{						
  _samp->reset();  
  VectorComplexFeatureStream::reset();
}

const gsl_vector_complex* DelayFeature::next(int frameX)
{
  if ( frameX == _frameX ) return _vector;
  
  const gsl_vector_complex* samp = _samp->next(frameX);
  const gsl_complex alpha = gsl_complex_polar( 1.0, _delayT );

  gsl_vector_complex_memcpy( (gsl_vector_complex*)_vector, samp );
  gsl_blas_zscal( alpha, (gsl_vector_complex*)_vector );

  _increment();
  return _vector;
}
