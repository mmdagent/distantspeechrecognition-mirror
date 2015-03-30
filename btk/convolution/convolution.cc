//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.modulated
//  Purpose: Block convolution realization of an LTI system with the FFT.
//  Author:  John McDonough
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


#include <math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>

#include "common/jpython_error.h"
#include "convolution/convolution.h"

#ifdef HAVE_CONFIG_H
#include <btk.h>
#endif
#ifdef HAVE_LIBFFTW3
#include <fftw3.h>
#endif


// ----- methods for class `OverlapAdd' -----
//
OverlapAdd::OverlapAdd(VectorFloatFeatureStreamPtr& samp,
		       const gsl_vector* impulseResponse, unsigned fftLen, const String& nm)
  : VectorFloatFeatureStream(samp->size(), nm), _samp(samp),
    _L(samp->size()), _P(impulseResponse->size), _N(_checkFFTLen(_L, _P, fftLen)), _N2(_N/2),
    _section(new double[_N]), _frequencyResponse(gsl_vector_complex_alloc(_N2+1)),
    _buffer(gsl_vector_float_alloc(_L+_P-1))
{
  _setImpulseResponse(impulseResponse);
}

OverlapAdd::~OverlapAdd()
{
  delete[] _section;
  gsl_vector_complex_free(_frequencyResponse);
  gsl_vector_float_free(_buffer);
}

void OverlapAdd::_setImpulseResponse(const gsl_vector* impulseResponse)
{
  if (impulseResponse == NULL) {
    gsl_vector_complex_set_zero(_frequencyResponse);
    return;
  }

  for (unsigned i = 0; i < _N; i++)
    _section[i] = 0.0;
  for (unsigned i = 0; i < _P; i++)
    _section[i] = gsl_vector_get(impulseResponse, i);
  gsl_fft_real_radix2_transform(_section, /* stride= */ 1, _N);
  _halfComplexUnpack(_frequencyResponse, _section);

  for (unsigned i = 0; i < _N; i++)
    _section[i] = 0.0;
  gsl_vector_float_set_zero(_buffer);
}

// check consistency of FFT length
unsigned OverlapAdd::_checkFFTLen(unsigned sectionLen, unsigned irLen, unsigned fftLen)
{
  printf("Section Length          = %d.\n", sectionLen);
  printf("Impulse Response Length = %d.\n", irLen);

  if (fftLen == 0) {

    fftLen = 1;
    while (fftLen < sectionLen + irLen - 1)
      fftLen *= 2;
    printf("Setting FFT length      = %d.\n", fftLen);

    return fftLen;

  } else {

    if (fftLen < sectionLen + irLen - 1)
      throw jdimension_error("Section (%d) and impulse response (%d) lengths inconsistent with FFT length (%d).",
			     sectionLen, irLen, fftLen);
    return fftLen;

  }
}

// unpack the 'complex_packed_array' into 'gsl_vector_complex'
void OverlapAdd::_halfComplexUnpack(gsl_vector_complex* tgt, const double* src)
{
  for (unsigned m = 0; m <= _N2; m++) {
    if (m == 0 || m == _N2) {
      gsl_vector_complex_set(tgt, m, gsl_complex_rect(src[m], 0.0));
    } else {
      gsl_vector_complex_set(tgt, m, gsl_complex_rect(src[m], src[_N-m]));
    }
  }
}

const gsl_vector_float* OverlapAdd::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem: %d != %d\n", frameX - 1, _frameX);

  const gsl_vector_float* block = _samp->next(_frameX + 1);

  assert(block->size == _L);

  // forward FFT on new data
  for (unsigned i = 0; i < _N; i++)
    _section[i] = 0.0;
  for (unsigned i = 0; i < _L; i++)
    _section[i] = gsl_vector_float_get(block, i);
  gsl_fft_real_radix2_transform(_section, /*stride=*/ 1, _N);

  // multiply with frequency response
  for (unsigned i = 0; i <= _N2; i++) {
    if (i == 0 || i == _N2) {
      _section[i] = _section[i] * GSL_REAL(gsl_vector_complex_get(_frequencyResponse, i));
    } else {
      gsl_complex val = gsl_complex_mul(gsl_complex_rect(_section[i], _section[_N-i]), gsl_vector_complex_get(_frequencyResponse, i));
      _section[i]     = GSL_REAL(val);
      _section[_N-i]  = GSL_IMAG(val);
    }
  }

  // inverse FFT
  gsl_fft_halfcomplex_radix2_inverse(_section, /* stride= */ 1, _N);

  // add contribution of new section to buffer
  for (unsigned i = 0; i < _L + _P - 1; i++)
    gsl_vector_float_set(_buffer, i, gsl_vector_float_get(_buffer, i) + _section[i]);

  // copy section length 'L' from buffer onto output
  for (unsigned i = 0; i < _L; i++)
    gsl_vector_float_set(_vector, i, gsl_vector_float_get(_buffer, i));

  // shift down buffer
  for (unsigned i = 0; i < _P - 1; i++)
    gsl_vector_float_set(_buffer, i, gsl_vector_float_get(_buffer, i + _L));
  for (unsigned i = _P - 1; i < _L + _P - 1; i++)
    gsl_vector_float_set(_buffer, i, 0.0);
  
  _increment();
  return _vector;
}

void OverlapAdd::reset()
{
  _samp->reset();  VectorFloatFeatureStream::reset();

  for (unsigned i = 0; i < _L + _P - 1; i++)
    gsl_vector_float_set(_buffer, i, 0.0);
}


// ----- methods for class `OverlapSave' -----
//
OverlapSave::OverlapSave(VectorFloatFeatureStreamPtr& samp,
			 const gsl_vector* impulseResponse, const String& nm)
  : VectorFloatFeatureStream(_checkOutputSize(impulseResponse->size, samp->size()), nm), _samp(samp),
    _L(_checkL(impulseResponse->size, samp->size())), _L2(_L/2), _P(impulseResponse->size),
    _section(new double[_L]), _frequencyResponse(gsl_vector_complex_alloc(_L/2+1))
{
  _setImpulseResponse(impulseResponse);
}

OverlapSave::~OverlapSave()
{
  delete[] _section;
  gsl_vector_complex_free(_frequencyResponse);
}

void OverlapSave::_setImpulseResponse(const gsl_vector* impulseResponse)
{
  if (impulseResponse == NULL) {
    gsl_vector_complex_set_zero(_frequencyResponse);
    return;
  }

  for (unsigned i = 0; i < _L; i++)
    _section[i] = 0.0;

  for (unsigned i = 0; i < _P; i++)
    _section[i] = gsl_vector_get(impulseResponse, i);
  gsl_fft_real_radix2_transform(_section, /* stride= */ 1, _L);
  _halfComplexUnpack(_frequencyResponse, _section);

  for (unsigned i = 0; i < _L; i++)
    _section[i] = 0.0;
}

unsigned OverlapSave::_checkOutputSize(unsigned irLen, unsigned sampLen)
{
  if (irLen >= sampLen)
    throw jdimension_error("Cannot have P = %d and L = %d", irLen, sampLen);

  return (sampLen - irLen);
}

// check consistency of FFT length
unsigned OverlapSave::_checkL(unsigned irLen, unsigned sampLen)
{
  // should check that _L is a power of 2
  if (irLen >= sampLen)
    throw jdimension_error("Cannot have P = %d and L = %d", irLen, sampLen);

  return sampLen;
}

// unpack the 'complex_packed_array' into 'gsl_vector_complex'
void OverlapSave::_halfComplexUnpack(gsl_vector_complex* tgt, const double* src)
{
  for (unsigned m = 0; m <= _L2; m++) {
    if (m == 0 || m == _L2) {
      gsl_vector_complex_set(tgt, m, gsl_complex_rect(src[m], 0.0));
    } else {
      gsl_vector_complex_set(tgt, m, gsl_complex_rect(src[m], src[_L-m]));
    }
  }
}

const gsl_vector_float* OverlapSave::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem: %d != %d\n", frameX - 1, _frameX);

  const gsl_vector_float* block = _samp->next(_frameX + 1);

  // forward FFT on new data
  for (unsigned i = 0; i < _L; i++)
    _section[i] = gsl_vector_float_get(block, i);
  gsl_fft_real_radix2_transform(_section, /*stride=*/ 1, _L);

  // multiply with frequency response
  for (unsigned i = 0; i <= _L2; i++) {
    if (i == 0 || i == _L2) {
      _section[i] = _section[i] * GSL_REAL(gsl_vector_complex_get(_frequencyResponse, i));
    } else {
      gsl_complex val = gsl_complex_mul(gsl_complex_rect(_section[i], _section[_L-i]), gsl_vector_complex_get(_frequencyResponse, i));
      _section[i]    = GSL_REAL(val);
      _section[_L-i] = GSL_IMAG(val);
    }
  }

  // inverse FFT
  gsl_fft_halfcomplex_radix2_inverse(_section, /* stride= */ 1, _L);

  // pick out linearly convolved portion
  for (unsigned i = _P ; i < _L; i++)
    gsl_vector_float_set(_vector, i - _P, _section[i]);

  _increment();
  return _vector;
}

void OverlapSave::reset()
{
  _samp->reset();  VectorFloatFeatureStream::reset();
}

void OverlapSave::update(const gsl_vector_complex* delta)
{
  if (delta->size != _L)
    throw jdimension_error("Dimension of udpate vector (%d) does not match frequency response (%d).",
			   delta->size, _L);

  for (unsigned i = 0; i < _L; i++)
    gsl_vector_complex_set(_frequencyResponse, i, gsl_complex_add(gsl_vector_complex_get(_frequencyResponse, i), gsl_vector_complex_get(delta, i)));
}
