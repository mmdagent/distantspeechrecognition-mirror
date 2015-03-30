//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.cancelVP
//  Purpose: Cancelation of a voice prompt based on either NLMS or Kalman
//	     filter algorithms.
//  Author:  John McDonough, Wei Chu and Kenichi Kumatani
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

#include <string.h>
#include <math.h>

#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_linalg.h>
#include "matrix/gslmatrix.h"

#include "common/jpython_error.h"
#include "cancelVP/cancelVP.h"

// ----- methods for class `NLMSAcousticEchoCancellationFeature' -----
//
NLMSAcousticEchoCancellationFeature::
NLMSAcousticEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded, 
				    double delta, double epsilon, double threshold, const String& nm)
  :  VectorComplexFeatureStream(played->size(), nm),
     _played(played), _recorded(recorded), _fftLen(played->size()), _fftLen2(_fftLen / 2), _filterCoefficient(gsl_vector_complex_alloc(_fftLen)),
     _delta(delta), _epsilon(epsilon), _threshold(threshold) { }


NLMSAcousticEchoCancellationFeature::~NLMSAcousticEchoCancellationFeature()
{
  gsl_vector_complex_free(_filterCoefficient);
}

bool NLMSAcousticEchoCancellationFeature::_update(const gsl_complex Vk)
{
  double energy = gsl_complex_abs2(Vk);

  return (energy > _threshold);
}

const gsl_vector_complex* NLMSAcousticEchoCancellationFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
                       name().c_str(), frameX - 1, _frameX);

  const gsl_vector_complex* playBlock = _played->next(_frameX + 1);
  const gsl_vector_complex* recordBlock = _recorded->next(_frameX + 1);

  for (unsigned k = 0; k <= _fftLen2; k++) {
    gsl_complex Vk = gsl_vector_complex_get(playBlock, k);
    gsl_complex Ak = gsl_vector_complex_get(recordBlock, k);
    gsl_complex Rk = gsl_vector_complex_get(_filterCoefficient, k);
    
    gsl_complex Ek = gsl_complex_sub(Ak, gsl_complex_mul(Rk, Vk));
    if (k > 0 && k < _fftLen2)
      gsl_vector_complex_set(_vector, _fftLen - k, gsl_complex_conjugate(Ek));

    gsl_vector_complex_set(_vector, k, Ek);

    if (_update(Vk)) {
      gsl_complex Gkhat = gsl_complex_div(Ak, Vk);
      gsl_complex dC    = gsl_complex_sub(Rk, Gkhat);
      double Vk2        = gsl_complex_abs2(Vk);
      double Ak2        = gsl_complex_abs2(Ak);

      gsl_complex deltaC = gsl_complex_mul_real(dC, _epsilon * Vk2/(_delta + Ak2));

      gsl_complex nC = gsl_complex_sub(Rk, deltaC);
      gsl_vector_complex_set(_filterCoefficient, k, nC);
      if (k > 0 && k < _fftLen2)
	gsl_vector_complex_set(_filterCoefficient, _fftLen - k, gsl_complex_conjugate(nC));

      if (k == 20) {
	printf("deltaC  = (%g + %gj)\n", GSL_REAL(deltaC), GSL_IMAG(deltaC));
	printf("Vk2     = %g\n", Vk2);
	printf("Ak2     = %g\n", Ak2);
	printf("Rk      = (%g + %gj)\n", GSL_REAL(nC), GSL_IMAG(nC));
	fflush(stdout);
      }
    }
  }

  _increment();
  return _vector;
}


// ----- methods for class `KalmanFilterEchoCancellationFeature' -----
//
KalmanFilterEchoCancellationFeature::
KalmanFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
				    double beta, double sigma2, double threshold, const String& nm)
  :  VectorComplexFeatureStream(played->size(), nm),
     _played(played), _recorded(recorded), _fftLen(played->size()), _fftLen2(_fftLen / 2),
     _filterCoefficient(gsl_vector_complex_calloc(_fftLen)), _sigma2_v(gsl_vector_calloc(_fftLen)),
     _K_k(gsl_vector_calloc(_fftLen)), _beta(beta), _threshold(threshold), _sigma2_u(sigma2)
{
  // Initialize variances
  for (unsigned m = 0; m < _fftLen; m++) {
    gsl_vector_set(_sigma2_v, m, sigma2);
    gsl_vector_set(_K_k, m, sigma2);
  }
}


KalmanFilterEchoCancellationFeature::~KalmanFilterEchoCancellationFeature()
{
  gsl_vector_complex_free(_filterCoefficient);
  gsl_vector_free(_sigma2_v);
  gsl_vector_free(_K_k);
}


bool KalmanFilterEchoCancellationFeature::_update(const gsl_complex Vk)
{
  double energy = gsl_complex_abs2(Vk);

  return (energy > _threshold);
}


const gsl_vector_complex* KalmanFilterEchoCancellationFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
                       name().c_str(), frameX - 1, _frameX);

  const gsl_vector_complex* playBlock	= _played->next(_frameX + 1);
  const gsl_vector_complex* recordBlock	= _recorded->next(_frameX + 1);

  for (unsigned m = 0; m <= _fftLen2; m++) {
          gsl_complex Ak = gsl_vector_complex_get(recordBlock, m);
          gsl_complex Rk = gsl_vector_complex_get(_filterCoefficient, m);
    const gsl_complex Vk = gsl_vector_complex_get(playBlock, m);
    
    // Calculate the residual signal; i.e., the desired speech, which is also the innovation
    gsl_complex Ek = gsl_complex_sub(Ak, gsl_complex_mul(Rk, Vk));
    gsl_vector_complex_set(_vector, m, Ek);
    if (m > 0 && m < _fftLen2)
      gsl_vector_complex_set(_vector, _fftLen - m, gsl_complex_conjugate(Ek));

    if (_update(Vk)) {

      // Estimate the observation noise variance
      double      Ek2		= gsl_complex_abs2(Ek);
      double	  sigma2_v	= _beta * gsl_vector_get(_sigma2_v, m) + (1.0 - _beta) * Ek2;
      gsl_vector_set(_sigma2_v, m, sigma2_v);

      // Calculate the Kalman gain
      double      Vk2		= gsl_complex_abs2(Vk);
      double      K_k_k1	= gsl_vector_get(_K_k, m) + _sigma2_u;
      double      sigma2_s	= Vk2 * K_k_k1 + sigma2_v;
      gsl_complex Gk		= gsl_complex_mul_real(gsl_complex_conjugate(Vk), K_k_k1 / sigma2_s);

      // Update the filter weight
      Rk			= gsl_complex_add(Rk, gsl_complex_mul(Gk, Ek));
      gsl_vector_complex_set(_filterCoefficient, m, Rk);

      // Store the state estimation error variance for next the iteration
      double 	  K_k		= (1.0 - K_k_k1 * Vk2 / sigma2_s) * K_k_k1;
      gsl_vector_set(_K_k, m, K_k);

      /*
      if (m == 20) {
	printf("FrameX		= %d\n", frameX);
	printf("-------------------------------------------\n");
	printf("Vk2		= %g\n", Vk2);
	printf("Ek2		= %g\n", Ek2);
	printf("K_k_k1		= %g\n", K_k_k1);
	printf("K_k		= %g\n", K_k);
	printf("sigma2_s	= %g\n", sigma2_s);
	printf("sigma2_v	= %g\n", sigma2_v);
	printf("Gk		= (%g + %gj)\n", GSL_REAL(Gk), GSL_IMAG(Gk));
	printf("Ek		= (%g + %gj)\n", GSL_REAL(Ek), GSL_IMAG(Ek));
	printf("Rk		= (%g + %gj)\n", GSL_REAL(Rk), GSL_IMAG(Rk));
	printf("Ak		= (%g + %gj)\n", GSL_REAL(Ak), GSL_IMAG(Ak));
	printf("Vk		= (%g + %gj)\n", GSL_REAL(Vk), GSL_IMAG(Vk));
	printf("\n");

	fflush(stdout);
      }
      */
    }
  }

  _increment();
  return _vector;
}

// ----- methods for class `BlockKalmanFilterEchoCancellationFeature' -----
//
gsl_complex BlockKalmanFilterEchoCancellationFeature::_ComplexOne  = gsl_complex_rect(1.0, 0.0);
gsl_complex BlockKalmanFilterEchoCancellationFeature::_ComplexZero = gsl_complex_rect(0.0, 0.0);

BlockKalmanFilterEchoCancellationFeature::
BlockKalmanFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
					 unsigned sampleN, double beta, double sigmau2, double sigmak2, double threshold, double amp4play, const String& nm)
  :  VectorComplexFeatureStream(played->size(), nm),
     _played(played), _recorded(recorded), _fftLen(played->size()), _fftLen2(_fftLen / 2), _sampleN(sampleN),
     _buffer(_fftLen, _sampleN), _filterCoefficient(new gsl_vector_complex*[_fftLen]),
     _sigma2_v(gsl_vector_calloc(_fftLen)), _K_k(new gsl_matrix_complex*[_fftLen]),
     _K_k_k1(gsl_matrix_complex_calloc(_sampleN, _sampleN)),
     _beta(beta), _threshold(threshold), _Sigma2_u(new gsl_matrix_complex*[_fftLen]),
     _Gk(gsl_vector_complex_calloc(_sampleN)),
     _scratch(gsl_vector_complex_calloc(_sampleN)),
     _scratch2(gsl_vector_complex_calloc(_sampleN)),
     _scratchMatrix(gsl_matrix_complex_calloc(_sampleN, _sampleN)),
     _scratchMatrix2(gsl_matrix_complex_calloc(_sampleN, _sampleN)),
     _amp4play(amp4play),_skippedN(0),_maxSkippedN(30)
{
  // Initialize variances
  for (unsigned m = 0; m < _fftLen; m++)
    gsl_vector_set(_sigma2_v, m, sigmau2);

  // Initialize subband-dependent covariance matrices
  for (unsigned m = 0; m < _fftLen; m++) {
    _filterCoefficient[m] = gsl_vector_complex_calloc(_sampleN);
    _K_k[m]               = gsl_matrix_complex_calloc(_sampleN, _sampleN);
    _Sigma2_u[m]          = gsl_matrix_complex_calloc(_sampleN, _sampleN);

    for (unsigned n = 0; n < _sampleN; n++) {
      gsl_matrix_complex_set(_K_k[m], n, n, gsl_complex_rect(sigmak2, 0.0));
      gsl_matrix_complex_set(_Sigma2_u[m], n, n, gsl_complex_rect(sigmau2, 0.0));
    }
  }
}


BlockKalmanFilterEchoCancellationFeature::~BlockKalmanFilterEchoCancellationFeature()
{
  gsl_vector_free(_sigma2_v);
  gsl_vector_complex_free(_Gk);
  gsl_matrix_complex_free(_K_k_k1);
  gsl_vector_complex_free(_scratch);
  gsl_vector_complex_free(_scratch2);
  gsl_matrix_complex_free(_scratchMatrix);
  gsl_matrix_complex_free(_scratchMatrix2);
  for (unsigned m = 0; m < _fftLen; m++) {
    gsl_vector_complex_free(_filterCoefficient[m]);
    gsl_matrix_complex_free(_K_k[m]);
    gsl_matrix_complex_free(_Sigma2_u[m]);
  }
  delete[] _filterCoefficient;
  delete[] _K_k;
  delete[] _Sigma2_u;
}


bool BlockKalmanFilterEchoCancellationFeature::_update(const gsl_vector_complex* Vk)
{
  double energy = gsl_complex_abs2(gsl_vector_complex_get(Vk, /* sampleX= */ 0));

  return (energy > _threshold);
}


void BlockKalmanFilterEchoCancellationFeature::_conjugate(gsl_vector_complex* dest, const gsl_vector_complex* src) const
{
  if (src->size != dest->size)
    throw jdimension_error("BlockKalmanFilterEchoCancellationFeature::_conjugate:: Vector sizes (%d vs. %d) do not match\n", src->size, dest->size);
  for (unsigned n = 0; n < dest->size; n++)
    gsl_vector_complex_set(dest, n, gsl_complex_conjugate((gsl_vector_complex_get(src, n))));
}


const gsl_vector_complex* BlockKalmanFilterEchoCancellationFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
                       name().c_str(), frameX - 1, _frameX);

  const gsl_vector_complex* playBlock	= _played->next(_frameX + 1);
  const gsl_vector_complex* recordBlock	= _recorded->next(_frameX + 1);
  _buffer.nextSample(playBlock,_amp4play);

  for (unsigned m = 0; m <= _fftLen2; m++) {
    gsl_complex         Ak = gsl_vector_complex_get(recordBlock, m);
    gsl_vector_complex* Rk = _filterCoefficient[m];
    const gsl_vector_complex* Vk = _buffer.getSamples(m);
    
    // Calculate the residual signal; i.e., the desired speech, which is also the innovation
    gsl_complex iprod;
    gsl_blas_zdotu(Rk, Vk, &iprod);
    gsl_complex Ek = gsl_complex_sub(Ak, iprod);
    gsl_vector_complex_set(_vector, m, Ek);
    if (m > 0 && m < _fftLen2)
      gsl_vector_complex_set(_vector, _fftLen - m, gsl_complex_conjugate(Ek));

    if (m == 20) {
      printf("FrameX		= %d\n", frameX);
      printf("-------------------------------------------\n");
      fflush(stdout);
    }
    double Ek2;
    if (_update(Vk)) {

      // Estimate the observation noise variance
      Ek2      = gsl_complex_abs2(Ek);
      double sigma2_v = _beta * gsl_vector_get(_sigma2_v, m) + (1.0 - _beta) * Ek2;
      gsl_vector_set(_sigma2_v, m, sigma2_v);

      // Calculate the Kalman gain
      gsl_matrix_complex_memcpy(_K_k_k1, _Sigma2_u[m]);
      gsl_matrix_complex_add(_K_k_k1, _K_k[m]);
      
      _conjugate(_scratch2, Vk);
      gsl_blas_zgemv(CblasNoTrans, _ComplexOne, _K_k_k1, _scratch2, _ComplexZero, _scratch);
      gsl_blas_zdotu(Vk, _scratch, &iprod);
      /*
      if (m == 20) {
	printf("Intermediate: iprod = %g : Vk2 = %g\n", GSL_REAL(iprod), gsl_complex_abs2(gsl_vector_complex_get(Vk, 0)));
	fflush(stdout);
      }
      */
      double sigma2_s = GSL_REAL(iprod) + sigma2_v;
      gsl_vector_complex_set_zero(_Gk);
      gsl_blas_zaxpy(gsl_complex_rect(1.0 / sigma2_s, 0.0), _scratch, _Gk);

      // Update the filter weights
      gsl_blas_zaxpy(Ek, _Gk, Rk);
      
      // Store the state estimation error variance for next the iteration
      gsl_matrix_complex_set_zero(_scratchMatrix);
      for (unsigned rowX = 0; rowX < _sampleN; rowX++) {
	for (unsigned colX = 0; colX < _sampleN; colX++) {
	  gsl_complex diagonal = ((rowX == colX) ? _ComplexOne : _ComplexZero);
	  gsl_complex value    =  gsl_complex_sub(diagonal, gsl_complex_mul(gsl_vector_complex_get(_Gk, rowX), gsl_vector_complex_get(Vk, colX)));
	  gsl_matrix_complex_set(_scratchMatrix, rowX, colX, value);
	}
      }
      gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, _ComplexOne, _scratchMatrix, _K_k_k1, _ComplexZero, _K_k[m]);

      if (m == 20) {
	printf("K_k_k1[0][0]	= %g\n", GSL_REAL(gsl_matrix_complex_get(_K_k_k1, 0, 0)));
	printf("K_k[0][0]	= %g\n", GSL_REAL(gsl_matrix_complex_get(_K_k[m], 0, 0)));
	printf("sigma2_s	= %g\n", sigma2_s);
	printf("sigma2_v	= %g\n", sigma2_v);
	printf("Gk[0]		= (%g + %gj)\n", GSL_REAL(gsl_vector_complex_get(_Gk, 0)), GSL_IMAG(gsl_vector_complex_get(_Gk, 0)));
	
	fflush(stdout);
      }
    }
    
    if (m == 20) {
      printf("Vk2		= %g\n", gsl_complex_abs2(gsl_vector_complex_get(Vk, 0)));
      printf("Ek2		= %g\n", Ek2);
      printf("Ek		= (%g + %gj)\n", GSL_REAL(Ek), GSL_IMAG(Ek));
      printf("Rk[0]		= (%g + %gj)\n", GSL_REAL(gsl_vector_complex_get(Rk, 0)), GSL_IMAG(gsl_vector_complex_get(Rk, 0)));
      printf("Rk2		= %g\n", gsl_complex_abs2(gsl_vector_complex_get(Rk, 0)));
      printf("Ak		= (%g + %gj)\n", GSL_REAL(Ak), GSL_IMAG(Ak));
      printf("Vk[0]		= (%g + %gj)\n", GSL_REAL(gsl_vector_complex_get(Vk, 0)), GSL_IMAG(gsl_vector_complex_get(Vk, 0)));
      printf("\n");

      fflush(stdout);
    }
  }

  _increment();
  return _vector;
}


// ----- methods for class `InformationFilterEchoCancellationFeature' -----
//
InformationFilterEchoCancellationFeature::
InformationFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
					 unsigned sampleN, double beta, double sigmau2, double sigmak2, double snrTh, double engTh, double smooth,
					 double loading, double amp4play, const String& nm)
  : BlockKalmanFilterEchoCancellationFeature(played, recorded, sampleN, beta, sigmau2, sigmak2, snrTh, amp4play, nm),
    _smoothEk(smooth), _smoothSk(smooth), _engTh(engTh),
    _snr(new double[_fftLen]), _EkEnergy(new double[_fftLen]), _SkEnergy(new double[_fftLen]), _loading(loading),
    _inverse(gsl_matrix_complex_calloc(_sampleN, _sampleN)), _eigenWorkSpace(gsl_eigen_hermv_alloc(_sampleN)), _evalues(gsl_vector_calloc(_sampleN)),
    _scratchInverse(gsl_vector_complex_calloc(_sampleN)),
    _scratchMatrixInverse(gsl_matrix_complex_calloc(_sampleN, _sampleN)),
    _scratchMatrixInverse2(gsl_matrix_complex_calloc(_sampleN, _sampleN)),
    _matrixCopy(gsl_matrix_complex_calloc(_sampleN, _sampleN))
{
  for (unsigned m = 0; m < _fftLen; m++) {
    _snr[m] = _EkEnergy[m] = _SkEnergy[m] = 0.0;

    // initialize filter coefficients
    gsl_vector_complex_set(_filterCoefficient[m], /* n= */ 0, gsl_complex_rect(1.0, 0.0));
    for (unsigned n = 1; n < _sampleN; n++)
      gsl_vector_complex_set(_filterCoefficient[m], n, gsl_complex_rect(/* 1.0e-04 */ 0.0, 0.0));
  }

  _floorVal = 0.01;
  // if debug open 
  // _fdb = fopen("/home/wei/src/wav/debug.txt", "w");
}


InformationFilterEchoCancellationFeature::~InformationFilterEchoCancellationFeature()
{
  delete[] _snr;
  delete[] _EkEnergy;
  delete[] _SkEnergy;

  gsl_matrix_complex_free(_inverse);
  gsl_eigen_hermv_free(_eigenWorkSpace);
  gsl_vector_complex_free(_scratchInverse);
  gsl_matrix_complex_free(_scratchMatrixInverse);
  gsl_matrix_complex_free(_scratchMatrixInverse2);
  gsl_matrix_complex_free(_matrixCopy);
}

void InformationFilterEchoCancellationFeature::_printMatrix(const gsl_matrix_complex* mat)
{
  for (unsigned m = 0; m < mat->size1; m++) {
    for (unsigned n = 0; n < mat->size2; n++) {
      gsl_complex value = gsl_matrix_complex_get(mat, m, n);
      printf("%8.4f %8.4f  ", GSL_REAL(value), GSL_IMAG(value));
    }
    printf("\n");
  }
}

void InformationFilterEchoCancellationFeature::_printVector(const gsl_vector_complex* vec)
{
  for (unsigned n = 0; n < vec->size; n++) {
    gsl_complex value = gsl_vector_complex_get(vec, n);
    printf("%8.4f %8.4f\n", GSL_REAL(value), GSL_IMAG(value));
  }
}

double InformationFilterEchoCancellationFeature::_updateBand(const gsl_complex Ak, const gsl_complex Ek, int frameX, unsigned m)
{
  double smthEk, smthSk;
  // if it is the first 100 frames
  if (frameX < 100) {
    smthEk = 1.0 - (double) frameX * (1.0 - _smoothEk) / 100.0;
    smthSk = 1.0 - (double) frameX * (1.0 - _smoothSk) / 100.0;
  } else {
    smthEk = _smoothEk;
    smthSk = _smoothSk;
  }

  double sf;
  const gsl_complex Sk = gsl_complex_sub(Ak, Ek);
  double currEkEng = gsl_complex_abs2(Ek);
  double currSkEng = gsl_complex_abs2(Sk); // Ek to Ak
  _EkEnergy[m] = currEkEng * smthEk + _EkEnergy[m] * (1.0 - smthEk);
  _SkEnergy[m] = currSkEng * smthSk + _SkEnergy[m] * (1.0 - smthSk);
  double currSnr = currSkEng / (currEkEng + 1.0e-15);
  _snr[m] = currSnr * smthEk + _snr[m] * (1.0 - smthEk);
  if (frameX < 100 || (_snr[m] > _threshold && _SkEnergy[m] > _engTh))
    sf = 2.0 / (1.0 + exp(-_snr[m])) - 1.0;             // snr -> inf, sf -> 1; snr-> 0
  else
    sf = -1.0;

  return sf;
}

double InformationFilterEchoCancellationFeature::_EigenValueThreshold = 1.0e-06;

void InformationFilterEchoCancellationFeature::_invert(gsl_matrix_complex* matrix)
{
  // perform eigen decomposition
  gsl_matrix_complex_memcpy(_matrixCopy, matrix);
  gsl_eigen_hermv(_matrixCopy, _evalues, _scratchMatrixInverse, _eigenWorkSpace);

  // find maximum eigenvalue
  double maxEvalue = 0.0;
  for (unsigned n = 0; n < _sampleN; n++) {
    double value = gsl_vector_get(_evalues, n);
    if (value > maxEvalue)
      maxEvalue = value;
  }

  // scale columns by inverse of eigenvector
  for (unsigned n = 0; n < _sampleN; n++) {
    double value = gsl_vector_get(_evalues, n);
    // if ((value / maxEvalue) < _EigenValueThreshold) continue;
    double scale = 1.0 / value;
    for (unsigned m = 0; m < _sampleN; m++) {
      gsl_matrix_complex_set(_scratchMatrixInverse2, m, n, gsl_complex_mul_real(gsl_matrix_complex_get(_scratchMatrixInverse, m, n), scale));
    }
  }

  // final matrix-matrix multiply to get the psuedo-inverse
  gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, _ComplexOne, _scratchMatrixInverse2, _scratchMatrixInverse, _ComplexZero, _inverse);

  /*
  gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, _ComplexOne, matrix, _inverse, _ComplexZero, _scratchMatrixInverse);
  gsl_matrix_complex_fprintf(stdout, _scratchMatrixInverse, "%8.4e");
  printf("Done\n");
  */
}

const gsl_vector_complex* InformationFilterEchoCancellationFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_complex* playBlock	= _played->next(_frameX + 1);
  const gsl_vector_complex* recordBlock	= _recorded->next(_frameX + 1);
  _buffer.nextSample(playBlock,_amp4play);

  for (unsigned m = 0; m <= _fftLen2; m++) {
    gsl_complex			Ak = gsl_vector_complex_get(recordBlock, m);
    gsl_vector_complex*		Rk = _filterCoefficient[m];
    const gsl_vector_complex*	Vk = _buffer.getSamples(m);

    // Calculate the residual signal; i.e., the desired speech, which is also the innovation
    gsl_complex iprod;
    gsl_blas_zdotu(Rk, Vk, &iprod);
    gsl_complex Ek = gsl_complex_sub(Ak, iprod);
    double absEk   = gsl_complex_abs(Ek);
    if ( absEk < _floorVal ) 
      Ek = gsl_complex_rect( GSL_REAL(Ek)/absEk, GSL_IMAG(Ek)/absEk ); 
    
    gsl_vector_complex_set(_vector, m, Ek);
    if (m > 0 && m < _fftLen2)
      gsl_vector_complex_set(_vector, _fftLen - m, gsl_complex_conjugate(Ek));

    /*
    if (m == 20) {
      printf("FrameX		= %d\n", frameX);
      printf("-------------------------------------------\n");
      printf("K_k_k1[0][0]	= %g\n", GSL_REAL(gsl_matrix_complex_get(_K_k_k1, 0, 0)));
      fflush(stdout);
    }
    */

    if (_update(Vk) == false || _updateBand(Ak, Ek, frameX, m) < 0.0){
      if( _skippedN >= _maxSkippedN ){
	// initialize filter coefficients
	gsl_vector_complex_set(_filterCoefficient[m], 0, gsl_complex_rect(1.0, 0.0));
	for (unsigned n = 1; n < _sampleN; n++)
	  gsl_vector_complex_set(_filterCoefficient[m], n, gsl_complex_rect(/* 1.0e-04 */ 0.0, 0.0));
	_skippedN = 0;
      }
      _skippedN++;
      continue;
    }

    // Estimate the observation noise variance
    double Ek2		= gsl_complex_abs2(Ek);
    double sigma2_v	= _beta * gsl_vector_get(_sigma2_v, m) + (1.0 - _beta) * Ek2;
    gsl_vector_set(_sigma2_v, m, sigma2_v);

    // Perform the prediction step; _scratch = y_{k|k-1}, and _inverse = Y_{k|k-1}
    gsl_matrix_complex_memcpy(_K_k_k1, _Sigma2_u[m]);
    gsl_matrix_complex_add(_K_k_k1, _K_k[m]);
    _invert(_K_k_k1);
    gsl_blas_zgemv(CblasNoTrans, _ComplexOne, _inverse, Rk, _ComplexZero, _scratch);

    /*
    if (m == 20) {
      printf("After temporal update:\n");
      _printMatrix(_inverse);
      printf("\n");
      _printVector(_scratch);
      printf("\n");
    }
    */

    // form the matrix I_k = _scratchMatrix and vector i_k = _scratch2
    double scale = 1.0 / sigma2_v;
    for (unsigned rowX = 0; rowX < _sampleN; rowX++) {
      gsl_complex value = gsl_complex_mul_real(gsl_complex_conjugate(gsl_vector_complex_get(Vk, rowX)), scale);
      gsl_vector_complex_set(_scratch2, rowX, gsl_complex_mul(value, Ak));

      for (unsigned colX = 0; colX < _sampleN; colX++) {
	gsl_complex colV = gsl_vector_complex_get(Vk, colX);
	gsl_matrix_complex_set(_scratchMatrix, rowX, colX, gsl_complex_mul(value, colV));
      }
    }

    // now perform the information correction/update step
    gsl_matrix_complex_add(_scratchMatrix, _inverse);
    gsl_vector_complex_add(_scratch, _scratch2);

    /*
    if (m == 20) {
      printf("After observational update:\n");
      _printMatrix(_scratchMatrix);
      printf("\n");
      _printVector(_scratch);
      printf("\n");
    }
    */

    // extra diagonal loading to limit the size of the filter coefficients
    static const gsl_complex load = gsl_complex_rect(_loading, 0.0);
    for (unsigned diagX = 0; diagX < _sampleN; diagX++) {
      gsl_complex diagonal = gsl_complex_add(gsl_matrix_complex_get(_scratchMatrix, diagX, diagX), load);
      gsl_matrix_complex_set(_scratchMatrix, diagX, diagX, diagonal);
    }

    // extract filter coefficients from information vector and store
    _invert(_scratchMatrix);
    gsl_matrix_complex_memcpy(_K_k[m], _inverse);
    gsl_blas_zgemv(CblasNoTrans, _ComplexOne, _inverse, _scratch, _ComplexZero, Rk);

    /*
    if (m == 20) {
      printf("After diagonal loading:\n");
      _printMatrix(_scratchMatrix);
      printf("\n");
      _printVector(Rk);
      printf("Done\n\n");
    }
    */

    /*
    if (m == 20) {
      printf("K_k[0][0]	= %g\n", GSL_REAL(gsl_matrix_complex_get(_K_k[m], 0, 0)));
      printf("sigma2_v	= %g\n", sigma2_v);
      printf("Vk2		= %g\n", gsl_complex_abs2(gsl_vector_complex_get(Vk, 0)));
      printf("Ek2		= %g\n", Ek2);
      printf("Ek		= (%g + %gj)\n", GSL_REAL(Ek), GSL_IMAG(Ek));
      printf("Rk[0]		= (%g + %gj)\n", GSL_REAL(gsl_vector_complex_get(Rk, 0)), GSL_IMAG(gsl_vector_complex_get(Rk, 0)));
      printf("Rk2		= %g\n", gsl_complex_abs2(gsl_vector_complex_get(Rk, 0)));
      printf("Ak		= (%g + %gj)\n", GSL_REAL(Ak), GSL_IMAG(Ak));
      printf("Vk[0]		= (%g + %gj)\n", GSL_REAL(gsl_vector_complex_get(Vk, 0)), GSL_IMAG(gsl_vector_complex_get(Vk, 0)));
      printf("\n");
      fflush(stdout);
    }
    */
  }

  _increment();
  return _vector;
}


// ----- methods for class `SquareRootInformationFilterEchoCancellationFeature' -----
//
SquareRootInformationFilterEchoCancellationFeature::
SquareRootInformationFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
						   unsigned sampleN, double beta, double sigmau2, double sigmak2, double snrTh, double engTh, double smooth,
						   double loading, double amp4play, const String& nm)
  : InformationFilterEchoCancellationFeature(played, recorded, sampleN, beta, sigmau2, sigmak2, snrTh, engTh, smooth, loading, amp4play, nm),
    _load(gsl_complex_rect(sqrt(_loading), 0.0)),
    _informationState(new gsl_vector_complex*[_fftLen])
{
  // Reallocated _scratchMatrix and _scratchMatrix2 for the temporal and observational updates respectively
  gsl_matrix_complex_free(_scratchMatrix);
  gsl_matrix_complex_free(_scratchMatrix2);
  _scratchMatrix  = gsl_matrix_complex_calloc((2 * _sampleN) + 1, 2 * _sampleN);
  _scratchMatrix2 = gsl_matrix_complex_calloc(_sampleN + 1, _sampleN + 1);

  // Initialize subband-dependent covariance matrices with the inverse Cholesky factors
  gsl_complex diagonal = gsl_complex_rect(1.0 / sqrt(sigmau2), 0.0);
  for (unsigned m = 0; m < _fftLen; m++) {
    gsl_matrix_complex_set_zero(_K_k[m]);
    gsl_matrix_complex_set_zero(_Sigma2_u[m]);
    for (unsigned n = 0; n < _sampleN; n++) {
      gsl_matrix_complex_set(_K_k[m], n, n, diagonal);
      gsl_matrix_complex_set(_Sigma2_u[m], n, n, diagonal);
    }

    _informationState[m] = gsl_vector_complex_calloc(_sampleN);
  }
}

SquareRootInformationFilterEchoCancellationFeature::~SquareRootInformationFilterEchoCancellationFeature()
{
  for (unsigned m = 0; m < _fftLen; m++)
    gsl_vector_complex_free(_informationState[m]);

  delete[] _informationState;
}

// calculate the cosine 'c' and sine 's' parameters of Givens
// rotation that rotates 'v2' into 'v1'
gsl_complex SquareRootInformationFilterEchoCancellationFeature::
_calcGivensRotation(const gsl_complex& v1, const gsl_complex& v2,
		    gsl_complex& c, gsl_complex& s)
{
  double norm = sqrt(gsl_complex_abs2(v1) + gsl_complex_abs2(v2));

  if (norm == 0.0)
    throw jarithmetic_error("calcGivensRotation: Norm is zero.");

  c = gsl_complex_div_real(v1, norm);
  s = gsl_complex_div_real(gsl_complex_conjugate(v2), norm);

  return gsl_complex_rect(norm, 0.0);
}

// apply a previously calculated Givens rotation
void SquareRootInformationFilterEchoCancellationFeature::
_applyGivensRotation(const gsl_complex& v1, const gsl_complex& v2,
		     const gsl_complex& c, const gsl_complex& s,
		     gsl_complex& v1p, gsl_complex& v2p)
{
  v1p =
    gsl_complex_add(gsl_complex_mul(gsl_complex_conjugate(c), v1),
		    gsl_complex_mul(s, v2));
  v2p =
    gsl_complex_sub(gsl_complex_mul(c, v2),
		    gsl_complex_mul(gsl_complex_conjugate(s), v1));
}

// extract covariance state from square-root information state
void SquareRootInformationFilterEchoCancellationFeature::
_extractCovarianceState(const gsl_matrix_complex* K_k, const gsl_vector_complex* sk, gsl_vector_complex* xk)
{
  for (int sampX = _sampleN - 1; sampX >= 0; sampX--) {
    gsl_complex skn = gsl_complex_conjugate(gsl_vector_complex_get(sk, sampX));
    for (int n = _sampleN - 1; n > sampX; n--) {
      gsl_complex xkn    = gsl_vector_complex_get(xk, n);
      gsl_complex K_k_mn = gsl_complex_conjugate(gsl_matrix_complex_get(K_k, n, sampX));
      skn = gsl_complex_sub(skn, gsl_complex_mul(K_k_mn, xkn));
    }
    gsl_vector_complex_set(xk, sampX, gsl_complex_div(skn, gsl_complex_conjugate(gsl_matrix_complex_get(K_k, sampX, sampX))));
  }
}

void SquareRootInformationFilterEchoCancellationFeature::
_negative(gsl_matrix_complex* dest, const gsl_matrix_complex* src)
{
  for (unsigned rowX = 0; rowX < _sampleN; rowX++) {
    for (unsigned colX = 0; colX < _sampleN; colX++) {
      gsl_complex value = gsl_matrix_complex_get(src, rowX, colX);
      gsl_matrix_complex_set(dest, rowX, colX, gsl_complex_rect(-GSL_REAL(value), -GSL_IMAG(value)));
    }
  }
}

const gsl_vector_complex* SquareRootInformationFilterEchoCancellationFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  const gsl_vector_complex* playBlock	= _played->next(_frameX + 1);
  const gsl_vector_complex* recordBlock	= _recorded->next(_frameX + 1);
  _buffer.nextSample(playBlock,_amp4play);

  for (unsigned m = 0; m <= _fftLen2; m++) {
    gsl_complex			Ak = gsl_vector_complex_get(recordBlock, m);
    gsl_vector_complex*		Rk = _filterCoefficient[m];
    const gsl_vector_complex*	Vk = _buffer.getSamples(m);

    // Calculate the residual signal; i.e., the desired speech, which is also the innovation
    gsl_complex iprod;
    gsl_blas_zdotu(Rk, Vk, &iprod);
    gsl_complex Ek = gsl_complex_sub(Ak, iprod);
    gsl_vector_complex_set(_vector, m, Ek);
    if (m > 0 && m < _fftLen2)
      gsl_vector_complex_set(_vector, _fftLen - m, gsl_complex_conjugate(Ek));

    /*
    if (m == 20) {
      printf("FrameX		= %d\n", frameX);
      printf("-------------------------------------------\n");
      printf("K_k_k1[0][0]	= %g\n", GSL_REAL(gsl_matrix_complex_get(_K_k_k1, 0, 0)));
      fflush(stdout);
    }
    */

    if (_update(Vk) == false || _updateBand(Ak, Ek, frameX, m) < 0.0) continue;

    // Estimate the observation noise variance
    double Ek2		= gsl_complex_abs2(Ek);
    double sigma2_v	= _beta * gsl_vector_get(_sigma2_v, m) + (1.0 - _beta) * Ek2;
    gsl_vector_set(_sigma2_v, m, sigma2_v);

    // perform prediction, correction, and add diagonal loading
    _temporalUpdate(m);

    /*
    if (m == 20) {
      printf("After temporal update:\n");
      gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, _ComplexOne, _K_k[m], _K_k[m], _ComplexZero, _inverse);
      _printMatrix(_inverse);
      printf("\n");
      _extractCovarianceState(_K_k[m], _informationState[m], _scratch);
      _printVector(_scratch);
      printf("\n");
    }
    */

    _observationalUpdate(m, Ak, sigma2_v);

    /*
    if (m == 20) {
      printf("After observational update:\n");
      gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, _ComplexOne, _K_k[m], _K_k[m], _ComplexZero, _inverse);
      _printMatrix(_inverse);
      printf("\n");
      _extractCovarianceState(_K_k[m], _informationState[m], _scratch);
      _printVector(_scratch);
      printf("\n");
    }
    */

    _diagonalLoading(m);

    // extract filter coefficients from information vector and store
    _extractCovarianceState(_K_k[m], _informationState[m], Rk);

    /*
    if (m == 20) {
      printf("After diagonal loading:\n");
      gsl_blas_zgemm(CblasNoTrans, CblasConjTrans, _ComplexOne, _K_k[m], _K_k[m], _ComplexZero, _inverse);
      _printMatrix(_inverse);
      printf("\n");
      _printVector(Rk);
      printf("Done\n\n");
    }
    */

    /*
    if (m == 20) {
      printf("State:\n");
      _printVector(Rk);
      printf("\n");
      printf("K_k[0][0]	= %g\n", GSL_REAL(gsl_matrix_complex_get(_K_k[m], 0, 0)));
      printf("sigma2_v	= %g\n", sigma2_v);
      printf("Vk2		= %g\n", gsl_complex_abs2(gsl_vector_complex_get(Vk, 0)));
      printf("Ek2		= %g\n", Ek2);
      printf("Ek		= (%g + %gj)\n", GSL_REAL(Ek), GSL_IMAG(Ek));
      printf("Rk[0]		= (%g + %gj)\n", GSL_REAL(gsl_vector_complex_get(Rk, 0)), GSL_IMAG(gsl_vector_complex_get(Rk, 0)));
      printf("Rk2		= %g\n", gsl_complex_abs2(gsl_vector_complex_get(Rk, 0)));
      printf("Ak		= (%g + %gj)\n", GSL_REAL(Ak), GSL_IMAG(Ak));
      printf("Vk[0]		= (%g + %gj)\n", GSL_REAL(gsl_vector_complex_get(Vk, 0)), GSL_IMAG(gsl_vector_complex_get(Vk, 0)));
      printf("\n");
      fflush(stdout);
    }
    */
  }

  _increment();
  return _vector;
}

static gsl_complex ComplexZero = gsl_complex_rect(0.0, 0.0);

void SquareRootInformationFilterEchoCancellationFeature::_temporalUpdate(unsigned m)
{
  // copy in elements of the pre-array
  gsl_matrix_complex_set_zero(_scratchMatrix);
  gsl_matrix_complex_view A11(gsl_matrix_complex_submatrix(_scratchMatrix,  /* k1= */ 0, /* k2= */ 0,
							   /* n1= */ _sampleN, /* n2= */ _sampleN));
  gsl_matrix_complex_memcpy(&A11.matrix, _Sigma2_u[m]);
  
  gsl_matrix_complex_view A12(gsl_matrix_complex_submatrix(_scratchMatrix,  /* k1= */ 0, /* k2= */ _sampleN,
							   /* n1= */ _sampleN, /* n2= */ _sampleN));
  _negative(&A12.matrix, _K_k[m]);

  gsl_matrix_complex_view A22(gsl_matrix_complex_submatrix(_scratchMatrix,  /* k1= */ _sampleN, /* k2= */ _sampleN,
							   /* n1= */ _sampleN, /* n2= */ _sampleN));
  gsl_matrix_complex_memcpy(&A22.matrix, _K_k[m]);
  
  gsl_vector_complex_view A32(gsl_matrix_complex_subrow(_scratchMatrix, /* rowX= */ 2 * _sampleN, /* offsetX= */ _sampleN, /* columnsN= */ _sampleN));
  gsl_vector_complex_memcpy(&A32.vector, _informationState[m]);

  /*
  if (m == 20) {
    printf("Temporal update prearray:\n");
    _printMatrix(_scratchMatrix);
    printf("\n");
  }
  */

  // zero out A12
  for (unsigned colX = 0; colX < _sampleN; colX++) {
    for (unsigned rowX = colX; rowX < _sampleN; rowX++) {    
      gsl_complex c, s;
      gsl_complex v1 = gsl_matrix_complex_get(&A11.matrix, rowX, rowX);
      gsl_complex v2 = gsl_matrix_complex_get(&A12.matrix, rowX, colX);
      gsl_matrix_complex_set(&A11.matrix, rowX, rowX, _calcGivensRotation(v1, v2, c, s));
      gsl_matrix_complex_set(&A12.matrix, rowX, colX, ComplexZero);

      for (unsigned n = rowX + 1; n <= 2 * _sampleN; n++) {
	gsl_complex v1p, v2p;
	v1 = gsl_matrix_complex_get(_scratchMatrix, n, rowX);
	v2 = gsl_matrix_complex_get(_scratchMatrix, n, colX + _sampleN);

	_applyGivensRotation(v1, v2, c, s, v1p, v2p);
	gsl_matrix_complex_set(_scratchMatrix, n, rowX, v1p);
	gsl_matrix_complex_set(_scratchMatrix, n, colX + _sampleN, v2p);
      }
    }
  }

  /*
  if (m == 20) {
    printf("Temporal update after annihilating A12:\n");
    _printMatrix(_scratchMatrix);
    printf("\n");
  }
  */

  // lower triangularize A22
  for (unsigned rowX = 0; rowX < _sampleN - 1; rowX++) {
    for (unsigned colX = _sampleN - 1; colX > rowX; colX--) {
      gsl_complex c, s;
      gsl_complex v1 = gsl_matrix_complex_get(&A22.matrix, rowX, rowX);
      gsl_complex v2 = gsl_matrix_complex_get(&A22.matrix, rowX, colX);
      gsl_matrix_complex_set(&A22.matrix, rowX, rowX, _calcGivensRotation(v1, v2, c, s));
      gsl_matrix_complex_set(&A22.matrix, rowX, colX, ComplexZero);

      for (unsigned n = rowX + 1; n <= _sampleN; n++) {
	gsl_complex v1p, v2p;
	v1 = gsl_matrix_complex_get(_scratchMatrix, _sampleN + n, _sampleN + rowX);
	v2 = gsl_matrix_complex_get(_scratchMatrix, _sampleN + n, _sampleN + colX);

	_applyGivensRotation(v1, v2, c, s, v1p, v2p);
	gsl_matrix_complex_set(_scratchMatrix, _sampleN + n, _sampleN + rowX, v1p);
	gsl_matrix_complex_set(_scratchMatrix, _sampleN + n, _sampleN + colX, v2p);
      }

      /*
      if (m == 20) {
	printf("After annihilating (%d, %d) of A22:\n", rowX, colX);
	_printMatrix(&A22.matrix);
	printf("\n");
	_printVector(&A32.vector);
	printf("\n");
      }
      */
    }
  }

  /*
  if (m == 20) {
    printf("Temporal update postarray:\n");
    _printMatrix(_scratchMatrix);
    printf("\n");
  }
  */

  // copy out inverse Cholesky factor and information state vector
  gsl_matrix_complex_memcpy(_K_k[m], &A22.matrix);
  gsl_vector_complex_memcpy(_informationState[m], &A32.vector);
}

void SquareRootInformationFilterEchoCancellationFeature::_observationalUpdate(unsigned m, const gsl_complex& Ak, double sigma2_v)
{
  // copy in elements of the pre-array
  gsl_matrix_complex_view A11(gsl_matrix_complex_submatrix(_scratchMatrix2,  /* k1= */ 0, /* k2= */ 0,
							   /* n1= */ _sampleN, /* n2= */ _sampleN));
  gsl_matrix_complex_memcpy(&A11.matrix, _K_k[m]);

  gsl_vector_complex_view a12(gsl_matrix_complex_subcolumn(_scratchMatrix2, /* colX= */ _sampleN, /* offsetX= */ 0, /* columnsN= */ _sampleN));
  _conjugate(&a12.vector, _buffer.getSamples(m));
  double scale = 1.0 / sqrt(sigma2_v);
  for (unsigned n = 0; n < _sampleN; n++) {
    gsl_complex value = gsl_complex_mul_real(gsl_vector_complex_get(&a12.vector, n), scale);
    gsl_vector_complex_set(&a12.vector, n, value);
  }

  /*
  if (m == 20) {
    printf("a12=\n");
    _printVector(&a12.vector);
    printf("\n");
  }
  */

  gsl_vector_complex_view a21(gsl_matrix_complex_subrow(_scratchMatrix2, /* rowX= */ _sampleN, /* offsetX= */ 0, /* columnsN= */ _sampleN));
  gsl_vector_complex_memcpy(&a21.vector, _informationState[m]);

  gsl_complex Akstar(gsl_complex_mul_real(gsl_complex_conjugate(Ak), scale));
  gsl_matrix_complex_set(_scratchMatrix2, _sampleN, _sampleN, Akstar);

  /*
  if (m == 20) {
    printf("Observational update prearray:\n");
    _printMatrix(_scratchMatrix2);
    printf("\n");
  }
  */

  // zero out a12
  for (unsigned rowX = 0; rowX < _sampleN; rowX++) {
    gsl_complex c, s;
    gsl_complex v1 = gsl_matrix_complex_get(&A11.matrix, rowX, rowX);
    gsl_complex v2 = gsl_vector_complex_get(&a12.vector, rowX);
    gsl_matrix_complex_set(&A11.matrix, rowX, rowX, _calcGivensRotation(v1, v2, c, s));
    gsl_vector_complex_set(&a12.vector, rowX, ComplexZero);

    for (unsigned n = rowX + 1; n <= _sampleN; n++) {
      gsl_complex v1p, v2p;
      v1 = gsl_matrix_complex_get(_scratchMatrix2, n, rowX);
      v2 = gsl_matrix_complex_get(_scratchMatrix2, n, _sampleN);

      _applyGivensRotation(v1, v2, c, s, v1p, v2p);
      gsl_matrix_complex_set(_scratchMatrix2, n, rowX, v1p);
      gsl_matrix_complex_set(_scratchMatrix2, n, _sampleN, v2p);
    }
  }

  /*
  if (m == 20) {
    printf("Observational update postarray:\n");
    _printMatrix(_scratchMatrix2);
    printf("\n");
  }
  */

  // copy out inverse Cholesky factor and information state vector
  gsl_vector_complex_memcpy(_informationState[m], &a21.vector);
  gsl_matrix_complex_memcpy(_K_k[m], &A11.matrix);
}

void SquareRootInformationFilterEchoCancellationFeature::_diagonalLoading(unsigned m)
{
  gsl_matrix_complex* A = _K_k[m];
  for (unsigned diagX = 0; diagX < _sampleN; diagX++) {
    gsl_vector_complex_set_zero(_scratch);
    gsl_vector_complex_set(_scratch, diagX, _load);

    for (unsigned colX = diagX; colX < _sampleN; colX++) {
      gsl_complex c, s;
      gsl_complex v1 = gsl_matrix_complex_get(A, colX, colX);
      gsl_complex v2 = gsl_vector_complex_get(_scratch, colX);
      gsl_matrix_complex_set(A, colX, colX, _calcGivensRotation(v1, v2, c, s));
      gsl_vector_complex_set(_scratch, colX, _ComplexZero);

      for (unsigned rowX = colX + 1; rowX < _sampleN; rowX++) {
	gsl_complex v1p, v2p;
	v1 = gsl_matrix_complex_get(A, rowX, colX);
	v2 = gsl_vector_complex_get(_scratch, rowX);

	_applyGivensRotation(v1, v2, c, s, v1p, v2p);
	gsl_matrix_complex_set(A, rowX, colX, v1p);
	gsl_vector_complex_set(_scratch, rowX, v2p);
      }
    }
  }
}


// ----- methods for class `DTDBlockKalmanFilterEchoCancellationFeature' -----
//
DTDBlockKalmanFilterEchoCancellationFeature::
DTDBlockKalmanFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
					    unsigned sampleN, double beta, double sigmau2, double sigmak2, double snrTh, double engTh, double smooth, double amp4play, const String& nm)
  : BlockKalmanFilterEchoCancellationFeature(played, recorded, sampleN, beta, sigmau2, sigmak2, snrTh, amp4play, nm),
    _smoothSk(smooth), _smoothEk(smooth), _engTh(engTh), _snr(0), _EkEnergy(0), _SkEnergy(0), _fdb(NULL)
{
  // if debug open 
  _fdb = fopen("/home/wei/src/wav/debug.txt", "w");
}


DTDBlockKalmanFilterEchoCancellationFeature::~DTDBlockKalmanFilterEchoCancellationFeature()
{
  if (_fdb != NULL)
    fclose(_fdb);
  printf("finished.\n");
}


double DTDBlockKalmanFilterEchoCancellationFeature::_updateBand(const gsl_complex Ak, const gsl_complex Ek, int frameX)
{
  double smthEk, smthSk;
  // if it is the first 100 frames
  if (frameX < 100) {
    smthEk = 1.0 - (double) frameX * (1.0 - _smoothEk) / 100.0;
    smthSk = 1.0 - (double) frameX * (1.0 - _smoothSk) / 100.0;
  } else {
    smthEk = _smoothEk;
    smthSk = _smoothSk;
  }

  double sf;
  const gsl_complex Sk = gsl_complex_sub(Ak, Ek);
  double currEkEng = gsl_complex_abs2(Ek);
  double currSkEng = gsl_complex_abs2(Sk); // Ek to Ak
  _EkEnergy = currEkEng * smthEk + _EkEnergy * (1.0 - smthEk);
  _SkEnergy = currSkEng * smthSk + _SkEnergy * (1.0 - smthSk);
  double currSnr = currSkEng / (currEkEng + 1.0e-15);
  _snr = currSnr * smthEk + _snr * (1.0 - smthEk);
  if (frameX < 100 || (_snr > _threshold && _SkEnergy > _engTh))
    sf = 2.0 / (1.0 + exp(-_snr)) - 1.0;             // snr -> inf, sf -> 1; snr-> 0
  else
    sf = -1.0;

  if (_fdb != NULL) {
    fwrite(&sf, sizeof(double), 1, _fdb);
    fwrite(&_snr, sizeof(double), 1, _fdb);
    fwrite(&_SkEnergy, sizeof(double), 1, _fdb);
    fwrite(&_EkEnergy, sizeof(double), 1, _fdb);
  }
  return sf;
}


void DTDBlockKalmanFilterEchoCancellationFeature::_conjugate(gsl_vector_complex* dest, const gsl_vector_complex* src) const
{
  if (src->size != dest->size)
    throw jdimension_error("BlockKalmanFilterEchoCancellationFeature::_conjugate:: Vector sizes (%d vs. %d) do not match\n", src->size, dest->size);
  for (unsigned n = 0; n < dest->size; n++)
    gsl_vector_complex_set(dest, n, gsl_complex_conjugate((gsl_vector_complex_get(src, n))));
}


const gsl_vector_complex* DTDBlockKalmanFilterEchoCancellationFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n",
                       name().c_str(), frameX - 1, _frameX);

  const gsl_vector_complex* playBlock	= _played->next(_frameX + 1);
  const gsl_vector_complex* recordBlock	= _recorded->next(_frameX + 1);
  _buffer.nextSample(playBlock,_amp4play);

  // Ek is stored in the _vector
  for (unsigned m = 0; m <= _fftLen2; m++) {
    gsl_complex Ak = gsl_vector_complex_get(recordBlock, m);
    gsl_vector_complex *Rk = _filterCoefficient[m];
    const gsl_vector_complex *Vk = _buffer.getSamples(m);

    // Calculate the residual signal; i.e., the desired speech, which is also the innovation
    gsl_complex iprod;
    gsl_blas_zdotu(Rk, Vk, &iprod);
    gsl_complex Ek = gsl_complex_sub(Ak, iprod);
    gsl_vector_complex_set(_vector, m, Ek);
    if (m > 0 && m < _fftLen2)
      gsl_vector_complex_set(_vector, _fftLen - m, gsl_complex_conjugate(Ek));
  }

  for (unsigned m = 0; m <= _fftLen2; m++) {
    gsl_complex Ak = gsl_vector_complex_get(recordBlock, m);
    gsl_vector_complex *Rk = _filterCoefficient[m];
    const gsl_vector_complex *Vk = _buffer.getSamples(m);

    // Calculate the residual signal; i.e., the desired speech, which is also the innovation
    gsl_complex Ek = gsl_vector_complex_get(_vector, m);
    gsl_complex iprod;
    gsl_blas_zdotu(Rk, Vk, &iprod);

    double sf = _updateBand(Ak, Ek, frameX);

    if (sf < 0.0) continue;

    // Estimate the observation noise variance
    gsl_complex zsf	= gsl_complex_rect(sf, 0.0);
    double Ek2		= gsl_complex_abs2(Ek);
    double sigma2_v	= _beta * gsl_vector_get(_sigma2_v, m) + (1.0 - _beta) * Ek2;
    gsl_vector_set(_sigma2_v, m, sigma2_v);

    // Calculate the Kalman gain
    gsl_matrix_complex_memcpy(_K_k_k1, _Sigma2_u[m]);
    gsl_matrix_complex_scale(_K_k_k1, zsf);
    gsl_matrix_complex_add(_K_k_k1, _K_k[m]);

    _conjugate(_scratch2, Vk);
    gsl_blas_zgemv(CblasNoTrans, _ComplexOne, _K_k_k1, _scratch2, _ComplexZero, _scratch);
    gsl_blas_zdotu(Vk, _scratch, &iprod);

    double sigma2_s = GSL_REAL(iprod) + sigma2_v;
    gsl_vector_complex_set_zero(_Gk);
    gsl_blas_zaxpy(gsl_complex_rect(1.0 / sigma2_s, 0.0), _scratch, _Gk);

    // Update the filter weights
    gsl_blas_zaxpy(Ek, _Gk, Rk);

    // Store the state estimation error variance for next the iteration
    gsl_matrix_complex_set_zero(_scratchMatrix);
    for (unsigned rowX = 0; rowX < _sampleN; rowX++) {
      for (unsigned colX = 0; colX < _sampleN; colX++) {
	gsl_complex diagonal = ((rowX == colX) ? _ComplexOne : _ComplexZero);
	gsl_complex value    =  gsl_complex_sub(diagonal, gsl_complex_mul(gsl_vector_complex_get(_Gk, rowX), gsl_vector_complex_get(Vk, colX)));
	gsl_matrix_complex_set(_scratchMatrix, rowX, colX, value);
      }
    }
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, _ComplexOne, _scratchMatrix, _K_k_k1, _ComplexZero, _K_k[m]);
  }

  _increment();
  return _vector;
}
