//                              -*- C++ -*-
//
//                            Speech Front End
//                                  (btk)
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

#ifndef _cancelVP_h_
#define _cancelVP_h_

#include <stdio.h>
#include <assert.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>
#include "common/jexception.h"
#include <gsl/gsl_eigen.h>

#include "stream/stream.h"
#include "btk.h"
#include "beamformer/tracker.h"

/**
* \defgroup NLMSAcousticEchoCancellationFeature NLMS Echo Cancellation Feature
*/
/*@{*/


// ----- definition for class `NLMSAcousticEchoCancellationFeature' -----
//
class NLMSAcousticEchoCancellationFeature : public VectorComplexFeatureStream {
 public:
   NLMSAcousticEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded, 
				       double delta = 100.0, double epsilon = 1.0E-04, double threshold = 100.0, const String& nm = "AEC");
  virtual ~NLMSAcousticEchoCancellationFeature();

  virtual const gsl_vector_complex* next(int frameX = -5);

  virtual void reset() { _played->reset(); _recorded->reset(); gsl_vector_complex_set_zero(_filterCoefficient); }

private:
  bool _update(const gsl_complex Vk);

  VectorComplexFeatureStreamPtr         	_played;                    // v(n)
  VectorComplexFeatureStreamPtr         	_recorded;                  // a(n)

  unsigned					_fftLen;
  unsigned					_fftLen2;
  gsl_vector_complex*				_filterCoefficient;

  const double					_delta;
  const double					_epsilon;
  const double 					_threshold;
};


typedef Inherit<NLMSAcousticEchoCancellationFeature, VectorComplexFeatureStreamPtr> NLMSAcousticEchoCancellationFeaturePtr;
/*@}*/


/**
* \defgroup KalmanFilterEchoCancellationFeature Kalman Filer Echo Cancellation Feature
*/
/*@{*/


// ----- definition for class `KalmanFilterEchoCancellationFeature' -----
//
class KalmanFilterEchoCancellationFeature : public VectorComplexFeatureStream {
 public:
  KalmanFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
				      double beta = 0.95, double sigma2 = 100.0, double threshold = 100.0, const String& nm = "KFEchoCanceller");
  virtual ~KalmanFilterEchoCancellationFeature();

  virtual const gsl_vector_complex* next(int frameX = -5);

  virtual void reset() { _played->reset(); _recorded->reset(); gsl_vector_complex_set_zero(_filterCoefficient); }

private:
  bool _update(const gsl_complex Vk);

  VectorComplexFeatureStreamPtr         	_played;                    // v(n)
  VectorComplexFeatureStreamPtr         	_recorded;                  // a(n)

  unsigned					_fftLen;
  unsigned					_fftLen2;

  gsl_vector_complex*				_filterCoefficient;
  gsl_vector*					_sigma2_v;
  gsl_vector*					_K_k;

  const double					_beta;
  const double 					_threshold;
  const double 					_sigma2_u;
};


typedef Inherit<KalmanFilterEchoCancellationFeature, VectorComplexFeatureStreamPtr> KalmanFilterEchoCancellationFeaturePtr;
/*@}*/

// ----- definition for class `BlockKalmanFilterEchoCancellationFeature' -----
//
class BlockKalmanFilterEchoCancellationFeature : public VectorComplexFeatureStream {
  public:
    BlockKalmanFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
                                             unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmauk2 = 5.0, double threshold = 100.0,
					     double amp4play = 1.0, 
                                             const String& nm = "KFEchoCanceller");
    virtual ~BlockKalmanFilterEchoCancellationFeature();

    virtual const gsl_vector_complex* next(int frameX = -5);

    virtual void reset()
    { 
      _played->reset(); _recorded->reset();

    /*
      for (unsigned m = 0; m < _fftLen; m++)
      gsl_vector_complex_set_zero(_filterCoefficient[m]);
    */
    }

  protected:
    class _ComplexBuffer {
      public:
    /*
        @brief Construct a circular buffer to hold past and current subband samples
        It keeps nsamp arrays which is completely updated with the period 'nsamp'.
        Each array holds actual values of the samples.
        @param unsigned len[in] is the size of each vector of samples
        @param unsigned nsamp[in] is the period of the circular buffer
    */
      _ComplexBuffer(unsigned len, unsigned sampleN)
        : _len(len), _sampleN(sampleN), _zero(_sampleN - 1),
	  _samples(new gsl_vector_complex*[_sampleN]), _subbandSamples(gsl_vector_complex_calloc(_sampleN))
      {
	for (unsigned i = 0; i < _sampleN; i++)
	  _samples[i] = gsl_vector_complex_calloc(_len);
      }

      ~_ComplexBuffer()
      {
	for (unsigned i = 0; i < _sampleN; i++)
	  gsl_vector_complex_free(_samples[i]);
	delete[] _samples;
	gsl_vector_complex_free(_subbandSamples);
      }

      gsl_complex sample(unsigned timeX, unsigned binX) const {
	unsigned idx = _index(timeX);
	const gsl_vector_complex* vec = _samples[idx];
	return gsl_vector_complex_get(vec, binX);
      }

      const gsl_vector_complex* getSamples(unsigned m)
      {
	for (unsigned timeX = 0; timeX < _sampleN; timeX++)
	  gsl_vector_complex_set(_subbandSamples, timeX, sample(timeX, m));

	return _subbandSamples;
      }

      void nextSample(const gsl_vector_complex* s = NULL, double amp4play = 1.0 ) {
	_zero = (_zero + 1) % _sampleN;

	gsl_vector_complex* nextBlock = _samples[_zero];

	if (s == NULL) {
	  gsl_vector_complex_set_zero(nextBlock);
	} else {
	  if (s->size != _len)
	    throw jdimension_error("'_ComplexBuffer': Sizes do not match (%d vs. %d)", s->size, _len);
	  assert( s->size == _len );
	  gsl_vector_complex_memcpy(nextBlock, s);
	  if( amp4play != 1.0 )
	    gsl_blas_zdscal( amp4play, nextBlock );
	}
      }

      void zero() {
	for (unsigned i = 0; i < _sampleN; i++)
	  gsl_vector_complex_set_zero(_samples[i]);
	_zero = _sampleN - 1;
      }

    private:
      unsigned _index(unsigned idx) const {
	assert(idx < _sampleN);
	unsigned ret = (_zero + _sampleN - idx) % _sampleN;
	return ret;
      }

      const unsigned				_len;
      const unsigned				_sampleN;
      unsigned					_zero;		// index of most recent sample
      gsl_vector_complex**			_samples;
      gsl_vector_complex*			_subbandSamples;
    };

  static gsl_complex				_ComplexOne;
  static gsl_complex				_ComplexZero;

  bool _update(const gsl_vector_complex* Vk);
  void _conjugate(gsl_vector_complex* dest, const gsl_vector_complex* src) const;

  VectorComplexFeatureStreamPtr         	_played;                    // v(n)
  VectorComplexFeatureStreamPtr         	_recorded;                  // a(n)

  unsigned					_fftLen;
  unsigned					_fftLen2;
  unsigned					_sampleN;

  _ComplexBuffer				_buffer;

  gsl_vector_complex**				_filterCoefficient;
  gsl_vector*					_sigma2_v;
  gsl_matrix_complex**				_K_k;
  gsl_matrix_complex*				_K_k_k1;

  const double					_beta;
  const double	 				_threshold;
  gsl_matrix_complex**				_Sigma2_u;
  gsl_vector_complex*				_Gk;
  gsl_vector_complex*				_scratch;
  gsl_vector_complex*				_scratch2;
  gsl_matrix_complex*				_scratchMatrix;
  gsl_matrix_complex*				_scratchMatrix2;
  double                                        _amp4play; 
  double                                        _floorVal;
  int                                           _skippedN;
  int                                           _maxSkippedN;
};


typedef Inherit<BlockKalmanFilterEchoCancellationFeature, VectorComplexFeatureStreamPtr> BlockKalmanFilterEchoCancellationFeaturePtr;
/*@}*/

/**
* \defgroup DTDBlockKalmanFilterEchoCancellationFeature Block Kalman Filer Echo Cancellation Feature
*/
/*@{*/

// ----- definition for class `InformationFilterEchoCancellationFeature' -----
//
class InformationFilterEchoCancellationFeature : public BlockKalmanFilterEchoCancellationFeature {
 public:
  InformationFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
					   unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmauk2 = 5.0, double snrTh = 2.0,
					   double engTh = 100.0, double smooth = 0.9, double loading = 1.0e-02,
					   double amp4play = 1.0, 
					   const String& nm = "Information Echo Canceller");

  virtual ~InformationFilterEchoCancellationFeature();

  virtual const gsl_vector_complex* next(int frameX = -5);

protected:
  static double _EigenValueThreshold;

  double _updateBand(const gsl_complex Ak, const gsl_complex Ek, int frameX, unsigned m);
  void _invert(gsl_matrix_complex* matrix);
  static void _printMatrix(const gsl_matrix_complex* mat);
  static void _printVector(const gsl_vector_complex* vec);

  const double						_smoothEk;                   // for smoothing the error signal, ->1, less smooth
  const double						_smoothSk;                   // for smoothing the estimated signal
  const double          	                        _engTh;                      // threshold of energy

  double*	 					_snr;
  double*        	                                _EkEnergy;
  double*                	                        _SkEnergy;

  gsl_matrix_complex*					_inverse;
  gsl_eigen_hermv_workspace*				_eigenWorkSpace;
  gsl_vector*						_evalues;

  gsl_vector_complex*					_scratchInverse;
  gsl_matrix_complex*					_scratchMatrixInverse;
  gsl_matrix_complex*					_scratchMatrixInverse2;
  gsl_matrix_complex*					_matrixCopy;

  const double						_loading;
};

typedef Inherit<InformationFilterEchoCancellationFeature, BlockKalmanFilterEchoCancellationFeaturePtr> InformationFilterEchoCancellationFeaturePtr;


// ----- definition for class `SquareRootInformationFilterEchoCancellationFeature' -----
//
class SquareRootInformationFilterEchoCancellationFeature : public InformationFilterEchoCancellationFeature {
 public:
  SquareRootInformationFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
						     unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmauk2 = 5.0, double snrTh = 2.0,
					   double engTh = 100.0, double smooth = 0.9, double loading = 1.0e-02,
					   double amp4play = 1.0, 
					   const String& nm = "Square Root Information Echo Canceller");

  virtual ~SquareRootInformationFilterEchoCancellationFeature();

  virtual const gsl_vector_complex* next(int frameX = -5);

private:
  const gsl_complex _load;
  static gsl_complex _calcGivensRotation(const gsl_complex& v1, const gsl_complex& v2, gsl_complex& c, gsl_complex& s);
  static void _applyGivensRotation(const gsl_complex& v1, const gsl_complex& v2, const gsl_complex& c, const gsl_complex& s,
				   gsl_complex& v1p, gsl_complex& v2p);
  void _negative(gsl_matrix_complex* dest, const gsl_matrix_complex* src);
  void _extractCovarianceState(const gsl_matrix_complex* K_k, const gsl_vector_complex* sk, gsl_vector_complex* xk);

  void _temporalUpdate(unsigned m);
  void _observationalUpdate(unsigned m, const gsl_complex& Ak, double sigma2_v);
  void _diagonalLoading(unsigned m);

  gsl_vector_complex**				_informationState;
};

typedef Inherit<SquareRootInformationFilterEchoCancellationFeature, InformationFilterEchoCancellationFeaturePtr> SquareRootInformationFilterEchoCancellationFeaturePtr;


// ----- definition for class `DTDBlockKalmanFilterEchoCancellationFeature' -----
//
class DTDBlockKalmanFilterEchoCancellationFeature : public BlockKalmanFilterEchoCancellationFeature {
 public:
   DTDBlockKalmanFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
					       unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmauk2 = 5.0, double snrTh = 2.0,
					       double engTh = 100.0, double smooth = 0.9,
                                               double amp4play = 1.0, 
					       const String& nm = "DTDKFEchoCanceller");
  virtual ~DTDBlockKalmanFilterEchoCancellationFeature();

  virtual const gsl_vector_complex* next(int frameX = -5);

private:
  double _updateBand(const gsl_complex Ak, const gsl_complex Ek, int frameX);
  void _conjugate(gsl_vector_complex* dest, const gsl_vector_complex* src) const;

  const double					_smoothEk;                   // for smoothing the error signal, ->1, less smooth
  const double					_smoothSk;                   // for smoothing the estimated signal
  const double                                  _engTh;                      // threshold of energy
  double 					_snr;

  double                                        _EkEnergy;
  double                                        _SkEnergy;
  FILE*						_fdb;
};


typedef Inherit<DTDBlockKalmanFilterEchoCancellationFeature, BlockKalmanFilterEchoCancellationFeaturePtr> DTDBlockKalmanFilterEchoCancellationFeaturePtr;
/*@}*/

#endif // _cancelVP_h_
