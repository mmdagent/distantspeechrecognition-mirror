//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.dereverberation
//  Purpose: Single- and multi-channel dereverberation base on linear
//	     prediction in the subband domain.
//  Author:  John McDonough

#include <math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>

#include "common/jpython_error.h"
#include "dereverberation/dereverberation.h"

#ifdef HAVE_CONFIG_H
#include <btk.h>
#endif
#ifdef HAVE_LIBFFTW3
#include <fftw3.h>
#endif


// ----- methods for class `SingleChannelWPEDereverberationFeature' -----
//
SingleChannelWPEDereverberationFeature::
SingleChannelWPEDereverberationFeature(VectorComplexFeatureStreamPtr& samples, unsigned lowerN, unsigned upperN, unsigned iterationsN, double loadDb, double bandWidth, double sampleRate, const String& nm)
  : VectorComplexFeatureStream(samples->size(), nm), _samples(samples),
    _lowerN(lowerN), _upperN(upperN), _predictionN(_upperN - _lowerN + 1), _iterationsN(iterationsN), _firstFrame(true), _framesN(0), _loadFactor(pow(10.0, loadDb / 10.0)),
    _lowerBandWidthN(_setBandWidthN(bandWidth, sampleRate)), _upperBandWidthN(size() - _lowerBandWidthN),
    _thetan(NULL), _gn(new gsl_vector_complex*[size()]), _R(gsl_matrix_complex_alloc(_predictionN, _predictionN)), _r(gsl_vector_complex_alloc(_predictionN)),
    _lagSamples(gsl_vector_complex_alloc(_predictionN))
{
  printf("Single-Channel WPE Feature Input Size  = %d\n", _samples->size());
  printf("Single-Channel WPE Feature Output Size = %d\n", size());

  // allocate prediction vectors
  for (unsigned n = 0; n < size(); n++)
    _gn[n] = gsl_vector_complex_calloc(_predictionN);
}

SingleChannelWPEDereverberationFeature::~SingleChannelWPEDereverberationFeature()
{
  if (_thetan != NULL) gsl_matrix_free(_thetan);
  
  for (unsigned n = 0; n < size(); n++)
    gsl_vector_complex_free(_gn[n]);
  delete[] _gn;

  gsl_matrix_complex_free(_R);
  gsl_vector_complex_free(_r);

  for (_SamplesIterator itr = _yn.begin(); itr != _yn.end(); itr++)
    gsl_vector_complex_free(*itr);
  _yn.clear();
}

const gsl_vector_complex* SingleChannelWPEDereverberationFeature::_getLags(unsigned subbandX, unsigned sampleX)
{
  static const gsl_complex _Zero = gsl_complex_rect(0.0, 0.0);

  for (unsigned lagX = 0; lagX < _predictionN; lagX++) {
    int index = sampleX;  index -= lagX;
    gsl_complex val = (index < 0) ? _Zero : gsl_vector_complex_get(_yn[index], subbandX);
    gsl_vector_complex_set(_lagSamples, lagX, val);
  }

  return _lagSamples;
}

void SingleChannelWPEDereverberationFeature::_fillBuffer()
{
  _framesN = 0;
  while (true) {
    const gsl_vector_complex* block;
    try {
      block = _samples->next();
    } catch (jiterator_error& e) {
      break;
    }
    gsl_vector_complex* sample = gsl_vector_complex_alloc(size());
    gsl_vector_complex_memcpy(sample, block);
    _yn.push_back(sample);
    _framesN++;    
  }
  _thetan  = gsl_matrix_alloc(_framesN, size());
  gsl_matrix_set_zero(_thetan);
}

void SingleChannelWPEDereverberationFeature::_calculateRr(unsigned subbandX)
{
  gsl_matrix_complex_set_zero(_R);
  gsl_vector_complex_set_zero(_r);

  // calculate _R
  for (unsigned sampleX = _lowerN; sampleX < _framesN; sampleX++) {
    double thetan = gsl_matrix_get(_thetan, sampleX, subbandX);
    const gsl_vector_complex* lag = _getLags(subbandX, sampleX - _lowerN);
    for (unsigned rowX = 0; rowX < _predictionN; rowX++) {
      gsl_complex rowS = gsl_vector_complex_get(lag, rowX);
      for (unsigned colX = 0; colX <= rowX; colX++) {
	gsl_complex colS = gsl_vector_complex_get(lag, colX);
	gsl_complex val = gsl_matrix_complex_get(_R, rowX, colX);
	val = gsl_complex_add(val, gsl_complex_div_real(gsl_complex_mul(rowS, gsl_complex_conjugate(colS)), thetan));
	gsl_matrix_complex_set(_R, rowX, colX, val);
      }
    }
  }

  // calculate _r
  unsigned sampleX = 0;
  double optimization = 0.0;
  for (_SamplesIterator itr = _yn.begin(); itr != _yn.end(); itr++) {
    if (sampleX < _lowerN) { sampleX++; continue; }
    double thetan = gsl_matrix_get(_thetan, sampleX, subbandX);
    gsl_complex current = gsl_vector_complex_get(*itr, subbandX);
    const gsl_vector_complex* lags = _getLags(subbandX, sampleX - _lowerN);

    gsl_complex dereverb;
    gsl_blas_zdotc(_gn[subbandX], lags, &dereverb);
    gsl_complex diff = gsl_complex_sub(current, dereverb);
    double dist = gsl_complex_abs(diff);
    optimization += dist * dist / thetan + log(thetan);

    /*
    if (subbandX == 100) {
      printf("Sample %d : Current (%0.4e, %0.4e) : Diff (%0.4e, %0.4e)\n",
	     sampleX, GSL_REAL(current), GSL_IMAG(current), GSL_REAL(diff), GSL_IMAG(diff));
    }
    */

    for (unsigned lagX = 0; lagX < _predictionN; lagX++) {
      gsl_complex val = gsl_vector_complex_get(_r, lagX);
      val = gsl_complex_add(val, gsl_complex_div_real(gsl_complex_mul(gsl_complex_conjugate(current), gsl_vector_complex_get(lags, lagX)), thetan));
      gsl_vector_complex_set(_r, lagX, val);
    }
    sampleX++;
  }

  if (subbandX == 100) {
    // gsl_vector_complex_fprintf(stdout, _gn[subbandX], "%0.2f");
    printf("Subband %4d : Criterion Value %10.4e\n", subbandX, optimization);
    // printf("\n");
  }
}

const double SingleChannelWPEDereverberationFeature::_SubbandFloor = 1.0E-03;

void SingleChannelWPEDereverberationFeature::_calculateThetan()
{
  unsigned sampleX = 0;
  for (_SamplesIterator itr = _yn.begin(); itr != _yn.end(); itr++) {
    for (unsigned subbandX = 0; subbandX < size(); subbandX++) {
      gsl_complex current = gsl_vector_complex_get(*itr, subbandX);

      if (sampleX >= _lowerN) {
	gsl_complex dereverb;
	const gsl_vector_complex* lags = _getLags(subbandX, sampleX - _lowerN);
	gsl_blas_zdotc(_gn[subbandX], lags, &dereverb);
	current = gsl_complex_sub(current, dereverb);
      }

      double thetan = gsl_complex_abs(current);
      if (thetan < _SubbandFloor) {
	// printf("Sample %d Subband %d theta_n = %0.2f\n", sampleX, subbandX, thetan);
	thetan = _SubbandFloor;
      }

      gsl_matrix_set(_thetan, sampleX, subbandX, thetan * thetan);
    }
    sampleX++;
  }
}

void SingleChannelWPEDereverberationFeature::_loadR()
{
  double maximumDiagonal = 0.0;
  for (unsigned componentX = 0; componentX < _predictionN; componentX++) {
    double diag = gsl_complex_abs(gsl_matrix_complex_get(_R, componentX, componentX));
    if (diag > maximumDiagonal) maximumDiagonal = diag;
  }

  for (unsigned componentX = 0; componentX < _predictionN; componentX++) {
    double diag = gsl_complex_abs(gsl_matrix_complex_get(_R, componentX, componentX)) + maximumDiagonal * _loadFactor;
    gsl_matrix_complex_set(_R, componentX, componentX, gsl_complex_rect(diag, 0.0));
  }
}

void SingleChannelWPEDereverberationFeature::_estimateGn()
{
  for (unsigned iterationX = 0; iterationX < _iterationsN; iterationX++) {
    _calculateThetan();
    for (unsigned subbandX = 0; subbandX < size(); subbandX++) {

      if ((subbandX > _lowerBandWidthN) && (subbandX < _upperBandWidthN)) continue;

      _calculateRr(subbandX);
      _loadR();
      gsl_linalg_complex_cholesky_decomp(_R);
      gsl_linalg_complex_cholesky_solve(_R, _r, _gn[subbandX]);

      if (subbandX == 100) {
	double sum = 0.0;
	for (unsigned componentX = 0; componentX < _predictionN; componentX++) {
	  double gn = gsl_complex_abs(gsl_vector_complex_get(_gn[subbandX], componentX));
	  sum += gn * gn;
	}

	double wng = 10.0 * log10(sum);
	printf("Iteration %d: Subband %4d WNG %6.2f\n", iterationX, subbandX, wng);
      }
    }
  }
}

const gsl_vector_complex* SingleChannelWPEDereverberationFeature::next(int frameX) {
  if (_firstFrame) { _fillBuffer();  _estimateGn();  _firstFrame = false; }

  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);

  _increment();
  if (_frameX == _yn.size())
    throw jiterator_error("end of samples!");

  const gsl_vector_complex* current = _yn[_frameX];

  for (unsigned subbandX = 0; subbandX < size(); subbandX++) {
    gsl_complex cur = gsl_vector_complex_get(current, subbandX);
    if ((_frameX >= _lowerN) && ((subbandX <= _lowerBandWidthN) || (subbandX >= _upperBandWidthN))) {
      gsl_complex dereverb;
      const gsl_vector_complex* lags = _getLags(subbandX, _frameX - _lowerN);
      gsl_blas_zdotc(_gn[subbandX], lags, &dereverb);

      cur = gsl_complex_sub(cur, dereverb);
    }
    gsl_vector_complex_set(_vector, subbandX, cur);
  }

  return _vector;
}

unsigned SingleChannelWPEDereverberationFeature::_setBandWidthN(double bandWidth, double sampleRate)
{
  if (bandWidth == 0.0) return (size() / 2);

  if (bandWidth > (sampleRate / 2.0))
    throw jdimension_error("Bandwidth is greater than the Nyquist rate.\n", bandWidth, (sampleRate / 2.0));

  return unsigned((bandWidth / (sampleRate / 2.0)) * (size() / 2));
}

void SingleChannelWPEDereverberationFeature::reset()
{
  _samples->reset();  VectorComplexFeatureStream::reset();  _firstFrame = true;  _framesN = 0;

  if (_thetan != NULL) { gsl_matrix_free(_thetan);  _thetan = NULL; }

  for (_SamplesIterator itr = _yn.begin(); itr != _yn.end(); itr++)
    gsl_vector_complex_free(*itr);
  _yn.clear();
}

void SingleChannelWPEDereverberationFeature::nextSpeaker()
{
  printf("Resetting 'SingleChannelWPEDereverberationFeature' for next speaker.\n");

  reset();
  for (unsigned n = 0; n < size(); n++)
    gsl_vector_complex_set_zero(_gn[n]);
}


// ----- methods for class `MultiChannelWPEDereverberation' -----
//
MultiChannelWPEDereverberation::MultiChannelWPEDereverberation(unsigned subbandsN, unsigned channelsN, unsigned lowerN, unsigned upperN, unsigned iterationsN, double loadDb, double bandWidth,  double sampleRate)
  : _sources(0), _subbandsN(subbandsN), _channelsN(channelsN),
    _lowerN(lowerN), _upperN(upperN), _predictionN(_upperN - _lowerN + 1), _iterationsN(iterationsN), _totalPredictionN(_predictionN * _channelsN),
    _firstFrame(true), _framesN(0), _loadFactor(pow(10.0, loadDb / 10.0)),
    _lowerBandWidthN(_setBandWidthN(bandWidth, sampleRate)), _upperBandWidthN(size() - _lowerBandWidthN),
    _thetan(new gsl_matrix*[_channelsN]), _Gn(new gsl_vector_complex**[channelsN]),
    _R(new gsl_matrix_complex*[_channelsN]), _r(new gsl_vector_complex*[_channelsN]), _lagSamples(gsl_vector_complex_alloc(_totalPredictionN)),
    _output(new gsl_vector_complex*[channelsN]), FrameResetX(-1), _frameX(FrameResetX)
{
  cout << "MultiChannelWPEDereverberation Subbands " << _subbandsN << endl;
  cout << "MultiChannelWPEDereverberation Channels " << _channelsN << endl;

  for (unsigned channelX = 0; channelX < _channelsN; channelX++) {
    _thetan[channelX] = NULL;
    _R[channelX]  = gsl_matrix_complex_alloc(_totalPredictionN, _totalPredictionN);
    _r[channelX]  = gsl_vector_complex_alloc(_totalPredictionN);
    _Gn[channelX] = new gsl_vector_complex*[_subbandsN];

    for (unsigned subbandX = 0; subbandX < _subbandsN; subbandX++)
      _Gn[channelX][subbandX] = gsl_vector_complex_alloc(_totalPredictionN);

    _output[channelX] = gsl_vector_complex_alloc(_subbandsN);
  }
}

MultiChannelWPEDereverberation::~MultiChannelWPEDereverberation()
{
  for (unsigned channelX = 0; channelX < _channelsN; channelX++) {
    if (_thetan[channelX] != NULL)
      gsl_matrix_free(_thetan[channelX]);

    for (unsigned subbandX = 0; subbandX < _subbandsN; subbandX++) {
      gsl_vector_complex_free(_Gn[channelX][subbandX]);
    }

    gsl_matrix_complex_free(_R[channelX]);
    gsl_vector_complex_free(_r[channelX]);
    delete[] _Gn[channelX];
    gsl_vector_complex_free(_output[channelX]);
  }

  delete[] _thetan;
  delete[] _Gn;
  delete[] _R;
  delete[] _r;
  delete[] _output;

  gsl_vector_complex_free(_lagSamples);
}

unsigned MultiChannelWPEDereverberation::_setBandWidthN(double bandWidth, double sampleRate)
{
  if (bandWidth == 0.0) return (size() / 2);

  if (bandWidth > (sampleRate / 2.0))
    throw jdimension_error("Bandwidth is greater than the Nyquist rate.\n", bandWidth, (sampleRate / 2.0));

  return unsigned((bandWidth / (sampleRate / 2.0)) * (size() / 2));
}

void MultiChannelWPEDereverberation::reset()
{
  _firstFrame = true;  _framesN = 0;  _frameX = FrameResetX;

  for (_SourceListIterator itr = _sources.begin(); itr != _sources.end(); itr++)
    (*itr)->reset();

  for (_FrameBraceListIterator itr = _frames.begin(); itr != _frames.end(); itr++) {
    _FrameBrace& fbrace(*itr);
    for (_FrameBraceIterator fitr = fbrace.begin(); fitr != fbrace.end(); fitr++) {
      gsl_vector_complex_free(*fitr);
    }
  }
  _frames.clear();
}

void MultiChannelWPEDereverberation::setInput(VectorComplexFeatureStreamPtr& samples)
{
  if (_sources.size() == _channelsN)
    throw jallocation_error("Channel capacity exceeded.");

  _sources.push_back(samples);
}

const gsl_vector_complex* MultiChannelWPEDereverberation::getOutput(unsigned channelX, int frameX)
{
  if (_firstFrame) { _fillBuffer();  _estimateGn();  _firstFrame = false; }

  if (frameX == _frameX) return _output[channelX];

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem in 'MultiChannelWPEDereverberation'\n");

  _increment();
  if (_frameX == _framesN)
    throw jiterator_error("end of samples!");

  // generate dereverberated output for *all* channels
  for (unsigned chanX = 0; chanX < _channelsN; chanX++ ) {
    const gsl_vector_complex* current = _frames[_frameX][chanX];

    for (unsigned subbandX = 0; subbandX < size(); subbandX++) {
      gsl_complex cur = gsl_vector_complex_get(current, subbandX);
      if ((_frameX >= _lowerN) && ((subbandX <= _lowerBandWidthN) || (subbandX >= _upperBandWidthN))) {
	gsl_complex dereverb;
	const gsl_vector_complex* lags = _getLags(subbandX, _frameX - _lowerN);
	gsl_blas_zdotc(_Gn[channelX][subbandX], lags, &dereverb);

	cur = gsl_complex_sub(cur, dereverb);
      }
      gsl_vector_complex_set(_output[chanX], subbandX, cur);
    }
  }
  return _output[channelX];
}

void MultiChannelWPEDereverberation::_fillBuffer()
{
  while (true) {
    _FrameBrace fbrace(_channelsN);
    try {
      for (unsigned channelX = 0; channelX < _channelsN; channelX++) {
	VectorComplexFeatureStreamPtr src(_sources[channelX]);
	const gsl_vector_complex* block = src->next();
	gsl_vector_complex* sample = gsl_vector_complex_alloc(size());
	gsl_vector_complex_memcpy(sample, block);
	fbrace[channelX] = sample;
      }
    } catch (jiterator_error& e) {
      break;
    }
    _frames.push_back(fbrace);
  }
  _framesN = _frames.size();
  for (unsigned channelX = 0; channelX < _channelsN; channelX++) {
    if (_thetan[channelX] != NULL) gsl_matrix_free(_thetan[channelX]);
    _thetan[channelX] = gsl_matrix_alloc(_framesN, size());
    gsl_matrix_set_zero(_thetan[channelX]);
  }
}

const gsl_vector_complex* MultiChannelWPEDereverberation::_getLags(unsigned subbandX, unsigned sampleX)
{
  static const gsl_complex _Zero = gsl_complex_rect(0.0, 0.0);

  unsigned totalX = 0;
  for (unsigned channelX = 0; channelX < _channelsN; channelX++) {
    for (unsigned lagX = 0; lagX < _predictionN; lagX++) {
      int index = sampleX;  index -= lagX;
      gsl_complex val = (index < 0) ? _Zero : gsl_vector_complex_get(_frames[index][channelX], subbandX);
      gsl_vector_complex_set(_lagSamples, totalX, val);
      totalX++;
    }
  }

  return _lagSamples;
}

void MultiChannelWPEDereverberation::_calculateRr(unsigned subbandX)
{
  // calculate (lower triangle of) _R for all channels
  for (unsigned channelX = 0; channelX < _channelsN; channelX++) {
    gsl_matrix_complex* R = _R[channelX];
    gsl_matrix_complex_set_zero(R);
    for (unsigned sampleX = _lowerN; sampleX < _framesN; sampleX++) {
      double thetan = gsl_matrix_get(_thetan[channelX], sampleX, subbandX);
      const gsl_vector_complex* lags = _getLags(subbandX, sampleX - _lowerN);
      for (unsigned rowX = 0; rowX < _totalPredictionN; rowX++) {
	gsl_complex rowS = gsl_vector_complex_get(lags, rowX);
	for (unsigned colX = 0; colX <= rowX; colX++) {
	  gsl_complex colS = gsl_vector_complex_get(lags, colX);
	  gsl_complex val = gsl_matrix_complex_get(R, rowX, colX);
	  val = gsl_complex_add(val, gsl_complex_div_real(gsl_complex_mul(rowS, gsl_complex_conjugate(colS)), thetan));
	  gsl_matrix_complex_set(R, rowX, colX, val);
	}
      }
    }
  }

  // calculate _r for all channels
  for (unsigned channelX = 0; channelX < _channelsN; channelX++) {
    gsl_vector_complex* r = _r[channelX];
    gsl_vector_complex_set_zero(r);

    unsigned sampleX = 0;
    double optimization = 0.0;
    for (_FrameBraceListIterator itr = _frames.begin(); itr != _frames.end(); itr++) {
      if (sampleX < _lowerN) { sampleX++; continue; }

      _FrameBrace& frame(*itr);
      double thetan = gsl_matrix_get(_thetan[channelX], sampleX, subbandX);
      gsl_complex current = gsl_vector_complex_get(frame[channelX], subbandX);
      const gsl_vector_complex* lags = _getLags(subbandX, sampleX - _lowerN);

      gsl_complex dereverb;
      gsl_blas_zdotc(_Gn[channelX][subbandX], lags, &dereverb);
      gsl_complex diff = gsl_complex_sub(current, dereverb);
      double dist = gsl_complex_abs(diff);
      optimization += dist * dist / thetan + log(thetan);

      for (unsigned lagX = 0; lagX < _totalPredictionN; lagX++) {
	gsl_complex val = gsl_vector_complex_get(r, lagX);
	val = gsl_complex_add(val, gsl_complex_div_real(gsl_complex_mul(gsl_complex_conjugate(current), gsl_vector_complex_get(lags, lagX)), thetan));
	gsl_vector_complex_set(r, lagX, val);
      }
      sampleX++;
    }

    if (subbandX == 100) {
      // gsl_vector_complex_fprintf(stdout, _Gn[channelX][subbandX], "%0.2f");
      printf("Channel %d : Subband %4d : Criterion Value %10.4e\n", channelX, subbandX, optimization);
      // printf("\n");
    }
  }
}

const double MultiChannelWPEDereverberation::_SubbandFloor = 1.0E-03;

void MultiChannelWPEDereverberation::_calculateThetan()
{
  unsigned sampleX = 0;
  for (_FrameBraceListIterator itr = _frames.begin(); itr != _frames.end(); itr++) {
    const _FrameBrace& brace(*itr);
    for (unsigned channelX = 0; channelX < _channelsN; channelX++) {
      const gsl_vector_complex* observation = brace[channelX];
      for (unsigned subbandX = 0; subbandX < size(); subbandX++) {
	gsl_complex current = gsl_vector_complex_get(observation, subbandX);

	if (sampleX >= _lowerN) {
	  gsl_complex dereverb;
	  const gsl_vector_complex* lags = _getLags(subbandX, sampleX - _lowerN);
	  gsl_blas_zdotc(_Gn[channelX][subbandX], lags, &dereverb);
	  current = gsl_complex_sub(current, dereverb);
	}

	double thetan = gsl_complex_abs(current);
	if (thetan < _SubbandFloor) {
	  // printf("Sample %d Subband %d theta_n = %0.2f\n", sampleX, subbandX, thetan);
	  thetan = _SubbandFloor;
	}

	gsl_matrix_set(_thetan[channelX], sampleX, subbandX, thetan * thetan);
      }
    }
    sampleX++;
  }
}

void MultiChannelWPEDereverberation::_loadR()
{
  for (unsigned channelX = 0; channelX < _channelsN; channelX++) {
    gsl_matrix_complex* R = _R[channelX];
    double maximumDiagonal = 0.0;
    for (unsigned componentX = 0; componentX < _totalPredictionN; componentX++) {
      double diag = gsl_complex_abs(gsl_matrix_complex_get(R, componentX, componentX));
      if (diag > maximumDiagonal) maximumDiagonal = diag;
    }

    for (unsigned componentX = 0; componentX < _totalPredictionN; componentX++) {
      double diag = gsl_complex_abs(gsl_matrix_complex_get(R, componentX, componentX)) + maximumDiagonal * _loadFactor;
      gsl_matrix_complex_set(R, componentX, componentX, gsl_complex_rect(diag, 0.0));
    }
  }
}

void MultiChannelWPEDereverberation::_estimateGn()
{
  for (unsigned iterationX = 0; iterationX < _iterationsN; iterationX++) {
    _calculateThetan();
    for (unsigned subbandX = 0; subbandX < size(); subbandX++) {

      if ((subbandX > _lowerBandWidthN) && (subbandX < _upperBandWidthN)) continue;

      _calculateRr(subbandX);
      _loadR();
      for (unsigned channelX = 0; channelX < _channelsN; channelX++) {
	gsl_linalg_complex_cholesky_decomp(_R[channelX]);
	gsl_linalg_complex_cholesky_solve(_R[channelX], _r[channelX], _Gn[channelX][subbandX]);

	if (subbandX == 100) {
	  double sum = 0.0;
	  for (unsigned componentX = 0; componentX < _predictionN; componentX++) {
	    double gn = gsl_complex_abs(gsl_vector_complex_get(_Gn[channelX][subbandX], componentX));
	    sum += gn * gn;
	  }

	  double wng = 10.0 * log10(sum);
	  printf("Channel %d: Iteration %d Subband %4d WNG %6.2f\n", channelX, iterationX, subbandX, wng);
	}
      }
    }
  }
}

void MultiChannelWPEDereverberation::nextSpeaker()
{
  printf("Resetting 'MultiChannelWPEDereverberation' for next speaker.\n");

  reset();
  for (unsigned channelX = 0; channelX < _channelsN; channelX++)
    for (unsigned n = 0; n < size(); n++)
      gsl_vector_complex_set_zero(_Gn[channelX][n]);
}


// ----- methods for class `MultiChannelWPEDereverberationFeature' -----
//
MultiChannelWPEDereverberationFeature::
MultiChannelWPEDereverberationFeature(MultiChannelWPEDereverberationPtr& source, unsigned channelX, const String& nm)
  : VectorComplexFeatureStream(source->size(), nm), _source(source), _channelX(channelX)
{
  cout << "MultiChannelWPEDereverberationFeature Subbands " << size()   << endl;
  cout << "MultiChannelWPEDereverberationFeature Channel  " << channelX << endl;
}

MultiChannelWPEDereverberationFeature::~MultiChannelWPEDereverberationFeature() { }

const gsl_vector_complex* MultiChannelWPEDereverberationFeature::next(int frameX) {
  return _source->getOutput(_channelX, frameX);
}

void MultiChannelWPEDereverberationFeature::reset() { _source->reset(); }
