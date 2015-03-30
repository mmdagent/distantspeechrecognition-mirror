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


#include "sad/sad.h"
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_sf.h>

#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_complex.h>

// ----- methods for class `NeuralNetVAD' -----
//
NeuralNetVAD::
NeuralNetVAD(VectorFloatFeatureStreamPtr& cep,
				unsigned context, unsigned hiddenUnitsN, unsigned outputUnitsN, float threshold,
				const String& neuralNetFile)
  : _cep(cep), FrameResetX(-1), _cepLen(_cep->size()),
    _context(context), _hiddenUnitsN(hiddenUnitsN), _outputUnitsN(outputUnitsN), _threshold(threshold),
    _mlp(Mlp_Param_Mem_Alloc(_cepLen, _context, _hiddenUnitsN, _outputUnitsN)),
    _frame(new float*[2*_context+1])
{
  printf("Cepstra Length  = %d\n", _cepLen);
  for (unsigned rowX = 0; rowX < 2*_context+1; rowX++)
    _frame[rowX] = new float[_cepLen];

  if (neuralNetFile != "")
    Read_Mlp_Param(neuralNetFile.c_str(), _mlp, _cepLen, _context);
}

NeuralNetVAD::~NeuralNetVAD()
{
  Free_Mlp_Param_Mem(_mlp);

  for (unsigned rowX = 0; rowX < 2*_context+1; rowX++)
    delete[] _frame[rowX];
  delete[] _frame;
}

void NeuralNetVAD::reset()
{
  _framesPadded = 0; _frameX = FrameResetX;  _cep->reset();
}

void NeuralNetVAD::read(const String& neuralNetFile)
{
  printf("Reading neural net from file \'%s\'.\n", neuralNetFile.c_str());  fflush(stdout);
  Read_Mlp_Param(neuralNetFile.c_str(), _mlp, _cepLen, _context);
}

void NeuralNetVAD::_shiftDown()
{
  float* tmp = _frame[0];
  for (unsigned rowX = 0; rowX < 2 * _context; rowX++)
    _frame[rowX] = _frame[rowX+1];
  _frame[2*_context] = tmp;
}

void NeuralNetVAD::_updateBuffer(int frameX)
{
  _shiftDown();

  if (_framesPadded == 0) {			// normal processing
    try {

      const gsl_vector_float* cep = _cep->next(frameX);
      memcpy(_frame[2*_context], cep->data, _cepLen * sizeof(float));

    } catch  (jiterator_error& e) {
      memcpy(_frame[2*_context], _frame[2*_context - 1], _cepLen * sizeof(float));

      // printf("Padding frame %d.\n", _framesPadded);

      _framesPadded++;
    }

  } else if (_framesPadded < _context) {	// repeat last frame

    memcpy(_frame[2*_context], _frame[2*_context - 1], _cepLen * sizeof(float));

    // printf("Padding frame %d.\n", _framesPadded);

    _framesPadded++;

  } else {					// end of utterance

    throw jiterator_error("end of samples!");

  }
}

// return true if current frame is speech
bool NeuralNetVAD::next(int frameX)
{
  if (frameX == _frameX) return _isSpeech;

  if (frameX >= 0 && frameX != _frameX + 1)
    throw jindex_error("Problem speech activity detector: %d != %d\n", frameX - 1, _frameX);

  // "prime" the buffer
  if (_frameX == FrameResetX) {
    for (unsigned itnX = 0; itnX < _context; itnX++)
      _updateBuffer(0);
    for (unsigned itnX = 0; itnX < _context; itnX++)
      _updateBuffer(itnX);
  }

  _increment();
  _updateBuffer(_frameX + _context);

  int nsp_flag;
  Neural_Spnsp_Det(_frame, _cepLen, _context, _mlp, _threshold, &nsp_flag);
  _isSpeech = (nsp_flag ? false : true);

  return _isSpeech;
}


// ----- methods for class `VAD' -----
//
VAD::VAD(VectorComplexFeatureStreamPtr& samp)
  : _samp(samp), FrameResetX(-1), _fftLen(_samp->size()),
    _frame(gsl_vector_complex_alloc(_fftLen))
{
  printf("FFT Length  = %d\n", _fftLen);
}

VAD::~VAD()
{
  gsl_vector_complex_free(_frame);
}


// ----- methods for class `SimpleEnergyVAD' -----
//
SimpleEnergyVAD::
SimpleEnergyVAD(VectorComplexFeatureStreamPtr& samp,
		double threshold, double gamma)
  : VAD(samp),
    _threshold(threshold), _gamma(gamma), _spectralEnergy(0.0) { }

SimpleEnergyVAD::
~SimpleEnergyVAD() { }

void SimpleEnergyVAD::nextSpeaker()
{
  _spectralEnergy = 0.0;  reset(); 
}

// return true if current frame is speech
bool SimpleEnergyVAD::next(int frameX)
{
  if (frameX == _frameX) return _isSpeech;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem speech activity detector: %d != %d\n", frameX - 1, _frameX);
  
  const gsl_vector_complex* samp = _samp->next(frameX);
  gsl_vector_complex_memcpy(_frame, samp);

  double currentEnergy = 0.0;
  for (unsigned k = 0; k < _fftLen; k++)
    currentEnergy += gsl_complex_abs2(gsl_vector_complex_get(samp, k));

  _spectralEnergy = _gamma * _spectralEnergy + (1.0 - _gamma) * currentEnergy;

  _isSpeech = (currentEnergy / _spectralEnergy) > _threshold;
  
  _increment();
  return _isSpeech;
}

void SimpleEnergyVAD::reset()
{
  VAD::reset();  _samp->reset();
}


// ----- methods for class `SimpleLikelihoodRatioVAD' -----
//
SimpleLikelihoodRatioVAD::
SimpleLikelihoodRatioVAD(VectorComplexFeatureStreamPtr& samp,
				      double threshold, double alpha)
  : VAD(samp),
    _noiseVariance(gsl_vector_alloc(samp->size())),
    _varianceSet(false),
    _prevAk(gsl_vector_alloc(samp->size())),
    _prevFrame(gsl_vector_complex_alloc(samp->size())),
    _threshold(threshold), _alpha(alpha)
{
  gsl_vector_complex_set_zero(_prevFrame);
}

SimpleLikelihoodRatioVAD::~SimpleLikelihoodRatioVAD()
{
  gsl_vector_free(_noiseVariance);
  gsl_vector_free(_prevAk);
  gsl_vector_complex_free(_prevFrame);
}

void SimpleLikelihoodRatioVAD::nextSpeaker()
{
  _varianceSet = false;  reset();
}

void SimpleLikelihoodRatioVAD::setVariance(const gsl_vector* variance)
{
  // initialize 'Ak[n-1]' to the noise floor
  if (_varianceSet == false)
    for (unsigned k = 0; k < _fftLen; k++)
      gsl_vector_set(_prevAk, k, sqrt(gsl_vector_get(variance, k)));

  if (variance->size != _samp->size())
    throw jdimension_error("Variance and sample sizes (%d vs. %d) do not match.",
			   variance->size, _samp->size());

  gsl_vector_memcpy(_noiseVariance, variance);
  _varianceSet = true;
}

double SimpleLikelihoodRatioVAD::_calcAk(double vk, double gammak, double Rk)
{
  return (sqrt(M_PI) / 2.0) * (sqrt(vk) / gammak) * gsl_sf_hyperg_1F1(-0.5, 1.0, -vk) * Rk;
}

// return true if current frame is speech
bool SimpleLikelihoodRatioVAD::next(int frameX)
{
  if (_varianceSet == false)
    throw jconsistency_error("Must set noise variance before calling 'next()'.");

  if (frameX == _frameX) return _isSpeech;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem speech activity detector: %d != %d\n", frameX - 1, _frameX);

  const gsl_vector_complex* samp = _samp->next(frameX);
  gsl_vector_complex_memcpy(_frame, samp);

  double logLR = 0.0;
  for (unsigned k = 0; k < _fftLen; k++) {
    double Rk       = gsl_complex_abs(gsl_vector_complex_get(samp, k));
    double lambdaNk = gsl_vector_get(_noiseVariance, k);
    double gammak   = Rk * Rk / lambdaNk;
    double prevAk   = gsl_vector_get(_prevAk, k);
    double xik      = _alpha * (prevAk * prevAk / lambdaNk) + (1.0 - _alpha) * max(gammak - 1.0, 0.0);
    double vk       = (xik / (1.0 + xik)) * gammak;
    double Ak       = _calcAk(vk, gammak, Rk);

    gsl_vector_set(_prevAk, k, Ak);

    logLR += -log(1.0 + xik) + (gammak * xik / (1.0 + xik));
  }

  gsl_vector_complex_memcpy(_prevFrame, samp);

  _isSpeech = (logLR / _fftLen) > _threshold;

  _increment();
  return _isSpeech;
}

void SimpleLikelihoodRatioVAD::reset()
{
  _samp->reset();
}

// -----------------------------------------------------------------
//
//  Implementation Notes:
//
//  This speech activity detector is based on [1], which makes
//  extensive references to the minimum mean square estimation
//  techniques reported in [2].
//
// References:
//
// [1] J. Sohn, N. S. Kim, W. Sung, "A statistical model-based
//     voice activity detection," IEEE Signal Processing Letters,
//     6(1), January, 1999.

// [2] Y. Ephraim, D. Malah, "Speech enhancement using a minimum
//     mean-square error short-time spectral amplitude estimator,"
//     IEEE Trans. Acoust. Speech Signal Proc., ASSP-32(6),
//     December, 1984.
//
// -----------------------------------------------------------------

// ----- Methods for class 'EnergyVADFeature' -----
//
EnergyVADFeature::
EnergyVADFeature(const VectorFloatFeatureStreamPtr& source, double threshold, unsigned bufferLength, unsigned energiesN, const String& nm)
  : VectorFloatFeatureStream(source->size(), nm),
    _source(source), _recognizing(false),
    _bufferLength(bufferLength), _buffer(NULL), _bufferIndex(0), _bufferedN(_bufferLength),
    _energiesN(energiesN), _energies(new double[_energiesN]), _medianIndex(unsigned(threshold * _energiesN))
{
  _buffer = new gsl_vector_float*[_bufferLength];
  for (unsigned n = 0; n < _bufferLength; n++)
    _buffer[n] = gsl_vector_float_calloc(_source->size());

  _energies       = new double[_energiesN];
  _sortedEnergies = new double[_energiesN];

  for (unsigned n = 0; n < _energiesN; n++)
    _energies[n] = HUGE;

  // printf("Median index %d\n", _medianIndex);
}

EnergyVADFeature::~EnergyVADFeature()
{
  for (unsigned n = 0; n < _bufferLength; n++)
    gsl_vector_float_free(_buffer[n]);

  delete[] _buffer;  delete[] _energies;  delete[] _sortedEnergies;
}

void EnergyVADFeature::reset()
{
  /* _source->reset(); */ VectorFloatFeatureStream::reset();

  _bufferedN   = _bufferIndex = _aboveThresholdN = _belowThresholdN = 0;
  _recognizing = false;
}

void EnergyVADFeature::nextSpeaker()
{
  for (unsigned n = 0; n < _energiesN; n++)
    _energies[n] = HUGE;
}

int EnergyVADFeature::_comparator(const void* elem1, const void* elem2)
{
  double* e1 = (double*) elem1;
  double* e2 = (double*) elem2;
  
  if (*e1 == *e2) return 0;
  if (*e1 <  *e2) return -1;
  return 1;
}

bool EnergyVADFeature::_aboveThreshold(const gsl_vector_float* vector)
{
  double sum = 0.0;
  for (unsigned n = 0; n < vector->size; n++) {
    double val = gsl_vector_float_get(vector, n);
    sum += val * val;
  }

  memcpy(_sortedEnergies, _energies, _energiesN * sizeof(double));
  qsort(_sortedEnergies, _energiesN, sizeof(double), _comparator);

  if (_recognizing == false && _aboveThresholdN == 0) {
    memmove(_energies, _energies + 1, (_energiesN - 1) * sizeof(double));
    _energies[_energiesN - 1] = sum;
  }

  // printf("Threshold = %10.2f\n", _sortedEnergies[_medianIndex]);

  return ((sum > _sortedEnergies[_medianIndex]) ? true : false);
}

const gsl_vector_float* EnergyVADFeature::next(int frameX)
{
  if (_recognizing) {

    // use up the buffered blocks
    if (_bufferedN > 0) {
      const gsl_vector_float* vector = _buffer[_bufferIndex];
      _bufferIndex = (_bufferIndex + 1) % _bufferLength;
      _bufferedN--;
      return vector;
    }

    // buffer is empty; take blocks directly from source
    const gsl_vector_float* vector = _source->next();
    if (_aboveThreshold(vector)) {
      _belowThresholdN = 0;
    } else {
      if (_belowThresholdN == _bufferLength)
	throw jiterator_error("end of samples!");
      _belowThresholdN++;
    }
    return vector;

  } else {

    // buffer sample blocks until sufficient blocks have energy above the threshold
    while (true) {
      const gsl_vector_float* vector = _source->next();
      gsl_vector_float_memcpy(_buffer[_bufferIndex], vector);
      _bufferIndex = (_bufferIndex + 1) % _bufferLength;
      _bufferedN = min(_bufferLength, _bufferedN + 1);

      if (_aboveThreshold(vector)) {
	if (_aboveThresholdN == _bufferLength) {
	  _recognizing = true;
	  vector = _buffer[_bufferIndex];
	  _bufferIndex = (_bufferIndex + 1) % _bufferLength;
	  _bufferedN--;
	  return vector;
	}
	_aboveThresholdN++;
      } else {
	_aboveThresholdN = 0;
      }
    }
  }
}


// ----- Methods for class 'EnergyVADMetric' -----
//
EnergyVADMetric::
EnergyVADMetric(const VectorFloatFeatureStreamPtr& source, double initialEnergy, double threshold, unsigned headN,
		unsigned tailN, unsigned energiesN, const String& nm)
  : _source(source), _initialEnergy(initialEnergy), _headN(headN), _tailN(tailN), _recognizing(false),
    _aboveThresholdN(0), _belowThresholdN(0), _energiesN(energiesN), _energies(new double[_energiesN]),
    _medianIndex(unsigned(threshold * _energiesN))
{
  _energies       = new double[_energiesN];
  _sortedEnergies = new double[_energiesN];

  for (unsigned n = 0; n < _energiesN; n++)
    _energies[n] = _initialEnergy;

  // printf("Median index %d\n", _medianIndex);
}

EnergyVADMetric::~EnergyVADMetric()
{
  delete[] _energies;  delete[] _sortedEnergies;
}

void EnergyVADMetric::reset()
{
  _aboveThresholdN = _belowThresholdN = 0;
  _recognizing = false;
}

void EnergyVADMetric::nextSpeaker()
{
  _aboveThresholdN = _belowThresholdN = 0;
  _recognizing = false;

  for (unsigned n = 0; n < _energiesN; n++)
    _energies[n] = _initialEnergy;
}

int EnergyVADMetric::_comparator(const void* elem1, const void* elem2)
{
  double* e1 = (double*) elem1;
  double* e2 = (double*) elem2;
  
  if (*e1 == *e2) return 0;
  if (*e1 <  *e2) return -1;
  return 1;
}

bool EnergyVADMetric::_aboveThreshold(const gsl_vector_float* vector)
{
  double sum = 0.0;
  for (unsigned n = 0; n < vector->size; n++) {
    double val = gsl_vector_float_get(vector, n);
    sum += val * val;
  }

  memcpy(_sortedEnergies, _energies, _energiesN * sizeof(double));
  qsort(_sortedEnergies, _energiesN, sizeof(double), _comparator);

  if (_recognizing == false && _aboveThresholdN == 0) {
    memmove(_energies, _energies + 1, (_energiesN - 1) * sizeof(double));
    _energies[_energiesN - 1] = sum;
  }

  // printf("Threshold = %12.4e\n", _sortedEnergies[_medianIndex]);
  _curScore = sum;
#ifdef _LOG_SAD_ 
  writeLog( "%d %e\n", _frameX, sum );
  setScore( sum );
#endif

  return ((sum > _sortedEnergies[_medianIndex]) ? true : false);
}

double EnergyVADMetric::energyPercentile(double percentile) const
{
  if ( percentile < 0.0 ||  percentile > 100.0)
    throw jdimension_error("Percentile %g is out of range [0.0, 100.0].", percentile);

  memcpy(_sortedEnergies, _energies, _energiesN * sizeof(double));
  qsort(_sortedEnergies, _energiesN, sizeof(double), _comparator);

  return _sortedEnergies[int((percentile / 100.0) * _energiesN)] / _energiesN;
}

double EnergyVADMetric::next(int frameX)
{
#ifdef  _LOG_SAD_ 
  _frameX = frameX;
#endif /* _LOG_SAD_ */

  const gsl_vector_float* vector = _source->next(frameX);
  if (_recognizing) {

    if (_aboveThreshold(vector)) {
      _belowThresholdN = 0;
      return 1.0;
    } else {
      _belowThresholdN++;
      if (_belowThresholdN == _tailN) {
	_recognizing = false;  _aboveThresholdN = 0;
      }
      return 0.0;
    }

  } else {

    if (_aboveThreshold(vector)) {
      _aboveThresholdN++;
      if (_aboveThresholdN == _headN) {
	_recognizing = true;  _belowThresholdN = 0;
      }
      return 1.0;
    } else {
      _aboveThresholdN = 0;
      return 0.0;
    }
  }
}

// ----- definition for class `MultiChannelVADMetric' -----
//

template<> MultiChannelVADMetric<VectorFloatFeatureStreamPtr>::MultiChannelVADMetric(unsigned fftLen, double sampleRate, double lowCutoff, double highCutoff, const String& nm)
  : _fftLen(fftLen), _fftLen2(fftLen / 2), _sampleRate(sampleRate), 
    _lowX(_setLowX(lowCutoff)), 
    _highX(_setHighX(highCutoff)), 
    _binN(_setBinN()),
    _logfp(NULL)
{
}

template<> MultiChannelVADMetric<VectorFloatFeatureStreamPtr>::~MultiChannelVADMetric()
{
  if( NULL != _logfp )
    fclose( _logfp );
}

template<> MultiChannelVADMetric<VectorComplexFeatureStreamPtr>::MultiChannelVADMetric(unsigned fftLen, double sampleRate, double lowCutoff, double highCutoff, const String& nm)
  : _fftLen(fftLen), _fftLen2(fftLen / 2), _sampleRate(sampleRate), 
    _lowX(_setLowX(lowCutoff)), 
    _highX(_setHighX(highCutoff)), 
    _binN(_setBinN()) 
{
}

template<> MultiChannelVADMetric<VectorComplexFeatureStreamPtr>::~MultiChannelVADMetric()
{}

template<> void MultiChannelVADMetric<VectorFloatFeatureStreamPtr>::setChannel(VectorFloatFeatureStreamPtr& chan)
{
  _channelList.push_back(chan);
}

template<> void MultiChannelVADMetric<VectorComplexFeatureStreamPtr>::setChannel(VectorComplexFeatureStreamPtr& chan)
{
  _channelList.push_back(chan);
}

template <typename ChannelType> unsigned MultiChannelVADMetric<ChannelType>::_setLowX(double lowCutoff) const
{
  if (lowCutoff < 0.0) return 0;
  
  if (lowCutoff >= _sampleRate / 2.0)
    throw jdimension_error("Low cutoff cannot be %10.1", lowCutoff);

  unsigned binX = (unsigned) ((lowCutoff / _sampleRate) * _fftLen);

  printf("Setting lowest bin to %d\n", binX);

  return binX;
}

template <typename ChannelType> unsigned MultiChannelVADMetric<ChannelType>::_setHighX(double highCutoff) const
{
  if (highCutoff < 0.0) return _fftLen2;
  
  if (highCutoff >= _sampleRate / 2.0)
    throw jdimension_error("Low cutoff cannot be %10.1", highCutoff);

  unsigned binX = (unsigned) ((highCutoff / _sampleRate) * _fftLen + 0.5);

  printf("Setting highest bin to %d\n", binX);

  return binX;
}

template <typename ChannelType> unsigned MultiChannelVADMetric<ChannelType>::_setBinN() const
{
  if (_lowX > 0)
    return 2 * (_highX - _lowX + 1);

  return 2 * (_highX - _lowX) + 1;
}

// ----- Methods for class 'PowerSpectrumVADMetric' -----
//
PowerSpectrumVADMetric::PowerSpectrumVADMetric(unsigned fftLen,
					       double sampleRate, double lowCutoff, double highCutoff,
					       const String& nm)
  : FloatMultiChannelVADMetric( fftLen, sampleRate, lowCutoff, highCutoff, nm)
{ 
  _powerList = NULL;
  _E0 = 1;
}

PowerSpectrumVADMetric::PowerSpectrumVADMetric(VectorFloatFeatureStreamPtr& source1, VectorFloatFeatureStreamPtr& source2,
					       double sampleRate, double lowCutoff, double highCutoff,
					       const String& nm)
  : FloatMultiChannelVADMetric( source1->size(), sampleRate, lowCutoff, highCutoff, nm)
{ 
  setChannel( source1 );
  setChannel( source2 ); 
  _powerList = NULL;
  _E0 = 1;
}

PowerSpectrumVADMetric::~PowerSpectrumVADMetric()
{
  if( NULL != _powerList )
    gsl_vector_free( _powerList );
  _powerList = NULL;
}

double PowerSpectrumVADMetric::next(int frameX)
{
  const gsl_vector_float* spectrum_n;
  unsigned chanX = 0;
  unsigned chanN = _channelList.size();
  unsigned argMaxChanX = -1;
  double maxPower = -1;
  double totalpower = 0;
  double power_ratio = 0;

#ifdef  _LOG_SAD_ 
  _frameX = frameX;
#endif /* _LOG_SAD_ */

  if( NULL == _powerList )
    _powerList = gsl_vector_alloc( chanN );
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++,chanX++) {
    double power_n = 0.0;

    spectrum_n = (*itr)->next(frameX);
    for (unsigned fbinX = _lowX; fbinX <= _highX; fbinX++) {
      if (fbinX == 0 || fbinX == (_fftLen2 + 1)) {
	power_n += gsl_vector_float_get(spectrum_n, fbinX);
      } else {
	power_n += 2.0 * gsl_vector_float_get(spectrum_n, fbinX);
      }
    }
    power_n /= _fftLen;
    totalpower += power_n;
    gsl_vector_set( _powerList, chanX, power_n );

    if( power_n > maxPower ){
      maxPower = power_n;
      argMaxChanX = chanX;
    }
  }

  power_ratio = gsl_vector_get( _powerList,0) / totalpower;

  if( power_ratio > (_E0/chanN) )
    return 1.0;
  else
    return -1.0;

  return 0.0;
}

void PowerSpectrumVADMetric::reset()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();
}

void PowerSpectrumVADMetric::nextSpeaker()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();
}

void PowerSpectrumVADMetric::clearChannel()
{
  if(  (int)_channelList.size() > 0 )
    _channelList.clear();
  if( NULL != _powerList )
    gsl_vector_free( _powerList );
  _powerList = NULL;
}

// ----- definition for class `NormalizedEnergyMetric' -----
//
NormalizedEnergyMetric::NormalizedEnergyMetric(unsigned fftLen, double sampleRate, double lowCutoff, double highCutoff, const String& nm ):
PowerSpectrumVADMetric( fftLen, sampleRate, lowCutoff, highCutoff, nm )
{
  _E0 = 1.0;
}

NormalizedEnergyMetric::~NormalizedEnergyMetric()
{}

/**
   @brief compuate the ratio of each channel's energy to the total energy at each frame.
   @return 1.0 if the voice detected. Otherwise, return 0.0.
 */
double NormalizedEnergyMetric::next(int frameX)
{
  const gsl_vector_float* spectrum_n;
  unsigned chanX = 0;
  unsigned chanN = _channelList.size();
  unsigned argMaxChanX = -1;
  double maxPower = -1;
  double totalenergy = 0;
  double energy_ratio = 0;

#ifdef  _LOG_SAD_ 
  _frameX = frameX;
#endif /* _LOG_SAD_ */

  if( NULL == _powerList )
    _powerList = gsl_vector_alloc( chanN );
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++,chanX++) {
    double power_n = 0.0;

    spectrum_n = (*itr)->next(frameX);
    for (unsigned fbinX = _lowX; fbinX <= _highX; fbinX++) {
      if (fbinX == 0 || fbinX == (_fftLen2 + 1)) {
	power_n += gsl_vector_float_get(spectrum_n, fbinX);
      } else {
	power_n += 2.0 * gsl_vector_float_get(spectrum_n, fbinX);
      }
    }
    power_n /= _fftLen;
    totalenergy += sqrt( power_n );
    gsl_vector_set( _powerList, chanX, power_n );

    if( power_n > maxPower ){
      maxPower = power_n;
      argMaxChanX = chanX;
    }
    //fprintf(stderr,"Energy Chan %d : %e\n", chanX, power_n  );
  }

  energy_ratio = sqrt( gsl_vector_get( _powerList,0) ) / totalenergy;
  _curScore = energy_ratio;

  //fprintf(stderr,"ER %e  %e\n", energy_ratio, _E0/chanN  );
#ifdef  _LOG_SAD_ 
  writeLog( "%d %e\n", _frameX, energy_ratio );
  setScore( energy_ratio );
#endif /* _LOG_SAD_ */
  
  if ( energy_ratio > (_E0/chanN) )
    return 1.0;
  else
    return -1.0;
  
  return 0.0;
}

void NormalizedEnergyMetric::reset()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();
}

// ----- definition for class `CCCVADMetric' -----
//

/**
   @brief constructor
   @param unsigned fftLen[in]
   @param unsigned nCand[in] the number of candidates
   @param double sampleRate[in]
   @param double lowCutoff[in]
   @param double highCutoff[in]
*/
CCCVADMetric::CCCVADMetric(unsigned fftLen, unsigned nCand, double sampleRate, double lowCutoff, double highCutoff, const String& nm ):
  ComplexMultiChannelVADMetric( fftLen, sampleRate, lowCutoff, highCutoff, nm)
{
  _nCand = nCand;
  _ccList = gsl_vector_alloc( _nCand );
  _sampleDelays = gsl_vector_int_alloc( _nCand );
  _packCrossSpectrum = new double[2*_fftLen];
  _threshold = 0.1;
}

CCCVADMetric::~CCCVADMetric()
{
  gsl_vector_free( _ccList );
  gsl_vector_int_free( _sampleDelays );
  delete [] _packCrossSpectrum;
}

void CCCVADMetric::setNCand(unsigned nCand)
{
  gsl_vector_free( _ccList );
  gsl_vector_int_free( _sampleDelays );

  _nCand = nCand;
  _ccList = gsl_vector_alloc( _nCand );
  _sampleDelays = gsl_vector_int_alloc( _nCand );
}

double CCCVADMetric::next(int frameX)
{
#define myREAL(z,i) ((z)[2*(i)])
#define myIMAG(z,i) ((z)[2*(i)+1])
  size_t stride = 1;
  const gsl_vector_complex* refSpectrum;
  const gsl_vector_complex* spectrum;
  double totalCCMetric = 0.0;

#ifdef  _LOG_SAD_ 
  _frameX = frameX;
#endif /* _LOG_SAD_ */

  for(unsigned fbinX=0;fbinX<2*_fftLen;fbinX++)
    _packCrossSpectrum[fbinX] = 0.0;

  _ChannelIterator itr = _channelList.begin();
  refSpectrum = (*itr)->next(frameX);
  itr++;
  for (unsigned chanX = 1; itr != _channelList.end(); itr++,chanX++) {
    spectrum = (*itr)->next(frameX);
    for (unsigned fbinX = _lowX; fbinX <= _highX; fbinX++) {
      gsl_complex val1, val2, cc;
    
      val1 = gsl_vector_complex_get( refSpectrum, fbinX );
      val2 = gsl_vector_complex_get( spectrum, fbinX );
      cc   = gsl_complex_mul( gsl_complex_conjugate( val1 ), val2 );
      cc   = gsl_complex_div_real( cc,  gsl_complex_abs( cc ) );
      myREAL( _packCrossSpectrum, fbinX ) = GSL_REAL(cc);
      myIMAG( _packCrossSpectrum, fbinX ) = GSL_IMAG(cc);
      if( fbinX > 0 ){
	myREAL( _packCrossSpectrum, (_fftLen-fbinX)*stride ) =  GSL_REAL(cc);
	myIMAG( _packCrossSpectrum, (_fftLen-fbinX)*stride ) = -GSL_IMAG(cc);
      }
    }
    gsl_fft_complex_radix2_inverse( _packCrossSpectrum, stride, _fftLen );// with scaling
    {/* detect _nHeldMaxCC peaks */
      /* _ccList[0] > _ccList[1] > _ccList[2] ... */

      gsl_vector_int_set( _sampleDelays, 0, 0 );
      gsl_vector_set( _ccList, 0, myREAL( _packCrossSpectrum, 0 ) );
      for(unsigned i=1;i<_nCand;i++){
	gsl_vector_int_set( _sampleDelays, i, -10 );
	gsl_vector_set( _ccList, i, -1e10 );
      }
      for(unsigned fbinX=1;fbinX<_fftLen;fbinX++){
	double cc = myREAL( _packCrossSpectrum, fbinX );
	
	if( cc > gsl_vector_get( _ccList, _nCand-1 ) ){
	  for(unsigned i=0;i<_nCand;i++){
	    if( cc > gsl_vector_get( _ccList, i ) ){
	      for(unsigned j=_nCand-1;j>i;j--){
		gsl_vector_int_set( _sampleDelays, j, gsl_vector_int_get( _sampleDelays, j-1 ) );
		gsl_vector_set(     _ccList,       j, gsl_vector_get( _ccList, j-1 ) );
	      }
	    }
	    gsl_vector_int_set( _sampleDelays, i, fbinX);
	    gsl_vector_set(     _ccList,       i, cc);
	    break;
	  }
	}
      }

      double ccMetric = 0.0;
      //set time delays to _vector
      for(unsigned i=0;i<_nCand;i++){
	unsigned sampleDelay = gsl_vector_int_get( _sampleDelays, i );
	double   cc = gsl_vector_get( _ccList, i );
	float timeDelay;
	
	if( sampleDelay < _fftLen/2 ){
	  timeDelay = sampleDelay * 1.0 / _sampleRate;
	}
	else{
	  timeDelay = - ( _fftLen - sampleDelay ) * 1.0 / _sampleRate;
	}
	//fprintf(stderr,"Chan %d : %d : SD = %d : TD = %e : CC = %e\n", chanX, i, sampleDelay, timeDelay, cc );
	ccMetric += cc;
      }
      ccMetric /= _nCand;
      totalCCMetric += ccMetric;
    }
  }
  totalCCMetric = totalCCMetric / ( _channelList.size() - 1 );
  _curScore = totalCCMetric;
  //fprintf(stderr,"Fr %d : %e\n", frameX, totalCCMetric );
#ifdef  _LOG_SAD_ 
  writeLog( "%d %e\n", _frameX, totalCCMetric );
  setScore( totalCCMetric );
#endif /* _LOG_SAD_ */

  if( totalCCMetric < _threshold )
    return 1.0;
  else
    return -1.0;
  return 0.0;

#undef myREAL
#undef myIMAG
}

void CCCVADMetric::reset()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();
}

void CCCVADMetric::nextSpeaker()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();
}

void CCCVADMetric::clearChannel()
{
  if(  (int)_channelList.size() > 0 )
    _channelList.clear();
}

// ----- definition for class `TSPSVADMetric' -----
//
TSPSVADMetric::TSPSVADMetric(unsigned fftLen, double sampleRate, double lowCutoff, double highCutoff, const String& nm ):
PowerSpectrumVADMetric( fftLen, sampleRate, lowCutoff, highCutoff, nm )
{
  _E0 = 5000;
}

TSPSVADMetric::~TSPSVADMetric()
{}

double TSPSVADMetric::next(int frameX)
{
  const gsl_vector_float* spectrum_n;
  unsigned chanX = 0;
  unsigned chanN = _channelList.size();
  unsigned argMaxChanX = -1;
  double maxPower = -1;
  double totalpower = 0;
  double TSPS = 0;
  double tgtPower;

#ifdef  _LOG_SAD_ 
  _frameX = frameX;
#endif /* _LOG_SAD_ */

  if( NULL == _powerList )
    _powerList = gsl_vector_alloc( chanN );

  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++,chanX++) {
    double power_n = 0.0;

    spectrum_n = (*itr)->next(frameX);
    for (unsigned fbinX = _lowX; fbinX <= _highX; fbinX++) {
      if (fbinX == 0 || fbinX == (_fftLen2 + 1)) {
	power_n += gsl_vector_float_get(spectrum_n, fbinX);
      } else {
	power_n += 2.0 * gsl_vector_float_get(spectrum_n, fbinX);
      }
    }
    power_n /= _fftLen;
    totalpower += power_n;
    gsl_vector_set( _powerList, chanX, power_n );

    if( power_n > maxPower ){
      maxPower = power_n;
      argMaxChanX = chanX;
    }
  }
  
  tgtPower = gsl_vector_get( _powerList, 0 );
  TSPS = log(tgtPower/(totalpower-tgtPower)) - log(_E0/totalpower);
#ifdef  _LOG_SAD_ 
  writeLog( "%d %e\n", _frameX, TSPS );
  setScore( TSPS );
#endif /* _LOG_SAD_ */

  if( TSPS > 0 )
    return 1.0;
  else
    return -1.0;

  return 0.0;
}

void TSPSVADMetric::reset()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();
}

// ----- Methods for class 'NegentropyVADMetric::_ComplexGeneralizedGaussian' -----
//
NegentropyVADMetric::_ComplexGeneralizedGaussian::
_ComplexGeneralizedGaussian(double shapeFactor)
  : _shapeFactor(shapeFactor), _Bc(_calcBc()), _normalization(_calcNormalization()) { }

double NegentropyVADMetric::_ComplexGeneralizedGaussian::_calcBc() const
{
  double lg1 = gsl_sf_lngamma(2.0 / _shapeFactor);
  double lg2 = gsl_sf_lngamma(4.0 / _shapeFactor);
  
  return exp((lg1 - lg2) / 2.0);
}

double NegentropyVADMetric::_ComplexGeneralizedGaussian::_calcNormalization() const
{
  return log(_shapeFactor / (2 * M_PI * _Bc * _Bc * gsl_sf_gamma(2.0 / _shapeFactor)));
}

double NegentropyVADMetric::_ComplexGeneralizedGaussian::logLhood(gsl_complex X, double scaleFactor) const
{
  double logDeterminant = 2.0 * log(scaleFactor);

  double s = pow(gsl_complex_abs(X) / (scaleFactor * _Bc), _shapeFactor);

  return _normalization - pow(gsl_complex_abs(X) / (scaleFactor * _Bc), _shapeFactor) - 2.0 * log(scaleFactor);
}


// ----- Methods for class 'NegentropyVADMetric' -----
//
NegentropyVADMetric::
NegentropyVADMetric(const VectorComplexFeatureStreamPtr& source, const VectorFloatFeatureStreamPtr& spectralEstimator,
		    const String& shapeFactorFileName, double threshold, double sampleRate, double lowCutoff, double highCutoff,
		    const String& nm)
  : _source(source),
    _spectralEstimator(spectralEstimator), _gaussian(new _ComplexGeneralizedGaussian()),
    _threshold(threshold), _fftLen(_source->size()), _fftLen2(_fftLen / 2),
    _sampleRate(sampleRate), _lowX(_setLowX(lowCutoff)), _highX(_setHighX(highCutoff)), _binN(_setBinN())
{
  size_t n      = 0;
  char*  buffer = NULL;
  static char fileName[256];

  // initialize the GG pdfs
  for (unsigned fbinX = 0; fbinX <= _fftLen2; fbinX++) {

    double factor = 2.0;	// default is Gaussian pdf
    if (shapeFactorFileName != "") {
      sprintf(fileName, "%s/_M-%04d", shapeFactorFileName.c_str(), fbinX);
      FILE* fp = fileOpen(fileName, "r");
      getline(&buffer, &n, fp);
      static char* token[2];
      token[0] = strtok(buffer, " ");
      token[1] = strtok(NULL, " ");

      factor = strtod(token[1], NULL);
      fileClose(fileName, fp);
    }

    // printf("Bin %d has shape factor %8.4f\n", fbinX, factor);  fflush(stdout);

    _generalizedGaussians.push_back(_ComplexGeneralizedGaussianPtr(new _ComplexGeneralizedGaussian(factor)));
  }

  if (_generalizedGaussians.size() != _fftLen2 + 1)
    throw jdimension_error("Numbers of spectral bins and shape factors do not match (%d vs. %d)", _generalizedGaussians.size(), _fftLen2 + 1);
}

NegentropyVADMetric::~NegentropyVADMetric() { }

double NegentropyVADMetric::calcNegentropy(int frameX)
{
  const gsl_vector_complex* sample   = _source->next(frameX);
  const gsl_vector_float*   envelope = _spectralEstimator->next(frameX);

  unsigned fbinX = 0;
  double logLikelihoodRatio = 0.0;
  for (_GaussianListConstIterator itr = _generalizedGaussians.begin(); itr != _generalizedGaussians.end(); itr++) {
    _ComplexGeneralizedGaussianPtr gg(*itr);
    gsl_complex X      = gsl_vector_complex_get(sample, fbinX);
    double      sigmaH = sqrt(gsl_vector_float_get(envelope, fbinX));
    double	lr     = gg->logLhood(X, sigmaH) - _gaussian->logLhood(X, sigmaH);

    if (fbinX >= _lowX && fbinX <= _highX) {
      if (fbinX == 0 || fbinX == (_fftLen2 + 1))
	logLikelihoodRatio += lr;
      else
	logLikelihoodRatio += 2.0 * lr;
    }

    fbinX++;
  }

  logLikelihoodRatio /= _binN;

  printf("Frame %d : Negentropy ratio = %12.4e\n", frameX, logLikelihoodRatio);

  return logLikelihoodRatio;
}

double NegentropyVADMetric::next(int frameX)
{
#ifdef  _LOG_SAD_ 
  _frameX = frameX;
#endif /* _LOG_SAD_ */

  if (calcNegentropy(frameX) > _threshold)
    return 1.0;
  return 0.0;
}

void NegentropyVADMetric::reset()
{
  _source->reset();  _spectralEstimator->reset();
}

void NegentropyVADMetric::nextSpeaker()
{
  cout << "NegentropyVADMetric::nextSpeaker" << endl;
}

bool NegentropyVADMetric::_aboveThreshold(int frameX)
{
  return next(frameX) > _threshold;
}

unsigned NegentropyVADMetric::_setLowX(double lowCutoff) const
{
  if (lowCutoff < 0.0) return 0;
  
  if (lowCutoff >= _sampleRate / 2.0)
    throw jdimension_error("Low cutoff cannot be %10.1", lowCutoff);

  unsigned binX = (unsigned) ((lowCutoff / _sampleRate) * _fftLen);

  printf("Setting lowest bin to %d\n", binX);

  return binX;
}

unsigned NegentropyVADMetric::_setHighX(double highCutoff) const
{
  if (highCutoff < 0.0) return _fftLen2;
  
  if (highCutoff >= _sampleRate / 2.0)
    throw jdimension_error("Low cutoff cannot be %10.1", highCutoff);

  unsigned binX = (unsigned) ((highCutoff / _sampleRate) * _fftLen + 0.5);

  printf("Setting highest bin to %d\n", binX);

  return binX;
}

unsigned NegentropyVADMetric::_setBinN() const
{
  if (_lowX > 0)
    return 2 * (_highX - _lowX + 1);

  return 2 * (_highX - _lowX) + 1;
}


// ----- Methods for class 'MutualInformationVADMetric::_JointComplexGeneralizedGaussian' -----
//
MutualInformationVADMetric::_JointComplexGeneralizedGaussian::
_JointComplexGeneralizedGaussian(const _ComplexGeneralizedGaussianPtr& ggaussian)
  : _ComplexGeneralizedGaussian(_match(ggaussian->shapeFactor())),
    _X(gsl_vector_complex_calloc(2)), _scratch(gsl_vector_complex_calloc(2)),
    _SigmaXinverse(gsl_matrix_complex_calloc(2, 2))
{
  _Bc = _calcBc();
  _normalization = _calcNormalization();

  /*
  unsigned Steps = 1000;
  static char fileName[] = "./entropy-shape-factors.txt";
  FILE* fp = fileOpen(fileName, "w");
  for (unsigned n = 100; n <= Steps; n++) {
    double f  = 1.0 * n / Steps;
    double fj = _match(f);
    fprintf(fp, "%14.6f  %14.6f\n", f, fj);
  }
  fileClose(fileName, fp);
  printf("All done\n");
  */
}

MutualInformationVADMetric::_JointComplexGeneralizedGaussian::
~_JointComplexGeneralizedGaussian()
{
  gsl_vector_complex_free(_X);
  gsl_vector_complex_free(_scratch);
  gsl_matrix_complex_free(_SigmaXinverse);
}

double MutualInformationVADMetric::_JointComplexGeneralizedGaussian::_calcBc() const
{
  double lg1 = gsl_sf_lngamma(4.0 / _shapeFactor);
  double lg2 = gsl_sf_lngamma(6.0 / _shapeFactor);

  // cout << "Executing _JointComplexGeneralizedGaussian::_calcBc()" << endl;

  double val = exp((lg1 - lg2) / 2.0);
  
  return exp((lg1 - lg2) / 2.0);
}

double MutualInformationVADMetric::_JointComplexGeneralizedGaussian::_calcNormalization() const
{
  // cout << "Executing _JointComplexGeneralizedGaussian::_calcNormalization()" << endl;

  return log(_shapeFactor / (8.0 * M_PI * M_PI * _Bc * _Bc * _Bc * _Bc * gsl_sf_gamma(4.0 / _shapeFactor)));
}

const double      MutualInformationVADMetric::_JointComplexGeneralizedGaussian::_sqrtTwo	= sqrt(2.0);
const gsl_complex MutualInformationVADMetric::_JointComplexGeneralizedGaussian::_complexOne	= gsl_complex_rect(1.0, 0.0);
const gsl_complex MutualInformationVADMetric::_JointComplexGeneralizedGaussian::_complexZero	= gsl_complex_rect(0.0, 0.0);

double MutualInformationVADMetric::_JointComplexGeneralizedGaussian::
logLhood(gsl_complex X1, gsl_complex X2, double scaleFactor1, double scaleFactor2, gsl_complex rho12) const
{
  // create the combined feature vector
  gsl_vector_complex_set(_X, 0, X1);
  gsl_vector_complex_set(_X, 1, X2);

  // calculate inverse of Sigma_X (up to scale factor)
  gsl_complex sigma12 = gsl_complex_mul_real(rho12, scaleFactor1 * scaleFactor2);
  gsl_complex sigma11 = gsl_complex_rect(scaleFactor1 * scaleFactor1, 0.0);
  gsl_complex sigma22 = gsl_complex_rect(scaleFactor2 * scaleFactor2, 0.0);

  gsl_matrix_complex_set(_SigmaXinverse, 0, 0, sigma22);
  gsl_matrix_complex_set(_SigmaXinverse, 1, 1, sigma11);
  gsl_matrix_complex_set(_SigmaXinverse, 0, 1, gsl_complex_rect(-GSL_REAL(sigma12), -GSL_IMAG(sigma12)));
  gsl_matrix_complex_set(_SigmaXinverse, 1, 0, gsl_complex_rect(-GSL_REAL(sigma12),  GSL_IMAG(sigma12)));

  // calculate determinant
  double determinant    = scaleFactor1 * scaleFactor1 * scaleFactor2 * scaleFactor2 * (1.0 - gsl_complex_abs2(rho12));
  double logDeterminant = log(determinant);

  // scale inverse of Sigma_X
  gsl_complex determinantComplex = gsl_complex_rect(1.0 / determinant, 0.0);
  gsl_matrix_complex_scale(_SigmaXinverse, determinantComplex);

  // calculate (square-root of) s
  gsl_complex s;
  gsl_blas_zgemv(CblasNoTrans, _complexOne, _SigmaXinverse, _X, _complexZero, _scratch);
  gsl_blas_zdotc(_X, _scratch, &s);
  double ssqrt = sqrt(gsl_complex_abs(s));

  return _normalization - pow(ssqrt / (_sqrtTwo * _Bc), _shapeFactor) - logDeterminant;
}

double MutualInformationVADMetric::_JointComplexGeneralizedGaussian::_lngammaRatio(double f) const
{
  return gsl_sf_lngamma(2.0 / f) + gsl_sf_lngamma(6.0 / f) - 2.0 * gsl_sf_lngamma(4.0 / f);
}

double MutualInformationVADMetric::_JointComplexGeneralizedGaussian::_lngammaRatioJoint(double f) const
{
  return gsl_sf_lngamma(4.0 / f) + gsl_sf_lngamma(8.0 / f) - 2.0 * gsl_sf_lngamma(6.0 / f);
}

#if 0

// these methods are for kurtosis matching
double MutualInformationVADMetric::_JointComplexGeneralizedGaussian::_matchScoreMarginal(double f) const
{
  return log(0.5 * (exp(_lngammaRatio(f)) + 1.0));
}

double MutualInformationVADMetric::_JointComplexGeneralizedGaussian::_matchScoreJoint(double f) const
{
  return _lngammaRatioJoint(f);
}

#else

// these methods are for entropy matching
double MutualInformationVADMetric::_JointComplexGeneralizedGaussian::_matchScoreMarginal(double f) const
{
  double lg1		= gsl_sf_lngamma(2.0 / f);
  double lg2		= gsl_sf_lngamma(4.0 / f);
  double Bc2		= exp(lg1 - lg2);
  double gamma2f	= gsl_sf_gamma(2.0 / f);
  double match		= 2.0 * ((2.0 / f) - log(f / (2.0 * M_PI * Bc2 * gamma2f)));

  return -match;
}

double MutualInformationVADMetric::_JointComplexGeneralizedGaussian::_matchScoreJoint(double fJ) const
{
  double lg1		= gsl_sf_lngamma(4.0 / fJ);
  double lg2		= gsl_sf_lngamma(6.0 / fJ);
  double BJ4 		= exp((lg1 - lg2) * 2.0);
  double gamma4fJ	= gsl_sf_gamma(4.0 / fJ);
  double match		= ((4.0 / fJ) - log(fJ / (8.0 * M_PI * M_PI * BJ4 * gamma4fJ)));

  return -match;
}

#endif

const double MutualInformationVADMetric::_JointComplexGeneralizedGaussian::_tolerance = 1.0e-06;

double MutualInformationVADMetric::_JointComplexGeneralizedGaussian::_match(double f) const
{
  double a     = f / 3.0;					// lower bound: univariate shape factor
  double c     = 2.0;						// upper bound: Gaussian shape factor
  double match = _matchScoreMarginal(f);			// must match this value

  // printf("f = %10.6f\n", f);

  // return f;

  // execute binary search
  while (true) {
    double b		= (a + c) / 2.0;
    double ratioa	= _matchScoreJoint(a);
    double ratiob	= _matchScoreJoint(b);
    double ratioc	= _matchScoreJoint(c);

    if (fabs(match - ratiob) < _tolerance) {
      /*
      printf("a = %10.6f : ratioa = %10.6f : b = %10.6f : ratiob = %10.6f : c = %10.6f : ratioc = %10.6f : match = %10.6f\n\n",
	     a, ratioa, b, ratiob, c, ratioc, match);
      printf("f_marginal = %10.6f : f_joint = %10.6f\n", f, b);
      */
      return b;
    }

    if (ratiob > match)
      a = b;		// ratio too high, take lower interval
    else		
      c = b;		// ratio too low, take upper interval
  }
}


// ----- Methods for class 'MutualInformationVADMetric' -----
//
MutualInformationVADMetric::
MutualInformationVADMetric(const VectorComplexFeatureStreamPtr& source1, const VectorComplexFeatureStreamPtr& source2,
			   const VectorFloatFeatureStreamPtr& spectralEstimator1, const VectorFloatFeatureStreamPtr& spectralEstimator2,
			   const String& shapeFactorFileName, double twiddle, double threshold, double beta,
			   double sampleRate, double lowCutoff, double highCutoff, const String& nm)
  : NegentropyVADMetric(source1, spectralEstimator1, shapeFactorFileName, /* threshold= */ 0.0,
			sampleRate, lowCutoff, highCutoff, nm),
    _source2(source2), _spectralEstimator2(spectralEstimator2),
    _crossCorrelations(_fftLen2 + 1), _fixedThreshold(_calcFixedThreshold()),
    _twiddle(twiddle), _threshold(threshold), _beta(beta) { }

MutualInformationVADMetric::~MutualInformationVADMetric() { }

void MutualInformationVADMetric::_initializePdfs()
{
  unsigned fbinX = 0;
  for (_GaussianListConstIterator itr = _generalizedGaussians.begin(); itr != _generalizedGaussians.end(); itr++) {
    // cout << "Initializing Bin " << fbinX << ":" << endl;
    const NegentropyVADMetric::_ComplexGeneralizedGaussianPtr& gg(*itr);
    _jointGeneralizedGaussians.push_back(_JointComplexGeneralizedGaussianPtr(new _JointComplexGeneralizedGaussian(gg)));
    fbinX++;
  }
}

// this is the fixed portion of the decision threshold
double MutualInformationVADMetric::_calcFixedThreshold()
{
  _initializePdfs();

  unsigned fbinX 		= 0;
  double   threshold		= 0.0;
  _GaussianListConstIterator mitr = _generalizedGaussians.begin();
  for (_JointGaussianListConstIterator itr = _jointGeneralizedGaussians.begin(); itr != _jointGeneralizedGaussians.end(); itr++) {
    _JointComplexGeneralizedGaussianPtr joint(*itr);
    _ComplexGeneralizedGaussianPtr marginal(*mitr);
    
    // marginal pdf contribution
    double f			= marginal->shapeFactor();
    double Bc2			= marginal->Bc() * marginal->Bc();
    double gamma2f		= gsl_sf_gamma(2.0 / f);
    double thresh		= 2.0 * ((2.0 / f) - log(f / (2.0 * M_PI * Bc2 * gamma2f)));

    // joint pdf contribution
    double fJ			= joint->shapeFactor();
    double BJ4			= pow(joint->Bc(), 4.0);
    double gamma4fJ		= gsl_sf_gamma(4.0 / fJ);
    thresh		       -= ((4.0 / fJ) - log(fJ / (8.0 * M_PI * M_PI * BJ4 * gamma4fJ)));

    if (fbinX >= _lowX && fbinX <= _highX) {
      if (fbinX == 0 || fbinX == (_fftLen2 + 1))
	threshold += thresh;
      else
	threshold += 2.0 * thresh;
    }

    fbinX++;  mitr++;
  }

  // don't normalize yet
  return threshold;
}

// complete threshold, fixed component and that depending on CC coefficients
double MutualInformationVADMetric::_calcTotalThreshold() const
{
  double totalThreshold = _fixedThreshold;
  for (unsigned fbinX = _lowX; fbinX <= _highX; fbinX++) {
    gsl_complex rho12	= _crossCorrelations[fbinX];
    double thresh	= - log(1.0 - gsl_complex_abs2(rho12));

    if (fbinX == 0 || fbinX == (_fftLen2 + 1))
      totalThreshold += thresh;
    else
      totalThreshold += 2.0 * thresh;
  }

  // now normalize the total threshold
  totalThreshold *= (_twiddle / _binN);

  return totalThreshold;
}

const double MutualInformationVADMetric::_epsilon = 0.10;

double MutualInformationVADMetric::calcMutualInformation(int frameX)
{
  const gsl_vector_complex* sample1   = _source->next(frameX);
  const gsl_vector_complex* sample2   = _source2->next(frameX);
  const gsl_vector_float*   envelope1 = _spectralEstimator->next(frameX);
  const gsl_vector_float*   envelope2 = _spectralEstimator2->next(frameX);
  unsigned fbinX 		= 0;
  double   mutualInformation	= 0.0;

#ifdef  _LOG_SAD_ 
  _frameX = frameX;
#endif /* _LOG_SAD_ */

  _GaussianListConstIterator mitr = _generalizedGaussians.begin();
  for (_JointGaussianListConstIterator itr = _jointGeneralizedGaussians.begin(); itr != _jointGeneralizedGaussians.end(); itr++) {
    _JointComplexGeneralizedGaussianPtr joint(*itr);
    _ComplexGeneralizedGaussianPtr marginal(*mitr);
    gsl_complex X1	= gsl_vector_complex_get(sample1, fbinX);
    gsl_complex X2	= gsl_vector_complex_get(sample2, fbinX);
    double      sigma1	= sqrt(gsl_vector_float_get(envelope1, fbinX));
    double      sigma2	= sqrt(gsl_vector_float_get(envelope2, fbinX));
    gsl_complex rho12	= _crossCorrelations[fbinX];

    // calculate empirical mutual information
    double jointLhood	= joint->logLhood(X1, X2, sigma1, sigma2, rho12);
    double marginal1	= marginal->logLhood(X1, sigma1);
    double marginal2	= marginal->logLhood(X2, sigma2);
    double mutual	= jointLhood - marginal1 - marginal2;

    /*
    if (fbinX == 15) {
      printf("(|X_1|, sigma1, |X_1| / sigma1) = (%12.4f, %12.4f, %12.4f)\n", gsl_complex_abs(X1), sigma1, gsl_complex_abs(X1) / sigma1);
      printf("(|X_2|, sigma2, |X_2| / sigma2) = (%12.4f, %12.4f, %12.4f)\n", gsl_complex_abs(X2), sigma2, gsl_complex_abs(X2) / sigma2);
      printf("rho12 = (%12.4f, %12.4f)\n", GSL_REAL(rho12), GSL_IMAG(rho12));
      printf("Mutual %12.4f\n", mutual);
      printf("Check here\n");
    } */

    if (fbinX >= _lowX && fbinX <= _highX) {
      if (fbinX == 0 || fbinX == (_fftLen2 + 1))
	mutualInformation += mutual;
      else
	mutualInformation += 2.0 * mutual;
    }

    // update cross-correlation coefficient for next frame
    gsl_complex cross	= gsl_complex_div_real(gsl_complex_mul(X1, gsl_complex_conjugate(X2)), sigma1 * sigma2);
    rho12		= gsl_complex_add(gsl_complex_mul_real(rho12, _beta), gsl_complex_mul_real(cross, 1.0 - _beta));
    if (gsl_complex_abs(rho12) >= (1.0 - _epsilon)) {
      // printf("Rescaling rho12 = (%12.4f, %12.4f) in bin %d\n", GSL_REAL(rho12), GSL_IMAG(rho12), fbinX);
      rho12 = gsl_complex_mul_real(rho12, ((1.0 - _epsilon) / gsl_complex_abs(rho12)));
    }
    _crossCorrelations[fbinX] = rho12;

    fbinX++;  mitr++;
  }

  mutualInformation /= _binN;

  // printf("Frame %d : MI = %12.4e\n", frameX, mutualInformation);

  return mutualInformation;
}

double MutualInformationVADMetric::next(int frameX)
{
  double threshold	= (_twiddle < 0.0) ? _threshold : _calcTotalThreshold();
  double mutual		= calcMutualInformation(frameX);

  // printf("Frame %d : Mutual Information %12.4f : Threshold %12.4f\n", frameX, mutual, threshold);
#ifdef  _LOG_SAD_ 
  _frameX = frameX;
#endif /* _LOG_SAD_ */

  if (mutual > threshold)
    return 1.0;
  return 0.0;
}

void MutualInformationVADMetric::reset()
{
  _source->reset();             _source2->reset();  
  _spectralEstimator->reset();  _spectralEstimator2->reset();
}

void MutualInformationVADMetric::nextSpeaker()
{
  cout << "MutualInformationVADMetric::nextSpeaker" << endl;

  // reset all cross correlations
  for (unsigned i = 0; i < _crossCorrelations.size(); i++)
    _crossCorrelations[i] = gsl_complex_rect(0.0, 0.0);
}

bool MutualInformationVADMetric::_aboveThreshold(int frameX)
{
  return next(frameX) > _threshold;
}


// ----- Methods for class 'LikelihoodRatioVADMetric' -----
//
LikelihoodRatioVADMetric::
LikelihoodRatioVADMetric(const VectorComplexFeatureStreamPtr& source1, const VectorComplexFeatureStreamPtr& source2,
			 const VectorFloatFeatureStreamPtr& spectralEstimator1, const VectorFloatFeatureStreamPtr& spectralEstimator2,
			 const String& shapeFactorFileName, double threshold,
			 double sampleRate, double lowCutoff, double highCutoff,
			 const String& nm)
  : NegentropyVADMetric(source1, spectralEstimator1, shapeFactorFileName, threshold,
			sampleRate, lowCutoff, highCutoff, nm),
    _source2(source2), _spectralEstimator2(spectralEstimator2) { }

LikelihoodRatioVADMetric::~LikelihoodRatioVADMetric() { }

double LikelihoodRatioVADMetric::calcLikelihoodRatio(int frameX)
{
  const gsl_vector_complex* sample1   = _source->next(frameX);
  const gsl_vector_complex* sample2   = _source2->next(frameX);
  const gsl_vector_float*   envelope1 = _spectralEstimator->next(frameX);
  const gsl_vector_float*   envelope2 = _spectralEstimator2->next(frameX);

  unsigned fbinX 		= 0;
  double likelihoodRatio	= 0.0;

#ifdef  _LOG_SAD_ 
  _frameX = frameX;
#endif /* _LOG_SAD_ */

  for (_GaussianListConstIterator itr = _generalizedGaussians.begin(); itr != _generalizedGaussians.end(); itr++) {
    _ComplexGeneralizedGaussianPtr marginal(*itr);
    gsl_complex X1	= gsl_vector_complex_get(sample1, fbinX);
    gsl_complex X2	= gsl_vector_complex_get(sample2, fbinX);
    // double      sigma1	= sqrt(gsl_vector_float_get(envelope1, fbinX));
    // double      sigma2	= sqrt(gsl_vector_float_get(envelope2, fbinX));
    // double	lr	= marginal->logLhood(X1, sigma1) - marginal->logLhood(X2, sigma2);

    double      sigma1	= gsl_vector_float_get(envelope1, fbinX);
    double      sigma2	= gsl_vector_float_get(envelope2, fbinX);
    double	sigma	= sqrt((sigma1 + sigma2) / 2);
    double	marg1	= marginal->logLhood(X1, sigma);
    double	marg2	= marginal->logLhood(X2, sigma);
    double	lr	= marg1 - marg2;

    if (fbinX == 4) {
      printf("(|X_1|, |X_1| / sigma) = (%12.4f, %12.4f)\n", gsl_complex_abs(X1), gsl_complex_abs(X1) / sigma);
      printf("(|X_2|, |X_2| / sigma) = (%12.4f, %12.4f)\n", gsl_complex_abs(X2), gsl_complex_abs(X2) / sigma);
      printf("Likelihood Ratio = %12.4f\n", lr);
      printf("Check here\n");
    }

    if (fbinX >= _lowX && fbinX <= _highX) {
      if (fbinX == 0 || fbinX == (_fftLen2 + 1))
	likelihoodRatio += lr;
      else
	likelihoodRatio += 2.0 * lr;
    }
    fbinX++;
  }
  likelihoodRatio /= _binN;

  printf("Frame %d : LR = %12.4e\n", frameX, likelihoodRatio);

  return likelihoodRatio;
}

double LikelihoodRatioVADMetric::next(int frameX)
{
  if (calcLikelihoodRatio(frameX) > _threshold)
    return 1.0;
  return 0.0;
}

void LikelihoodRatioVADMetric::reset()
{
  _source->reset();             _source2->reset();  
  _spectralEstimator->reset();  _spectralEstimator2->reset();
}

void LikelihoodRatioVADMetric::nextSpeaker()
{
  cout << "LikelihoodRatioVADMetric::nextSpeaker" << endl;
}


// ----- Methods for class 'LowFullBandEnergyRatioVADMetric' -----
//
LowFullBandEnergyRatioVADMetric::
LowFullBandEnergyRatioVADMetric(const VectorFloatFeatureStreamPtr& source, const gsl_vector* lowpass, double threshold, const String& nm)
  : _source(source), _lagsN(lowpass->size), _lowpass(gsl_vector_calloc(_lagsN)), _scratch(gsl_vector_calloc(_lagsN)),
    _autocorrelation(new double[_lagsN]), _covariance(gsl_matrix_calloc(_lagsN, _lagsN))
{
  gsl_vector_memcpy(_lowpass, lowpass);
}

LowFullBandEnergyRatioVADMetric::~LowFullBandEnergyRatioVADMetric()
{
  gsl_vector_free(_lowpass);  gsl_vector_free(_scratch);  delete[] _autocorrelation;  gsl_matrix_free(_covariance);
}

void LowFullBandEnergyRatioVADMetric::_calcAutoCorrelationVector(int frameX)
{
  const gsl_vector_float* samples = _source->next(frameX);
  unsigned		  sampleN = samples->size;

  for (unsigned lag = 0; lag < _lagsN; lag++) {
    double r_xx = 0.0;
    for (unsigned i = lag; i < sampleN; i++)
      r_xx += gsl_vector_float_get(samples, i) * gsl_vector_float_get(samples, i - lag);
    _autocorrelation[lag] = r_xx / (sampleN - lag);
  }
}

void LowFullBandEnergyRatioVADMetric::_calcCovarianceMatrix()
{
  for (unsigned rowX = 0; rowX < _lagsN; rowX++) {
    for (unsigned colX = rowX; colX < _lagsN; colX++) {
      gsl_matrix_set(_covariance, rowX, colX, _autocorrelation[colX - rowX]);
      gsl_matrix_set(_covariance, colX, rowX, _autocorrelation[colX - rowX]);
    }
  }
}

double LowFullBandEnergyRatioVADMetric::_calcLowerBandEnergy()
{
  double innerProduct;
  gsl_blas_dgemv(CblasNoTrans, 1.0, _covariance, _lowpass, 1.0, _scratch);
  gsl_blas_ddot(_lowpass, _scratch, &innerProduct);

  return innerProduct;
}

double LowFullBandEnergyRatioVADMetric::next(int frameX)
{
  _calcAutoCorrelationVector(frameX);
  _calcCovarianceMatrix();
  double le = _calcLowerBandEnergy();
  
  return le / _autocorrelation[0];
}

void LowFullBandEnergyRatioVADMetric::reset() { }

void LowFullBandEnergyRatioVADMetric::nextSpeaker() { }

bool LowFullBandEnergyRatioVADMetric::_aboveThreshold(int frameX) { return false; }


// ----- Methods for class 'HangoverVADFeature' -----
//
HangoverVADFeature::
HangoverVADFeature(const VectorFloatFeatureStreamPtr& source, const VADMetricPtr& metric, double threshold,
		   unsigned headN, unsigned tailN, const String& nm)
  : VectorFloatFeatureStream(source->size(), nm),
    _source(source), _recognizing(false), _headN(headN), _tailN(tailN),
    _buffer(NULL), _bufferIndex(0), _bufferedN(0),
    _aboveThresholdN(0), _belowThresholdN(0), _prefixN(0)
{
  _metricList.push_back(_MetricPair(metric, threshold));

  _buffer = new gsl_vector_float*[_headN];
  for (unsigned n = 0; n < _headN; n++)
    _buffer[n] = gsl_vector_float_calloc(_source->size());
}  

HangoverVADFeature::~HangoverVADFeature()
{
  for (unsigned n = 0; n < _headN; n++)
    gsl_vector_float_free(_buffer[n]);

  delete[] _buffer;
}

void HangoverVADFeature::reset()
{
  _source->reset();  VectorFloatFeatureStream::reset();

  _bufferIndex = _bufferedN = _aboveThresholdN = _belowThresholdN = _prefixN = 0;
  _recognizing = false;

  for (_MetricListIterator itr = _metricList.begin(); itr != _metricList.end(); itr++) {
    VADMetricPtr& metric((*itr).first);
    metric->reset();
  }
}

void HangoverVADFeature::nextSpeaker()
{
  _source->reset();  VectorFloatFeatureStream::reset();

  _bufferIndex = _bufferedN = _aboveThresholdN = _belowThresholdN = _prefixN = 0;
  _recognizing = false;

  for (_MetricListIterator itr = _metricList.begin(); itr != _metricList.end(); itr++) {
    VADMetricPtr& metric((*itr).first);
    metric->nextSpeaker();
  }
}

bool HangoverVADFeature::_aboveThreshold(int frameX)
{
  double sum = 0.0;
  _MetricListIterator itr = _metricList.begin();
  VADMetricPtr& metric((*itr).first);
  double threshold((*itr).second);
  double val = metric->next(frameX);

  return (val > threshold);
}

const gsl_vector_float* HangoverVADFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX){
    fprintf(stderr,"Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);
    throw jindex_error("Problem in Feature %s: %d != %d\n", name().c_str(), frameX - 1, _frameX);
  }

  // printf("HangoverVADFeature::next: FrameX = %d : Source FrameX = %d\n", frameX, _source->frameX()); fflush(stdout);
  if (_recognizing) {

    // use up the buffered blocks
    if (_bufferedN > 0) {
      const gsl_vector_float* vector = _buffer[_bufferIndex];
      _bufferIndex = (_bufferIndex + 1) % _headN;
      _bufferedN--;
      _increment();
      return vector;
    }

    // buffer is empty; take blocks directly from source
    const gsl_vector_float* vector = _source->next(frameX + prefixN());
    if (_aboveThreshold(frameX + prefixN())) {
      _belowThresholdN = 0;

      // printf("Decoding: FrameX = %d : Source FrameX = %d\n", frameX, _source->frameX()); fflush(stdout);

    } else {
      _belowThresholdN++;

      // printf("Tail: FrameX = %d : Source FrameX = %d : Below Threshold = %d\n", frameX, _source->frameX(), _belowThresholdN); fflush(stdout);

      if (_belowThresholdN == _tailN)
	throw jiterator_error("end of samples!");
    }
    _increment();
    return vector;

  } else {

    // buffer sample blocks until sufficient blocks have energy above the threshold
    while (true) {
      const gsl_vector_float* vector = _source->next(_prefixN);

      gsl_vector_float_memcpy(_buffer[_bufferIndex], vector);
      _bufferIndex = (_bufferIndex + 1) % _headN;
      _bufferedN = min(_headN, _bufferedN + 1);

      if (_aboveThreshold(_prefixN++)) {
	_aboveThresholdN++;

	// printf("FrameX = %d : Source FrameX = %d : Above Threshold = %d : Prefix = %d\n", frameX, _source->frameX(), _aboveThresholdN, _prefixN - 1);  fflush(stdout);

	if (_aboveThresholdN == _headN) {
	  _recognizing = true;
	  vector = _buffer[_bufferIndex];
	  _bufferIndex = (_bufferIndex + 1) % _headN;
	  _bufferedN--;
	  _increment();
	  return vector;
	}
      } else {

	// printf("Waiting : FrameX = %d : Source FrameX = %d : Prefix = %d\n", frameX, _source->frameX(), _prefixN - 1);  fflush(stdout);

	_aboveThresholdN = 0;
      }
    }
  }
}

// ----- Methods for class 'HangoverMIVADFeature' -----
//
HangoverMIVADFeature::
HangoverMIVADFeature(const VectorFloatFeatureStreamPtr& source,
		     const VADMetricPtr& energyMetric, const VADMetricPtr& mutualInformationMetric, const VADMetricPtr& powerMetric,
		     double energyThreshold, double mutualInformationThreshold, double powerThreshold,
		     unsigned headN, unsigned tailN, const String& nm)
  : HangoverVADFeature(source, energyMetric, energyThreshold, headN, tailN, nm)
{
  _metricList.push_back(_MetricPair(mutualInformationMetric, mutualInformationThreshold));
  _metricList.push_back(_MetricPair(powerMetric, powerThreshold));
}  

// ad hoc decision making process (should create a separate class for this)
bool HangoverMIVADFeature::_aboveThreshold(int frameX)
{
  VADMetricPtr& energyMetric            = _metricList[EnergyVADMetricX].first;
  VADMetricPtr& mutualInformationMetric = _metricList[MutualInformationVADMetricX].first;
  VADMetricPtr& likelihoodRatioMetric   = _metricList[LikelihoodRatioVADMetricX].first;

  if (energyMetric->next(frameX) < 0.5) {
    _decisionMetric = -1;
    mutualInformationMetric->next(frameX);
    likelihoodRatioMetric->next(frameX);
    return false;
  }

  if (mutualInformationMetric->next(frameX) < 0.5) {
    _decisionMetric = 2;
    likelihoodRatioMetric->next(frameX);
    return true;
  }

  if (likelihoodRatioMetric->next(frameX) > 0.5) {
    _decisionMetric = 3;
    return true;
  }

  _decisionMetric = -3;
  return false;
}

// ----- Methods for class 'HangoverMIVADFeature' -----
//
HangoverMultiStageVADFeature::
HangoverMultiStageVADFeature(const VectorFloatFeatureStreamPtr& source,
			     const VADMetricPtr& energyMetric, double energyThreshold, 
			     unsigned headN, unsigned tailN, const String& nm)
  : HangoverVADFeature(source, energyMetric, energyThreshold, headN, tailN, nm)
#ifdef _LOG_SAD_ 
  ,_scores(NULL)
#endif /* _LOG_SAD_ */
{
}  

HangoverMultiStageVADFeature::~HangoverMultiStageVADFeature()
{
#ifdef _LOG_SAD_ 
  if( NULL!=_scores ){
    gsl_vector_free( _scores );
  }
#endif /* _LOG_SAD_ */
}

// ad hoc decision making process (should create a separate class for this)
bool HangoverMultiStageVADFeature::_aboveThreshold(int frameX)
{
  if( _metricList.size() < 3 ){
    fprintf(stderr,"HangoverMultiStage::setMetric()\n");
    return false;
  }
#ifdef _LOG_SAD_ 
  if( NULL==_scores )
    initScores();
#endif /* _LOG_SAD_ */

  VADMetricPtr& energyMetric = _metricList[0].first; // the first stage

  if (energyMetric->next(frameX) < 0.5) {
    _decisionMetric = -1; // determine non-voice activity based on the energy measure.

    for(unsigned metricX=1;metricX<_metricList.size();metricX++){
      VADMetricPtr& vadMetricPtr = _metricList[metricX].first;
      vadMetricPtr->next(frameX);
    }
    return false;
  }

  for(unsigned stageX=1;stageX<_metricList.size();stageX++){
    VADMetricPtr& currentStageMetric = _metricList[stageX].first;
    
    if ( currentStageMetric->next(frameX) > 0.5 ){
      _decisionMetric = stageX + 1;
      for(unsigned metricX=2;metricX<_metricList.size();metricX++){
	VADMetricPtr& vadMetricPtr = _metricList[metricX].first;
	vadMetricPtr->next(frameX);
      }
      return true; // determine  voice activity.
    }
    else{
      _decisionMetric = - ( stageX + 1 );
    }
  }
  _decisionMetric = - ( _metricList.size() );

  return false;
}


#ifdef _LOG_SAD_

void HangoverMultiStageVADFeature::initScores()
{
  if( NULL==_scores ){
    _scores = gsl_vector_calloc( _metricList.size() );
  }
  for(unsigned metricX=0;metricX<_metricList.size();metricX++){
    VADMetricPtr& vadMetricPtr = _metricList[metricX].first;
    vadMetricPtr->initScore();
  }
}

gsl_vector *HangoverMultiStageVADFeature::getScores()
{
  for(unsigned metricX=0;metricX<_metricList.size();metricX++){
    VADMetricPtr& vadMetricPtr = _metricList[metricX].first;
    gsl_vector_set( _scores, metricX, vadMetricPtr->getAverageScore() );
  }
  
  return _scores;
}

#endif /* _LOG_SAD_ */

