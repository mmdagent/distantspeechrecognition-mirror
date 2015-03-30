//
//                       Beamforming Toolkit
//                              (btk)
//
//  Module:  btk.sad
//  Purpose: Speech activity detection.
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

#include "sad/sadFeature.h"


// ----- define auxiliary functions -----
//
static float norm(float* vec, int n) {
  double norm = 0.0;
  for (unsigned i = 0; i < n; i++)
    norm += vec[i] * vec[i];

  return sqrt(norm);
}

static void normalize(float* vec, int n) {
  double sigma = norm(vec, n);
  for (unsigned i = 0; i < n; i++)
    vec[i] /= sigma;
}


// ----- methods for class `BrightnessFeature' -----
//
BrightnessFeature::BrightnessFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, const String& nm)
  : VectorFloatFeatureStream( /* sz= */ 1, nm),
    _src(src), _sampleRate(sampleRate), _max(_sampleRate / 2.0), _med(_max / 2.0), _df(_max / _src->size()), _frs(new float[_src->size()])
{
  for (unsigned i = 0; i < _src->size(); i++)
    _frs[i] = _df * (float) (i + 1);
}

BrightnessFeature::~BrightnessFeature() { delete[] _frs; }

const gsl_vector_float* BrightnessFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem: %d != %d\n", frameX - 1, _frameX);

  const gsl_vector_float* block = _src->next(frameX);

  float n = 0.0f;
  float d = 0.0f;
  for (unsigned j = 0; j < _src->size(); j++) {

    if (_weight)
      n += _frs[j] * gsl_vector_float_get(block, j);
    else
      n += j * gsl_vector_float_get(block, j);

    d += gsl_vector_float_get(block, j);

    float val = n / d;
    if (!_weight)
      val /= _src->size();

    gsl_vector_float_set(_vector, j, val);
  }

  _increment();
  return _vector;
}


// ----- methods for class `EnergyDiffusionFeature' -----
//
EnergyDiffusionFeature::EnergyDiffusionFeature(const VectorFloatFeatureStreamPtr& src, const String& nm)
  : VectorFloatFeatureStream( /* sz= */ 1, nm), _src(src) { }

EnergyDiffusionFeature::~EnergyDiffusionFeature() { }

const gsl_vector_float* EnergyDiffusionFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem: %d != %d\n", frameX - 1, _frameX);

  const gsl_vector_float* block = _src->next(frameX);

  double norm = 0.0;
  for (unsigned j = 0; j < _src->size(); j++) {
    double val = gsl_vector_float_get(block, j);
    norm += val * val;
  }

  norm = sqrt(norm);
  double diff = 0.0;
  for (unsigned j = 0; j < _src->size(); j++) {
    double nval =  gsl_vector_float_get(block, j) / norm;
    diff -= (nval > 0.0 ? nval * log10(nval) : 0.0);
  }
  gsl_vector_float_set(_vector, 0, diff);

  _increment();
  return _vector;
}


// ----- methods for class `BandEnergyRatioFeature' -----
//
BandEnergyRatioFeature::BandEnergyRatioFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, float threshF, const String& nm)
  : VectorFloatFeatureStream( /* sz= */ 1, nm), _src(src), _sampleRate(sampleRate), _max(_sampleRate / 2.0),
    _df(_max / _src->size()), _threshF((threshF > 0.0) ? threshF : _max / 2.0f), _threshX(int(floor(_threshF / _df))) { }

BandEnergyRatioFeature::~BandEnergyRatioFeature() { }

const gsl_vector_float* BandEnergyRatioFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem: %d != %d\n", frameX - 1, _frameX);

  const gsl_vector_float* block = _src->next(frameX);

  float ssLow  = 0.0f;
  float ssHigh = 0.0f;
  for (unsigned j = 0; j < _threshX; j++) {
    float val = gsl_vector_float_get(block, j);
    ssLow += val * val;
  }

  for (unsigned j = _threshX; j < _src->size(); j++) {
    float val = gsl_vector_float_get(block, j);
    ssHigh += val * val;
  }
  gsl_vector_float_set(_vector, 0, sqrt(ssLow / ssHigh));

  _increment();
  return _vector;
}


// ----- methods for class `NormalizedFluxFeature' -----
//
NormalizedFluxFeature::NormalizedFluxFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, float threshF, const String& nm)
  : VectorFloatFeatureStream( /* sz= */ 1, nm), _src(src), _win0(new float[_src->size()]), _win1(new float[_src->size()]) { }

NormalizedFluxFeature::~NormalizedFluxFeature() { delete[] _win0; delete[] _win1; }

const gsl_vector_float* NormalizedFluxFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem: %d != %d\n", frameX - 1, _frameX);

  const gsl_vector_float* block = _src->next(frameX);

  if (frameX == 0) {
    memcpy(_win0, block->data, _src->size() * sizeof(float));
    normalize(_win0, _src->size());

    gsl_vector_float_set(_vector, 0, 0.0);
    _increment();
    return _vector;
  }
    
  memcpy(_win1, block->data, _src->size() * sizeof(float));
  normalize(_win1, _src->size());

  double sum = 0.0;
  for (unsigned j = 0; j < _src->size(); j++) {
    float diff = _win0[j] - _win1[j];
    sum += diff * diff;
  }
  gsl_vector_float_set(_vector, 0, sqrt(sum));
    
  memcpy(_win1, _win0, _src->size() * sizeof(float));

  _increment();
  return _vector;
}


// ----- methods for class `NegativeEntropyFeature' -----
//
NegativeEntropyFeature::NegativeEntropyFeature(const VectorFloatFeatureStreamPtr& src, const String& nm)
  : VectorFloatFeatureStream( /* sz= */ 1, nm), _src(src), _win(new float[_src->size()]) { }

NegativeEntropyFeature::~NegativeEntropyFeature() { delete[] _win; }

const gsl_vector_float* NegativeEntropyFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem: %d != %d\n", frameX - 1, _frameX);

  const gsl_vector_float* block = _src->next(frameX);

  // half-wave rectify
  for (unsigned j = 0; j < _src->size(); j++) {
    float val = gsl_vector_float_get(block, j);
    _win[j] = (val < 0 ? -val : val);
  }

  // normalize window to 0 mean, unity variance
  double sum  = 0.0;
  double sumS = 0.0;
  for (unsigned j = 0; j < _src->size(); j++) {
    sum  += _win[j];
    sumS += _win[j] * _win[j];
  }
  double mean = sum / _src->size();
  double dev  = sqrt((sumS / (_src->size() - 1)) - (mean * mean));
  for (unsigned j = 0; j < _src->size(); j++)
    _win[j] = (_win[j] - mean) / dev;

  // calculate E(G(y)), where G(u) = ln cosh(u) */
  sum = 0.0;
  for (unsigned j = 0; j < _src->size(); j++)
    sum += log(cosh(_win[j]));

  double EGy = sum / _src->size();
  static const double EGgy = 0.374576;		// according to mathematica
  gsl_vector_float_set(_vector, 0, 100.0 * (EGy - EGgy) * (EGy - EGgy));

  _increment();
  return _vector;
}


// ----- methods for class `SignificantSubbandsFeature' -----
//
SignificantSubbandsFeature::SignificantSubbandsFeature(const VectorFloatFeatureStreamPtr& src, float thresh, const String& nm)
  : VectorFloatFeatureStream( /* sz= */ 1, nm), _src(src), _thresh(thresh), _win(new float[_src->size()]) { }

SignificantSubbandsFeature::~SignificantSubbandsFeature() { delete[] _win; }

const gsl_vector_float* SignificantSubbandsFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem: %d != %d\n", frameX - 1, _frameX);

  unsigned dimN = _src->size();
  const gsl_vector_float* block = _src->next(frameX);

  memcpy(_win, block->data, dimN * sizeof(float));
  normalize(_win, dimN);
  double sum = 0.0;
  for (unsigned j = 0; j < dimN; j++)
    if (_win[j] > _thresh)
      sum += 1.0f;

  gsl_vector_float_set(_vector, 0, sum);

  _increment();
  return _vector;
}


// ----- methods for class `NormalizedBandwidthFeature' -----
//
NormalizedBandwidthFeature::NormalizedBandwidthFeature(const VectorFloatFeatureStreamPtr& src, float sampleRate, float thresh, const String& nm)
  : VectorFloatFeatureStream( /* sz= */ 1, nm), _src(src), _thresh(thresh), _sampleRate(sampleRate), _df((_sampleRate / 2.0f) / _src->size()),
    _frs(new float[_src->size()]), _win(new float[_src->size()])
{
  for (unsigned i = 0; i < _src->size(); i++) {
    _frs[i] = _df * (i + 1);
  }
}

NormalizedBandwidthFeature::~NormalizedBandwidthFeature() { delete[] _frs; delete[] _win; }

const gsl_vector_float* NormalizedBandwidthFeature::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if (frameX >= 0 && frameX - 1 != _frameX)
    throw jindex_error("Problem: %d != %d\n", frameX - 1, _frameX);

  unsigned dimN = _src->size();
  const gsl_vector_float* block = _src->next(frameX);

  memcpy(_win, block->data, dimN * sizeof(float));
  normalize(_win, dimN);

  int min = dimN;
  for (unsigned j = 0; j < dimN && min == dimN; j++) {
    if (_win[j] > _thresh) min = j;
  }
  int max = 0;
  for (int j = dimN - 1; j >= min && max == 0; j--) {
    if (_win[j] > _thresh) max = j;
  }
  gsl_vector_float_set(_vector, 0, _frs[max] - _frs[min]);

  _increment();
  return _vector;
}
