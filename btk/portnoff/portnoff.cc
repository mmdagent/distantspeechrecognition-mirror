//
//                       Beamforming Toolkit
//                              (btk)
//
//  Module:  btk.portnoff
//  Purpose: Portnoff filter bank.
//  Author:  John McDonough.

#include <math.h>
#include "portnoff.h"
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>

#include "feature/feature.h"

// ----- members for class `LowPassImpulseResp' -----
//
LowPassImpulseResp::LowPassImpulseResp(unsigned blockLen, unsigned nBlocks,
				       bool useHammingWindow)
  : _blockLen(blockLen), _nBlocks(nBlocks), _filterLen(_blockLen * _nBlocks+1),
    _zeroX(_filterLen-1), _firstSample(1-_filterLen), _lastSample(_filterLen-1)
{
  // allocate and initialize analysis LPF impulse response
  //
  _lowPassFilter = new double[2*_filterLen-1];
  _lowPassFilter[_zeroX] = 1.0;
  for (unsigned n = 1; n < _filterLen; n++) {
    double val = sin (M_PI * n / _blockLen) / (M_PI * n / _blockLen);
    if ( useHammingWindow )
      val *= (0.54 + 0.46 * cos (M_PI * n / _lastSample));

    _lowPassFilter[_zeroX - n] = _lowPassFilter[_zeroX + n] = val;
  }
}

LowPassImpulseResp::~LowPassImpulseResp()
{
  delete _lowPassFilter;
}

void LowPassImpulseResp::print() const
{
  printf("\nLow-Pass Filter:\n");
  for (int i = _firstSample; i <= _lastSample; i++)
    printf("  %4d  %g\n", i, (*this)[i]);
  printf("\n");
}


// ----- members for class `AnalysisFilterBank' -----
//
PortnoffAnalysisBank::
PortnoffAnalysisBank(VectorShortFeatureStreamPtr& src,
		     unsigned fftLen, unsigned nBlocks, unsigned subSampRate,
		     const String& nm)
  : VectorComplexFeatureStream(fftLen, nm),
    _src(src),
    _fftLen(fftLen),   _fftLen2(fftLen/2),
    _nBlocks(nBlocks), _nBlocks2(nBlocks/2),
    _subSampRate(subSampRate),
    _sampleBuffer(_fftLen, _nBlocks+2),
    _spectralBuffer(_fftLen, _nBlocks, _subSampRate),
    _lpfImpulse(_fftLen, _nBlocks),
    _nextR(0)
{
  _fft = new double[_fftLen];
  _Skr = gsl_vector_complex_alloc(_fftLen);

  zeroAll();
}

PortnoffAnalysisBank::~PortnoffAnalysisBank()
{
  delete _fft;
  gsl_vector_complex_free(_Skr);
}

// reset all entries to zero to prepare for the next utterance
//
void PortnoffAnalysisBank::zeroAll()
{
  _nextR = -_nBlocks2 * _fftLen + _fftLen / _subSampRate;

  _sampleBuffer.zeroAll();
  _spectralBuffer.zeroAll();
}

void PortnoffAnalysisBank::print()
{
  _sampleBuffer.print();

  _spectralBuffer.print();
}

const gsl_vector_complex* PortnoffAnalysisBank::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  _increment();

  if ((_frameX % _subSampRate) == 0) {
    const gsl_vector_short* smp = _src->next(frameX);
    _sampleBuffer.nextBlock(smp);

    for (int s = 0;  s < int(_subSampRate); s++) {
      for (int m = 0; m < int(_fftLen); m++) {
	double sum = 0.0;
	for (int blk = -int(_nBlocks2); blk < int(_nBlocks2); blk++) {
	  sum += _sampleBuffer[blk * int(_fftLen) + m + _nextR] *
	    _lpfImpulse[-blk * int(_fftLen) - m];
	}

	_fft[(m+_nextR)%_fftLen] = sum;
      }

      gsl_fft_real_radix2_transform(_fft, /*stride=*/ 1, _fftLen);

      halfComplexUnpack(_Skr, _fft);
      _spectralBuffer.nextBlock(_Skr, _nextR);

      // printf("\nSkr %4d\n", _nextR);
      // gsl_vector_complex_fprintf(stdout, _Skr, "%10.4f");
      // printf("\n");

      _nextR  += (_fftLen / _subSampRate);
    }
  }

  // _spectralBuffer.print();

  const gsl_vector_complex* retVec = getBlock(_frameX * (_fftLen / _subSampRate));

  return retVec;
}


// ----- members for class `BaseSynthesisBank' -----
//
BaseSynthesisBank::
BaseSynthesisBank(unsigned fftLen, unsigned nBlocks,unsigned subSampRate)
  : _fftLen(fftLen), _fftLen2(fftLen/2),
    _nBlocks(nBlocks), _nBlocks2(nBlocks/2),
    _subSampRate(subSampRate),
    _nextR(0), _nextX(0),
    _lpfImpulse(fftLen / subSampRate, nBlocks * subSampRate,
		/*useHammingWindow=*/ false) { }

BaseSynthesisBank::~BaseSynthesisBank() { }

// reset all entries to zero to prepare for the next utterance
//
void BaseSynthesisBank::zeroAll()
{
  _nextR  = -_nBlocks2 * _fftLen + _fftLen / _subSampRate;
  _nextX  = 0;
}

// ----- members for class `PortnoffSynthesisBank' -----
//
PortnoffSynthesisBank::
PortnoffSynthesisBank(VectorComplexFeatureStreamPtr& src,
		      unsigned fftLen, unsigned nBlocks, unsigned subSampRate,
		      const String& nm)
  : BaseSynthesisBank(fftLen, nBlocks, subSampRate),
    VectorShortFeatureStream(fftLen / subSampRate, nm),
    _src(src),
    _smnBuffer(fftLen, 2*nBlocks, subSampRate),
    _xBuffer(fftLen / subSampRate,    nBlocks * subSampRate)
{
  _fft = new double[_fftLen];
  _x   = new short[_fftLen / _subSampRate];

  zeroAll();
}

PortnoffSynthesisBank::~PortnoffSynthesisBank()
{
  delete[] _fft;  delete[] _x;
}

// reset all entries to zero to prepare for the next utterance
//
void PortnoffSynthesisBank::zeroAll()
{
  BaseSynthesisBank::zeroAll();
  
  _smnBuffer.zeroAll();
  _xBuffer.zeroAll();
}

const gsl_vector_short* PortnoffSynthesisBank::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  do {
    const gsl_vector_complex* smp = _src->next(frameX);

    halfComplexPack(_fft, smp);
    gsl_fft_halfcomplex_radix2_inverse(_fft, /*stride=*/ 1, _fftLen);

    _smnBuffer.nextBlock(_fft, _nextR);

    // resynthesis loop
    //
    if (_nextR >= int(_nBlocks2 * _fftLen)) {
      for (int n = 0; n < int( _fftLen / _subSampRate ); n++) {
	double   sum = 0.0;
	unsigned m   = (n + _nextX) % _fftLen;
	// printf("synth. m = %4d\n", m); fflush(stdout);
	for (int r = -int(_nBlocks2 * _subSampRate);
	     r < int(_nBlocks2 * _subSampRate); r++) {
	  sum += _smnBuffer.smn(_nextX + r * int(_fftLen / _subSampRate), m)
	    * _lpfImpulse[n - (r * int(_fftLen / _subSampRate))];
	}
	_x[n] = short(sum);
      }
      _xBuffer.nextBlock(_x);
      _nextX += (_fftLen / _subSampRate);
    }

    // _xBuffer.print();

    _nextR   += (_fftLen / _subSampRate);

  } while (_nextR <= int(_nBlocks2 * _fftLen) );

  _increment();

  const gsl_vector_short* retVec = getBlock(_frameX * (_fftLen / _subSampRate));
  return retVec;
}


// ----- members for class `SpectralSynthesisBank' -----
//
SpectralSynthesisBank::
SpectralSynthesisBank(unsigned fftLen, unsigned nBlocks,
			    unsigned subSampRate, unsigned D)
  : BaseSynthesisBank(fftLen, nBlocks, subSampRate),
    _R(fftLen / subSampRate), _D(D), _R_upper(0), _R_lower(0),
    _spectralBuffer(_fftLen, _nBlocks, _subSampRate)
{
  _fft = new double[_fftLen];
  _x   = new double[_fftLen / _subSampRate];

  zeroAll();
}

SpectralSynthesisBank::~SpectralSynthesisBank()
{
  delete _fft;  delete _x;
}

// reset all entries to zero to prepare for the next utterance
//
void SpectralSynthesisBank::zeroAll()
{
  BaseSynthesisBank::zeroAll();
  
  _spectralBuffer.zeroAll();
}

void SpectralSynthesisBank::calcFmdr(int d)
{
  _R_upper = (int) (d+1)*_D / _R;
  _R_upper += _nBlocks2;

  _R_lower = (int) (d-1)*_D / _R;
  _R_lower += (1 - _nBlocks2);
}

// get the next spectral sample and perform synthesis
//
void SpectralSynthesisBank::
nextSpectralBlock(unsigned k, const gsl_complex e_krd,
		  const gsl_vector_complex* Z_k) { }


#ifdef MAIN

#include <stdio.h>
#include <iostream.h>

int main(int argc, char** argv)
{
  unsigned fftLen      = 64;
  unsigned fftLen2     = fftLen/2;
  unsigned nBlocks     = 16;
  unsigned nBlocks2    = nBlocks/2;
  unsigned subSampRate = 4;
  double*  sample      = new double[fftLen];

  PortnoffAnalysisBank  anal(fftLen, nBlocks, subSampRate);
  PortnoffSynthesisBank synth(fftLen, nBlocks, subSampRate);

  int nextR  = -nBlocks2 * fftLen + fftLen / subSampRate;
  int nextX  = 0;

  unsigned cnt = 0;
  for (unsigned i = 0; i < 40; i++) {
    for (unsigned j = 0; j < fftLen; j++)
      sample[j] =  (cnt++) % 15;
      // sample[j] = cos((2.0 * M_PI * cnt++) / (fftLen/4));
    anal.nextSampleBlock(sample);

    for (int s = 0;  s < int(subSampRate); s++) {
      const gsl_vector_complex* nextSpecSample = anal.getBlock(nextR);
      synth.nextSpectralBlock(nextSpecSample);

      if ( nextR >= int(nBlocks2 * fftLen) ) {
	const gsl_vector* nextSpeechSample = synth.getBlock(nextX);
	printf("Resynthesized Speech Samples:\n");
	for (unsigned j = 0; j < fftLen / subSampRate; j++)
	  printf("  %4d. %10.4f\n",
		 nextX+j, gsl_vector_get(nextSpeechSample, j));
	nextX += fftLen / subSampRate;
      }

      nextR += fftLen / subSampRate;
    }
  }

  delete sample;

  return 0;
}

#endif // SWIG
