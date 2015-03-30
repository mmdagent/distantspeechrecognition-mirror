//                              -*- C++ -*-
//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.portnoff
//  Purpose: Portnoff filter bank.
//  Author:  John McDonough.

#ifndef _portnoff_h_
#define _portnoff_h_

#include <stdio.h>
#include <assert.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include "common/jexception.h"

#include "stream/stream.h"

// ----- definition for class template `CircularBuffer' -----
// 
template <class GSLType, class Type>
class CircularBuffer {
 public:
  CircularBuffer(unsigned blockLen, unsigned nBlocks,
		 unsigned subSampRate = 1);
  ~CircularBuffer();

  void  zeroAll();
  void  nextBlock(const GSLType* smp);
  void  nextBlock(const GSLType* smp, int blkX);

  void  nextBlock(const Type* smp);
  void  nextBlock(const Type* smp, int blkX);

  int   getBlockIndex(int blkX) const;
  GSLType* getBlock(int blkX) const;

  inline Type operator[](int n) const;
  inline Type smn(int blkX, unsigned sampleX) const;

  void print() const;

 private:
  inline int firstSample() const;
  inline int lastSample()  const;

  inline int firstBlock() const;
  inline int lastBlock()  const;

  GSLType* alloc(unsigned blockLen);
  void free(GSLType* tp);
  void zero(GSLType* tp);
  void copy(GSLType* tgt, const GSLType* src);

  void increment();

  const unsigned	_blockLen;
  const unsigned	_nBlocks;
  const unsigned	_nBlocks2;
  const unsigned	_subSampRate;
  GSLType**		_blocks;
  unsigned		_zeroBlock;
  unsigned		_nextBlock;
  int			_sampleAtZero;
};

// Hide this from SWIG
#ifndef SWIGPYTHON

// ----- members for class template `CircularBuffer' -----
//
template <class GSLType, class Type>
CircularBuffer<GSLType, Type>::CircularBuffer(unsigned blockLen, unsigned nBlocks,
					      unsigned subSampRate)
  : _blockLen(blockLen), _nBlocks(nBlocks), _nBlocks2(nBlocks/2),
    _subSampRate(subSampRate)
{
  _blocks = new GSLType*[_nBlocks * _subSampRate];
  for (unsigned n = 0; n < _nBlocks * _subSampRate; n++)
    _blocks[n] = alloc(_blockLen);

  zeroAll();
}

// here follow many template specializations; they would not be
// necessary if GSL were written in C++ 
//
template<>
gsl_vector* CircularBuffer<gsl_vector, double>::alloc(unsigned blockLen)
{
  return gsl_vector_alloc(blockLen);
}

template<>
void CircularBuffer<gsl_vector, double>::free(gsl_vector* tp)
{
  gsl_vector_free(tp);
}

template<>
void CircularBuffer<gsl_vector, double>::zero(gsl_vector* tp)
{
  gsl_vector_set_zero(tp);
}

template<>
void CircularBuffer<gsl_vector, double>::copy(gsl_vector* tgt, const gsl_vector* src)
{
  gsl_vector_memcpy(tgt, src);
}

template<>
gsl_vector_complex*
CircularBuffer<gsl_vector_complex, double>::alloc(unsigned blockLen)
{
  return gsl_vector_complex_alloc(blockLen);
}

template<>
void CircularBuffer<gsl_vector_complex, double>::free(gsl_vector_complex* tp)
{
  gsl_vector_complex_free(tp);
}

template<>
void CircularBuffer<gsl_vector_complex, double>::zero(gsl_vector_complex* tp)
{
  gsl_vector_complex_set_zero(tp);
}

template<>
void CircularBuffer<gsl_vector_complex, double>::
copy(gsl_vector_complex* tgt, const gsl_vector_complex* src)
{
  gsl_vector_complex_memcpy(tgt, src);
}

template<>
gsl_vector_short* CircularBuffer<gsl_vector_short, short>::alloc(unsigned blockLen)
{
  return gsl_vector_short_alloc(blockLen);
}

template<>
void CircularBuffer<gsl_vector_short, short>::free(gsl_vector_short* tp)
{
  gsl_vector_short_free(tp);
}

template<>
void CircularBuffer<gsl_vector_short, short>::zero(gsl_vector_short* tp)
{
  gsl_vector_short_set_zero(tp);
}

template<>
void CircularBuffer<gsl_vector_short, short>::copy(gsl_vector_short* tgt, const gsl_vector_short* src)
{
  gsl_vector_short_memcpy(tgt, src);
}

template <class GSLType, class Type>
CircularBuffer<GSLType, Type>::~CircularBuffer()
{
  for (unsigned n = 0; n < _nBlocks * _subSampRate; n++)
    free(_blocks[n]);

  delete _blocks;
}

// reset to zero to get ready for new utterance
//
template <class GSLType, class Type>
void CircularBuffer<GSLType, Type>::zeroAll()
{
  _sampleAtZero = -(_nBlocks2 * _blockLen);
  _zeroBlock    = _nBlocks2 * _subSampRate;
  _nextBlock    = 0;
  for (unsigned n = 0; n < _nBlocks * _subSampRate; n++)
    zero(_blocks[n]);
}

// store next block of samples in circular buffer
//
template<>
void CircularBuffer<gsl_vector_complex, double>::nextBlock(const gsl_vector_complex* smp)
{
  gsl_vector_complex_memcpy(_blocks[_nextBlock], smp);

  increment();
}

// store next block of samples in circular buffer
//
template<>
void CircularBuffer<gsl_vector_complex, double>::nextBlock(const gsl_vector_complex* smp, int blkX)
{
  gsl_vector_complex_memcpy(_blocks[getBlockIndex(blkX)], smp);

  increment();
}

// store next block of samples in circular buffer
//
template<>
void CircularBuffer<gsl_vector_short, short>::nextBlock(const gsl_vector_short* smp)
{
  gsl_vector_short_memcpy(_blocks[_nextBlock], smp);

  increment();
}

// store next block of samples in circular buffer
//
template<>
void CircularBuffer<gsl_vector_short, short>::nextBlock(const gsl_vector_short* smp, int blkX)
{
  gsl_vector_short_memcpy(_blocks[getBlockIndex(blkX)], smp);

  increment();
}

// store next block of samples in circular buffer
//
template <class GSLType, class Type>
void CircularBuffer<GSLType, Type>::nextBlock(const Type* smp)
{
  copy(_blocks[_nextBlock], smp);

  increment();
}

// store next block of samples in circular buffer
//
template <class GSLType, class Type>
void CircularBuffer<GSLType, Type>::nextBlock(const Type* smp, int blkX)
{
  copy(_blocks[getBlockIndex(blkX)], smp);

  increment();
}

// store next block of samples in circular buffer
//
template<>
void CircularBuffer<gsl_vector, double>::nextBlock(const double* smp)
{
  gsl_vector* next = _blocks[_nextBlock];
  for (unsigned n = 0; n < _blockLen; n++)
    gsl_vector_set(next, n, smp[n]);

  increment();
}

// store next block of samples in circular buffer
//
template<>
void CircularBuffer<gsl_vector, double>::nextBlock(const double* smp, int blkX)
{
  gsl_vector* next = _blocks[getBlockIndex(blkX)];
  for (unsigned n = 0; n < _blockLen; n++)
    gsl_vector_set(next, n, smp[n]);

  increment();
}

// store next block of samples in circular buffer
//
template<>
void CircularBuffer<gsl_vector_short, short>::nextBlock(const short* smp)
{
  gsl_vector_short* next = _blocks[_nextBlock];
  for (unsigned n = 0; n < _blockLen; n++)
    gsl_vector_short_set(next, n, smp[n]);

  increment();
}

// store next block of samples in circular buffer
//
template<>
void CircularBuffer<gsl_vector_short, short>::nextBlock(const short* smp, int blkX)
{
  gsl_vector_short* next = _blocks[getBlockIndex(blkX)];
  for (unsigned n = 0; n < _blockLen; n++)
    gsl_vector_short_set(next, n, smp[n]);

  increment();
}

template <class GSLType, class Type>
void CircularBuffer<GSLType, Type>::increment()
{
  _sampleAtZero += _blockLen / _subSampRate;
  _zeroBlock     = (++_zeroBlock) % (_nBlocks * _subSampRate);
  _nextBlock     = (++_nextBlock) % (_nBlocks * _subSampRate);
}

template <class GSLType, class Type>
int CircularBuffer<GSLType, Type>::firstSample() const {
  return _sampleAtZero - (_nBlocks2 * _blockLen / _subSampRate);
}

template <class GSLType, class Type>
int CircularBuffer<GSLType, Type>::lastSample() const {
  return _sampleAtZero + ((_nBlocks2 * _blockLen / _subSampRate) - 1);
}

template <class GSLType, class Type>
int CircularBuffer<GSLType, Type>::firstBlock() const {
  return _sampleAtZero - (_nBlocks2 * _blockLen);
}

template <class GSLType, class Type>
int CircularBuffer<GSLType, Type>::lastBlock() const {
  return _sampleAtZero + ((_nBlocks2 * _blockLen) - (_blockLen / _subSampRate));
}

template<>
int CircularBuffer<gsl_vector_complex, double>::firstBlock() const {
  return _sampleAtZero - (_nBlocks2 * _blockLen);
}

template<>
int CircularBuffer<gsl_vector_complex, double>::lastBlock() const {
  return _sampleAtZero + ((_nBlocks2 * _blockLen) - (_blockLen / _subSampRate));
}

template<>
int CircularBuffer<gsl_vector_short, short>::firstBlock() const {
  return _sampleAtZero - (_nBlocks2 * _blockLen);
}

template<>
int CircularBuffer<gsl_vector_short, short>::lastBlock() const {
  return _sampleAtZero + ((_nBlocks2 * _blockLen) - (_blockLen / _subSampRate));
}

template <class GSLType, class Type>
Type CircularBuffer<GSLType, Type>::operator[](int n) const {
  assert( _subSampRate == 1 );
  assert( n >= firstSample() && n <= lastSample() );

  unsigned sampleX =
    (n - _sampleAtZero) % _blockLen;

  int blockX = (_blockLen * _nBlocks + n - _sampleAtZero) / _blockLen;

  blockX = (blockX + _zeroBlock) % _nBlocks;

  // printf(" block index %d\n", blockX);

  return gsl_vector_get(_blocks[blockX], sampleX);
}

template<>
short CircularBuffer<gsl_vector_short, short>::operator[](int n) const {
  assert( _subSampRate == 1 );
  assert( n >= firstSample() && n <= lastSample() );

  unsigned sampleX =
    (n - _sampleAtZero) % _blockLen;

  int blockX = (_blockLen * _nBlocks + n - _sampleAtZero) / _blockLen;

  blockX = (blockX + _zeroBlock) % _nBlocks;

  // printf(" block index %d\n", blockX);

  return gsl_vector_short_get(_blocks[blockX], sampleX);
}

template<>
void CircularBuffer<gsl_vector, double>::print() const
{
  printf("\nSample Buffer:\n");
  for (int i = firstSample(); i <= lastSample(); i++)
    printf("  %4d  %g\n", i, (*this)[i]);
  printf("\n");  fflush(stdout);
}

template<>
void CircularBuffer<gsl_vector_complex, double>::print() const
{
  printf("\nSpectral Buffer:\n");
  for (int b = firstBlock(); b <= lastBlock(); b += (_blockLen/_subSampRate)) {
    printf("Block %d:\n", b);
    gsl_vector_complex* block = getBlock(b);
    gsl_vector_complex_fprintf(stdout, block, "%10.4f");
  }
  printf("\n");  fflush(stdout);
}

template<>
void CircularBuffer<gsl_vector_short, short>::print() const
{
  printf("\nSample Buffer:\n");
  for (int i = firstSample(); i <= lastSample(); i++)
    printf("  %4d  %d\n", i, (*this)[i]);
  printf("\n");  fflush(stdout);
}

template <class GSLType, class Type>
Type CircularBuffer<GSLType, Type>::smn(int blkX, unsigned sampleX) const {
  assert( sampleX < _blockLen );
  assert( blkX >= firstBlock() && blkX <= lastBlock() );

  int blockX = (blkX - _sampleAtZero) / int(_blockLen / _subSampRate);

  blockX = (_nBlocks * _subSampRate + blockX + _zeroBlock)
    % (_nBlocks * _subSampRate);

  return gsl_vector_get(_blocks[blockX], sampleX);
}

#endif //SWIG

template <class GSLType, class Type>
GSLType* CircularBuffer<GSLType, Type>::getBlock(int blkX) const {
  return _blocks[getBlockIndex(blkX)];
}

template <class GSLType, class Type>
int CircularBuffer<GSLType, Type>::getBlockIndex(int blkX) const {
  assert( blkX >= firstBlock() && blkX <= lastBlock() );

  int blockX = (blkX - _sampleAtZero) / int(_blockLen / _subSampRate);

  // printf(" block index1 %d\n", blockX);

  blockX = (_nBlocks * _subSampRate + blockX + _zeroBlock)
    % (_nBlocks * _subSampRate);

  // printf(" _zeroBlock = %d\n", _zeroBlock);
  // printf(" block index2 %d\n", blockX);

  return blockX;
}

typedef CircularBuffer<gsl_vector, double>		SampleFloatBuffer;
typedef CircularBuffer<gsl_vector_short, short>		SampleBuffer;
typedef CircularBuffer<gsl_vector_complex, double>	SpectralBuffer;


// ----- definition for class `LowPassImpulseResp' -----
// 
class LowPassImpulseResp {
 public:
  LowPassImpulseResp(unsigned blockLen, unsigned nBlocks,
		     bool useHammingWindow = true);
  ~LowPassImpulseResp();

  void print() const;
  inline double operator[](int n) const;

 private:
  const unsigned	_blockLen;
  const unsigned	_nBlocks;
  const unsigned	_filterLen;

  const unsigned	_zeroX;
  const int		_firstSample;
  const int		_lastSample;

  double*		_lowPassFilter;
};

double LowPassImpulseResp::operator[](int n) const {
  assert(n >= _firstSample && n <= _lastSample);
  return _lowPassFilter[n + _zeroX];
}


// ----- definition for class `PortnoffAnalysisBank' -----
// 
class PortnoffAnalysisBank : public VectorComplexFeatureStream {
 public:
  PortnoffAnalysisBank(VectorShortFeatureStreamPtr& src,
		       unsigned fftLen = 256, unsigned nBlocks = 6, unsigned subSampRate = 2,
		       const String& nm = "PortnoffAnalysisBank");
  ~PortnoffAnalysisBank();

  inline const gsl_vector_complex* getBlock(int blkX) const;

  void zeroAll();

  void print();

  virtual const gsl_vector_complex* next(int frameX = -5);

  virtual void reset() { _src->reset();  zeroAll();  VectorComplexFeatureStream::reset(); }

  unsigned fftLen()      const { return _fftLen;      }
  unsigned fftLen2()     const { return _fftLen2;     }
  unsigned nBlocks()     const { return _nBlocks;     }
  unsigned nBlocks2()    const { return _nBlocks2;    }
  unsigned subSampRate() const { return _subSampRate; }

 private:
  VectorShortFeatureStreamPtr			_src;
  const unsigned				_fftLen;
  const unsigned				_fftLen2;
  const unsigned				_nBlocks;
  const unsigned				_nBlocks2;
  const unsigned				_subSampRate;

  SampleBuffer					_sampleBuffer;
  SpectralBuffer				_spectralBuffer;
  LowPassImpulseResp				_lpfImpulse;
  double*					_fft;

  gsl_vector_complex*				_Skr;
  int						_nextR;
};

const gsl_vector_complex* PortnoffAnalysisBank::getBlock(int blkX) const {
  return _spectralBuffer.getBlock(blkX);
}

typedef Inherit<PortnoffAnalysisBank, VectorComplexFeatureStreamPtr> PortnoffAnalysisBankPtr;


// ----- definition for class `BaseSynthesisBank' -----
// 
class BaseSynthesisBank {
 public:
  void zeroAll();

 protected:
  BaseSynthesisBank(unsigned fftLen = 256, unsigned nBlocks = 6,
		      unsigned subSampRate = 2);
  ~BaseSynthesisBank();

  const unsigned	_fftLen;
  const unsigned	_fftLen2;
  const unsigned	_nBlocks;
  const unsigned	_nBlocks2;
  const unsigned	_subSampRate;

  int			_nextR;
  int			_nextX;

  LowPassImpulseResp	_lpfImpulse;
};


// ----- definition for class `PortnoffSynthesisBank' -----
// 
class PortnoffSynthesisBank :
private BaseSynthesisBank, public VectorShortFeatureStream {
 public:
  PortnoffSynthesisBank(VectorComplexFeatureStreamPtr& src,
			unsigned fftLen = 256, unsigned nBlocks = 6, unsigned subSampRate = 2,
			const String& nm = "PortnoffSynthesisBank");
  ~PortnoffSynthesisBank();

  inline const gsl_vector_short* getBlock(int blkX) const;

  void zeroAll();

  virtual const gsl_vector_short* next(int frameX = -5);

  virtual void reset() { _src->reset();  zeroAll();  VectorShortFeatureStream::reset(); }

 private:
  VectorComplexFeatureStreamPtr			_src;
  SampleFloatBuffer				_smnBuffer;
  SampleBuffer					_xBuffer;

  double*					_fft;
  short*					_x;
};

const gsl_vector_short* PortnoffSynthesisBank::getBlock(int blkX) const {
  return _xBuffer.getBlock(blkX);
}

typedef Inherit<PortnoffSynthesisBank, VectorShortFeatureStreamPtr> PortnoffSynthesisBankPtr;


// ----- definition for class `SpectralSynthesisBank' -----
// 
class SpectralSynthesisBank : private BaseSynthesisBank {
 public:
  SpectralSynthesisBank(unsigned fftLen = 256, unsigned nBlocks = 6,
			      unsigned subSampRate = 2, unsigned D = 160);
  ~SpectralSynthesisBank();

  inline const gsl_vector_complex* getBlock(int blkX) const;

  void nextSpectralBlock(unsigned k, const gsl_complex e_krd,
			 const gsl_vector_complex* Z_k);

  void zeroAll();

 private:
  void calcFmdr(int d);

  const unsigned _R;
  const unsigned _D;
  int _R_upper;
  int _R_lower;
  SpectralBuffer	_spectralBuffer;

  double*		_fft;
  double*		_x;
};

const gsl_vector_complex* SpectralSynthesisBank::getBlock(int blkX) const {
  return _spectralBuffer.getBlock(blkX);
}

#endif // _portnoff_h_
