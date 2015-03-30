
//
//                       Beamforming Toolkit
//                              (btk)
//
//  Module:  btk.TDEstimator
//  Purpose: 
//  Author:  ABC
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

#include "CCTDE.h"
#include <gsl/gsl_blas.h>
#include <matrix/blas1_c.H>
#include <matrix/linpack_c.H>


static unsigned getFFTLen( unsigned tmpi )
{/* calculate the FFT  length */
  unsigned int pp = 1; // 2^i
  for(unsigned i=0; pp < tmpi;i++){
    pp = pp * 2;
  }
  
  return pp;
}

/*
	@brief calculate the time delay of arrival (TDOA) between two sounds.
	@param const String& fn1[in] file name of the first sound 
	@param const String& fn2[in] file name of the second sound
	@param unsigned nHeldMaxCC[in] the number of the max cross-correlation values held during the process
*/
CCTDE::CCTDE( SampleFeaturePtr& samp1, SampleFeaturePtr& samp2, int fftLen, unsigned nHeldMaxCC, int freqLowerLimit, int freqUpperLimit, const String& nm )
:VectorFeatureStream(nHeldMaxCC, nm),_nHeldMaxCC(nHeldMaxCC),_freqLowerLimit(_freqLowerLimit),_freqUpperLimit(_freqUpperLimit)
{
  _sampleDelays = new unsigned[_nHeldMaxCC];
  _ccValues     = new double[_nHeldMaxCC];

  if( samp1->getSampleRate() != samp2->getSampleRate() ){
    printf("The sampling rates must be the same but %d != %d\n", samp1->getSampleRate(), samp2->getSampleRate() );
    throw jdimension_error("The sampling rates must be the same but %d != %d\n", samp1->getSampleRate(), samp2->getSampleRate() );
  }
  _sampleRate = samp1->getSampleRate();

  if( samp1->size() != samp2->size() ){
    printf("Block sizes must be the same but %d != %d \n",samp1->size(), samp2->size() );
    throw jdimension_error("Block sizes must be the same but %d != %d \n",samp1->size(),samp2->size() );
  }
  unsigned tmpi = (samp1->size()>samp2->size())? samp1->size():samp2->size();
  _fftLen = getFFTLen( tmpi );

  if( _nHeldMaxCC >= _fftLen ){
    printf("The number of the held cross-correlation coefficients should be less than the FFT length but %d > %d \n", _nHeldMaxCC, _fftLen );
    throw jdimension_error("The number of the held cross-correlation coefficients should be less than the FFT length but %d > %d \n", _nHeldMaxCC, _fftLen );	
  }
  _window = getWindow( 2, _fftLen ); // Hanning window  
  _channelList.push_back( samp1 );
  _channelList.push_back( samp2 );
  _frameCounter.push_back( 0 );
  _frameCounter.push_back( 0 );
}

CCTDE::~CCTDE()
{
  if(  (int)_channelList.size() > 0 )
    _channelList.erase( _channelList.begin(), _channelList.end() );
  
  delete [] _sampleDelays;
  delete [] _ccValues;

  gsl_vector_free( _window );
}

void  CCTDE::allsamples( int fftLen )
{
  size_t stride = 1;
  size_t chanN = _channelList.size();

  if( fftLen < 0 ){
    _ChannelIterator itr = _channelList.begin();
    unsigned samplesMax = (*itr)->samplesN(); 
    itr++;
    while( itr != _channelList.end() ) {// loop for 2 sources
      if( (*itr)->samplesN() > samplesMax )
	samplesMax = (*itr)->samplesN();
      itr++;
    }
    _fftLen = getFFTLen( samplesMax );
  }
  else
    _fftLen = (unsigned)fftLen;

  if( NULL != _window )
    delete [] _window;
  _window = getWindow( 2, _fftLen );

  double **samples = (double **)malloc( chanN * sizeof(double *));
  if( NULL == samples ){
    printf("cannot allocate memory\n");
    throw j_error("cannot allocate memory\n");
  }

  _ChannelIterator itr = _channelList.begin();
  for(unsigned i=0; itr != _channelList.end(); i++, itr++) {// loop for 2 sources
    const gsl_vector_float *block;
    unsigned blockLen;
	
    samples[i] = (double *)calloc(_fftLen,sizeof(double));
    if( samples[i] == NULL ){
      printf("cannot allocate memory\n");
      throw j_error("cannot allocate memory\n");
    }
    block = (*itr)->data();
    blockLen = (*itr)->samplesN();
    for(unsigned j =0;j<_fftLen;j++){
      if( j >= blockLen )
	break;
      samples[i][j] = gsl_vector_get(_window,j) * gsl_vector_float_get(block,j);
    }
    gsl_fft_real_radix2_transform( samples[i], stride, _fftLen );// FFT for real data
  }
  
  this->detectPeaksOfCCFunction( samples, stride );
  
  for (unsigned i=0;i<2;i++)
    free( samples[i] );
  free( samples );
}

/*
  @note re-write _sampleDelays, _ccValues and _vector.
*/
const gsl_vector* CCTDE::detectPeaksOfCCFunction( double **samples, size_t stride )
{

  double *ccA = new double[2*_fftLen];

  { // calculate the CC function in the frequency domain
#define myREAL(z,i) ((z)[2*(i)])
#define myIMAG(z,i) ((z)[2*(i)+1])
    double hc_r[2], hc_i[2];
    double val;

    hc_r[0] = samples[0][0];
    hc_i[0] = 0.0;
    hc_r[1] = samples[1][0];
    hc_i[1] = 0.0;
    val = atan2( hc_i[1], hc_r[1] ) - atan2( hc_i[0], hc_r[0] );
    myREAL( ccA, 0 ) = cos(val);
    myIMAG( ccA, 0 ) = sin(val);
    for(unsigned j =1;j<_fftLen/2;j++){
      hc_r[0] = samples[0][j*stride];
      hc_i[0] = samples[0][(_fftLen-j)*stride];
      hc_r[1] = samples[1][j*stride];
      hc_i[1] = samples[1][(_fftLen-j)*stride];
      
      val = atan2( hc_i[1], hc_r[1] ) - atan2( hc_i[0], hc_r[0] );
      myREAL( ccA, j ) = cos(val);
      myIMAG( ccA, j ) = sin(val);
      
      val = atan2( -hc_i[1], hc_r[1] ) - atan2( -hc_i[0], hc_r[0] );
      myREAL( ccA, (_fftLen - j)*stride ) = cos(val);
      myIMAG( ccA, (_fftLen - j)*stride ) = sin(val);
    }
    hc_r[0] = samples[0][(_fftLen/2)*stride];
    hc_i[0] = 0.0;
    hc_r[1] = samples[1][(_fftLen/2)*stride];
    hc_i[1] = 0.0;
    val = atan2( hc_i[1], hc_r[1] ) - atan2( hc_i[0], hc_r[0] );
    myREAL( ccA, _fftLen/2 ) = cos(val);
    myIMAG( ccA, _fftLen/2 ) = sin(val);
    
    {// discard a band 
      if( _freqUpperLimit <= 0) 
	_freqUpperLimit = _sampleRate / 2;
      if( _freqLowerLimit >= 0 && _freqUpperLimit <= 0 ){
	
	int s1 = (int)(_freqLowerLimit * _fftLen / (float)_sampleRate );
	int e1 = (int)(_freqUpperLimit * _fftLen / (float)_sampleRate );			
	for(int i=1;i<=s1;i++){
	  myREAL( ccA, i ) = 0.0;
	  myIMAG( ccA, i ) = 0.0;
	  myREAL( ccA, _fftLen - 1 - i ) = 0.0;
	  myIMAG( ccA, _fftLen - 1 - i ) = 0.0;
	}
	for(int i=e1;i<(int)_fftLen/2;i++){
	  myREAL( ccA, i ) = 0.0;
	  myIMAG( ccA, i ) = 0.0;
	  myREAL( ccA, _fftLen - 1 - i ) = 0.0;
	  myIMAG( ccA, _fftLen - 1 - i ) = 0.0;
	}
      }
    }
    gsl_fft_complex_radix2_inverse( ccA, stride, _fftLen );// with scaling

  }
  {/* detect _nHeldMaxCC peaks */
    unsigned *maxArgs = _sampleDelays;
    double   *maxVals = _ccValues; /* maxVals[0] > maxVals[1] > maxVals[2] ... */
    
    maxArgs[0] = 0;
    maxVals[0] = myREAL( ccA, 0 );
    for(unsigned i1=1;i1<_nHeldMaxCC;i1++){
      maxArgs[i1]  = -1;
      maxVals[i1] = -10e10;
    }
    for(unsigned i=1;i<_fftLen;i++){
      double cc = myREAL( ccA, i );
      
      if( cc > maxVals[_nHeldMaxCC-1] ){
	for(unsigned i1=0;i1<_nHeldMaxCC;i1++){
	  if( cc >= maxVals[i1] ){
	    for(unsigned j=_nHeldMaxCC-1;j>i1;j--){
	      maxVals[j] = maxVals[j-1];
	      maxArgs[j] = maxArgs[j-1];
	    }
	    maxVals[i1] = cc;
	    maxArgs[i1] = i;
	    break;
	  }
	}
      }
    }
    
    //set time delays to _vector
    printf("# Nth candidate : delay (sample) : delay (msec) : CC\n"); fflush(stdout);
    for(unsigned i=0;i<_nHeldMaxCC;i++){
      int sampleDelay;
      float timeDelay;
      
      if( maxArgs[i] < _fftLen/2 ){
	timeDelay   = maxArgs[i] * 1.0 / _sampleRate;
	sampleDelay = maxArgs[i];
      }
      else{
	timeDelay   = - ( (float)_fftLen - maxArgs[i] ) * 1.0 / _sampleRate;
	sampleDelay = - ( _fftLen - maxArgs[i] );
      }
      gsl_vector_set( _vector, i, timeDelay );
      printf("%d : %d : %f : %f\n",i, sampleDelay, timeDelay * 1000, maxVals[i] ); fflush(stdout);
    }
#undef myREAL
#undef myIMAG
  }
  
  delete [] ccA;
  return _vector;
}

/**
   @brief
   @param
   @return TDOAs btw. two signals (sec).
*/
const gsl_vector* CCTDE::next(int frameX)
{
  if (frameX == _frameX) return _vector;
  
  size_t stride = 1;
  
  double **samples = (double **)malloc( _channelList.size() * sizeof(double *));
  if( NULL == samples ){
    printf("cannot allocate memory\n");
    throw j_error("cannot allocate memory\n");
  }
  
  _ChannelIterator itr = _channelList.begin();
  for(unsigned i=0; itr != _channelList.end(); i++, itr++) {// loop for 2 sources
    const gsl_vector_float *block;

    block = (*itr)->next(frameX);    
    samples[i] = (double *)calloc(_fftLen,sizeof(double));
    unsigned blockLen = (*itr)->size();
    for(unsigned j =0;j<_fftLen;j++){
      if( j >= blockLen )
	break;
      samples[i][j] = gsl_vector_get(_window,j) * gsl_vector_float_get(block,j);
    }
    gsl_fft_real_radix2_transform( samples[i], stride, _fftLen );// FFT for real data
  }
  
  this->detectPeaksOfCCFunction( samples, stride );
  for (unsigned i=0;i<_channelList.size();i++)
    free( samples[i] );
  free( samples );
  
  _increment();
  return _vector;
}

const gsl_vector* CCTDE::nextX( unsigned chanX, int frameX )
{
  size_t stride = 1;
  
  double **samples = (double **)malloc( _channelList.size() * sizeof(double *));
  if( NULL == samples ){
    printf("cannot allocate memory\n");
    throw j_error("cannot allocate memory\n");
  }
  
  _ChannelIterator itr = _channelList.begin();
  for(unsigned i=0; itr != _channelList.end(); i++, itr++) {// loop for 2 sources
    const gsl_vector_float *block;

    if( i != chanX )
      block = (*itr)->current();
    else
      block = (*itr)->next(frameX);
    samples[i] = (double *)calloc(_fftLen,sizeof(double));
    unsigned blockLen = (*itr)->size();
    for(unsigned j =0;j<_fftLen;j++){
      if( j >= blockLen )
	break;
      samples[i][j] = gsl_vector_get(_window,j) * gsl_vector_float_get(block,j);
    }
    gsl_fft_real_radix2_transform( samples[i], stride, _fftLen );// FFT for real data
  }
  
  this->detectPeaksOfCCFunction( samples, stride );
  for (unsigned i=0;i<_channelList.size();i++)
    free( samples[i] );
  free( samples );

  if( chanX == 0 )
    _increment();
  _frameCounter[chanX]++;

  return _vector;
}

void CCTDE::reset()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++) {
    (*itr)->reset();
  }
  VectorFeatureStream::reset();
}
