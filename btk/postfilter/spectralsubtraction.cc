#include "spectralsubtraction.h"
#include <gsl/gsl_blas.h>
#include <matrix/blas1_c.H>
#include <matrix/linpack_c.H>

PSDEstimator::PSDEstimator(unsigned fftLen2)
{
  _estimates = gsl_vector_alloc( fftLen2+1 );
  gsl_vector_set_zero( _estimates );
}

PSDEstimator::~PSDEstimator()
{
  gsl_vector_free( _estimates );
}

bool PSDEstimator::readEstimates( const String& fn )
{  
  FILE *fp = fopen( fn.c_str(), "r" );
  if( NULL == fp ){
    fprintf(stderr,"could not read %s\n", fn.c_str() );
    return false;
  }
  
  for(size_t i=0;i<_estimates->size;i++){
    double val;
    
    fscanf( fp, "%lf\n", &val );
    gsl_vector_set( _estimates, i, val );
  }
  fclose(fp);
  return true;
}

bool PSDEstimator::writeEstimates( const String& fn )
{
  
  FILE *fp = fopen( fn.c_str(), "w" );
  if( NULL == fp ){
    fprintf(stderr,"could not write %s\n", fn.c_str() );
    return false;
  }
  
  for(size_t i=0;i<_estimates->size;i++){
    fprintf( fp, "%lf\n", gsl_vector_get(_estimates, i ) );
  }
  fclose(fp);
  return true;
}


averagePSDEstimator::averagePSDEstimator(unsigned fftLen2, double alpha ):
  PSDEstimator(fftLen2), _alpha(alpha), _isSampleAdded(false)
{
  _sampleL.clear();
}

averagePSDEstimator::~averagePSDEstimator()
{
  this->clearSamples();
}

void averagePSDEstimator::clear()
{
  //fprintf(stderr,"averagePSDEstimator::clear()\n");
  _isSampleAdded = false;
  this->clearSamples();
}

const gsl_vector* averagePSDEstimator::average()
{
  if( _alpha < 0 ){
    list<gsl_vector *>::iterator itr = _sampleL.begin();
    unsigned sampleN = 0;
    
    gsl_vector_set_zero( _estimates );
    while( itr != _sampleL.end() ){
      //gsl_vector_add ( _estimates, *itr );
      gsl_vector_add ( _estimates, (gsl_vector*)*itr );
      sampleN++;
      itr++;
    }
      
    gsl_vector_scale( _estimates, 1.0/(double)sampleN );
  }
  return (const gsl_vector*)_estimates;
}

bool averagePSDEstimator::addSample( const gsl_vector_complex *sample )
{
  
  size_t fftLen2 = _estimates->size - 1;
  gsl_vector *tmp = gsl_vector_alloc( fftLen2+1 );
  
  for(size_t i=0;i<=fftLen2;i++){
    gsl_complex val = gsl_vector_complex_get( sample, i );
    gsl_vector_set( tmp, i, gsl_complex_abs2 ( val ) );
  }

  if( _alpha < 0 ){/* to calc. the average */
    _sampleL.push_back( tmp );
  }
  else{/* for recursive averaging */
    if( _isSampleAdded==false ){
      gsl_vector_memcpy( _estimates, tmp );
      _isSampleAdded = true;
    }
    else{
      gsl_vector_scale( _estimates, _alpha );
      gsl_vector_scale( tmp,       (1.0-_alpha) );
      gsl_vector_add( _estimates, tmp );
    }
    gsl_vector_free(tmp);
  }
  
  //_isSampleAdded = true;

  return true;
}

void averagePSDEstimator::clearSamples()
{
  list<gsl_vector *>::iterator itr = _sampleL.begin();
  while( itr != _sampleL.end() ){
    gsl_vector_free( *itr );
    itr++;
  }
  _sampleL.clear();
}

/*
@brief

@param unsigned fftLen
@param bool halfBandShift
@param float ft[in] subtraction coeffient
@param float flooringV
@param const String& nm

*/
SpectralSubtractor::SpectralSubtractor(unsigned fftLen, bool halfBandShift, float ft, float flooringV, const String& nm)
: VectorComplexFeatureStream(fftLen, nm)
{
  _fftLen = fftLen;
  _fftLen2 = fftLen/2;
  _halfBandShift = halfBandShift;
  _isTrainingStarted = true;
  _totalTrainingSampleN = 0;
  _ft = ft;
  _flooringV = flooringV;
  _startNoiseSubtraction = false;
}

SpectralSubtractor::~SpectralSubtractor()
{
  if(  (int)_channelList.size() > 0 )
    _channelList.erase( _channelList.begin(), _channelList.end() );

  if(  (int)_noisePSDList.size() > 0 )
    _noisePSDList.erase( _noisePSDList.begin(), _noisePSDList.end() );
}

void SpectralSubtractor::reset()
{
  //fprintf(stderr,"SpectralSubtractor::reset\n");

  _totalTrainingSampleN  = 0;

  _ChannelIterator  itr1 = _channelList.begin();
  //_NoisePSDIterator itr2 = _noisePSDList.begin();
  for(; itr1 != _channelList.end(); itr1++){
    (*itr1)->reset();
    //(*itr2)->reset();
    //itr2++;
  }
  
  VectorComplexFeatureStream::reset();
  //_endOfSample = false;
}

/*
@brief
@param
*/
void SpectralSubtractor::setChannel( VectorComplexFeatureStreamPtr& chan, double alpha )
{
  _channelList.push_back(chan);
  _noisePSDList.push_back( new averagePSDEstimator(_fftLen2, alpha) );
}

/*
@brief If the noise PSD model has been trained, which means _isTrainingStarted == false, 
       this method performs spectral subtraction on the audio data set by setChannel().  
       Otherwise, this updates the noise PSD model.
@param 
@return 
*/
const gsl_vector_complex* SpectralSubtractor::next(int frameX)
{
  size_t chanN = _channelList.size();
  _ChannelIterator  itr1 = _channelList.begin();
  _NoisePSDIterator itr2 = _noisePSDList.begin();
  gsl_complex c1 = gsl_complex_rect(1.0, 0.0);
  
  gsl_vector_complex_set_zero( _vector );
  for(size_t i=0;i<chanN;i++,itr1++,itr2++){
	const gsl_vector_complex* samp_i = (*itr1)->next(frameX);
	
	if( _isTrainingStarted == true ){
	  (*itr2)->addSample( samp_i ); // for computating the variance of the noise PSD
	  _totalTrainingSampleN++;
	}
	if( _startNoiseSubtraction == false ){
	  gsl_blas_zaxpy( c1, samp_i, _vector ); // _vector = samp_i + _vector
	}
	else{
	  const gsl_vector* noisePSD = (*itr2)->getEstimate();
	  gsl_complex Xt, tmp;

#ifdef IGNORE_DC
	  //for fbinX = 0
	  Xt  = gsl_vector_complex_get( samp_i, 0 );
	  tmp = gsl_vector_complex_get( _vector, 0 );
	  gsl_vector_complex_set( _vector, 0, gsl_complex_add( Xt, tmp) );
	  for(unsigned fbinX=1;fbinX<=_fftLen2;fbinX++){
	    Xt  = gsl_vector_complex_get( samp_i, fbinX );
	    tmp = gsl_vector_complex_get( _vector, fbinX );
	    double th = gsl_complex_arg( Xt );
	    double X2 = gsl_complex_abs2( Xt );
	    double N2 = gsl_vector_get( noisePSD, fbinX );
	    double S2 = X2 - _ft * N2;
	    if( S2 <= _flooringV ){ // flooring
	      S2 = _flooringV;
	    }
	    gsl_complex St  = gsl_complex_polar( sqrt(S2), th );
	    gsl_complex Stp = gsl_complex_add( St, tmp );
	    gsl_vector_complex_set( _vector, fbinX, Stp );
	    if( fbinX < _fftLen2 )
	      gsl_vector_complex_set( _vector, _fftLen - fbinX, gsl_complex_conjugate(Stp) );
	  }
	  //for fbinX = fftLen2
#else
	  for(unsigned fbinX=0;fbinX<=_fftLen2;fbinX++){
	    Xt  = gsl_vector_complex_get( samp_i, fbinX );
	    tmp = gsl_vector_complex_get( _vector, fbinX );
	    double th = gsl_complex_arg( Xt );
	    double X2 = gsl_complex_abs2( Xt );
	    double N2 = gsl_vector_get( noisePSD, fbinX );
	    double S2 = X2 - _ft * N2;
	    if( S2 <= _flooringV ){ // flooring
	      S2 = _flooringV;
	    }
	    gsl_complex St  = gsl_complex_polar( sqrt(S2), th );
	    gsl_complex Stp = gsl_complex_add( St, tmp );
	    gsl_vector_complex_set( _vector, fbinX, Stp );
	    if( fbinX > 0 && fbinX < _fftLen2 )
	      gsl_vector_complex_set( _vector, _fftLen - fbinX, gsl_complex_conjugate(Stp) );
	  }
#endif /* IGNORE_DC */
	  
	}
  }
  
  gsl_blas_zdscal( 1.0/(double)chanN, _vector );
  _increment();
  return _vector;
}

WienerFilter::WienerFilter( VectorComplexFeatureStreamPtr &targetSignal, VectorComplexFeatureStreamPtr &noiseSignal, bool halfBandShift, float alpha, float flooringV, double beta, const String& nm )
  : VectorComplexFeatureStream(targetSignal->size(), nm),
    _targetSignal(targetSignal), _noiseSignal(noiseSignal),
    _halfBandShift(halfBandShift), _alpha(alpha), _flooringV(flooringV), _beta(beta), _updateNoisePSD(true)
{
  _fftLen  = _targetSignal->size();
  _fftLen2 = _fftLen / 2 ;

  //fprintf(stderr,"WienerFilter::WienerFilter\n");
  if( _fftLen != _noiseSignal->size() ){
    fprintf(stderr,"Input block length (%d) != fftLen (%d)\n", _fftLen, _noiseSignal->size() );
    throw jdimension_error("Input block length (%d) != fftLen (%d)\n", _fftLen, _noiseSignal->size() );
  }

  _prevPSDs = gsl_vector_calloc( _fftLen2 + 1 );
  _prevPSDn = gsl_vector_calloc( _fftLen2 + 1 );
}

WienerFilter::~WienerFilter()
{
  gsl_vector_free( _prevPSDs );
  gsl_vector_free( _prevPSDn );
}

const gsl_vector_complex *WienerFilter::next( int frameX )
{
  if (frameX == _frameX) return _vector;
  
  const gsl_vector_complex *St = _targetSignal->next( frameX );
  const gsl_vector_complex *Nt = _noiseSignal->next( frameX );
  double alpha, H, PSDs, PSDn, prevPSDs, prevPSDn, currPSDs, currPSDn;
  gsl_complex val;

  if( _frameX > 0 )
    alpha =  _alpha;
  else
    alpha = 0.0;

  if( false == _halfBandShift ){
    gsl_vector_complex_set( _vector, 0, gsl_vector_complex_get( St, 0) );
    for (unsigned fbinX = 1; fbinX <= _fftLen2; fbinX++) {
      prevPSDs = gsl_vector_get( _prevPSDs, fbinX );
      currPSDs = gsl_complex_abs2( gsl_vector_complex_get( St, fbinX ) );
      PSDs = alpha * prevPSDs + (1-alpha) * currPSDs;

      prevPSDn = gsl_vector_get( _prevPSDn, fbinX );
      if( _updateNoisePSD ){
	currPSDn = gsl_complex_abs2( gsl_vector_complex_get( Nt, fbinX ) );
	if( currPSDn < _flooringV )
	  currPSDn = _flooringV;
	PSDn = alpha * prevPSDn + (1-alpha) * currPSDn;
      }
      else
	PSDn = prevPSDn;

      H = PSDs / ( PSDs + _beta * PSDn );
      val = gsl_complex_mul_real( gsl_vector_complex_get( St, fbinX ), H );
      gsl_vector_complex_set( _vector, fbinX, val );
      gsl_vector_set( _prevPSDs, fbinX, PSDs );
      if( _updateNoisePSD )
	gsl_vector_set( _prevPSDn, fbinX, PSDn );
      if( fbinX == _fftLen2 )
	gsl_vector_complex_set(_vector, _fftLen - fbinX, gsl_complex_conjugate(val) );
    }
  }
  else{
    fprintf(stderr,"WienerFilter::next() for the half band shift is not implemented\n");
    throw  j_error("WienerFilter::next() for the half band shift is not implemented\n");
  }

  _increment();
  return _vector;
}

void WienerFilter::reset()
{
  _targetSignal->reset();
  _noiseSignal->reset();
}

