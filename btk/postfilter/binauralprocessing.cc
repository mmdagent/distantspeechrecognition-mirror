#include <math.h>
#include "postfilter/binauralprocessing.h"

/*
  @brief calculate the interaural time delay (ITD) at each frequency bin
  @param fbinX[in] the index of the frequency component
  @param fftLen[in]
  @param X_L_f[in]
  @param X_R_f[in]
  @return ITD
*/
static double calcITDf( unsigned fbinX, unsigned fftLen, gsl_complex X_L_f, gsl_complex X_R_f )
{
  double ITD; /* => |d_{s^*[m,k][m,k]}| Eq. (4) */
  double ad_X_LAngle_f = gsl_complex_arg( X_L_f );
  double ad_X_RAngle_f = gsl_complex_arg( X_R_f );
  double adPhaseDiff1  = fabs( ad_X_LAngle_f - ad_X_RAngle_f );
  double adPhaseDiff2  = fabs( ad_X_LAngle_f - ad_X_RAngle_f - 2 * M_PI );
  double adPhaseDiff3  = fabs( ad_X_LAngle_f - ad_X_RAngle_f + 2 * M_PI );
  double adPhaseDiff; /* => |w_k| * |d_{s^*[m,k][m,k]}| Eq.  */
  
  if( adPhaseDiff1 < adPhaseDiff2 ){
    adPhaseDiff = adPhaseDiff1;
  }
  else{
    adPhaseDiff = adPhaseDiff2;
  }
  if( adPhaseDiff3 < adPhaseDiff )
    adPhaseDiff = adPhaseDiff3;

  ITD = adPhaseDiff / ( 2 * M_PI * fbinX / fftLen );
  return ITD;
}

// ----- definition for class 'BinaryMaskFilter' -----
// 

/**
   @brief construct this object
   @param VectorComplexFeatureStreamPtr srcL[in]
   @param VectorComplexFeatureStreamPtr srcR[in]
   @param unsigned M[in] the FFT-poit
   @param float threshold[in] threshold for the ITD
   @param float alpha[in] forgetting factor
   @param float dEta[in] flooring value 
*/
BinaryMaskFilter::BinaryMaskFilter( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M,
				    float threshold, float alpha, float dEta , const String& nm ):
  VectorComplexFeatureStream(M, nm),
  _chanX(chanX),
  _srcL(srcL),
  _srcR(srcR),
  _threshold(threshold),
  _alpha(alpha),
  _dEta(dEta),
  _thresholdAtFreq(NULL)
{
  if( srcL->size() != M ){
    fprintf(stderr,"Left input block length (%d) != M (%d)\n", srcL->size(), M );
    throw jdimension_error("Left input block length (%d) != M (%d)\n", srcL->size(), M );
  }
  
  if( srcR->size() != M ){
    fprintf(stderr,"Right input block length (%d) != M (%d)\n", srcR->size(), M );
    throw jdimension_error("Right input block length (%d) != M (%d)\n", srcR->size(), M );
  }

  _prevMu = gsl_vector_float_alloc( M/2+1 );
  gsl_vector_float_set_all( _prevMu, 1.0 );
}

BinaryMaskFilter::~BinaryMaskFilter()
{
  gsl_vector_float_free( _prevMu );
  if( NULL != _thresholdAtFreq )
    gsl_vector_free( _thresholdAtFreq );
}

void BinaryMaskFilter::setThresholds( const gsl_vector *thresholds )
{
  unsigned fftLen2 = (unsigned)_srcL->size()/2;

  if( NULL != _thresholdAtFreq ){
    for(unsigned int fbinX=1;fbinX<=fftLen2;fbinX++){
      gsl_vector_set( _thresholdAtFreq, fbinX, gsl_vector_get( thresholds, fbinX ) );
    }
  }
  else{
    _thresholdAtFreq = gsl_vector_alloc( fftLen2+1 );
  }

}

const gsl_vector_complex* BinaryMaskFilter::next(int frameX)
{
  _increment();
  return _vector;
}

void BinaryMaskFilter::reset()
{
  _srcL->reset();
  _srcR->reset();
  VectorComplexFeatureStream::reset();
  _endOfSamples = false;
}

// ----- definition for class 'KimBinaryMaskFilter' -----
// 

/**
   @brief construct this object
   @param VectorComplexFeatureStreamPtr srcL[in]
   @param VectorComplexFeatureStreamPtr srcR[in]
   @param unsigned M[in] the FFT-poit
   @param float threshold[in] threshold for the ITD
   @param float alpha[in] forgetting factor
   @param float dEta[in] flooring value 
   @param float dPowerCoeff[in] power law non-linearity
*/
KimBinaryMaskFilter::KimBinaryMaskFilter( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M,
					 float threshold, float alpha, float dEta , float dPowerCoeff, const String& nm ):
  BinaryMaskFilter( chanX, srcL, srcR, M, threshold, alpha, dEta, nm ),
  _dPowerCoeff(dPowerCoeff)
{
}

KimBinaryMaskFilter::~KimBinaryMaskFilter()
{
}

/**
   @brief perform binary masking which picks up the left channel when the ITD <= threshold
*/
const gsl_vector_complex* KimBinaryMaskFilter::masking1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R, float threshold )
{
  unsigned fftLen  = (unsigned)ad_X_L->size;
  unsigned fftLen2 = fftLen / 2;
  gsl_complex val;

  // Direct component : fbinX = 0
  val = gsl_vector_complex_get( ad_X_L, 0 );
  gsl_vector_complex_set( _vector, 0, val );

  for(unsigned fbinX=1;fbinX<=fftLen2;fbinX++){
    float  mu;
    double ITD = calcITDf( fbinX, fftLen, 
			   gsl_vector_complex_get( ad_X_L, fbinX ),
			   gsl_vector_complex_get( ad_X_R, fbinX ) );
    if( _chanX == 0 ){
      if( ITD <= _threshold )
	mu = _alpha * gsl_vector_float_get( _prevMu, fbinX ) + ( 1 - _alpha );
      else
	mu = _alpha * gsl_vector_float_get( _prevMu, fbinX ) + ( 1 - _alpha ) * _dEta;
      val = gsl_complex_mul_real ( gsl_vector_complex_get( ad_X_L, fbinX ), mu );
    }
    else{
      if( ITD <= _threshold )
	mu = _alpha * gsl_vector_float_get( _prevMu, fbinX ) + ( 1 - _alpha ) * _dEta;
      else
	mu = _alpha * gsl_vector_float_get( _prevMu, fbinX ) + ( 1 - _alpha );
      val = gsl_complex_mul_real ( gsl_vector_complex_get( ad_X_R, fbinX ), mu );
    }

    if( fbinX < fftLen2 ){
      gsl_vector_complex_set(_vector, fbinX, val);
      gsl_vector_complex_set(_vector, fftLen - fbinX, gsl_complex_conjugate(val) );
    }
    else
      gsl_vector_complex_set(_vector, fftLen2, val);
    
    gsl_vector_float_set( _prevMu, fbinX, mu );
  }

  return _vector;
}

const gsl_vector_complex* KimBinaryMaskFilter::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  const gsl_vector_complex* ad_X_L;
  const gsl_vector_complex* ad_X_R;

  if( frameX >= 0 ){
    ad_X_L = _srcL->next(frameX);
    ad_X_R = _srcR->next(frameX);
  }
  else{
    if( _frameX == FrameResetX ){
      ad_X_L = _srcL->next(0);
      ad_X_R = _srcR->next(0);
    }
    else{
      ad_X_L = _srcL->next(_frameX+1);
      ad_X_R = _srcR->next(_frameX+1);
    }
  }
  masking1( ad_X_L, ad_X_R, _threshold );
  
  _increment();
  return _vector;
}

void KimBinaryMaskFilter::reset()
{
  _srcL->reset();
  _srcR->reset();
  VectorComplexFeatureStream::reset();
  _endOfSamples = false;
}

// ----- definition for class 'KimITDThresholdEstimator' -----
// 

/**
   @brief construct this object
   @param VectorComplexFeatureStreamPtr srcL[in]
   @param VectorComplexFeatureStreamPtr srcR[in]
   @param unsigned M[in] the FFT-poit
   @param float minThreshold
   @param float maxThreshold
   @param float width
   @param float minFreq
   @param float maxFreq
   @param int sampleRate
   @param float threshold[in] threshold for the ITD
   @param float alpha[in] forgetting factor
   @param float dEta[in] flooring value 
   @param float dPowerCoeff[in] power law non-linearity
*/
KimITDThresholdEstimator::KimITDThresholdEstimator(VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, 
						   float minThreshold, float maxThreshold, float width,
						   float minFreq, float maxFreq, int sampleRate, 
						   float dEta, float dPowerCoeff, const String& nm ):
  KimBinaryMaskFilter( 0, srcL, srcR, M, 0.0, 0.0 /* alha must be zero */, dEta, dPowerCoeff ),
  _width(width),
  _isCostFunctionComputed(false),
  _nCand(0)
{
  if( srcL->size() != M ){
    fprintf(stderr,"Left input block length (%d) != M (%d)\n", srcL->size(), M );
    throw jdimension_error("Left input block length (%d) != M (%d)\n", srcL->size(), M );
  }
  
  if( srcR->size() != M ){
    fprintf(stderr,"Right input block length (%d) != M (%d)\n", srcR->size(), M );
    throw jdimension_error("Right input block length (%d) != M (%d)\n", srcR->size(), M );
  }

  if( minThreshold == maxThreshold ){
    _minThreshold = - 0.2 * 16000 / 340;
    _maxThreshold =   0.2 * 16000 / 340;
  }
  else{
    _minThreshold = minThreshold;
    _maxThreshold = maxThreshold;
  }

  if( minFreq < 0 || maxFreq < 0 || sampleRate < 0 ){
    _minFbinX = 1;
    _maxFbinX = M/2 + 1; 
  }
  else{
    _minFbinX = (unsigned) ( M * minFreq / (float)sampleRate );
    _maxFbinX = (unsigned) ( M * maxFreq / (float)sampleRate );
  }

  int nCand = (int)( (_maxThreshold-_minThreshold)/width + 1.5 );
  _nCand = (unsigned int)nCand;

  _costFunctionValues = (double *)calloc(nCand,sizeof(double));
  _sigma_T = (double *)calloc(nCand,sizeof(double));
  _sigma_I = (double *)calloc(nCand,sizeof(double));
  _mean_T  = (double *)calloc(nCand,sizeof(double));
  _mean_I  = (double *)calloc(nCand,sizeof(double));
  if( NULL==_costFunctionValues || 
      NULL==_sigma_T || NULL==_sigma_I || 
      NULL==_mean_T  || NULL==_mean_I  ){
    fprintf(stderr,"KimITDThresholdEstimator:cannot allocate memory\n");
    throw jallocation_error("KimITDThresholdEstimator:cannot allocate memory\n");    
  }

  _buffer = gsl_vector_alloc( _nCand );

  _nSamples = 0;
}

KimITDThresholdEstimator::~KimITDThresholdEstimator()
{
  free( _costFunctionValues );
  free( _sigma_T );
  free( _sigma_I );
  free( _mean_T );
  free( _mean_I );
  if( NULL != _buffer ){
    gsl_vector_free( _buffer );
    _buffer = NULL;
  }
}

const gsl_vector* KimITDThresholdEstimator::getCostFunction()
{
  if( _isCostFunctionComputed == false ){
    fprintf(stderr,"KimITDThresholdEstimator:call calcThreshold() first\n");
  }
  for(unsigned int iSearchX=0;iSearchX<_nCand;iSearchX++){
    gsl_vector_set( _buffer, iSearchX, _costFunctionValues[iSearchX] );
  }

  return (const gsl_vector*)_buffer;
}

void KimITDThresholdEstimator::accumStats1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R )
{
  unsigned fftLen = (unsigned)ad_X_L->size;
  gsl_complex X_T, X_I;
  double      mu_T, mu_I;
  unsigned    iSearchX=0;

  for(float threshold=_minThreshold;threshold<=_maxThreshold;threshold+=_width,iSearchX++){
    double P_T = 0.0, R_T;
    double P_I = 0.0, R_I;

    for(unsigned fbinX=_minFbinX;fbinX<_maxFbinX;fbinX++){
      double ITD = calcITDf(fbinX, fftLen, 
			    gsl_vector_complex_get( ad_X_L, fbinX ),
			    gsl_vector_complex_get( ad_X_R, fbinX ));
      if( ITD <= threshold ){
	mu_T = 1.0;
	mu_I = _dEta;
      }
      else{
	mu_T = _dEta;
	mu_I = 1.0;
      }
      X_T = gsl_complex_mul_real ( gsl_vector_complex_get( ad_X_L, fbinX ), mu_T );
      X_I = gsl_complex_mul_real ( gsl_vector_complex_get( ad_X_R, fbinX ), mu_I );
      P_T += gsl_complex_abs2( X_T );
      P_I += gsl_complex_abs2( X_I );
    }
    R_T = (double)pow( (double)P_T, (double)_dPowerCoeff );
    R_I = (double)pow( (double)P_I, (double)_dPowerCoeff );
    _costFunctionValues[iSearchX] += R_T * R_I;
    _mean_T[iSearchX]  += R_T;
    _mean_I[iSearchX]  += R_I;
    _sigma_T[iSearchX] += R_T * R_T;
    _sigma_I[iSearchX] += R_I * R_I;
  }
  
  _nSamples++;
  return;
}

const gsl_vector_complex* KimITDThresholdEstimator::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  const gsl_vector_complex* ad_X_L;
  const gsl_vector_complex* ad_X_R;

  if( frameX >= 0 ){
    ad_X_L = _srcL->next(frameX);
    ad_X_R = _srcR->next(frameX);
  }
  else{
    if( _frameX == FrameResetX ){
      ad_X_L = _srcL->next(0);
      ad_X_R = _srcR->next(0);
    }
    else{
      ad_X_L = _srcL->next(_frameX+1);
      ad_X_R = _srcR->next(_frameX+1);
    }
  }
  accumStats1( ad_X_L, ad_X_R );
  
  _increment();
  return _vector;
}

double KimITDThresholdEstimator::calcThreshold()
{
  float argMin = _minThreshold;
  double min_rho = 1000000;  
  unsigned iSearchX=0;

  for(float threshold=_minThreshold;threshold<=_maxThreshold;threshold+=_width,iSearchX++){
    double rho;

    _mean_T[iSearchX]  /= _nSamples;
    _mean_I[iSearchX]  /= _nSamples;
    _sigma_T[iSearchX] = ( _sigma_T[iSearchX] / _nSamples ) - _mean_T[iSearchX] * _mean_T[iSearchX];
    _sigma_I[iSearchX] = ( _sigma_I[iSearchX] / _nSamples ) - _mean_I[iSearchX] * _mean_I[iSearchX];
    _costFunctionValues[iSearchX] /= _nSamples;
    rho = fabs( ( _costFunctionValues[iSearchX] - _mean_T[iSearchX] *_mean_I[iSearchX] ) / ( sqrt( _sigma_T[iSearchX] ) * sqrt( _sigma_I[iSearchX] ) ) ); 
    //fprintf(stderr,"%f %f\n",threshold,rho);
    if( rho < min_rho ){
      argMin = threshold;
      min_rho = rho;
    }
  }

  _threshold = argMin;
  _isCostFunctionComputed = true;
  return argMin;
}

void KimITDThresholdEstimator::reset()
{
  _srcL->reset();
  _srcR->reset();
  VectorComplexFeatureStream::reset();
  _endOfSamples = false;
  
  unsigned iSearchX=0;
  for(float threshold=_minThreshold;threshold<=_maxThreshold;threshold+=_width,iSearchX++){
    _costFunctionValues[iSearchX] = 0.0;
    _mean_T[iSearchX]  = 0.0;
    _mean_I[iSearchX]  = 0.0;
    _sigma_T[iSearchX] = 0.0;
    _sigma_I[iSearchX] = 0.0;
  }
  _nSamples = 0;
  _isCostFunctionComputed = false;
}

// ----- definition for class 'IIDBinaryMaskFilter' -----
// 

IIDBinaryMaskFilter::IIDBinaryMaskFilter( unsigned chanX, VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M,
										 float threshold, float alpha, float dEta, const String& nm ):
BinaryMaskFilter( chanX, srcL, srcR, M, threshold, alpha, dEta, nm )
{
}

IIDBinaryMaskFilter::~IIDBinaryMaskFilter()
{
}

const gsl_vector_complex* IIDBinaryMaskFilter::masking1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R, float threshold )
{
  unsigned fftLen  = (unsigned)ad_X_L->size;
  unsigned fftLen2 = fftLen / 2;
  gsl_complex val;
  
  // Direct component : fbinX = 0
  val = gsl_vector_complex_get( ad_X_L, 0 );
  gsl_vector_complex_set( _vector, 0, val );
  
  for(unsigned fbinX=1;fbinX<=fftLen2;fbinX++){
    gsl_complex X_T, X_I;
    double P_T, P_I;
    float  mu;
    
    if( NULL != _thresholdAtFreq )
      _threshold = gsl_vector_get( _thresholdAtFreq, fbinX );
    
    if( _chanX == 0 ){ /* the left channel contains the stronger target signal */
      X_T = gsl_vector_complex_get( ad_X_L, fbinX );
      X_I = gsl_vector_complex_get( ad_X_R, fbinX );
    }
    else{
      X_T = gsl_vector_complex_get( ad_X_R, fbinX );
      X_I = gsl_vector_complex_get( ad_X_L, fbinX );
    }
    P_T = gsl_complex_abs( X_T );
    P_I = gsl_complex_abs( X_I );
    
    if( P_T <= ( P_I + _threshold ) )
      mu = _alpha * gsl_vector_float_get( _prevMu, fbinX ) + ( 1 - _alpha ) * _dEta;
    else
      mu = _alpha * gsl_vector_float_get( _prevMu, fbinX ) + ( 1 - _alpha ) ;
    val = gsl_complex_mul_real ( X_T, mu );
    if( fbinX < fftLen2 ){
      gsl_vector_complex_set(_vector, fbinX, val);
      gsl_vector_complex_set(_vector, fftLen - fbinX, gsl_complex_conjugate(val) );
    }
    else
      gsl_vector_complex_set(_vector, fftLen2, val);
    gsl_vector_float_set( _prevMu, fbinX, mu );
  }
  
  return _vector;
}

const gsl_vector_complex* IIDBinaryMaskFilter::next(int frameX)
{
  if (frameX == _frameX) return _vector;
  
  const gsl_vector_complex* ad_X_L;
  const gsl_vector_complex* ad_X_R;
  
  if( frameX >= 0 ){
    ad_X_L = _srcL->next(frameX);
    ad_X_R = _srcR->next(frameX);
  }
  else{
    if( _frameX == FrameResetX ){
      ad_X_L = _srcL->next(0);
      ad_X_R = _srcR->next(0);
    }
    else{
      ad_X_L = _srcL->next(_frameX+1);
      ad_X_R = _srcR->next(_frameX+1);
    }
  }
  masking1( ad_X_L, ad_X_R, _threshold );
	
  _increment();
  return _vector;
}

void IIDBinaryMaskFilter::reset()
{
  _srcL->reset();
  _srcR->reset();
  VectorComplexFeatureStream::reset();
  _endOfSamples = false;
}

// ----- definition for class 'IIDThresholdEstimator' -----
// 

IIDThresholdEstimator::IIDThresholdEstimator( VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, 
											  float minThreshold, float maxThreshold, float width,
											 float minFreq, float maxFreq, int sampleRate, float Eta, float dPowerCoeff, const String& nm ):
  KimITDThresholdEstimator( srcL, srcR, M, minThreshold, maxThreshold, width, minFreq, maxFreq, sampleRate, Eta, dPowerCoeff, nm ),
  _beta(3.0)
{
  _Y4_T = (double *)calloc(_nCand,sizeof(double));
  _Y4_I = (double *)calloc(_nCand,sizeof(double));
  
  if( NULL==_Y4_T || NULL==_Y4_I  ){
    fprintf(stderr,"IIDThresholdEstimator:cannot allocate memory\n");
    throw jallocation_error("IIDThresholdEstimator:cannot allocate memory\n");    
  }
}

IIDThresholdEstimator::~IIDThresholdEstimator()
{
  free(_Y4_T);
  free(_Y4_I);
}

/*
  @brief calculate kurtosis of beamformer's outputs
*/
void IIDThresholdEstimator::accumStats1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R )
{
	unsigned fftLen = (unsigned)ad_X_L->size;
	unsigned    iSearchX=0;
	
	//fprintf(stderr,"IIDThresholdEstimator::accumStats1 %d\n",_frameX);
	for(float threshold=_minThreshold;threshold<=_maxThreshold;threshold+=_width,iSearchX++){
        	double Y1_T = 0.0, Y2_T = 0.0, Y4_T = 0.0;
		double Y1_I = 0.0, Y2_I = 0.0, Y4_I = 0.0;
		
		for(unsigned fbinX=_minFbinX;fbinX<_maxFbinX;fbinX++){
			gsl_complex X_T_f,  X_I_f;
			double      P_T_f,  P_I_f;
			double      mu_T,   mu_I;
			double      Y1_T_f, Y1_I_f; /* the magnitude of the binary masked value */
			double      Y2_T_f, Y2_I_f; /* the power of the binary masked value */

			X_T_f = gsl_vector_complex_get( ad_X_L, fbinX );
			X_I_f = gsl_vector_complex_get( ad_X_R, fbinX );
			P_T_f = gsl_complex_abs( X_T_f );
			P_I_f = gsl_complex_abs( X_I_f );
			
			if( P_T_f <= ( P_I_f + threshold ) )
				mu_T = _dEta; //_alpha * gsl_vector_float_get( _prevMu, fbinX ) + ( 1 - _alpha ) * _dEta;
			else
				mu_T = 1.0;   //_alpha * gsl_vector_float_get( _prevMu, fbinX ) + ( 1 - _alpha ) ;
			
			if( P_I_f <= ( P_T_f + threshold ) )
				mu_I = _dEta; //_alpha * gsl_vector_float_get( _prevMu, fbinX ) + ( 1 - _alpha ) * _dEta;
			else
				mu_I = 1.0;   //_alpha * gsl_vector_float_get( _prevMu, fbinX ) + ( 1 - _alpha ) ;
			
			Y1_T_f = pow( gsl_complex_abs( gsl_complex_mul_real( X_T_f, mu_T ) ), (double)2.0*_dPowerCoeff );
			Y1_I_f = pow( gsl_complex_abs( gsl_complex_mul_real( X_I_f, mu_I ) ), (double)2.0*_dPowerCoeff );
			//Y1_T_f = gsl_complex_abs( gsl_complex_mul_real( X_T_f, mu_T ) );
			//Y1_I_f = gsl_complex_abs( gsl_complex_mul_real( X_I_f, mu_I ) );
			Y2_T_f = Y1_T_f * Y1_T_f;
			Y2_I_f = Y1_I_f * Y1_I_f;
			Y1_T += Y1_T_f;
			Y1_I += Y1_I_f;
			Y2_T += Y2_T_f;
			Y2_I += Y2_I_f;
			Y4_T += Y2_T_f * Y2_T_f;
			Y4_I += Y2_I_f * Y2_I_f;
		}
		//_costFunctionValues[iSearchX] += Y4_T + Y4_I;
		_Y4_T[iSearchX]    += Y4_T;
		_Y4_I[iSearchX]    += Y4_I;
		_mean_T[iSearchX]  += Y1_T;
		_mean_I[iSearchX]  += Y1_I;
		_sigma_T[iSearchX] += Y2_T;
		_sigma_I[iSearchX] += Y2_I;
	}
	
	_nSamples++;
	return;
}

const gsl_vector_complex* IIDThresholdEstimator::next(int frameX)
{
	if (frameX == _frameX) return _vector;
	
	const gsl_vector_complex* ad_X_L;
	const gsl_vector_complex* ad_X_R;
	
	if( frameX >= 0 ){
		ad_X_L = _srcL->next(frameX);
		ad_X_R = _srcR->next(frameX);
	}
	else{
		if( _frameX == FrameResetX ){
			ad_X_L = _srcL->next(0);
			ad_X_R = _srcR->next(0);
		}
		else{
			ad_X_L = _srcL->next(_frameX+1);
			ad_X_R = _srcR->next(_frameX+1);
		}
	}
	accumStats1( ad_X_L, ad_X_R );
	
	_increment();
	return _vector;
}

double IIDThresholdEstimator::calcThreshold()
{
  float argMin = _minThreshold;
  double min_rho = 1000000;  
  unsigned iSearchX=0;

  for(float threshold=_minThreshold;threshold<=_maxThreshold;threshold+=_width,iSearchX++){
    double rho, sig2;

    _mean_T[iSearchX]  /= _nSamples;
    _mean_I[iSearchX]  /= _nSamples;
    _sigma_T[iSearchX] /= _nSamples;
    _sigma_I[iSearchX] /= _nSamples;
    _Y4_T[iSearchX]    /= _nSamples;
    _Y4_I[iSearchX]    /= _nSamples;
    sig2 = _sigma_T[iSearchX] + _sigma_I[iSearchX];
    _costFunctionValues[iSearchX] = ( _Y4_T[iSearchX] + _Y4_I[iSearchX] ) - _beta * sig2 * sig2;
    rho = - _costFunctionValues[iSearchX]; /* negative kurtosis */
    //fprintf(stderr,"%f %e\n",threshold,rho);
    if( rho < min_rho ){
      argMin = threshold;
      min_rho = rho;
    }
  }

  _threshold = argMin;
  _isCostFunctionComputed = true;
  return argMin;
}

void IIDThresholdEstimator::reset()
{
	_srcL->reset();
	_srcR->reset();
	VectorComplexFeatureStream::reset();
	_endOfSamples = false;
	
	unsigned iSearchX=0;
	for(float threshold=_minThreshold;threshold<=_maxThreshold;threshold+=_width,iSearchX++){
		_costFunctionValues[iSearchX] = 0.0;
		_Y4_T[iSearchX] = 0.0;
		_Y4_I[iSearchX] = 0.0;
		_mean_T[iSearchX]  = 0.0;
		_mean_I[iSearchX]  = 0.0;
		_sigma_T[iSearchX] = 0.0;
		_sigma_I[iSearchX] = 0.0;
	}
	_nSamples = 0;
	_isCostFunctionComputed = false;
}


// ----- definition for class 'FDIIDThresholdEstimator' -----
// 

/**
   @brief construct this object
   @param VectorComplexFeatureStreamPtr srcL[in]
   @param VectorComplexFeatureStreamPtr srcR[in]
   @param unsigned M[in] the FFT-poit
   @param float minThreshold
   @param float maxThreshold
   @param float width
   @param float threshold[in] threshold for the ITD
   @param float alpha[in] forgetting factor
   @param float dEta[in] flooring value 
   @param float dPowerCoeff[in] power law non-linearity
*/
FDIIDThresholdEstimator::FDIIDThresholdEstimator(VectorComplexFeatureStreamPtr &srcL, VectorComplexFeatureStreamPtr &srcR, unsigned M, 
						 float minThreshold, float maxThreshold, float width,
						 float dEta, float dPowerCoeff, const String& nm ):
  BinaryMaskFilter( 0, srcL, srcR, M, 0.0, 0.0 /* alha must be zero */, dEta ),
  _width(width),
  _isCostFunctionComputed(false),
  _dPowerCoeff(dPowerCoeff),
  _nCand(0)
{
  if( srcL->size() != M ){
    fprintf(stderr,"Left input block length (%d) != M (%d)\n", srcL->size(), M );
    throw jdimension_error("Left input block length (%d) != M (%d)\n", srcL->size(), M );
  }
  
  if( srcR->size() != M ){
    fprintf(stderr,"Right input block length (%d) != M (%d)\n", srcR->size(), M );
    throw jdimension_error("Right input block length (%d) != M (%d)\n", srcR->size(), M );
  }

  if( minThreshold == maxThreshold ){
    _minThreshold = - 100000;
    _maxThreshold =   100000;
  }
  else{
    _minThreshold = minThreshold;
    _maxThreshold = maxThreshold;
  }

  unsigned fftLen2 = M/2;
  int nCand = (int)( (_maxThreshold-_minThreshold)/width + 1.5 );
  _nCand = (unsigned int)nCand;

  _costFunctionValues = (double **)malloc((fftLen2+1)*sizeof(double *));
  _Y4    = (double **)malloc((fftLen2+1)*sizeof(double *));
  _sigma = (double **)malloc((fftLen2+1)*sizeof(double *));
  _mean  = (double **)malloc((fftLen2+1)*sizeof(double *));
  if( NULL==_costFunctionValues || NULL==_Y4 || NULL==_sigma || NULL==_mean ){
    fprintf(stderr,"FDIIDThresholdEstimator:cannot allocate memory\n");
    throw jallocation_error("FDIIDThresholdEstimator:cannot allocate memory\n");    
  }
  
  _costFunctionValues[0] = (double *)calloc((fftLen2+1)*_nCand,sizeof(double));
  _Y4[0]    = (double *)calloc((fftLen2+1)*_nCand,sizeof(double));
  _sigma[0] = (double *)calloc((fftLen2+1)*_nCand,sizeof(double));
  _mean[0]  = (double *)calloc((fftLen2+1)*_nCand,sizeof(double));
  if( NULL==_costFunctionValues[0] || NULL==_Y4[0] || NULL==_sigma[0] || NULL==_mean[0] ){
    fprintf(stderr,"FDIIDThresholdEstimator:cannot allocate memory\n");
    throw jallocation_error("FDIIDThresholdEstimator:cannot allocate memory\n");    
  }
  for(unsigned int fbinX=1;fbinX<=fftLen2;fbinX++){
    _costFunctionValues[fbinX] = &_costFunctionValues[0][fbinX*_nCand];
    _Y4[fbinX]    = &(_Y4[0][fbinX*_nCand]);
    _sigma[fbinX] = &(_sigma[0][fbinX*_nCand]);
    _mean[fbinX]  = &(_mean[0][fbinX*_nCand]);
  }

  _buffer = gsl_vector_alloc( _nCand );

  _thresholdAtFreq = gsl_vector_alloc( fftLen2+1 );

  _nSamples = 0;
}

FDIIDThresholdEstimator::~FDIIDThresholdEstimator()
{
  free( _costFunctionValues[0] );
  free( _Y4[0] );
  free( _sigma[0] );
  free( _mean[0] );
  free( _costFunctionValues );
  free( _Y4 );
  free( _sigma );
  free( _mean );
  if( NULL != _buffer ){
    gsl_vector_free( _buffer );
    _buffer = NULL;
  }
  if( NULL != _thresholdAtFreq ){
    gsl_vector_free( _thresholdAtFreq );
    _thresholdAtFreq = NULL;
  }
}

const gsl_vector* FDIIDThresholdEstimator::getCostFunction( unsigned freqX )
{
  if( _isCostFunctionComputed == false ){
    fprintf(stderr,"FDIIDThresholdEstimator:call calcThreshold() first\n");
  }
  for(unsigned int iSearchX=0;iSearchX<_nCand;iSearchX++){
    gsl_vector_set( _buffer, iSearchX, _costFunctionValues[freqX][iSearchX] );
  }

  return (const gsl_vector*)_buffer;
}

/*
  @brief calculate kurtosis of beamformer's outputs
*/
void FDIIDThresholdEstimator::accumStats1( const gsl_vector_complex* ad_X_L, const gsl_vector_complex* ad_X_R )
{
  unsigned fftLen2  = (unsigned)ad_X_L->size / 2;
	
  //fprintf(stderr,"IIDThresholdEstimator::accumStats1 %d\n",_frameX);
  for(unsigned fbinX=1;fbinX<=fftLen2;fbinX++){
    unsigned iSearchX=0;
    for(float threshold=_minThreshold;threshold<=_maxThreshold;threshold+=_width,iSearchX++){
      gsl_complex X_T_f,  X_I_f;
      double      P_T_f,  P_I_f;
      double      mu_T,   mu_I;
      double      Y1_T_f, Y1_I_f; /* the magnitude of the binary masked value */
      double      Y2_T_f, Y2_I_f; /* the power of the binary masked value */
      
      X_T_f = gsl_vector_complex_get( ad_X_L, fbinX );
      X_I_f = gsl_vector_complex_get( ad_X_R, fbinX );
      P_T_f = gsl_complex_abs( X_T_f );
      P_I_f = gsl_complex_abs( X_I_f );
      
      if( P_T_f <= ( P_I_f + threshold ) )
	mu_T = _dEta; //_alpha * gsl_vector_float_get( _prevMu, fbinX ) + ( 1 - _alpha ) * _dEta;
      else
	mu_T = 1.0;   //_alpha * gsl_vector_float_get( _prevMu, fbinX ) + ( 1 - _alpha ) ;
      
      if( P_I_f <= ( P_T_f + threshold ) )
	mu_I = _dEta; //_alpha * gsl_vector_float_get( _prevMu, fbinX ) + ( 1 - _alpha ) * _dEta;
      else
	mu_I = 1.0;   //_alpha * gsl_vector_float_get( _prevMu, fbinX ) + ( 1 - _alpha ) ;
      
      Y1_T_f = pow( gsl_complex_abs( gsl_complex_mul_real( X_T_f, mu_T ) ), (double)2.0*_dPowerCoeff );
      Y1_I_f = pow( gsl_complex_abs( gsl_complex_mul_real( X_I_f, mu_I ) ), (double)2.0*_dPowerCoeff );
      //Y1_T_f = gsl_complex_abs( gsl_complex_mul_real( X_T_f, mu_T ) );
      //Y1_I_f = gsl_complex_abs( gsl_complex_mul_real( X_I_f, mu_I ) );
      Y2_T_f = Y1_T_f * Y1_T_f;
      Y2_I_f = Y1_I_f * Y1_I_f;
      _Y4[fbinX][iSearchX]    += Y2_T_f * Y2_T_f + Y2_I_f *Y2_I_f;
      _mean[fbinX][iSearchX]  += Y1_T_f          + Y1_I_f;
      _sigma[fbinX][iSearchX] += Y2_T_f          + Y2_I_f;
    }
    //_costFunctionValues[iSearchX] += Y4_T + Y4_I;
  }
  
  _nSamples++;
  return;
}

const gsl_vector_complex* FDIIDThresholdEstimator::next(int frameX)
{
  if (frameX == _frameX) return _vector;
	
  const gsl_vector_complex* ad_X_L;
  const gsl_vector_complex* ad_X_R;
	
  if( frameX >= 0 ){
    ad_X_L = _srcL->next(frameX);
    ad_X_R = _srcR->next(frameX);
  }
  else{
    if( _frameX == FrameResetX ){
      ad_X_L = _srcL->next(0);
      ad_X_R = _srcR->next(0);
    }
    else{
      ad_X_L = _srcL->next(_frameX+1);
      ad_X_R = _srcR->next(_frameX+1);
    }
  }
  accumStats1( ad_X_L, ad_X_R );
  
  _increment();
  return _vector;
}

double FDIIDThresholdEstimator::calcThreshold()
{
  unsigned fftLen2 = (unsigned)_srcL->size() / 2;
  double min_rho   = 1000000;  

  for(unsigned int fbinX=1;fbinX<=fftLen2;fbinX++){	
    unsigned iSearchX=0;
    double local_min_rho = 1000000;  

    for(float threshold=_minThreshold;threshold<=_maxThreshold;threshold+=_width,iSearchX++){
      double rho;

      _mean[fbinX][iSearchX]  /= _nSamples;
      _sigma[fbinX][iSearchX] /= _nSamples;
      _Y4[fbinX][iSearchX]    /= _nSamples;
      _costFunctionValues[fbinX][iSearchX] = _Y4[fbinX][iSearchX] - _beta * _sigma[fbinX][iSearchX] * _sigma[fbinX][iSearchX];
      rho = - _costFunctionValues[fbinX][iSearchX]; /* negative kurtosis */
      //fprintf(stderr,"%f %e\n",threshold,rho);
      if( rho <= min_rho ){
	_threshold = threshold;
	min_rho = rho;
      }
      if( rho <= local_min_rho ){
	gsl_vector_set( _thresholdAtFreq, fbinX, threshold );
	local_min_rho = rho;
      }
    }
  }

  _isCostFunctionComputed = true;
  return _threshold;
}

void FDIIDThresholdEstimator::reset()
{
  unsigned fftLen2 = (unsigned)_srcL->size()/2;

  _srcL->reset();
  _srcR->reset();
  VectorComplexFeatureStream::reset();
  _endOfSamples = false;

  for(unsigned int fbinX=1;fbinX<=fftLen2;fbinX++){	
    unsigned iSearchX=0;

    for(float threshold=_minThreshold;threshold<=_maxThreshold;threshold+=_width,iSearchX++){
      _costFunctionValues[fbinX][iSearchX] = 0.0;
      _Y4[fbinX][iSearchX]    = 0.0;
      _mean[fbinX][iSearchX]  = 0.0;
      _sigma[fbinX][iSearchX] = 0.0;
    }
  }

  _nSamples = 0;
  _isCostFunctionComputed = false;
}
