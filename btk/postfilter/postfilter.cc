#include "postfilter/postfilter.h"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_sf_trig.h>

/**
   @brief calculate cross spectral density (CSD).
 */
gsl_complex calcCSD( gsl_complex &xi, gsl_complex &xj,  gsl_complex &prevPhi, double alpha )
{
  gsl_complex xjA   = gsl_complex_conjugate( xj );
  gsl_complex xixjA = gsl_complex_mul( xi, xjA );

  if( alpha > 0.0 ){
    gsl_complex currPhi;
   
    currPhi = gsl_complex_add( gsl_complex_mul_real( prevPhi, alpha ), gsl_complex_mul_real( xixjA, ( 1.0 - alpha ) ) );
    return(currPhi); 
  }
    
  return(xixjA);
}

/** 
    @brief this function compensates a time delay for the signal at each channel.
    @param const gsl_vector_complex *arrayManifold[in] 
    @param const gsl_vector_complex *snapShot[in]
    @param int nChan[in]
    @param gsl_vector_complex *output[out]
*/
void TimeAlignment(  const gsl_vector_complex *arrayManifold_f,
		     const gsl_vector_complex *snapShot_f, 
		     int nChan,
		     gsl_vector_complex *output )
{    
  gsl_complex dsf, xsf, ysf;
  
  for(int i=0;i<nChan;i++){
    dsf = gsl_complex_conjugate( gsl_vector_complex_get( arrayManifold_f, i ) );
    xsf = gsl_vector_complex_get( snapShot_f, i );
    ysf = gsl_complex_mul( dsf, xsf );
    gsl_vector_complex_set( output, i, ysf );
  }
}

/**
   @brief This methods enhances the signal with Zelinski post-filter.

   The filter corresponds to Eq.(4) in [1].
   This post-filter gave better noise reduction [1].
   
   @param gsl_vector_complex *snapShot[in] the complex signal at a frequency 
   @param int nChan[in] the number of channels, snapShot[nChan]
   @param void *pvPrevCSD[in]
   @return the window value
*/
#define SPECTRAL_FLOOR 0.0001 // to avoid  very small transfer function components
double ZelinskiFilter_f(const gsl_vector_complex *arrayManifold,
			const gsl_vector_complex *snapShot, 
			int nChan,
			gsl_vector_complex *prevCSDf, double alpha, int pfType )
{
  double W1f = 1.0;

  if( 1 >= nChan ){
    fprintf(stderr,"ZelinskiFilter:The number of channels %d is <= 1 \n",nChan);
    throw jdimension_error("The number of channels %d is <= 1 \n",nChan);
  }
  else{
    int idx;
    double numerator=0.0, denominator=0.0;
    double estPSD, prevPSD;
    gsl_complex estCSD, sum, xi, xj, prevPhi;
    gsl_vector_complex *timeAlignedSignalf = gsl_vector_complex_calloc( nChan );
    
    // estimate the CSD of the desired signal
    TimeAlignment( arrayManifold, snapShot, nChan, timeAlignedSignalf );
    GSL_SET_COMPLEX( &sum, 0, 0 );
    for(int i=0;i<nChan-1;i++){
      for(int j=i+1;j<nChan;j++){
	idx = i * nChan + j;
	xi = gsl_vector_complex_get( timeAlignedSignalf, i );
	xj = gsl_vector_complex_get( timeAlignedSignalf, j );
	prevPhi = gsl_vector_complex_get( prevCSDf, idx );
	estCSD = calcCSD( xi, xj, prevPhi, alpha );
	sum = gsl_complex_add ( sum, estCSD );
	gsl_vector_complex_set( prevCSDf, idx, estCSD );
      }
    }

    // the real operation is performed 
    if( TYPE_ZELINSKI1_REAL & pfType ){
      numerator = GSL_REAL( sum );
      if( numerator < 0.0 )
	numerator = 0.0;
      //fprintf(stderr,"Zelinski post-filter  1 is used\n");
    }
    else{
      numerator = gsl_complex_abs( sum );
    }
    
    {  // calculate the PSD of the noisy signal
      gsl_complex tmpC;

      for(int i=0;i<nChan;i++){
	idx = i * nChan + i;
	xi = gsl_vector_complex_get( timeAlignedSignalf, i );
	if( alpha > 0.0 ){
	  prevPSD = GSL_REAL( gsl_vector_complex_get( prevCSDf, idx ) );
	  estPSD = alpha * prevPSD + (1.0-alpha) * gsl_complex_abs2( xi );
	}
	else
	  estPSD = gsl_complex_abs2( xi );
	denominator += estPSD;
	GSL_SET_COMPLEX( &tmpC, estPSD, 0.0 );
	gsl_vector_complex_set( prevCSDf, idx, tmpC );
      }
    }
    
    W1f = ( numerator / denominator ) * ( 2.0 / ( nChan - 1.0 ) );
    // to avoid artificial amplification,
    if( W1f >=  1.0 ) W1f =  1.0;
    if( W1f < SPECTRAL_FLOOR ) W1f = SPECTRAL_FLOOR;
    
#ifdef _DEBUG_PRINT_
    fprintf(stderr,"%e=%f/%f (%f,%f)\n", W1f, numerator, denominator,
	    GSL_REAL(sum), GSL_IMAG(sum) );
    {
      gsl_complex tmpsum;
      GSL_SET_COMPLEX( &tmpsum, 0, 0 );
      for(int i=0;i<nChan;i++){
	tmpsum = gsl_complex_add( tmpsum, gsl_vector_complex_get( timeAlignedSignalf, i ) );
      }
      tmpsum = gsl_complex_mul_real( tmpsum, 1.0/nChan );
    }
#endif

    gsl_vector_complex_free( timeAlignedSignalf );
  }

  return(W1f);
}


/**
   @brief 

   @param gsl_vector_complex **arrayManifold [in]
   @param SnapShotArrayPtr     snapShotArray [in] 
   @param int fftLen [in] 
   @param bool halfBandShift [in]
   @param int nChan  [in]
   @param gsl_vector_complex *beamformedSignal[in/out] 
   @param gsl_vector_complex **prevCSDf [in/out] 
   @param gsl_vector_complex *pfweights  [out]  the weights of postfilter
   @param double alpha [in] 
   @param int pfType [in] You can select a real operator in Eq. (4). If pfType==TYPE_ZELINSKI1_REAL, the real value of the sum of cross spectral densities is taken. If pfType==TYPE_ZELINSKI1_ABS, the absolute of the sum of cross spectral densities is taken. If pfType==NO_USE_POST_FILTER, *prevCSDf is just updated.
*/
void ZelinskiFilter(gsl_vector_complex **arrayManifold,
		    SnapShotArrayPtr     snapShotArray, 
		    bool halfBandShift, 
		    gsl_vector_complex *beamformedSignal,
		    gsl_vector_complex **prevCSDs, 
		    gsl_vector_complex *pfweights,
		    double alpha, int pfType )
{
  int fftLen = snapShotArray->fftLen();
  int nChan  = snapShotArray->nChan();
  unsigned fftLen2 = fftLen/2;
  gsl_complex wf;
  double r;

  // calculate coefficients of the post filter 
  if( halfBandShift==true ){
    for(int fbinX=0;fbinX<fftLen;fbinX++){
      const gsl_vector_complex* propagation = arrayManifold[fbinX];
      const gsl_vector_complex* snapShot    = snapShotArray->getSnapShot(fbinX);
      gsl_vector_complex* prevCSDf          = prevCSDs[fbinX];
    
      r = ZelinskiFilter_f( propagation, snapShot, nChan, prevCSDf, alpha, pfType );
      wf = gsl_complex_polar( r, 0 );
      gsl_vector_complex_set( pfweights, fbinX, wf );
    }
  } else {
    // calculate weights by using the property of the symmetry.
    for(int fbinX=0;fbinX<=fftLen2;fbinX++){
      const gsl_vector_complex* propagation = arrayManifold[fbinX];
      const gsl_vector_complex* snapShot    = snapShotArray->getSnapShot(fbinX);
      gsl_vector_complex* prevCSDf          = prevCSDs[fbinX];
    
      r = ZelinskiFilter_f( propagation, snapShot, nChan, prevCSDf, alpha, pfType );
      wf = gsl_complex_polar( r, 0 );
      gsl_vector_complex_set( pfweights, fbinX, wf );
      if( fbinX > 0 && fbinX < fftLen2 )// substitute a conjugate component
	gsl_vector_complex_set( pfweights, fftLen - fbinX, gsl_complex_conjugate(wf) );
    }
  }
  
  if( NO_USE_POST_FILTER == pfType ){// CSD is just updated.
    return;
  }
    
  // filter the wave
  if( halfBandShift==true ){
    for(int fbinX=0;fbinX<fftLen;fbinX++){
      wf = gsl_vector_complex_get( pfweights, fbinX );
      gsl_complex outf = gsl_complex_mul( wf, gsl_vector_complex_get( beamformedSignal, fbinX ) );
      gsl_vector_complex_set( beamformedSignal, fbinX, outf );
    }
  } else {
    for(int fbinX=0;fbinX<=fftLen2;fbinX++){
      wf = gsl_vector_complex_get( pfweights, fbinX );
      gsl_complex outf = gsl_complex_mul( wf, gsl_vector_complex_get( beamformedSignal, fbinX ) );
      gsl_vector_complex_set( beamformedSignal, fbinX, outf );
      if( fbinX > 0 && fbinX < fftLen2 ){// substitute a conjugate component
	gsl_vector_complex_set( beamformedSignal, fftLen - fbinX, gsl_complex_conjugate( outf ) );
      }
    }
  }

}

/**
   @brief the adaptive post-filter for an arbitrary beamformer(APAB)
*/
double ApabFilter_f( const gsl_vector_complex *propagation,
		     const gsl_vector_complex *snapShot, 
		     int nChan,
		     gsl_complex beamFormedSignalf,
		     int channelX )
{
  double phi_yy;
  double phi_xx;
  double Wf;

  phi_yy = gsl_complex_abs2( beamFormedSignalf );
  if( channelX < 0 ){// averaging the time aligned signals == delay-and-sum
    //fprintf(stderr,"APAB post-filter 1 is used\n");
    gsl_complex avgSignalf; 

    gsl_blas_zdotc( propagation, snapShot, &avgSignalf );
    phi_xx = gsl_complex_abs2( avgSignalf );
  }
  else{ // choose one channel
    //fprintf(stderr,"APAB post-filter 2 is used\n");
    gsl_complex dsf, xsf, ysf;

    if( channelX >= nChan ){
      fprintf(stderr,"ApabFilter:Invalid channel ID %d \n",channelX);
      throw jdimension_error("Invalid channel ID %d \n",channelX);
    }
    dsf = gsl_complex_conjugate( gsl_vector_complex_get( propagation, channelX ) );
    xsf = gsl_vector_complex_get( snapShot, channelX );
    ysf = gsl_complex_mul( dsf, xsf );
    phi_xx = gsl_complex_abs2(  ysf );
  }

  Wf = phi_yy / phi_xx;
  // to avoid artificial amplification,
  if( Wf >=  1.0 ) Wf =  1.0;
  if( Wf <= -1.0 ) Wf = -1.0;
  
  return(Wf);
}

/**
   @brief the adaptive post-filter for an arbitrary beamformer(APAB)
  
   @param gsl_vector_complex **arrayManifold[in]
   @param SnapShotArrayPtr     snapShotArray[in]
   @param int fftLen[in]
   @param int nChan[in]
   @param gsl_vector_complex *beamformedSignal[in/out]
   @param int channelX[in]
*/
void ApabFilter( gsl_vector_complex **arrayManifold,
		 SnapShotArrayPtr     snapShotArray, 
		 int fftLen, int nChan, bool halfBandShift,
		 gsl_vector_complex *beamformedSignal,
		 int channelX )
{
  int fftLen2 = fftLen / 2;
  double *window = new double[2*fftLen];
  gsl_complex beamFormedSignalf;
  gsl_complex wf, outf;
  gsl_vector_complex* windowV;
  int length;

  if( channelX < 0 )
    channelX = nChan / 2;

  for(int fbinX=0;fbinX<fftLen2;fbinX++){
    const gsl_vector_complex* propagation = arrayManifold[fbinX];
    const gsl_vector_complex* snapShot = snapShotArray->getSnapShot(fbinX);
    double r;
    gsl_complex wf;

    beamFormedSignalf = gsl_vector_complex_get( beamformedSignal, fbinX );
    r = ApabFilter_f( propagation, snapShot, nChan, beamFormedSignalf, channelX );
    wf = gsl_complex_polar( r, 0 );
    window[2*fbinX]   = GSL_REAL(wf);
    window[2*fbinX+1] = GSL_IMAG(wf);
  }

  // remove the aliasing and convert the data structure
  //MyRefineFilter( window, fftLen2 );
  windowV = gsl_vector_complex_calloc(fftLen);
  if( halfBandShift ){
    // make an mirror image
    for(int fbinX=0;fbinX<fftLen2;fbinX++){
      GSL_SET_COMPLEX( &wf, window[2*fbinX], window[2*fbinX+1]);
      gsl_vector_complex_set( windowV, fbinX,          wf );
      gsl_vector_complex_set( windowV, fftLen-1-fbinX, gsl_complex_conjugate(wf) );      
    }
  }
  else{
    for(int fbinX=0;fbinX<fftLen2;fbinX++){
      GSL_SET_COMPLEX( &wf, window[2*fbinX], window[2*fbinX+1]);
      gsl_vector_complex_set( windowV, fbinX,          wf );
    }
  }
  delete [] window;

  // filter the wave  
  if( halfBandShift )
    length = fftLen;    
  else
    length = fftLen2;
  for(int fbinX=0;fbinX<length;fbinX++){
    gsl_complex outf;

    wf = gsl_vector_complex_get( windowV, fbinX );
    //wf = gsl_complex_abs( wf );
    outf = gsl_complex_mul( wf, gsl_vector_complex_get( beamformedSignal, fbinX ) );
    gsl_vector_complex_set( beamformedSignal, fbinX, outf );
  }

  gsl_vector_complex_free( windowV );
}

/**
   @brief construct an object for Zelinski post-filtering
   @param VectorComplexFeatureStreamPtr &output[in] The output of the beamformer
   @param unsigned M[in] no. subbands (must be the same as output->size)
   @param double alpha[in] forgetting factor
   @param int type[in] type of the operation for the cross correlation of the PSD.
                       if type==1, Re(*) and if type==2, |*|.
   @param unsigned minFrames[in]
   @notice the object of the beamformer must not use post-filtering.
 */
ZelinskiPostFilter::ZelinskiPostFilter( VectorComplexFeatureStreamPtr &output, unsigned fftLen, double alpha, int type, int minFrames, const String& nm ):
  VectorComplexFeatureStream(fftLen, nm),
  _fftLen(fftLen),
  _samp(output),
  _alpha(alpha),
  _minFrames(minFrames),
  _bfWeightsPtr(NULL),
  _hasBeamformerPtr(false)
{
  if( output->size() != fftLen ){
    fprintf(stderr,"Input block length (%d) != fftLen (%d)\n", output->size(), fftLen );
    throw jdimension_error("Input block length (%d) != fftLen (%d)\n", output->size(), fftLen );
  }

  _type = (PostfilterType)type;
}

ZelinskiPostFilter::~ZelinskiPostFilter()
{
  if( false ==  _hasBeamformerPtr ){
    if( NULL != _bfWeightsPtr ){
      delete _bfWeightsPtr;
    }
  }
}

void ZelinskiPostFilter::setBeamformer( SubbandDSPtr &bfptr )
{
  if( false ==  _hasBeamformerPtr && NULL != _bfWeightsPtr ){
    delete _bfWeightsPtr;
    _bfWeightsPtr = NULL;
  }
  _hasBeamformerPtr = true;
  _beamformerPtr = bfptr;
  //_samp = (VectorComplexFeatureStreamPtr)(bfptr);
}

void ZelinskiPostFilter::setSnapShotArray( SnapShotArrayPtr &snapShotArray )
{
  _snapShotArray = snapShotArray;
}

/**
   @brief 
   @param gsl_matrix_complex *arrayManifold[in] arrayManifold[fftLen][chanN]
 */
void ZelinskiPostFilter::setArrayManifoldVector( unsigned fbinX, gsl_vector_complex *arrayManifoldVector, bool halfBandShift, unsigned NC )
{
  if( fbinX >= size() ){
    fprintf(stderr,"fbinX %d must be less than %d\n", fbinX, size() );
    throw jdimension_error("fbinX %d must be less than %d\n", fbinX, size() );
  }

  unsigned chanN = arrayManifoldVector->size;
  gsl_vector_complex** wq;

  if( NULL == _bfWeightsPtr ){
    _bfWeightsPtr = new beamformerWeights( size(), chanN, halfBandShift, NC );
  }

  if( (int)TYPE_ZELINSKI2 & _type ){
    wq = _bfWeightsPtr->wq();
  }
  else{
    wq = _bfWeightsPtr->arrayManifold();
  }
  for(unsigned chanX=0;chanX<chanN;chanX++){
    gsl_vector_complex_set( wq[fbinX], chanX, gsl_vector_complex_get( arrayManifoldVector, chanX ) );
  }
}

void ZelinskiPostFilter::reset()
{
  _samp->reset();
  VectorComplexFeatureStream::reset();
  _endOfSamples = false;
}

const gsl_vector_complex* ZelinskiPostFilter::next(int frameX)
{
  //fprintf(stderr,"pf %d %d %d\n",frameX,_frameX, FrameResetX); 
  if (frameX == _frameX) return _vector;

  const gsl_vector_complex* output;

  if( frameX >= 0 )
    output = _samp->next(frameX);
  else{
    if( _frameX == FrameResetX )
      output = _samp->next(0);
    else
      output = _samp->next(_frameX+1);
  }

  if( true ==  _hasBeamformerPtr ){
    _snapShotArray = _beamformerPtr->getSnapShotArray();
    _bfWeightsPtr = _beamformerPtr->getBeamformerWeightObject(0);
  }
  if( NULL == _bfWeightsPtr ){
    fprintf(stderr,"set beamformer's weights \n");
    throw  j_error("set beamformer's weights \n");
  }
  
  gsl_vector_complex*  wp1         = _bfWeightsPtr->wp1();
  gsl_vector_complex** prevCSDs    = _bfWeightsPtr->CSDs();
  unsigned fftLen = _bfWeightsPtr->fftLen();
  double alpha;
  gsl_vector_complex** wq;

  if( (int)TYPE_ZELINSKI2 & _type ){
    wq = _bfWeightsPtr->wq(); // just use a beamformer output as a clean signal.
  }
  else{
    wq = _bfWeightsPtr->arrayManifold();
  }

  if( _frameX > 0 )
    alpha =  _alpha;
  else
    alpha = 0.0;

  for (unsigned fbinX = 0; fbinX < fftLen; fbinX++)
    gsl_vector_complex_set(_vector, fbinX, gsl_vector_complex_get( output, fbinX ) );

  if( _frameX < _minFrames ){// just update cross spectral densities
    ZelinskiFilter( wq, _snapShotArray, _bfWeightsPtr->isHalfBandShift(), _vector, prevCSDs, wp1, alpha, (int)NO_USE_POST_FILTER);
  }
  else{
    ZelinskiFilter( wq, _snapShotArray, _bfWeightsPtr->isHalfBandShift(), _vector, prevCSDs, wp1, alpha, _type );
  }

#if 0
  unsigned showChanN = 2;// _snapShotArray->nChan();
  for(unsigned chanX=0;chanX< showChanN;chanX++){
    fprintf(stderr,"AM %f + i%f, ", GSL_REAL(gsl_vector_complex_get( wq[128], chanX)), GSL_IMAG(gsl_vector_complex_get( wq[128], chanX )) ); 
  }
  fprintf(stderr," \n");
  for(unsigned chanX=0;chanX< showChanN;chanX++){
    const gsl_vector_complex* sa = _snapShotArray->getSnapShot(10);
    fprintf(stderr,"SA %f + i%f, ", GSL_REAL(gsl_vector_complex_get( sa, chanX )), GSL_IMAG(gsl_vector_complex_get( sa, chanX )) ); 
  }
  fprintf(stderr," \n");
  fprintf(stderr,"NF %f + i%f \n", GSL_REAL(gsl_vector_complex_get( _vector, 10 )), GSL_IMAG(gsl_vector_complex_get( _vector, 10 )) ); 
#endif

  _increment();
  return _vector;
}

// ----- definition for class `McCowanPostFilter' -----
//

McCowanPostFilter::McCowanPostFilter( VectorComplexFeatureStreamPtr &output, unsigned fftLen, double alpha, int type, int minFrames, float threshold, const String& nm ):
  ZelinskiPostFilter( output, fftLen, alpha, type, minFrames, nm ),
  _thresholdOfRij(threshold),
  _timeAlignedSignalf(NULL),
  _isInvRcomputed(false)
{
  unsigned fftLen2 = _fftLen / 2;

  _R = (gsl_matrix_complex** )malloc( (fftLen2+1) * sizeof(gsl_matrix_complex*) );
  if( _R == NULL ){
    fprintf(stderr,"McCowanPostFilter: gsl_matrix_complex_alloc failed\n");
    throw jallocation_error("McCowanPostFilter: gsl_matrix_complex_alloc failed\n");
  }

  _diagonalWeights = (float *)calloc( (fftLen2+1), sizeof(float) );
  if( _diagonalWeights == NULL ){
    fprintf(stderr,"McCowanPostFilter: cannot allocate RAM\n");
    throw jallocation_error("McCowanPostFilter: cannot allocate RAM\n");
  }

  for( unsigned fbinX=0;fbinX<=fftLen2;fbinX++){
    _R[fbinX] = NULL;
  }
}
 
McCowanPostFilter::~McCowanPostFilter()
{
  //fprintf(stderr,"~McCowanPostFilter()\n");
  unsigned fftLen2 = _fftLen / 2;
  
  for( unsigned fbinX=0;fbinX<=fftLen2;fbinX++){
    if( NULL!=_R[fbinX] )
      gsl_matrix_complex_free( _R[fbinX] );
  }
  free(_R);
  free(_diagonalWeights);

  if( NULL != _timeAlignedSignalf )
    gsl_vector_complex_free( _timeAlignedSignalf );
}

const gsl_matrix_complex *McCowanPostFilter::getNoiseSpatialSpectralMatrix( unsigned fbinX )
{
  return _R[fbinX];
}

bool McCowanPostFilter::setNoiseSpatialSpectralMatrix( unsigned fbinX, gsl_matrix_complex* Rnn )
{
  if( Rnn->size1 != Rnn->size2 ){
    fprintf(stderr,"The noise coherence matrix should be the square matrix\n");
    return false;
  }
  
  unsigned chanN = Rnn->size1;

  if( _R[fbinX] == NULL ){
    _R[fbinX] = gsl_matrix_complex_alloc( chanN, chanN );
  }

  for(unsigned m=0;m<chanN;m++){
    for(unsigned n=0;n<chanN;n++){
      gsl_matrix_complex_set( _R[fbinX], m, n, gsl_matrix_complex_get( Rnn, m, n ) );
    }
  }
  _isInvRcomputed = false;
  return true;
}

bool McCowanPostFilter::setDiffuseNoiseModel( const gsl_matrix* micPositions, double sampleRate, double sspeed )
{
  size_t chanN   = micPositions->size1;
  gsl_matrix *dm = gsl_matrix_alloc( chanN, chanN );
  
  if( micPositions->size2 < 3 ){
    fprintf(stderr,"The microphone positions should be described in the three dimensions\n");
    return false;
  }

  for(unsigned fbinX=0;fbinX<=_fftLen/2;fbinX++){
    if( _R[fbinX] == NULL ){
      _R[fbinX] = gsl_matrix_complex_alloc( chanN, chanN );
    }
  }

  {// calculate the distance matrix.
     for(unsigned m=0;m<chanN;m++){
       for(unsigned n=0;n<m;n++){ //for(unsigned n=0;n<chanN;n++){
	 double Xm = gsl_matrix_get( micPositions, m, 0 );
	 double Xn = gsl_matrix_get( micPositions, n, 0 );
	 double Ym = gsl_matrix_get( micPositions, m, 1 );
	 double Yn = gsl_matrix_get( micPositions, n, 1 );
	 double Zm = gsl_matrix_get( micPositions, m, 2 );
	 double Zn = gsl_matrix_get( micPositions, n, 2 );

	 double dx = Xm - Xn;
	 double dy = Ym - Yn;
	 double dz = Zm - Zn;
	 gsl_matrix_set( dm, m, n, sqrt( dx * dx + dy * dy + dz * dz ) );
       }
     }
     //for(unsigned m=0;m<chanN;m++){ gsl_matrix_set( dm, m, m, 0.0 );}
  }

  {
    for(unsigned fbinX=0;fbinX<=_fftLen/2;fbinX++){
      //double omega_d_c = 2.0 * M_PI * sampleRate * fbinX / ( _fftLen * sspeed );
      double omega_d_c = 2.0 * sampleRate * fbinX / ( _fftLen * sspeed );

      for(unsigned m=0;m<chanN;m++){
	for(unsigned n=0;n<m;n++){ 
	  double Gamma_mn = gsl_sf_sinc( omega_d_c * gsl_matrix_get( dm, m, n ) );
	  gsl_matrix_complex_set( _R[fbinX], m, n, gsl_complex_rect( Gamma_mn, 0.0 ) );
	}// for(unsigned n=0;n<chanN;n++){ 
      }// for(unsigned m=0;m<chanN;m++){
      for(unsigned m=0;m<chanN;m++){ 
	 gsl_matrix_complex_set( _R[fbinX], m, m, gsl_complex_rect( 1.0, 0.0 ) );
      }
      for(unsigned m=0;m<chanN;m++){
	for(unsigned n=m+1;n<chanN;n++){
	  gsl_complex val = gsl_matrix_complex_get( _R[fbinX], n, m );
	  gsl_matrix_complex_set( _R[fbinX], m, n, val );
	}
      }
    }// for(unsigned fbinX=0;fbinX<=_fftLen/2;fbinX++){
  }
  //gsl_sf_sinc (double x);
  
  gsl_matrix_free(dm);
  _isInvRcomputed = false;

  return true;
};

void McCowanPostFilter::setAllLevelsOfDiagonalLoading( float diagonalWeight )
{
  if( _R[0] == NULL ){
    fprintf(stderr,"Construct/set first a noise coherence matrix\n");
    throw j_error("Construct/set first a noise coherence matrix\n");
  }

  unsigned chanN = _R[0]->size1;
  for(unsigned fbinX=0;fbinX<=_fftLen/2;fbinX++){
    _diagonalWeights[fbinX] = diagonalWeight;
    for( unsigned chanX=0 ; chanX < chanN ;chanX++ ){// diagonal loading
      gsl_complex val = gsl_matrix_complex_get( _R[fbinX], chanX, chanX );
      gsl_matrix_complex_set( _R[fbinX], chanX, chanX, gsl_complex_add_real( val, _diagonalWeights[fbinX] ) );
    }
  }
}

void McCowanPostFilter::setLevelOfDiagonalLoading( unsigned fbinX, float diagonalWeight )
{
  if( _R[0] == NULL ){
    fprintf(stderr,"Construct/set first a noise coherence matrix\n");
    throw j_error("Construct/set first a noise coherence matrix\n");
  }

  unsigned chanN = _R[0]->size1;
  _diagonalWeights[fbinX] = diagonalWeight;
  for( unsigned chanX=0 ; chanX < chanN ;chanX++ ){// diagonal loading
    gsl_complex val = gsl_matrix_complex_get( _R[fbinX], chanX, chanX );
    gsl_matrix_complex_set( _R[fbinX], chanX, chanX, gsl_complex_add_real( val, _diagonalWeights[fbinX] ) );
  }
}

/**
   @brief Divide each non-diagonal elemnt by ( 1 + myu ) instead of diagonal loading. 
   myu can be interpreted as the ratio of the sensor noise to the ambient noise power.
   @param float myu[in]
*/
void McCowanPostFilter::divideAllNonDiagonalElements( float myu )
{
  for(unsigned fbinX=0;fbinX<=_fftLen/2;fbinX++){
    divideNonDiagonalElements( fbinX, myu );    
  }
}

void McCowanPostFilter::divideNonDiagonalElements( unsigned fbinX, float myu )
{
  for ( size_t chanX=0; chanX<_R[fbinX]->size1; chanX++ ){
    for ( size_t chanY=0; chanY<_R[fbinX]->size2; chanY++ ){
      if( chanX != chanY ){
	gsl_complex Rxy = gsl_matrix_complex_get( _R[fbinX], chanX, chanY );
	gsl_matrix_complex_set( _R[fbinX], chanX, chanY, gsl_complex_div( Rxy, gsl_complex_rect( (1.0+myu), 0.0 ) ) );
      }
    }
  }
}

void McCowanPostFilter::reset()
{
  //fprintf(stderr,"McCowanPostFilter::reset() 1\n");
  _samp->reset();
  VectorComplexFeatureStream::reset();
  _endOfSamples = false;
}

/*
   @brief calculate the auto and cross spectral densities at a frequnecy bin.
   @param gsl_vector_complex *timeAlignedSignal_f[in] Estimates of the desired signal
   @param gsl_vector_complex **prevCSDf[in/out] PSD at the prevous frame 
                                                PSD will be updated.
   param double alpha[in] forgetting factor for the recursive average
*/
double calculateSpectralDensities_f( gsl_vector_complex *timeAlignedSignal_f, gsl_vector_complex* prevCSDf, double alpha  )
{
  unsigned nChan = timeAlignedSignal_f->size;
  double sumOfPSD = 0.0;
  
  {/* compute the CSD of the estimates of the desired signal, 
     that is, either delay-compensated or beamformed signal */ 
    gsl_complex estCSD;

    for(unsigned i=0;i<nChan-1;i++){
      for(unsigned j=i+1;j<nChan;j++){
	unsigned idx = i * nChan + j;
	gsl_complex x_i = gsl_vector_complex_get( timeAlignedSignal_f, i );
	gsl_complex x_j = gsl_vector_complex_get( timeAlignedSignal_f, j );
	gsl_complex prevPhi = gsl_vector_complex_get( prevCSDf, idx );
	estCSD = calcCSD( x_i, x_j, prevPhi, alpha );
	gsl_vector_complex_set( prevCSDf, idx, estCSD );
      }
    }
  }
  {// calculate the PSD of the noisy signal
    double estPSD;
    //gsl_complex tmpC;
    
    for(unsigned i=0;i<nChan;i++){
      unsigned  idx = i * nChan + i;
      gsl_complex x_i = gsl_vector_complex_get( timeAlignedSignal_f, i );
      if( alpha > 0.0 ){
	double prevPSD = GSL_REAL( gsl_vector_complex_get( prevCSDf, idx ) );
	estPSD = alpha * prevPSD + (1.0-alpha) * gsl_complex_abs2( x_i );
      }
      else
	estPSD = gsl_complex_abs2( x_i );
      sumOfPSD += estPSD;
      //GSL_SET_COMPLEX( &tmpC, estPSD, 0.0 );
      gsl_vector_complex_set( prevCSDf, idx, gsl_complex_rect( estPSD, 0.0 ) );
    }
  }

  return ( sumOfPSD / nChan );
}

/*
  @brief estimate the power spectral density of the clean signal and average them over pairs of mics.
  @return the averaged PSD
*/
//#define ORIGINAL_IAIN_PAPER   
#ifdef  ORIGINAL_IAIN_PAPER   
double McCowanPostFilter::estimateAverageOfCleanSignalPSD( unsigned fbinX, gsl_vector_complex* currCSDf )
{
  size_t nChan = _R[fbinX]->size1;
  double nu,de;
  double R_ij;
  double phi_ij;    /* cross power specral density */
  double phi_ii, phi_jj; /* auto power specral density */
  double avg=0.0;

  if( TYPE_ZELINSKI1_REAL & _type ){
    for(size_t i=0;i<nChan-1;i++){
      phi_ii = GSL_REAL( gsl_vector_complex_get( currCSDf, i * nChan + i ) );
      for(size_t j=i+1;j<nChan;j++){
	phi_ij = GSL_REAL( gsl_vector_complex_get( currCSDf, i * nChan + j ) );
	phi_jj = GSL_REAL( gsl_vector_complex_get( currCSDf, j * nChan + j ) );
	R_ij   = GSL_REAL( gsl_matrix_complex_get( _R[fbinX], i, j ) );

	if( R_ij > _thresholdOfRij ){
	  R_ij = _thresholdOfRij;
	}
	else if( R_ij == 1 ){
	  R_ij = 0.99;
	}
	nu = phi_ij - 0.5 * R_ij * ( phi_ii + phi_jj );
	de = 1 - R_ij;
	avg += (nu/de); // phi_ss 
      }
    }
  }
  else{
    for(size_t i=0;i<nChan-1;i++){
      phi_ii = gsl_complex_abs( gsl_vector_complex_get( currCSDf, i * nChan + i ) );
      for(size_t j=i+1;j<nChan;j++){
	phi_ij = gsl_complex_abs( gsl_vector_complex_get( currCSDf, i * nChan + j ) );
	phi_jj = gsl_complex_abs( gsl_vector_complex_get( currCSDf, j * nChan + j ) );
	R_ij   = gsl_complex_abs( gsl_matrix_complex_get( _R[fbinX], i, j ) );

	if( R_ij > _thresholdOfRij ){
	  R_ij = _thresholdOfRij;
	}
	else if( R_ij  == 1 ){
	  R_ij = 0.99;
	}
	//fprintf(stderr,"%d : %d %d :%e - 0.5 * %e ( %e + %e )\n ",fbinX,i,j,phi_ij,R_ij,phi_ii,phi_jj);
	nu = phi_ij - 0.5 * R_ij * ( phi_ii + phi_jj );
	de = 1 - R_ij;
	avg += (nu/de); // phi_ss 
      }
    }
  }

  return ( 2.0*avg/(nChan*(nChan-1)) );
}
#else
double McCowanPostFilter::estimateAverageOfCleanSignalPSD( unsigned fbinX, gsl_vector_complex* currCSDf )
{
  size_t nChan = _R[fbinX]->size1;
  gsl_complex nu,de,val;
  gsl_complex R_ij;
  gsl_complex phi_ij;    /* cross power specral density */
  double phi_ii, phi_jj; /* auto power specral density */
  gsl_complex sum;
  double avg=0.0;

  GSL_SET_COMPLEX( &sum, 0, 0 );
  for(size_t i=0;i<nChan-1;i++){
    phi_ii = GSL_REAL( gsl_vector_complex_get( currCSDf, i * nChan + i ) );
    for(size_t j=i+1;j<nChan;j++){
      phi_ij = gsl_vector_complex_get( currCSDf, i * nChan + j );
      phi_jj = GSL_REAL( gsl_vector_complex_get( currCSDf, j * nChan + j ) );
      R_ij   = gsl_matrix_complex_get( _R[fbinX], i, j );
      
      if( GSL_REAL( R_ij ) > _thresholdOfRij && GSL_IMAG( R_ij ) <= 0.0 ){
	GSL_SET_COMPLEX( &R_ij, _thresholdOfRij, 0 );
      }
      
      val = gsl_complex_mul_real( R_ij, 0.5 * ( phi_ii + phi_jj ) );
      nu = gsl_complex_sub( phi_ij, val ); //nu = phi_ij - 0.5 * R_ij * ( phi_ii + phi_jj );
      de = gsl_complex_add_real( gsl_complex_negative( R_ij ), 1.0 ); // de = 1 - R_ij;
      sum = gsl_complex_add( sum, gsl_complex_div( nu, de ) ); // phi_ss 
    }
  }

  if( TYPE_ZELINSKI1_REAL & _type ){
    avg = GSL_REAL(sum);
  }
  else{
    avg = gsl_complex_abs(sum);
  }

  return ( 2.0*avg/(nChan*(nChan-1)) );
}

#endif

/*
  @brief calculate the weight of the post-filter and its output
  @note In the case that the number of samples are less than _minFrames, the weights are only updated.
*/
void McCowanPostFilter::PostFiltering()
{
  if( NULL == _R[0] ){
    fprintf(stderr,"McCowanPostFilter: construct/set a noise coherence matrix\n");
    throw j_error("McCowanPostFilter:  construct/set a noise coherence matrix\n");
  }
  unsigned nChan  = _snapShotArray->nChan();;
  unsigned fftLen2 = _fftLen / 2;
  gsl_vector_complex*  wp       = _bfWeightsPtr->wp1(); /* post-filter's weights */
  gsl_vector_complex** prevCSDs = _bfWeightsPtr->CSDs();/* auto and cross spectral densities */
  gsl_vector_complex** wq;
  double alpha; /* forgetting factor */

  if( NULL == _timeAlignedSignalf )
    _timeAlignedSignalf = gsl_vector_complex_calloc( nChan );

  if( (int)TYPE_ZELINSKI2 & _type ){
    wq = _bfWeightsPtr->wq(); // just use a beamformer output as a clean signal.
  }
  else{
    wq = _bfWeightsPtr->arrayManifold();
  }

  if( _frameX > 0 )
    alpha =  _alpha;
  else
    alpha = 0.0;

  if(  _bfWeightsPtr->isHalfBandShift()==false ){
    for(unsigned fbinX=0;fbinX<=fftLen2;fbinX++){
      double nu,de, weight;
      const gsl_vector_complex *propagation = wq[fbinX];
      const gsl_vector_complex *snapShot    = _snapShotArray->getSnapShot(fbinX);
      gsl_vector_complex* prevCSDf          = prevCSDs[fbinX];

      TimeAlignment( propagation, snapShot, nChan, _timeAlignedSignalf ); // beamforming
      de = calculateSpectralDensities_f( _timeAlignedSignalf, prevCSDf, alpha  );
      //fprintf(stderr,"de %d %f\n",fbinX,de);
      nu = estimateAverageOfCleanSignalPSD(fbinX,prevCSDf);
      //fprintf(stderr,"%d %d : %e = %e / %e \n",_frameX,fbinX,nu/de,nu,de);
      weight = nu / de;
      if( weight >  1.0 ) weight =  1.0;
      if( weight < SPECTRAL_FLOOR ) weight = SPECTRAL_FLOOR;
      gsl_vector_complex_set( wp, fbinX, gsl_complex_polar( weight, 0 ) );
      if( fbinX > 0 && fbinX < fftLen2 )
	gsl_vector_complex_set( wp, _fftLen - fbinX, gsl_complex_polar( weight, 0 ) );
      
      if( _frameX >= _minFrames ){
	gsl_complex outf = gsl_complex_mul_real( gsl_vector_complex_get( _vector, fbinX ), weight );
	gsl_vector_complex_set(_vector, fbinX, outf );
	if( fbinX > 0 && fbinX < fftLen2 )// substitute a conjugate component
	  gsl_vector_complex_set( _vector, _fftLen - fbinX, gsl_complex_conjugate( outf ) );
      }
    }
  }
  else{
    fprintf(stderr,"McCowanPostFilter: The opition, halfBandShift=True, is not yet implemented\n");
    throw j_error("McCowanPostFilter: The opition, halfBandShift=True, is not yet implemented\n");
  }

}

const gsl_vector_complex* McCowanPostFilter::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  const gsl_vector_complex* output;

  if( frameX >= 0 )
    output = _samp->next(frameX);
  else{
    if( _frameX == FrameResetX )
      output = _samp->next(0);
    else
      output = _samp->next(_frameX+1);
  }

  if( true ==  _hasBeamformerPtr ){
    _snapShotArray = _beamformerPtr->getSnapShotArray();
    _bfWeightsPtr = _beamformerPtr->getBeamformerWeightObject(0);
  }
  if( NULL == _bfWeightsPtr ){
    fprintf(stderr,"set beamformer's weights \n");
    throw  j_error("set beamformer's weights \n");
  }
  
  for (unsigned fbinX = 0; fbinX <= _fftLen/2; fbinX++)
    gsl_vector_complex_set(_vector, fbinX, gsl_vector_complex_get( output, fbinX ) );

  PostFiltering();

  _increment();
  return _vector;
}

// ----- definition for class `LefkimmiatisPostFilter' -----
//
LefkimmiatisPostFilter::LefkimmiatisPostFilter( VectorComplexFeatureStreamPtr &output, unsigned fftLen, double minSV, unsigned fbinX1, double alpha, int type, int minFrames, float threshold, const String& nm ):
  McCowanPostFilter( output, fftLen, alpha, type, minFrames, threshold, nm ),
  _minSV(minSV),
  _fbinX1(fbinX1)
{
  _tmpH = NULL;
  _invR = (gsl_matrix_complex** )malloc( (fftLen/2+1) * sizeof(gsl_matrix_complex*) );
  if( _invR == NULL ){
    fprintf(stderr,"LefkimmiatisPostFilter: gsl_matrix_complex_alloc failed\n");
    throw jallocation_error("LefkimmiatisPostFilter: gsl_matrix_complex_alloc failed\n");
  }
  for( unsigned fbinX=0;fbinX<=_fftLen/2;fbinX++){
    _invR[fbinX]  = NULL;
  }
}

LefkimmiatisPostFilter::~LefkimmiatisPostFilter()
{
  unsigned fftLen2 = _fftLen / 2;
  
  for( unsigned fbinX=0;fbinX<=fftLen2;fbinX++){
    if( NULL!=_invR[fbinX] )
      gsl_matrix_complex_free( _invR[fbinX] );
  }
  free(_invR);

  if( NULL != _tmpH ){
    gsl_vector_complex_free( _tmpH );
  }
}

void LefkimmiatisPostFilter::calcInverseNoiseSpatialSpectralMatrix()
{
  size_t nChan = _R[0]->size1;

  for(unsigned fbinX=0;fbinX<=_fftLen/2;fbinX++){ 
    if( NULL == _invR[fbinX] )
      _invR[fbinX] = gsl_matrix_complex_alloc( nChan, nChan );    
    // calculate the inverse matrix of the coherence matrix
    bool ret = pseudoinverse( _R[fbinX], _invR[fbinX], _minSV );
    if( false == ret )
      gsl_matrix_complex_set_identity( _invR[fbinX] );     
  }
  _isInvRcomputed = true;
}

gsl_complex LefkimmiatisPostFilter::calcLambda( unsigned fbinX )
{
  gsl_complex Lambda;
  const gsl_complex val1 = gsl_complex_rect( 1.0, 0.0 );
  const gsl_complex val0 = gsl_complex_rect( 0.0, 0.0 );
  gsl_vector_complex** arrayManifold = _bfWeightsPtr->arrayManifold();
      
  gsl_blas_zgemv( CblasConjTrans, val1, _invR[fbinX], arrayManifold[fbinX], val0, _tmpH ); // tmpH = invR^H * d
  gsl_blas_zdotc( _tmpH, arrayManifold[fbinX], &Lambda ); // Lambda = d^H * invR * d
  //norm = gsl_complex_mul_real( Lambda, nChan );
  
  return Lambda;
}

#ifdef  ORIGINAL_IAIN_PAPER   
double LefkimmiatisPostFilter::estimateAverageOfNoiseSignalPSD( unsigned fbinX, gsl_vector_complex* currCSDf )
{
  size_t nChan = _R[fbinX]->size1;
  double nu,de;
  double R_ij;
  double phi_ij; /* cross power specral density */
  double phi_ii, phi_jj; /* auto power specral density */
  double avg=0.0;

  if( TYPE_ZELINSKI1_REAL & _type ){
    for(size_t i=0;i<nChan-1;i++){
      phi_ii = GSL_REAL( gsl_vector_complex_get( currCSDf, i * nChan + i ) );
      for(size_t j=i+1;j<nChan;j++){
	phi_ij = GSL_REAL( gsl_vector_complex_get( currCSDf, i * nChan + j ) );
	phi_jj = GSL_REAL( gsl_vector_complex_get( currCSDf, j * nChan + j ) );
	R_ij   = GSL_REAL( gsl_matrix_complex_get( _R[fbinX], i, j ) );

	if( R_ij > _thresholdOfRij ){
	  R_ij = _thresholdOfRij;
	}
	else if( R_ij == 1 ){
	  R_ij = 0.99;
	}
	nu = 0.5 * ( phi_ii + phi_jj ) - phi_ij;
	de = 1 - R_ij;
	avg += (nu/de); // phi_ss 
      }
    }
  }
  else{
    for(size_t i=0;i<nChan-1;i++){
      phi_ii = gsl_complex_abs( gsl_vector_complex_get( currCSDf, i * nChan + i ) );
      for(size_t j=i+1;j<nChan;j++){
	phi_ij = gsl_complex_abs( gsl_vector_complex_get( currCSDf, i * nChan + j ) );
	phi_jj = gsl_complex_abs( gsl_vector_complex_get( currCSDf, j * nChan + j ) );
	R_ij   = gsl_complex_abs( gsl_matrix_complex_get( _R[fbinX], i, j ) );

	if( R_ij > _thresholdOfRij ){
	  R_ij = _thresholdOfRij;
	}
	else if( R_ij == 1 ){
	  R_ij = 0.99;
	}
	//fprintf(stderr,"%d : %d %d :%e - 0.5 * %e ( %e + %e )\n ",fbinX,i,j,phi_ij,R_ij,phi_ii,phi_jj);
	nu = 0.5 * ( phi_ii + phi_jj ) - phi_ij;
	de = 1 - R_ij;
	avg += (nu/de); // phi_ss 
      }
    }
  }

  return ( 2.0*avg/(nChan*(nChan-1)) );
}
#else
double LefkimmiatisPostFilter::estimateAverageOfNoiseSignalPSD( unsigned fbinX, gsl_vector_complex* currCSDf )
{
  size_t nChan = _R[fbinX]->size1;
  gsl_complex nu, de, val, sum;
  gsl_complex R_ij;
  gsl_complex phi_ij; /* cross power specral density */
  gsl_complex phi_ii, phi_jj; /* auto power specral density */
  double avg=0.0;

  GSL_SET_COMPLEX( &sum, 0, 0 );
  for(size_t i=0;i<nChan-1;i++){
    phi_ii = gsl_vector_complex_get( currCSDf, i * nChan + i );
    for(size_t j=i+1;j<nChan;j++){
      phi_ij = gsl_vector_complex_get( currCSDf, i * nChan + j );
      phi_jj = gsl_vector_complex_get( currCSDf, j * nChan + j );
      R_ij   = gsl_matrix_complex_get( _R[fbinX], i, j );
      
      if( GSL_REAL( R_ij ) > _thresholdOfRij ){
	GSL_SET_COMPLEX( &R_ij, _thresholdOfRij, 0 );
      }
      else if( GSL_REAL( R_ij ) == 1 ){
	GSL_SET_COMPLEX( &R_ij, 0.99, 0 );
      }
      val = gsl_complex_mul_real( gsl_complex_add( phi_ii, phi_jj ), 0.5 );
      nu  = gsl_complex_sub( val, phi_ij ); //nu = phi_ij - 0.5 * R_ij * ( phi_ii + phi_jj );
      de  = gsl_complex_add_real( gsl_complex_negative( R_ij ), 1.0 ); // de = 1 - R_ij;
      sum = gsl_complex_add( sum, gsl_complex_div( nu, de ) ); // phi_ss 
    }
  }

  if( TYPE_ZELINSKI1_REAL & _type ){
    avg = GSL_REAL(sum);
  }
  else{
    avg = gsl_complex_abs(sum);
  }
  
  return ( 2.0*avg/(nChan*(nChan-1)) );
}
#endif

void LefkimmiatisPostFilter::PostFiltering()
{
  unsigned nChan = _snapShotArray->nChan();

  if( NULL == _R[0] ){
    fprintf(stderr,"LefkimmiatisPostFilter: construct/set a noise coherence matrix\n");
    throw j_error("LefkimmiatisPostFilter:  construct/set a noise coherence matrix\n");
  }
  if( NULL == _tmpH )
    _tmpH = gsl_vector_complex_alloc( nChan );
  if( false == _isInvRcomputed )
    calcInverseNoiseSpatialSpectralMatrix();
  
  unsigned fftLen2 = _fftLen / 2;
  gsl_vector_complex** arrayManifold = _bfWeightsPtr->arrayManifold();
  gsl_vector_complex* wp             = _bfWeightsPtr->wp1(); /* post-filter's weights */
  gsl_vector_complex** prevCSDs      = _bfWeightsPtr->CSDs();/* auto and cross spectral densities */
  double alpha; /* forgetting factor */

  if( NULL == _timeAlignedSignalf )
    _timeAlignedSignalf = gsl_vector_complex_calloc( nChan );

  if( _frameX > 0 )
    alpha =  _alpha;
  else
    alpha = 0.0;

  if(  _bfWeightsPtr->isHalfBandShift()==false ){
    for(unsigned fbinX=0;fbinX<=fftLen2;fbinX++){
      double weight;
      double phi_ss, phi_vv, phi_nn;
      gsl_vector_complex* prevCSDf = prevCSDs[fbinX];
      const gsl_vector_complex *snapShot = _snapShotArray->getSnapShot(fbinX);

      TimeAlignment( arrayManifold[fbinX], snapShot, nChan, _timeAlignedSignalf ); // beamforming
      calculateSpectralDensities_f( _timeAlignedSignalf, prevCSDf, alpha );
      phi_ss = estimateAverageOfCleanSignalPSD( fbinX, prevCSDf );
      phi_vv = estimateAverageOfNoiseSignalPSD( fbinX, prevCSDf );

      if( fbinX < _fbinX1 ){
	weight = phi_ss / ( phi_ss + phi_vv );
      }
      else{
	if( TYPE_ZELINSKI1_REAL & _type ){
	  phi_nn = phi_vv / GSL_REAL( calcLambda( fbinX ) );
	}
	else
	  phi_nn = phi_vv / gsl_complex_abs( calcLambda( fbinX ) );
	weight = phi_ss / ( phi_ss + phi_nn );
      }

      if( weight >  1.0 ) weight =  1.0;
      if( weight < SPECTRAL_FLOOR ) weight = SPECTRAL_FLOOR;
      gsl_vector_complex_set( wp, fbinX, gsl_complex_polar( weight, 0 ) );
      if( fbinX > 0 && fbinX < fftLen2 )
	gsl_vector_complex_set( wp, _fftLen - fbinX, gsl_complex_polar( weight, 0 ) );
      
      if( _frameX >= _minFrames ){
	//fprintf(stderr, "%0.4e \n", weight );
	
	gsl_complex outf = gsl_complex_mul_real( gsl_vector_complex_get( _vector, fbinX ), weight );
	gsl_vector_complex_set(_vector, fbinX, outf );
	if( fbinX > 0 && fbinX < fftLen2 )// substitute a conjugate component
	  gsl_vector_complex_set( _vector, _fftLen - fbinX, gsl_complex_conjugate( outf ) );
      }
    }
  }
  else{
    fprintf(stderr,"LefkimmiatisPostFilter: The opition, halfBandShift=True, is not yet implemented\n");
    throw j_error( "LefkimmiatisPostFilter: The opition, halfBandShift=True, is not yet implemented\n");
  }
}

const gsl_vector_complex* LefkimmiatisPostFilter::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  const gsl_vector_complex* output;

  if( frameX >= 0 )
    output = _samp->next(frameX);
  else{
    if( _frameX == FrameResetX )
      output = _samp->next(0);
    else
      output = _samp->next(_frameX+1);
  }

  if( true ==  _hasBeamformerPtr ){
    _snapShotArray = _beamformerPtr->getSnapShotArray();
    _bfWeightsPtr = _beamformerPtr->getBeamformerWeightObject(0);
  }
  if( NULL == _bfWeightsPtr ){
    fprintf(stderr,"set beamformer's weights \n");
    throw  j_error("set beamformer's weights \n");
  }
  
  for (unsigned fbinX = 0; fbinX <= _fftLen/2; fbinX++)
    gsl_vector_complex_set(_vector, fbinX, gsl_vector_complex_get( output, fbinX ) );

  PostFiltering();

  _increment();
  return _vector;
}

void LefkimmiatisPostFilter::reset()
{
  //fprintf(stderr,"LefkimmiatisPostFilter::reset() 1\n");
  _samp->reset();
  VectorComplexFeatureStream::reset();
  _endOfSamples = false;
}

// ----- definition for class `highPassFilter' -----
//

highPassFilter::highPassFilter( VectorComplexFeatureStreamPtr &output, float cutOffFreq, int sampleRate, const String& nm):
  VectorComplexFeatureStream(output->size(), nm),
  _src(output),
  _cutOffFreq(cutOffFreq)
{
  unsigned fftLen = output->size();
  _cutOfffreqBin = (unsigned) ( fftLen * cutOffFreq / (float)sampleRate );
  //fprintf(stderr,"Cut 0 - %d\n",_cutOfffreqBin);
}

const gsl_vector_complex* highPassFilter::next(int frameX)
{
  if (frameX == _frameX) return _vector;
  
  const gsl_vector_complex* srcVec = _src->next();

  for(unsigned fbinX=1;fbinX<_cutOfffreqBin;fbinX++){// now brainlessly cut...
    gsl_vector_complex_set( _vector, fbinX, gsl_complex_rect( 0, 0 ));
  }

  unsigned fftLen  = _src->size();
  unsigned fftLen2 = fftLen/2;
  for(unsigned fbinX=_cutOfffreqBin;fbinX<=fftLen2;fbinX++){
    gsl_complex val = gsl_vector_complex_get( srcVec, fbinX );
    if( fbinX < fftLen2 ){
      gsl_vector_complex_set(_vector, fbinX, val);
      gsl_vector_complex_set(_vector, fftLen - fbinX, gsl_complex_conjugate(val) );
    }
    else
      gsl_vector_complex_set(_vector, fftLen2, val);    
  }

  _increment();
  return _vector;
}

void highPassFilter::reset()
{
  _src->reset();  VectorComplexFeatureStream::reset();
}
