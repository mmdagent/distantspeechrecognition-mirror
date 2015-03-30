#include "beamformer/modalBeamformer.h"
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_trig.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_legendre.h>
#include <gsl/gsl_blas.h>

void calcTimeDelaysOfSphericalArray( double theta, double phi, unsigned chanN, double rad, gsl_vector *theta_s, gsl_vector *phi_s,
				     gsl_vector* delays )
{
  double dist, theta_sn, phi_sn;

  for(unsigned chanX=0;chanX<chanN;chanX++){
    theta_sn = gsl_vector_get( theta_s, chanX );
    phi_sn   = gsl_vector_get( phi_s,   chanX );
    
    dist = rad * ( sin(theta_sn) * sin(theta) * cos(phi_sn-phi) + cos(theta_sn) * cos(theta) );
    gsl_vector_set( delays, chanX, - dist / SSPEED );
  }
}

void normalizeWeights( gsl_vector_complex *weights, float wgain )
{  
  double nrm = wgain / gsl_blas_dznrm2( weights );// / gsl_blas_dznrm2() returns the Euclidean norm
  
  for(unsigned i=0;i<weights->size;i++)
    gsl_vector_complex_set( weights, i, gsl_complex_mul_real( gsl_vector_complex_get( weights, i ), nrm ) );
}

/**
	@brief compute the mode amplitude.
	@param int order[in]
	@param double ka[in] k : wavenumber, a : radius of the rigid sphere
	@param double kr[in] k : wavenumber, r : distance of the observation point from the origin
 */
gsl_complex modeAmplitude( int order, double ka )
{
  if( ka == 0 )
    return gsl_complex_rect( 1, 0 );
  
  gsl_complex bn;
  switch (order){
  case 0:
    {
      double ka2 = ka  * ka;
      double      j0 = gsl_sf_sinc( ka/M_PI );
      double      y0 = - cos(ka)/ka;
      gsl_complex h0 = gsl_complex_rect( j0, y0 );
      double      val1 = cos(ka)/ka - sin(ka)/ka2;
      gsl_complex eika = gsl_complex_polar(1,ka);
      gsl_complex val2 = gsl_complex_div_real( gsl_complex_mul( gsl_complex_rect(ka,1), eika ), ka2 );
      gsl_complex grad = gsl_complex_div( gsl_complex_rect( val1, 0 ), val2 );
      bn = gsl_complex_sub( gsl_complex_rect(j0, 0), gsl_complex_mul( grad, h0 ) );
    }
    break;
  case 1:
    {
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka;
      double j1 =  ( sin(ka)/ka2 ) - ( cos(ka)/ka ) ;
      double y1 = -( cos(ka)/ka2 ) - ( sin(ka)/ka ) ;
      gsl_complex h1 = gsl_complex_rect( j1, y1 );
      double val1 = ( - 0.5/ka ) * ( -cos(ka)/ka + sin(ka)/ka2 ) + 0.5 * ( 3*cos(ka)/ka2 + sin(ka)/ka - (3-ka2)*sin(ka)/ka3 );
      double j0 = gsl_sf_sinc( ka/M_PI );
      double y0 = -cos(ka)/ka;
      gsl_complex h0 = gsl_complex_rect( j0, y0 );
      double j2 =  ( 3/ka3 - 1/ka ) * sin(ka) - ( 3/ka2 ) * cos(ka);
      double y2 = -( 3/ka3 - 1/ka ) * cos(ka) - ( 3/ka2 ) * sin(ka);
      gsl_complex h2 = gsl_complex_rect( j2, y2 );
      gsl_complex hdiff = gsl_complex_sub( h0, h2 );
      gsl_complex val2 = gsl_complex_div_real( gsl_complex_sub( hdiff, gsl_complex_div_real( h1, ka) ), 2 );
      gsl_complex grad = gsl_complex_div( gsl_complex_rect( val1, 0 ), val2 );

      bn = gsl_complex_sub( gsl_complex_rect(j1, 0), gsl_complex_mul( grad, h1 ) );
    }
    break;
  case 2:
    {
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka; 
      double ka4 = ka3 * ka;
      double j2 =  ( 3/ka3 - 1/ka )*sin(ka) - ( 3*cos(ka)/ka2 );
      double y2 = -( 3/ka3 - 1/ka )*cos(ka) - ( 3*sin(ka)/ka2 );
      gsl_complex h2 = gsl_complex_rect( j2, y2 );
      double val1 = 0.5 * ( -cos(ka)/ka + sin(ka)/ka2 + (18-ka2)*cos(ka)/ka3 + (-18+7*ka2)*sin(ka)/ka4 );
      double j1 =  ( sin(ka)/ka2 ) - ( cos(ka)/ka ) ;
      double y1 = -( cos(ka)/ka2 ) - ( sin(ka)/ka ) ;
      gsl_complex h1 = gsl_complex_rect( j1, y1 );
      double j3 = ( -15+ka2 )*cos(ka)/ka3 - ( -15+6*ka2 )*sin(ka)/ka4;
      double y3 = ( -15+ka2 )*sin(ka)/ka3 + ( -15+6*ka2 )*cos(ka)/ka4;
      gsl_complex h3 = gsl_complex_rect( j3, y3 );
      gsl_complex hdiff = gsl_complex_sub( h1, h3 );
      gsl_complex val2 = gsl_complex_div_real( gsl_complex_sub( hdiff, gsl_complex_div_real( h2, ka) ), 2 );
      gsl_complex grad = gsl_complex_div( gsl_complex_rect( val1, 0 ), val2 );
      
      bn = gsl_complex_sub( gsl_complex_rect(j2, 0), gsl_complex_mul( grad, h2 ) );
    }
    break;
  case 3:
    {
      double ka2 = ka  * ka;
      double ka3 = ka2 * ka; 
      double ka4 = ka2 * ka2;
      double ka5 = ka4 * ka;
      double j3 = ( -15+ka2 )*cos(ka)/ka3 - ( -15+6*ka2 )*sin(ka)/ka4;
      double y3 = ( -15+ka2 )*sin(ka)/ka3 + ( -15+6*ka2 )*cos(ka)/ka4;
      gsl_complex h3 = gsl_complex_rect( j3, y3 );
      double val1 = 0.5 * ( -3*cos(ka)/ka2 + (3-ka2)*sin(ka)/ka3 + (120-11*ka2)*cos(ka)/ka4 + (-120+51*ka2-ka4)*sin(ka)/ka5 );
      double j2 =  ( 3/ka3 - 1/ka )*sin(ka) - ( 3*cos(ka)/ka2 );
      double y2 = -( 3/ka3 - 1/ka )*cos(ka) - ( 3*sin(ka)/ka2 );
      gsl_complex h2 = gsl_complex_rect( j2, y2 );
      double j4 = ( -105+10*ka2 )*cos(ka)/ka4 + (105-45*ka2+ka4)*sin(ka)/ka5 ;
      double y4 = ( -105+10*ka2 )*sin(ka)/ka4 - (105-45*ka2+ka4)*cos(ka)/ka5;
      gsl_complex h4 = gsl_complex_rect( j4, y4 );
      gsl_complex hdiff = gsl_complex_sub( h2, h4 );
      gsl_complex val2 = gsl_complex_div_real( gsl_complex_sub( hdiff, gsl_complex_div_real( h3, ka) ), 2 );
      gsl_complex grad = gsl_complex_div( gsl_complex_rect( val1, 0 ), val2 );
      
      bn = gsl_complex_sub( gsl_complex_rect(j3, 0), gsl_complex_mul( grad, h3 ) );
    }
    break;
  default:
    {
      int status;
      gsl_sf_result jn, jn_p, jn_n;
      gsl_sf_result yn, yn_p, yn_n;
      gsl_complex   hn, hn_p, hn_n;
      double djn, dyn;
      gsl_complex dhn;
      gsl_complex val, grad;
      gsl_complex hdiff;

      status = gsl_sf_bessel_jl_e( order, ka, &jn);// the (regular) spherical Bessel function of the first kind
      status = gsl_sf_bessel_yl_e( order, ka, &yn);// the (irregular) spherical Bessel function of the second kind
      hn = gsl_complex_rect(jn.val, yn.val); // Spherical Hankel function of the first kind
      
      status = gsl_sf_bessel_jl_e( order-1, ka, &jn_p );
      status = gsl_sf_bessel_jl_e( order+1, ka, &jn_n );
      djn = ( jn_p.val - jn.val / ka - jn_n.val ) / 2;

      status = gsl_sf_bessel_yl_e( order-1, ka, &yn_p );
      status = gsl_sf_bessel_yl_e( order+1, ka, &yn_n );
      dyn = ( yn_p.val - yn.val / ka - yn_n.val ) / 2;

      hn_p = gsl_complex_rect( jn_p.val, yn_p.val );
      hn_n = gsl_complex_rect( jn_n.val, yn_n.val );
      
      hdiff = gsl_complex_sub( hn_p, hn_n );
      dhn = gsl_complex_div_real( gsl_complex_sub( hdiff, gsl_complex_div_real( hn, ka) ), 2 );
      
      //printf ("status  = %s\n", gsl_strerror(status));
      //printf ("J0(5.0) = %.18f +/- % .18f\n", result.val, result.err);  
      
      grad = gsl_complex_div( gsl_complex_rect( djn, 0 ), dhn );
      bn   = gsl_complex_add_real( gsl_complex_negative( gsl_complex_mul( grad, hn ) ), jn.val );
    }
    break;
  }

  return bn;
}

/**
   @brief compute the spherical harmonics transformation of the input shapshot
   @param int maxOrder[in] 
   @param const gsl_vector_complex *XVec[in] the input snapshot vector
   @param gsl_vector_complex **sh_s[in] the orthogonal bases (spherical harmonics) sh_s[sensors index][basis index]
   @param gsl_vector_complex *F[out] the spherical transformation coefficinets will be stored.
 */
static void sphericalHarmonicsTransformation( int maxOrder, const gsl_vector_complex *XVec, gsl_vector_complex **sh_s,
					      gsl_vector_complex *F )
{
  gsl_complex Fmn;

  for(int n=0,idx=0;n<maxOrder;n++){/* order */
    for(int m=-n;m<=n;m++){/* degree */
      gsl_blas_zdotu( XVec, sh_s[idx], &Fmn );
      gsl_vector_complex_set( F, idx, Fmn );
      idx++;
    }
  }

  return;
}

/**
   @brief calculate the spherical harmonic for a steering vector
   @param int degree[in]
   @param int order[in]
   @param double theta [in] the look direction
   @param double phi[in] the look direction
   @return the spherical harmonic
 */
static gsl_complex sphericalHarmonic( int degree, int order, double theta, double phi )
{
  int status;
  gsl_sf_result sphPnm;
  gsl_complex Ymn;
  
  if( order < degree || order < -degree ){// 
    fprintf( stderr, "The oder must be less than the degree but %d > %dn", order, degree );
  }
  
  //fprintf(stderr,"%d %d %e ", order, degree, cos(theta)); 
  if( degree >= 0 ){
    // \sqrt{(2l+1)/(4\pi)} \sqrt{(l-m)!/(l+m)!} P_l^m(x), and derivatives, m >= 0, l >= m, |x| <= 1
    status = gsl_sf_legendre_sphPlm_e( order /* =l */, degree /* =m */,cos(theta), &sphPnm);
  }
  else{
    status = gsl_sf_legendre_sphPlm_e( order /* =l */, -  degree /* =m */,cos(theta), &sphPnm);
    if( ( (-degree) % 2 ) != 0 )
      sphPnm.val = - sphPnm.val;
  }
  
  Ymn = gsl_complex_mul_real( gsl_complex_polar( 1.0, degree*phi ), sphPnm.val );
  //fprintf(stderr,"%e %e \n", sphPnm.val, sphPnm.err); 

  return Ymn;
}

static void calcDCWeights( unsigned maxOrder, gsl_vector_complex *weights )
{
  for(int n=0,idx=0;n<maxOrder;n++){
     for( int m=-n;m<=n;m++){/* degree */
       if( n == 0 ){
	 gsl_vector_complex_set( weights, idx, gsl_complex_rect(1,0) );
       }
       else{
	 gsl_vector_complex_set( weights, idx, gsl_complex_rect(0,0) );
       }
       idx++;
     }
  }
}

// ----- definition for class `EigenBeamformer' -----
// 
EigenBeamformer::EigenBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, const String& nm )
  :  SubbandDS( fftLen, halfBandShift, nm ),
     _sampleRate(sampleRate),
     _NC(NC),
     _maxOrder(maxOrder),
     _areWeightsNormalized(normalizeWeight),
     _modeAmplitudes(NULL),
     _F(NULL),
     _sh_s(NULL),
     _theta(0.0), 
     _phi(0.0),
     _sphericalTransformSnapShotArray(NULL),
     _a(0.0),
     _theta_s(NULL),
     _phi_s(NULL),
     _beamPattern(NULL),
     _WNG(NULL),
     _wgain(1.0),
     _sigma2(0.0)
{
  _bfWeightV.resize(1); // the steering vector
  _bfWeightV[0] = NULL;

  _dim = _maxOrder * _maxOrder;
  //_dim = 0;
  //for(int n=0;n<_maxOrder;n++)
  //for( int m=-n;m<=n;m++)
  //_dim++;
}
  
EigenBeamformer::~EigenBeamformer()
{
  if( NULL != _modeAmplitudes )
    gsl_matrix_complex_free( _modeAmplitudes );

  if( NULL != _F ){
    gsl_vector_complex_free( _F );
    _F = NULL;
  }

  if( NULL != _sh_s ){
    for(unsigned dimX=0;dimX<_dim;dimX++)
      gsl_vector_complex_free( _sh_s[dimX] );
    free(_sh_s);
    _sh_s = NULL;
  }

  if( NULL != _theta_s )
    gsl_vector_free( _theta_s );
  if( NULL != _phi_s )
    gsl_vector_free( _phi_s );
  if( NULL != _beamPattern )
    gsl_matrix_free( _beamPattern );
  if( NULL != _WNG )
    gsl_vector_free(_WNG );
}

/**
   @brief compute the output of the eigenbeamformer
   @param unsigned fbinX[in]
   @param gsl_matrix_complex *bMat[in] mode amplitudes 
   @param double theta[in] the steering direction
   @param double phi[in] the steering direction
   @param gsl_vector_complex *weights[out]
*/
void EigenBeamformer::_calcWeights( unsigned fbinX, gsl_vector_complex *weights )
{
  unsigned norm = _dim * (unsigned)_phi_s->size;

  for(int n=0,idx=0;n<_maxOrder;n++){/* order */
    gsl_complex bn = gsl_matrix_complex_get( _modeAmplitudes, fbinX, n ); //bn = modeAmplitude( order, ka );
    double      bn2 = gsl_complex_abs2( bn ) + _sigma2;
    double      de  = norm * bn2;
    gsl_complex in;
    gsl_complex inbn;

    if( 0 == ( n % 4 ) ){
      in = gsl_complex_rect(1,0);
    }
    else if( 1 == ( n % 4 ) ){
      in = gsl_complex_rect(0,1);
    }
    else if( 2 == ( n % 4 ) ){
      in = gsl_complex_rect(-1,0);
    }
    else{
      in = gsl_complex_rect(0,-1);
    }

    inbn = gsl_complex_mul( in, bn );
    for( int m=-n;m<=n;m++){/* degree */
      gsl_complex weight;
      gsl_complex YmnA = gsl_complex_conjugate(sphericalHarmonic( m, n, _theta, _phi ));

      //weight = gsl_complex_div( sphericalHarmonic( m, n, _theta, _phi ), gsl_complex_mul_real( gsl_complex_mul( in, bn ), 4 * M_PI ) ); // follow Rafaely's paper
      //gsl_vector_complex_set( weights, idx, gsl_complex_conjugate(weight) ); 
      weight = gsl_complex_div_real( gsl_complex_mul( gsl_complex_mul_real( YmnA, 4 * M_PI ), inbn ), de ); // HMDI beamfomrer; see S Yan's paper
      gsl_vector_complex_set( weights, idx, weight ); 
      idx++;
    }
  }
  
  if( true==_areWeightsNormalized )
    normalizeWeights( weights, _wgain );

  return;
}

const gsl_vector_complex* EigenBeamformer::next( int frameX )
{
  //fprintf(stderr, "EigenBeamformer::next" );

  if (frameX == _frameX) return _vector;

  unsigned chanX = 0;
  const gsl_vector_complex* XVec;   /* snapshot at each frequency */
  const gsl_vector_complex* weights; /* weights of the combining-unit */
  gsl_complex val;

  this->_allocImage();
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frameX);
    if( true==(*itr)->isEnd() ) _endOfSamples = true;
    _snapShotArray->newSample( samp, chanX);  chanX++;
  }
  _snapShotArray->update();
  
  if( _halfBandShift == false ){
    // calculate a direct component.
    XVec    = _snapShotArray->getSnapShot(0);
    sphericalHarmonicsTransformation( _maxOrder, XVec, _sh_s, _F );
    _sphericalTransformSnapShotArray->newSnapShot( (const gsl_vector_complex*)_F, 0 );
    weights = _bfWeightV[0]->wq_f(0);
    gsl_blas_zdotc( weights, _F, &val ); //gsl_blas_zdotc( weights, _F, &val );
    gsl_vector_complex_set(_vector, 0, val);
    //gsl_vector_complex_set( _vector, 0, gsl_vector_complex_get( XVec, XVec->size/2) );

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX <= _fftLen2; fbinX++) {
      XVec  = _snapShotArray->getSnapShot(fbinX);
      sphericalHarmonicsTransformation( _maxOrder, XVec, _sh_s, _F );
      _sphericalTransformSnapShotArray->newSnapShot( (const gsl_vector_complex*)_F, fbinX );
      weights =  _bfWeightV[0]->wq_f(fbinX);
      gsl_blas_zdotc( weights, _F, &val ); // gsl_blas_zdotc( weights, _F, &val ); x^H y
      
      if( fbinX < _fftLen2 ){
	gsl_vector_complex_set(_vector, fbinX, val);
	gsl_vector_complex_set(_vector, _fftLen - fbinX, gsl_complex_conjugate(val) );
      }
      else
	gsl_vector_complex_set(_vector, _fftLen2, val);
    }
  }
  else{
    fprintf(stderr,"_halfBandShift == true is not implemented yet\n");
    throw j_error("_halfBandShift == true is not implemented yet\n");
  }

  _increment();
  return _vector;
}
 
void EigenBeamformer::reset()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();

  if ( NULL != _snapShotArray )
    _snapShotArray->zero();

  if( NULL != _sphericalTransformSnapShotArray )
    _sphericalTransformSnapShotArray->zero();

  VectorComplexFeatureStream::reset();
  _endOfSamples = false;
}

/**
   @brief set the geometry of the EigenMikeR
*/
void EigenBeamformer::setEigenMikeGeometry( )
{ 
   gsl_vector *theta_s = gsl_vector_alloc( 32 );
   gsl_vector *phi_s   = gsl_vector_alloc( 32 );

   gsl_vector_set( theta_s, 0, 69 * M_PI / 180 );
   gsl_vector_set( phi_s,   0, 0.0 );

   gsl_vector_set( theta_s, 1, 90 * M_PI / 180 );
   gsl_vector_set( phi_s,   1, 32 * M_PI / 180 );

   gsl_vector_set( theta_s, 2, 111 * M_PI / 180 );
   gsl_vector_set( phi_s,   2, 0 );

   gsl_vector_set( theta_s, 3, 90 * M_PI / 180 );
   gsl_vector_set( phi_s,   3, 328 * M_PI / 180 );

   gsl_vector_set( theta_s, 4, 32 * M_PI / 180 );
   gsl_vector_set( phi_s,   4, 0);

   gsl_vector_set( theta_s, 5, 55 * M_PI / 180 );
   gsl_vector_set( phi_s,   5, 45 * M_PI / 180 );

   gsl_vector_set( theta_s, 6, 90 * M_PI / 180 );
   gsl_vector_set( phi_s,   6, 69 * M_PI / 180 );

   gsl_vector_set( theta_s, 7, 125 * M_PI / 180);
   gsl_vector_set( phi_s,   7, 45  * M_PI / 180);

   gsl_vector_set( theta_s, 8, 148 * M_PI / 180);
   gsl_vector_set( phi_s,   8, 0 );

   gsl_vector_set( theta_s, 9, 125 * M_PI / 180 );
   gsl_vector_set( phi_s,   9, 315 * M_PI / 180 );

   gsl_vector_set( theta_s, 10, 90 * M_PI / 180 );
   gsl_vector_set( phi_s,   10, 291 * M_PI / 180 );

   gsl_vector_set( theta_s, 11, 55 * M_PI / 180 );
   gsl_vector_set( phi_s,   11, 315 * M_PI / 180 );

   gsl_vector_set( theta_s, 12, 21 * M_PI / 180 );
   gsl_vector_set( phi_s,   12, 91 * M_PI / 180 );

   gsl_vector_set( theta_s, 13, 58 * M_PI / 180 );
   gsl_vector_set( phi_s,   13, 90 * M_PI / 180 );

   gsl_vector_set( theta_s, 14, 121 * M_PI / 180 );
   gsl_vector_set( phi_s,   14, 90 * M_PI / 180 );

   gsl_vector_set( theta_s, 15, 159 * M_PI / 180 );
   gsl_vector_set( phi_s,   15, 89 * M_PI / 180 );

   gsl_vector_set( theta_s, 16, 69 * M_PI / 180 );
   gsl_vector_set( phi_s,   16, 180 * M_PI / 180 );

   gsl_vector_set( theta_s, 17, 90 * M_PI / 180 );
   gsl_vector_set( phi_s,   17, 212 * M_PI / 180 );

   gsl_vector_set( theta_s, 18, 111 * M_PI / 180 );
   gsl_vector_set( phi_s,   18, 180 * M_PI / 180 );

   gsl_vector_set( theta_s, 19, 90 * M_PI / 180 );
   gsl_vector_set( phi_s,   19, 148 * M_PI / 180 );

   gsl_vector_set( theta_s, 20, 32 * M_PI / 180 );
   gsl_vector_set( phi_s,   20, 180 * M_PI / 180 );

   gsl_vector_set( theta_s, 21, 55 * M_PI / 180 );
   gsl_vector_set( phi_s,   21, 225 * M_PI / 180 );

   gsl_vector_set( theta_s, 22, 90 * M_PI / 180 );
   gsl_vector_set( phi_s,   22, 249 * M_PI / 180 );

   gsl_vector_set( theta_s, 23, 125 * M_PI / 180 );
   gsl_vector_set( phi_s,   23, 225 * M_PI / 180 );
   
   gsl_vector_set( theta_s, 24, 148 * M_PI / 180 );
   gsl_vector_set( phi_s,   24, 180 * M_PI / 180 );

   gsl_vector_set( theta_s, 25, 125 * M_PI / 180 );
   gsl_vector_set( phi_s,   25, 135 * M_PI / 180 );

   gsl_vector_set( theta_s, 26, 90 * M_PI / 180 );
   gsl_vector_set( phi_s,   26, 111 * M_PI / 180 );

   gsl_vector_set( theta_s, 27, 55 * M_PI / 180 );
   gsl_vector_set( phi_s,   27, 135 * M_PI / 180 );

   gsl_vector_set( theta_s, 28, 21 * M_PI / 180 );
   gsl_vector_set( phi_s,   28, 269 * M_PI / 180 );

   gsl_vector_set( theta_s, 29, 58 * M_PI / 180 );
   gsl_vector_set( phi_s,   29, 270 * M_PI / 180 );

   gsl_vector_set( theta_s, 30, 122 * M_PI / 180 );
   gsl_vector_set( phi_s,   30, 270 * M_PI / 180 );

   gsl_vector_set( theta_s, 31, 159 * M_PI / 180 );
   gsl_vector_set( phi_s,   31, 271 * M_PI / 180 );

   for ( unsigned i = 0; i < 32; i++ ) {
     fprintf(stderr, "%d : %e %e\n", i, gsl_vector_get( theta_s, i ),
	     gsl_vector_get( phi_s, i ) );
     fflush(stderr);
   }

   setArrayGeometry( 42, theta_s, phi_s );
   
   gsl_vector_free( theta_s );
   gsl_vector_free( phi_s );
}

void EigenBeamformer::setArrayGeometry( double a, gsl_vector *theta_s, gsl_vector *phi_s )
{ 
  if ( theta_s->size != phi_s->size ) {
    fprintf( stderr,
	     "The numbers of the sensor positions have to be the same\n" );
    return;
  }

  _a = a; // radius 

  if ( NULL != _theta_s )
    gsl_vector_free( _theta_s );
  _theta_s = gsl_vector_alloc( theta_s->size );

  if ( NULL != _phi_s )
    gsl_vector_free( _phi_s );
  _phi_s = gsl_vector_alloc( phi_s->size );

  for ( unsigned i = 0; i < theta_s->size; i++) {
    gsl_vector_set( _theta_s, i, gsl_vector_get( theta_s, i ) );
    gsl_vector_set( _phi_s,   i, gsl_vector_get( phi_s,   i ) );
  }

  if ( false == _calcSphericalHarmonicsAtEachPosition( theta_s, phi_s ) ) {
    fprintf(stderr,"_calcSphericalHarmonicsAtEachPosition() failed\n");
  }
}

bool EigenBeamformer::_calcSphericalHarmonicsAtEachPosition( gsl_vector *theta_s, gsl_vector *phi_s )
{
  int nChan = (int)theta_s->size;

  if( NULL != _F )
    gsl_vector_complex_free( _F );
  _F = gsl_vector_complex_alloc( _dim );

  if( NULL != _sh_s )
    free(_sh_s);
  _sh_s = (gsl_vector_complex **)malloc(_dim*sizeof(gsl_vector_complex *));
  if( NULL == _sh_s ){
    fprintf(stderr,"_calcSphericalHarmonicsAtEachPosition : cannot allocate memory\n");
    return false;
  }
  for(int dimX=0;dimX<_dim;dimX++)
    _sh_s[dimX] = gsl_vector_complex_alloc( nChan );

  // compute spherical harmonics for each sensor
  for(int n=0,idx=0;n<_maxOrder;n++){/* order */
    for(int m=-n;m<=n;m++){/* degree */
      for(int chanX=0;chanX<nChan;chanX++){
	gsl_complex Ymn_s = sphericalHarmonic( m, n, 
					       gsl_vector_get( theta_s, chanX ), 
					       gsl_vector_get( phi_s, chanX ) );
	//gsl_vector_complex_set( _sh_s[chanX], idx, gsl_complex_div_real( Ymn_s, 2 * sqrt( M_PI ) ) );// based on Meyer and Elko's descripitons. Do not gsl_complex_conjugate
	gsl_vector_complex_set( _sh_s[idx], chanX, gsl_complex_conjugate( Ymn_s ) );// based on Rafaely's definition
      }
      idx++;
    }
  }

  return true;
}

/**
   @brief set the look direction for the steering vector.
   @param double theta[in] 
   @param double phi[in]
   @param isGSC[in]
 */
void EigenBeamformer::setLookDirection( double theta, double phi )
{
  //fprintf(stderr, "EigenBeamformer::setLookDirection\n" );
  fflush(stderr);

  if( theta < 0 || theta >  M_PI ){
    fprintf(stderr,"ERROR: Out of range of theta\n");
  }

  _theta = theta;
  _phi   = phi;

  if( NULL == _modeAmplitudes )
    _calcModeAmplitudes();

  if( NULL == _bfWeightV[0] )
    _allocSteeringUnit(1);
  
  _calcSteeringUnit( 0, false /* isGSC */  );
}

const gsl_matrix_complex *EigenBeamformer::getModeAmplitudes()
{
  if( _a == 0.0 ){
    fprintf(stderr,"set the radius of the rigid sphere\n");
    return NULL;
  }

  if( NULL == _modeAmplitudes )
    if( false == _calcModeAmplitudes() ){
      fprintf(stderr,"Did you set the multi-channel data?\n");
      return NULL;
    }

  return (const gsl_matrix_complex *)_modeAmplitudes;
}

gsl_vector *EigenBeamformer::getArrayGeometry( int type )
{
  if( type == 0 ){
    return _theta_s;
  }
  return _phi_s;
}

void EigenBeamformer::_allocImage( bool flag )
{
  if( _a == 0.0 ){
    fprintf(stderr,"set the radius of the rigid sphere\n");
    throw j_error("set the radius of the rigid sphere\n");
  }

  if( NULL == _snapShotArray )
    _snapShotArray = new SnapShotArray( _fftLen, _channelList.size() );  

  if( NULL == _sphericalTransformSnapShotArray ){
    _sphericalTransformSnapShotArray = new SnapShotArray( _fftLen, _dim );
    _sphericalTransformSnapShotArray->zero();
  }

  if( NULL == _modeAmplitudes )
    _calcModeAmplitudes();

  if( NULL == _bfWeightV[0] && flag ){
    _allocSteeringUnit(1);
    _calcSteeringUnit(0);
  }
}

bool EigenBeamformer::_calcModeAmplitudes()
{  
  _modeAmplitudes = gsl_matrix_complex_alloc( _fftLen2+1, _maxOrder );
  for (unsigned fbinX = 0; fbinX <= _fftLen2; fbinX++) {
    double ka = 2.0 * M_PI * fbinX * _a * _sampleRate / ( _fftLen * SSPEED );
    
    for(int n=0;n<_maxOrder;n++){/* order */
      gsl_matrix_complex_set( _modeAmplitudes, fbinX, n, modeAmplitude( n, ka ) );
    }
  }

  return true;
}

bool EigenBeamformer::_allocSteeringUnit( int unitN )
{
  for(unsigned unitX=0;unitX<_bfWeightV.size();unitX++){
    if( NULL != _bfWeightV[unitX] ){
      delete _bfWeightV[unitX];
    }
    _bfWeightV[unitX] = NULL;
  }

  if( _bfWeightV.size() != unitN )
    _bfWeightV.resize( unitN );
      
  for(unsigned unitX=0;unitX<unitN;unitX++){
    _bfWeightV[unitX] = new beamformerWeights( _fftLen, _dim, _halfBandShift, _NC );
  }
  
  return true;
}

bool EigenBeamformer::_calcSteeringUnit( int unitX, bool isGSC )
{
#if 0
  const String thisname = name();
  printf("_calcSteeringUnit %s %d\n", thisname.c_str(), (int)isGSC);
#endif

  gsl_vector_complex* weights;
  unsigned nChan = _channelList.size();

  if( unitX >= _bfWeightV.size() ){
    fprintf(stderr,"Invalid argument %d\n", unitX);
    return false;
  }

  //fprintf(stderr, "calcDCWeights\n");
  weights = _bfWeightV[unitX]->wq_f(0); 
  calcDCWeights( _maxOrder, weights );

  for(unsigned fbinX=1;fbinX<=_fftLen2;fbinX++){
    //fprintf(stderr, "_calcWeights(%d)\n", fbinX);
    weights = _bfWeightV[unitX]->wq_f(fbinX); 
    _calcWeights( fbinX, weights );

    if( true == isGSC ){// calculate a blocking matrix for each frequency bin.
      _bfWeightV[unitX]->calcBlockingMatrix( fbinX );
    }
  }
  _bfWeightV[unitX]->setTimeAlignment();

  return true;
}


void planeWaveOnSphericalAperture( double ka, double theta, double phi, 
				   gsl_vector *theta_s, gsl_vector *phi_s, gsl_vector_complex *p )
{
  for(unsigned chanX=0;chanX<theta_s->size;chanX++){
    double theta_sn = gsl_vector_get( theta_s, chanX );
    double phi_sn   = gsl_vector_get( phi_s,   chanX );
    double ang = ka * ( sin(theta_sn) * sin(theta) * cos( phi_sn - phi ) + cos(theta_sn) * cos(theta) );
    gsl_vector_complex_set( p, chanX,  gsl_complex_polar( 1, ang ) );
  }
  return;
}

/**
   @brief compute the beam pattern at a frequnecy
   @param unsigned fbinX[in] frequency bin
   @param double theta[in] the look direction
   @param double phi[in]   the look direction
   @return the matrix of the beam patters where each column and row indicate the direction of the plane wave impinging on the sphere.
 */
gsl_matrix *EigenBeamformer::getBeamPattern( unsigned fbinX, double theta, double phi,
					     double minTheta, double maxTheta, double minPhi, double maxPhi, double widthTheta, double widthPhi )
{
  float nTheta = ( maxTheta - minTheta ) / widthTheta + 0.5 + 1;
  float nPhi   = ( maxPhi - minPhi ) / widthPhi + 0.5 + 1;
  double ka = 2.0 * M_PI * fbinX * _a * _sampleRate / ( _fftLen * SSPEED );
  gsl_vector_complex *p = gsl_vector_complex_alloc( _theta_s->size );

  if( NULL != _beamPattern )
    gsl_matrix_free( _beamPattern );
  _beamPattern = gsl_matrix_alloc( (int)nTheta, (int)nPhi );
						
  setLookDirection( theta, phi );
  unsigned thetaIdx = 0;
  for(double theta=minTheta;thetaIdx<(int)nTheta;theta+=widthTheta,thetaIdx++){
    unsigned phiIdx = 0;;
    for(double phi=minPhi;phiIdx<(int)nPhi;phi+=widthPhi,phiIdx++){
      gsl_complex val;
      
      planeWaveOnSphericalAperture( ka, theta, phi, _theta_s, _phi_s, p );
      sphericalHarmonicsTransformation( _maxOrder, p, _sh_s, _F );
      gsl_vector_complex *weights = _bfWeightV[0]->wq_f(fbinX);
      gsl_blas_zdotc( weights, _F, &val );
      //gsl_blas_zdotu( weights, _F, &val );
      gsl_matrix_set( _beamPattern, thetaIdx, phiIdx, gsl_complex_abs( val ) );
    }
  }

  gsl_vector_complex_free( p );

  return _beamPattern;
}


// ----- definition for class DOAEstimatorSRPEB' -----
// 

DOAEstimatorSRPEB::DOAEstimatorSRPEB( unsigned nBest, unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, const String& nm ):
  DOAEstimatorSRPBase( nBest, fftLen/2 ),
  EigenBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm )
{
  //_beamformer = new EigenBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, nm );
}

DOAEstimatorSRPEB::~DOAEstimatorSRPEB()
{
}

void DOAEstimatorSRPEB::_calcSteeringUnitTable()
{
  int nChan = (int)chanN();
  if( nChan == 0 ){
    return;
  }

  this->_allocImage( false );

  _nTheta = (unsigned)( ( _maxTheta - _minTheta ) / _widthTheta + 0.5 );
  _nPhi   = (unsigned)( ( _maxPhi - _minPhi ) / _widthPhi + 0.5 );
  int maxUnit  = _nTheta * _nPhi;

  _svTbl.resize(maxUnit);

  for(unsigned i=0;i<maxUnit;i++){
    _svTbl[i] = (gsl_vector_complex **)malloc((_fbinMax+1)*sizeof(gsl_vector_complex *));
    if( NULL == _svTbl[i] ){
      fprintf(stderr,"could not allocate image : %d\n", maxUnit );
    }
    for(unsigned fbinX=0;fbinX<=_fbinMax;fbinX++)
      _svTbl[i][fbinX] = gsl_vector_complex_calloc( _dim );
  }

  if( NULL != _accRPs )
    gsl_vector_free( _accRPs );
  _accRPs = gsl_vector_calloc( maxUnit );

  unsigned unitX = 0;
  unsigned thetaIdx = 0;
  for(double theta=_minTheta;thetaIdx<_nTheta;theta+=_widthTheta,thetaIdx++){
    unsigned phiIdx = 0;
    for(double phi=_minPhi;phiIdx<_nPhi;phi+=_widthPhi,phiIdx++){
      gsl_vector_complex *weights;
      
      setLookDirection( theta, phi );
      weights = _svTbl[unitX][0];
      for(unsigned n=0;n<weights->size;n++)
	gsl_vector_complex_set( weights, n, gsl_complex_rect(1,0) );
      for(unsigned fbinX=_fbinMin;fbinX<=_fbinMax;fbinX++){
	weights = _svTbl[unitX][fbinX];
	_calcWeights( fbinX, weights ); // call the function through the pointer
	//for(unsigned n=0;n<weights->size;n++)
	//gsl_vector_complex_set( weights, n, gsl_complex_conjugate( gsl_vector_complex_get( weights, n ) ) );
      }
      unitX++;
    }    
  }
  
#ifdef __MBDEBUG__
  allocDebugWorkSapce();
#endif /* #ifdef __MBDEBUG__ */

  _isTableInitialized = true;
}

double DOAEstimatorSRPEB::_calcResponsePower( unsigned unitX )
{  
  const gsl_vector_complex* F;       /* spherical transformation coefficient */
  const gsl_vector_complex* weights; /* weights of the combining-unit */
  gsl_complex val;
  double rp  = 0.0;
  
  if( _halfBandShift == false ){

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = _fbinMin; fbinX <= _fbinMax; fbinX++) {
      F = _sphericalTransformSnapShotArray->getSnapShot(fbinX);
      weights = _svTbl[unitX][fbinX];
      gsl_blas_zdotc( weights, F, &val ); // x^H y

      if( fbinX < _fftLen2 ){
	gsl_vector_complex_set(_vector, fbinX, val);
	gsl_vector_complex_set(_vector, _fftLen - fbinX, gsl_complex_conjugate(val) );
	rp += 2 * gsl_complex_abs2( val );
      }
      else{
	gsl_vector_complex_set(_vector, _fftLen2, val);
	rp += gsl_complex_abs2( val );
      }
    }
  }
  else{
    fprintf(stderr,"_halfBandShift == true is not implemented yet\n");
    throw  j_error("_halfBandShift == true is not implemented yet\n");
  }

  return rp / ( _fbinMax - _fbinMin + 1 ); // ( X0^2 + X1^2 + ... + XN^2 )
}

const gsl_vector_complex* DOAEstimatorSRPEB::next( int frameX )
{
  if (frameX == _frameX) return _vector;

  unsigned chanX = 0;
  double rp;

  for(unsigned n=0;n<_nBest;n++){
    gsl_vector_set( _nBestRPs, n, -10e10 );
    gsl_matrix_set( _argMaxDOAs, n, 0, -M_PI);
    gsl_matrix_set( _argMaxDOAs, n, 1, -M_PI);
  }

  if( false == _isTableInitialized )
    _calcSteeringUnitTable();
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frameX);
    if( true==(*itr)->isEnd() ) _endOfSamples = true;
    _snapShotArray->newSample( samp, chanX);  chanX++;
  }
  _snapShotArray->update();

  _energy = calcEnergy( _snapShotArray, _fbinMin, _fbinMax, _fftLen2, _halfBandShift );
  if( _energy < _engeryThreshold ){
#ifdef __MBDEBUG__
    fprintf(stderr,"Energy %e is less than threshold\n", _energy);
#endif /* #ifdef __MBDEBUG__ */
    _increment();
    return _vector;
  }

  // update the spherical harmonics transformation coefficients
  if( _halfBandShift == false ){
    const gsl_vector_complex* XVec;    /* snapshot at each frequency */

    for (unsigned fbinX = _fbinMin; fbinX <=  _fbinMax; fbinX++) {
      XVec    = _snapShotArray->getSnapShot(fbinX);
      sphericalHarmonicsTransformation( _maxOrder, XVec, _sh_s, _F );
      _sphericalTransformSnapShotArray->newSnapShot( (const gsl_vector_complex*)_F, fbinX );
    }
  }

  unsigned unitX = 0;
  unsigned thetaIdx = 0;
  for(double theta=_minTheta;thetaIdx<_nTheta;theta+=_widthTheta,thetaIdx++){
    unsigned phiIdx = 0;
    for(double phi=_minPhi;phiIdx<_nPhi;phi+=_widthPhi,phiIdx++){
      //setLookDirection( theta, phi );
      rp = _calcResponsePower( unitX );
      gsl_vector_set( _accRPs, unitX, gsl_vector_get( _accRPs, unitX ) + rp );
      unitX++;
#ifdef __MBDEBUG__
      gsl_matrix_set( _rpMat, thetaIdx, phiIdx, rp);
#endif /* #ifdef __MBDEBUG__ */
      //   fprintf( stderr, "t=%0.8f p=%0.8f rp=%e\n" , theta, phi, rp );
      if( rp > gsl_vector_get( _nBestRPs, _nBest-1 ) ){
	//  decide the order of the candidates
	for(unsigned n1=0;n1<_nBest;n1++){ 
	  if( rp > gsl_vector_get( _nBestRPs, n1 ) ){
	    // shift the other candidates
	    for(unsigned n2=_nBest-1;n2>n1;n2--){
	      gsl_vector_set( _nBestRPs,   n2, gsl_vector_get( _nBestRPs, n2-1 ) );
	      gsl_matrix_set( _argMaxDOAs, n2, 0, gsl_matrix_get( _argMaxDOAs, n2-1, 0 ) );
	      gsl_matrix_set( _argMaxDOAs, n2, 1, gsl_matrix_get( _argMaxDOAs, n2-1, 1 ) );
	    }
	    // keep this as the n1-th best candidate
	    gsl_vector_set( _nBestRPs, n1, rp );
	    gsl_matrix_set( _argMaxDOAs, n1, 0, theta);
	    gsl_matrix_set( _argMaxDOAs, n1, 1, phi);
	    break;
	  }
	}
	// for(unsinged n1=0;n1<_nBest-1;n1++)
      }
    }
  }

  _increment();
  return _vector;
}

void DOAEstimatorSRPEB::reset()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();

  if ( _snapShotArray != NULL )
    _snapShotArray->zero();

  VectorComplexFeatureStream::reset();
  _endOfSamples = false;
}

// ----- definition for class `SphericalDSBeamformer' -----
//

SphericalDSBeamformer::SphericalDSBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, const String& nm ):
  EigenBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm )
{}

SphericalDSBeamformer::~SphericalDSBeamformer()
{}

gsl_vector *SphericalDSBeamformer::calcWNG()
{
  if( NULL == _WNG )
    _WNG = gsl_vector_alloc( _fftLen2 + 1 );
  double norm = _theta_s->size / ( M_PI * M_PI );

  for (unsigned fbinX = 0; fbinX <= _fftLen2; fbinX++) {
    double val = 0;
    for(int n=0;n<_maxOrder;n++){/* order */
      gsl_complex bn = gsl_matrix_complex_get( _modeAmplitudes, fbinX, n ); 
      double bn2 =  gsl_complex_abs2( bn );
      val += ( (2*n+1) * bn2 );
    }
    gsl_vector_set( _WNG, fbinX, val * val * norm );
  }

  return _WNG;
}

/**
   @brief compute the weights of the delay-and-sum beamformer with the spherical microphone array
   @param unsigned fbinX[in]
   @param gsl_vector_complex *weights[out]
   @notice This is supposed to be called by the member of the child class, EigenBeamformer::_calcSteeringUnit()
 */
void SphericalDSBeamformer::_calcWeights( unsigned fbinX,
					  gsl_vector_complex *weights )
{
  //fprintf( stderr, "SphericalDSBeamformer::_calcWeights\n" );

  for(int n=0,idx=0;n<_maxOrder;n++){/* order */
    gsl_complex bn = gsl_matrix_complex_get( _modeAmplitudes, fbinX, n ); //bn = modeAmplitude( order, ka );
    gsl_complex in;

    if( 0 == ( n % 4 ) ){
      in = gsl_complex_rect(1,0);
    }
    else if( 1 == ( n % 4 ) ){
      in = gsl_complex_rect(0,1);
    }
    else if( 2 == ( n % 4 ) ){
      in = gsl_complex_rect(-1,0);
    }
    else{
      in = gsl_complex_rect(0,-1);
    }
	
    for( int m=-n;m<=n;m++){/* degree */
      gsl_complex weight;
  
      weight = gsl_complex_mul( sphericalHarmonic( m, n, _theta, _phi ),
				gsl_complex_conjugate( gsl_complex_mul( in, bn ) ) );
      gsl_vector_complex_set( weights, idx, gsl_complex_conjugate( gsl_complex_mul_real( weight, 4 * M_PI ) ) );
      idx++;
    }
  }

  if( true==_areWeightsNormalized )
    normalizeWeights( weights, _wgain );

  return;
}

bool SphericalDSBeamformer::_calcSphericalHarmonicsAtEachPosition( gsl_vector *theta_s, gsl_vector *phi_s )
{
  int nChan = (int)theta_s->size;

  if( NULL != _F )
    gsl_vector_complex_free( _F );
  _F = gsl_vector_complex_alloc( _dim );

  if( NULL != _sh_s )
    free(_sh_s);
  _sh_s = (gsl_vector_complex **)malloc(_dim*sizeof(gsl_vector_complex *));
  if( NULL == _sh_s ){
    fprintf(stderr,"_calcSphericalHarmonicsAtEachPosition : cannot allocate memory\n");
    return false;
  }
  for(int dimX=0;dimX<_dim;dimX++)
    _sh_s[dimX] = gsl_vector_complex_alloc( nChan );

  // compute spherical harmonics for each sensor
  for(int n=0,idx=0;n<_maxOrder;n++){/* order */
    for(int m=-n;m<=n;m++){/* degree */
      for(int chanX=0;chanX<nChan;chanX++){
	gsl_complex Ymn_s = sphericalHarmonic( m, n, 
					       gsl_vector_get( theta_s, chanX ), 
					       gsl_vector_get( phi_s, chanX ) );
	gsl_vector_complex_set( _sh_s[idx], chanX, gsl_complex_conjugate( Ymn_s ) );// 
      }
      idx++;
    }
  }

  return true;
}

// ----- definition for class `DualSphericalDSBeamformer' -----
//

DualSphericalDSBeamformer::DualSphericalDSBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, const String& nm ):
  SphericalDSBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm )
{
  _bfWeightV2.resize(1); // the steering vector
  _bfWeightV2[0] = NULL;

}

DualSphericalDSBeamformer::~DualSphericalDSBeamformer()
{}

bool DualSphericalDSBeamformer::_allocSteeringUnit( int unitN )
{
  for(unsigned unitX=0;unitX<_bfWeightV.size();unitX++){
    if( NULL != _bfWeightV[unitX] ){
      delete _bfWeightV[unitX];
    }
    _bfWeightV[unitX] = NULL;

    if( NULL != _bfWeightV2[unitX] ){
      delete _bfWeightV2[unitX];
    }
    _bfWeightV2[unitX] = NULL;
  }

  if( _bfWeightV.size() != unitN ){
    _bfWeightV.resize( unitN );
    _bfWeightV2.resize( unitN );
  }
      
  for(unsigned unitX=0;unitX<unitN;unitX++){
    _bfWeightV[unitX]  = new beamformerWeights( _fftLen, _dim, _halfBandShift, _NC );
    _bfWeightV2[unitX] = new beamformerWeights( _fftLen, chanN(), _halfBandShift, _NC );
  }
  
  return true;
}

/**
   @brief compute the weights of the delay-and-sum beamformer with the spherical microphone array
   @param unsigned fbinX[in]
   @param gsl_vector_complex *weights[out]
   @notice This is supposed to be called by the member of the child class, EigenBeamformer::_calcSteeringUnit()
 */
void DualSphericalDSBeamformer::_calcWeights( unsigned fbinX,
					  gsl_vector_complex *weights )
{
  //fprintf(stderr,"calcSphericalDSBeamformersWeights\n");
  for(int n=0,idx=0;n<_maxOrder;n++){/* order */
    gsl_complex bn = gsl_matrix_complex_get( _modeAmplitudes, fbinX, n ); //bn = modeAmplitude( order, ka );
    gsl_complex in;

    if( 0 == ( n % 4 ) ){
      in = gsl_complex_rect(1,0);
    }
    else if( 1 == ( n % 4 ) ){
      in = gsl_complex_rect(0,1);
    }
    else if( 2 == ( n % 4 ) ){
      in = gsl_complex_rect(-1,0);
    }
    else{
      in = gsl_complex_rect(0,-1);
    }
	
    for( int m=-n;m<=n;m++){/* degree */
      gsl_complex weight;
  
      weight = gsl_complex_mul( sphericalHarmonic( m, n, _theta, _phi ),
				gsl_complex_conjugate( gsl_complex_mul( in, bn ) ) );
      gsl_vector_complex_set( weights, idx,  gsl_complex_conjugate( gsl_complex_mul_real( weight, 4 * M_PI ) ) );
      idx++;
    }
  }

  if( true==_areWeightsNormalized )
    normalizeWeights( weights, _wgain );

  {//  construct the delay-and-sum beamformer in the normal subband domain
    gsl_vector* delays = gsl_vector_alloc( chanN() );

    calcTimeDelaysOfSphericalArray( _theta, _phi, chanN(), _a, _theta_s, _phi_s, delays );
    _bfWeightV2[0]->calcMainlobe( _sampleRate, delays, false );
    _bfWeightV2[0]->setTimeAlignment();
    gsl_vector_free( delays );
  }

  return;
}

// ----- definition for class DOAEstimatorSRPSphDSB' -----
// 

DOAEstimatorSRPSphDSB::DOAEstimatorSRPSphDSB( unsigned nBest, unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, const String& nm ):
  DOAEstimatorSRPBase( nBest, fftLen/2 ),
  SphericalDSBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm )
{
}

DOAEstimatorSRPSphDSB::~DOAEstimatorSRPSphDSB()
{
}

void DOAEstimatorSRPSphDSB::_calcSteeringUnitTable()
{
  int nChan = (int)chanN();
  if( nChan == 0 ){
    return;
  }

  this->_allocImage( false );

  _nTheta = (unsigned)( ( _maxTheta - _minTheta ) / _widthTheta + 0.5 );
  _nPhi   = (unsigned)( ( _maxPhi - _minPhi ) / _widthPhi + 0.5 );
  int maxUnit  = _nTheta * _nPhi;

  _svTbl.resize(maxUnit);

  for(unsigned i=0;i<maxUnit;i++){
    _svTbl[i] = (gsl_vector_complex **)malloc((_fbinMax+1)*sizeof(gsl_vector_complex *));
    if( NULL == _svTbl[i] ){
      fprintf(stderr,"could not allocate image : %d\n", maxUnit );
    }
    for(unsigned fbinX=0;fbinX<=_fbinMax;fbinX++)
      _svTbl[i][fbinX] = gsl_vector_complex_calloc( _dim );
  }

  _accRPs = gsl_vector_calloc( maxUnit );
   
  unsigned unitX = 0;
  unsigned thetaIdx = 0;
  for(double theta=_minTheta;thetaIdx<_nTheta;theta+=_widthTheta,thetaIdx++){
    unsigned phiIdx = 0;
    for(double phi=_minPhi;phiIdx<_nPhi;phi+=_widthPhi,phiIdx++){
      gsl_vector_complex *weights;

      setLookDirection( theta, phi );
      weights = _svTbl[unitX][0];
      for(unsigned n=0;n<weights->size;n++)
	gsl_vector_complex_set( weights, n, gsl_complex_rect(1,0) );
      for(unsigned fbinX=_fbinMin;fbinX<=_fbinMax;fbinX++){
	weights = _svTbl[unitX][fbinX];
	_calcWeights( fbinX, weights ); // call the function through the pointer
	//for(unsigned n=0;n<weights->size;n++)
	//gsl_vector_complex_set( weights, n, gsl_complex_conjugate( gsl_vector_complex_get( weights, n ) ) );
      }
      unitX++;
    }
    
  }
  
#ifdef __MBDEBUG__
  allocDebugWorkSapce();
#endif /* #ifdef __MBDEBUG__ */

  _isTableInitialized = true;
}

double DOAEstimatorSRPSphDSB::_calcResponsePower( unsigned unitX )
{  
  const gsl_vector_complex* weights; /* weights of the combining-unit */
  const gsl_vector_complex* F;       /* spherical transformation coefficient */
  gsl_complex val;
  double rp  = 0.0;
  
  if( _halfBandShift == false ){

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = _fbinMin; fbinX <= _fbinMax; fbinX++) {      
      F = _sphericalTransformSnapShotArray->getSnapShot(fbinX);
      weights = _svTbl[unitX][fbinX];
      gsl_blas_zdotc( weights, F, &val ); // x^H y
      
      if( fbinX < _fftLen2 ){
	gsl_vector_complex_set(_vector, fbinX, val);
	gsl_vector_complex_set(_vector, _fftLen - fbinX, gsl_complex_conjugate(val) );
	rp += 2 * gsl_complex_abs2( val );
      }
      else{
	gsl_vector_complex_set(_vector, _fftLen2, val);
	rp += gsl_complex_abs2( val );
      }
    }
  }
  else{
    fprintf(stderr,"_halfBandShift == true is not implemented yet\n");
    throw  j_error("_halfBandShift == true is not implemented yet\n");
  }

  return rp / ( _fbinMax - _fbinMin + 1 ); // ( X0^2 + X1^2 + ... + XN^2 )
}

const gsl_vector_complex* DOAEstimatorSRPSphDSB::next( int frameX )
{
  if (frameX == _frameX) return _vector;

  unsigned chanX = 0;
  double rp;

  for(unsigned n=0;n<_nBest;n++){
    gsl_vector_set( _nBestRPs, n, -10e10 );
    gsl_matrix_set( _argMaxDOAs, n, 0, -M_PI);
    gsl_matrix_set( _argMaxDOAs, n, 1, -M_PI);
  }

  if( false == _isTableInitialized )
    _calcSteeringUnitTable();
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frameX);
    if( true==(*itr)->isEnd() ) _endOfSamples = true;
    _snapShotArray->newSample( samp, chanX);  chanX++;
  }
  _snapShotArray->update();

  _energy = calcEnergy( _snapShotArray, _fbinMin, _fbinMax, _fftLen2, _halfBandShift );
  if( _energy < _engeryThreshold ){
#ifdef __MBDEBUG__
    fprintf(stderr,"Energy %e is less than threshold\n", _energy);
#endif /* #ifdef __MBDEBUG__ */
    _increment();
    return _vector;
  }

  // update the spherical harmonics transformation coefficients
  if( _halfBandShift == false ){
    const gsl_vector_complex* XVec;    /* snapshot at each frequency */

    for (unsigned fbinX = _fbinMin; fbinX <=  _fbinMax; fbinX++) {
      XVec    = _snapShotArray->getSnapShot(fbinX);
      sphericalHarmonicsTransformation( _maxOrder, XVec, _sh_s, _F );
      _sphericalTransformSnapShotArray->newSnapShot( (const gsl_vector_complex*)_F, fbinX );
    }
  }

  unsigned unitX = 0;
  unsigned thetaIdx = 0;
  for(double theta=_minTheta;thetaIdx<_nTheta;theta+=_widthTheta,thetaIdx++){
    unsigned phiIdx = 0;
    for(double phi=_minPhi;phiIdx<_nPhi;phi+=_widthPhi,phiIdx++){
      //setLookDirection( theta, phi );
      rp = _calcResponsePower( unitX );
      gsl_vector_set( _accRPs, unitX, gsl_vector_get( _accRPs, unitX ) + rp );
      unitX++;
#ifdef __MBDEBUG__
      gsl_matrix_set( _rpMat, thetaIdx, phiIdx, rp);
#endif /* #ifdef __MBDEBUG__ */
      //   fprintf( stderr, "t=%0.8f p=%0.8f rp=%e\n" , theta, phi, rp );
      if( rp > gsl_vector_get( _nBestRPs, _nBest-1 ) ){
	//  decide the order of the candidates
	for(unsigned n1=0;n1<_nBest;n1++){ 
	  if( rp > gsl_vector_get( _nBestRPs, n1 ) ){
	    // shift the other candidates
	    for(unsigned n2=_nBest-1;n2>n1;n2--){
	      gsl_vector_set( _nBestRPs,   n2, gsl_vector_get( _nBestRPs, n2-1 ) );
	      gsl_matrix_set( _argMaxDOAs, n2, 0, gsl_matrix_get( _argMaxDOAs, n2-1, 0 ) );
	      gsl_matrix_set( _argMaxDOAs, n2, 1, gsl_matrix_get( _argMaxDOAs, n2-1, 1 ) );
	    }
	    // keep this as the n1-th best candidate
	    gsl_vector_set( _nBestRPs, n1, rp );
	    gsl_matrix_set( _argMaxDOAs, n1, 0, theta);
	    gsl_matrix_set( _argMaxDOAs, n1, 1, phi);
	    break;
	  }
	}
	// for(unsinged n1=0;n1<_nBest-1;n1++)
      }
    }
  }

  _increment();
  return _vector;
}
 
void DOAEstimatorSRPSphDSB::reset()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();

  if ( _snapShotArray != NULL )
    _snapShotArray->zero();

  VectorComplexFeatureStream::reset();
  _endOfSamples = false;

  //_initAccs();
}

// ----- definition for class `SphericalDSBeamformer' -----
//

SphericalHWNCBeamformer::SphericalHWNCBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, float ratio, const String& nm ):
  EigenBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm )
{
  _ratio = ratio;
}

SphericalHWNCBeamformer::~SphericalHWNCBeamformer()
{
}

gsl_vector *SphericalHWNCBeamformer::calcWNG()
{
  double nrm = _theta_s->size / ( 16 * M_PI * M_PI );

  if( NULL == _WNG )
    _WNG = gsl_vector_calloc( _fftLen/2+1 );

  for (unsigned fbinX = 0; fbinX <= _fftLen2; fbinX++) {
    double val = 0.0;
    double wng;

    for(int n=0;n<_maxOrder;n++){
      double bn2 = gsl_complex_abs2( gsl_matrix_complex_get( _modeAmplitudes, fbinX, n ) );
      val += ( ( 2 * n + 1 ) * bn2 );
    }
    wng = nrm * val * _ratio;
    gsl_vector_set( _WNG, fbinX, wng );
  }

  //fprintf(stderr,"%e\n", wng);
  return _WNG;
}

/**
   @brief compute the weights of the delay-and-sum beamformer with the spherical microphone array
   @param unsigned fbinX[in]
   @param gsl_vector_complex *weights[out]
   @notice This is supposed to be called by the member of the child class, EigenBeamformer::_calcSteeringUnit()
 */
void SphericalHWNCBeamformer::_calcWeights( unsigned fbinX,
					    gsl_vector_complex *weights )
{
  unsigned nChan = (unsigned)_phi_s->size;
  unsigned norm = _dim * nChan;

  //fprintf(stderr,"calcSphericalDSBeamformersWeights\n");
  for(int n=0,idx=0;n<_maxOrder;n++){/* order */
    gsl_complex bn = gsl_matrix_complex_get( _modeAmplitudes, fbinX, n ); //bn = modeAmplitude( order, ka );
    double      bn2 = gsl_complex_abs2( bn ) + _sigma2;
    double      de  = norm * bn2;
    gsl_complex in;
    gsl_complex inbn;

    if( 0 == ( n % 4 ) ){
      in = gsl_complex_rect(1,0);
    }
    else if( 1 == ( n % 4 ) ){
      in = gsl_complex_rect(0,1);
    }
    else if( 2 == ( n % 4 ) ){
      in = gsl_complex_rect(-1,0);
    }
    else{
      in = gsl_complex_rect(0,-1);
    }
	
    inbn = gsl_complex_mul( in, bn );
    for( int m=-n;m<=n;m++){/* degree */
      gsl_complex weight;
      gsl_complex YmnA = gsl_complex_conjugate(sphericalHarmonic( m, n, _theta, _phi ));

      weight = gsl_complex_div_real( gsl_complex_mul( gsl_complex_mul_real( YmnA, 4 * M_PI ), inbn ), de ); // HMDI beamfomrer; see S Yan's paper
      gsl_vector_complex_set( weights, idx, weight ); 
      idx++;
    }
  }

  if( _ratio > 0.0 ){ 
    // control the white noise gain
    if( NULL == _WNG ){ calcWNG();}
    double wng = gsl_vector_get( _WNG, fbinX );
    normalizeWeights( weights, 2 * sqrt( M_PI / ( nChan * wng) ) );
  }
  else{
    double coeff = ( 16 * M_PI * M_PI ) / ( nChan * _maxOrder * _maxOrder );
    gsl_blas_zdscal( coeff, weights );
    //for(unsigned i=0;i<weights->size;i++)
    //  gsl_vector_complex_set( weights, i, gsl_complex_mul_real( gsl_vector_complex_get( weights, i ), coeff ) );
  }

  return;
}

// ----- definition for class `SphericalGSCBeamformer' -----
// 

SphericalGSCBeamformer::SphericalGSCBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, const String& nm )
  :SphericalDSBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm )
{}

SphericalGSCBeamformer::~SphericalGSCBeamformer()
{}

const gsl_vector_complex* SphericalGSCBeamformer::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  unsigned chanX = 0;
  const gsl_vector_complex* XVec;   /* snapshot at each frequency */
  gsl_vector_complex* wq_f; /* weights of the combining-unit */
  gsl_vector_complex* wl_f;
  gsl_complex val;

  this->_allocImage();
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frameX);
    if( true==(*itr)->isEnd() ) _endOfSamples = true;
    _snapShotArray->newSample( samp, chanX);  chanX++;
  }
  _snapShotArray->update();
  
  if( _halfBandShift == false ){
    // calculate a direct component.
    XVec    = _snapShotArray->getSnapShot(0);
    sphericalHarmonicsTransformation( _maxOrder, XVec, _sh_s, _F );
    _sphericalTransformSnapShotArray->newSnapShot( (const gsl_vector_complex*)_F, 0 );
    wq_f = _bfWeightV[0]->wq_f(0);
    gsl_blas_zdotc( wq_f, _F, &val );
    gsl_vector_complex_set(_vector, 0, val);
    //gsl_vector_complex_set( _vector, 0, gsl_vector_complex_get( XVec, XVec->size/2) );
    
    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX <= _fftLen2; fbinX++) {
      XVec  = _snapShotArray->getSnapShot(fbinX);
      sphericalHarmonicsTransformation( _maxOrder, XVec, _sh_s, _F );
      _sphericalTransformSnapShotArray->newSnapShot( (const gsl_vector_complex*)_F, fbinX );
      wq_f = _bfWeightV[0]->wq_f(fbinX);
      wl_f = _bfWeightV[0]->wl_f(fbinX);
      
      calcOutputOfGSC( _F, wl_f, wq_f, &val, _areWeightsNormalized );
      if( fbinX < _fftLen2 ){
	gsl_vector_complex_set(_vector, fbinX, val);
	gsl_vector_complex_set(_vector, _fftLen - fbinX, gsl_complex_conjugate(val) );
      }
      else
	gsl_vector_complex_set(_vector, _fftLen2, val);
    }
  }
  else{
    fprintf(stderr,"_halfBandShift == true is not implemented yet\n");
    throw j_error("_halfBandShift == true is not implemented yet\n");
  }

  _increment();
  return _vector;
}
 
void SphericalGSCBeamformer::reset()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();

  if ( NULL != _snapShotArray )
    _snapShotArray->zero();
  
  if( NULL != _sphericalTransformSnapShotArray )
    _sphericalTransformSnapShotArray->zero();
  
  VectorComplexFeatureStream::reset();
  _endOfSamples = false;
}

/**
   @brief set the look direction for the steering vector.
   @param double theta[in] 
   @param double phi[in]
   @param isGSC[in]
 */
void SphericalGSCBeamformer::setLookDirection( double theta, double phi )
{
  //fprintf(stderr," SphericalGSCBeamformer::setLookDirection()\n");

  _theta = theta;
  _phi   = phi;

  if( NULL == _modeAmplitudes )
    _calcModeAmplitudes();

  if( NULL == _bfWeightV[0] )
    _allocSteeringUnit(1);
  
  _calcSteeringUnit( 0, true /* isGSC */  );
}

/**
   @brief set active weights for each frequency bin.
   @param int fbinX
   @param const gsl_vector* packedWeight
*/
void SphericalGSCBeamformer::setActiveWeights_f( unsigned fbinX, const gsl_vector* packedWeight )
{
  if( 0 == _bfWeightV.size() ){
    fprintf(stderr,"call setLookDirection() once\n");
    throw  j_error("call setLookDirection() once\n");
  }

  _bfWeightV[0]->calcSidelobeCancellerP_f( fbinX, packedWeight );
}

// ----- definition for class `SphericalHWNCGSCBeamformer' -----
// 

SphericalHWNCGSCBeamformer::SphericalHWNCGSCBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, float ratio, const String& nm )
  :SphericalHWNCBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, ratio, nm )
{}

SphericalHWNCGSCBeamformer::~SphericalHWNCGSCBeamformer()
{}

const gsl_vector_complex* SphericalHWNCGSCBeamformer::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  unsigned chanX = 0;
  const gsl_vector_complex* XVec;   /* snapshot at each frequency */
  gsl_vector_complex* wq_f; /* weights of the combining-unit */
  gsl_vector_complex* wl_f;
  gsl_complex val;

  this->_allocImage();
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frameX);
    if( true==(*itr)->isEnd() ) _endOfSamples = true;
    _snapShotArray->newSample( samp, chanX);  chanX++;
  }
  _snapShotArray->update();
  
  if( _halfBandShift == false ){
    // calculate a direct component.
    XVec    = _snapShotArray->getSnapShot(0);
    sphericalHarmonicsTransformation( _maxOrder, XVec, _sh_s, _F );
    _sphericalTransformSnapShotArray->newSnapShot( (const gsl_vector_complex*)_F, 0 );
    wq_f = _bfWeightV[0]->wq_f(0);
    gsl_blas_zdotc( wq_f, _F, &val );
    gsl_vector_complex_set(_vector, 0, val);
    //gsl_vector_complex_set( _vector, 0, gsl_vector_complex_get( XVec, XVec->size/2) );
    
    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX <= _fftLen2; fbinX++) {
      XVec  = _snapShotArray->getSnapShot(fbinX);
      sphericalHarmonicsTransformation( _maxOrder, XVec, _sh_s, _F );
      _sphericalTransformSnapShotArray->newSnapShot( (const gsl_vector_complex*)_F, fbinX );
      wq_f = _bfWeightV[0]->wq_f(fbinX);
      wl_f = _bfWeightV[0]->wl_f(fbinX);
      
      calcOutputOfGSC( _F, wl_f, wq_f, &val, _areWeightsNormalized );
      if( fbinX < _fftLen2 ){
	gsl_vector_complex_set(_vector, fbinX, val);
	gsl_vector_complex_set(_vector, _fftLen - fbinX, gsl_complex_conjugate(val) );
      }
      else
	gsl_vector_complex_set(_vector, _fftLen2, val);
    }
  }
  else{
    fprintf(stderr,"_halfBandShift == true is not implemented yet\n");
    throw j_error("_halfBandShift == true is not implemented yet\n");
  }

  _increment();
  return _vector;
}
 
void SphericalHWNCGSCBeamformer::reset()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();

  if ( NULL != _snapShotArray )
    _snapShotArray->zero();
  
  if( NULL != _sphericalTransformSnapShotArray )
    _sphericalTransformSnapShotArray->zero();
  
  VectorComplexFeatureStream::reset();
  _endOfSamples = false;
}

/**
   @brief set the look direction for the steering vector.
   @param double theta[in] 
   @param double phi[in]
   @param isGSC[in]
 */
void SphericalHWNCGSCBeamformer::setLookDirection( double theta, double phi )
{
  //fprintf(stderr," SphericalGSCBeamformer::setLookDirection()\n");

  _theta = theta;
  _phi   = phi;

  if( NULL == _modeAmplitudes )
    _calcModeAmplitudes();

  if( NULL == _bfWeightV[0] )
    _allocSteeringUnit(1);
  
  _calcSteeringUnit( 0, true /* isGSC */  );
}

/**
   @brief set active weights for each frequency bin.
   @param int fbinX
   @param const gsl_vector* packedWeight
*/
void SphericalHWNCGSCBeamformer::setActiveWeights_f( unsigned fbinX, const gsl_vector* packedWeight )
{
  if( 0 == _bfWeightV.size() ){
    fprintf(stderr,"call setLookDirection() once\n");
    throw  j_error("call setLookDirection() once\n");
  }

  _bfWeightV[0]->calcSidelobeCancellerP_f( fbinX, packedWeight );
}

// ----- definition for class `DualSphericalGSCBeamformer' -----
// 

DualSphericalGSCBeamformer::DualSphericalGSCBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, const String& nm )
  :SphericalGSCBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm )
{}

DualSphericalGSCBeamformer::~DualSphericalGSCBeamformer()
{}

bool DualSphericalGSCBeamformer::_allocSteeringUnit( int unitN )
{
  for(unsigned unitX=0;unitX<_bfWeightV.size();unitX++){
    if( NULL != _bfWeightV[unitX] ){
      delete _bfWeightV[unitX];
    }
    _bfWeightV[unitX] = NULL;

    if( NULL != _bfWeightV2[unitX] ){
      delete _bfWeightV2[unitX];
    }
    _bfWeightV2[unitX] = NULL;
  }

  if( _bfWeightV.size() != unitN ){
    _bfWeightV.resize( unitN );
    _bfWeightV2.resize( unitN );
  }
      
  for(unsigned unitX=0;unitX<unitN;unitX++){
    _bfWeightV[unitX]  = new beamformerWeights( _fftLen, _dim, _halfBandShift, _NC );
    _bfWeightV2[unitX] = new beamformerWeights( _fftLen, chanN(), _halfBandShift, _NC );
  }
  
  return true;
}

/**
   @brief compute the weights of the delay-and-sum beamformer with the spherical microphone array
   @param unsigned fbinX[in]
   @param gsl_vector_complex *weights[out]
   @notice This is supposed to be called by the member of the child class, EigenBeamformer::_calcSteeringUnit()
 */
void DualSphericalGSCBeamformer::_calcWeights( unsigned fbinX, gsl_vector_complex *weights )
{
  //fprintf(stderr,"calcSphericalDSBeamformersWeights\n");
  for(int n=0,idx=0;n<_maxOrder;n++){/* order */
    gsl_complex bn = gsl_matrix_complex_get( _modeAmplitudes, fbinX, n ); //bn = modeAmplitude( order, ka );
    gsl_complex in;

    if( 0 == ( n % 4 ) ){
      in = gsl_complex_rect(1,0);
    }
    else if( 1 == ( n % 4 ) ){
      in = gsl_complex_rect(0,1);
    }
    else if( 2 == ( n % 4 ) ){
      in = gsl_complex_rect(-1,0);
    }
    else{
      in = gsl_complex_rect(0,-1);
    }
	
    for( int m=-n;m<=n;m++){/* degree */
      gsl_complex weight;
  
      weight = gsl_complex_mul( sphericalHarmonic( m, n, _theta, _phi ),
				gsl_complex_conjugate( gsl_complex_mul( in, bn ) ) );
      gsl_vector_complex_set( weights, idx,  gsl_complex_conjugate( gsl_complex_mul_real( weight, 4 * M_PI ) ) );
      idx++;
    }
  }

  if( true==_areWeightsNormalized )
    normalizeWeights( weights, _wgain );

  {//  construct the delay-and-sum beamformer in the normal subband domain
    gsl_vector* delays = gsl_vector_alloc( chanN() );

    calcTimeDelaysOfSphericalArray( _theta, _phi, chanN(), _a, _theta_s, _phi_s, delays );
    _bfWeightV2[0]->calcMainlobe( _sampleRate, delays, false );
    _bfWeightV2[0]->setTimeAlignment();
    gsl_vector_free( delays );
  }

  return;
}


// ----- definition for class `SphericalMOENBeamformer' -----
// 

SphericalMOENBeamformer::SphericalMOENBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, const String& nm )
  :SphericalDSBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm )
{
  _dThreshold = 1.0E-8;
  _isTermFixed = false;
  _CN = 2.0 / ( maxOrder * maxOrder ); // maxOrder = N + 1
  _orderOfBF = maxOrder;

  _A      = (gsl_matrix_complex** )malloc( (fftLen/2+1) * sizeof(gsl_matrix_complex*) );
  _fixedW = (gsl_matrix_complex** )malloc( (fftLen/2+1) * sizeof(gsl_matrix_complex*) );
  if( _A == NULL || _fixedW == NULL ){
    fprintf(stderr,"SphericalMOENBeamformer: gsl_matrix_complex_alloc failed\n");
    throw jallocation_error("SphericalMOENBeamformer: gsl_matrix_complex_alloc failed\n");
  }
  
  _BN = (gsl_vector_complex** )malloc( (fftLen/2+1) * sizeof(gsl_vector_complex*) );
  if( _BN == NULL ){
    fprintf(stderr,"SphericalMOENBeamformer: gsl_vector_complex_alloc failed\n");
    throw jallocation_error("SphericalMOENBeamformer: gsl_vector_complex_alloc failed\n");
  }

  _diagonalWeights = (float *)calloc( (fftLen/2+1), sizeof(float) );
  if( _diagonalWeights == NULL ){
    fprintf(stderr,"SphericalMOENBeamformer: cannot allocate RAM\n");
    throw jallocation_error("SphericalMOENBeamformer: cannot allocate RAM\n");
  }

  for( unsigned fbinX=0;fbinX<=fftLen/2;fbinX++){
    _A[fbinX]       = NULL;
    _fixedW[fbinX]  = NULL;
    _BN[fbinX]      = NULL;
    _diagonalWeights[fbinX] = 0.0;
  }

}

SphericalMOENBeamformer::~SphericalMOENBeamformer()
{
  unsigned fftLen2 = _fftLen / 2;
  
  for( unsigned fbinX=0;fbinX<=fftLen2;fbinX++){
    if( _A[fbinX] != NULL )
      gsl_matrix_complex_free( _A[fbinX] );
    if( _fixedW[fbinX] != NULL )
      gsl_matrix_complex_free( _fixedW[fbinX] );
    if( _BN[fbinX] != NULL )
      gsl_vector_complex_free( _BN[fbinX] );
  }
  free(_A);
  free(_fixedW);
  free(_BN);
  free(_diagonalWeights);
}

const gsl_vector_complex* SphericalMOENBeamformer::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  unsigned chanX = 0;
  const gsl_vector_complex* XVec;   /* snapshot at each frequency */
  gsl_vector_complex* wq_f; /* weights of the combining-unit */
  gsl_vector_complex* wl_f;
  gsl_complex val;

  this->_allocImage();
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frameX);
    if( true==(*itr)->isEnd() ) _endOfSamples = true;
    _snapShotArray->newSample( samp, chanX);  chanX++;
  }
  _snapShotArray->update();
  
  if( _halfBandShift == false ){
    // calculate a direct component.
    XVec = _snapShotArray->getSnapShot(0);
    wq_f = _bfWeightV[0]->wq_f(0);
    gsl_blas_zdotc( wq_f, XVec, &val );
    //gsl_blas_zdotu( wq_f, XVec, &val );
    gsl_vector_complex_set(_vector, 0, val);
    
    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX <= _fftLen2; fbinX++) {
      XVec = _snapShotArray->getSnapShot(fbinX);
      wq_f = _bfWeightV[0]->wq_f(fbinX);
      gsl_blas_zdotc( wq_f, XVec, &val );
      //gsl_blas_zdotu( wq_f, XVec, &val ); 

      if( fbinX < _fftLen2 ){
	gsl_vector_complex_set(_vector, fbinX, val);
	gsl_vector_complex_set(_vector, _fftLen - fbinX, gsl_complex_conjugate(val) );
      }
      else
	gsl_vector_complex_set(_vector, _fftLen2, val);
    }
  }
  else{
    fprintf(stderr,"_halfBandShift == true is not implemented yet\n");
    throw j_error("_halfBandShift == true is not implemented yet\n");
  }

  _increment();
  return _vector;
}
 
void SphericalMOENBeamformer::reset()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();

  if ( NULL != _snapShotArray )
    _snapShotArray->zero();
  
  if( NULL != _sphericalTransformSnapShotArray )
    _sphericalTransformSnapShotArray->zero();
  
  VectorComplexFeatureStream::reset();
  _endOfSamples = false;
}

void SphericalMOENBeamformer::setLevelOfDiagonalLoading( unsigned fbinX, float diagonalWeight )
{
  if( fbinX > _fftLen/2 ){
    fprintf(stderr,"SphericalMOENBeamformer::setLevelOfDiagonalLoading() : Invalid freq. bin %d\n", fbinX);
    return;
  }
  _diagonalWeights[fbinX] = diagonalWeight;
}

/*
  @brief calcualte matrices, _A and _BN.

  @note this function is called by _calcSteeringUnit() which is called by setLookDirection().
 */
void SphericalMOENBeamformer::_calcWeights( unsigned fbinX, gsl_vector_complex *weights )
{
  //fprintf(stderr,"calcSphericalDSBeamformersWeights\n");
  unsigned nChan = _theta_s->size;

  if( _A[fbinX] == NULL ){
    _A[fbinX]  = gsl_matrix_complex_alloc( _dim, nChan );
    _BN[fbinX] = gsl_vector_complex_calloc( _dim );
  }
  else
    gsl_vector_complex_set_zero( _BN[fbinX] );
  
  for(int n=0,idx=0;n<_maxOrder;n++){/* order */
    gsl_complex bn = gsl_matrix_complex_get( _modeAmplitudes, fbinX, n ); //bn = modeAmplitude( order, ka );
    gsl_complex in;
    
    if( 0 == ( n % 4 ) ){
      in = gsl_complex_rect(1,0);
    }
    else if( 1 == ( n % 4 ) ){
      in = gsl_complex_rect(0,1);
    }
    else if( 2 == ( n % 4 ) ){
      in = gsl_complex_rect(-1,0);
    }
    else{
      in = gsl_complex_rect(0,-1);
    }
    
    for( int m=-n;m<=n;m++){/* degree */
      for(int chanX=0;chanX<nChan;chanX++){
	if( false == _isTermFixed ) {
	  gsl_complex YAmn_s = gsl_vector_complex_get( _sh_s[idx], chanX );
	  gsl_complex val = gsl_complex_mul( YAmn_s, gsl_complex_mul( in, bn ) );
	  //gsl_complex val = gsl_complex_div( gsl_complex_conjugate(YAmn_s), gsl_complex_mul( in, bn ) );
	  gsl_matrix_complex_set( _A[fbinX], idx, chanX, gsl_complex_mul_real( val, 4 * M_PI ) );
	}
	if( n < _orderOfBF ){
	  //gsl_vector_complex_set( _BN[fbinX], idx, gsl_complex_mul_real( sphericalHarmonic( m, n, _theta, _phi ), 2 * M_PI ) );
	   gsl_vector_complex_set( _BN[fbinX], idx, gsl_complex_mul_real( gsl_complex_conjugate( sphericalHarmonic( m, n, _theta, _phi ) ), 2 * M_PI ) );
	}
      }
      idx++;
    }
  }

  _calcMOENWeights( fbinX, weights, _dThreshold, _isTermFixed, 0 );

  return;
}


bool SphericalMOENBeamformer::_calcMOENWeights( unsigned fbinX, gsl_vector_complex *weights, double dThreshold, bool calcFixedTerm, unsigned unitX )
{
  unsigned nChan = _theta_s->size;
  
  if( NULL != _fixedW[fbinX] ){
    gsl_matrix_complex_free( _fixedW[fbinX] );
  }
  _fixedW[fbinX] = gsl_matrix_complex_calloc( nChan, _dim );
  
#if 0
  for(unsigned chanX=0;chanX<weights->size;chanX++)
    gsl_vector_complex_set( weights, chanX, gsl_complex_rect( 1.0, 0.0 ) );
#endif

  if( false == calcFixedTerm ){
    gsl_matrix_complex* tmp = gsl_matrix_complex_calloc( nChan, nChan );
    //gsl_matrix_complex* AH  = gsl_matrix_complex_calloc( nChan, _dim );
    for(unsigned chanX=0;chanX<nChan;chanX++)
      gsl_matrix_complex_set( tmp, chanX, chanX, gsl_complex_rect( 1.0, 0.0 ) );
    
    gsl_blas_zherk( CblasUpper, CblasConjTrans, 1.0, _A[fbinX], _diagonalWeights[fbinX], tmp ); // A^H A + l^2 I 
    // can be implemented in the faster way
    for(unsigned chanX=0;chanX<nChan;chanX++)
      for(unsigned chanY=chanX;chanY<nChan;chanY++)
	gsl_matrix_complex_set( tmp, chanY, chanX, gsl_complex_conjugate( gsl_matrix_complex_get( tmp, chanX, chanY) ) );
    if( false==pseudoinverse( tmp, tmp, dThreshold )){ //( A^H A + l^2 I )^{-1}
      fprintf(stderr,"fbinX %d : pseudoinverse() failed\n",fbinX);
#if 1
      for(unsigned chanX=0;chanX<nChan;chanX++){
	for(unsigned chanY=0;chanY<_dim;chanY++){
	  fprintf(stderr,"%0.2e + i %0.2e, ", GSL_REAL(gsl_matrix_complex_get(tmp, chanX, chanY)), GSL_IMAG(gsl_matrix_complex_get(tmp, chanX, chanY)) );
	}
	fprintf(stderr,"\n");
      }
#endif

    }

    gsl_blas_zgemm( CblasNoTrans, CblasConjTrans, gsl_complex_rect( 1.0, 0.0 ), tmp, _A[fbinX], gsl_complex_rect( 0.0, 0.0 ), _fixedW[fbinX] ); //( A^H A + l^2 I )^{-1} A^H
    gsl_blas_zgemv( CblasNoTrans, gsl_complex_rect( _CN, 0.0 ), (const gsl_matrix_complex*)_fixedW[fbinX], (const gsl_vector_complex*)_BN[fbinX], gsl_complex_rect( 0.0, 0.0 ), weights );// ( A^H A + l^2 I )^{-1} A^H BN  
    gsl_matrix_complex_free( tmp );
  }
  else{
    //gsl_blas_zhemv( CblasUpper, gsl_complex_rect( _CN, 0.0 ), (const gsl_matrix_complex*)_fixedW[fbinX], (const gsl_matrix_complex*)_BN[fbinX], gsl_complex_rect( 0.0, 0.0 ), weights );
    gsl_blas_zgemv( CblasNoTrans, gsl_complex_rect( _CN, 0.0 ), (const gsl_matrix_complex*)_fixedW[fbinX], (const gsl_vector_complex*)_BN[fbinX], gsl_complex_rect( 0.0, 0.0 ), weights );// ( A^H A + l^2 I )^{-1} A^H BN  
  }

  if( true==_areWeightsNormalized )
    normalizeWeights( weights, _wgain );

  return true;
}

bool SphericalMOENBeamformer::_allocSteeringUnit( int unitN )
{
  for(unsigned unitX=0;unitX<_bfWeightV.size();unitX++){
    if( NULL != _bfWeightV[unitX] ){
      delete _bfWeightV[unitX];
    }
    _bfWeightV[unitX] = NULL;
  }

  if( _bfWeightV.size() != unitN )
    _bfWeightV.resize( unitN );
      
  for(unsigned unitX=0;unitX<unitN;unitX++){
    _bfWeightV[unitX] = new beamformerWeights( _fftLen, _theta_s->size, _halfBandShift, _NC );
  }
  
  return true;
}

/**
   @brief compute the beam pattern at a frequnecy
   @param unsigned fbinX[in] frequency bin
   @param double theta[in] the look direction
   @param double phi[in]   the look direction
   @return the matrix of the beam patters where each column and row indicate the direction of the plane wave impinging on the sphere.
 */
gsl_matrix *SphericalMOENBeamformer::getBeamPattern( unsigned fbinX, double theta, double phi,
						     double minTheta, double maxTheta, double minPhi, double maxPhi,
						     double widthTheta, double widthPhi )
{
  float nTheta = ( maxTheta - minTheta ) / widthTheta + 0.5 + 1;
  float nPhi   = ( maxPhi - minPhi ) / widthPhi + 0.5 + 1;
  double ka = 2.0 * M_PI * fbinX * _a * _sampleRate / ( _fftLen * SSPEED );
  gsl_vector_complex *p = gsl_vector_complex_alloc( _theta_s->size );

  if( NULL != _beamPattern )
    gsl_matrix_free( _beamPattern );
  _beamPattern = gsl_matrix_alloc( (int)nTheta, (int)nPhi );
						
  setLookDirection( theta, phi );
  unsigned thetaIdx = 0;
  for(double theta=minTheta;thetaIdx<(int)nTheta;theta+=widthTheta,thetaIdx++){
    unsigned phiIdx = 0;;
    for(double phi=minPhi;phiIdx<(int)nPhi;phi+=widthPhi,phiIdx++){
      gsl_complex val;
      
      planeWaveOnSphericalAperture( ka, theta, phi, _theta_s, _phi_s, p );
      gsl_vector_complex *weights = _bfWeightV[0]->wq_f(fbinX);
      //gsl_blas_zdotc( weights, p, &val );
      gsl_blas_zdotu( weights, p, &val );
      gsl_matrix_set( _beamPattern, thetaIdx, phiIdx, gsl_complex_abs( val ) );
    }
  }

  gsl_vector_complex_free( p );

  return _beamPattern;
}



// ----- definition for class `SphericalSpatialDSBeamformer' -----
//

SphericalSpatialDSBeamformer::SphericalSpatialDSBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, bool normalizeWeight, const String& nm ):
  SphericalDSBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm )
{}

SphericalSpatialDSBeamformer::~SphericalSpatialDSBeamformer()
{}

/**
   @brief compute the weights of the delay-and-sum beamformer with the spherical microphone array
   @param unsigned fbinX[in]
   @param gsl_vector_complex *weights[out]
   @notice This is supposed to be called by the member of the child class, EigenBeamformer::_calcSteeringUnit()
 */
void SphericalSpatialDSBeamformer::_calcWeights( unsigned fbinX,
						 gsl_vector_complex *weights )
{
  int nChan = (int)chanN();

  for ( int s = 0; s < nChan; s++ ) { /* channnel */
    /* compute the approximation of the sound pressure at sensor s with the spherical harmonics coefficients, */
    /* G(Omega_s,ka,Omega) = 4pi \sum_{n=0}^{N} i^n b_n(ka) \sum_{m=-n}^{n} Ymn(Omega_s) Ymn(Omega)^*         */
    gsl_complex weight = gsl_complex_rect( 0, 0 );
    for ( int n = 0, idx = 0; n < _maxOrder; n++ ) { /* order */
      gsl_complex bn = gsl_matrix_complex_get( _modeAmplitudes, fbinX, n ); // bn = modeAmplitude( order, ka );
      gsl_complex in, inbn;
      gsl_complex tmp_weight;

      if ( 0 == ( n % 4 ) ) {
	in = gsl_complex_rect( 1, 0 );
      }
      else if ( 1 == ( n % 4 ) ) {
	in = gsl_complex_rect( 0, 1 );
      }
      else if ( 2 == ( n % 4 ) ) {
	in = gsl_complex_rect( -1, 0 );
      }
      else {
	in = gsl_complex_rect( 0, -1 );
      }
      inbn = gsl_complex_mul( in, bn );

      tmp_weight = gsl_complex_rect( 0, 0 );
      for ( int m = -n; m <= n; m++ ) { /* degree */
	tmp_weight = gsl_complex_add( tmp_weight, 
				      gsl_complex_mul( gsl_complex_conjugate( gsl_vector_complex_get( _sh_s[idx], s ) ), /* this has to be conjugated to get Ymn */
						       gsl_complex_conjugate( sphericalHarmonic( m, n, _theta, _phi ) ) ) );
	idx++;
      }
      weight = gsl_complex_add( weight, gsl_complex_mul( inbn, tmp_weight ) );
    }
    gsl_vector_complex_set( weights, s, gsl_complex_mul_real( weight, 4 * M_PI / nChan ) );// the Euclidean norm gsl_blas_dznrm2( weights )
  }

  /* Normalization */
  // if ( true == _areWeightsNormalized )
  // normalizeWeights( weights, _wgain ); /* <- Correct? */
  
  //double ka = 2.0 * M_PI * fbinX * _a * _sampleRate / ( _fftLen * SSPEED );
  //gsl_complex Gpw = gsl_complex_polar( 1.0, ka * cos( _theta ) );
  //for ( unsigned i = 0; i < nChan; i++) {
  //gsl_vector_complex_set( weights, i, gsl_complex_conjugate(gsl_complex_mul( Gpw, norm_weight ) ) );
  //gsl_complex norm_weight = gsl_complex_div_real( gsl_vector_complex_get( weights, i ), nrm * nChan );
  //gsl_vector_complex_set( weights, i, gsl_complex_conjugate(gsl_complex_mul( Gpw, norm_weight ) ) );
  //}

  return;
}

const gsl_vector_complex* SphericalSpatialDSBeamformer::next( int frameX )
{
  if ( frameX == _frameX ) return _vector;

  unsigned chanX = 0;
  const gsl_vector_complex* XVec;   /* snapshot at each frequency */
  const gsl_vector_complex* weights; /* weights of the combining-unit */
  gsl_complex val;

  this->_allocImage();
  for ( _ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++ ) {
    const gsl_vector_complex* samp = ( *itr )->next( frameX );
    if ( true == ( *itr )->isEnd() ) _endOfSamples = true;
    _snapShotArray->newSample( samp, chanX );
    chanX++;
  }
  _snapShotArray->update();
  
  if ( _halfBandShift == false ) {
    // calculate a direct component.

    XVec    = _snapShotArray->getSnapShot(0);
    weights = _bfWeightV[0]->wq_f(0);

    gsl_blas_zdotc( weights, XVec, &val );
    gsl_vector_complex_set( _vector, 0, val );

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for ( unsigned fbinX = 1; fbinX <= _fftLen2; fbinX++ ) {
      XVec    = _snapShotArray->getSnapShot( fbinX );
      weights = _bfWeightV[0]->wq_f( fbinX );

      gsl_blas_zdotc( weights, XVec, &val );      
      if( fbinX < _fftLen2 ){
	gsl_vector_complex_set( _vector, fbinX, val );
	gsl_vector_complex_set( _vector, _fftLen - fbinX, gsl_complex_conjugate(val) );
      }
      else
	gsl_vector_complex_set( _vector, _fftLen2, val );
    }
  }
  else{
    fprintf( stderr, "_halfBandShift == true is not implemented yet\n" );
    throw j_error( "_halfBandShift == true is not implemented yet\n" );
  }

  _increment();
  return _vector;
}

bool SphericalSpatialDSBeamformer::_allocSteeringUnit( int unitN )
{
  for ( unsigned unitX = 0; unitX < _bfWeightV.size(); unitX++ ) {
    if ( NULL != _bfWeightV[unitX] ) {
      delete _bfWeightV[unitX];
    }
    _bfWeightV[unitX] = NULL;
  }

  if ( _bfWeightV.size() != unitN )
    _bfWeightV.resize( unitN );

  for ( unsigned unitX = 0; unitX < unitN; unitX++ ) {
    _bfWeightV[unitX]
      = new beamformerWeights( _fftLen, (int)chanN(), _halfBandShift, _NC );
  }

  return true;
}

bool SphericalSpatialDSBeamformer::_calcSteeringUnit( int unitX, bool isGSC )
{
  gsl_vector_complex* weights;
  unsigned nChan = _channelList.size();

  if( unitX >= _bfWeightV.size() ){
    fprintf(stderr,"Invalid argument %d\n", unitX);
    return false;
  }

  // weights = _bfWeightV[unitX]->wq_f(0); 
  // calcDCWeights( _maxOrder, weights );

  // for(unsigned fbinX=1;fbinX<=_fftLen2;fbinX++){
  for ( unsigned fbinX = 0; fbinX <= _fftLen2; fbinX++ ) {
    // fprintf(stderr, "_calcWeights(%d)\n", fbinX);
    weights = _bfWeightV[unitX]->wq_f(fbinX); 
    _calcWeights( fbinX, weights );

    if( true == isGSC ){// calculate a blocking matrix for each frequency bin.
      _bfWeightV[unitX]->calcBlockingMatrix( fbinX );
    }
  }
  _bfWeightV[unitX]->setTimeAlignment();

  return true;
}


// ----- definition for class `SphericalSpatialHWNCBeamformer' -----
//
SphericalSpatialHWNCBeamformer::SphericalSpatialHWNCBeamformer( unsigned sampleRate, unsigned fftLen, bool halfBandShift, unsigned NC, unsigned maxOrder, double sigmaSI2, bool normalizeWeight, float ratio, const String& nm ):
  SphericalHWNCBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, ratio, nm ),
  _sigmaSI2(sigmaSI2),_dThreshold(1.0E-8),_SigmaSI(NULL)
{
  unsigned fftLen2 = fftLen/2;

  _SigmaSI = new gsl_matrix_complex*[fftLen2];
  for(unsigned fbinX=0;fbinX<=fftLen2;fbinX++)   
    _SigmaSI[fbinX] = NULL;
}

SphericalSpatialHWNCBeamformer::~SphericalSpatialHWNCBeamformer()
{
  unsigned fftLen2 = _fftLen/2;
    
  for(unsigned fbinX=0;fbinX<=fftLen2;fbinX++)
    if( NULL != _SigmaSI[fbinX] ){
      gsl_matrix_complex_free( _SigmaSI[fbinX] );
    }
  
  delete [] _SigmaSI;
}

/*
  @brief compute the coherence matrix for the diffuse noise field
  @param unsigned fbinX[in]
*/
gsl_matrix_complex *SphericalSpatialHWNCBeamformer::_calcDiffuseNoiseModel( unsigned fbinX )
{
  //fprintf(stderr, "SphericalSpatialHWNCBeamformer::_calcDiffuseNoiseModel %d\n",_dim );
  unsigned fftLen2 = _fftLen/2;
  int nChan = (int)chanN();
  gsl_matrix_complex *A = gsl_matrix_complex_alloc( nChan, _dim );
  gsl_matrix_complex *SigmaSIp = gsl_matrix_complex_calloc( _dim, _dim );
  gsl_matrix_complex *A_SigmaSIp = gsl_matrix_complex_calloc( nChan, _dim );

  if( NULL == _SigmaSI[fbinX] ){
    _SigmaSI[fbinX] = gsl_matrix_complex_calloc( nChan, nChan );
  }

  /* set the left-side matrix, A */
  /* Note: this matrix A is correct! Eq. (180) in the book chapter is incorrect! */
  for(int chanX=0;chanX<nChan;chanX++){
    for(int n=0,idx=0;n<_maxOrder;n++){/* order */
      for(int m=-n;m<=n;m++){/* degree */
	gsl_complex YmnA_s = gsl_vector_complex_get( _sh_s[idx], chanX );
	gsl_matrix_complex_set( A, chanX, idx, gsl_complex_conjugate(YmnA_s) );
	idx++;
      }
    }
  }

  /* compute the covariance matrix in the spherical harmonics domain, SigmaSIp */
  for(int n=0,idx=0;n<_maxOrder;n++){/* order */
    gsl_complex bn = gsl_matrix_complex_get( _modeAmplitudes, fbinX, n ); //bn = modeAmplitude( order, ka );
    double      bn2 = gsl_complex_abs2( bn );
    
    //if( (n%2)!=0 ) bn2 = -bn2;
    for( int m=-n;m<=n;m++){/* degree */
      gsl_matrix_complex_set( SigmaSIp, idx, idx, gsl_complex_rect( bn2, 0.0 ) );
      idx++;
    }
  }

  /* compute the covariance matrix SigmaSI = A * SigmaSIp * A^H */
  gsl_blas_zgemm( CblasNoTrans, CblasNoTrans,   gsl_complex_rect(1,0), A, SigmaSIp, gsl_complex_rect(0,0), A_SigmaSIp );
  gsl_blas_zgemm( CblasNoTrans, CblasConjTrans, gsl_complex_rect(1,0), A_SigmaSIp, A, gsl_complex_rect(0,0), _SigmaSI[fbinX] );

  gsl_matrix_complex_free( A );
  gsl_matrix_complex_free( SigmaSIp );
  gsl_matrix_complex_free( A_SigmaSIp );

  // add the diagonal component
  for(unsigned chanX=0;chanX<nChan;chanX++){
    gsl_matrix_complex_set( _SigmaSI[fbinX], chanX, chanX, gsl_complex_add_real( gsl_matrix_complex_get( _SigmaSI[fbinX], chanX, chanX ), _sigma2 ) );
  }

  return _SigmaSI[fbinX];
}

void SphericalSpatialHWNCBeamformer::_calcWeights( unsigned fbinX, gsl_vector_complex *weights )
{
  //fprintf(stderr, "SphericalSpatialHWNCBeamformer::_calcWeights\n" );
  int nChan = chanN();

  for ( int s = 0; s < nChan; s++ ) { /* channnel */
    /* compute the approximation of the sound pressure at sensor s with the spherical harmonics coefficients, */
    /* G(Omega_s,ka,Omega) = 4pi \sum_{n=0}^{N} i^n b_n(ka) \sum_{m=-n}^{n} Ymn(Omega_s) Ymn(Omega)^*         */
    gsl_complex weight = gsl_complex_rect( 0, 0 );
    for ( int n = 0, idx = 0; n < _maxOrder; n++ ) { /* order */
      gsl_complex bn = gsl_matrix_complex_get( _modeAmplitudes, fbinX, n ); // bn = modeAmplitude( order, ka );
      gsl_complex in, inbn;
      gsl_complex tmp_weight;

      if ( 0 == ( n % 4 ) ) {
	in = gsl_complex_rect( 1, 0 );
      }
      else if ( 1 == ( n % 4 ) ) {
	in = gsl_complex_rect( 0, 1 );
      }
      else if ( 2 == ( n % 4 ) ) {
	in = gsl_complex_rect( -1, 0 );
      }
      else {
	in = gsl_complex_rect( 0, -1 );
      }
      inbn = gsl_complex_mul( in, bn );

      tmp_weight = gsl_complex_rect( 0, 0 );
      for ( int m = -n; m <= n; m++ ) { /* degree */
	tmp_weight = gsl_complex_add( tmp_weight, 
				      gsl_complex_mul( gsl_complex_conjugate( gsl_vector_complex_get( _sh_s[idx], s ) ), /* this has to be conjugated to get Ymn */
						       gsl_complex_conjugate( sphericalHarmonic( m, n, _theta, _phi ) ) ) );
	idx++;
      }
      weight = gsl_complex_add( weight, gsl_complex_mul( inbn, tmp_weight ) );
    }
    //gsl_vector_complex_set( weights, s, gsl_complex_mul_real( weight, 4 * M_PI ) ); 
  }

  {// compute the coherence matrix of the diffuse noise field; the result is set to _SigmaSI. 
    _calcDiffuseNoiseModel( fbinX );

    gsl_matrix_complex *iSigmaSI   = gsl_matrix_complex_calloc( nChan, nChan );
    gsl_vector_complex *iSigmaSI_v = gsl_vector_complex_calloc( nChan );
    gsl_complex lambda, ilambda;   
    double norm = gsl_blas_dznrm2( weights );
    
    gsl_blas_zdscal( 1/norm, weights );
    
    if ( false == pseudoinverse( _SigmaSI[fbinX], iSigmaSI, _dThreshold ) )
      fprintf( stderr, "fbinX %d : pseudoinverse() failed\n", fbinX );
    
    gsl_blas_zgemv( CblasNoTrans, gsl_complex_rect( 1.0, 0.0 ), iSigmaSI, weights, 
		    gsl_complex_rect( 0.0, 0.0), iSigmaSI_v ); // SigmaSI^{-1} v
    gsl_blas_zdotc( weights, iSigmaSI_v, &lambda ); // lambda = vH * SigmaSI^{-1} * v    
    ilambda = gsl_complex_inverse( lambda ); // ilambda = 1 / lambda
    gsl_blas_zscal( ilambda, iSigmaSI_v );   // iSigmaSI_v = ilambda * iSigmaSI_v 
    gsl_vector_complex_memcpy( weights, iSigmaSI_v );
    
    gsl_matrix_complex_free( iSigmaSI );
    gsl_vector_complex_free( iSigmaSI_v );

  }
  {
    if( _ratio > 0.0 ){ 
      // control the white noise gain
      if( NULL == _WNG ){ calcWNG();}
      double wng = gsl_vector_get( _WNG, fbinX );
      normalizeWeights( weights, 2 * sqrt( M_PI / ( nChan * wng) ) );
    }
    else{
      double coeff = ( 16 * M_PI * M_PI ) / ( nChan * _maxOrder * _maxOrder );
      gsl_blas_zdscal( coeff, weights );
    }
  }

  return;
}

const gsl_vector_complex* SphericalSpatialHWNCBeamformer::next( int frameX )
{
  if ( frameX == _frameX ) return _vector;

  unsigned chanX = 0;
  const gsl_vector_complex* XVec;   /* snapshot at each frequency */
  const gsl_vector_complex* weights; /* weights of the combining-unit */
  gsl_complex val;

  this->_allocImage();
  for ( _ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++ ) {
    const gsl_vector_complex* samp = ( *itr )->next( frameX );
    if ( true == ( *itr )->isEnd() ) _endOfSamples = true;
    _snapShotArray->newSample( samp, chanX );
    chanX++;
  }
  _snapShotArray->update();
  
  if ( _halfBandShift == false ) {
    // calculate a direct component.

    XVec    = _snapShotArray->getSnapShot(0);
    weights = _bfWeightV[0]->wq_f(0);
    gsl_blas_zdotc( weights, XVec, &val );
    gsl_vector_complex_set( _vector, 0, val );

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for ( unsigned fbinX = 1; fbinX <= _fftLen2; fbinX++ ) {
      XVec    = _snapShotArray->getSnapShot( fbinX );
      weights = _bfWeightV[0]->wq_f( fbinX );

      gsl_blas_zdotc( weights, XVec, &val );
      
      if( fbinX < _fftLen2 ){
	gsl_vector_complex_set( _vector, fbinX, val );
	gsl_vector_complex_set( _vector, _fftLen - fbinX, gsl_complex_conjugate(val) );
      }
      else
	gsl_vector_complex_set( _vector, _fftLen2, val );
    }
  }
  else{
    fprintf( stderr, "_halfBandShift == true is not implemented yet\n" );
    throw j_error( "_halfBandShift == true is not implemented yet\n" );
  }

  _increment();
  return _vector;
}

bool SphericalSpatialHWNCBeamformer::_allocSteeringUnit( int unitN )
{
  for ( unsigned unitX = 0; unitX < _bfWeightV.size(); unitX++ ) {
    if ( NULL != _bfWeightV[unitX] ) {
      delete _bfWeightV[unitX];
    }
    _bfWeightV[unitX] = NULL;
  }

  if ( _bfWeightV.size() != unitN )
    _bfWeightV.resize( unitN );
  
  for ( unsigned unitX = 0; unitX < unitN; unitX++ ) {
    _bfWeightV[unitX] = new beamformerWeights( _fftLen, (int)chanN(), _halfBandShift, _NC );
  }

  return true;
}

bool SphericalSpatialHWNCBeamformer::_calcSteeringUnit( int unitX, bool isGSC )
{
  gsl_vector_complex* weights;
  unsigned nChan = _channelList.size();

  if( unitX >= _bfWeightV.size() ){
    fprintf(stderr,"Invalid argument %d\n", unitX);
    return false;
  }

  // weights = _bfWeightV[unitX]->wq_f(0); 
  // calcDCWeights( _maxOrder, weights );

  // for(unsigned fbinX=1;fbinX<=_fftLen2;fbinX++){
  for ( unsigned fbinX = 0; fbinX <= _fftLen2; fbinX++ ) {
    // fprintf(stderr, "_calcWeights(%d)\n", fbinX);
    weights = _bfWeightV[unitX]->wq_f(fbinX); 
    _calcWeights( fbinX, weights );

    if( true == isGSC ){// calculate a blocking matrix for each frequency bin.
      _bfWeightV[unitX]->calcBlockingMatrix( fbinX );
    }
  }
  _bfWeightV[unitX]->setTimeAlignment();

  return true;
}
