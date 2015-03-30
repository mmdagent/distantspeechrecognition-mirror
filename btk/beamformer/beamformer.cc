//
//                       Beamforming Toolkit
//                              (btk)
//
//  Module:  btk.beamforming
//  Purpose: Beamforming in the subband domain.
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


#include "beamformer/beamformer.h"
#include <gsl/gsl_blas.h>
#include <gsl/gsl_sf_trig.h>
#include <matrix/blas1_c.H>
#include <matrix/linpack_c.H>
#include "postfilter/postfilter.h"

//double  sspeed = 343740.0;

// ----- members for class `SnapShotArray' -----
//
SnapShotArray::SnapShotArray(unsigned fftLn, unsigned nChn)
  : _fftLen(fftLn), _nChan(nChn)
{
  // printf("Snap Shot Array FFT Length = %d\n", _fftLen);
  // printf("Snap Shot Array N Channels = %d\n", _nChan);

  _specSamples  = new gsl_vector_complex*[_nChan];
  for (unsigned i = 0; i < _nChan; i++)
    _specSamples[i] = gsl_vector_complex_alloc(_fftLen);

  _specSnapShots = new gsl_vector_complex*[_fftLen];
  for (unsigned i = 0; i < _fftLen; i++)
    _specSnapShots[i] = gsl_vector_complex_alloc(_nChan);
}

SnapShotArray::~SnapShotArray()
{
  for (unsigned i = 0; i < _nChan; i++)
    gsl_vector_complex_free(_specSamples[i]);
  delete[] _specSamples;

  for (unsigned i = 0; i < _fftLen; i++)
    gsl_vector_complex_free(_specSnapShots[i]);
  delete[] _specSnapShots;
}

void SnapShotArray::zero()
{
  for (unsigned i = 0; i < _nChan; i++)
    gsl_vector_complex_set_zero(_specSamples[i]);

  for (unsigned i = 0; i < _fftLen; i++)
    gsl_vector_complex_set_zero(_specSnapShots[i]);
}

/**
   @brief set the snapshot at each chanell bin to the inner member
   @param const gsl_vector_complex* samp[in]
   @param unsigned chanX[in]
   @note after you set all the channel data, call update() method.
*/
void SnapShotArray::newSample(const gsl_vector_complex* samp, unsigned chanX) const
{
  assert(chanX < _nChan);
  gsl_vector_complex_memcpy(_specSamples[chanX], samp);
}

void SnapShotArray::update()
{
  for (unsigned ifft = 0; ifft < _fftLen; ifft++) {
    for (unsigned irow = 0; irow < _nChan; irow++) {
      gsl_complex rowVal = gsl_vector_complex_get(_specSamples[irow], ifft);
      gsl_vector_complex_set(_specSnapShots[ifft], irow, rowVal);
    }
  }
}

/**
   @brief set the snapshot at each frequnecy bin to the internal member.
   @param const gsl_vector_complex* snapshots[in]
   @param unsigned fbinX[in]
   @note if this method is used for setting all the frequency components, 
         don NOT call update() method.
*/
void SnapShotArray::newSnapShot(const gsl_vector_complex* snapshots, unsigned fbinX )
{
  unsigned fftLen2 = _fftLen/2;
  assert( fbinX <= fftLen2 );

  gsl_vector_complex_memcpy( _specSnapShots[fbinX], snapshots );
  if( fbinX == 0 || fbinX == fftLen2 )
    return;

  for(unsigned chanX=0;chanX<_nChan;chanX++){
    gsl_vector_complex_set( _specSnapShots[fftLen2-fbinX], chanX, 
			    gsl_complex_conjugate( gsl_vector_complex_get( snapshots, chanX ) ) );
  }
  return;
}

// ----- members for class `SpectralMatrixArray' -----
//
SpectralMatrixArray::SpectralMatrixArray(unsigned fftLn, unsigned nChn,
					 double forgetFact)
  : SnapShotArray(fftLn, nChn),
    _mu(gsl_complex_rect(forgetFact, 0))
{
  _specMatrices = new gsl_matrix_complex*[_fftLen];
  for (unsigned i = 0; i < _fftLen; i++)
    _specMatrices[i] = gsl_matrix_complex_alloc(_nChan, _nChan);
}

SpectralMatrixArray::~SpectralMatrixArray()
{
  for (unsigned i = 0; i < _fftLen; i++)
    gsl_matrix_complex_free(_specMatrices[i]);
  delete[] _specMatrices;
}

void SpectralMatrixArray::zero()
{
  SnapShotArray::zero();

  for (unsigned i = 0; i < _fftLen; i++)
    gsl_matrix_complex_set_zero(_specMatrices[i]);
}

void SpectralMatrixArray::update()
{
  SnapShotArray::update();

  for (unsigned ifft = 0; ifft < _fftLen; ifft++) {
    gsl_matrix_complex* smat = _specMatrices[ifft];
    gsl_matrix_complex_scale(smat, _mu);

    for (unsigned irow = 0; irow < _nChan; irow++) {
      gsl_complex rowVal = gsl_vector_complex_get(_specSnapShots[ifft], irow);
      for (unsigned icol = 0; icol < _nChan; icol++) {
	gsl_complex colVal = gsl_vector_complex_get(_specSnapShots[ifft], icol);
	gsl_complex newVal = gsl_complex_mul(rowVal, colVal);
	gsl_complex oldVal = gsl_matrix_complex_get(smat, irow, icol);
	gsl_complex alpha  = gsl_complex_sub( gsl_complex_rect( 1.0, 0 ), _mu );
	newVal =  gsl_complex_mul( alpha, newVal );
	gsl_matrix_complex_set(smat, irow, icol,
			       gsl_complex_add(oldVal, newVal));
      }
    }
  }
}

// ----- members for class `FBSpectralMatrixArray' -----
//
FBSpectralMatrixArray::FBSpectralMatrixArray(unsigned fftLn, unsigned nChn,
					     double forgetFact)
  : SpectralMatrixArray(fftLn, nChn, forgetFact) { }

FBSpectralMatrixArray::~FBSpectralMatrixArray()
{
}

void FBSpectralMatrixArray::update()
{
  SnapShotArray::update();

  for (unsigned ifft = 0; ifft < _fftLen; ifft++) {
    gsl_matrix_complex* smat = _specMatrices[ifft];
    gsl_matrix_complex_scale(smat, _mu);

    for (unsigned irow = 0; irow < _nChan; irow++) {
      gsl_complex rowVal = gsl_vector_complex_get(_specSnapShots[ifft], irow);
      for (unsigned icol = 0; icol < _nChan; icol++) {
	gsl_complex colVal = gsl_vector_complex_get(_specSamples[ifft], icol);
	gsl_complex newVal = gsl_complex_mul(rowVal, colVal);
	gsl_complex oldVal = gsl_matrix_complex_get(smat, irow, icol);
	gsl_matrix_complex_set(smat, irow, icol,
			       gsl_complex_add(oldVal, newVal));
      }
    }
  }
}

/**
   @brief calculate the inverse matrixof 2 x 2 matrix,
          inv( A + bI ) 
   @note It is fast because this function doesn't use a special mothod.
   @param mat[in/out]
 */
static void putInverseMat22( gsl_matrix_complex *mat, double beta = 0.01 )
{
  gsl_complex mat00, mat11, mat01, mat10, det, val;

  mat00 = gsl_matrix_complex_get( mat, 0, 0 );
  mat11 = gsl_matrix_complex_get( mat, 1, 1 );
  mat01 = gsl_matrix_complex_get( mat, 0, 1 );
  mat10 = gsl_matrix_complex_get( mat, 1, 0 );

#define USE_TH_INVERSION
#ifdef USE_TH_INVERSION
#define MINDET_THRESHOLD (1.0E-07)
  det = gsl_complex_sub( gsl_complex_mul( mat00, mat11 ), gsl_complex_mul( mat01, mat10 ) );
  if ( gsl_complex_abs( det ) < MINDET_THRESHOLD ){

    fprintf(stderr,"putInverseMat22:compensate for non-invertibility\n");
    mat00 = gsl_complex_add_real( mat00, beta );
    mat11 = gsl_complex_add_real( mat11, beta );
    det = gsl_complex_sub( gsl_complex_mul( mat00, mat11 ), gsl_complex_mul( mat01, mat10 ) );
  }
#else
  mat00 = gsl_complex_add_real( mat00, beta );
  mat11 = gsl_complex_add_real( mat11, beta );
  det = gsl_complex_sub( gsl_complex_mul( mat00, mat11 ), gsl_complex_mul( mat01, mat10 ) );
#endif

  val = gsl_complex_div( mat11, det );
  gsl_matrix_complex_set( mat, 0, 0, val );

  val = gsl_complex_div( mat00, det );
  gsl_matrix_complex_set( mat, 1, 1, val );

  val = gsl_complex_div( mat01, det );
  val = gsl_complex_mul_real(val, -1.0 );
  gsl_matrix_complex_set( mat, 0, 1, val );

  val = gsl_complex_div( mat10, det );
  val = gsl_complex_mul_real(val, -1.0 );
  gsl_matrix_complex_set( mat, 1, 0, val );

}

/**
   @brief calculate the Moore-Penrose pseudoinverse matrix

   @param gsl_matrix_complex *A[in] an input matrix. A[M][N] where M > N
   @param gsl_matrix_complex *invA[out] an pseudoinverse matrix of A
   @param float dThreshold[in]

   @note if M>N, invA * A = E. Otherwise, A * invA = E.
 */
bool pseudoinverse( gsl_matrix_complex *A, gsl_matrix_complex *invA, float dThreshold )
{
   size_t M = A->size1;
   size_t N = A->size2;
   int info;
   complex <float> a[M*N];
   complex <float> s[M+N];
   complex <float> e[M+N];
   complex <float> u[M*M];
   complex <float> v[N*N];
   size_t lda = M;
   size_t ldu = M;
   size_t ldv = N;
   size_t minN = (M<N)? M:N;
   bool ret = true;

   for ( size_t i = 0; i < M; i++ )
     for ( size_t j = 0; j < N; j++ ){
       gsl_complex gx = gsl_matrix_complex_get( A, i, j );
       a[i+j*lda] = complex <float>( GSL_REAL(gx), GSL_IMAG(gx) );
     }
 
   info = csvdc ( a, lda, M, N, s, e, u, ldu, v, ldv, 11 );
   if ( info != 0 ){
     cout << "\n";
     cout << "Warning:\n";
     cout << "  CSVDC returned nonzero INFO = " << info << "\n";
     //gsl_matrix_complex_set_identity( invA );
     ret = false;//return(true);
   }

   for (size_t k = 0; k <N; k++ ){
     if ( abs(s[k]) < dThreshold ){
       s[k] = complex<float>( 0.0, 0.0 );
       fprintf( stderr, "pseudoinverse: s[%d] = 0 because of %e < %e\n", k,  abs(s[k]), dThreshold  );
       ret = false;
     }
     else
       s[k] = complex<float>( 1.0, 0.0 ) / s[k];
   }

   for ( size_t i = 0; i < M; i++ ){
     for ( size_t j = 0; j < N; j++ ){
       complex <float> x( 0.0, 0.0 );
       for ( size_t k = 0; k < N/*minN*/; k++ ){
	 x = x + v[j+k*ldv] * s[k] * conj ( u[i+k*ldu] );
       }
       gsl_matrix_complex_set( invA, j, i, gsl_complex_rect( x.real(), x.imag() ) );
     }
   }
   
   return(ret);
}

/**
   @brief calculate a quiescent weight vector wq = C * v = C * inv( C_H * C ) * g. 
   @param gsl_vector_complex* wt[in/out] array manifold for the signal of a interest. wt[chanN]
   @param gsl_vector_complex** pWj[in] array manifolds for the interferences. pWj[NC-1][chanN]
   @param int chanN[in] the number of sensors
   @param int NC[in] the number of constraints. 
   @return 
 */
static bool calcNullBeamformer( gsl_vector_complex* wt, gsl_vector_complex** pWj, int chanN, int NC = 2 )
{
  gsl_matrix_complex* constraintMat;  // the constraint matrix
  gsl_matrix_complex* constraintMat1; // the copy of constraintMat
  gsl_vector_complex* g; // [NC] the gain vector which has constant elements
  gsl_complex val;
  gsl_complex alpha;
  gsl_matrix_complex* invMat;
  gsl_vector_complex* v;

  constraintMat  = gsl_matrix_complex_alloc( chanN, NC );
  constraintMat1 = gsl_matrix_complex_alloc( chanN, NC );
  invMat = gsl_matrix_complex_alloc( NC, NC );
  if( NULL == constraintMat ||  NULL == constraintMat1  || NULL == invMat ){
    fprintf(stderr,"gsl_matrix_complex_alloc failed\n");
    return false;
  }

  for (int i = 0; i < chanN; i++){
    gsl_matrix_complex_set( constraintMat, i, 0, gsl_vector_complex_get( wt, i ) );
    for(int j = 1; j < NC; j++)
      gsl_matrix_complex_set( constraintMat, i, j, gsl_vector_complex_get( pWj[j-1], i ) );
  }
  gsl_matrix_complex_memcpy( constraintMat1, constraintMat );

  g = gsl_vector_complex_alloc( NC );
  v = gsl_vector_complex_alloc( NC );
  if( NULL == g || NULL == v ){
    gsl_matrix_complex_free( constraintMat );
    gsl_matrix_complex_free( constraintMat1 );
    gsl_matrix_complex_free( invMat );
    fprintf(stderr,"gsl_vector_complex_alloc failed\n");
    return false;
  }

  gsl_vector_complex_set( g, 0, gsl_complex_rect( 1.0, 0.0 ) );
  for(int j = 1; j < NC; j++)
    gsl_vector_complex_set( g, j, gsl_complex_rect( 0.0, 0.0 ) );
  
  GSL_SET_COMPLEX( &alpha, 1.0, 0.0 );
  // calculate C_H * C 
  gsl_matrix_complex_set_zero( invMat );
  gsl_blas_zgemm( CblasConjTrans, CblasNoTrans, alpha, constraintMat, constraintMat1, alpha, invMat );
  // calculate inv( C_H * C ) 
  if( 2!=NC ){
    // write a code which calculates a NxN inverse matrix.
    pseudoinverse( invMat, invMat );
  }
  else{
    putInverseMat22( invMat );
  }
  // calculate inv( C_H * C ) * g
  gsl_vector_complex_set_zero( v );
  gsl_blas_zgemv( CblasNoTrans, alpha, invMat, g, alpha, v );
  // calculate C * v = C * inv( C_H * C ) *g
  gsl_vector_complex_set_zero( wt );
  gsl_blas_zgemv( CblasNoTrans, alpha, constraintMat1, v, alpha, wt );
  
  gsl_matrix_complex_free( constraintMat );
  gsl_matrix_complex_free( constraintMat1 );
  gsl_matrix_complex_free( invMat );
  gsl_vector_complex_free( g );
  gsl_vector_complex_free( v );

  return true;

 ERROR_calcNullBeamformer:
  gsl_matrix_complex_free( constraintMat );
  gsl_matrix_complex_free( constraintMat1 );
  gsl_matrix_complex_free( invMat );
  gsl_vector_complex_free( g );
  gsl_vector_complex_free( v );
  return false;
}

/**
   @brief Calculate the blocking matrix for a distortionless beamformer and return its Hermitian transpose.      
   @param  gsl_vector_complex* arrayManifold[in] the fixed weights in an upper branch
   @param  int NC[in] the number of constraints in an upper branch
   @param  gsl_matrix_complex* blockMatA[out]
   @return success -> true, error -> false
   @note This code was transported from subbandBeamforming.py. 
 */
static bool _calcBlockingMatrix( gsl_vector_complex* arrayManifold, int NC, gsl_matrix_complex* blockMat )
{
  gsl_matrix_complex* PcPerp;
  gsl_vector_complex *vec, *rvec, *conj1;
  int vsize    = arrayManifold->size;
  int bsize    = vsize - NC;

  if( bsize <= 0 ){
    fprintf(stderr,"The number of sensors %d > the number of constraints %d\n",vsize,NC);
    return false;
  }

  gsl_matrix_complex_set_zero( blockMat );

  PcPerp = gsl_matrix_complex_alloc( vsize, vsize );
  if( NULL == PcPerp ){
    fprintf(stderr,"gsl_matrix_complex_alloc failed\n");
    return false;
  }

  vec = gsl_vector_complex_alloc( vsize );
  rvec = gsl_vector_complex_alloc( vsize );
  conj1 = gsl_vector_complex_alloc( vsize );
  if( NULL==vec || NULL == rvec || NULL==conj1 ){
    fprintf(stderr,"gsl_vector_complex_alloc\n");
    return false;
  }

  double norm_vs = gsl_blas_dznrm2( arrayManifold );
  norm_vs = norm_vs * norm_vs; // note ...
  gsl_complex alpha;
  GSL_SET_COMPLEX ( &alpha, -1.0/norm_vs, 0 );

  gsl_matrix_complex_set_identity( PcPerp );
  for(int i = 0; i < vsize; i++)
    gsl_vector_complex_set( conj1, i, gsl_complex_conjugate( gsl_vector_complex_get( arrayManifold, i ) ) );
  gsl_blas_zgeru( alpha, conj1, arrayManifold, PcPerp );

  for(int idim=0;idim<bsize;idim++){
    double norm_vec;

    gsl_matrix_complex_get_col( vec, PcPerp, idim );
    for(int jdim=0;jdim<idim;jdim++){
      gsl_complex ip;

      gsl_matrix_complex_get_col( rvec, blockMat, jdim );
      for (int j = 0; j < vsize; j++){
	gsl_complex out;
	out = gsl_vector_complex_get(rvec, j);
      }
      gsl_blas_zdotc( rvec, vec, &ip );
      ip = gsl_complex_mul_real( ip, -1.0 );
      gsl_blas_zaxpy( ip, rvec, vec );
    }
    // I cannot understand why the calculation of the normalization coefficient norm_vec is different from that of norm_vs.
    // But I imitated the original python code faithfully.
    norm_vec = gsl_blas_dznrm2( vec );
    gsl_blas_zdscal ( 1.0 / norm_vec, vec );
    gsl_matrix_complex_set_col(blockMat, idim, vec );
  }

  gsl_matrix_complex_free( PcPerp );
  gsl_vector_complex_free( vec );
  gsl_vector_complex_free( rvec );
  gsl_vector_complex_free( conj1 );

#ifdef _MYDEBUG_
  // conjugate the blocking matrix
  for (int i = 0; i < vsize; i++){
    for (int j = 0; j < bsize; j++){
      gsl_complex val;

      val = gsl_matrix_complex_get( blockMat, i, j );
      gsl_matrix_complex_set( blockMat, i, j,  gsl_complex_conjugate( val ) );
      //printf ("%e %e, ",  GSL_REAL(val), GSL_IMAG(val) );
    }
    //printf("\n");
  }
#endif

  return true;
}

 /**
   @brief a wrapper function for _calcBlockingMatrix( gsl_vector_complex* , int, gsl_matrix_complex* )
   @note  you have to free the returned pointer afterwards. 
   @param  gsl_vector_complex* arrayManifold[in] the fixed weights in an upper branch
   @param  int NC[in] the number of constraints in an upper branch
   @return the pointer to a blocking matrix
 */
static gsl_matrix_complex* getBlockingMatrix( gsl_vector_complex* arrayManifold, int NC )
{
  int vsize    = arrayManifold->size;
  int bsize    = vsize - NC;
  gsl_matrix_complex* blockMat;

  blockMat = gsl_matrix_complex_alloc( vsize, bsize );
  if( NULL == blockMat ){
    fprintf(stderr,"gsl_matrix_complex_alloc failed\n");
    return NULL;
  }

  if( false==_calcBlockingMatrix( arrayManifold, NC, blockMat ) ){
    fprintf(stderr,"getBlockingMatrix() failed\n");
    return NULL;
  }
  return blockMat;
}

// ----- members for class `beamformerWeights' -----
//

beamformerWeights::beamformerWeights( unsigned fftLen, unsigned chanN, bool halfBandShift, unsigned NC )
{
  _fftLen  = fftLen;
  _chanN  = chanN;
  _halfBandShift = halfBandShift;
  _NC     = NC;

  this->_allocWeights();
}

beamformerWeights::~beamformerWeights()
{
  this->_freeWeights();
}

/**
   @brief calculate array manifold vectors for the delay & sum beamformer.
   @param double sampleRate[in]
   @param const gsl_vector* delays[in] a vector whose element indicates time delay. delaysT[chanN]
   @param bool isGSC[in] if it's 'true', blocking matrices will be calculated.
 */
void beamformerWeights::calcMainlobe( double sampleRate, const gsl_vector* delays, bool isGSC )
{
  if (delays->size != _chanN )
    throw jdimension_error("Number of delays does not match number of channels (%d vs. %d).\n",
			   delays->size, _chanN );
  if( true == isGSC && _chanN <=1 ){
    throw jdimension_error("The number of channels must be > 1 but it is %d\n",
			   _chanN );
  }
  if( _wq == NULL ) this->_allocWeights();

  unsigned fftLen2 = _fftLen / 2;

  if ( _halfBandShift==true ) {
    float fshift = 0.5;

    for (unsigned fbinX = 0; fbinX < fftLen2; fbinX++) {
      gsl_vector_complex* vec     = _wq[fbinX];
      gsl_vector_complex* vecConj = _wq[_fftLen - 1 - fbinX];
      for (unsigned chanX = 0; chanX < _chanN; chanX++) {
	double val = -2.0 * M_PI * (fshift+fbinX) * sampleRate * gsl_vector_get(delays, chanX) / _fftLen;
	gsl_vector_complex_set(vec,     chanX, gsl_complex_div_real( gsl_complex_polar(1.0,  val), _chanN ) );
	gsl_vector_complex_set(vecConj, chanX, gsl_complex_div_real( gsl_complex_polar(1.0, -val), _chanN ) );
      }
    }

  } else {
    gsl_vector_complex* vec;
    gsl_vector_complex* vecConj;

    // calculate weights of a direct component.
    vec     = _wq[0];
    for (unsigned chanX = 0; chanX < _chanN; chanX++) 
      gsl_vector_complex_set(vec,     chanX, gsl_complex_div_real( gsl_complex_polar(1.0,  0.0), _chanN ) );    
    // calculate weights from FFT bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX < fftLen2; fbinX++) {
      vec     = _wq[fbinX];
      vecConj = _wq[_fftLen - fbinX];
      for (unsigned chanX = 0; chanX < _chanN; chanX++) {
	double val = -2.0 * M_PI * fbinX * gsl_vector_get(delays, chanX) * sampleRate / _fftLen;
	gsl_vector_complex_set(vec,     chanX, gsl_complex_div_real( gsl_complex_polar(1.0,  val), _chanN ) );
	gsl_vector_complex_set(vecConj, chanX, gsl_complex_div_real( gsl_complex_polar(1.0, -val), _chanN ) );
      }
    }
    // for exp(-j*pi)
    vec     = _wq[fftLen2];
    for (unsigned chanX = 0; chanX < _chanN; chanX++) {
      double val = -M_PI * sampleRate * gsl_vector_get(delays, chanX);
      gsl_vector_complex_set(vec, chanX, gsl_complex_div_real( gsl_complex_polar(1.0,  val), _chanN ) );
    }
  }

  this->setTimeAlignment();

  // calculate a blocking matrix for each frequency bin.
  if( true == isGSC ){  
    for (unsigned fbinX = 0; fbinX < _fftLen; fbinX++) {
      if( false==_calcBlockingMatrix(  _wq[fbinX], 1, _B[fbinX] ) ){
	throw j_error("_calcBlockingMatrix() failed\n");
      }
    }
  }

}

/**
   @brief you can constrain a beamformer to preserve a target signal and, at the same time, to suppress an interference signal.
   @param double sampleRate[in]
   @param const gsl_vector*  delaysT[in] a time delay vector for a target signal. delaysT[chanN]
   @param const gsl_vector** delaysJ[in] a time delay vector for an interference. delaysI[chanN]
 */
void beamformerWeights::calcMainlobe2( double sampleRate, const gsl_vector* delaysT, const gsl_vector* delaysI, bool isGSC  )
{
  if( delaysI->size != _chanN )
    throw jdimension_error("The number of delays for an interference signal does not match number of channels (%d vs. %d).\n",
			   delaysI->size, _chanN );
  if( _chanN < 2 ){
    fprintf(stderr,"The number of channels must be > 2 but it is %d\n",
	    _chanN );
    throw jdimension_error("The number of channels must be > 2 but it is %d\n",
			   _chanN );
  }

  gsl_matrix* delaysIs = gsl_matrix_alloc( 1 /* = 2 - 1 */, _chanN ); // the number of interferences = 1
  
  if( NULL == delaysIs )
    throw jallocation_error("gsl_matrix_complex_alloc failed\n");
  gsl_matrix_set_row( delaysIs, 0, delaysI );
  this->calcMainlobeN( sampleRate, delaysT, delaysIs, 2, isGSC );

  gsl_matrix_free( delaysIs );
}

/**
   @brief put multiple constraints to preserve a target signal and, at the same time, to suppress interference signals.
   @param double sampleRate[in]
   @param const gsl_vector*  delaysT[in] delaysT[chanN]
   @param const gsl_vector** delaysJ[in] delaysJ[NC-1][chanN]
   @param int NC[in] the number of constraints (= the number of target and interference signals )
 */
void beamformerWeights::calcMainlobeN( double sampleRate, const gsl_vector* delaysT, const gsl_matrix* delaysIs, unsigned NC, bool isGSC  )
{
  if( NC < 2  || NC > _chanN )
    throw jdimension_error("1 < the number of constraints %d <= the number of sensors %d.\n",
			   NC, _chanN );

  if( delaysT->size != _chanN )
    throw jdimension_error("The number of delays does not match number of channels (%d vs. %d).\n",
			   delaysT->size, _chanN );

  unsigned fftLen2 = _fftLen / 2;
  gsl_vector_complex** pWj      = new gsl_vector_complex*[NC-1];
  gsl_vector_complex** pWjConj  = new gsl_vector_complex*[NC-1];
  for(int n=0;n<NC-1;n++){
     pWj[n]     = gsl_vector_complex_alloc( _chanN );
     pWjConj[n] = gsl_vector_complex_alloc( _chanN );
  }

#ifdef _MYDEBUG_
  for (int n = 0; n < NC-1; n++){
    printf("delay %d\n",n);
    for (unsigned chanX = 0; chanX < chanN(); chanX++ ){
      float val = gsl_matrix_get(delaysJ, n, chanX);
      printf ("%e, ",  (val) );
    }
    printf("\n");
    fflush(stdout);
  }
#endif

  // set values to _wq[].
  this->calcMainlobe( sampleRate, delaysT, false );

  if ( _halfBandShift==true ) {
    float fshift = 0.5;

    for (unsigned fbinX = 0; fbinX < fftLen2; fbinX++) {
      gsl_vector_complex* vec     = _wq[fbinX];
      gsl_vector_complex* vecConj = _wq[_fftLen - 1 - fbinX];
      for (unsigned chanX = 0; chanX < _chanN; chanX++) {
	gsl_complex wq_f  = gsl_vector_complex_get( vec,     chanX );
	gsl_complex wqc_f = gsl_vector_complex_get( vecConj, chanX );
	gsl_vector_complex_set( vec,     chanX, gsl_complex_mul_real( wq_f,  _chanN ) );
	gsl_vector_complex_set( vecConj, chanX, gsl_complex_mul_real( wqc_f, _chanN ) );
	for(int n=0;n<NC-1;n++){
	  double valJ = -2.0 * M_PI * (fshift+fbinX) * sampleRate * gsl_matrix_get(delaysIs, n, chanX) / _fftLen;
	  gsl_vector_complex_set(pWj[n],     chanX, gsl_complex_polar(1.0,  valJ) );
	  gsl_vector_complex_set(pWjConj[n], chanX, gsl_complex_polar(1.0, -valJ) );
	}
      }
      if( false==calcNullBeamformer( vec, pWj, _chanN, NC ) ){
	throw j_error("calcNullBeamformer() failed\n");
      }
      if( false==calcNullBeamformer( vecConj, pWjConj, _chanN, NC ) ){
	throw j_error("calcNullBeamformer() failed\n");
      }
    }
    
  } else {
    gsl_vector_complex* vec;
    gsl_vector_complex* vecConj;

    // calculate weights of a direct component
    vec = _wq[0];
    for (unsigned chanX = 0; chanX < _chanN; chanX++)
      gsl_vector_complex_set( vec, chanX, gsl_complex_rect( 1.0/_chanN, 0.0 ) );
    
    // use the property of the symmetry : wq[1] = wq[fftLen-1]*, wq[2] = wq[fftLen-2]*,...
    for (unsigned fbinX = 1; fbinX < fftLen2; fbinX++) { 
      vec     = _wq[fbinX];
      vecConj = _wq[_fftLen - fbinX];
      for (unsigned chanX = 0; chanX < _chanN; chanX++) {
	gsl_complex wq_f  = gsl_vector_complex_get( vec,     chanX );
	//gsl_complex wqc_f = gsl_vector_complex_get( vecConj, chanX );
	gsl_vector_complex_set( vec,     chanX, gsl_complex_mul_real( wq_f,  _chanN ) );
	//gsl_vector_complex_set( vecConj, chanX, gsl_complex_mul_real( wqc_f, _chanN ) );
 	for(int n=0;n<NC-1;n++){
	  double valJ = -2.0 * M_PI * fbinX * sampleRate * gsl_matrix_get(delaysIs, n, chanX) / _fftLen;
	  gsl_vector_complex_set(pWj[n],     chanX, gsl_complex_polar(1.0,  valJ) );
	  //gsl_vector_complex_set(pWjConj[n], chanX, gsl_complex_polar(1.0, -valJ) );
	}
      }
      if( false==calcNullBeamformer( vec, pWj, _chanN, NC ) ){
	throw j_error("calcNullBeamformer() failed\n");
      }
      //if( false==calcNullBeamformer( vecConj, pWjConj, _chanN, NC ) ){
      //throw j_error("calcNullBeamformer() failed\n");
      //}
    }

    // for exp(-j*pi)
    vec     = _wq[fftLen2];
    for (unsigned chanX = 0; chanX < _chanN; chanX++) {
      gsl_complex wq_f  = gsl_vector_complex_get( vec, chanX );
      gsl_vector_complex_set( vec, chanX, gsl_complex_mul_real( wq_f, _chanN ) );
      for(int n=0;n<NC-1;n++){
	double val = -M_PI * sampleRate *  gsl_matrix_get(delaysIs, n, chanX);
	gsl_vector_complex_set(vec, chanX, gsl_complex_div_real( gsl_complex_polar(1.0,  val), _chanN ) );
      }
      if( false==calcNullBeamformer( vec, pWj, _chanN, NC ) ){
	throw j_error("calcNullBeamformer() failed\n");
      }
    }

  }

  // calculate a blocking matrix for each frequency bin.
  if( true == isGSC ){
    for (unsigned fbinX = 0; fbinX < _fftLen; fbinX++) {
      if( false==_calcBlockingMatrix( _wq[fbinX], NC, _B[fbinX] ) ){
	throw j_error("_calcBlockingMatrix() failed\n");
      }
    }
  }

  for(int n=0;n<NC-1;n++){
    gsl_vector_complex_free( pWj[n] );
    gsl_vector_complex_free( pWjConj[n] );
  }
  delete [] pWj;
  delete [] pWjConj;

}

/**
   @brief set an active weight vector for frequency bin 'fbinX' and calculate B[fbinX] * wa[fbinX].

   @param int fbinX[in] a frequency bin.
   @param const gsl_vector* packedWeight[in] a packed weight vector.
 */
void beamformerWeights::calcSidelobeCancellerP_f( unsigned fbinX, const gsl_vector* packedWeight )
{

  if( packedWeight->size != ( 2 * ( _chanN - _NC ) ) )
    throw jdimension_error("the size of an active weight vector must be %d but it is %d\n",
			   ( 2 * ( _chanN - _NC ) ), packedWeight->size );

  if( fbinX >= _fftLen )
    throw jdimension_error("Must be a frequency bin %d < the length of FFT %d\n",
			   fbinX, _fftLen );

  // copy an active weight vector to a member.
  gsl_complex val;

  for (unsigned chanX = 0; chanX < _chanN - _NC ; chanX++) {
    GSL_SET_COMPLEX( &val, gsl_vector_get( packedWeight, 2*chanX ), gsl_vector_get( packedWeight, 2*chanX+1 ) );
    gsl_vector_complex_set( _wa[fbinX], chanX, val );
  }

  // calculate B[f] * wa[f]
  // gsl_vector_complex_set_zero( _wl[fbinX] );
  gsl_blas_zgemv( CblasNoTrans, gsl_complex_rect(1.0,0.0), _B[fbinX], _wa[fbinX], gsl_complex_rect(0.0,0.0), _wl[fbinX] );
}

void beamformerWeights::calcSidelobeCancellerU_f( unsigned fbinX, const gsl_vector_complex* wa )
{
  if( fbinX >= _fftLen )
    throw jdimension_error("Must be a frequency bin %d < the length of FFT %d\n",
			   fbinX, _fftLen );

  // copy an active weight vector to a member.
  for (unsigned chanX = 0; chanX < _chanN - _NC ; chanX++) {
    gsl_vector_complex_set( _wa[fbinX], chanX, gsl_vector_complex_get( wa, chanX  ) );
  }
  
  // calculate B[f] * wa[f]
  // gsl_vector_complex_set_zero( _wl[fbinX] );
  gsl_blas_zgemv( CblasNoTrans, gsl_complex_rect(1.0,0.0), _B[fbinX], _wa[fbinX], gsl_complex_rect(0.0,0.0), _wl[fbinX] );
}

/**
   @brief multiply each weight of a beamformer with exp(j*k*pi), transform them back into time domain and write the coefficients multiplied with a window function.

   @param const String& fn[in]
   @param unsigned winType[in]
 */
bool beamformerWeights::writeFIRCoeff( const String& fn, unsigned winType )
{
  if( _wq == NULL || NULL == _wl ){
    printf("beamformerWeights::writeFIRCoeff :\n");
    return false;
  }

  unsigned fftLen2 = _fftLen / 2;
  FILE* fp = fileOpen(fn.c_str(), "w");

  if( NULL==fp ){
    printf("could not open %s\n",fn.c_str());
    return false;
  }

  fprintf(fp,"%d %d\n", _chanN, _fftLen );
  gsl_vector *window = getWindow( winType, _fftLen );
  double *weights_n = new double[_fftLen*2];

  if( false== _halfBandShift ){

    for( unsigned chanX = 0 ; chanX < _chanN ; chanX++ ){
      for( unsigned fbinX = 0 ; fbinX <= fftLen2 ; fbinX++ ){
	// calculate wq(f) - B(f) wa(f) at each channel
	gsl_complex wq_fn = gsl_vector_complex_get( _wq[fbinX], chanX );
	gsl_complex wl_fn = gsl_vector_complex_get( _wl[fbinX], chanX );
	gsl_complex wH_fn = gsl_complex_conjugate( gsl_complex_sub( wq_fn, wl_fn ) ) ;
	gsl_complex H_f   = gsl_complex_polar(1.0,  M_PI * (fbinX+1) );     
	gsl_complex val   = gsl_complex_mul ( H_f, wH_fn ); // shift fftLen/2

	weights_n[2*fbinX]   = GSL_REAL( val );
	weights_n[2*fbinX+1] = GSL_IMAG( val );
	if( fbinX > 0 && fbinX < fftLen2 ){
	  weights_n[2*(_fftLen-fbinX)]   = GSL_REAL( gsl_complex_conjugate(val) );
	  weights_n[2*(_fftLen-fbinX)+1] = GSL_IMAG( gsl_complex_conjugate(val) );
	}
      }
      gsl_fft_complex_radix2_inverse( weights_n, /* stride= */ 1, _fftLen );
      for( unsigned fbinX = 0 ; fbinX < _fftLen ; fbinX++ ){// multiply a window
	double coeff = gsl_vector_get(window,fbinX)* weights_n[2*fbinX];
	fprintf(fp,"%e ",coeff);
      }
      fprintf(fp,"\n");

    }//for( unsigned chanX = 0 ; chanX < _chanN ; chanX++ ){

  }

  gsl_vector_free( window );
  delete [] weights_n;
  fclose(fp);

  return(true);
}

void beamformerWeights::setQuiescentVector( unsigned fbinX, gsl_vector_complex *wq_f, bool isGSC )
{
  for(unsigned chanX=0;chanX<_chanN;chanX++)
    gsl_vector_complex_set( _wq[fbinX], chanX, gsl_vector_complex_get( wq_f, chanX ) );
  
  if( true == isGSC ){// calculate a blocking matrix for each frequency bin.
      if( false==_calcBlockingMatrix(  _wq[fbinX], _NC, _B[fbinX] ) ){
	throw j_error("_calcBlockingMatrix() failed\n");
      }
  }
}

void beamformerWeights::setQuiescentVectorAll( gsl_complex z, bool isGSC )
{
  for (unsigned fbinX = 0; fbinX < _fftLen; fbinX++){
    gsl_vector_complex_set_all( _wq[fbinX], z );
      if( true == isGSC ){// calculate a blocking matrix for each frequency bin.
	if( false==_calcBlockingMatrix(  _wq[fbinX], _NC, _B[fbinX] ) ){
	  throw j_error("_calcBlockingMatrix() failed\n");
	}
      }
  }
}

/**
   @brief calculate the blocking matrix at the frequency bin and 
          set results to the internal member _B[].
          The quiescent vector has to be provided
   @param unsigned fbinX[in]
 */
void beamformerWeights::calcBlockingMatrix( unsigned fbinX )
{
  if( false==_calcBlockingMatrix( _wq[fbinX], _NC, _B[fbinX] ) ){
    throw j_error("_calcBlockingMatrix() failed\n");
  }
}

void beamformerWeights::_allocWeights()
{
  _wq = new gsl_vector_complex*[_fftLen];
  _wa = new gsl_vector_complex*[_fftLen];
  _wl = new gsl_vector_complex*[_fftLen];
  _ta = new gsl_vector_complex*[_fftLen];
  _B  = new gsl_matrix_complex*[_fftLen];
  _CSDs  = new gsl_vector_complex*[_fftLen];
  _wp1   = gsl_vector_complex_alloc( _fftLen );

  for (unsigned i = 0; i < _fftLen; i++){
    _wq[i] = gsl_vector_complex_alloc( _chanN );
    _wl[i] = gsl_vector_complex_alloc( _chanN );
    _ta[i] = gsl_vector_complex_alloc( _chanN );
    _CSDs[i] = gsl_vector_complex_alloc( _chanN * _chanN );
    if( NULL == _wq[i] || NULL == _wl[i] || NULL == _ta[i] || NULL == _CSDs[i] )
      throw jallocation_error("gsl_matrix_complex_alloc failed\n");
    gsl_vector_complex_set_zero( _wq[i] );
    gsl_vector_complex_set_zero( _wl[i] );
    gsl_vector_complex_set_zero( _ta[i] );
    gsl_vector_complex_set_zero( _CSDs[i] );

    if( _chanN == 1 || _chanN == _NC ){
      _B[i]  = NULL;
      _wa[i] = NULL;
    }
    else{
      _B[i]  = gsl_matrix_complex_alloc( _chanN, _chanN - _NC );
      _wa[i] = gsl_vector_complex_alloc( _chanN - _NC );
      if( NULL == _wa[i] || NULL == _B[i] )
	throw jallocation_error("gsl_matrix_complex_alloc failed\n");
      gsl_matrix_complex_set_zero( _B[i]  );
      gsl_vector_complex_set_zero( _wa[i] );
    }

  }
  gsl_vector_complex_set_zero( _wp1 );
}

void beamformerWeights::_freeWeights()
{
  if( NULL != _wq ){
    for(unsigned fbinX=0;fbinX<_fftLen;fbinX++)
      gsl_vector_complex_free( _wq[fbinX] );
    delete [] _wq;
    _wq = NULL;
  }

  if( NULL != _wa ){
    for(unsigned fbinX=0;fbinX<_fftLen;fbinX++){
      if( NULL != _wa[fbinX] )
	gsl_vector_complex_free( _wa[fbinX] );
    }
    delete [] _wa;
    _wa = NULL;
  }

  if( NULL != _B ){
    for(unsigned fbinX=0;fbinX<_fftLen;fbinX++){
      if( NULL != _B[fbinX] )
	gsl_matrix_complex_free( _B[fbinX] );
    }
    delete [] _B;
     _B = NULL;
  }

  if( NULL != _wl ){
    for(unsigned fbinX=0;fbinX<_fftLen;fbinX++)
      gsl_vector_complex_free( _wl[fbinX] );
    delete [] _wl;
    _wl = NULL;
  }

  if( NULL != _ta ){    
    for(unsigned fbinX=0;fbinX<_fftLen;fbinX++)
      gsl_vector_complex_free( _ta[fbinX] );
    delete [] _ta;
    _ta = NULL;
  }

  if( NULL!=_CSDs ){
    for (unsigned i = 0; i < _fftLen; i++)
      gsl_vector_complex_free(_CSDs[i]);
    delete [] _CSDs;
    _CSDs = NULL;
  }

  if( NULL!=_wp1 ){
    gsl_vector_complex_free(_wp1);
    _wp1 = NULL;
  }
}

void beamformerWeights::setTimeAlignment()
{
  for (unsigned i = 0; i < _fftLen; i++){
    gsl_vector_complex_memcpy( _ta[i], _wq[i] );
  }
}


// ----- members for class `SubbandBeamformer' -----
//
SubbandBeamformer::SubbandBeamformer(unsigned fftLen, bool halfBandShift, const String& nm)
  : VectorComplexFeatureStream(fftLen, nm),
    _fftLen(fftLen),
    _halfBandShift(halfBandShift),
    _snapShotArray(NULL)
{ 
  _fftLen2 = _fftLen/2;
}

SubbandBeamformer::~SubbandBeamformer()
{
  if(  (int)_channelList.size() > 0 )
    _channelList.erase( _channelList.begin(), _channelList.end() );
}

void SubbandBeamformer::setChannel(VectorComplexFeatureStreamPtr& chan)
{
  _channelList.push_back(chan);
}

void SubbandBeamformer::clearChannel()
{
  //fprintf(stderr,"void SubbandDS::clearChannel()\n");
  
  if(  (int)_channelList.size() > 0 )
    _channelList.clear();
  _snapShotArray = NULL;
}

const gsl_vector_complex* SubbandBeamformer::next(int frameX)
{
  if (frameX == _frameX) return _vector;
  _increment();
  return _vector;
}

void SubbandBeamformer::reset()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();

  if ( _snapShotArray != NULL )
    _snapShotArray->zero();

  VectorComplexFeatureStream::reset();
  _endOfSamples = false;
}

// ----- members for class `SubbandDS' -----
//
SubbandDS::SubbandDS(unsigned fftLen, bool halfBandShift, const String& nm)
  : SubbandBeamformer( fftLen, halfBandShift, nm )
{ 
  _bfWeightV.clear();
}


SubbandDS::~SubbandDS()
{
  for(unsigned i=0;i<_bfWeightV.size();i++)
    delete _bfWeightV[i];

  //fprintf(stderr,"deleting...\n");
  //if( _snapShotArray != NULL )
  //delete _snapShotArray;
  //fprintf(stderr,"deletedn");
}

void SubbandDS::clearChannel()
{
  //fprintf(stderr,"void SubbandDS::clearChannel()\n");
  
  if(  (int)_channelList.size() > 0 )
    _channelList.clear();
  for(unsigned i=0;i<_bfWeightV.size();i++)
    delete _bfWeightV[i];
  _bfWeightV.clear();
  _snapShotArray = NULL;
}

/**
   @brief calculate an array manifold vectors for the delay & sum beamformer.
   @param double sampleRate[in]
   @param const gsl_vector* delays[in] delaysT[chanN]
 */
void SubbandDS::calcArrayManifoldVectors( double sampleRate, const gsl_vector* delays )
{
  this->_allocBFWeight( 1, 1 );
  _bfWeightV[0]->calcMainlobe( sampleRate, delays, false );
}

/**
   @brief you can put 2 constraints. You can constrain a beamformer to preserve a target signal and, at the same time, to suppress an interference signal.
   @param double sampleRate[in]
   @param const gsl_vector*  delaysT[in] delaysT[chanN]
   @param const gsl_vector** delaysJ[in] delaysJ[chanN]
 */
void SubbandDS::calcArrayManifoldVectors2(double sampleRate, const gsl_vector* delaysT, const gsl_vector* delaysJ )
{
  this->_allocBFWeight( 1, 2 );
  _bfWeightV[0]->calcMainlobe2( sampleRate, delaysT, delaysJ, false );
}

/**
   @brief you can put multiple constraints. For example, you can constrain a beamformer to preserve a target signal and, at the same time, to suppress interference signals.
   @param double sampleRate[in]
   @param const gsl_vector* delaysT[in] delaysT[chanN]
   @param const gsl_matrix* delaysJ[in] delaysJ[NC-1][chanN]
   @param int NC[in] the number of constraints (= the number of target and interference signals )
 */
void SubbandDS::calcArrayManifoldVectorsN(double sampleRate, const gsl_vector* delaysT, const gsl_matrix* delaysJ, unsigned NC )
{
  this->_allocBFWeight( 1, NC );
  _bfWeightV[0]->calcMainlobeN( sampleRate, delaysT, delaysJ, NC, false );
}

void SubbandDS::_allocImage()
{
  if( NULL == _snapShotArray )
    _snapShotArray = new SnapShotArray( _fftLen, chanN() );
}

void SubbandDS::_allocBFWeight( int nSource, int NC )
{
  for(unsigned i=0;i<_bfWeightV.size();i++){
    delete _bfWeightV[i];
  }
  _bfWeightV.resize(nSource);
  for(unsigned i=0;i<_bfWeightV.size();i++){
    _bfWeightV[i] = new beamformerWeights( _fftLen, chanN(), _halfBandShift, NC );
  }
  this->_allocImage();
}

#define MINFRAMES 0 // the number of frames for estimating CSDs.
const gsl_vector_complex* SubbandDS::next(int frameX)
{
  if (frameX == _frameX) return _vector;
  if( 0 == _bfWeightV.size() ){
    fprintf(stderr,"call calcArrayManifoldVectorsX() once\n");
    throw j_error("call calcArrayManifoldVectorsX() once\n");
  }
  
  unsigned chanX = 0;
  const gsl_vector_complex* snapShot_f;
  const gsl_vector_complex* arrayManifold_f;
  gsl_complex val;
  unsigned fftLen = _fftLen;

  this->_allocImage();
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frameX);
    if( true==(*itr)->isEnd() ) _endOfSamples = true;
    _snapShotArray->newSample( samp, chanX);  chanX++;
  }
  _snapShotArray->update();

  if( _halfBandShift == true ){
    for (unsigned fbinX = 0; fbinX < fftLen; fbinX++) {
      snapShot_f      = _snapShotArray->getSnapShot(fbinX);
      arrayManifold_f = _bfWeightV[0]->wq_f(fbinX);    
      gsl_blas_zdotc(arrayManifold_f, snapShot_f, &val);
      gsl_vector_complex_set(_vector, fbinX, val);
#ifdef  _MYDEBUG_
      if ( fbinX % 100 == 0 ){
	fprintf(stderr,"fbinX %d\n",fbinX );
	for (unsigned chX = 0; chX < chanN(); chX++) 
	  fprintf(stderr,"%f %f\n",GSL_REAL(  gsl_vector_complex_get( arrayManifold_f, chX ) ), GSL_IMAG(  gsl_vector_complex_get( arrayManifold_f, chX ) ) );
	fprintf(stderr,"VAL %f %f\n",GSL_REAL( val ), GSL_IMAG( val ) );
      }
#endif //_MYDEBUG_
    }
  }
  else{
    unsigned fftLen2 = fftLen/2;

    // calculate a direct component.
    snapShot_f      = _snapShotArray->getSnapShot(0);
    arrayManifold_f = _bfWeightV[0]->wq_f(0);    
    gsl_blas_zdotc(arrayManifold_f, snapShot_f, &val);
    gsl_vector_complex_set(_vector, 0, val);

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
      snapShot_f      = _snapShotArray->getSnapShot(fbinX);
      arrayManifold_f = _bfWeightV[0]->wq_f(fbinX);
      gsl_blas_zdotc(arrayManifold_f, snapShot_f, &val);
      if( fbinX < fftLen2 ){
	gsl_vector_complex_set(_vector, fbinX, val);
	gsl_vector_complex_set(_vector, _fftLen - fbinX, gsl_complex_conjugate(val) );
      }
      else
	gsl_vector_complex_set(_vector, fftLen2, val);
    }
  }

  _increment();
  return _vector;
}

void SubbandDS::reset()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();

  if ( _snapShotArray != NULL )
    _snapShotArray->zero();

  VectorComplexFeatureStream::reset();
  _endOfSamples = false;
}

void calcAllDelays(double x, double y, double z, const gsl_matrix* mpos, gsl_vector* delays)
{
  unsigned chanN = mpos->size1;
	
  for (unsigned chX = 0; chX < chanN; chX++) {
    double x_m = gsl_matrix_get(mpos, chX, 0);
    double y_m = gsl_matrix_get(mpos, chX, 1);
    double z_m = gsl_matrix_get(mpos, chX, 2);

    double delta = sqrt(x_m*x_m + y_m*y_m + z_m*z_m) / SSPEED;
    gsl_vector_set(delays, chX, delta);
  }

  // Normalize by delay of the middle element
  double mid = gsl_vector_get(delays, chanN/2);
  for (unsigned chX = 0; chX < chanN; chX++)
    gsl_vector_set(delays, chX, gsl_vector_get(delays, chX) - mid);
}


void calcProduct(gsl_vector_complex* synthesisSamples, gsl_matrix_complex* gs_W, gsl_vector_complex* product)
{
  gsl_complex a = gsl_complex_rect(1,0);
  gsl_complex b = gsl_complex_rect(0,0);
  gsl_blas_zgemv(CblasTrans, a, gs_W, synthesisSamples, b, product);
}

/**
   @brief calculate the output of the GSC beamformer for a frequency bin, that is,
          Y(f) = ( wq(f) - B(f) wa(f) )_H * X(f)

   @param gsl_vector_complex* snapShot[in] an input subband snapshot
   @param gsl_vector_gsl_vector_complex* wq[in/out] a quiescent weight vector
   @param gsl_vector_gsl_vector_complex* wa[in]     B * wa (sidelobe canceller)
   @param gsl_complex *pYf[out] (gsl_complex *)the output of the GSC beamformer
   @param bool normalizeWeight[in] Normalize the entire weight vector if true.
*/
void calcOutputOfGSC( const gsl_vector_complex* snapShot, 
		      gsl_vector_complex* wl_f, gsl_vector_complex* wq_f, 
		      gsl_complex *pYf, bool normalizeWeight )
{
  unsigned chanN = wq_f->size;
  gsl_complex wq_fn, wl_fn;
  gsl_vector_complex *myWq_f = gsl_vector_complex_alloc( chanN );

  if( wq_f->size != wl_f->size ){
    fprintf(stderr,"calcOutputOfGSC:The lengths of weight vectors must be the same.\n");
    throw  j_error("calcOutputOfGSC:The lengths of weight vectors must be the same.\n");
  }

  // calculate wq(f) - B(f) wa(f)
  //gsl_vector_complex_sub( wq_f, wl_f );
  for(unsigned i=0;i<chanN;i++){
    wq_fn = gsl_vector_complex_get( wq_f ,i );
    wl_fn = gsl_vector_complex_get( wl_f ,i );
    gsl_vector_complex_set( myWq_f, i, gsl_complex_sub( wq_fn, wl_fn ) );
  }

  /* normalize the entire weight */
  /* w <- w / ( ||w|| * chanN )  */
  if ( true==normalizeWeight ){ 
    double norm = gsl_blas_dznrm2( myWq_f );
    for(unsigned i=0;i<chanN;i++){
      gsl_complex val;
      val = gsl_complex_div_real( gsl_vector_complex_get( myWq_f, i ), norm * chanN);
      gsl_vector_complex_set( myWq_f, i, val );
    }
    //double norm1 = gsl_blas_dznrm2( myWq_f ); fprintf(stderr,"norm %e %e\n",norm,norm1 );
  }

  // calculate  ( wq(f) - B(f) wa(f) )^H * X(f)
  gsl_blas_zdotc( myWq_f, snapShot, pYf );
  gsl_vector_complex_free( myWq_f );
}

SubbandGSC::~SubbandGSC()
{
  fprintf(stderr,"SubbandGSC::~SubbandGSC()\n");
}

/**
   @brief  calculate the outputs of the GSC beamformer at each frame
 */
const gsl_vector_complex* SubbandGSC::next(int frameX)
{
  if (frameX == _frameX) return _vector;
  
  const gsl_vector_complex* snapShot_f;
  gsl_vector_complex* wl_f;
  gsl_vector_complex* wq_f;
  gsl_complex val;
  unsigned chanX = 0;
  unsigned fftLen = _fftLen;

  if( 0 == _bfWeightV.size() ){
    fprintf(stderr,"call calcGSCWeightsX() once\n");
    throw  j_error("call calcGSCWeightsX() once\n");
  }

  this->_allocImage();
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frameX);
    if( true==(*itr)->isEnd() ) _endOfSamples = true;
    _snapShotArray->newSample( samp, chanX);  chanX++;
  }
  _snapShotArray->update();

  if( _halfBandShift == true ){
    for (unsigned fbinX = 0; fbinX < fftLen; fbinX++) {
      snapShot_f = _snapShotArray->getSnapShot(fbinX);
      wq_f = _bfWeightV[0]->wq_f(fbinX);
      wl_f = _bfWeightV[0]->wl_f(fbinX);
      
      calcOutputOfGSC( snapShot_f, wl_f,  wq_f, &val, _normalizeWeight );
      gsl_vector_complex_set(_vector, fbinX, val);
    }
  }
  else{
    unsigned fftLen2 = fftLen/2;

    // calculate a direct component.
    snapShot_f = _snapShotArray->getSnapShot(0);
    wq_f       = _bfWeightV[0]->wq_f(0);
    gsl_blas_zdotc( wq_f, snapShot_f, &val);
    gsl_vector_complex_set(_vector, 0, val);
    //wq_f = _bfWeights->wq_f(0);
    //wl_f = _bfWeights->wl_f(0);
    //calcOutputOfGSC( snapShot_f, chanN(), wl_f, wq_f, &val );
    //gsl_vector_complex_set(_vector, 0, val);

    // calculate outputs from bin 1 to fftLen-1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
      snapShot_f = _snapShotArray->getSnapShot(fbinX);
      wq_f = _bfWeightV[0]->wq_f(fbinX);
      wl_f = _bfWeightV[0]->wl_f(fbinX);
      
      calcOutputOfGSC( snapShot_f, wl_f, wq_f, &val, _normalizeWeight );
      if( fbinX < fftLen2 ){
	gsl_vector_complex_set(_vector, fbinX,           val);
	gsl_vector_complex_set(_vector, _fftLen - fbinX, gsl_complex_conjugate(val) );
      }
      else
	gsl_vector_complex_set(_vector, fftLen2, val);
    }
  }

  _increment();
  
  return _vector;
}

void SubbandGSC::setQuiescentWeights_f( unsigned fbinX, const gsl_vector_complex * srcWq )
{
  this->_allocBFWeight( 1, 1 );
  gsl_vector_complex* destWq = _bfWeightV[0]->wq_f(fbinX); 
  gsl_vector_complex_memcpy( destWq, srcWq );
  _bfWeightV[0]->calcBlockingMatrix( fbinX );
}

void SubbandGSC::calcGSCWeights(double sampleRate, const gsl_vector* delaysT )
{
  this->_allocBFWeight( 1, 1 );
  _bfWeightV[0]->calcMainlobe( sampleRate, delaysT, true );
}

void SubbandGSC::calcGSCWeights2( double sampleRate, const gsl_vector* delaysT, const gsl_vector* delaysI )
{
  this->_allocBFWeight( 1, 2 );
  _bfWeightV[0]->calcMainlobe2( sampleRate, delaysT, delaysI, true );
}

/**
   @brief calculate the quescent vectors with N linear constraints
   @param double sampleRate[in]
   @param const gsl_vector* delaysT[in] delaysT[chanN]
   @param const gsl_matrix* delaysJ[in] delaysJ[NC-1][chanN]
   @param int NC[in] the number of constraints (= the number of target and interference signals )
   @note you can put multiple constraints. For example, you can constrain a beamformer to preserve a target signal and, at the same time, to suppress interference signals.
 */
void SubbandGSC::calcGSCWeightsN( double sampleRate, const gsl_vector* delaysT, const gsl_matrix* delaysIs, unsigned NC )
{
  if( NC == 2 ){
    gsl_vector* delaysI = gsl_vector_alloc( chanN() );

    for( unsigned i=0;i<chanN();i++)
      gsl_vector_set( delaysI, i, gsl_matrix_get( delaysIs, 0, i ) );
    this->_allocBFWeight( 1, 2 );
    _bfWeightV[0]->calcMainlobe2( sampleRate, delaysT, delaysI, true );

    gsl_vector_free(delaysI);
  }
  else{
    this->_allocBFWeight( 1, NC );
    _bfWeightV[0]->calcMainlobeN( sampleRate, delaysT, delaysIs, NC, true );
  }
}

bool SubbandGSC::writeFIRCoeff( const String& fn, unsigned winType )
{
  if( 0 == _bfWeightV.size() ){
    fprintf(stderr,"call calcArrayManifoldVectorsX() once\n");
    return(false);
  }
  return( _bfWeightV[0]->writeFIRCoeff(fn,winType) );
}

/**
   @brief set active weights for each frequency bin.
   @param int fbinX
   @param const gsl_vector* packedWeight
*/
void SubbandGSC::setActiveWeights_f( unsigned fbinX, const gsl_vector* packedWeight )
{
  if( 0 == _bfWeightV.size() ){
    fprintf(stderr,"call calcGSCWeightsX() once\n");
    throw  j_error("call calcGSCWeightsX() once\n");
  }

  _bfWeightV[0]->calcSidelobeCancellerP_f( fbinX, packedWeight );
}

void SubbandGSC::zeroActiveWeights()
{
  if( 0 == _bfWeightV.size() ){
    fprintf(stderr,"call calcGSCWeightsX() once\n");
    throw  j_error("call calcGSCWeightsX() once\n");
  }

  gsl_vector_complex *wa = gsl_vector_complex_calloc( chanN() - _bfWeightV[0]->NC() );
  for (unsigned fbinX = 0; fbinX < _fftLen; fbinX++){
   _bfWeightV[0]->calcSidelobeCancellerU_f( fbinX, wa );
  }
  gsl_vector_complex_free( wa );
}

/**
   @brief solve the scaling ambiguity
   
   @param gsl_matrix_complex *W[in/out] MxN unmixing matrix. M and N are the number of sources and sensors, respectively.
   @param float dThreshold[in]
 */
static bool scaling( gsl_matrix_complex *W, float dThreshold = 1.0E-8 )
{
  size_t M = W->size1; // the number of sources
  size_t N = W->size2; // the number of sensors
  gsl_matrix_complex *Wp = gsl_matrix_complex_alloc( N, M );

  pseudoinverse( W, Wp, dThreshold );

#if 0
  // W <= diag[W+] * W, where W+ is a pseudoinverse matrix.
  for ( size_t i = 0; i < M; i++ ){// src
    for ( size_t j = 0; j < N; j++ ){// mic
      //gsl_complex dii   = gsl_vector_complex_get( diagWp, i );
      gsl_complex dii   = gsl_matrix_complex_get(Wp, i, i );
      gsl_complex W_ij  = gsl_matrix_complex_get( W , i ,j );
      gsl_matrix_complex_set( W, i, j, gsl_complex_mul( dii, W_ij ) );
    }
  }
  
#else
  int stdsnsr = (int) N / 2;
  for ( size_t i = 0; i < M; i++ ){// src
    for ( size_t j = 0; j < N; j++ ){// mic
      gsl_complex Wp_mi = gsl_matrix_complex_get( Wp, stdsnsr, i );
      gsl_complex W_ij  = gsl_matrix_complex_get( W , i ,j );

      gsl_matrix_complex_set( W, i, j, gsl_complex_mul( Wp_mi, W_ij ) );
    }
  }
#endif

  gsl_matrix_complex_free( Wp );
  return(true);
}

/**
   @brief 
   @param unsigned fftLen[in] The point of the FFT
   @param bool halfBandShift[in]
   @param float myu[in] the forgetting factor for the covariance matrix
   @param float sigma2[in] the amount of diagonal loading
 */
SubbandGSCRLS::SubbandGSCRLS(unsigned fftLen, bool halfBandShift, float myu, float sigma2, const String& nm ): 
  SubbandGSC( fftLen, halfBandShift, nm ),
  _myu(myu), _isActiveWeightVectorUpdated(true),_alpha(-1.0),_qctype(NO_QUADRATIC_CONSTRAINT)
{
  _gz = new gsl_vector_complex*[fftLen];
  _Pz = new gsl_matrix_complex*[fftLen];
  _Zf = NULL;
  _wa = NULL;
  _diagonalWeights = new float[fftLen];
  for (unsigned fbinX = 0; fbinX < fftLen; fbinX++){
    _gz[fbinX] = NULL;
    _Pz[fbinX] = NULL;
    _diagonalWeights[fbinX] = sigma2;
  }
  _PzH_Z = NULL;
  _I     = NULL;
  _mat1  = NULL;
}

SubbandGSCRLS::~SubbandGSCRLS()
{
  _freeImage4SubbandGSCRLS();
  delete [] _gz;
  delete [] _Pz;
  delete [] _diagonalWeights;
}

void SubbandGSCRLS::initPrecisionMatrix( float sigma2 )
{
  _freeImage4SubbandGSCRLS();
  if( false == _allocImage4SubbandGSCRLS() )
    throw j_error("call calcGSCWeightsX() once\n");

  for (unsigned fbinX = 0; fbinX < _fftLen; fbinX++){
    gsl_matrix_complex_set_zero( _Pz[fbinX] );
    for (unsigned chanX = 0; chanX < _Pz[fbinX]->size1; chanX++){
      gsl_matrix_complex_set( _Pz[fbinX], chanX, chanX, gsl_complex_rect( 1/sigma2, 0 ) );
    }
  }
}

void SubbandGSCRLS::setPrecisionMatrix( unsigned fbinX, gsl_matrix_complex *Pz )
{
  _freeImage4SubbandGSCRLS();
  if( false == _allocImage4SubbandGSCRLS() )
    throw j_error("call calcGSCWeightsX() once\n");
  
  unsigned nChan = chanN();
  for (unsigned chanX = 0; chanX < nChan; chanX++) {
    for (unsigned chanY = 0; chanY < nChan; chanY++) {
      gsl_matrix_complex_set( _Pz[fbinX], chanX, chanY, gsl_matrix_complex_get(Pz, chanX, chanY) );
    }
  }
}

const gsl_vector_complex* SubbandGSCRLS::next(int frameX)
{
  if (frameX == _frameX) return _vector;
  const gsl_vector_complex* snapShot_f;
  gsl_vector_complex* wl_f;
  gsl_vector_complex* wq_f;
  gsl_complex val;
  unsigned chanX = 0;
  unsigned fftLen = _fftLen;

  if( 0 == _bfWeightV.size() ){
    fprintf(stderr,"call calcGSCWeightsX() once\n");
    throw  j_error("call calcGSCWeightsX() once\n");
  }
   if( NULL == _Zf ){
     fprintf(stderr,"set the precision matrix with initPrecisionMatrix() or setPrecisionMatrix()\n");
     throw  j_error("set the precision matrix with initPrecisionMatrix() or setPrecisionMatrix()\n");
   }

  this->_allocImage();
 for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frameX);
    if( true==(*itr)->isEnd() ) _endOfSamples = true;
    _snapShotArray->newSample( samp, chanX);  chanX++;
  }
  _snapShotArray->update();
  
  if( _halfBandShift == true ){
    fprintf(stderr,"not yet implemented\n");
    throw  j_error("not yet implemented\n");
  }
  else{
    unsigned fftLen2 = fftLen/2;

    // calculate a direct component.
    snapShot_f = _snapShotArray->getSnapShot(0);
    wq_f       = _bfWeightV[0]->wq_f(0);
    gsl_blas_zdotc( wq_f, snapShot_f, &val);
    gsl_vector_complex_set(_vector, 0, val);

    // calculate outputs from bin 1 to fftLen-1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
      snapShot_f = _snapShotArray->getSnapShot(fbinX);
      wq_f = _bfWeightV[0]->wq_f(fbinX);
      wl_f = _bfWeightV[0]->wl_f(fbinX);
      
      calcOutputOfGSC( snapShot_f, wl_f, wq_f, &val, _normalizeWeight );
      if( fbinX < fftLen2 ){
	gsl_vector_complex_set(_vector, fbinX,           val);
	gsl_vector_complex_set(_vector, _fftLen - fbinX, gsl_complex_conjugate(val) );
      }
      else
	gsl_vector_complex_set(_vector, fftLen2, val);
    }
    
    this->_updateActiveWeightVector2( frameX );
  }

  _increment();
  
  return _vector;
}

void SubbandGSCRLS::reset()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();

  if ( _snapShotArray != NULL )
    _snapShotArray->zero();

  VectorComplexFeatureStream::reset();
  _endOfSamples = false;  

}

void SubbandGSCRLS::_updateActiveWeightVector2( int frameX )
{
  if( false == _isActiveWeightVectorUpdated )
    return;
  
  unsigned NC    = _bfWeightV[0]->NC();
  unsigned nChan = chanN();
  gsl_matrix_complex** B  = _bfWeightV[0]->B();
  gsl_vector_complex** old_wa = _bfWeightV[0]->wa();

  for (unsigned fbinX = 1; fbinX <= _fftLen/2; fbinX++){
    const gsl_vector_complex *Xf = _snapShotArray->getSnapShot(fbinX);
    gsl_complex nu, de;

    // calc. output of the blocking matrix.    
    gsl_blas_zgemv( CblasConjTrans, gsl_complex_rect(1.0,0.0), B[fbinX], Xf, gsl_complex_rect(0.0,0.0), _Zf );

    // calc. the gain vector 
    gsl_blas_zgemv( CblasConjTrans, gsl_complex_rect(1.0,0.0), _Pz[fbinX], _Zf, gsl_complex_rect(0.0,0.0), _PzH_Z );
    gsl_blas_zgemv( CblasNoTrans,   gsl_complex_rect(1.0/_myu,0.0), _Pz[fbinX], _Zf, gsl_complex_rect(0.0,0.0), _gz[fbinX] );
    gsl_blas_zdotc( _PzH_Z, _Zf, &de );
    de = gsl_complex_add_real( gsl_complex_mul_real( de, 1.0/_myu ), 1.0 );
    for(unsigned chanX =0;chanX<nChan-NC;chanX++){
      gsl_complex val;
      nu = gsl_vector_complex_get( _gz[fbinX], chanX );      
      val = gsl_complex_div( nu, de );
      gsl_vector_complex_set( _gz[fbinX], chanX, val );
    }

    // calc. the precision matrix
    for(unsigned chanX=0;chanX<nChan-NC;chanX++){
      for(unsigned chanY=0;chanY<nChan-NC;chanY++){
	gsl_complex oldPz, val1, val2;
	oldPz = gsl_matrix_complex_get( _Pz[fbinX], chanX, chanY );
	val1  = gsl_complex_mul( gsl_vector_complex_get( _gz[fbinX], chanX ), gsl_complex_conjugate( gsl_vector_complex_get( _PzH_Z, chanY ) ) );
	val2  = gsl_complex_mul_real( gsl_complex_sub( oldPz, val1 ),  1.0/_myu );
	gsl_matrix_complex_set( _Pz[fbinX], chanX, chanY, val2 );
      }
    }

    { // update the active weight vecotr
      gsl_complex epA  = gsl_complex_conjugate( gsl_vector_complex_get( _vector, fbinX ) );

      gsl_matrix_complex_memcpy( _mat1, _Pz[fbinX] );
      gsl_matrix_complex_scale(  _mat1, gsl_complex_rect( - _diagonalWeights[fbinX], 0.0 ) );
      gsl_matrix_complex_add( _mat1, _I );
      gsl_blas_zgemv( CblasNoTrans, gsl_complex_rect(1.0,0.0), _mat1, old_wa[fbinX], gsl_complex_rect(0.0,0.0), _wa );
      for(unsigned chanX=0;chanX<nChan-NC;chanX++){
	gsl_complex val1 = gsl_vector_complex_get( _wa, chanX );
	gsl_complex val2 = gsl_complex_mul( gsl_vector_complex_get( _gz[fbinX], chanX ), epA );
	gsl_vector_complex_set( _wa, chanX, gsl_complex_add( val1, val2 ) );
      }
      if( _qctype == CONSTANT_NORM ){
	double nrmwa = gsl_blas_dznrm2( _wa );
	for(unsigned chanX=0;chanX<nChan-NC;chanX++)
	  gsl_vector_complex_set( _wa, chanX, gsl_complex_mul_real( gsl_vector_complex_get( _wa, chanX), _alpha/nrmwa ) );
      }
      else if( _qctype == THRESHOLD_LIMITATION ){
	double nrmwa = gsl_blas_dznrm2( _wa );
	if( ( nrmwa * nrmwa ) >= _alpha ){
	  for(unsigned chanX=0;chanX<nChan-NC;chanX++)
	    gsl_vector_complex_set( _wa, chanX, gsl_complex_mul_real( gsl_vector_complex_get( _wa, chanX), _alpha/nrmwa ) );
	}
      }
      //fprintf( stderr, "%d: %e\n", frameX, gsl_blas_dznrm2 ( _wa ) );
      _bfWeightV[0]->calcSidelobeCancellerU_f( fbinX, _wa );
    }
  }

}

/*
  @brief allocate image blocks for _gz[] and _Pz[].
 */
bool SubbandGSCRLS::_allocImage4SubbandGSCRLS()
{
  if( 0 == _bfWeightV.size() ){
    fprintf(stderr,"call calcWeightsX() once\n");
    return false;
  }
  unsigned NC    = _bfWeightV[0]->NC();
  unsigned nChan = chanN();

  _Zf = gsl_vector_complex_calloc( nChan - NC );
  _wa = gsl_vector_complex_calloc( nChan - NC );

  for (unsigned fbinX = 0; fbinX < _fftLen; fbinX++){
    _Pz[fbinX] = gsl_matrix_complex_calloc( nChan - NC , nChan - NC );
    _gz[fbinX] = gsl_vector_complex_calloc( nChan - NC );
    _bfWeightV[0]->calcSidelobeCancellerU_f( fbinX, _wa );
  }

  _PzH_Z = gsl_vector_complex_calloc( nChan - NC );
  _I     = gsl_matrix_complex_alloc( nChan - NC, nChan - NC );
  gsl_matrix_complex_set_identity( _I );
  _mat1  = gsl_matrix_complex_alloc( nChan - NC, nChan - NC );

  return true;
}

void SubbandGSCRLS::_freeImage4SubbandGSCRLS()
{
  for (unsigned fbinX = 0; fbinX < _fftLen; fbinX++) {
    if( NULL != _Pz[fbinX] ){
      gsl_vector_complex_free( _gz[fbinX] );
      gsl_matrix_complex_free( _Pz[fbinX] );
      _gz[fbinX] = NULL;
      _Pz[fbinX] = NULL;
    }
  }

  if( NULL != _Zf ){
    gsl_vector_complex_free( _Zf );
    gsl_vector_complex_free( _wa );    
    _Zf = NULL;
    _wa = NULL;
    gsl_vector_complex_free( _PzH_Z );
    gsl_matrix_complex_free( _I );
    gsl_matrix_complex_free( _mat1 );
    _PzH_Z = NULL;
    _I     = NULL;
    _mat1  = NULL;
  }
}

SubbandMMI::~SubbandMMI()
{
  if( NULL != _interferenceOutputs ){
    for (unsigned i = 0; i < _nSource; i++)
      gsl_vector_complex_free(_interferenceOutputs[i]);
    delete [] _interferenceOutputs;
  }
  if( NULL != _avgOutput )
    gsl_vector_complex_free( _avgOutput );
}

/**
   @brief calculate a quiescent weight vector and blocking matrix for each frequency bin.
   @param double sampleRate[in]
   @param const gsl_matrix* delayMat[in] delayMat[nSource][nChan]
 */
void SubbandMMI::calcWeights( double sampleRate, const gsl_matrix* delayMat )
{
  this->_allocBFWeight( _nSource, 1 );
  gsl_vector *delaysT = gsl_vector_alloc( chanN() );

  for( unsigned srcX=0;srcX<_nSource;srcX++){
    gsl_matrix_get_row( delaysT, delayMat, srcX );
    _bfWeightV[srcX]->calcMainlobe( sampleRate, delaysT, true );
  }

  gsl_vector_free( delaysT );
}

/**
   @brief calculate a quiescent weight vector with N constraints and blocking matrix for each frequency bin.
   @param double sampleRate[in]
   @param const gsl_matrix* delayMat[in] delayMat[nSource][nChan]
   @param unsigned NC[in] the number of linear constraints
 */
void SubbandMMI::calcWeightsN( double sampleRate, const gsl_matrix* delayMat, unsigned NC )
{
  if( 0 == _bfWeightV.size() )
    this->_allocBFWeight( _nSource, NC );

  gsl_vector* delaysT  = gsl_vector_alloc( chanN() );
  gsl_vector* tmpV     = gsl_vector_alloc( chanN() );
  gsl_matrix* delaysIs = gsl_matrix_alloc( NC - 1, chanN() ); // the number of interferences = 1

  for(unsigned srcX=0;srcX<_nSource;srcX++){
    gsl_matrix_get_row( delaysT, delayMat, srcX );
    for(unsigned srcY=0,i=0;i<NC-1;srcY++){
      if( srcY == srcX ) continue;
      gsl_matrix_get_row( tmpV, delayMat, srcY );
      gsl_matrix_set_row( delaysIs, i, tmpV );
      i++;
    }
    _bfWeightV[srcX]->calcMainlobeN( sampleRate, delaysT, delaysIs, NC, true );
  }

  gsl_vector_free( delaysT );
  gsl_vector_free( tmpV );
  gsl_matrix_free( delaysIs );
}

/**
   @brief set an active weight vector for each frequency bin, 
          calculate the entire weights of a lower branch, and make a NxM demixing matrix. 
          N is the number of sound sources and M is the number of channels.
   @param unsigned fbinX[in]
   @param const gsl_matrix* packedWeights[in] [nSource][nChan]
   @param int option[in] 0:do nothing, 1:solve scaling, 
 */
void SubbandMMI::setActiveWeights_f( unsigned fbinX, const gsl_matrix* packedWeights, int option )
{ 
  if( 0 == _bfWeightV.size() ){
    fprintf(stderr,"call calcWeightsX() once\n");
    throw  j_error("call calcWeightsX() once\n");
  }
  if( packedWeights->size1 != _nSource ){
    fprintf(stderr,"The number of columns must be the number of sources %d\n", _nSource);
    throw  j_error("The number of columns must be the number of sources %d\n", _nSource);
  }

  {// calculate the entire weight of a lower branch.
    gsl_vector* packedWeight = gsl_vector_alloc( packedWeights->size2 );

    for(unsigned srcX=0;srcX<_nSource;srcX++){
      gsl_matrix_get_row( packedWeight, packedWeights, srcX );
      _bfWeightV[srcX]->calcSidelobeCancellerP_f( fbinX, packedWeight );
    }
    gsl_vector_free( packedWeight );
  }


  {// make a demixing matrix and solve scaling ambiguity
    gsl_vector_complex* tmpV;
    gsl_matrix_complex* Wl_f = gsl_matrix_complex_alloc( _nSource, chanN() ); // a demixing matrix
    gsl_vector_complex* new_wl_f = gsl_vector_complex_alloc( chanN() );
    gsl_complex val;

    for(unsigned srcX=0;srcX<_nSource;srcX++){
      tmpV = _bfWeightV[srcX]->wl_f( fbinX );
      gsl_matrix_complex_set_row( Wl_f, srcX, tmpV );
    }
    // conjugate a matrix
    for(unsigned i=0;i<Wl_f->size1;i++){// Wl_f[_nSource][_nChan]
       for(unsigned j=0;j<Wl_f->size2;j++){
	 val = gsl_matrix_complex_get( Wl_f, i, j );
	 gsl_matrix_complex_set( Wl_f, i, j, gsl_complex_conjugate( val ) );
       }
    }

    if( option == 1 ){
      if( false==scaling( Wl_f, 1.0E-7 ) )
	fprintf(stderr,"%d : scaling is not performed\n", fbinX);
    }

    // put an updated vector back
    for(unsigned srcX=0;srcX<_nSource;srcX++){
      gsl_matrix_complex_get_row( new_wl_f, Wl_f, srcX );
      for(unsigned i=0;i<new_wl_f->size;i++){// new_wl_f[_nChan]
	val = gsl_vector_complex_get( new_wl_f, i );
	gsl_vector_complex_set( new_wl_f, i, gsl_complex_conjugate( val ) );
      }
      _bfWeightV[srcX]->setSidelobeCanceller_f( fbinX, new_wl_f );
    }

    gsl_vector_complex_free( new_wl_f );
    gsl_matrix_complex_free( Wl_f );
  }

}

/**
   @brief set active weight matrix & vector for each frequency bin, 
          calculate the entire weights of a lower branch, and make a Ns x Ns demixing matrix, 
          where Ns is the number of sound sources.
   @param unsigned fbinX[in]
   @param const gsl_vector* pkdWa[in] active weight matrices [2 * nSrc * (nChan-NC) * nSrc]
   @param const gsl_vector* pkdwb[in] active weight vectors  [2 * nSrc * nSrc]
   @param int option[in] 0:do nothing, 1:solve scaling, 
 */
void SubbandMMI::setHiActiveWeights_f( unsigned fbinX, const gsl_vector* pkdWa, const gsl_vector* pkdwb, int option )
{

  if( 0 == _bfWeightV.size() ){
    fprintf(stderr,"call calcWeightsX() once\n");
    throw  j_error("call calcWeightsX() once\n");
  }
  unsigned NC = _bfWeightV[0]->NC();
  if( pkdWa->size != (2*_nSource*(chanN()-NC)*_nSource) ){
    fprintf(stderr,"The size of the 2nd arg must be 2 * %d * %d * %d\n", _nSource, (chanN()-NC), _nSource );
    throw  j_error("The size of the 2nd arg must be 2 * %d * %d * %d\n", _nSource, (chanN()-NC), _nSource );
  }
  if( pkdwb->size != (2*_nSource*_nSource) ){
    fprintf(stderr,"The size of the 3rd arg must be 2 * %d * %d\n", _nSource, _nSource );
    throw  j_error("The size of the 3rd arg must be 2 * %d * %d\n", _nSource, _nSource );
  }

  {// make a demixing matrix and solve scaling ambiguity
    gsl_matrix_complex** Wa_f = new gsl_matrix_complex*[_nSource];
    gsl_vector_complex** wb_f = new gsl_vector_complex*[_nSource];
    gsl_matrix_complex*  Wc_f = gsl_matrix_complex_alloc( _nSource, _nSource );// concatenate wb_f[] and conjugate it
    gsl_vector_complex*  we_f = gsl_vector_complex_alloc( chanN()-NC ); // the entire weight ( Wa_f[] * wb_f[] )
    const gsl_complex alpha = gsl_complex_rect(1.0,0.0);
    const gsl_complex beta  = gsl_complex_rect(0.0,0.0);

    for(unsigned srcX=0;srcX<_nSource;srcX++){
      Wa_f[srcX] = gsl_matrix_complex_alloc( chanN()-NC, _nSource ); // active matrices
      wb_f[srcX] = gsl_vector_complex_alloc( _nSource ); // active vectors
    }

    // unpack vector data for active matrices
    for(unsigned srcX=0,i=0;srcX<_nSource;srcX++){
      for(unsigned chanX=0;chanX<chanN()-NC;chanX++){
	for(unsigned srcY=0;srcY<_nSource;srcY++){
	  gsl_complex val = gsl_complex_rect( gsl_vector_get( pkdWa, 2*i ), gsl_vector_get( pkdWa, 2*i+1 ) );
	  gsl_matrix_complex_set( Wa_f[srcX], chanX, srcY, val );
	  i++;
	}
      }
    }
    // unpack vector data for active vectors and make a demixing matrix
    for(unsigned srcX=0,i=0;srcX<_nSource;srcX++){
      for(unsigned srcY=0;srcY<_nSource;srcY++){
	gsl_complex val = gsl_complex_rect( gsl_vector_get( pkdwb, 2*i ), gsl_vector_get( pkdwb, 2*i+1 ) );
	gsl_matrix_complex_set( Wc_f, srcX, srcY, gsl_complex_conjugate( val ) );
	//gsl_matrix_complex_set( Wc_f, srcX, srcY, val );
	i++;
      }
    }

    if( option == 1 ){
      if( false==scaling( Wc_f, 1.0E-7 ) )
	fprintf(stderr,"%d : scaling is not performed\n", fbinX);
    }
    
    // put an updated vector back and calculate the entire active weights
    for(unsigned srcX=0;srcX<_nSource;srcX++){
      for(unsigned srcY=0;srcY<_nSource;srcY++){
	gsl_complex val = gsl_matrix_complex_get( Wc_f, srcX, srcY );
	gsl_vector_complex_set( wb_f[srcX], srcY, gsl_complex_conjugate( val ) );
	//gsl_vector_complex_set( wb_f[srcX], srcY, val );
      }
      gsl_blas_zgemv( CblasNoTrans, alpha, Wa_f[srcX], wb_f[srcX], beta, we_f );
      _bfWeightV[srcX]->calcSidelobeCancellerU_f( fbinX, we_f );
    }

    for(unsigned srcX=0;srcX<_nSource;srcX++){
      gsl_matrix_complex_free( Wa_f[srcX] );
      gsl_vector_complex_free( wb_f[srcX] );
    }
    
    delete [] Wa_f;
    delete [] wb_f;
    gsl_matrix_complex_free( Wc_f );
    gsl_vector_complex_free( we_f );
  }
  
}

/**
   @brief  calculate the outputs of the GSC beamformer at each frame
 */
const gsl_vector_complex* SubbandMMI::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  const gsl_vector_complex* snapShot_f;
  gsl_vector_complex* wl_f;
  gsl_vector_complex* wq_f;
  gsl_complex val;
  unsigned chanX = 0;
  unsigned fftLen = _fftLen;

  if( 0 == _bfWeightV.size() ){
    fprintf(stderr,"call calcWeightsX() once\n");
    throw  j_error("call calcWeightsX() once\n");
  }

  this->_allocImage();
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frameX);
    if( true==(*itr)->isEnd() ) _endOfSamples = true;
    _snapShotArray->newSample( samp, chanX);  chanX++;
  }
  _snapShotArray->update();

  if( _halfBandShift == true ){
    for (unsigned fbinX = 0; fbinX < fftLen; fbinX++) {
      snapShot_f = _snapShotArray->getSnapShot(fbinX);
      wq_f = _bfWeightV[_targetSourceX]->wq_f(fbinX);
      wl_f = _bfWeightV[_targetSourceX]->wl_f(fbinX);
      
      calcOutputOfGSC( snapShot_f, wl_f,  wq_f, &val );
      gsl_vector_complex_set(_vector, fbinX, val);
    }
  }
  else{
    unsigned fftLen2 = fftLen/2;

    // calculate a direct component.
    snapShot_f = _snapShotArray->getSnapShot(0);
    wq_f       = _bfWeightV[_targetSourceX]->wq_f(0);
    gsl_blas_zdotc( wq_f, snapShot_f, &val);
    gsl_vector_complex_set(_vector, 0, val);
    //wq_f = _bfWeightV[_targetSourceX]->wq_f(0);
    //wl_f = _bfWeightV[_targetSourceX]->wl_f(0);      
    //calcOutputOfGSC( snapShot_f, chanN(), wl_f,  wq_f, &val );
    //gsl_vector_complex_set(_vector, 0, val);

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
      snapShot_f = _snapShotArray->getSnapShot(fbinX);
      wq_f = _bfWeightV[_targetSourceX]->wq_f(fbinX);
      wl_f = _bfWeightV[_targetSourceX]->wl_f(fbinX);

      calcOutputOfGSC( snapShot_f, wl_f,  wq_f, &val );
      if( fbinX < fftLen2 ){
	gsl_vector_complex_set(_vector, fbinX,           val);
	gsl_vector_complex_set(_vector, _fftLen - fbinX, gsl_complex_conjugate(val) );
      }
      else
	gsl_vector_complex_set(_vector, fftLen2, val);
    }
  }

  {// post-filtering
    double alpha;
    gsl_vector_complex** wq  = _bfWeightV[_targetSourceX]->arrayManifold(); // use a D&S beamformer output as a clean signal.
    gsl_vector_complex*  wp1 = _bfWeightV[0]->wp1();
    gsl_vector_complex** prevCSDs = _bfWeightV[_targetSourceX]->CSDs();

    if(  _frameX > 0 )
      alpha =  _alpha;
    else
      alpha = 0.0;

    if( (int)TYPE_APAB & _pfType ){
      ApabFilter( wq, _snapShotArray, fftLen, chanN(), _halfBandShift, _vector, chanN()/2 );
    }
    else if( (int)TYPE_ZELINSKI1_REAL & _pfType || (int)TYPE_ZELINSKI1_ABS & _pfType ){

      if( (int)TYPE_ZELINSKI2 & _pfType  )
	wq =  _bfWeightV[_targetSourceX]->wq(); // just use a beamformer output as a clean signal.

      if( _frameX < MINFRAMES )// just update cross spectral densities
	ZelinskiFilter( wq, _snapShotArray, _halfBandShift, _vector, prevCSDs, wp1, alpha, (int)NO_USE_POST_FILTER);
      else
	ZelinskiFilter( wq, _snapShotArray, _halfBandShift, _vector, prevCSDs, wp1, alpha, _pfType );
    }
  }

  if( _useBinaryMask==true ){// binary mask
    this->calcInterferenceOutputs();

    if( _binaryMaskType == 0 )
      gsl_vector_complex_memcpy( _interferenceOutputs[_targetSourceX], _vector );
    this->binaryMasking( _interferenceOutputs, _targetSourceX, _vector );
  }

  _increment();
  return _vector;
}


/**
   @brief construct beamformers for all the sources and get the outputs.
   @return Interference signals are stored in _interferenceOutputs[].
 */
void SubbandMMI::calcInterferenceOutputs()
{
  const gsl_vector_complex* snapShot_f;
  gsl_vector_complex* wl_f;
  gsl_vector_complex* wq_f;
  gsl_complex val;
  unsigned chanX = 0;
  unsigned fftLen = _fftLen;

  if( _halfBandShift == true ){
    for (unsigned srcX = 0; srcX < _nSource; srcX++) {
      if( _binaryMaskType == 0 ){// store GSC's outputs in _interferenceOutputs[].
	if( srcX == _targetSourceX )
	  continue;
	for (unsigned fbinX = 0; fbinX < fftLen; fbinX++) {
	  snapShot_f = _snapShotArray->getSnapShot(fbinX);
	  wq_f = _bfWeightV[srcX]->wq_f(fbinX);
	  wl_f = _bfWeightV[srcX]->wl_f(fbinX);
	  calcOutputOfGSC( snapShot_f, wl_f,  wq_f, &val );
	  gsl_vector_complex_set( _interferenceOutputs[srcX], fbinX, val);
	}
      }
      else{ // store outputs of an upper branch _interferenceOutputs[].
	for (unsigned fbinX = 0; fbinX < fftLen; fbinX++) {
	  snapShot_f = _snapShotArray->getSnapShot(fbinX);
	  wq_f = _bfWeightV[srcX]->wq_f(fbinX);
	  gsl_blas_zdotc(wq_f, snapShot_f, &val);
	  gsl_vector_complex_set( _interferenceOutputs[srcX], fbinX, val);
	}
      }
    }// for (unsigned srcX = 0; srcX < _nSource; srcX++)
  }
  else{
    unsigned fftLen2 = fftLen/2;

    for (unsigned srcX = 0; srcX < _nSource; srcX++) {
      if( _binaryMaskType == 0 ){// store GSC's outputs.
	if( srcX == _targetSourceX )
	  continue;

	// calculate a direct component.
	snapShot_f = _snapShotArray->getSnapShot(0);
	wq_f       = _bfWeightV[srcX]->wq_f(0);
	gsl_blas_zdotc( wq_f, snapShot_f, &val);
	gsl_vector_complex_set(_interferenceOutputs[srcX], 0, val);

	// calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
	for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
	  snapShot_f = _snapShotArray->getSnapShot(fbinX);
	  wq_f = _bfWeightV[srcX]->wq_f(fbinX);
	  wl_f = _bfWeightV[srcX]->wl_f(fbinX);
	  calcOutputOfGSC( snapShot_f, wl_f,  wq_f, &val );
	  if( fbinX < fftLen2 ){
	    gsl_vector_complex_set(_interferenceOutputs[srcX], fbinX, val);
	    gsl_vector_complex_set(_interferenceOutputs[srcX], _fftLen - fbinX, gsl_complex_conjugate(val) );
	  }
	  else
	    gsl_vector_complex_set(_interferenceOutputs[srcX], fftLen2, val);
	}// for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++)
      }
      else{ // store outputs of an upper branch _interferenceOutputs[].
	// calculate a direct component.
	snapShot_f = _snapShotArray->getSnapShot(0);
	wq_f       = _bfWeightV[srcX]->wq_f(0);
	gsl_blas_zdotc( wq_f, snapShot_f, &val);
	gsl_vector_complex_set(_interferenceOutputs[srcX], 0, val);

	// calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
	for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
	  snapShot_f = _snapShotArray->getSnapShot(fbinX);
	  wq_f = _bfWeightV[srcX]->wq_f(fbinX);
	  gsl_blas_zdotc( wq_f, snapShot_f, &val);
	  if( fbinX < fftLen2 ){
	    gsl_vector_complex_set(_interferenceOutputs[srcX], fbinX, val);
	    gsl_vector_complex_set(_interferenceOutputs[srcX], _fftLen - fbinX, gsl_complex_conjugate(val) );
	  }
	  else
	    gsl_vector_complex_set(_interferenceOutputs[srcX], fftLen2, val);
	}// for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++)
      }
    }// for (unsigned srcX = 0; srcX < _nSource; srcX++)
  }

  {// post-filtering
    for (unsigned srcX = 0; srcX < _nSource; srcX++) {
      if( _binaryMaskType == 0 && srcX == _targetSourceX )
	continue;

      double alpha;
      gsl_vector_complex** wq  = _bfWeightV[srcX]->arrayManifold(); // use a D&S beamformer output as a clean signal.
      gsl_vector_complex*  wp1 = _bfWeightV[0]->wp1();
      gsl_vector_complex** prevCSDs = _bfWeightV[srcX]->CSDs();

      if(  _frameX > 0 )
	alpha =  _alpha;
      else
	alpha = 0.0;

      if( (int)TYPE_APAB & _pfType ){
	ApabFilter( wq, _snapShotArray, fftLen, chanN(), _halfBandShift, _interferenceOutputs[srcX], chanN()/2 );
      }
      else if( (int)TYPE_ZELINSKI1_REAL & _pfType || (int)TYPE_ZELINSKI1_ABS & _pfType ){

	if( (int)TYPE_ZELINSKI2 & _pfType  )
	  wq =  _bfWeightV[srcX]->wq(); // just use a beamformer output as a clean signal.

	if( _frameX < MINFRAMES )// just update cross spectral densities
	  ZelinskiFilter( wq, _snapShotArray, _halfBandShift, _interferenceOutputs[srcX], prevCSDs, wp1, alpha, (int)NO_USE_POST_FILTER);
	else
	  ZelinskiFilter( wq, _snapShotArray, _halfBandShift, _interferenceOutputs[srcX], prevCSDs, wp1, alpha, _pfType );
      }
    }// for (unsigned srcX = 0; srcX < _nSource; srcX++)
  }

  return;
}

/*
  @brief averaging the output of a beamformer recursively.
          Y'(f,t) = a * Y(f,t) + ( 1 -a ) *  Y'(f,t-1)

*/
static void setAveragedOutput( gsl_vector_complex* avgOutput, unsigned fbinX, gsl_vector_complex* curOutput, double avgFactor )
{
  gsl_complex prev = gsl_complex_mul_real( gsl_vector_complex_get( avgOutput, fbinX ), avgFactor );
  gsl_complex curr = gsl_complex_mul_real( gsl_vector_complex_get( curOutput, fbinX ), ( 1.0 - avgFactor ) );
  gsl_vector_complex_set( avgOutput, fbinX, gsl_complex_add(prev,curr) );
}

/**
   @brief calculate a mean of subband components over frequency bins.

 */
static gsl_complex getMeanOfSubbandC( int fbinX, gsl_vector_complex *output, unsigned fftLen, unsigned fwidth )
{
  if( fwidth <= 1 )
    return( gsl_vector_complex_get( output, fbinX ) );

  int fbinStart, fbinEnd;
  unsigned count = 0;
  gsl_complex sum = gsl_complex_rect( 0.0, 0.0 );

  fbinStart = fbinX - fwidth/2;
  if( fbinStart < 1 ) fbinStart = 1; // a direct component is not used
  fbinEnd = fbinX + fwidth/2;
  if( fbinEnd >= fftLen ) fbinEnd = fftLen - 1;

  for(int i=fbinStart;i<=fbinEnd;i++,count++){
    sum = gsl_complex_add( sum, gsl_vector_complex_get( output, i ) );
  }

  return( gsl_complex_div_real( sum, (double) count ) );
}

/**
   @brief Do binary masking. If an output power of a target source > outputs of interferences, the target signal is set to 0. 

   @note if avgFactor >= 0, a recursively averaged subband compoent is set instead of 0.
   @param gsl_vector_complex** interferenceOutputs[in]
   @param unsinged targetSourceX[in]
   @param gsl_vector_complex* output[in/out]
*/
void SubbandMMI::binaryMasking( gsl_vector_complex** interferenceOutputs, unsigned targetSourceX, gsl_vector_complex* output )
{
  unsigned fftLen = _fftLen;
  gsl_complex tgtY, valY, itfY;
  gsl_complex newVal, sumVal;
  gsl_vector_complex* targetOutput = _interferenceOutputs[_targetSourceX];

  if( _halfBandShift == true ){
    for (unsigned fbinX = 0; fbinX < fftLen; fbinX++) {
      double tgtPow, valPow, maxPow = 0.0;

      tgtY = gsl_vector_complex_get( targetOutput, fbinX );
      tgtPow = gsl_complex_abs2( tgtY );
      for (unsigned srcX = 0; srcX < _nSource; srcX++) {
	if( srcX == _targetSourceX )
	  continue;
	valY = gsl_vector_complex_get( interferenceOutputs[srcX], fbinX );
	valPow = gsl_complex_abs2( valY );
	if( valPow > maxPow ){
	  maxPow = valPow;
	  itfY = valY;
	}
      }// for (unsigned srcX = 0; srcX < _nSource; srcX++)
      if( _avgFactor >= 0.0 )
	newVal = gsl_complex_mul_real( getMeanOfSubbandC( (int)fbinX, _avgOutput, _fftLen, _fwidth ), _avgFactor );
      else
	newVal = gsl_complex_rect( 0.0, 0.0 );
      if( tgtPow < maxPow ){
	gsl_vector_complex_set( output, fbinX, newVal );
	if( _avgFactor >= 0.0 )
	  gsl_vector_complex_set( _avgOutput, fbinX, newVal );
      }
      else{
	if( _avgFactor >= 0.0 )
	  setAveragedOutput( _avgOutput, fbinX, output, _avgFactor );
      }
    }
  }
  else{
    unsigned fftLen2 = fftLen/2;

    for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
      double tgtPow, valPow, maxPow = 0.0;

      tgtY = gsl_vector_complex_get( targetOutput, fbinX );
      tgtPow = gsl_complex_abs2( tgtY );
      for (unsigned srcX = 0; srcX < _nSource; srcX++) {// seek a subband component with maximum power
	if( srcX == _targetSourceX )
	  continue;
	valY = gsl_vector_complex_get( interferenceOutputs[srcX], fbinX );
	valPow = gsl_complex_abs2( valY );
	if( valPow > maxPow ){
	  maxPow = valPow;
	  itfY = valY;
	}
      }// for (unsigned srcX = 0; srcX < _nSource; srcX++)
      if( _avgFactor >= 0.0 ) // set the estimated value (_avgFactor * _avgOutput[t-1])
	newVal = gsl_complex_mul_real( getMeanOfSubbandC( (int)fbinX, _avgOutput, _fftLen/2, _fwidth ), _avgFactor );
      else // set 0 to the output
	newVal = gsl_complex_rect( 0.0, 0.0 );
      if( tgtPow < maxPow ){
	if( fbinX < fftLen2 ){
	  gsl_vector_complex_set( output, fbinX, newVal );
	  gsl_vector_complex_set( output, _fftLen - fbinX, newVal );
	}
	else
	  gsl_vector_complex_set( output, fftLen2, newVal );
	if( _avgFactor >= 0.0 )
	  gsl_vector_complex_set( _avgOutput, fbinX, newVal );
      }
      else{
	if( _avgFactor >= 0.0 )
	  setAveragedOutput( _avgOutput, fbinX, output, _avgFactor );
      }
    }//for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++)
  }

  return;
}

SubbandMVDR::SubbandMVDR( unsigned fftLen, bool halfBandShift, const String& nm)
  : SubbandDS( fftLen, halfBandShift, nm )
{ 
  if( halfBandShift == true ){
    fprintf(stderr,"halfBandShift==true is not yet supported\n");
    throw jallocation_error("halfBandShift==true is not yet supported\n");
  }
  
  _R    = (gsl_matrix_complex** )malloc( (fftLen/2+1) * sizeof(gsl_matrix_complex*) );
  _invR = (gsl_matrix_complex** )malloc( (fftLen/2+1) * sizeof(gsl_matrix_complex*) );
  if( _R == NULL || _invR == NULL ){
    fprintf(stderr,"SubbandMVDR: gsl_matrix_complex_alloc failed\n");
    throw jallocation_error("SubbandMVDR: gsl_matrix_complex_alloc failed\n");
  }

  _wmvdr = (gsl_vector_complex** )malloc( (fftLen/2+1) * sizeof(gsl_vector_complex*) );
  if( _wmvdr == NULL ){
    fprintf(stderr,"SubbandMVDR: gsl_vector_complex_alloc failed\n");
    throw jallocation_error("SubbandMVDR: gsl_vector_complex_alloc failed\n");
  }

  _diagonalWeights = (float *)calloc( (fftLen/2+1), sizeof(float) );
  if( _diagonalWeights == NULL ){
    fprintf(stderr,"SubbandMVDR: cannot allocate RAM\n");
    throw jallocation_error("SubbandMVDR: cannot allocate RAM\n");
  }

  for( unsigned fbinX=0;fbinX<=_fftLen/2;fbinX++){
    _R[fbinX] = NULL;
    _invR[fbinX]  = NULL;
    _wmvdr[fbinX] = NULL;
  }

}

SubbandMVDR::~SubbandMVDR()
{
  unsigned fftLen2 = _fftLen / 2;
  
  for( unsigned fbinX=0;fbinX<=fftLen2;fbinX++){
    if( NULL!=_R[fbinX] )
      gsl_matrix_complex_free( _R[fbinX] );
    if( NULL!=_invR[fbinX] )
      gsl_matrix_complex_free( _invR[fbinX] );
    if( NULL!= _wmvdr[fbinX] )
      gsl_vector_complex_free( _wmvdr[fbinX] );
  }
  free(_R);
  free(_invR);
  free(_wmvdr);
  free(_diagonalWeights);
}

void SubbandMVDR::clearChannel()
{
  unsigned fftLen2 = _fftLen / 2;

  SubbandDS::clearChannel();
  
  for( unsigned fbinX=0;fbinX<=fftLen2;fbinX++){
    if( NULL!=_R[fbinX] ){
      gsl_matrix_complex_free( _R[fbinX] );
      _R[fbinX] = NULL;
    }
    if( NULL!= _wmvdr[fbinX] ){
      gsl_vector_complex_free( _wmvdr[fbinX] );
      _wmvdr[fbinX] = NULL;
    }
  }
}

bool SubbandMVDR::calcMVDRWeights( double sampleRate, double dThreshold, bool calcInverseMatrix )
{
  if( NULL == _R[0] ){
    fprintf(stderr,"Set a spatial spectral matrix before calling calcMVDRWeights()\n");
    throw jallocation_error("Set a spatial spectral matrix before calling calcMVDRWeights()\n");
  }
  if( 0 == _bfWeightV.size() ){
    fprintf(stderr,"call calcArrayManifoldVectorsX() once\n");
    throw j_error("call calcArrayManifoldVectorsX() once\n");
  }

  unsigned nChan = chanN();
  gsl_vector_complex *tmpH = gsl_vector_complex_alloc( nChan );
  gsl_complex val1 = gsl_complex_rect( 1.0, 0.0 );
  gsl_complex val0 = gsl_complex_rect( 0.0, 0.0 );
  gsl_complex Lambda;
  bool ret;

  if( NULL == _wmvdr[0] ){
    _wmvdr[0] = gsl_vector_complex_alloc( nChan );
  }
  for( unsigned chanX=0 ; chanX < nChan ;chanX++ ){
    gsl_vector_complex_set( _wmvdr[0], chanX, val1 );
  }
  for(unsigned fbinX=1;fbinX<=_fftLen/2;fbinX++){
    gsl_complex norm;
    const gsl_vector_complex* arrayManifold_f = _bfWeightV[0]->wq_f(fbinX);

    if( NULL == _invR[fbinX] )
      _invR[fbinX] = gsl_matrix_complex_alloc( nChan, nChan );

    // calculate the inverse matrix of the coherence matrix
    if( true == calcInverseMatrix ){
      ret = pseudoinverse( _R[fbinX], _invR[fbinX], dThreshold );
      if( false==ret )
	gsl_matrix_complex_set_identity( _invR[fbinX] );
    }

    gsl_blas_zgemv( CblasConjTrans, val1, _invR[fbinX], arrayManifold_f, val0, tmpH ); // tmpH = invR^H * d
    gsl_blas_zdotc( tmpH, arrayManifold_f, &Lambda ); // Lambda = d^H * invR * d
    norm = gsl_complex_mul_real( Lambda, nChan );

    if( NULL == _wmvdr[fbinX] ){
      _wmvdr[fbinX] = gsl_vector_complex_alloc( nChan );
    }
    for( unsigned chanX=0 ; chanX < nChan ;chanX++ ){
      gsl_complex val = gsl_vector_complex_get( tmpH, chanX );// val = invR^H * d
      gsl_vector_complex_set( _wmvdr[fbinX], chanX, gsl_complex_div( val, norm /*Lambda*/ ) );
    }
  }

  gsl_vector_complex_free( tmpH );

  return true;
}

/**
   @brief set the spatial spectral matrix for the MVDR beamformer

   @param unsigned fbinX[in]
   @param gsl_matrix_complex* Rnn[in]
 */
bool SubbandMVDR::setNoiseSpatialSpectralMatrix( unsigned fbinX, gsl_matrix_complex* Rnn )
{

  if( Rnn->size1 != chanN() ){
    fprintf(stderr,"The number of the rows of the matrix must be %d but it is %d\n", chanN(), Rnn->size1 );
    return false;
  }
  if( Rnn->size2 != chanN() ){
    fprintf(stderr,"The number of the columns of the matrix must be %d but it is %d\n", chanN(), Rnn->size2 );
    return false;
  }

  if( _R[fbinX] == NULL ){
    _R[fbinX] = gsl_matrix_complex_alloc( chanN(), chanN() );
  }

  for(unsigned m=0;m<chanN();m++){
    for(unsigned n=0;n<chanN();n++){
      gsl_matrix_complex_set( _R[fbinX], m, n, gsl_matrix_complex_get( Rnn, m, n ) );
    }
  }

  return true;
}

/**
   @brief calculate the coherence matrix in the case of the diffuse noise field.

   @param const gsl_matrix* micPositions[in] geometry of the microphone array. micPositions[no. channels][x,y,z]
   @param double sampleRate[in]
   @param double sspeed[in]
 */
bool SubbandMVDR::setDiffuseNoiseModel( const gsl_matrix* micPositions, double sampleRate, double sspeed )
{
  size_t micN  = micPositions->size1;
  
  if( micN != chanN() ){
    fprintf(stderr,"The number of microphones must be %d but it is %d\n", chanN(), micN );
    return false;
  }
  if( micPositions->size2 < 3 ){
    fprintf(stderr,"The microphone positions should be described in the three dimensions\n");
    return false;
  }

  gsl_matrix *dm = gsl_matrix_alloc( micN, micN );

  for(unsigned fbinX=0;fbinX<=_fftLen/2;fbinX++){
    if( _R[fbinX] == NULL ){
      _R[fbinX] = gsl_matrix_complex_alloc( micN, micN );
    }
  }

  {// calculate the distance matrix.
     for(unsigned m=0;m<micN;m++){
       for(unsigned n=0;n<m;n++){ //for(unsigned n=0;n<micN;n++){
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
     //for(unsigned m=0;m<micN;m++){ gsl_matrix_set( dm, m, m, 0.0 );}
  }

  {
    for(unsigned fbinX=0;fbinX<=_fftLen/2;fbinX++){
      //double omega_d_c = 2.0 * M_PI * sampleRate * fbinX / ( _fftLen * sspeed );
      double omega_d_c = 2.0 * sampleRate * fbinX / ( _fftLen * sspeed );

      for(unsigned m=0;m<micN;m++){
	for(unsigned n=0;n<m;n++){ 
	  double Gamma_mn = gsl_sf_sinc( omega_d_c * gsl_matrix_get( dm, m, n ) );
	  gsl_matrix_complex_set( _R[fbinX], m, n, gsl_complex_rect( Gamma_mn, 0.0 ) );
	}// for(unsigned n=0;n<micN;n++){ 
      }// for(unsigned m=0;m<micN;m++){
      for(unsigned m=0;m<micN;m++){ 
	 gsl_matrix_complex_set( _R[fbinX], m, m, gsl_complex_rect( 1.0, 0.0 ) );
      }
      for(unsigned m=0;m<micN;m++){
	for(unsigned n=m+1;n<micN;n++){
	  gsl_complex val = gsl_matrix_complex_get( _R[fbinX], n, m );
	  gsl_matrix_complex_set( _R[fbinX], m, n, val );
	}
      }
    }// for(unsigned fbinX=0;fbinX<=_fftLen/2;fbinX++){
  }
  //gsl_sf_sinc (double x);
  
  gsl_matrix_free(dm);

  return true;
}

void SubbandMVDR::setAllLevelsOfDiagonalLoading( float diagonalWeight )
{
  if( _R == NULL ){
    fprintf(stderr,"Construct first a noise covariance matrix\n");
    throw j_error("Construct first a noise covariance matrix\n");
  }
  for(unsigned fbinX=0;fbinX<=_fftLen/2;fbinX++){       
    _diagonalWeights[fbinX] = diagonalWeight;
    for( unsigned chanX=0 ; chanX < chanN() ;chanX++ ){// diagonal loading
      gsl_complex val = gsl_matrix_complex_get( _R[fbinX], chanX, chanX );
      gsl_matrix_complex_set( _R[fbinX], chanX, chanX, gsl_complex_add_real( val, _diagonalWeights[fbinX] ) );
    }
  }
}

void SubbandMVDR::setLevelOfDiagonalLoading( unsigned fbinX, float diagonalWeight )
{
  if( _R == NULL ){
    fprintf(stderr,"Construct first a noise covariance matrix\n");
    throw j_error("Construct first a noise covariance matrix\n");
  }
  _diagonalWeights[fbinX] = diagonalWeight;
  for( unsigned chanX=0 ; chanX < chanN() ;chanX++ ){// diagonal loading
    gsl_complex val = gsl_matrix_complex_get( _R[fbinX], chanX, chanX );
    gsl_matrix_complex_set( _R[fbinX], chanX, chanX, gsl_complex_add_real( val, _diagonalWeights[fbinX] ) );
  }
}

const gsl_vector_complex* SubbandMVDR::next(int frameX)
{
  if (frameX == _frameX) return _vector;

  if( 0 == _bfWeightV.size() ){
    fprintf(stderr,"call calcArrayManifoldVectorsX() once\n");
    throw j_error("call calcArrayManifoldVectorsX() once\n");
  }
  if( NULL == _wmvdr[0] ){
    fprintf(stderr,"call calcMVDRWeights() once\n");
    throw j_error("call calcMVDRWeights() once\n");
  }

  unsigned chanX = 0;
  const gsl_vector_complex* snapShot_f;
  gsl_complex val;
  unsigned fftLen = _fftLen;

  this->_allocImage();
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frameX);
    if( true==(*itr)->isEnd() ) _endOfSamples = true;
    _snapShotArray->newSample( samp, chanX);  chanX++;
  }
  _snapShotArray->update();

   if( _halfBandShift == true ){
     // TODO : implement
   }
   else{
     unsigned fftLen2 = fftLen/2;

     // calculate a direct component.
     snapShot_f = _snapShotArray->getSnapShot(0);
     gsl_blas_zdotc( _wmvdr[0], snapShot_f, &val );
     gsl_vector_complex_set(_vector, 0, val);

     // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
     for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
       snapShot_f = _snapShotArray->getSnapShot(fbinX);
       gsl_blas_zdotc( _wmvdr[fbinX], snapShot_f, &val );
       if( fbinX < fftLen2 ){
	 gsl_vector_complex_set(_vector, fbinX, val);
	 gsl_vector_complex_set(_vector, _fftLen - fbinX, gsl_complex_conjugate(val) );
       }
       else
	 gsl_vector_complex_set(_vector, fftLen2, val);
     }
   }

   _increment();
   return _vector;
}

SubbandMVDRGSC::SubbandMVDRGSC( unsigned fftLen, bool halfBandShift, const String& nm)
  : SubbandMVDR( fftLen, halfBandShift, nm ), _normalizeWeight(false)
{
}

SubbandMVDRGSC::~SubbandMVDRGSC()
{
}

void SubbandMVDRGSC::setActiveWeights_f( unsigned fbinX, const gsl_vector* packedWeight )
{
  if( 0 == _bfWeightV.size() ){
    fprintf(stderr,"set the quiescent vector once\n");
    throw  j_error("set the quiescent vector once\n");
  }
  _bfWeightV[0]->calcSidelobeCancellerP_f( fbinX, packedWeight );
}

void SubbandMVDRGSC::zeroActiveWeights()
{
  if( 0 == _bfWeightV.size() ){
    fprintf(stderr,"call calcGSCWeightsX() once\n");
    throw  j_error("call calcGSCWeightsX() once\n");
  }

  gsl_vector_complex *wa = gsl_vector_complex_calloc( chanN() - _bfWeightV[0]->NC() );
  for (unsigned fbinX = 0; fbinX < _fftLen; fbinX++){
   _bfWeightV[0]->calcSidelobeCancellerU_f( fbinX, wa );
  }
  gsl_vector_complex_free( wa );
}

/**
   @brief compute the blocking matrix so as to satisfy the orthogonal condition 
          with the delay-and-sum beamformer's weight.
 */
bool SubbandMVDRGSC::calcBlockingMatrix1( double sampleRate, const gsl_vector* delaysT )
{
  this->_allocBFWeight( 1, 1 );
  _bfWeightV[0]->calcMainlobe( sampleRate, delaysT, true );
  return true;
}

/**
   @brief compute the blocking matrix so as to satisfy the orthogonal condition 
          with the MVDR beamformer's weight.
 */
bool SubbandMVDRGSC::calcBlockingMatrix2()
{
  if( NULL == _wmvdr[0] ){
    fprintf(stderr,"You have to call calcMVDRWeights() first\n");
    return false;
  }
  
  this->_allocBFWeight( 1, 1 );

  if( _halfBandShift == true ){
     // TODO : implement
    fprintf(stderr,"Not yet implemented\n");
    return false;
  }
  else{
    unsigned fftLen2 = _fftLen/2;

    for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
      gsl_vector_complex* destWq = _bfWeightV[0]->wq_f(fbinX); 
      gsl_vector_complex_memcpy( destWq, _wmvdr[fbinX] );
      _bfWeightV[0]->calcBlockingMatrix( fbinX );
    }
  }

  return true;
}

void SubbandMVDRGSC::upgradeBlockingMatrix()
{
  unsigned fbinX;
  const unsigned fftLen2 = _fftLen/2;
  gsl_matrix_complex** B = _bfWeightV[0]->B();
  gsl_vector_complex *weight = gsl_vector_complex_alloc( this->chanN() );

  /* printf("SubbandMVDRGSC: set the orthogonal matrix of the entire vector to the blocking matrix\n");*/
  
  for (unsigned fbinX = 1; fbinX < _fftLen; fbinX++) {
    gsl_vector_complex_memcpy( weight, _bfWeightV[0]->wq_f(fbinX) );
    gsl_vector_complex_sub(    weight, _bfWeightV[0]->wl_f(fbinX) );

    if( false==_calcBlockingMatrix( weight, _bfWeightV[0]->NC(), B[fbinX] ) ){
      throw j_error("_calcBlockingMatrix() failed\n");
    }
  }

  gsl_vector_complex_free( weight );
}

const gsl_vector_complex* SubbandMVDRGSC::blockingMatrixOutput( int outChanX )
{
  const gsl_vector_complex* snapShot_f;
   
  if( _halfBandShift == true ){
    // TODO : implement
  }
  else{
    const unsigned fftLen2 = _fftLen/2;
    gsl_matrix_complex** B = _bfWeightV[0]->B();
    gsl_vector_complex* bi = gsl_vector_complex_alloc( this->chanN() );
    gsl_complex val;
       
    for (unsigned fbinX = 0; fbinX <= fftLen2; fbinX++) {
      gsl_matrix_complex_get_col( bi, B[fbinX], outChanX );
      snapShot_f = _snapShotArray->getSnapShot(fbinX);
      gsl_blas_zdotc( bi, snapShot_f, &val);
      gsl_vector_complex_set( _vector, fbinX, val );
    }

    gsl_vector_complex_free( bi );
  }

  return _vector;
}

const gsl_vector_complex* SubbandMVDRGSC::next(int frameX)
{

  if (frameX == _frameX) return _vector;

  if( 0 == _bfWeightV.size() ){
    fprintf(stderr,"call calcArrayManifoldVectorsX() once\n");
    throw j_error("call calcArrayManifoldVectorsX() once\n");
  }
  if( NULL == _wmvdr[0] ){
    fprintf(stderr,"call calcMVDRWeights() once\n");
    throw j_error("call calcMVDRWeights() once\n");
  }

  unsigned chanX = 0;
  const gsl_vector_complex* snapShot_f;
  gsl_vector_complex* wl_f;
  gsl_vector_complex* wq_f;
  gsl_complex val;
  unsigned fftLen = _fftLen;

  this->_allocImage();
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frameX);
    if( true==(*itr)->isEnd() ) _endOfSamples = true;
    _snapShotArray->newSample( samp, chanX);  chanX++;
  }
  _snapShotArray->update();

   if( _halfBandShift == true ){
     // TODO : implement
   }
   else{
     unsigned fftLen2 = fftLen/2;

     // calculate a direct component.
     snapShot_f = _snapShotArray->getSnapShot(0);
     gsl_blas_zdotc( _wmvdr[0], snapShot_f, &val );
     gsl_vector_complex_set(_vector, 0, val);

     // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
     for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
       snapShot_f = _snapShotArray->getSnapShot(fbinX);
       wl_f = _bfWeightV[0]->wl_f(fbinX);
      
       calcOutputOfGSC( snapShot_f, wl_f, _wmvdr[fbinX], &val, _normalizeWeight );
       if( fbinX < fftLen2 ){
	 gsl_vector_complex_set(_vector, fbinX, val);
	 gsl_vector_complex_set(_vector, _fftLen - fbinX, gsl_complex_conjugate(val) );
       }
       else
	 gsl_vector_complex_set(_vector, fftLen2, val);
     }
   }

   _increment();
   return _vector;
}

// ----- members for class `SubbandOrthogonalizer' -----
//
SubbandOrthogonalizer::SubbandOrthogonalizer(SubbandMVDRGSCPtr &beamformer, int outChanX,  const String& nm)
  : VectorComplexFeatureStream(beamformer->fftLen(), nm),
    _beamformer(beamformer),
    _outChanX(outChanX)
{ 
}

SubbandOrthogonalizer::~SubbandOrthogonalizer()
{
  
}

const gsl_vector_complex* SubbandOrthogonalizer::next(int frameX)
{
  const gsl_vector_complex* vector;
  if (frameX == _frameX) return _vector;
  
  
  if( _outChanX <= 0 ){
    vector = _beamformer->next(frameX);
  }
  else{
    vector = _beamformer->blockingMatrixOutput( _outChanX - 1 );
  }

  gsl_vector_complex_memcpy( _vector, vector );

  _increment();

  return _vector;
}


const gsl_vector_complex* SubbandBlockingMatrix::next(int frameX)
{
  if (frameX == _frameX) return _vector;
  
  const gsl_vector_complex* snapShot_f;
  gsl_vector_complex* wl_f;
  gsl_vector_complex* wq_f;
  gsl_complex val;
  unsigned chanX = 0;
  unsigned fftLen = _fftLen;

  if( 0 == _bfWeightV.size() ){
    fprintf(stderr,"call calcGSCWeightsX() once\n");
    throw  j_error("call calcGSCWeightsX() once\n");
  }

  this->_allocImage();
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++) {
    const gsl_vector_complex* samp = (*itr)->next(frameX);
    if( true==(*itr)->isEnd() ) _endOfSamples = true;
    _snapShotArray->newSample( samp, chanX);  chanX++;
  }
  _snapShotArray->update();

  if( _halfBandShift == true ){
    for (unsigned fbinX = 0; fbinX < fftLen; fbinX++) {
      snapShot_f = _snapShotArray->getSnapShot(fbinX);
      wq_f = _bfWeightV[0]->wq_f(fbinX);
      wl_f = _bfWeightV[0]->wl_f(fbinX);
      
      calcOutputOfGSC( snapShot_f, wl_f,  wq_f, &val, _normalizeWeight );
      gsl_vector_complex_set(_vector, fbinX, val);
    }
  }
  else{
    unsigned fftLen2 = fftLen/2;

    // calculate a direct component.
    snapShot_f = _snapShotArray->getSnapShot(0);
    wq_f       = _bfWeightV[0]->wq_f(0);
    gsl_blas_zdotc( wq_f, snapShot_f, &val);
    gsl_vector_complex_set(_vector, 0, val);
    //wq_f = _bfWeights->wq_f(0);
    //wl_f = _bfWeights->wl_f(0);
    //calcOutputOfGSC( snapShot_f, chanN(), wl_f, wq_f, &val );
    //gsl_vector_complex_set(_vector, 0, val);

    // calculate outputs from bin 1 to fftLen-1 by using the property of the symmetry.
    for (unsigned fbinX = 1; fbinX <= fftLen2; fbinX++) {
      snapShot_f = _snapShotArray->getSnapShot(fbinX);
      wq_f = _bfWeightV[0]->wq_f(fbinX);
      wl_f = _bfWeightV[0]->wl_f(fbinX);
      
      calcOutputOfGSC( snapShot_f, wl_f, wq_f, &val, _normalizeWeight );
      if( fbinX < fftLen2 ){
	gsl_vector_complex_set(_vector, fbinX,           val);
	gsl_vector_complex_set(_vector, _fftLen - fbinX, gsl_complex_conjugate(val) );
      }
      else
	gsl_vector_complex_set(_vector, fftLen2, val);
    }
  }

  _increment();
  
  return _vector;
}

// ----- definition for class DOAEstimatorSRPBase' -----
// 
DOAEstimatorSRPBase::DOAEstimatorSRPBase( unsigned nBest, unsigned fbinMax ):
  _widthTheta(0.25),
  _widthPhi(0.25),
  _minTheta(-M_PI),
  _maxTheta(M_PI),
  _minPhi(-M_PI),
  _maxPhi(M_PI),
  _fbinMin(1),
  _fbinMax(fbinMax),
  _accRPs(NULL),
  _nBest(nBest),
  _isTableInitialized(false),
  _rpMat(NULL),
  _engeryThreshold(0.0)
{
  _nBestRPs   = gsl_vector_calloc( _nBest );
  _argMaxDOAs = gsl_matrix_calloc( _nBest, 2 );
}

DOAEstimatorSRPBase::~DOAEstimatorSRPBase()
{
  if( NULL != _nBestRPs )
    gsl_vector_free( _nBestRPs );
  if( NULL != _argMaxDOAs )
    gsl_matrix_free( _argMaxDOAs );

  clearTable();
}

#ifdef __MBDEBUG__
void DOAEstimatorSRPBase::allocDebugWorkSapce()
{
  float nTheta = ( _maxTheta - _minTheta ) / _widthTheta + 0.5 + 1;
  float nPhi   = ( _maxPhi - _minPhi ) / _widthPhi  + 0.5 + 1;
  _rpMat = gsl_matrix_calloc( (int)nTheta, (int)nPhi );
}
#endif /* #ifdef __MBDEBUG__ */

void DOAEstimatorSRPBase::clearTable()
{
  //fprintf(stderr,"DOAEstimatorSRPBase::clearTable()\n");
  if( true == _isTableInitialized ){
    for(unsigned i=0;i<_svTbl.size();i++){
      for(unsigned fbinX=0;fbinX<=_fbinMax;fbinX++)
	gsl_vector_complex_free( _svTbl[i][fbinX] );
      free( _svTbl[i] );      
    }
    _svTbl.clear();
#ifdef __MBDEBUG__
    if( NULL != _rpMat ){
      gsl_matrix_free( _rpMat );
      _rpMat = NULL;
    }
#endif /* #ifdef __MBDEBUG__ */
    if( NULL != _accRPs ){
      gsl_vector_free( _accRPs );
      _accRPs = NULL;
    }
  }
  _isTableInitialized = false;
  //fprintf(stderr,"DOAEstimatorSRPBase::clearTable()2\n");
}


void DOAEstimatorSRPBase::_getNBestHypothesesFromACCRP()
{
  for(unsigned n=0;n<_nBest;n++){
    gsl_vector_set( _nBestRPs, n, -10e10 );
    gsl_matrix_set( _argMaxDOAs, n, 0, -M_PI);
    gsl_matrix_set( _argMaxDOAs, n, 1, -M_PI);
  }

  unsigned unitX = 0;
  unsigned thetaIdx = 0;
  for(double theta=_minTheta;thetaIdx<_nTheta;theta+=_widthTheta,thetaIdx++){
    unsigned phiIdx = 0;
    for(double phi=_minPhi;phiIdx<_nPhi;phi+=_widthPhi,phiIdx++){
      double rp = gsl_vector_get( _accRPs, unitX++ );
#ifdef __MBDEBUG__
      gsl_matrix_set( _rpMat, thetaIdx, phiIdx, rp );
#endif /* #ifdef __MBDEBUG__ */
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
  
}

void DOAEstimatorSRPBase::_initAccs()
{
  if( NULL != _accRPs )
    gsl_vector_set_zero(_accRPs);
#ifdef __MBDEBUG__
  if( NULL != _rpMat )
    gsl_matrix_set_zero( _rpMat );
#endif /* #ifdef __MBDEBUG__ */

  for(unsigned n=0;n<_nBest;n++){
    gsl_vector_set( _nBestRPs, n, -10e10 );
    gsl_matrix_set( _argMaxDOAs, n, 0, -M_PI);
    gsl_matrix_set( _argMaxDOAs, n, 1, -M_PI);
  }
} 

float calcEnergy( SnapShotArrayPtr snapShotArray, unsigned fbinMin, unsigned fbinMax, unsigned fftLen2, bool  halfBandShift )
{
  float rp = 0.0;
  unsigned chanN;
  gsl_complex val;

  if( halfBandShift == false ){

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = fbinMin; fbinX <= fbinMax; fbinX++) {
      const gsl_vector_complex* F = snapShotArray->getSnapShot(fbinX);
      chanN = F->size;

      gsl_blas_zdotc( F, F, &val ); // x^H y

      if( fbinX < fftLen2 ){
	rp += 2 * gsl_complex_abs2( val );
      }
      else{
	rp += gsl_complex_abs2( val );
      }
    }
  }
  else{
    fprintf(stderr,"_halfBandShift == true is not implemented yet\n");
    throw  j_error("_halfBandShift == true is not implemented yet\n");
  }

  //fprintf(stderr,"Engery %e\n", rp / ( 2* fftLen2 * chanN ) );
  
  return rp / ( 2* fftLen2 * chanN );
}

// ----- definition for class DOAEstimatorSRPDSBLA' -----
// 
DOAEstimatorSRPDSBLA::DOAEstimatorSRPDSBLA( unsigned nBest, unsigned sampleRate, unsigned fftLen, const String& nm ):
  DOAEstimatorSRPBase( nBest, fftLen/2 ),
  SubbandDS(fftLen, false, nm ),  
  _sampleRate(sampleRate)
{
  //fprintf(stderr,"DOAEstimatorSRPDSBLA\n");
  _arraygeometry = NULL;
  setSearchParam();
}

DOAEstimatorSRPDSBLA::~DOAEstimatorSRPDSBLA()
{
  if( NULL != _arraygeometry )
    gsl_matrix_free( _arraygeometry );
}

void DOAEstimatorSRPDSBLA::setArrayGeometry( gsl_vector *positions )
{
  if( NULL != _arraygeometry )
    gsl_matrix_free( _arraygeometry );

  _arraygeometry = gsl_matrix_alloc( positions->size, 3 );
  for(unsigned i=0;i<positions->size;i++){
    gsl_matrix_set( _arraygeometry, i, 0, gsl_vector_get( positions, i ) );
  }
}

void DOAEstimatorSRPDSBLA::_calcSteeringUnitTable()
{
  int nChan = (int)chanN();
  if( nChan == 0 ){
    fprintf(stderr,"Set the channel\n");
    return;
  }

  _nTheta = (unsigned)( ( _maxTheta - _minTheta ) / _widthTheta + 0.5 );
  _nPhi   = 1;
  int maxUnit  = _nTheta * _nPhi;
  _svTbl.resize(maxUnit);

  for(unsigned i=0;i<maxUnit;i++){
    _svTbl[i] = (gsl_vector_complex **)malloc((_fbinMax+1)*sizeof(gsl_vector_complex *));
    if( NULL == _svTbl[i] ){
      fprintf(stderr,"could not allocate image : %d\n", maxUnit );
    }
    for(unsigned fbinX=0;fbinX<=_fbinMax;fbinX++)
      _svTbl[i][fbinX] = gsl_vector_complex_calloc( nChan );
  }

  if( NULL != _accRPs )
    gsl_vector_free( _accRPs );
  _accRPs = gsl_vector_calloc( maxUnit );

  unsigned unitX = 0;
  unsigned thetaIdx = 0;
  for(double theta=_minTheta;thetaIdx<_nTheta;theta+=_widthTheta,thetaIdx++){
    gsl_vector_complex *weights;
      
    setLookDirection( nChan, theta );
    weights = _svTbl[unitX][0];
    for(unsigned chanX=0;chanX<nChan;chanX++)
      gsl_vector_complex_set( weights, chanX, gsl_complex_rect(1,0) );
    for(unsigned fbinX=_fbinMin;fbinX<=_fbinMax;fbinX++){
      weights = _svTbl[unitX][fbinX];
      gsl_vector_complex_memcpy( weights, _bfWeightV[0]->wq_f(fbinX)) ;
    }
    unitX++;
  }
  
#ifdef __MBDEBUG__
  allocDebugWorkSapce();
#endif /* #ifdef __MBDEBUG__ */

  _isTableInitialized = true;
}

double DOAEstimatorSRPDSBLA::_calcResponsePower( unsigned unitX )
{  
  const gsl_vector_complex* weights; /* weights of the combining-unit */
  const gsl_vector_complex* snapShot_f;
  gsl_complex val;
  double rp  = 0.0;

  if( _halfBandShift == false ){

    // calculate outputs from bin 1 to fftLen - 1 by using the property of the symmetry.
    for (unsigned fbinX = _fbinMin; fbinX <= _fbinMax; fbinX++) {
      snapShot_f = _snapShotArray->getSnapShot(fbinX);
      weights    = _svTbl[unitX][fbinX];
      gsl_blas_zdotc( weights, snapShot_f, &val);

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

const gsl_vector_complex* DOAEstimatorSRPDSBLA::next( int frameX )
{
  if (frameX == _frameX) return _vector;

  unsigned chanX = 0;
  double rp;

  for(unsigned n=0;n<_nBest;n++){
    gsl_vector_set( _nBestRPs, n, -10e10 );
    gsl_matrix_set( _argMaxDOAs, n, 0, -M_PI);
    gsl_matrix_set( _argMaxDOAs, n, 1, -M_PI);
  }

  this->_allocImage();
#define __MBDEBUG__
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

  unsigned unitX = 0;
  unsigned thetaIdx = 0;
  for(double theta=_minTheta;thetaIdx<_nTheta;theta+=_widthTheta,thetaIdx++){
      //setLookDirection( theta, phi );
    rp = _calcResponsePower( unitX );
    gsl_vector_set( _accRPs, unitX, gsl_vector_get( _accRPs, unitX ) + rp );
    unitX++;
#ifdef __MBDEBUG__
    gsl_matrix_set( _rpMat, thetaIdx, 0, rp);
#endif /* #ifdef __MBDEBUG__ */
    //fprintf( stderr, "t=%0.8f rp=%e\n" , theta, rp );
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
	  gsl_matrix_set( _argMaxDOAs, n1, 1, 0);
	  break;
	}
      }
      // for(unsinged n1=0;n1<_nBest-1;n1++)
    }
  }

  _increment();
  return _vector;
}

void DOAEstimatorSRPDSBLA::setLookDirection( int nChan, double theta )
{
  gsl_vector* delays = gsl_vector_alloc(nChan);
  double refPosition = gsl_matrix_get( _arraygeometry, 0, 0 );
  
  gsl_vector_set( delays, 0, 0 );
  for(int chanX=1;chanX<nChan;chanX++){
    double dist = gsl_matrix_get( _arraygeometry, chanX, 0 ) - refPosition;
    if( dist < 0 ){ dist = -dist; }

    gsl_vector_set( delays, chanX, dist * cos(theta) );
  }
  calcArrayManifoldVectors( _sampleRate, delays );
  gsl_vector_free( delays );
}

void DOAEstimatorSRPDSBLA::reset()
{
  for (_ChannelIterator itr = _channelList.begin(); itr != _channelList.end(); itr++)
    (*itr)->reset();

  if ( _snapShotArray != NULL )
    _snapShotArray->zero();

  VectorComplexFeatureStream::reset();
  _endOfSamples = false;
}
