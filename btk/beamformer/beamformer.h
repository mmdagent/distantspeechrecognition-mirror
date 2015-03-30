//                              -*- C++ -*-
//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.beamforming
//  Purpose: Beamforming in the subband domain.
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

#ifndef _beamformer_
#define _beamformer_

#ifdef BTK_MEMDEBUG
#include "memcheck/memleakdetector.h"
#endif
#include <stdio.h>
#include <assert.h>
#include <float.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_fft_complex.h>
#include <common/refcount.h>
#include "common/jexception.h"

#include "stream/stream.h"
//#include "stream/pyStream.h"
#include "beamformer/spectralinfoarray.h"
#include "modulated/modulated.h"

#define SSPEED 343740.0

class beamformerWeights {
  bool _halfBandShift;
  unsigned _fftLen;
  unsigned _chanN;
  unsigned _NC; // the numbef of constraints
  gsl_vector_complex** _wq; // a quiescent weight vector for each frequency bin, _wq[fbinX][chanN]
  gsl_matrix_complex** _B;  // a blocking matrix for each frequency bin,         _B[fbinX][chanN][chanN-NC]
  gsl_vector_complex** _wa; // an active weight vector for each frequency bin,   _wa[fbinX][chanN-NC]
  gsl_vector_complex** _wl; // _wl[fbinX] = _B[fbinX] * _wa[fbinX]
  gsl_vector_complex** _ta; // do time alignment for multi-channel waves. It is also called an array manifold. _ta[fbinX][chanN].
  gsl_vector_complex*  _wp1;  // a weight vector of postfiltering,   _wp[fbinX]
  gsl_vector_complex** _CSDs; // cross spectral density for the post-filtering
  
public:
  beamformerWeights( unsigned fftLen, unsigned chanN, bool halfBandShift, unsigned NC = 1 );
  ~beamformerWeights();

  void calcMainlobe(  double sampleRate, const gsl_vector* delays,  bool isGSC );
  void calcMainlobe2( double sampleRate, const gsl_vector* delaysT, const gsl_vector* delaysJ, bool isGSC );
  void calcMainlobeN( double sampleRate, const gsl_vector* delaysT, const gsl_matrix* delaysIs, unsigned NC, bool isGSC );
  void calcSidelobeCancellerP_f( unsigned fbinX, const gsl_vector* packedWeight );
  void calcSidelobeCancellerU_f( unsigned fbinX, const gsl_vector_complex* wa );
  void setSidelobeCanceller_f( unsigned fbinX, gsl_vector_complex* wl_f ){
    gsl_vector_complex_memcpy( _wl[fbinX], wl_f );
  }
  bool writeFIRCoeff( const String& fn, unsigned winType );

public:
  void setQuiescentVector( unsigned fbinX, gsl_vector_complex *wq_f, bool isGSC=false );
  void setQuiescentVectorAll( gsl_complex z, bool isGSC=false );
  void calcBlockingMatrix( unsigned fbinX );

  unsigned NC(){return(_NC);}
  bool     isHalfBandShift(){return(_halfBandShift);}
  unsigned fftLen(){return(_fftLen);}
  unsigned chanN(){return(_chanN);}

  gsl_vector_complex* wq_f( unsigned fbinX ){
    return _wq[fbinX];
  }

  gsl_vector_complex* wl_f( unsigned fbinX ){
    return _wl[fbinX];
  }

  gsl_vector_complex** wq(){
    return (_wq);
  }
  gsl_matrix_complex** B(){
    return (_B);
  }
  gsl_vector_complex** wa(){
    return (_wa);
  }
  gsl_vector_complex** arrayManifold(){
    return (_ta);
  }
  gsl_vector_complex** CSDs(){
    return _CSDs;
  }
  gsl_vector_complex* wp1(){
    return _wp1;
  }

  void setTimeAlignment();

private:
  void _allocWeights();
  void _freeWeights();
};

typedef refcount_ptr<beamformerWeights>     beamformerWeightsPtr;


// ----- definition for class `SubbandBeamformer' -----
// 

class SubbandBeamformer : public VectorComplexFeatureStream {
 public:
  SubbandBeamformer(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandBeamformer");
  ~SubbandBeamformer();

  bool isEnd(){return _endOfSamples;}
  unsigned fftLen() const { return _fftLen; }
  unsigned fftLen2() const { return _fftLen2; }
  unsigned chanN() const { return _channelList.size(); }
  virtual unsigned dim() const { return chanN();}

  const gsl_vector_complex* snapShotArray_f(unsigned fbinX){ return ( _snapShotArray->getSnapShot(fbinX) ); }
  virtual SnapShotArrayPtr getSnapShotArray(){return(_snapShotArray);}

  void setChannel(VectorComplexFeatureStreamPtr& chan);
  virtual void clearChannel();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();

protected:
  typedef list<VectorComplexFeatureStreamPtr>	_ChannelList;
  typedef _ChannelList::iterator		_ChannelIterator;

  SnapShotArrayPtr				_snapShotArray;
  unsigned					_fftLen;
  unsigned					_fftLen2;
  _ChannelList					_channelList;
  bool						_halfBandShift;
};

// ----- definition for class `SubbandDS' -----
// 

class SubbandDS : public SubbandBeamformer {
 public:
  SubbandDS(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandDS");
  ~SubbandDS();

  virtual void clearChannel();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();

  virtual void calcArrayManifoldVectors(double sampleRate, const gsl_vector* delays);
  virtual void calcArrayManifoldVectors2(double sampleRate, const gsl_vector* delaysT, const gsl_vector* delaysJ );
  virtual void calcArrayManifoldVectorsN(double sampleRate, const gsl_vector* delaysT, const gsl_matrix* delaysJ, unsigned NC=2 );
  virtual beamformerWeights* getBeamformerWeightObject(unsigned srcX=0){
    return _bfWeightV[srcX];
  }
  virtual const gsl_vector_complex *getWeights( unsigned fbinX ){return _bfWeightV[0]->wq_f(fbinX);}

protected:
  void _allocImage();
  void _allocBFWeight( int nSrc, int NC );

protected:
  vector<beamformerWeights *>                   _bfWeightV; // weights of a beamformer per source.
};

#define NO_PROCESSING 0x00
#define SCALING_MDP   0x01
class SubbandGSC : public SubbandDS {
public:
  SubbandGSC(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandGSC")
    : SubbandDS( fftLen, halfBandShift, nm ),_normalizeWeight(false){}
  ~SubbandGSC();

  bool isEnd(){return _endOfSamples;}  
  void normalizeWeight( bool flag ){
    _normalizeWeight = flag;
  }
  void setQuiescentWeights_f( unsigned fbinX, const gsl_vector_complex * srcWq );
  void setActiveWeights_f( unsigned fbinX, const gsl_vector* packedWeight );
  void zeroActiveWeights();
  void calcGSCWeights( double sampleRate, const gsl_vector* delaysT );
  void calcGSCWeights2( double sampleRate, const gsl_vector* delaysT, const gsl_vector* delaysJ );
  void calcGSCWeightsN( double sampleRate, const gsl_vector* delaysT, const gsl_matrix* delaysJ, unsigned NC=2  );
  virtual const gsl_vector_complex* next(int frameX = -5);
  bool writeFIRCoeff( const String& fn, unsigned winType=1 );
  gsl_matrix_complex* getBlockingMatrix(unsigned srcX, unsigned fbinX){
    gsl_matrix_complex** B = _bfWeightV[srcX]->B();
    return (B[fbinX]);
  }
protected:
  bool _normalizeWeight;
};

/**   
   @class SubbandGSCRLS
   @brief implementation of recursive least squares of a GSC 
   @usage
   1. calcGSCWeights()
   2. initPrecisionMatrix() or setPrecisionMatrix()
   3. updateActiveWeightVecotrs( false ) if you want to stop adapting the active weight vectors.
   @note notations are  based on Van Trees, "Optimum Array Processing", pp. 766-767.
 */
typedef enum {
  CONSTANT_NORM           = 0x01,
  THRESHOLD_LIMITATION    = 0x02,
  NO_QUADRATIC_CONSTRAINT = 0x00
} QuadraticConstraintType;

// ----- definition for class `SubbandGSCRLS' -----
// 

class SubbandGSCRLS : public SubbandGSC {
 public:
  SubbandGSCRLS(unsigned fftLen = 512, bool halfBandShift = false, float myu = 0.9, float sigma2=0.0, const String& nm = "SubbandGSCRLS");
  ~SubbandGSCRLS();
  
  void initPrecisionMatrix( float sigma2 = 0.01 );
  void setPrecisionMatrix(unsigned fbinX, gsl_matrix_complex *Pz);
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  void updateActiveWeightVecotrs( bool flag ){ _isActiveWeightVectorUpdated = flag; }
  void setQuadraticConstraint( float alpha, int qctype=1 ){ _alpha=alpha; _qctype=(QuadraticConstraintType)qctype; }
private:
  void _updateActiveWeightVector1( int frameX ); /* the case of the half band shift = True */
  void _updateActiveWeightVector2( int frameX ); /* the case of the half band shift = False */
  bool _allocImage4SubbandGSCRLS();
  void _freeImage4SubbandGSCRLS();

private:
  gsl_vector_complex** _gz; /* Gain vectors */
  gsl_matrix_complex** _Pz; /* Precision matrices */
  gsl_vector_complex* _Zf;  /* output of the blocking matrix at each frequency */
  gsl_vector_complex* _wa;
  float  _myu;              /* Exponential factor for the covariance matrix */
  float* _diagonalWeights;
  float  _alpha;            /* Weight for the quadratic constraint*/
  QuadraticConstraintType _qctype;
  bool _isActiveWeightVectorUpdated;
private:
  /* work space for updating active weight vectors */
  gsl_vector_complex* _PzH_Z;
  gsl_matrix_complex* _I;
  gsl_matrix_complex* _mat1;
};

// ----- definition for class `SubbandMMI' -----
//

class SubbandMMI : public SubbandDS {
public:
  SubbandMMI(unsigned fftLen = 512, bool halfBandShift = false, unsigned targetSourceX=0, unsigned nSource=2, int pfType=0, double alpha=0.9, const String& nm = "SubbandMMI")
    : SubbandDS( fftLen, halfBandShift, nm )
  {
    _targetSourceX = targetSourceX;
    _nSource = nSource;
    _useBinaryMask = false;
    _binaryMaskType = 0;
    _interferenceOutputs = NULL;
    _avgOutput = NULL;
    _pfType = pfType;
    _alpha  = alpha;
  }
  ~SubbandMMI();

  void useBinaryMask( double avgFactor=-1.0, unsigned fwidth=1, unsigned type=0 )
  {
    _avgFactor = avgFactor;
    _fwidth = fwidth;
    _useBinaryMask = true;
    _binaryMaskType = type;
    
    _interferenceOutputs = new gsl_vector_complex*[_nSource];
    for (unsigned i = 0; i < _nSource; i++)
      _interferenceOutputs[i] = gsl_vector_complex_alloc(_fftLen);
    
    _avgOutput = gsl_vector_complex_alloc(_fftLen);
    for (unsigned fbinX = 0; fbinX < _fftLen; fbinX++)
      gsl_vector_complex_set( _avgOutput, fbinX, gsl_complex_rect( 0.0, 0.0 ) );
  }
  void calcWeights(  double sampleRate, const gsl_matrix* delays );
  void calcWeightsN( double sampleRate, const gsl_matrix* delays, unsigned NC=2 );
  void setActiveWeights_f(   unsigned fbinX, const gsl_matrix* packedWeights, int option=0 );
  void setHiActiveWeights_f( unsigned fbinX, const gsl_vector* pkdWa, const gsl_vector* pkdwb, int option=0 );
  virtual const gsl_vector_complex* next(int frameX = -5);

private:
  void calcInterferenceOutputs();
  void binaryMasking( gsl_vector_complex** interferenceOutputs, unsigned targetSourceX, gsl_vector_complex* output );
private:
  unsigned                                      _nSource;       // the number of sound sources
  unsigned                                      _targetSourceX; // the n-th source will be emphasized
  bool                                          _useBinaryMask; // true if you use a binary mask
  unsigned                                      _binaryMaskType;// 0:use GSC's outputs, 1:use outputs of the upper branch.
  gsl_vector_complex**                          _interferenceOutputs;
  gsl_vector_complex*                           _avgOutput;
  double                                        _avgFactor;
  unsigned                                      _fwidth;
  int                                           _pfType; 
  double                                        _alpha;
};


// ----- definition for class `SubbandMVDR' -----
//

/**
   @class SubbandMVDR 

   @usage
   1. setChannel()
   2. calcArrayManifoldVectors(), calcArrayManifoldVectors2() or calcArrayManifoldVectorsN().
   3. setNoiseSpatialSpectralMatrix() or setDiffuseNoiseModel()
   4. calcMVDRWeights()
 */
class SubbandMVDR : public SubbandDS {
 public:
  /**
     @brief 
     @param int fftLen[in]
     @param bool halfBandShift[in]
   */  
  SubbandMVDR(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandMVDR");
  ~SubbandMVDR();

  virtual void clearChannel();
  bool isEnd(){return _endOfSamples;}
  bool calcMVDRWeights( double sampleRate, double dThreshold = 1.0E-8, bool calcInverseMatrix = true );
  const gsl_vector_complex* getMVDRWeights( unsigned fbinX ){
    return _wmvdr[fbinX];
  }
  virtual const gsl_vector_complex* next(int frameX = -5);

  const gsl_matrix_complex *getNoiseSpatialSpectralMatrix( unsigned fbinX ){
    return _R[fbinX];
  }
  bool setNoiseSpatialSpectralMatrix( unsigned fbinX, gsl_matrix_complex* Rnn );
  bool setDiffuseNoiseModel( const gsl_matrix* micPositions, double sampleRate, double sspeed = 343740.0 ); /* micPositions[][x,y,z] */
  void setAllLevelsOfDiagonalLoading( float diagonalWeight );
  void setLevelOfDiagonalLoading( unsigned fbinX, float diagonalWeight );
  /**
     @brief Divide each non-diagonal elemnt by 1 + myu instead of diagonal loading. myu can be interpreted as the ratio of the sensor noise to the ambient noise power.
     @param float myu[in]
   */
  void divideAllNonDiagonalElements( float myu ){

    for(unsigned fbinX=0;fbinX<=_fftLen/2;fbinX++){
      divideNonDiagonalElements( fbinX, myu );
    }
  }
  void divideNonDiagonalElements( unsigned fbinX, float myu ){

    for ( size_t chanX=0; chanX<chanN(); chanX++ ){
      for ( size_t chanY=0; chanY<chanN(); chanY++ ){
	if( chanX != chanY ){
	  gsl_complex Rxy = gsl_matrix_complex_get( _R[fbinX], chanX, chanY );
	  gsl_matrix_complex_set( _R[fbinX], chanX, chanY, gsl_complex_div( Rxy, gsl_complex_rect( (1.0+myu), 0.0 ) ) );
	}
      }
    }
  }
  gsl_matrix_complex**  getNoiseSpatialSpectralMatrix(){
    return _R;
  }

protected:
  gsl_matrix_complex**                           _R; /* Noise spatial spectral matrices */
  gsl_matrix_complex**                           _invR;
  gsl_vector_complex**                           _wmvdr;
  float*                                         _diagonalWeights;
};

// ----- definition for class `SubbandMVDRGSC' -----
//

/**
   @class SubbandMVDRGSC 

   @usage
   1. setChannel()
   2. calcArrayManifoldVectors(), calcArrayManifoldVectors2() or calcArrayManifoldVectorsN().
   3. setNoiseSpatialSpectralMatrix() or setDiffuseNoiseModel()
   4. calcMVDRWeights()
   5. calcBlockingMatrix1() or calcBlockingMatrix2()
   6. setActiveWeights_f()
 */
class SubbandMVDRGSC : public SubbandMVDR {
 public:
  /**
     @brief 
     @param int fftLen[in]
     @param bool halfBandShift[in]
   */  
  SubbandMVDRGSC(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandMVDR");
  ~SubbandMVDRGSC();

  void setActiveWeights_f( unsigned fbinX, const gsl_vector* packedWeight );
  void zeroActiveWeights();
  bool calcBlockingMatrix1( double sampleRate, const gsl_vector* delaysT );
  bool calcBlockingMatrix2( );

  void upgradeBlockingMatrix();
  const gsl_vector_complex* blockingMatrixOutput( int outChanX=0 );

  virtual const gsl_vector_complex* next(int frameX = -5);
protected:
  bool _normalizeWeight;
};

typedef Inherit<SubbandBeamformer, VectorComplexFeatureStreamPtr> SubbandBeamformerPtr;
typedef Inherit<SubbandDS, SubbandBeamformerPtr> SubbandDSPtr;
typedef Inherit<SubbandGSC, SubbandDSPtr> SubbandGSCPtr;
typedef Inherit<SubbandGSCRLS, SubbandGSCPtr> SubbandGSCRLSPtr;
typedef Inherit<SubbandMMI, SubbandDSPtr> SubbandMMIPtr;
typedef Inherit<SubbandMVDR, SubbandDSPtr> SubbandMVDRPtr;
typedef Inherit<SubbandMVDRGSC, SubbandMVDRPtr> SubbandMVDRGSCPtr;

// ----- members for class `SubbandOrthogonalizer' -----
//

class SubbandOrthogonalizer : public VectorComplexFeatureStream {
public:
  SubbandOrthogonalizer( SubbandMVDRGSCPtr &beamformer, int outChanX=0, const String& nm = "SubbandOrthogonalizer" );
  
  ~SubbandOrthogonalizer();
  virtual const gsl_vector_complex* next(int frameX = -5);

private:
  SubbandMVDRGSCPtr _beamformer;
  int _outChanX;
  gsl_matrix_complex** _B;
};

typedef Inherit<SubbandOrthogonalizer, VectorComplexFeatureStreamPtr> SubbandOrthogonalizerPtr;

class SubbandBlockingMatrix : public SubbandGSC {
public:
  SubbandBlockingMatrix(unsigned fftLen=512, bool halfBandShift=false, const String& nm = "SubbandBlockingMatrix")
    :SubbandGSC(fftLen, halfBandShift, nm ){;}

  ~SubbandBlockingMatrix();
  virtual const gsl_vector_complex* next(int frameX = -5);
};

// ----- definition for class DOAEstimatorSRPBase' -----
// 
class DOAEstimatorSRPBase {
public:
  DOAEstimatorSRPBase( unsigned nBest, unsigned fbinMax );
  ~DOAEstimatorSRPBase();
  virtual const gsl_vector *getNBestRPs(){ return (const gsl_vector *)_nBestRPs; }
  virtual const gsl_matrix *getNBestDOAs(){ return (const gsl_matrix *)_argMaxDOAs;}
  gsl_matrix *getResponsePowerMatrix(){ return _rpMat;}

protected:
  void clearTable();
  void _getNBestHypothesesFromACCRP();
  void _initAccs();

protected:
  double _widthTheta;
  double _widthPhi;
  double _minTheta;
  double _maxTheta;
  double _minPhi;
  double _maxPhi;
  unsigned _nTheta;
  unsigned _nPhi;
  unsigned _fbinMin;
  unsigned _fbinMax; 
  unsigned _nBest;
  bool   _isTableInitialized;

  gsl_vector *_accRPs;  
  gsl_vector *_nBestRPs;
  gsl_matrix *_argMaxDOAs;  
  vector<gsl_vector_complex **> _svTbl; // [][fftL2+1][_dim]
  gsl_matrix         *_rpMat;

  float _engeryThreshold;
  float _energy;

#define __MBDEBUG__
#ifdef  __MBDEBUG__
  void allocDebugWorkSapce();
#endif /* #ifdef __MBDEBUG__ */
};

// ----- definition for class DOAEstimatorSRPDSBLA' -----
// 
/**
   @brief estimate the direction of arrival based on the maximum steered response power

   @usage
   1. construct an object
   2. set the geometry of the linear array
   3. call next()
 */
class DOAEstimatorSRPDSBLA : 
  public DOAEstimatorSRPBase, public SubbandDS {
public:
  DOAEstimatorSRPDSBLA( unsigned nBest, unsigned sampleRate, unsigned fftLen, const String& nm="DOAEstimatorSRPDSBLAPtr" );
  ~DOAEstimatorSRPDSBLA();
  
  void setArrayGeometry(gsl_vector *positions);
  const gsl_vector_complex* next(int frameX = -5);
  void reset();
  virtual const gsl_vector *getNBestRPs(){ return (const gsl_vector *)_nBestRPs; }
  virtual const gsl_matrix *getNBestDOAs(){ return (const gsl_matrix *)_argMaxDOAs;}

  void setEnergyThreshold(float engeryThreshold){
    _engeryThreshold = engeryThreshold;
  }
  float getEnergy(){return _energy;}

  void setFrequencyRange( unsigned fbinMin, unsigned fbinMax ){ _fbinMin = fbinMin; _fbinMax = fbinMax;}  
  void setSearchParam( double minTheta=-M_PI/2, double maxTheta=M_PI/2, double widthTheta=0.1 ){
    if( minTheta > maxTheta ){
      fprintf(stderr,"Invalid argument\n");
      _minTheta = maxTheta;
      _maxTheta = minTheta;
    }
    else{
      _minTheta = minTheta;
      _maxTheta = maxTheta;
    }
    _minPhi = 0;
    _maxPhi = 1;
    _widthTheta = widthTheta;
    _widthPhi   = 1;  
    clearTable();
  }

  void getFinalNBestHypotheses(){_getNBestHypothesesFromACCRP();}
  void initAccs(){_initAccs();}

protected:
  virtual void   _calcSteeringUnitTable();
  virtual double _calcResponsePower( unsigned uttX );  

private:
  virtual void setLookDirection( int nChan, double theta );

  unsigned    _sampleRate;
  gsl_matrix *_arraygeometry; // [micX][x,y,z]
};

typedef refcount_ptr<DOAEstimatorSRPBase> DOAEstimatorSRPBasePtr;
typedef Inherit<DOAEstimatorSRPDSBLA, SubbandDSPtr> DOAEstimatorSRPDSBLAPtr;

void calcAllDelays(double x, double y, double z, const gsl_matrix* mpos, gsl_vector* delays);

void calcProduct(gsl_vector_complex* synthesisSamples, gsl_matrix_complex* gs_W, gsl_vector_complex* product);

void calcOutputOfGSC( const gsl_vector_complex* snapShot, 
		      gsl_vector_complex* wl_f, gsl_vector_complex* wq_f, 
		      gsl_complex *pYf, bool normalizeWeight=false );

bool pseudoinverse( gsl_matrix_complex *A, gsl_matrix_complex *invA, float dThreshold =  1.0E-8 );
#endif
