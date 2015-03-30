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

#ifndef _modalBeamformer_
#define _modalBeamformer_

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
#include "beamformer/beamformer.h"


// ----- definition for class `ModeAmplitudeCalculator' -----
//
gsl_complex modeAmplitude( int order, double ka );

class ModeAmplitudeCalculator {
public:
  ModeAmplitudeCalculator( int order, double minKa=0.01, double maxKa=20, double wid=0.01):
    _minKa(minKa),_maxKa(maxKa),_wid(wid),_modeAmplitude(NULL)
  {
    float sizef = ( _maxKa - _minKa ) / _wid;
    int size = (int)(sizef + 0.5);
    unsigned idx = 0;
    _modeAmplitude = gsl_vector_complex_alloc( size );

    //fprintf(stderr,"[%d]\n",size);
    for(double ka=minKa;idx<size;ka+=wid,idx++){
      gsl_complex val = modeAmplitude( order, ka );
      gsl_vector_complex_set( _modeAmplitude, idx, val );
      //fprintf(stderr,"[%d] = b_n (%e) = %e\n", idx, ka, GSL_REAL(val) );
    }
  }
  ~ModeAmplitudeCalculator()
  {
    if(_modeAmplitude!=NULL) 
      gsl_vector_complex_free(_modeAmplitude);
  }
  gsl_vector_complex *get()
  {
    return _modeAmplitude;
  }

private:
  gsl_vector_complex *_modeAmplitude;
  double _minKa;
  double _maxKa;
  double _wid;
};

typedef refcount_ptr<ModeAmplitudeCalculator> ModeAmplitudeCalculatorPtr;

// ----- definition for class `EigenBeamformer' -----
//

/**
   @class EigenBeamformer
   @brief This beamformer is implemented based on Meyer and Elko's ICASSP paper. 
          In Boaz Rafaely's paper, this method is referred to as the phase-mode beamformer
   @usage 
   1) construct this object, bf = EigenBeamformer(...)
   2) set the radious of the spherical array with bf.setEigenMikeGeometry() or bf.setArrayGeometry(..).
   3) set the look direction with bf.setLookDirection(..)
   4) process each block with bf.next() until it hits the end
*/
class EigenBeamformer : public  SubbandDS {
public:
  EigenBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "EigenBeamformer");
  ~EigenBeamformer();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  virtual void _calcWeights( unsigned fbinX, gsl_vector_complex *weights );
  virtual unsigned dim() const { return _dim;}

  void setSigma2(float sigma2){_sigma2=sigma2;}
  void setWeightGain(float wgain){_wgain=wgain;}
  void setEigenMikeGeometry();
  void setArrayGeometry( double a,  gsl_vector *theta_s, gsl_vector *phi_s );
  virtual void setLookDirection( double theta, double phi );
  
  const gsl_matrix_complex *getModeAmplitudes();
  gsl_vector *getArrayGeometry( int type ); // type==0 -> theta, type==1 -> phi
  virtual gsl_matrix *getBeamPattern( unsigned fbinX, double theta = 0, double phi = 0,
				      double minTheta=-M_PI, double maxTheta=M_PI, double minPhi=-M_PI, double maxPhi=M_PI, 
				      double widthTheta=0.1, double widthPhi=0.1 );
  /**
     @brief obtain the spherical transformation coefficients at each frame
     @return spherical harmonics transformation coefficients at the current frame
   */
  virtual SnapShotArrayPtr getSnapShotArray(){return(_sphericalTransformSnapShotArray);}  
  virtual SnapShotArrayPtr getSnapShotArray2(){return(_snapShotArray);}  

  const gsl_matrix_complex *getBlockingMatrix(unsigned fbinX, unsigned unitX=0 ){
    gsl_matrix_complex** B = _bfWeightV[unitX]->B();
    return (const gsl_matrix_complex *)B[fbinX];
  }

protected:
  virtual bool _calcSphericalHarmonicsAtEachPosition( gsl_vector *theta_s, gsl_vector *phi_s ); // need to be tested!!
  virtual bool _calcSteeringUnit(  int unitX=0, bool isGSC=false );
  virtual bool _allocSteeringUnit( int unitN=1 );
  void _allocImage( bool flag=true );
  bool _calcModeAmplitudes();
  
  unsigned _sampleRate;
  unsigned _NC;
  unsigned _maxOrder;
  unsigned _dim; // the number of the spherical harmonics transformation coefficients
  bool _areWeightsNormalized;
  gsl_matrix_complex *_modeAmplitudes; // [_maxOrder] the mode amplitudes.
  gsl_vector_complex *_F; // Spherical Transform coefficients [_dim]
  gsl_vector_complex **_sh_s; // Conjugate of spherical harmonics at each sensor position [_dim][nChan]: Y_n^{m*}
  SnapShotArrayPtr _sphericalTransformSnapShotArray; // for compatibility with a post-filtering object

  double _theta; // look direction 
  double _phi;   // look direction 

  double _a;     // the radius of the rigid sphere.
  gsl_vector *_theta_s; // sensor positions
  gsl_vector *_phi_s;   // sensor positions
  gsl_matrix *_beamPattern;
  gsl_vector *_WNG; // white noise gain
  float _wgain; //
  float _sigma2; // dialog loading
};

typedef Inherit<EigenBeamformer,  SubbandDSPtr> EigenBeamformerPtr;

float calcEnergy( SnapShotArrayPtr snapShotArray, unsigned fbinMin, unsigned fbinMax, unsigned fftLen2, bool  halfBandShift=false );


// ----- definition for class DOAEstimatorSRPEB' -----
// 

/**
   @class DOAEstimatorSRPEB
   @brief estimate the direction of arrival based on the maximum steered response power
   @usage 
   1) construct this object, doaEstimator = DOAEstimatorSRPEB(...)
   2) set the radious of the spherical array, doaEstimator.setEigenMikeGeometry() or doaEstimator.setArrayGeometry(..).
   3) process each block, doaEstimator.next() 
   4) get the N-best hypotheses at the current instantaneous frame through doaEstimator.getNBestDOAs()
   5) do doaEstimator.getFinalNBestHypotheses() after a static segment is processed.
      You can then obtain the averaged N-best hypotheses of the static segment with doaEstimator.getNBestDOAs().
*/
class DOAEstimatorSRPEB : 
  public DOAEstimatorSRPBase, public EigenBeamformer {
  public:
  DOAEstimatorSRPEB( unsigned nBest, unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "DirectionEstimatorSRPBase");
  ~DOAEstimatorSRPEB();

  const gsl_vector_complex* next(int frameX = -5);
  void reset();

  void setEnergyThreshold(float engeryThreshold){
    _engeryThreshold = engeryThreshold;
  }
  float getEnergy(){return _energy;}

  void setFrequencyRange( unsigned fbinMin, unsigned fbinMax ){ _fbinMin = fbinMin; _fbinMax = fbinMax;}  
  void setSearchParam( double minTheta=0, double maxTheta=M_PI, double minPhi=-M_PI, double maxPhi=M_PI, double widthTheta=0.1, double widthPhi=0.1 ){
    _minTheta = minTheta;
    _maxTheta = maxTheta;
    _minPhi = minPhi;
    _maxPhi = maxPhi;
    _widthTheta = widthTheta;
    _widthPhi   = widthPhi;  
    clearTable();
  }

  void getFinalNBestHypotheses(){_getNBestHypothesesFromACCRP();}
  void initAccs(){_initAccs();}

protected:
  virtual void   _calcSteeringUnitTable();
  virtual double _calcResponsePower( unsigned uttX );  
};

typedef Inherit<DOAEstimatorSRPEB, EigenBeamformerPtr> DOAEstimatorSRPEBPtr;

// ----- definition for class `SphericalDSBeamformer' -----
// 

/**
   @class SphericalDSBeamformer
   @usage 
   1) construct this object, mb = SphericalDSBeamformer(...)
   2) set the radious of the spherical array mb.setArrayGeometry(..) or 
   3) set the look direction mb.setLookDirection()
   4) process each block mb.next()
   @note this implementation is based on Boaz Rafaely's letter,
   "Phase-Mode versus Delay-and-Sum Spherical Microphone Array", IEEE Signal Processing Letters, vol. 12, Oct. 2005.
*/
class SphericalDSBeamformer : public EigenBeamformer {
public:
  SphericalDSBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "SphericalDSBeamformer");
  ~SphericalDSBeamformer();
  virtual gsl_vector *calcWNG();

protected:
  virtual void _calcWeights( unsigned fbinX, gsl_vector_complex *weights );
  virtual bool _calcSphericalHarmonicsAtEachPosition( gsl_vector *theta_s, gsl_vector *phi_s );
};

typedef Inherit<SphericalDSBeamformer, EigenBeamformerPtr> SphericalDSBeamformerPtr;

// ----- definition for class `DualSphericalDSBeamformer' -----
// 

/**
   @class DualSphericalDSBeamformer
   @usage 
   1) construct this object, mb = SphericalDSBeamformer(...)
   2) set the radious of the spherical array mb.setArrayGeometry(..) or 
   3) set the look direction mb.setLookDirection()
   4) process each block mb.next()
   @note In addition to  SphericalDSBeamformer, this class has an object of the *normal* D&S beamformer
*/
class DualSphericalDSBeamformer : public SphericalDSBeamformer {
public:
  DualSphericalDSBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "SphericalDSBeamformer");
  ~DualSphericalDSBeamformer();

  virtual SnapShotArrayPtr getSnapShotArray(){return(_snapShotArray);}  
  virtual beamformerWeights* getBeamformerWeightObject(unsigned srcX=0){
    return _bfWeightV2[srcX];
  }

protected:
  virtual void _calcWeights( unsigned fbinX, gsl_vector_complex *weights );
  virtual bool _allocSteeringUnit( int unitN=1 );

  vector<beamformerWeights *>                   _bfWeightV2; // weights of a normal D&S beamformer.
};

typedef Inherit<DualSphericalDSBeamformer, SphericalDSBeamformerPtr> DualSphericalDSBeamformerPtr;

// ----- definition for class DOAEstimatorSRPPSphDSB' -----
// 

class DOAEstimatorSRPSphDSB : public DOAEstimatorSRPBase, public SphericalDSBeamformer {
public:
  DOAEstimatorSRPSphDSB( unsigned nBest, unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "DOAEstimatorSRPPSphDSB" );
  ~DOAEstimatorSRPSphDSB();
  const gsl_vector_complex* next(int frameX = -5);
  void reset();

  void setEnergyThreshold(float engeryThreshold){
    _engeryThreshold = engeryThreshold;
  }
  float getEnergy(){return _energy;}

  void setFrequencyRange( unsigned fbinMin, unsigned fbinMax ){ _fbinMin = fbinMin; _fbinMax = fbinMax;}  
  void setSearchParam( double minTheta=0, double maxTheta=M_PI, double minPhi=-M_PI, double maxPhi=M_PI, double widthTheta=0.1, double widthPhi=0.1 ){
    _minTheta = minTheta;
    _maxTheta = maxTheta;
    _minPhi = minPhi;
    _maxPhi = maxPhi;
    _widthTheta = widthTheta;
    _widthPhi   = widthPhi;  
    clearTable();
  }

  const gsl_vector *getNBestRPs(){ return (const gsl_vector *)_nBestRPs; }
  const gsl_matrix *getNBestDOAs(){ return (const gsl_matrix *)_argMaxDOAs;}
  gsl_matrix *getResponsePowerMatrix(){ return _rpMat;}
  void getFinalNBestHypotheses(){_getNBestHypothesesFromACCRP();}
  void initAccs(){_initAccs();}

protected:
  virtual void   _calcSteeringUnitTable();
  virtual double _calcResponsePower( unsigned uttX );  
};

typedef Inherit<DOAEstimatorSRPSphDSB, SphericalDSBeamformerPtr> DOAEstimatorSRPSphDSBPtr;

// ----- definition for class `SphericalHWNCBeamformer' -----
// 

/**
   @class SphericalHWNCBeamformer
   @usage 
   1) construct this object, mb = SphericalDSBeamformer(...)
   2) set the radious of the spherical array mb.setArrayGeometry(..) or 
   3) set the look direction mb.setLookDirection()
   4) process each block mb.next()
*/
class SphericalHWNCBeamformer : public EigenBeamformer {
public:
  SphericalHWNCBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, float ratio=1.0, const String& nm = "SphericalHWNCBeamformer");
  ~SphericalHWNCBeamformer();
  virtual gsl_vector *calcWNG();
  virtual void setWNG( double ratio){ _ratio=ratio; calcWNG();}

protected:
  virtual void _calcWeights( unsigned fbinX, gsl_vector_complex *weights );

protected:
  float _ratio;
};

typedef Inherit<SphericalHWNCBeamformer, EigenBeamformerPtr> SphericalHWNCBeamformerPtr;

// ----- definition for class `SphericalGSCBeamformer' -----
// 

/**
   @class SphericalGSCBeamformer
   @usage 
   1) construct this object, mb = SphericalDSBeamformer(...)
   2) set the radious of the spherical array mb.setArrayGeometry(..) or 
   3) set the look direction mb.setLookDirection()
   4) process each block mb.next()
   @note this implementation is based on Boaz Rafaely's letter,
   "Phase-Mode versus Delay-and-Sum Spherical Microphone Array", IEEE Signal Processing Letters, vol. 12, Oct. 2005.
*/
class SphericalGSCBeamformer : public SphericalDSBeamformer {
public:
  SphericalGSCBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "SphericalGSCBeamformer");
  ~SphericalGSCBeamformer();

  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();

  void setLookDirection( double theta, double phi );
  void setActiveWeights_f( unsigned fbinX, const gsl_vector* packedWeight );
};

typedef Inherit<SphericalGSCBeamformer, SphericalDSBeamformerPtr> SphericalGSCBeamformerPtr;

// ----- definition for class `SphericalHWNCGSCBeamformer' -----
// 

/**
   @class SphericalHWNCGSCBeamformer
   @usage 
   1) construct this object, mb = SphericalHWNCGSCBeamformer(...)
   2) set the radious of the spherical array mb.setArrayGeometry(..) or 
   3) set the look direction mb.setLookDirection()
   4) process each block mb.next()
*/
class SphericalHWNCGSCBeamformer : public SphericalHWNCBeamformer {
public:
  SphericalHWNCGSCBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, float ratio=1.0, const String& nm = "SphericalHWNCGSCBeamformer");
  ~SphericalHWNCGSCBeamformer();

  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();

  void setLookDirection( double theta, double phi );
  void setActiveWeights_f( unsigned fbinX, const gsl_vector* packedWeight );
};

typedef Inherit<SphericalHWNCGSCBeamformer, SphericalHWNCBeamformerPtr> SphericalHWNCGSCBeamformerPtr;

// ----- definition for class `DualSphericalGSCBeamformer' -----
// 

/**
   @class DualSphericalGSCBeamformer
   @usage 
   1) construct this object, mb = DualSphericalGSCBeamformer(...)
   2) set the radious of the spherical array mb.setArrayGeometry(..) or 
   3) set the look direction mb.setLookDirection()
   4) process each block mb.next()
   @note In addition to DualSphericalGSCBeamformer, this class has an object of the *normal* D&S beamformer
*/
class DualSphericalGSCBeamformer : public SphericalGSCBeamformer {
public:
  DualSphericalGSCBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "DualSphericalGSCBeamformer");
  ~DualSphericalGSCBeamformer();

  virtual SnapShotArrayPtr getSnapShotArray(){return(_snapShotArray);}  
  virtual beamformerWeights* getBeamformerWeightObject(unsigned srcX=0){
    return _bfWeightV2[srcX];
  }

protected:
  virtual void _calcWeights( unsigned fbinX, gsl_vector_complex *weights );
  virtual bool _allocSteeringUnit( int unitN=1 );

  vector<beamformerWeights *>                   _bfWeightV2; // weights of a normal D&S beamformer.
};

typedef Inherit<DualSphericalGSCBeamformer, SphericalGSCBeamformerPtr> DualSphericalGSCBeamformerPtr;

// ----- definition for class `SphericalMOENBeamformer' -----
// 

/**
   @class SphericalGSCBeamformer
   @usage 
   1) construct this object, mb = SphericalDSBeamformer(...)
   2) set the radious of the spherical array mb.setArrayGeometry(..) or 
   3) set the look direction mb.setLookDirection()
   4) process each block mb.next()
   @note this implementation is based on Z. Li and R. Duraiswami's letter,
   "Flexible and Optimal Design of Spherical Microphone Arrays for Beamforming", IEEE Trans. SAP.
*/
class SphericalMOENBeamformer : public SphericalDSBeamformer {
public:
  SphericalMOENBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "SphericalMOENBeamformer");
  ~SphericalMOENBeamformer();

  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  void fixTerms( bool flag ){ _isTermFixed = flag; }
  void setLevelOfDiagonalLoading( unsigned fbinX, float diagonalWeight );
  virtual SnapShotArrayPtr getSnapShotArray(){return(_snapShotArray);}
  virtual gsl_matrix *getBeamPattern( unsigned fbinX, double theta = 0, double phi = 0,
				      double minTheta=-M_PI, double maxTheta=M_PI, double minPhi=-M_PI, double maxPhi=M_PI, 
				      double widthTheta=0.1, double widthPhi=0.1 );
protected:
  virtual bool _allocSteeringUnit( int unitN=1 );
  virtual void _calcWeights( unsigned fbinX, gsl_vector_complex *weights );
  bool _calcMOENWeights( unsigned fbinX, gsl_vector_complex *weights, double dThreshold = 1.0E-8, bool calcInverseMatrix = true, unsigned unitX=0 );
  
private:
  // _maxOrder == Neff in the Li's paper.
  unsigned             _orderOfBF; // N in the Li's paper.
  float                _CN;
  gsl_matrix_complex** _A; /* _A[fftLen2+1][_dim][nChan]; Coeffcients of the spherical harmonics expansion; See Eq. (31) & (32) */
  gsl_matrix_complex** _fixedW; /* _fixedW[fftLen2+1][nChan][_dim]; [ A^H A + l^2 I ]^{-1} A^H */
  gsl_vector_complex** _BN;     // _BN[fftLen2+1][_dim]
  float*               _diagonalWeights;
  bool                 _isTermFixed;
  float                _dThreshold;
};

typedef Inherit<SphericalMOENBeamformer, SphericalDSBeamformerPtr> SphericalMOENBeamformerPtr;


// ----- definition for class `SphericalSpatialDSBeamformer' -----
// 

/**
   @class SphericalSpatialDSBeamformer
   @usage 
   1) construct this object, mb = SphericalSpatialDSBeamformer(...)
   2) set the radious of the spherical array mb.setArrayGeometry(..) or 
   3) set the look direction mb.setLookDirection()
   4) process each block mb.next()
   @note this implementation is based on Boaz Rafaely's letter,
   "Phase-Mode versus Delay-and-Sum Spherical Microphone Array", IEEE Signal Processing Letters, vol. 12, Oct. 2005.
*/
class SphericalSpatialDSBeamformer : public SphericalDSBeamformer {
public:
  SphericalSpatialDSBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "SphericalSpatialDSBeamformer");
  ~SphericalSpatialDSBeamformer();
  virtual const gsl_vector_complex* next(int frameX = -5);

protected:
  virtual void _calcWeights( unsigned fbinX, gsl_vector_complex *weights );
  virtual bool _allocSteeringUnit( int unitN = 1 );
  virtual bool _calcSteeringUnit( int unitX = 0, bool isGSC = false );
};

typedef Inherit<SphericalSpatialDSBeamformer, SphericalDSBeamformerPtr> SphericalSpatialDSBeamformerPtr;


// ----- definition for class `SphericalSpatialHWNCBeamformer' -----
// 

/**
   @class SphericalSpatialHWNCBeamformer
   @usage 
   1) construct this object, mb = SphericalDSBeamformer(...)
   2) set the radious of the spherical array mb.setArrayGeometry(..) or 
   3) set the look direction mb.setLookDirection()
   4) process each block mb.next()
*/
class SphericalSpatialHWNCBeamformer : public SphericalHWNCBeamformer {
public:
  SphericalSpatialHWNCBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, double sigmaSI2=1.0, bool normalizeWeight=false, float ratio=1.0, const String& nm = "SphericalHWNCBeamformer");
  ~SphericalSpatialHWNCBeamformer();
  virtual const gsl_vector_complex* next(int frameX = -5);

protected:
  virtual void _calcWeights( unsigned fbinX, gsl_vector_complex *weights );
  virtual bool _allocSteeringUnit( int unitN = 1 );
  virtual bool _calcSteeringUnit( int unitX = 0, bool isGSC = false );

private:
  gsl_matrix_complex *_calcDiffuseNoiseModel( unsigned fbinX );
  gsl_matrix_complex **_SigmaSI; // _SigmaSI[fftLen/2+1][chanN]
  double  _dThreshold;
  double  _sigmaSI2;
};

typedef Inherit<SphericalSpatialHWNCBeamformer, SphericalHWNCBeamformerPtr> SphericalSpatialHWNCBeamformerPtr;

#endif
