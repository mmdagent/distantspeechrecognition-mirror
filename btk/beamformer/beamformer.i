//                           -*- C++ -*-
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

%module(package="btk") beamformer

%{
#include "beamformer/beamformer.h"
#include "beamformer/taylorseries.h"
#include "beamformer/modalBeamformer.h"
#include "beamformer/tracker.h"
#include <numpy/arrayobject.h>
#include "stream/pyStream.h"
#include "postfilter/postfilter.h"
#include "postfilter/spectralsubtraction.h"
%}

%init {
  // NumPy needs to set up callback functions
  import_array();
}

//typedef int size_t;

%include gsl/gsl_types.h
%include jexception.i
%include complex.i
%include matrix.i
%include vector.i

%pythoncode %{
import btk
from btk import stream
oldimport = """
%}
%import stream/stream.i
%pythoncode %{
"""
%}

%import postfilter/postfilter.i

// ----- definition for class `SnapShotArray' -----
// 
%ignore SnapShotArray;
class SnapShotArray {
 public:
  SnapShotArray(unsigned fftLn, unsigned nChn);
  virtual ~SnapShotArray();

  const gsl_vector_complex* getSnapShot(unsigned fbinX) const;

  void newSample(const gsl_vector_complex* samp, unsigned chanX) const;

  unsigned fftLen() const;
  unsigned nChan()  const;

  virtual void update();
  virtual void zero();
};

class SnapShotArrayPtr {
 public:
  %extend {
    SnapShotArrayPtr(unsigned fftLn, unsigned nChn) {
      return new SnapShotArrayPtr(new SnapShotArray(fftLn, nChn));
    }
  }

  SnapShotArray* operator->();
};

// ----- definition for class `SpectralMatrixArray' -----
// 
%ignore SpectralMatrixArray;
class SpectralMatrixArray : public SnapShotArray {
public:
  SpectralMatrixArray(unsigned fftLn, unsigned nChn, double forgetFact = 0.95);
  virtual ~SpectralMatrixArray();

  gsl_matrix_complex* getSpecMatrix(unsigned idx) const;
  virtual void update();
  virtual void zero();
};

class SpectralMatrixArrayPtr {
 public:
  %extend {
    SpectralMatrixArrayPtr(unsigned fftLn, unsigned nChn, double forgetFact = 0.95) {
      return new SpectralMatrixArrayPtr(new SpectralMatrixArray(fftLn, nChn, forgetFact));
    }
  }

  SpectralMatrixArray* operator->();
};

// ----- definition for class `SubbandBeamformer' -----
// 
%ignore SubbandBeamformer;
class SubbandBeamformer : public VectorComplexFeatureStream {
 public:
  SubbandBeamformer(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandBeamformer");
  ~SubbandBeamformer();

  bool isEnd();
  unsigned fftLen();
  unsigned chanN();
  virtual unsigned dim();

  const gsl_vector_complex* snapShotArray_f(unsigned fbinX);
  SnapShotArrayPtr getSnapShotArray();

  void setChannel(VectorComplexFeatureStreamPtr& chan);
  virtual void clearChannel();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
};

class SubbandBeamformerPtr : public VectorComplexFeatureStreamPtr {
 public:
  %extend {
    SubbandBeamformerPtr(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandBeamformer") {
       return new SubbandBeamformerPtr(new SubbandBeamformer(fftLen, halfBandShift, nm));
    }
    
    SubbandBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SubbandBeamformer* operator->();
};

// ----- definition for class `SubbandDS' -----
//
%ignore SubbandDS;
class SubbandDS : public SubbandBeamformer {
public:
  SubbandDS(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandDS");
  ~SubbandDS();

  virtual void clearChannel();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  
  virtual void calcArrayManifoldVectors(double sampleRate, const gsl_vector* delays);
  virtual void calcArrayManifoldVectors2(double sampleRate, const gsl_vector* delaysT, const gsl_vector* delaysJ );
  virtual void calcArrayManifoldVectorsN(double sampleRate, const gsl_vector* delaysT, const gsl_matrix* delaysJ, int NC );
  virtual const gsl_vector_complex *getWeights( unsigned fbinX );
};

class SubbandDSPtr : public SubbandBeamformerPtr {
 public:
  %extend {
    SubbandDSPtr(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandDS") {
      return new SubbandDSPtr(new SubbandDS(fftLen, halfBandShift, nm));
    }

    SubbandDSPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SubbandDS* operator->();
};

// ----- definition for class `SubbandGSC' -----
//
%ignore SubbandGSC;
class SubbandGSC : public SubbandDS {
 public:
 SubbandGSC(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandGSC");
  ~SubbandGSC();

  bool isEnd();
  virtual const gsl_vector_complex* next(int frameX = -5);

  void normalizeWeight( bool flag );
  void calcGSCWeights( double sampleRate, const gsl_vector* delaysT );
  void calcGSCWeights2( double sampleRate, const gsl_vector* delaysT, const gsl_vector* delaysJ );
  void calcGSCWeightsN( double sampleRate, const gsl_vector* delaysT, const gsl_matrix* delaysJ, int NC=2  );
  void setActiveWeights_f( int fbinX, const gsl_vector* packedWeight );
  void zeroActiveWeights();
  bool writeFIRCoeff( const String& fn, unsigned winType=1 );
  gsl_matrix_complex* getBlockingMatrix(unsigned srcX, unsigned fbinX);

  virtual void reset();
};

class SubbandGSCPtr : public SubbandDSPtr {
 public:
  %extend {
    SubbandGSCPtr(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandGSC") {
      return new SubbandGSCPtr(new SubbandGSC(fftLen, halfBandShift, nm));
    }

    SubbandGSCPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SubbandGSC* operator->();
};

// ----- definition for class `SubbandGSCRLS' -----
//
%ignore SubbandGSCRLS;
class SubbandGSCRLS : public SubbandGSC {
 public:
 SubbandGSCRLS(unsigned fftLen = 512, bool halfBandShift = false, float myu = 0.9, float sigma2=0.0, const String& nm = "SubbandGSCRLS");
  ~SubbandGSCRLS();

  void updateActiveWeightVecotrs( bool flag );
  void initPrecisionMatrix( float sigma2 = 0.01 );
  void setPrecisionMatrix(unsigned fbinX, gsl_matrix_complex *Pz);
  void setQuadraticConstraint( float alpha, int qctype=1 );
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
};

class SubbandGSCRLSPtr : public SubbandGSCPtr {
 public:
  %extend {
    SubbandGSCRLSPtr ( unsigned fftLen = 512, bool halfBandShift = false, float myu = 0.9, float sigma2=0.01, const String& nm = "SubbandGSCRLS") {
      return new SubbandGSCRLSPtr(new SubbandGSCRLS(fftLen, halfBandShift, myu, sigma2, nm));
    }

    SubbandGSCRLSPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SubbandGSCRLS* operator->();
};

// ----- definition for class `SubbandMMI' -----
//
%ignore SubbandMMI;
class SubbandMMI : public SubbandDS {
 public:
  SubbandMMI(unsigned fftLen = 512, bool halfBandShift = false, unsigned targetSourceX=0, unsigned nSource=2, int pfType=0, double alpha=0.9, const String& nm = "SubbandMMI");
  ~SubbandMMI();

  virtual const gsl_vector_complex* next(int frameX = -5);

  void useBinaryMask( double avgFactor=-1.0, unsigned fwidth=1, unsigned type=0  );
  void calcWeights(  double sampleRate, const gsl_matrix* delays );
  void calcWeightsN( double sampleRate, const gsl_matrix* delays, unsigned NC=2 );
  void setActiveWeights_f( unsigned fbinX, const gsl_matrix* packedWeights, int option=0 );
  void setHiActiveWeights_f( unsigned fbinX, const gsl_vector* pkdWa, const gsl_vector* pkdwb, int option=0 );

  virtual void reset();
};

class SubbandMMIPtr : public SubbandDSPtr {
 public:
  %extend {
   SubbandMMIPtr(unsigned fftLen = 512, bool halfBandShift = false, unsigned targetSourceX=0, unsigned nSource=2, int pfType=0, double alpha=0.9, const String& nm = "SubbandMMI") {
      return new SubbandMMIPtr(new SubbandMMI(fftLen, halfBandShift, targetSourceX, nSource, pfType, alpha, nm));
    }

    SubbandMMIPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SubbandMMI* operator->();
};

void calcAllDelays(double x, double y, double z, const gsl_matrix* mpos, gsl_vector* delays);

void calcProduct(gsl_vector_complex* synthesisSamples, gsl_matrix_complex* gs_W, gsl_vector_complex* product);

// ----- definition for class `SubbandMVDR' -----
//
%ignore SubbandMVDR;
class SubbandMVDR : public SubbandDS {
 public:
   SubbandMVDR(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandMVDR");
  ~SubbandMVDR();

  virtual void clearChannel();
  bool isEnd();
  virtual const gsl_vector_complex* next(int frameX = -5);

  bool calcMVDRWeights( double sampleRate, double dThreshold = 1.0E-8, bool calcInverseMatrix=true );
  const gsl_vector_complex* getMVDRWeights( unsigned fbinX );

  const gsl_matrix_complex *getNoiseSpatialSpectralMatrix( unsigned fbinX );
  bool setNoiseSpatialSpectralMatrix( unsigned fbinX, gsl_matrix_complex* Rnn );
  bool setDiffuseNoiseModel( const gsl_matrix* micPositions, double sampleRate, double sspeed = 343740.0 );
  void setAllLevelsOfDiagonalLoading( float diagonalWeight );
  void setLevelOfDiagonalLoading( unsigned fbinX, float diagonalWeight );
  void divideAllNonDiagonalElements( float myu );
  void divideNonDiagonalElements( unsigned fbinX, float myu );
  gsl_matrix_complex**  getNoiseSpatialSpectralMatrix();
  virtual void reset();
};

class SubbandMVDRPtr : public SubbandDSPtr {
 public:
  %extend {
    SubbandMVDRPtr(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandMVDR"){
      return new SubbandMVDRPtr(new SubbandMVDR( fftLen, halfBandShift, nm ));
    }

    SubbandMVDRPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SubbandMVDR* operator->();
};

// ----- definition for class `SubbandMVDRGSC' -----
//
%ignore SubbandMVDRGSC;
class SubbandMVDRGSC : public SubbandMVDR {
 public:
  SubbandMVDRGSC(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandMVDRGSC");
  ~SubbandMVDRGSC();

  void setActiveWeights_f( unsigned fbinX, const gsl_vector* packedWeight );
  void zeroActiveWeights();
  bool calcBlockingMatrix1( double sampleRate, const gsl_vector* delaysT );
  bool calcBlockingMatrix2( );
  void upgradeBlockingMatrix();

  virtual const gsl_vector_complex* next(int frameX = -5);
};

class SubbandMVDRGSCPtr : public SubbandMVDRPtr {
 public:
  %extend {
    SubbandMVDRGSCPtr(unsigned fftLen = 512, bool halfBandShift = false, const String& nm = "SubbandMVDRGSC"){
      return new SubbandMVDRGSCPtr(new SubbandMVDRGSC( fftLen, halfBandShift, nm ));
    }

    SubbandMVDRGSCPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SubbandMVDRGSC* operator->();
};

// ----- definition for class `SubbandOrthogonalizer' -----
//
%ignore SubbandOrthogonalizer;
class SubbandOrthogonalizer : public VectorComplexFeatureStream {
 public:
  SubbandOrthogonalizer( SubbandMVDRGSCPtr &beamformer, int outChanX=0, const String& nm = "SubbandOrthogonalizer");
  ~SubbandOrthogonalizer();
  
  virtual const gsl_vector_complex* next(int frameX = -5);
};

class SubbandOrthogonalizerPtr : public VectorComplexFeatureStreamPtr {
public:
  %extend {
    SubbandOrthogonalizerPtr( SubbandMVDRGSCPtr &beamformer, int outChanX=0, const String& nm = "SubbandOrthogonalizer"){
      return new SubbandOrthogonalizerPtr(new SubbandOrthogonalizer( beamformer, outChanX, nm ));
    }

    SubbandOrthogonalizerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SubbandOrthogonalizer* operator->();
};


gsl_complex modeAmplitude( int order, double ka );

// ----- definition for class `ModeAmplitudeCalculator' -----
//
%ignore ModeAmplitudeCalculator;
class   ModeAmplitudeCalculator {
public:
  ModeAmplitudeCalculator( int order, double minKa=0.01, double maxKa=20, double wid=0.01 );
  ~ModeAmplitudeCalculator();
  gsl_vector_complex *get();
};

class ModeAmplitudeCalculatorPtr {
public:
  %extend {
    ModeAmplitudeCalculatorPtr( int order, double minKa=0.01, double maxKa=20, double wid=0.01 ){
      return new ModeAmplitudeCalculatorPtr(new ModeAmplitudeCalculator( order, minKa, maxKa, wid ));
    }
  }

  ModeAmplitudeCalculator* operator->();
};

// ----- definition for class `EigenBeamformer' -----
//
%ignore EigenBeamformer;
class EigenBeamformer : public  SubbandDS {
public:
  EigenBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=8, bool normalizeWeight=false, const String& nm = "EigenBeamformer");
  ~EigenBeamformer();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  virtual unsigned dim();

  void setSigma2(float simga2);
  void setWeightGain(float wgain);
  void setEigenMikeGeometry();
  void setArrayGeometry( double a,  gsl_vector *theta_s, gsl_vector *phi_s );
  void setLookDirection( double theta, double phi );
  const gsl_matrix_complex *getModeAmplitudes();
  gsl_vector *getArrayGeometry( int type );
  gsl_matrix *getBeamPattern( unsigned fbinX, double theta = 0, double phi = 0,
			      double minTheta=-M_PI, double maxTheta=M_PI, double minPhi=-M_PI, double maxPhi=M_PI, 
			      double widthTheta=0.1, double widthPhi=0.1 );
  virtual SnapShotArrayPtr getSnapShotArray();
  virtual SnapShotArrayPtr getSnapShotArray2();
  const gsl_matrix_complex *getBlockingMatrix(unsigned fbinX, unsigned unitX=0 );
};

class EigenBeamformerPtr : public  SubbandDSPtr {
public:
  %extend {
    EigenBeamformerPtr( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=8, bool normalizeWeight=false, const String& nm = "EigenBeamformer"){
      return new EigenBeamformerPtr(new EigenBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm ));
    }

    EigenBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  EigenBeamformer* operator->();
};

// ----- definition for class `DOAEstimatorSRPBase' -----
//
%ignore DOAEstimatorSRPBase;
class DOAEstimatorSRPBase {
public:
  DOAEstimatorSRPBase( unsigned nBest, unsigned fbinMax );
  ~DOAEstimatorSRPBase();
  const gsl_vector *getNBestRPs();
  const gsl_matrix *getNBestDOAs();
  gsl_matrix *getResponsePowerMatrix();
};

class DOAEstimatorSRPBasePtr {
public:
  %extend {
    DOAEstimatorSRPBasePtr( unsigned nBest, unsigned fbinMax ){
      return new DOAEstimatorSRPBasePtr( new DOAEstimatorSRPBase( nBest, fbinMax ) );
    }
  }
  DOAEstimatorSRPBase* operator->();
};

// ----- definition for class `DOAEstimatorSRPDSBLA' -----
//
%ignore DOAEstimatorSRPDSBLA;
class DOAEstimatorSRPDSBLA : public SubbandDS {
public:
  DOAEstimatorSRPDSBLA( unsigned nBest, unsigned sampleRate, unsigned fftLen, const String& nm="DOAEstimatorSRPDSBLAPtr" );
  ~DOAEstimatorSRPDSBLA();

  const gsl_vector *getNBestRPs();
  const gsl_matrix *getNBestDOAs();
  void setArrayGeometry(gsl_vector *positions);
  const gsl_vector_complex* next(int frameX = -5);
  void reset();

  void setEnergyThreshold(float engeryThreshold);
  float getEnergy();
  void setFrequencyRange( unsigned fbinMin, unsigned fbinMax );
  void setSearchParam( double minTheta=0, double maxTheta=M_PI/2, double widthTheta=0.1 );
  void getFinalNBestHypotheses();
  void initAccs();
};

class DOAEstimatorSRPDSBLAPtr : public SubbandDSPtr {
public:
  %extend {
    DOAEstimatorSRPDSBLAPtr( unsigned nBest, unsigned sampleRate, unsigned fftLen, const String& nm="DOAEstimatorSRPDSBLAPtr" ){
      return new DOAEstimatorSRPDSBLAPtr( new DOAEstimatorSRPDSBLA( nBest, sampleRate, fftLen, nm ) );
    }
    DOAEstimatorSRPDSBLAPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  DOAEstimatorSRPDSBLA* operator->();
};

// ----- definition for class `DOAEstimatorSRPEB' -----
//
%ignore DOAEstimatorSRPEB;
class DOAEstimatorSRPEB : public EigenBeamformer {
public:
  DOAEstimatorSRPEB( unsigned nBest, unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=8, bool normalizeWeight=false, const String& nm = "DirectionEstimatorSRPMB");
  ~DOAEstimatorSRPEB();

  const gsl_vector_complex* next(int frameX = -5);
  void reset();

  void setEnergyThreshold(float engeryThreshold);
  float getEnergy();
  void setFrequencyRange( unsigned fbinMin, unsigned fbinMax );
  void setSearchParam( double minTheta=0, double maxTheta=M_PI, double minPhi=-M_PI, double maxPhi=M_PI, double widthTheta=0.1, double widthPhi=0.1 );
  const gsl_vector *getNBestRPs();
  const gsl_matrix *getNBestDOAs();
  gsl_matrix *getResponsePowerMatrix();
  void getFinalNBestHypotheses();
  void initAccs();
};

class DOAEstimatorSRPEBPtr : public EigenBeamformerPtr {
public:
  %extend {
    DOAEstimatorSRPEBPtr( unsigned nBest, unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=8, bool normalizeWeight=false, const String& nm = "DirectionEstimatorSRPMB" ){
      return new DOAEstimatorSRPEBPtr( new DOAEstimatorSRPEB( nBest, sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm ) );
    }
    DOAEstimatorSRPEBPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  DOAEstimatorSRPEB* operator->();
};

// ----- definition for class `SphericalDSBeamformer' -----
//
%ignore SphericalDSBeamformer;
class SphericalDSBeamformer : public EigenBeamformer {
public:
  SphericalDSBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, const String& nm = "SphericalDSBeamformer");
  ~SphericalDSBeamformer();
  virtual gsl_vector *calcWNG();
  const gsl_vector_complex* next(int frameX = -5);
  void reset();
};

class SphericalDSBeamformerPtr : public EigenBeamformerPtr {
public:
  %extend {
    SphericalDSBeamformerPtr( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, const String& nm = "SphericalDSBeamformer"){
      return new SphericalDSBeamformerPtr( new SphericalDSBeamformer(sampleRate, fftLen, halfBandShift, NC, maxOrder,  normalizeWeight, nm ) );
    }    
    SphericalDSBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }
  
  SphericalDSBeamformer* operator->();
};

// ----- definition for class `DualSphericalDSDSBeamformer' -----
//
%ignore DualSphericalDSBeamformer;
class DualSphericalDSBeamformer : public SphericalDSBeamformer {
public:
  DualSphericalDSBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, const String& nm = "DualSphericalDSBeamformer");
  ~DualSphericalDSBeamformer();
  virtual SnapShotArrayPtr getSnapShotArray();
};

class DualSphericalDSBeamformerPtr : public SphericalDSBeamformerPtr {
public:
  %extend {
    DualSphericalDSBeamformerPtr( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, const String& nm = "DualSphericalDSBeamformer"){
      return new DualSphericalDSBeamformerPtr( new DualSphericalDSBeamformer(sampleRate, fftLen, halfBandShift, NC, maxOrder,  normalizeWeight, nm ) );
    }    
    DualSphericalDSBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }
  
  DualSphericalDSBeamformer* operator->();
};

// ----- definition for class DOAEstimatorSRPSphDSB' -----
// 
%ignore DOAEstimatorSRPSphDSB;
class DOAEstimatorSRPSphDSB : public SphericalDSBeamformer {
public:
  DOAEstimatorSRPSphDSB( unsigned nBest, unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, const String& nm = "DOAEstimatorSRPSphDSB");
  ~DOAEstimatorSRPSphDSB();
  const gsl_vector_complex* next(int frameX = -5);
  void reset();
  void setEnergyThreshold(float engeryThreshold);
  float getEnergy();
  void setFrequencyRange( unsigned fbinMin, unsigned fbinMax );
  void setSearchParam( double minTheta=0, double maxTheta=M_PI, double minPhi=-M_PI, double maxPhi=M_PI, double widthTheta=0.1, double widthPhi=0.1 );
  const gsl_vector *getNBestRPs();
  const gsl_matrix *getNBestDOAs();
  gsl_matrix *getResponsePowerMatrix();
  void getFinalNBestHypotheses();
  void initAccs();
};

class DOAEstimatorSRPSphDSBPtr : public SphericalDSBeamformerPtr {
public:
  %extend {
    DOAEstimatorSRPSphDSBPtr( unsigned nBest, unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=8, bool normalizeWeight=false, const String& nm = "DirectionEstimatorSRPMB" ){
      return new DOAEstimatorSRPSphDSBPtr( new DOAEstimatorSRPSphDSB( nBest, sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm ) );
    }
    DOAEstimatorSRPSphDSBPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  DOAEstimatorSRPSphDSB* operator->();
};

// ----- definition for class `SphericalHWNCBeamformer' -----
//
%ignore SphericalHWNCBeamformer;
class SphericalHWNCBeamformer : public EigenBeamformer {
public:
  SphericalHWNCBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, float ratio=0.1, const String& nm = "SphericalHWNCBeamformer");
  ~SphericalHWNCBeamformer();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual gsl_vector *calcWNG();
  void setWNG( double ratio);
  void reset();
};

class SphericalHWNCBeamformerPtr : public EigenBeamformerPtr {
public:
  %extend {
    SphericalHWNCBeamformerPtr( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, float ratio=0.1, const String& nm = "SphericalHWNCBeamformer"){
      return new SphericalHWNCBeamformerPtr( new SphericalHWNCBeamformer(sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, ratio, nm ) );
    }    
    SphericalHWNCBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }
  
  SphericalHWNCBeamformer* operator->();
};

// ----- definition for class `SphericalGSCBeamformer' -----
// 
%ignore SphericalGSCBeamformer;
class SphericalGSCBeamformer : public SphericalDSBeamformer {
public:
  SphericalGSCBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "SphericalGSCBeamformer");
  ~SphericalGSCBeamformer();

  const gsl_vector_complex* next(int frameX = -5);
  void reset();

  void setLookDirection( double theta, double phi );
  void setActiveWeights_f( unsigned fbinX, const gsl_vector* packedWeight );
};

class SphericalGSCBeamformerPtr : public SphericalDSBeamformerPtr {
public:
  %extend {
    SphericalGSCBeamformerPtr( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "SphericalGSCBeamformer" ){
      return new SphericalGSCBeamformerPtr( new SphericalGSCBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm ) );
    }
    SphericalGSCBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SphericalGSCBeamformer* operator->();
};

// ----- definition for class `SphericalHWNCGSCBeamformer' -----
// 
%ignore SphericalHWNCGSCBeamformer;
class SphericalHWNCGSCBeamformer : public SphericalHWNCBeamformer {
public:
  SphericalHWNCGSCBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, float ratio=1.0, const String& nm = "SphericalHWNCGSCBeamformer");
  ~SphericalHWNCGSCBeamformer();

  const gsl_vector_complex* next(int frameX = -5);
  void reset();
  
  void setLookDirection( double theta, double phi );
  void setActiveWeights_f( unsigned fbinX, const gsl_vector* packedWeight );
};

class SphericalHWNCGSCBeamformerPtr : public SphericalHWNCBeamformerPtr {
public:
  %extend {
    SphericalHWNCGSCBeamformerPtr( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, float ratio=1.0, const String& nm = "SphericalHWNCGSCBeamformer" ){
      return new SphericalHWNCGSCBeamformerPtr( new SphericalHWNCGSCBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, ratio, nm ) );
    }
    SphericalHWNCGSCBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SphericalHWNCGSCBeamformer* operator->();
};

// ----- definition for class `DualSphericalGSCBeamformer' -----
//
%ignore DualSphericalGSCBeamformer;
class DualSphericalGSCBeamformer : public SphericalGSCBeamformer {
public:
  DualSphericalGSCBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, const String& nm = "DualSphericalGSCBeamformer");
  ~DualSphericalGSCBeamformer();
  virtual SnapShotArrayPtr getSnapShotArray();
};

class DualSphericalGSCBeamformerPtr : public SphericalGSCBeamformerPtr {
public:
  %extend {
    DualSphericalGSCBeamformerPtr( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, const String& nm = "DualSphericalGSCBeamformer"){
      return new DualSphericalGSCBeamformerPtr( new DualSphericalGSCBeamformer(sampleRate, fftLen, halfBandShift, NC, maxOrder,  normalizeWeight, nm ) );
    }    
    DualSphericalGSCBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }
  
  DualSphericalGSCBeamformer* operator->();
};

// ----- definition for class `SphericalMOENBeamformer' -----
// 
%ignore SphericalMOENBeamformer;
class SphericalMOENBeamformer : public SphericalDSBeamformer {
public:
  SphericalMOENBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "SphericalMOENBeamformer");
  ~SphericalMOENBeamformer();

  const gsl_vector_complex* next(int frameX = -5);
  void reset();
  void fixTerms( bool flag );
  void setLevelOfDiagonalLoading( unsigned fbinX, float diagonalWeight );
  gsl_matrix *getBeamPattern( unsigned fbinX, double theta = 0, double phi = 0,
			      double minTheta=-M_PI, double maxTheta=M_PI, double minPhi=-M_PI, double maxPhi=M_PI, 
			      double widthTheta=0.1, double widthPhi=0.1 );
};

class SphericalMOENBeamformerPtr : public SphericalDSBeamformerPtr {
public:
  %extend {
    SphericalMOENBeamformerPtr( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=4, bool normalizeWeight=false, const String& nm = "SphericalMOENBeamformer" ){
      return new SphericalMOENBeamformerPtr( new SphericalMOENBeamformer( sampleRate, fftLen, halfBandShift, NC, maxOrder, normalizeWeight, nm ) );
    }    
    SphericalMOENBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SphericalMOENBeamformer* operator->();
};

// ----- definition for class `TaylorSeries' -----
//
%ignore nonamePdf;
class nonamePdf {
 public:
  nonamePdf();
  ~nonamePdf();

  bool loadCoeffDescFile( const String &coefDescfn );
};

class nonamePdfPtr {
 public:
  %extend {
    nonamePdfPtr(){
      return new nonamePdfPtr(new nonamePdf());
    }
  }

  nonamePdf* operator->();
};

%ignore gammaPdf;
class gammaPdf : public nonamePdf {
 public:
  gammaPdf(int numberOfVariate = 2 );
  ~gammaPdf();
  double calcLog( double x, int N );
  double calcDerivative1( double x, int N );
  void  bi();
  void  four();
  void  printCoeff();
};

class gammaPdfPtr : public nonamePdfPtr {
 public:
  %extend {
    gammaPdfPtr(int numberOfVariate = 2){
      return new gammaPdfPtr(new gammaPdf(numberOfVariate));
    }
  }

  gammaPdf* operator->();
};

// ----- definition of class 'ModalDecomposition' -----
//
%ignore ModalDecomposition;
class ModalDecomposition {
 public:
  ModalDecomposition(unsigned orderN, unsigned subbandsN, double a, double sampleRate, unsigned useSubbandsN = 0);
  ~ModalDecomposition();

  gsl_complex harmonic(int order, int degree, double theta, double phi);
  gsl_complex harmonicDerivPolarAngle(int order, int degree, double theta, double phi);
  gsl_complex harmonicDerivAzimuth(int order, int degree, double theta, double phi);
  gsl_complex modalCoefficient(unsigned order, double ka);
  void transform(const gsl_vector_complex* initial, gsl_vector_complex* transformed);
  gsl_complex estimateBkl(double theta, double phi, const gsl_vector_complex* snapshot, int frameX, unsigned subbandX);
  gsl_matrix_complex* linearize(gsl_vector* xk);
};

class ModalDecompositionPtr {
public:
  %extend {
    ModalDecompositionPtr(unsigned orderN, unsigned subbandsN, double a, double sampleRate, unsigned useSubbandsN = 0) {
      return new ModalDecompositionPtr(new ModalDecomposition(orderN, subbandsN, a, sampleRate, useSubbandsN));
    }
  }
};


// ----- definition of class 'SpatialDecomposition' -----
//
%ignore SpatialDecomposition;
class SpatialDecomposition {
 public:
  SpatialDecomposition(unsigned orderN, unsigned subbandsN, double a, double sampleRate, unsigned useSubbandsN = 0);
  ~SpatialDecomposition();

  gsl_complex harmonic(int order, int degree, double theta, double phi);
  gsl_complex harmonicDerivPolarAngle(int order, int degree, double theta, double phi);
  gsl_complex harmonicDerivAzimuth(int order, int degree, double theta, double phi);
  gsl_complex modalCoefficient(unsigned order, double ka);
  void transform(const gsl_vector_complex* initial, gsl_vector_complex* transformed);
  gsl_complex estimateBkl(double theta, double phi, const gsl_vector_complex* snapshot, int frameX, unsigned subbandX);
  gsl_matrix_complex* linearize(gsl_vector* xk);
};

class SpatialDecompositionPtr {
public:
  %extend {
    SpatialDecompositionPtr(unsigned orderN, unsigned subbandsN, double a, double sampleRate, unsigned useSubbandsN = 0) {
      return new SpatialDecompositionPtr(new SpatialDecomposition(orderN, subbandsN, a, sampleRate, useSubbandsN));
    }
  }
};


// ----- definition of class 'ModalSphericalArrayTracker' -----
//
%ignore ModalSphericalArrayTracker;
class ModalSphericalArrayTracker {
 public:
  ModalSphericalArrayTracker(ModalDecompositionPtr& modalDecomposition, double sigma2_u = 10.0, double sigma2_v = 10.0, double sigma2_init = 10.0,
			unsigned maxLocalN = 1, const String& nm = "ModalSphericalArrayTracker");
  ~ModalSphericalArrayTracker();

  void setChannel(VectorComplexFeatureStreamPtr& chan);
  void setV(const gsl_matrix_complex* Vk, unsigned subbandX);
  unsigned chanN() const { return _channelList.size(); }

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

  void nextSpeaker();
  //void setInitialPostion(double theta, double phi);
};

class ModalSphericalArrayTrackerPtr {
public:
  %extend {
    ModalSphericalArrayTrackerPtr(ModalDecompositionPtr& modalDecomposition, double sigma2_u = 10.0, double sigma2_v = 10.0, double sigma2_init = 10.0,
				  unsigned maxLocalN = 1, const String& nm = "ModalSphericalArrayTracker") {
      return new ModalSphericalArrayTrackerPtr(new ModalSphericalArrayTracker(modalDecomposition, sigma2_u, sigma2_v, sigma2_init, maxLocalN, nm));
    }

    ModalSphericalArrayTrackerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  ModalSphericalArrayTracker* operator->();
};


// ----- definition of class 'SpatialSphericalArrayTracker' -----
//
%ignore SpatialSphericalArrayTracker;
class SpatialSphericalArrayTracker {
 public:
  SpatialSphericalArrayTracker(SpatialDecompositionPtr& spatialDecomposition, double sigma2_u = 10.0, double sigma2_v = 10.0, double sigma2_init = 10.0,
			       unsigned maxLocalN = 1, const String& nm = "SpatialSphericalArrayTracker");
  ~SpatialSphericalArrayTracker();

  void setChannel(VectorComplexFeatureStreamPtr& chan);
  void setV(const gsl_matrix_complex* Vk, unsigned subbandX);
  unsigned chanN() const { return _channelList.size(); }

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

  void nextSpeaker();
  //void setInitialPosition(double theta, double phi);
};

class SpatialSphericalArrayTrackerPtr {
public:
  %extend {
    SpatialSphericalArrayTrackerPtr(SpatialDecompositionPtr& spatialDecomposition, double sigma2_u = 10.0, double sigma2_v = 10.0, double sigma2_init = 10.0,
			     unsigned maxLocalN = 1, const String& nm = "SpatialSphericalArrayTracker") {
      return new SpatialSphericalArrayTrackerPtr(new SpatialSphericalArrayTracker(spatialDecomposition, sigma2_u, sigma2_v, sigma2_init, maxLocalN, nm));
    }

    SpatialSphericalArrayTrackerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SpatialSphericalArrayTracker* operator->();
};


// ----- definition of class 'PlaneWaveSimulator' -----
//
%ignore PlaneWaveSimulator;
class PlaneWaveSimulator : public VectorComplexFeatureStream {
 public:
  PlaneWaveSimulator(const VectorComplexFeatureStreamPtr& source, ModalDecompositionPtr& modalDecomposition,
		     unsigned channelX, double theta, double phi, const String& nm = "Plane Wave Simulator");
  ~PlaneWaveSimulator();

  virtual const gsl_complex_float* next(int frameX = -5);

  virtual void reset();
};

class PlaneWaveSimulatorPtr : public VectorComplexFeatureStreamPtr {
public:
  %extend {
    PlaneWaveSimulatorPtr(const VectorComplexFeatureStreamPtr& source, ModalDecompositionPtr& modalDecomposition,
			  unsigned channelX, double theta, double phi, const String& nm = "Plane Wave Simulator") {
      return new PlaneWaveSimulatorPtr(new PlaneWaveSimulator(source, modalDecomposition, channelX, theta, phi, nm));
    }

    PlaneWaveSimulatorPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  PlaneWaveSimulator* operator->();
};


// ----- definition for class `SphericalSpatialDSBeamformer' -----
//
%ignore SphericalSpatialDSBeamformer;
class SphericalSpatialDSBeamformer : public SphericalDSBeamformer {
public:
  SphericalSpatialDSBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, const String& nm = "SphericalSpatialDSBeamformer");
  ~SphericalSpatialDSBeamformer();
  // virtual void _calcWeights(unsigned fbinX, gsl_vector_complex *weights);
  // virtual bool _allocSteeringUnit( int unitN = 1);
  virtual const gsl_vector_complex* next(int frameX = -5);
  void reset();
};

class SphericalSpatialDSBeamformerPtr : public SphericalDSBeamformerPtr {
public:
  %extend {
    SphericalSpatialDSBeamformerPtr( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=3, bool normalizeWeight=false, const String& nm = "SphericalSpatialDSBeamformer"){
      return new SphericalSpatialDSBeamformerPtr( new SphericalSpatialDSBeamformer(sampleRate, fftLen, halfBandShift, NC, maxOrder,  normalizeWeight, nm ) );
    }    
    SphericalSpatialDSBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }
  
  SphericalSpatialDSBeamformer* operator->();
};


// ----- definition for class `SphericalSpatialHWNCBeamformer' -----
//
%ignore SphericalSpatialHWNCBeamformer;
class SphericalSpatialHWNCBeamformer : public SphericalHWNCBeamformer {
public:
  SphericalSpatialHWNCBeamformer( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=3, double sigmaSI2=1.0, bool normalizeWeight=false, float ratio=0.1, const String& nm = "SphericalSpatialHWNCBeamformer");
  ~SphericalSpatialHWNCBeamformer();
  virtual const gsl_vector_complex* next(int frameX = -5);
  // virtual gsl_vector *calcWNG();
  // virtual void setWNG( double ratio);
  void reset();
};

class SphericalSpatialHWNCBeamformerPtr : public SphericalHWNCBeamformerPtr {
public:
  %extend {
    SphericalSpatialHWNCBeamformerPtr( unsigned sampleRate, unsigned fftLen = 512, bool halfBandShift = false, unsigned NC=1, unsigned maxOrder=3, double sigmaSI2=1.0, bool normalizeWeight=false, float ratio=0.1, const String& nm = "SphericalSpatialHWNCBeamformer"){
      return new SphericalSpatialHWNCBeamformerPtr( new SphericalSpatialHWNCBeamformer(sampleRate, fftLen, halfBandShift, NC, maxOrder, sigmaSI2, normalizeWeight, ratio, nm ) );
    }
    SphericalSpatialHWNCBeamformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }
  
  SphericalSpatialHWNCBeamformer* operator->();
};
