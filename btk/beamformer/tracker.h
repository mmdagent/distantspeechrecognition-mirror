//                              -*- C++ -*-
//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.beamforming
//  Purpose: Beamforming and speaker tracking with a spherical microphone array.
//  Author:  John McDonough
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

#ifndef _tracker_
#define _tracker_

#include <stdio.h>
#include <assert.h>
#include <float.h>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_complex_math.h>

#include "common/refcount.h"
#include "common/jexception.h"

#include "stream/stream.h"
#include "beamformer/spectralinfoarray.h"
#include "modulated/modulated.h"
#include "cancelVP/cancelVP.h"

// ----- definition of class 'BaseDecomposition' -----
//
class BaseDecomposition : public Countable {
public:
  class SubbandEntry {
  public:
    SubbandEntry(unsigned subbandX, const gsl_complex& Bkl);

    unsigned subbandX() const { return _subbandX; }
    gsl_complex bkl()   const { return _bkl; }

    void* operator new(size_t sz) { return memoryManager().newElem(); }
    void  operator delete(void* e) { memoryManager().deleteElem(e); }

    static MemoryManager<SubbandEntry>& memoryManager();

  private:
    const unsigned			_subbandX;
    const gsl_complex			_bkl;
  };

  class GreaterThan {
  public:
    bool operator()(SubbandEntry* sbX1, SubbandEntry* sbX2) {
      return (gsl_complex_abs(sbX1->bkl()) > gsl_complex_abs(sbX2->bkl()));
    }
  };

  class Iterator;

  class SubbandList : public Countable {
  public:
    SubbandList(const gsl_vector_complex* bkl, unsigned useSubbandsN = 0);
    ~SubbandList();

    unsigned useSubbandsN() const { return _useSubbandsN; }
    SubbandEntry** subbands() { return _subbands; }

    friend class Iterator;

  private:
    const unsigned			_subbandsN;
    const unsigned			_useSubbandsN;
    SubbandEntry**			_subbands;
  };

  typedef refcountable_ptr<SubbandList> SubbandListPtr;

  class Iterator {
  public:
    Iterator(const SubbandListPtr& subbandList)
      : _subbandX(0), _useSubbandsN(subbandList->useSubbandsN()), _subbands(subbandList->subbands()) { }
    void operator++(int) { _subbandX++; }
    bool more()          { return _subbandX < _useSubbandsN; }
    const SubbandEntry& operator*() { return *(_subbands[_subbandX]); }

  private:
    unsigned				_subbandX;
    const unsigned			_useSubbandsN;
    SubbandEntry**			_subbands;
  };

 public:
  BaseDecomposition(unsigned orderN, unsigned subbandsN, double a, double sampleRate, unsigned useSubbandsN = 0, bool spatial = false);
  ~BaseDecomposition();

  unsigned orderN()         const { return _orderN;         }
  unsigned modesN()         const { return _modesN;         }
  unsigned subbandsN2()     const { return _subbandsN2;     }
  unsigned subbandsN()      const { return _subbandsN;      }
  unsigned useSubbandsN()   const { return _useSubbandsN;   }
  unsigned subbandLengthN() const { return _subbandLengthN; }

  SubbandListPtr subbandList()  { return _subbandList;    }

  virtual void reset();

  static gsl_complex harmonic(int order, int degree, double theta, double phi);
  gsl_complex harmonic(int order, int degree, unsigned channelX) const;
  static gsl_complex harmonicDerivPolarAngle(int order, int degree, double theta, double phi);
  static gsl_complex harmonicDerivAzimuth(int order, int degree, double theta, double phi);
  static gsl_complex modalCoefficient(unsigned order, double ka);
  gsl_complex modalCoefficient(unsigned order, unsigned subbandX) const;
  virtual void transform(const gsl_vector_complex* initial, gsl_vector_complex* transformed) { }
  virtual void estimateBkl(double theta, double phi, const gsl_vector_complex* snapshot, unsigned subbandX) = 0;
  virtual void calculate_gkl(double theta, double phi, unsigned subbandX) = 0;
  virtual const gsl_matrix_complex* linearize(gsl_vector* xk, int frameX) = 0;
  virtual const gsl_vector_complex* predictedObservation(gsl_vector* xk, int frameX) = 0;

  static const unsigned				_StateN;
  static const unsigned				_ChannelsN;
  static const gsl_complex 			_ComplexZero;
  static const gsl_complex 			_ComplexOne;
  static const double				_SpeedOfSound;

protected:
  static gsl_complex _calc_in(int n);
  void _setEigenMikeGeometry();
  gsl_complex _calculate_Gnm(unsigned subbandX, int n, int m, double theta, double phi);
  gsl_complex _calculate_dGnm_dtheta(unsigned subbandX, int n, int m, double theta, double phi);
  static double _calculatePnm(int order, int degree, double theta);
  static double _calculate_dPnm_dtheta(int n, int m, double theta);
  static double _calculateNormalization(int order, int degree);

  const unsigned				_orderN;
  const unsigned				_modesN;
  const unsigned				_subbandsN;
  const unsigned				_subbandsN2;
  const unsigned				_useSubbandsN;
  const unsigned				_subbandLengthN;
  const double					_sampleRate;
  const double					_a;
  gsl_vector_complex**				_bn;
  gsl_vector*					_theta_s;
  gsl_vector*					_phi_s;
  gsl_vector_complex**				_sphericalComponent;
  gsl_vector_complex*				_bkl;
  gsl_vector_complex*				_dbkl_dtheta;
  gsl_vector_complex*				_dbkl_dphi;
  gsl_vector_complex**				_gkl;
  gsl_vector_complex**				_dgkl_dtheta;
  gsl_vector_complex**				_dgkl_dphi;
  gsl_vector_complex*				_vkl;
  gsl_matrix_complex*				_Hbar_k;
  gsl_vector_complex*				_yhat_k;

  SubbandListPtr				_subbandList;
};

typedef refcountable_ptr<BaseDecomposition> BaseDecompositionPtr;


// ----- definition of class 'ModalDecomposition' -----
//
class ModalDecomposition : public BaseDecomposition {
public:
  ModalDecomposition(unsigned orderN, unsigned subbandsN, double a, double sampleRate, unsigned useSubbandsN = 0);
  ~ModalDecomposition() { }

  virtual void estimateBkl(double theta, double phi, const gsl_vector_complex* snapshot, unsigned subbandX);
  virtual void transform(const gsl_vector_complex* initial, gsl_vector_complex* transformed);
  virtual const gsl_matrix_complex* linearize(gsl_vector* xk, int frameX);
  virtual const gsl_vector_complex* predictedObservation(gsl_vector* xk, int frameX);

  virtual void calculate_gkl(double theta, double phi, unsigned subbandX);
};

typedef Inherit<ModalDecomposition, BaseDecompositionPtr>	ModalDecompositionPtr;


// ----- definition of class 'SpatialDecomposition' -----
//
class SpatialDecomposition : public BaseDecomposition {
public:
  SpatialDecomposition(unsigned orderN, unsigned subbandsN, double a, double sampleRate, unsigned useSubbandsN = 0);
  ~SpatialDecomposition() { }

  virtual void estimateBkl(double theta, double phi, const gsl_vector_complex* snapshot, unsigned subbandX);
  virtual const gsl_matrix_complex* linearize(gsl_vector* xk, int frameX);
  virtual const gsl_vector_complex* predictedObservation(gsl_vector* xk, int frameX);

  virtual void calculate_gkl(double theta, double phi, unsigned subbandX);
};

typedef Inherit<SpatialDecomposition, BaseDecompositionPtr>	SpatialDecompositionPtr;


// ----- definition of class 'BaseSphericalArrayTracker' -----
//
class BaseSphericalArrayTracker : public VectorFloatFeatureStream {
  typedef list<VectorComplexFeatureStreamPtr>	_ChannelList;
  typedef _ChannelList::iterator		_ChannelIterator;
  typedef BaseDecomposition::SubbandEntry	SubbandEntry;
  typedef BaseDecomposition::SubbandListPtr	SubbandListPtr;
  typedef BaseDecomposition::Iterator		Iterator;

public:
  BaseSphericalArrayTracker(BaseDecompositionPtr& baseDecomposition, double sigma2_u = 10.0, double sigma2_v = 10.0, double sigma2_init = 10.0,
			    unsigned maxLocalN = 1, const String& nm = "BaseSphericalArrayTracker");
  ~BaseSphericalArrayTracker();

  void setChannel(VectorComplexFeatureStreamPtr& chan);
  void setV(const gsl_matrix_complex* Vk, unsigned subbandX);
  unsigned chanN() const { return _channelList.size(); }

  virtual const gsl_vector_float* next(int frameX = -5) = 0;

  virtual void reset();
  void nextSpeaker();
  void setInitialPosition(double theta, double phi);

protected:
  static void _printMatrix(const gsl_matrix_complex* mat);
  static void _printMatrix(const gsl_matrix* mat);
  static void _printVector(const gsl_vector_complex* vec);
  static void _printVector(const gsl_vector* vec);

  static double _calcGivensRotation(double v1, double v2, double& c, double& s);
  static void _applyGivensRotation(double v1, double v2, double c, double s, double& v1p, double& v2p);

  void _allocImage();
  void _update(const gsl_matrix_complex* Hbar_k, const gsl_vector_complex* yhat_k, const SubbandListPtr& subbandList);
  void _lowerTriangularize();
  void _copyPosition();
  void _checkPhysicalConstraints();
  double _calculateResidual();
  void _realify(const gsl_matrix_complex* Hbar_k, const gsl_vector_complex* yhat_k);
  void _realifyResidual();

  static const unsigned 			_StateN;
  static const gsl_complex 			_ComplexZero;
  static const gsl_complex 			_ComplexOne;
  static const double      			_Epsilon;
  static const double				_Tolerance;

  bool						_firstFrame;
  const unsigned				_subbandsN;
  const unsigned				_subbandsN2;
  const unsigned				_useSubbandsN;
  const unsigned 				_modesN;
  const unsigned 				_subbandLengthN;  
  const unsigned 				_observationN;
  const unsigned 				_maxLocalN;
  bool						_endOfSamples;
  const double 					_sigma_init;

  SnapShotArrayPtr				_snapShotArray;
  BaseDecompositionPtr				_baseDecomposition;
  _ChannelList					_channelList;

  // these quantities are stored as Cholesky factors
  gsl_matrix*					_U;
  gsl_matrix*					_V;
  gsl_matrix*					_K_k_k1;

  // work space for state estimate update
  gsl_matrix*					_prearray;
  gsl_vector_complex*				_vk;
  gsl_matrix*					_Hbar_k;
  gsl_vector*					_yhat_k;
  gsl_vector*					_correction;
  gsl_vector*					_position;
  gsl_vector*					_eta_i;
  gsl_vector*					_delta;
  gsl_vector_complex*				_residual;
  gsl_vector*					_residual_real;
  gsl_vector*					_scratch;
};

typedef Inherit<BaseSphericalArrayTracker, VectorFloatFeatureStreamPtr> BaseSphericalArrayTrackerPtr;


// ----- definition of class 'ModalSphericalArrayTracker' -----
//
class ModalSphericalArrayTracker : public BaseSphericalArrayTracker {
  typedef list<VectorComplexFeatureStreamPtr>	_ChannelList;
  typedef _ChannelList::iterator		_ChannelIterator;

public:
  ModalSphericalArrayTracker(ModalDecompositionPtr& modalDecomposition, double sigma2_u = 10.0, double sigma2_v = 10.0, double sigma2_init = 10.0,
			     unsigned maxLocalN = 1, const String& nm = "ModalSphericalArrayTracker");
  ~ModalSphericalArrayTracker() { }

  const gsl_vector_float* next(int frameX);
};

typedef Inherit<ModalSphericalArrayTracker, VectorFloatFeatureStreamPtr> ModalSphericalArrayTrackerPtr;


// ----- definition of class 'SpatialSphericalArrayTracker' -----
//
class SpatialSphericalArrayTracker : public BaseSphericalArrayTracker {
  typedef list<VectorComplexFeatureStreamPtr>	_ChannelList;
  typedef _ChannelList::iterator		_ChannelIterator;

public:
 SpatialSphericalArrayTracker(SpatialDecompositionPtr& spatialDecomposition, double sigma2_u = 10.0, double sigma2_v = 10.0, double sigma2_init = 10.0,
			       unsigned maxLocalN = 1, const String& nm = "SpatialSphericalArrayTracker");
  ~SpatialSphericalArrayTracker() { }

  const gsl_vector_float* next(int frameX);
};

typedef Inherit<SpatialSphericalArrayTracker, VectorFloatFeatureStreamPtr> SpatialSphericalArrayTrackerPtr;


// ----- definition of class 'PlaneWaveSimulator' -----
//
class PlaneWaveSimulator : public VectorComplexFeatureStream {
public:
  PlaneWaveSimulator(const VectorComplexFeatureStreamPtr& source, ModalDecompositionPtr& modalDecomposition,
		     unsigned channelX, double theta, double phi, const String& nm = "Plane Wave Simulator");
  ~PlaneWaveSimulator();

  const gsl_vector_complex* next(int frameX = -5);

  virtual void reset();

private:
  static const gsl_complex			_ComplexZero;

  const unsigned				_subbandsN;
  const unsigned				_subbandsN2;
  const	unsigned				_channelX;
  const double					_theta;
  const double					_phi;

  VectorComplexFeatureStreamPtr			_source;
  ModalDecompositionPtr				_modalDecomposition;
  gsl_vector_complex*				_subbandCoefficients;
};

typedef Inherit<PlaneWaveSimulator, VectorComplexFeatureStreamPtr> PlaneWaveSimulatorPtr;

#ifdef HAVE_GSL_V11X

int gsl_matrix_complex_add( gsl_matrix_complex *am, gsl_matrix_complex *bm )
{
  if( am->size1 != bm->size1 ){
    fprintf(stderr,"Dimension error \n",am->size1, bm->size1 );
    return 0;
  }

  if( am->size2 != bm->size2 ){
    fprintf(stderr,"Dimension error \n",am->size2, bm->size2 );
    return 0;
  }
  
  for(size_t i=0;i<am->size1;i++){
    for(size_t j=0;j<am->size2;j++){
      gsl_complex val = gsl_complex_add( gsl_matrix_complex_get(am,i,j), gsl_matrix_complex_get(bm,i,j) );
      gsl_matrix_complex_set(am,i,j,val);
    }
  }

  return 1;
}

int gsl_vector_complex_add( gsl_vector_complex *av, gsl_vector_complex *bv )
{
  if( av->size != bv->size ){
    fprintf(stderr,"Dimension error \n",av->size, bv->size);
    return 0;
  }
  
  for(size_t i=0;i<av->size;i++){
    gsl_complex val = gsl_complex_add( gsl_vector_complex_get(av,i), gsl_vector_complex_get(bv,i) );
    gsl_vector_complex_set(av,i,val);
  }

  return 1;
}


int gsl_vector_complex_sub( const gsl_vector_complex *av, const gsl_vector_complex *bv )
{
  if( av->size != bv->size ){
    fprintf(stderr,"Dimension error \n",av->size, bv->size);
    return 0;
  }
  
  for(size_t i=0;i<av->size;i++){
    gsl_complex val = gsl_complex_sub( gsl_vector_complex_get(av,i), gsl_vector_complex_get(bv,i) );
    gsl_vector_complex_set((gsl_vector_complex*)av,i,val);
  }

  return 1;
}

#endif /* End of HAVE_GSL_V11X */

#endif
