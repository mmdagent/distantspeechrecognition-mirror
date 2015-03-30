//                              -*- C++ -*-
//
//                              Millennium
//                  Automatic Speech Recognition System
//                                 (asr)
//
//  Module:  asr.train
//  Purpose: Speaker adaptation parameter estimation and conventional
//           ML and discriminative HMM training with state clustering.
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


#ifndef _estimateAdapt_h_
#define _estimateAdapt_h_

using namespace std;

#include <map>
#include <vector>
#include <list>
#include <set>
#include <math.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>

#include "path/distribPath.h"
#include "train/distribTrain.h"
#include "adapt/transform.h"


// ----- definition of class `GaussListTrain' -----
//
class GaussListTrain {
  typedef list< CodebookTrain::GaussDensity >	_GaussListTrain;
  typedef _GaussListTrain::iterator		_GaussListTrainIter;
  typedef _GaussListTrain::const_iterator	_GaussListTrainConstIter;

 public:
  GaussListTrain() { }
  GaussListTrain(CodebookSetTrainPtr& cb, unsigned idx = 0);

  unsigned size() const { return _glist.size(); }
  void add(CodebookTrain::GaussDensity& pdf) { _glist.push_back(pdf); }

  class Iterator;       friend class Iterator;
  class ConstIterator;  friend class ConstIterator;

 private:
  _GaussListTrain	_glist;
};


// ----- definition of class `GaussListTrain::Iterator' -----
//
class GaussListTrain::Iterator {
 public:
  Iterator(GaussListTrain& l)
    : _glist(l._glist), _itr(_glist.begin()) { }

  void operator++(int)  { _itr++; }
  bool more()           { return _itr != _glist.end(); }
  CodebookTrain::GaussDensity& mix()   { return _mix(); }

 protected:
  CodebookTrain::GaussDensity& _mix() { return *_itr; }

 private:
  _GaussListTrain&			_glist;
  _GaussListTrainIter			_itr;
};


// ----- definition of class `GaussListTrain::ConstIterator' -----
//
class GaussListTrain::ConstIterator {
 public:
  ConstIterator(const GaussListTrain& l)
    : _glist(l._glist), _itr(_glist.begin()) { }

  void operator++(int)  { _itr++; }
  bool more()           { return _itr != _glist.end(); }
  const CodebookTrain::GaussDensity& mix() { return _mix(); }
  unsigned size() const { return _glist.size(); }

 protected:
  const CodebookTrain::GaussDensity& _mix() { return *_itr; }

 private:
  const _GaussListTrain&		_glist;
  _GaussListTrainConstIter		_itr;
};

/**
* \defgroup Adaptation Estimating Speaker Adaptation Parameters
* This hierarchy of classes provides the capability to perform speaker-adaptation
* based on one of several linear transformations.
*
* Assume the Gaussian means are to be adapted for a particular
* speaker, denoted by the index \f$ s \f$, prior to speech recognition. The
* adaptation of a single mean is achieved by forming the product
* \f$\hat{\mathbf{\mu}}_m = \mathbf{A} \mathbf{\mu}_m\f$ for some speaker-dependent
* transformation matrix \f$\mathbf{A}\f$.  Very often the transformation is
* assumed to include an additive shift \f$\mathbf{b}\f$ such that
\f[
\hat{\mathbf{\mu}}_m = \mathbf{A} \mathbf{\mu}_m + \mathbf{b}.
\f]
* We shall account for this case, however, by assuming the shift is
* represented by the last column of \f$\mathbf{A}\f$, and by appending a final
* component of unity to \f$\mathbf{\mu}_k\f$.  The technique of estimating the
* components of \f$\mathbf{A}\f$ and \f$\mathbf{b}\f$ directly based on a ML criterion was
* originally proposed by Leggetter and Woodland (1995), who called this technique
* maximum likelihood linear regression. Representing the additive bias by the final component of
* \f$\mathbf{\mu}_k\f$ and suppressing the summation over \f$s\f$
* enables the foregoing to be rewritten as
\f[
        \mathcal{G}(\mathcal{Y} ; \{(\mathbf{A}^{(s)},\mathbf{b}^{(s)})\}, \Lambda) = \frac{1}{2}
        \sum_{s,m} c_m[(\tilde{\mathbf{\mu}}_m^{(s)} - \mathbf{A}^{(s)} \mathbf{\mu}_{m})^T \mathbf{\Sigma}_{m}^{-1}
(\tilde{\mathbf{\mu}}_m^{(s)} - \mathbf{A}^{(s)} \mathbf{\mu}_{m})],
\f]
* where
\f{eqnarray*}
\tilde{\mathbf{\mu}}_m &=& \frac{1}{c_m} \sum_k c_{m,k} \, \mathbf{y}_k, \\
c_m &=& \sum_{m,k} c_{m,k},
\f}
* as before.  If the covariance matrix \f$\mathbf{\Sigma}_m\f$ is again assumed to
* be diagonal, then the above can once more be
* decomposed row by row, and each row can be optimized independently.
* Let \f$\mathbf{a}^T_n\f$ denote the \f$ n \f$th row of \f$\mathbf{A}\f$,
* and let \f$L\f$ denote the length of the acoustic feature.
* Then the optimization function can be rewritten as
\f[
\mathcal{G}(\mathcal{Y} ; \{(\mathbf{A}^{(s)},\mathbf{b}^{(s)})\}, \Lambda) = \sum_{n=0}^{L-1} \mathcal{G}_n(\mathcal{Y}; \mathbf{a}_n, \Lambda),
\f]
* where
\f{equation}
\mathcal{G}_n(\mathcal{Y}; \mathbf{a}_n, \Lambda) = \frac{1}{2} \sum_m \frac{\hat{c}_m}{\sigma_{m,n}^2}
\, (\tilde{\mathbf{\mu}}_m - a_n^T \mathbf{\mu}_m).
\label{eq:mllr-optimization-function-modified}
\f}
* Taking the derivative with respect to \f$ \mathbf{a}_n^T \f$ yields
\f[
\frac{\partial \mathcal{G}_n(\mathcal{Y}; \mathbf{a}_n, \Lambda)}{\partial \mathbf{a}_n} =
\frac{1}{2} \sum_m \frac{\hat{c}_m}{\sigma_{m,n}^2} \, \left[- \tilde{\mu}_{m,n} \, \mathbf{\mu}_m + \left(\mathbf{\mu}_m \mathbf{\mu}_m^T\right)\mathbf{a}_n^T\right].
\f]
* Upon equating the right hand side to zero and
* solving, we find
\f[
\mathbf{a}_n = \tilde{\mathbf{H}}^{-1}_n \tilde{\mathbf{v}}_n,
\f]
* where
\f{eqnarray*}
\tilde{\mathbf{H}}^{-1}_n &=& \sum_m \frac{\hat{c}_m}{\sigma_{m,n}^2} \, \mathbf{\mu}_m \mathbf{\mu}_m^T, \\
\tilde{\mathbf{v}}_n &=& \sum_m \frac{\hat{c}_m}{\sigma_{m,n}^2} \tilde{\mu}_{m,n} \, \mathbf{\mu}_m.
\f}
*
* In the final portion of this section, we consider a different type of
* all-pass transform that shares may of the characteristcs of the RAPT.
* Its chief advantage over the RAPT is its simplicity of form and
* amenability to numerical computation. Regrettably, this simplicity is
* not immediately apparent from the abbreviated presentation given
* here.
*
* Let us begin by defining the sine-log all-pass transform as
\f[
Q(z) = z \, {\rm exp} \, F(z),
\f]
* where
\f{eqnarray*}
  F(z) &=& \sum_{k=1}^K \alpha_k \, F_k(z) \;\, \forall\;\,\alpha_1, \ldots, \alpha_K \in \mathcal{R}, \;\,{\rm and} \\
  F_k(z) &=& j \, \pi \, \sin \left(\frac{k}{j}\log z\right),
\f}
* and \f$K\f$ is the number of free parameters in the transform. The
* designation ``sine-log'' is due to the functional form of \f$F_k(z)\f$. It is
* worth noting that \f$F_k(z)\f$ is single-valued even though \f$\log z\f$ is
* multiple-valued. Moreover, applying the well-known relation,
\f[
        \sin z = \frac{1}{2j} \left(e^{jz} - e^{-jz}\right),
\f]
* provides
\f[
        F_k(z) = \frac{\pi}{2} \left( z^k - z^{-k} \right),
\f]
* which is a more tractable form for computation. As \f$z\f$ traverses the unit circle, \f$Q(z)\f$ winds
* exactly once about the origin, just as the RAPTs considered earlier, which is necessary to ensure
* that spectral content is not doubled or
* tripled.
*
* In order to calculate the coefficients of a transformed cepstral
* sequence in the manner described above, it is first necessary to
* calculate the coefficients \f$q\f$ in the Laurent series expansion of \f$Q\f$;
* this can be done as follows: For \f$F\f$ as above, set
\f[
G(z) = {\rm exp} \, F(z).
\f]
Let \f$g\f$ denote the coefficients of the Laurent series expansion of
\f$G\f$ valid in an annular region including the unit circle. Then,
\f[
  g[n] = \frac{1}{2\pi j} \oint G(z) \, z^{-(n+1)} \, dz.
\f]
* Moreover, the natural exponential admits the series expansion
\f[
        e^z = \sum_{m=0}^\infty \frac{z^m}{m!},
\f]
so that
\f[
G(z) = \sum_{m=0}^\infty \frac{F^m(z)}{m!}.
\f]
* Substituting provides
\f{eqnarray*}
        g[n] &=& \frac{1}{2\pi j} \oint \, \sum_{m=0}^\infty \frac{F^m(z)}{m!} \,
        z^{-(n+1)} \, dz \\
        &=& \sum_{m=0}^\infty \frac{1}{m!} \frac{1}{2\pi j}
        \oint \, F^m(z) \,    z^{-(n+1)} \, dz.
\f}
* The sequence \f$f\f$ of coefficients in the series expansion of \f$F\f$ are
* available by inspection. Letting
* \f$ f^{(m)}\f$ denote the coefficients in the series expansion of \f$ F^m\f$, we have
\f[
        f^{(m)}[n] = \frac{1}{2\pi j} \oint F^m(z) \, z^{-(n+1)} \, dz,
\f]
* and upon substituting this, we find
\f[
        g[n] = \sum_{m=0}^\infty \frac{1}{m!} \, f^{(m)}[n].
\f]
* Moreover, from the Cauchy product it follows
\f[
f^{(m)} = f * f^{(m-1)}~~\forall~~m = 1,2,3,\ldots .
\f]
* These equations imply that \f$Q(z) = z \, G(z)\f$,
* so the desired coefficients are given by
\f[
        q[n] = g[n-1]~~\forall~~n = 0, \pm 1, \pm 2, \ldots .
\f]
*
*/
/*@{*/

/*@{*/

// ----- interface class for estimation of RAPT parameters -----
//
class EstimatorBase {
 protected:
  virtual ~EstimatorBase() { }
  bool bracket(double& ax, double& bx, double& cx);
  double brentsMethod(double ax, double bx, double cx, double& xmin);

  virtual double calcLogLhood(double mu) = 0;

  NaturalVector        paramDerivs;

  static const double BarrierScale;
  static const double MaximumRho;
  static const double RhoEpsilon;

 private:
  static const unsigned MaxItns;
  static const double   Tolerance;
  static const double   ConstGold;
  static const double   Gold;
  static const double   GoldLimit;
  static const double   Tiny;
  static const double   ZEpsilon;

  void _swap(double& a, double& b) {
    double temp = a; a = b; b = temp;
  }
  void _shift(double& a, double& b, double& c, double d) {
    a = b; b = c; c = d;
  }
  double _sign(double a, double b) {
    return (b > 0.0) ? fabs(a) : -fabs(a);
  }

  void bound(double min, double max, double& q);
  virtual void findRange(double rho, double& mmin, double& mmax) = 0;
};

class BLTEstimatorBase : protected EstimatorBase, protected BLTTransformer {
 protected:
  BLTEstimatorBase(UnShrt sbFtSz, UnShrt nSubFt)
    : BLTTransformer(sbFtSz, nSubFt) { }

  virtual void _writeParm(FILE* fp);
  virtual void findRange(double rho, double& min, double& max);
};


// ----- definition of class `RAPTSeriesDerivs' -----
//
typedef enum { NoBias, CepstralFeat, AllFeat } BiasType;

BiasType getBiasType(const char* ctype);

class RAPTSeriesDerivs : protected RAPTTransformer {
 protected:
  // methods for 1st derivative calculations
  void  a_Alpha(double _alpha, LaurentSeries& _sequence);
  void  b_Rho(double _rho, double _theta, LaurentSeries& _sequence,
	      bool _invFlag);
  void  b_Theta(double _rho, double _theta, LaurentSeries& _sequence,
		bool _invFlag);

  // methods for 2nd derivative calculations
  void  a_AlphaAlpha(double _alpha, LaurentSeries& _sequence);
  void  b_RhoRho(double _rho, double _theta, LaurentSeries& _sequence,
		 bool _invFlag);
  void  b_RhoTheta(double _rho, double _theta, LaurentSeries& _sequence,
		   bool _invFlag);
  void  b_ThetaTheta(double _rho, double _theta, LaurentSeries& _sequence,
		     bool _invFlag);
  bool  b_RhoSmallTheta(double _rho, double _theta, LaurentSeries& _sequence,
			bool _invFlag);
  bool  b_ThetaSmallTheta(double _rho, double _theta, LaurentSeries& _sequence,
			  bool _invFlag);
  bool  b_RhoRhoSmallTheta(double _rho, double _theta, LaurentSeries& _sequence,
			   bool _invFlag);
  bool  b_RhoThetaSmallTheta(double _rho, double _theta,
			     LaurentSeries& _sequence, bool _invFlag);
  bool  b_ThetaThetaSmallTheta(double _rho, double _theta,
			       LaurentSeries& _sequence, bool _invFlag);
};


// ----- definition of class `APTEstimatorBase' -----
//
class APTEstimatorBase :
virtual protected APTTransformerBase, protected EstimatorBase {
 protected:
  APTEstimatorBase();
  APTEstimatorBase(unsigned noItns);
  ~APTEstimatorBase();

  virtual void _writeParm(FILE* fp) = 0;

  double optimize();

  bool  converged(double newFp, double oldFp);
  void  calcGradientMatrix(const CoeffSequence& allPassParams);
  void  calcHessianMatrix(const CoeffSequence& allPassParams);

  void  xformGradVec(const NaturalVector& initFeat, NaturalVector& parDeriv,
		     unsigned ipar, bool useBias = true);
  void  xformHessVec(const NaturalVector& initFeat, NaturalVector& parDeriv,
		     unsigned ipar, unsigned jpar, bool useBias = true);
  virtual void
    calcSeriesDerivs(const CoeffSequence& _params, LaurentSeries& _sequence,
		     unsigned derivIndex) = 0;
  virtual void
    calcSeries2ndDerivs(const CoeffSequence& _params, LaurentSeries& _sequence,
			unsigned derivIndex1, unsigned derivIndex2) = 0;

  void  copyLowerTriangle(gsl_matrix* mat);

  typedef enum { None, Gradient, Hessian } Derivatives;
  Derivatives derivFlag;

  CoeffSequence	_xi, _p1;
  LaurentSeries	_laurentDerivs;
  gsl_matrix*		_hessian;
  gsl_matrix**		_gradMatrix;
  gsl_matrix***	_hessMatrix;
  gsl_matrix**		_cepGradMatrix;
  gsl_matrix***	_cepHessMatrix;
  gsl_matrix*		_dbh_daj;
  gsl_matrix**		_d2bh_daj_dai;

  static const double GradientTolerance;
  static const double SearchTolerance;
  static unsigned MaxIterations;

  class NewtonDirection {
  public:
    NewtonDirection();
    NewtonDirection(UnShrt n);
    ~NewtonDirection();
    void solve(gsl_matrix* hess, CoeffSequence& grad, CoeffSequence& newDirection);
  private:
    const UnShrt nParms;

    gsl_matrix *_eigenVecs;
    gsl_vector *_tempVector, *_eigenVals, *_newGrad, *_pk;
    gsl_eigen_symmv_workspace* _workSpace;
  };

  LaurentSeries q_Alpha, q_Rho, qp_AlphaRho, q_AlphaRho;
  NewtonDirection direction;
};


// ----- definition of class `EstimatorAdapt' -----
//
class EstimatorAdapt : virtual public TransformBase {
 public:
  virtual EstimatorAdapt& estimate(unsigned rc) = 0;

  double ttlFrames() const;

 protected:
  EstimatorAdapt(ParamTreePtr& pt, CodebookSetTrainPtr& cb, unsigned idx, BiasType bt);

  virtual void _writeParm(FILE* fp) = 0;

  void   _calcBias();
  double _calcMixLhood(const CodebookTrain::GaussDensity& mp, double& nCounts);

  ParamTreePtr				_paramTree;
  const unsigned			_nodeIndex;
  const GaussListTrain			_glist;
  NaturalVector				_denom;
  BiasType				_biasType;
};

typedef Inherit<EstimatorAdapt, TransformBasePtr> EstimatorAdaptPtr;


// ----- definition of class `APTEstimatorAdaptBase' -----
//
class APTEstimatorAdaptBase
: virtual public APTEstimatorBase, public EstimatorAdapt {
 public:
  APTEstimatorAdaptBase(ParamTreePtr& pt, CodebookSetTrainPtr& cb, unsigned idx,
			BiasType bt, int trace);
  virtual ~APTEstimatorAdaptBase();

  virtual EstimatorAdapt& estimate(unsigned rc);

 protected:
  virtual void			_writeParm(FILE* fp) = 0;
  virtual void			_resetParams(UnShrt nParams = 0) = 0;
  virtual AdaptationType	_type() = 0;

  void calcGradient(const CodebookTrain::GaussDensity& pdf);
  void calcHessian(const CodebookTrain::GaussDensity& pdf);
  virtual double calcLogLhood(double rho) = 0;
  void _calcBiasGradient();
  void _calcBiasHessian();
  double _accumLogLhood(double rho, double& nCounts);
  void _dumpDebugInfo();

  NaturalVector*			_dmu_daj;
};


// ----- definition of class `RAPTEstimatorBase' -----
//
class RAPTEstimatorBase
: virtual protected APTEstimatorBase, protected RAPTSeriesDerivs {
 protected:
  RAPTEstimatorBase();
  ~RAPTEstimatorBase();

  virtual void _writeParm(FILE* fp);
  virtual void
    calcSeriesDerivs(const CoeffSequence& _params, LaurentSeries& _sequence,
		     unsigned derivIndex);
  virtual void
    calcSeries2ndDerivs(const CoeffSequence& _params, LaurentSeries& _sequence,
			unsigned derivIndex1, unsigned derivIndex2);

  void convertGrad2Rect();
  void convertHess2Rect();
  void _resetParams(UnShrt nParams = 0);

  void boundRange(double x, double y, double dx, double dy, double rho,
		  double& min, double& max);
  virtual void findRange(double rho, double& min, double& max);

  void calcConvMatrix();

  double barrier();
  void addGradBarrier();
  void addHessBarrier();

  gsl_matrix*					_convMatrix;
  gsl_matrix*					_rectHessian;
};


// ----- definition of class `SLAPTEstimatorBase' -----
//
class SLAPTEstimatorBase
  : virtual protected APTEstimatorBase, public SLAPTTransformer {
 protected:
  virtual void _writeParm(FILE* fp);
  virtual void findRange(double rho, double& min, double& max);
  virtual void
    calcSeriesDerivs(const CoeffSequence& _params, LaurentSeries& _sequence,
		     unsigned derivIndex);
  virtual void
    calcSeries2ndDerivs(const CoeffSequence& _params, LaurentSeries& _sequence,
			unsigned derivIndex1, unsigned derivIndex2);
  void _resetParams(UnShrt nParams = 0);

 private:
  LaurentSeries _temp1, _temp2, _temp3;
};


// ----- definition of class `BLTEstimatorAdapt' -----
//
class BLTEstimatorAdapt
: public EstimatorAdapt, private BLTEstimatorBase {
 public:
  BLTEstimatorAdapt(ParamTreePtr& pt, CodebookSetTrainPtr& cb, unsigned idx,
		    BiasType bt = NoBias, int trace = 0001);

  virtual EstimatorAdapt& estimate(unsigned rc);

 private:
  virtual void _writeParm(FILE* fp) {
    BLTEstimatorBase::_writeParm(fp);
  }

  virtual double calcLogLhood(double mu);
};


// ----- definition of class `RAPTEstimatorAdapt' -----
//
class RAPTEstimatorAdapt
: public APTEstimatorAdaptBase, private RAPTEstimatorBase {
 public:
  RAPTEstimatorAdapt(ParamTreePtr& pt, CodebookSetTrainPtr& cb, unsigned idx, UnShrt noParams,
		     BiasType bt, unsigned noItns, int trace);
 private:
  virtual void _writeParm(FILE* fp);
  virtual void _resetParams(UnShrt nParams = 0);
  virtual AdaptationType _type() { return RAPT; }
  virtual double calcLogLhood(double rho);
};


// ----- definition of class `SLAPTEstimatorAdapt' -----
//
class SLAPTEstimatorAdapt
: public APTEstimatorAdaptBase, public SLAPTEstimatorBase {
 public:
  SLAPTEstimatorAdapt(ParamTreePtr& pt, CodebookSetTrainPtr& cb, unsigned idx, UnShrt noParams,
		      BiasType bt, unsigned noItns, int trace);
 private:
  virtual void _writeParm(FILE* fp);
  virtual void _resetParams(UnShrt nParams = 0);
  virtual AdaptationType _type() { return SLAPT; }
  virtual double calcLogLhood(double rho);
};


// ----- definition of class `MLLREstimatorAdapt' -----
//
class MLLREstimatorAdapt :
public EstimatorAdapt, public MLLRTransformer {
 public:
  MLLREstimatorAdapt(ParamTreePtr& pt, CodebookSetTrainPtr& cb, unsigned idx,
		     BiasType bt = NoBias, int trace = 0001);
  virtual ~MLLREstimatorAdapt();

  virtual EstimatorAdapt& estimate(unsigned rc);

 private:
  virtual void _writeParm(FILE* fp);

  void _calcZ();
  void _calcGi(UnShrt dim);
  void _copyParams(unsigned rc) const;
  void _copyLowerTriangle();

  static unsigned nUse;
  static const double MaxSingularValueRatio;

  static gsl_matrix *U, *V;
  static gsl_vector *singularVals, *tempVec;
  static gsl_vector *_workSpace;

  gsl_matrix* Gi;
  gsl_matrix* W;
  gsl_matrix* Z;
};

/*@}*/

/**
* \defgroup FeatureTransformation Feature Transformations
*/
/*@{*/

// ----- definition of class `STCEstimator' -----
//
class STCEstimator :
public EstimatorAdapt, public STCTransformer {
 public:
  class Accu;
  typedef refcount_ptr<Accu> AccuPtr;

  STCEstimator(VectorFloatFeatureStreamPtr& src, ParamTreePtr& pt, DistribSetTrainPtr& dss,
	       unsigned idx = 0, BiasType bt = CepstralFeat, bool cascade = false, bool mmiEstimation = false,
	       int trace = 0001, const String& nm = "STC Estimator");

  virtual ~STCEstimator();

  EstimatorAdapt& estimate(unsigned rc = 1) { return estimate(rc, /* minE= */ 1.0, /* multE= */ 1.0); }

  EstimatorAdapt& estimate(unsigned rc, double minE, double multE);

  void accumulate(DistribPathPtr& path, float factor = 1.0);

 	  void save(const String& fileName, const gsl_matrix_float* trans = NULL);
  virtual void load(const String& fileName);

  const gsl_matrix_float* transMatrix();

  void zeroAccu();

  void saveAccu(FILE* fp) const;
  void loadAccu(FILE* fp);

  void saveAccu(const String& fileName) const;
  void loadAccu(const String& fileName);

  inline double totalPr()  const;
  inline unsigned totalT() const;

 private:
  void _identity();
  void _update(gsl_matrix* stcMatrix, double minE, double multE);
  virtual void _writeParm(FILE* fp);

  const bool			_cascade;
  const bool			_mmiEstimation;

  gsl_vector*				_cofactor;
  gsl_matrix*				_scratchMatrix;
  gsl_matrix*				_cascadeMatrix;
  gsl_vector*				_scratchVector;
  gsl_matrix*				_stcInv;
  gsl_matrix_float*		_transMatrix;
  gsl_permutation*		_permutation;
  gsl_matrix**				_GiInv;
  float*			_addCount;
  gsl_vector_float*		_transFeature;

  DistribSetTrainPtr		_dss;
  AccuPtr			_accu;
};

typedef Inherit<STCEstimator, STCTransformerPtr> STCEstimatorPtr;


// ----- definition for class `STCEstimator::Accu' -----
// 
class STCEstimator::Accu {
 public:
  Accu(unsigned subFeatLen, unsigned nSubFeat);
  ~Accu();

  void zero();
  void save(FILE* fp, unsigned index = 0);
  void load(FILE* fp, unsigned index = 0);

  void accumulateW1(const CodebookBasicPtr& cb, const gsl_vector_float* pattern,
		    const float* addCount, float factor = 1.0);

  void accumulateW2(const CodebookBasicPtr& cb, const gsl_matrix* invMatrix,
		    const float* addCount, float factor);

  void increment(double totalPr, unsigned totalT = 0) { _totalPr += totalPr;  _totalT += totalT; }

  void copyUpperTriangle(gsl_matrix* cov);
  void makePosDef(double minE, double multE);
  double logLhood();
  double calcAuxFunc(const gsl_matrix* stcMatrix, bool mmiEstimation);

  const gsl_matrix* sumOsq(unsigned i) { copyUpperTriangle(_sumOsq[i]);  return _sumOsq[i]; }
  const gsl_matrix* regSq(unsigned i)  { copyUpperTriangle(_regSq[i]);   return _regSq[i];  }

  double count()    const { return _count;    }
  double denCount() const { return _denCount; }

  double E() const { return _E; }

  double   totalPr() const { return _totalPr; }
  unsigned totalT()  const { return _totalT;  }

 private:
  static double					CountThreshold;

  const unsigned				_subFeatLen;	// size of reference vectors
  const unsigned				_nSubFeat;	// number of sub-features
  double					_count;		// count[j] = training counts for j-th vector
  double					_denCount;	// denCount[j] = training counts for j-th vector
  double					_totalPr;
  unsigned					_totalT;
  double*					_obs;

  double					_E;		// normalization factor for MMI

  gsl_matrix*					_scratchMatrix;
  gsl_vector*					_scratchVector;
  gsl_permutation*				_permutation;
  gsl_eigen_symm_workspace*			_workSpace;

  gsl_matrix**					_sumOsq;	// sumOsq[j] = sum of squares for j-th vector
  gsl_matrix**					_regSq;		// regSq[j]  = regularization term for j-th vector
};

double   STCEstimator::totalPr() const { return _accu->totalPr(); }
unsigned STCEstimator::totalT()  const { return _accu->totalT(); }


// ----- definition of class `LDAEstimator' -----
//
class LDAEstimator :
public EstimatorAdapt, public LDATransformer {

  class _SortItem {
  public:
    double			evalue;
    unsigned			index;
  };

  class _GreaterThan {
  public:
    bool operator()(const _SortItem& first, const _SortItem& second) {
      return first.evalue > second.evalue;
    }
  };
  class _LessThan {
  public:
    bool operator()(const _SortItem& first, const _SortItem& second) {
      return first.evalue < second.evalue;
    }
  };

 public:
  LDAEstimator(VectorFloatFeatureStreamPtr& src, ParamTreePtr& pt, DistribSetTrainPtr& dss,
	       const CodebookBasicPtr& globalCodebook, unsigned idx = 0, BiasType bt = CepstralFeat, int trace = 0001);
  virtual ~LDAEstimator();

  class Accu;

  typedef refcountable_ptr<Accu>	AccuPtr;

  virtual EstimatorAdapt& estimate(unsigned rc = 1);
  double accumulate(DistribPathPtr& dspath, float factor = 1.0);

  void zeroAccu();

  inline void saveAccu(FILE* fp) const;
  inline void loadAccu(FILE* fp);

  void saveAccu(const String& fileName) const;
  void loadAccu(const String& fileName);

 private:
  void _update(unsigned iblk);
  void _scaleEigenvectors(unsigned iblk);
  void _calculatePositiveEigenvalues(const gsl_matrix* A, gsl_vector* evalues, gsl_matrix* evectors);

  static const double ConditionNumber;

  virtual void _writeParm(FILE* fp);

  static const double MaxSingularValueRatio;

  CodebookBasicPtr		_globalCodebook;
  gsl_matrix*			_ldaMatrix;
  gsl_matrix*			_scatterMatrix;
  gsl_matrix*			_scratchMatrix;
  gsl_vector*			_eigenValue;
  gsl_matrix*			_whitening;
  gsl_matrix*			_eigenVector;
  gsl_eigen_symmv_workspace*	_workSpace;

  DistribSetTrainPtr		_dss;
  AccuPtr			_accu;
};

typedef Inherit<LDAEstimator, LDATransformerPtr> LDAEstimatorPtr;


// ----- definition for class `LDAEstimator::Accu' -----
// 
typedef Countable LDAEstimatorAccuCountable;
class LDAEstimator::Accu : public LDAEstimatorAccuCountable {
 public:
  Accu(unsigned subFeatLen, unsigned nblks);
  ~Accu();

  void zero();
  void save(FILE* fp, unsigned index = 0);
  void load(FILE* fp, unsigned index = 0);
  void copyUpperTriangle(gsl_matrix* cov);
  void accumulate(const CodebookBasicPtr& cb, const CodebookBasicPtr& globalCodebook,
		  const gsl_vector_float* pattern, const float* addCount, float addFactor = 1.0);

  void scaleScatterMatrices();

  const gsl_matrix* within(unsigned i)  { copyUpperTriangle(_within[i]);  return _within[i];  }
  const gsl_matrix* between(unsigned i) { copyUpperTriangle(_between[i]); return _between[i]; }
  const gsl_matrix* mixture(unsigned i) { copyUpperTriangle(_mixture[i]); return _mixture[i]; }

 private:
  static double CountThreshold;

  const unsigned		_subFeatLen;	// size of reference vectors
  const unsigned		_nSubFeat;	// number of sub-features

  double*			_obsW;
  double*			_obsB;
  double*			_obsM;
  gsl_matrix**			_within;
  gsl_matrix**			_between;
  gsl_matrix**			_mixture;
  double			_count;
};


// ----- definition of class `EstimatorTree' -----
//
class EstimatorTree : public TransformerTree {

 public:
  void writeParams(const String& fileName);
  double ttlFrames();

  class Node;          friend class Node;
  class LeafIterator;  friend class classLeafIterator;

  typedef Inherit<Node, BaseTree::NodePtr> NodePtr;

        CodebookSetTrainPtr& cbs()       { return Cast<CodebookSetTrainPtr>(_cbs()); }
  const CodebookSetTrainPtr& cbs() const { return Cast<CodebookSetTrainPtr>(_cbs()); }

 protected:
  typedef set<unsigned> RCSet;
  typedef RCSet::const_iterator RCSetIter;

  RCSet leafNodes();

  EstimatorTree(CodebookSetTrainPtr& cb, ParamTreePtr& paramTree,
		int trace, float threshold);

 private:
  static float EstimateThreshold;

  	NodePtr& node(int idx) { return Cast<NodePtr>(BaseTree::node(idx)); }
  const NodePtr& node(int idx) const {
    return Cast<const NodePtr>(BaseTree::node(idx));
  }
};

typedef Inherit<EstimatorTree, TransformerTreePtr> EstimatorTreePtr;

void estimateAdaptationParameters(EstimatorTreePtr& tree);


// ----- definition of class `EstimatorTree::Node' -----
//
class EstimatorTree::Node : public TransformerTree::Node {
  friend class EstimatorTree::LeafIterator;
 public:
  Node(EstimatorTree& tr, UnShrt idx, NodeType type, EstimatorAdapt* estimator)
    : TransformerTree::Node(tr, idx, type, estimator), _estimator(estimator) { }
  ~Node() { }

  double ttlFrames() const { return _estimator->ttlFrames(); }
  void estimate(unsigned rc) { _estimator->estimate(rc); }

 private:
  EstimatorAdapt*			_estimator;
};


// ----- definition of class `EstimatorTree::LeafIterator' -----
//
class EstimatorTree::LeafIterator : public TransformerTree::Iterator {
 public:
  LeafIterator(EstimatorTreePtr& tr)
    : TransformerTree::Iterator(tr, /* onlyLeaves= */ true) { }

  NodePtr& node() { return Cast<NodePtr>(BaseTree::Iterator::node()); }

  double ttlFrames() { return node()->_estimator->ttlFrames(); }

  void estimate();

  const EstimatorTreePtr& tree() const {
    if (_tree.isNull()) {
      printf("problem here.\n");  fflush(stdout);
    }
    return Cast<EstimatorTreePtr>(_tree);
  }
};

void initializeNaturalIndex(UnShrt nsub);


// ----- definition of class `EstimatorTreeMLLR' -----
//
class EstimatorTreeMLLR : public EstimatorTree {
 public:
  EstimatorTreeMLLR(CodebookSetTrainPtr& cb, ParamTreePtr& paramTree,
		    int trace = 1, float threshold = 1500.0);
};

typedef Inherit<EstimatorTreeMLLR, EstimatorTreePtr> EstimatorTreeMLLRPtr;


// ----- definition of class `EstimatorTreeSLAPT' -----
//
class EstimatorTreeSLAPT : public EstimatorTree {
 public:
  EstimatorTreeSLAPT(CodebookSetTrainPtr& cb, ParamTreePtr& paramTree,
		     const String& biasType = "CepstraOnly",
		     UnShrt paramN = 9, unsigned noItns = 10, int trace = 1, float threshold = 150.0);
};

typedef Inherit<EstimatorTreeSLAPT, EstimatorTreePtr> EstimatorTreeSLAPTPtr;


// ----- definition of class `EstimatorTreeSTC' -----
//
class EstimatorTreeSTC : public EstimatorTree {
 public:
  EstimatorTreeSTC(VectorFloatFeatureStreamPtr& src, DistribSetTrainPtr& dss, ParamTreePtr& paramTree,
		   const String& biasType = "CepstraOnly", int trace = 1, float threshold = 150.0);
};

typedef Inherit<EstimatorTreeSTC, EstimatorTreePtr> EstimatorTreeSTCPtr;


// ----- definition of class `EstimatorTreeLDA' -----
//
class EstimatorTreeLDA : public EstimatorTree {
 public:
  EstimatorTreeLDA(VectorFloatFeatureStreamPtr& src, DistribSetTrainPtr& dss, ParamTreePtr& paramTree,
		   const CodebookBasicPtr& globalCodebook, const String& biasType = "CepstraOnly",
		   int trace = 1, float threshold = 150.0);
};

typedef Inherit<EstimatorTreeLDA, EstimatorTreePtr> EstimatorTreeLDAPtr;

/*@}*/

/*@}*/

#endif
