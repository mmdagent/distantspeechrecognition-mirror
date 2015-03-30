//                              -*- C++ -*-
//
//                              Millennium
//                  Distant Speech Recognition System
//                                (dsr)
//
//  Module:  asr.sat
//  Purpose: Speaker-adapted ML and discriminative HMM training.
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


#ifndef _sat_h_
#define _sat_h_

#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_linalg.h>

#include "sat/codebookFastSAT.h"
#include "train/estimateAdapt.h"


/**
* \defgroup SpeakerAdaptedTraining Speaker-Adapted Training Class Hierarchy.
* This hierarchy of classes provides the capability to perform speaker-adapted
* training using either a maximum likelihood (ML) or maximum mutual information (MMI)
* criterion. The ML re-estimation formulae can be expressed as
\f{eqnarray*}
\mathbf{M}_m &=& \mathbf{A} \, \mathbf{\Sigma} \, \mathbf{A}^T \\
\mathbf{v}_m &=& \mathbf{A}^T \underline{\mathbf{\Sigma}}_m \, (\mathbf{\mu}_m - \underline{\mathbf{b}}) \\
&=& \sum_s c_m^{(s)} \mathbf{A}^{(s)T} \mathbf{\Sigma}_m \left(\tilde{\mathbf{\mu}}_m - \mathbf{b}^{(s)}\right).
\f}
* The final solution for the speaker-independent mean is then given by
\f[
\mathbf{\mu}_m = \mathbf{M}^{-1}_m \, \mathbf{v}_m.
\f]
* The solution for the \f$m\f$th diagonal component of the speaker-independent
* covariance matrix is given by
\f[
\sigma^2_m = \tilde{\sigma}^2_m + \ldots
\f]
*/
/*@{*/


// ----- definition for class `SATMean' -----
//
class SATMean {
  class BlockAcc {
  public:
    BlockAcc() : _mat(NULL), _vec(NULL), _varVec(NULL), _scalar(0.0) { }

    gsl_matrix*					_mat;
    gsl_vector*					_vec;
    gsl_vector*					_varVec;

    double					_scalar;
  };
 public:
  class UpdateInfo;

  SATMean(UnShrt len, UnShrt orgLen);
  virtual ~SATMean();

  void accumulate(const TransformBasePtr& transformer, float c_k,
		  const InverseVariance& invVar, const DNaturalVector& sumO,
		  bool addScalar = true);

  virtual double update(NaturalVector satMean, UpdateInfo* info = NULL,
			UnShrt* dl = NULL);
  double fixup(NaturalVector satMean);
  void   zero();

 protected:
  static const double MaxSingularValueRatio;

  virtual void _checkBlocks(UnShrt nBlks, bool allocVarVec = false);
  virtual void _solve(gsl_matrix* _mat, gsl_vector* _vec, gsl_vector* _newMean, UpdateInfo* info);
  static double
    _auxFunc(const gsl_matrix* mat, const gsl_vector* vec, double scalar, const gsl_vector* mean);

  void _validate(gsl_matrix* mat, gsl_vector* vec, double scalar);

  double					_occ;

  const UnShrt					_featLen;
  const UnShrt					_origFeatLen;

  UnShrt					_nBlocks;
  UnShrt					_blockLen;
  UnShrt					_origBlockLen;

  BlockAcc*					_block;

  static unsigned				nUse;
  static bool					allocated;
  static gsl_matrix* 				temp1;
  static gsl_matrix*				temp2;
  static gsl_matrix*				matCopy;
  static gsl_vector*				vec1;
  static gsl_vector*				vec2;
  static gsl_vector*				vecCopy;
  static gsl_vector*				vecProduct;

  static gsl_matrix*				U;
  static gsl_matrix				*V;
  static gsl_vector*				newMean;
  static gsl_vector*				singularVals;
  static gsl_vector*				tempVec;
  static gsl_vector*				_workSpace;
};

typedef refcount_ptr<SATMean>	SATMeanPtr;


// ----- definition for class `SATMeanFullCovar' -----
//
class SATMeanFullCovar : public SATMean {
 public:
  SATMeanFullCovar(UnShrt len, UnShrt orgLen) : SATMean(len, orgLen) { }

  void accumulate(const TransformBasePtr& transformer, float c_k,
		  const gsl_matrix* invCovar, const DNaturalVector& sumO);

  void dump(const String& fileName) const;
  void load(const String& fileName);
};

typedef Inherit<SATMeanFullCovar, SATMeanPtr>	SATMeanFullCovarPtr;


// ----- definition for class `MDLSATMean' -----
//
class MDLSATMean : public SATMean {
 public:
  MDLSATMean(UnShrt len, UnShrt orgLen, double thr);
  virtual ~MDLSATMean();

  virtual double update(NaturalVector satMean, UpdateInfo* info = NULL,
			UnShrt* dl = NULL);

 protected:
  virtual void _checkBlocks(UnShrt nblk, bool allocVarVec = false);
  virtual void _solve(gsl_matrix* _mat, gsl_vector* _vec, gsl_vector* _newMean, UpdateInfo* info);

 private:
  const double    			_lhoodThresh;
  static gsl_eigen_symmv_workspace*	_workSpace;
};

typedef Inherit<MDLSATMean, SATMeanPtr>		MDLSATMeanPtr;


// ----- definition for class `FastSAT' -----
//
class FastSAT : public MDLSATMean {
 public:
  FastSAT(UnShrt len, UnShrt orgLen, double thr);

  void accumulate(const TransformBasePtr& transformer, double c_k,
		  const InverseVariance& invVar);
  virtual void addFastAcc(const DNaturalVector& meanAcc,
			  const DNaturalVector& varAcc,
			  const double* scalar, int nblks);

  double update(NaturalVector satMean, NaturalVector satVar,
		UpdateInfo* info = NULL, UnShrt* dl = NULL);

  SATMean::fixup;
  SATMean::zero;

protected:
  double _updateMean(NaturalVector satMean, UpdateInfo* info = NULL,
		     UnShrt* dl = NULL);
  void _updateVariance(NaturalVector satVar);
};

typedef Inherit<FastSAT, MDLSATMeanPtr>		FastSATPtr;


// ----- definition for class `SATMean::UpdateInfo' -----
//
class SATMean::UpdateInfo {
  friend class SATMean;  friend class MDLSATMean;
  friend ostream& operator<<(ostream& os, const UpdateInfo& ui);

 public:
  UpdateInfo();
  UpdateInfo(unsigned tb, unsigned tbu);

  void zero();
  void operator+=(const UpdateInfo& ui);

 private:
  unsigned					ttlBlocks;
  unsigned					ttlBlocksUpdated;
  unsigned					ttlDimensions;
  unsigned					ttlDimensionsUpdated;
};


// ---- definition for class `SATVariance' ----
//
class SATVariance {
 public:
  SATVariance(UnShrt len);
  ~SATVariance();

  void accumulate(const TransformBasePtr& transformer, double c_k,
		  const NaturalVector& mean, const DNaturalVector& sumO,
		  const DNaturalVector& sumOsq);

  double update(NaturalVector var);
  void zero();
  void fixup(NaturalVector var);

  double operator[](UnShrt idx) const { return gsl_vector_get(_vec, idx); }

 private:
  void _validate(gsl_vector* vec, double occ);

  UnShrt					_featLen;
  gsl_vector*					_vec;
  double					_occ;
};


// ----- definition for class `MMISAT' -----
//
class FastMMISAT : public FastSAT {
  typedef SATMean::UpdateInfo Info;
 public:
  FastMMISAT(UnShrt len, UnShrt orgLen, double thr = 0.0, bool addScalar = true);
  virtual ~FastMMISAT();

  void accumulate(const TransformBasePtr& transformer, double ck, double dk,
		  const InverseVariance& invVar, const NaturalVector& mdlMean);

  void accumulateVk(const TransformBasePtr& transformer0, const TransformBasePtr& transformer,
		    double dk, const InverseVariance& invVar, const NaturalVector& mdlMean);

  double update(NaturalVector satMean, NaturalVector satVar, Info* info = NULL,
		UnShrt* dl = NULL);

  static void zeroNoNegativeComponents();
  static void announceNoNegativeComponents();

 private:
  void _accumulateMMIVar(const NaturalVector &varAcc, double dk);

  class SpkrTrans {
  public:
    SpkrTrans(const TransformBasePtr& tr, double d_k)
      : _trans(tr), _dk(d_k) { }

    const TransformBasePtr& trans() const { return _trans; }
    double                  dk()    const { return _dk; }

  private:
    const TransformBasePtr			_trans;
    double					_dk;
  };

  typedef refcount_ptr<SpkrTrans>		SpkrTransPtr;

  class SpkrTransList {
    typedef list<SpkrTransPtr>			_SpkrTransList;
    typedef _SpkrTransList::const_iterator	_SpkrTransListIter;

  public:
    ~SpkrTransList() {
      _meanList.erase(_meanList.begin(), _meanList.end());
    }
    void add(const SpkrTransPtr& spkrMean) { _meanList.push_back(spkrMean); }
    unsigned nSpkrs() const { return _meanList.size(); }

    class Iterator;  friend class Iterator;

    class Iterator {
    public:
      Iterator(const SpkrTransList& list)
	: _meanList(list._meanList), _itr(_meanList.begin()) { }

      void operator++(int) { _itr++; }
      bool more() { return _itr != _meanList.end(); }
      const SpkrTransPtr& trans() { return *_itr; }

    private:
      const _SpkrTransList&			_meanList;
      _SpkrTransListIter			_itr;
    };

  private:
    _SpkrTransList				_meanList;
  };

  static unsigned TtlVarianceComponents;
  static unsigned TtlNegativeVarianceComponents;

  SpkrTransList					_transList;

  NaturalVector					_mmiMean;
  NaturalVector					_difference;
  NaturalVector					_transMean;
  bool						_addScalar;
};

typedef Inherit<FastMMISAT, MDLSATMeanPtr>	FastMMISATPtr;


// ----- definition of class `TransformerTreeList' -----
//
class TransformerTreeList {
  typedef map<String, TransformerTreePtr>	_TreeList;
  typedef _TreeList::const_iterator		_TreeListIter;
  typedef _TreeList::value_type			_ValueType;

 public:
  TransformerTreeList(CodebookSetFastSATPtr& cb, const String& parmFile,
		      UnShrt orgSubFeatLen = 0, const SpeakerList* spkList = NULL);
  ~TransformerTreeList();

  const TransformerTreePtr& getTree(const String& spkrLabel);
  const TransformerTreePtr& getTree(const String& spkrLabel) const;

 private:
  CodebookSetFastSATPtr				_cbs;
  const String					_parmFile;
  const UnShrt					_orgSubFeatLen;
  _TreeList					_treeList;
};


// ----- definition of base class 'SATBase' -----
//
class SATBase {
 public:
  virtual ~SATBase();

  virtual double estimate(const SpeakerListPtr& sList, const String& spkrAdaptParms,
			  const String& meanVarsStats);

  class BaseAcc;		friend class BaseAcc;
  class MeanAcc;		friend class MeanAcc;
  class FastAcc;		friend class FastAcc;
  class VarianceAcc;		friend class VarianceAcc;
  class RegClassAcc;		friend class RegClassAcc;
  class MMIAcc;			friend class MMIAcc;
  class MMIRegClassAcc;		friend class MMIRegClassAcc;

  typedef refcount_ptr<BaseAcc>				BaseAccPtr;
  typedef Inherit<MeanAcc,		BaseAccPtr>	MeanAccPtr;
  typedef Inherit<FastAcc,		BaseAccPtr>	FastAccPtr;
  typedef Inherit<VarianceAcc,		BaseAccPtr>	VarianceAccPtr;
  typedef Inherit<RegClassAcc,		BaseAccPtr>	RegClassAccPtr;
  typedef Inherit<MMIAcc,		BaseAccPtr>	MMIAccPtr;
  typedef Inherit<MMIRegClassAcc,	BaseAccPtr>	MMIRegClassAccPtr;

  static double MMISATMultiplier;
  static bool MeanOnly;
     
 protected:
  typedef CodebookSetFastSAT::CodebookIterator CodebookIterator;
  typedef CodebookFastSAT::Iterator    	       Iterator;

  static double LogLhoodThreshhold;
  static double UpdateThreshhold;

  friend class AccList;
  class AccList {
    typedef list<BaseAccPtr>			_AccList;
    typedef _AccList::iterator			_AccListIterator;
    typedef _AccList::const_iterator		_AccListConstIterator;
  public:
    ~AccList();
    void add(BaseAccPtr& acc) { _list.push_back(acc); }

    class Iterator;  friend class Iterator;
    class Iterator {
    public:
      Iterator(AccList& al)
	: _list(al._list), _itr(_list.begin()) { }
      bool more() const { return _itr != _list.end(); }
      void operator++(int) { _itr++; }
      SATBase::BaseAccPtr& accu() { return *_itr; }

    private:
      _AccList&					_list;
      _AccListIterator				_itr;
    };

  private:
    _AccList					_list;
  };

  SATBase(CodebookSetFastSATPtr& cb, UnShrt meanLen, double lhoodThreshhold,
	  double massThreshhold,
	  double multiplier, unsigned nParts, unsigned part, bool meanOnly);
  
  void zero();
  void accumulate(const TransformerTreePtr& tree);
  void update();

  virtual void announceAccumulate() = 0;
  virtual void announceAccumulateDone() = 0;
  virtual void announceUpdate() = 0;
  virtual void announceUpdateDone() = 0;

  	CodebookSetFastSATPtr& cbs()       { return _cbs; }
  const CodebookSetFastSATPtr& cbs() const { return _cbs; }

  UnShrt orgFeatLen() const { return cbs()->orgFeatLen(); }
  UnShrt nSubFeat()   const { return cbs()->nSubFeat(); }

  CodebookSetFastSATPtr	_cbs;

  AccList					_accList;  // list of -> SAT accumulators
  const UnShrt					_origFeatLen;
  unsigned					_nParts;
  unsigned					_part;

 private:
  UnShrt _setFeatLen(UnShrt meanLen);
};

// definition of nested classes to accumulate SAT stats for each Gaussian
//
class SATBase::BaseAcc : protected CodebookFastSAT::GaussDensity {
 public:
  virtual void accumulate(const TransformerTreePtr& tree) = 0;

  virtual void update() = 0;
  virtual void zero() = 0;
  virtual void addFastAcc() { }
  virtual ~BaseAcc() { }

 protected:
  BaseAcc(GaussDensity pdf, UnShrt meanLen);

  void   _zero() { _occurences = 0.0; }
  void   _increment(double cnt) { _occurences += cnt; }
  double _occ() { return _occurences; }

  const UnShrt					_featLen;
  const UnShrt					_origFeatLen;

  static unsigned				_TotalGaussians;
  static unsigned				_TotalUpdatedGaussians;

 private:
  double					_occurences;
};

// accumulate SAT mean re-estimation statistics
//
class SATBase::MeanAcc : public SATBase::BaseAcc {
  typedef SATMean::UpdateInfo Info;
 public:
  MeanAcc(GaussDensity pdf, UnShrt meanLen, double thr);
  ~MeanAcc();

  virtual void accumulate(const TransformerTreePtr& tree);
  virtual void update();
  virtual void zero();

  static void announceAccumulate();
  static void announceAccumulateDone();
  static void announceUpdate();
  static void announceUpdateDone();

 protected:
  static Info& updateInfo();

  SATMeanPtr					_satMean;
};

// accumulate fast SAT re-estimation statistics
//
class SATBase::FastAcc : public SATBase::BaseAcc {
  typedef SATMean::UpdateInfo Info;
 public:
  FastAcc(GaussDensity pdf, UnShrt meanLen, double thr);

  virtual void accumulate(const TransformerTreePtr& tree);
  virtual void update();
  virtual void addFastAcc();
  virtual void zero();

  static void announceAccumulate();
  static void announceAccumulateDone();
  static void announceUpdate();
  static void announceUpdateDone();

 protected:
  static Info& updateInfo();

  FastSAT					_fastSAT;
};

// accumulate MMI-SAT mean and variance re-estimation statistics
//
class SATBase::MMIAcc : public SATBase::BaseAcc {
  typedef SATMean::UpdateInfo Info;
 public:
  MMIAcc(GaussDensity pdf, UnShrt meanLen, double thr);
  ~MMIAcc();

  virtual void accumulate(const TransformerTreePtr& tree);
  virtual void update();
  virtual void zero();
  virtual void addFastAcc();

  static void announceAccumulate();
  static void announceAccumulateDone();
  static void announceUpdate();
  static void announceUpdateDone();

 private:
  static Info& updateInfo();

  FastMMISAT					_mmiSAT;
};

// accumulate SAT variance re-estimation statistics
//
class SATBase::VarianceAcc : public SATBase::BaseAcc {
 public:
  VarianceAcc(GaussDensity pdf, UnShrt meanLen);

  virtual void accumulate(const TransformerTreePtr& tree);
  virtual void update();
  virtual void zero();

  static void announceAccumulate();
  static void announceAccumulateDone();
  static void announceUpdate();
  static void announceUpdateDone();

 private:
  SATVariance _satVar;
};

class ClassTotals {
  typedef map<short, unsigned>			_ClassTotals;
  typedef _ClassTotals::iterator		_ClassTotalsIter;
  typedef _ClassTotals::const_iterator		_ConstClassTotalsIter;
  friend ostream& operator<<(ostream& os, const ClassTotals& cl);
 public:
  void zero();
  unsigned& operator[](short rc) { return _classTotals[rc]; }
  inline unsigned operator[](short rc) const;

private:
  _ClassTotals					_classTotals;
};

unsigned ClassTotals::operator[](short rc) const
{
  _ConstClassTotalsIter itr = _classTotals.find(rc);

  return (*itr).second;
}

// accumulate SAT statistics for joint re-estimation of
// mean and optimal regression class
//
class SATBase::RegClassAcc : public SATBase::BaseAcc {
  typedef SATMean::UpdateInfo Info;
 public:
  RegClassAcc(GaussDensity pdf, UnShrt meanLen, double thr);

  virtual void accumulate(const TransformerTreePtr& tree);
  virtual void update();
  virtual void zero();
  virtual void addFastAcc();

  static void announceAccumulate();
  static void announceAccumulateDone();
  static void announceUpdate();
  static void announceUpdateDone();
  static void setMaxRegClasses(unsigned max) { _maxRegClasses = max; }

 private:
  class FastSATList {
    typedef map<UnShrt, FastSATPtr>		_FastSATList;
    typedef _FastSATList::iterator		_FastSATListIter;
    typedef _FastSATList::value_type		_ValueType;
  public:
    ~FastSATList();

    class Iterator;  friend class Iterator;

    class Iterator {
    public:
      Iterator(FastSATList& l)
	: _fastSATList(l._fastSATList), _itr(_fastSATList.begin()) { }

      void operator++(int) { _itr++; }
      bool more() { return _itr != _fastSATList.end(); }
      UnShrt regClass() const { return (*_itr).first; }
      FastSATPtr& acc() { return (*_itr).second; }

    private:
      _FastSATList&				_fastSATList;
      _FastSATList::iterator			_itr;
    };

    FastSATPtr& operator[](UnShrt rc);
    FastSATPtr& find(UnShrt rc, UnShrt fl, UnShrt ol, double pl);

  private:
    _FastSATList				_fastSATList;
  };

  typedef CodebookFastSAT::GaussDensity::RCArray RCArray;
  typedef CodebookFastSAT::GaussDensity::RegClassScore RegClassScore;

  class LessThan {    // function object for regression class score sorting
  public:
    bool operator()(const RegClassScore& first, const RegClassScore& second) {
      return first._score < second._score;
    }
  };

  void _fixup();
  void _setClass(short cl);
  void _setClass(RCArray& rcsarray);

  static Info& updateInfo();
  static ClassTotals& classTotals();

  static UnShrt					_firstLeaf;
  static unsigned				_ttlChanges;
  static unsigned				_ttlUpdates;

  static unsigned				_maxRegClasses;

  const double					_lhoodThresh;
  FastSATList					_fastSATList;
};


// ----- definition for class template `SATBase::MMIRegClassAcc' -----
//
class SATBase::MMIRegClassAcc : public SATBase::BaseAcc {
  typedef SATMean::UpdateInfo Info;
 public:
  MMIRegClassAcc(GaussDensity pdf, UnShrt meanLen, double thr);

  virtual void accumulate(const TransformerTreePtr& tree);
  virtual void update();
  virtual void zero();
  virtual void addFastAcc();

  static void announceAccumulate();
  static void announceAccumulateDone();
  static void announceUpdate();
  static void announceUpdateDone();
  static void setMaxRegClasses(unsigned max) { _maxRegClasses = max; }

 private:
  class FastSATList {
    typedef map<UnShrt, FastMMISATPtr>		_FastSATList;
    typedef _FastSATList::iterator		_FastSATListIter;
    typedef _FastSATList::value_type		_ValueType;
  public:
    ~FastSATList();

    class Iterator;  friend class Iterator;

    class Iterator {
    public:
      Iterator(FastSATList& l)
	: _fastSATList(l._fastSATList), _itr(_fastSATList.begin()) { }

      void operator++(int) { _itr++; }
      bool more() { return _itr != _fastSATList.end(); }
      UnShrt regClass() const { return (*_itr).first; }
      FastMMISATPtr& acc() { return (*_itr).second; }

    private:
      _FastSATList&				_fastSATList;
      _FastSATList::iterator			_itr;
    };

    FastMMISATPtr& operator[](UnShrt rc);
    FastMMISATPtr& find(UnShrt rc, UnShrt fl, UnShrt ol, double pl);
  private:
    _FastSATList				_fastSATList;
  };

  typedef CodebookFastSAT::GaussDensity::RCArray RCArray;
  typedef CodebookFastSAT::GaussDensity::RegClassScore RegClassScore;

  class LessThan {    // function object for regression class score sorting
  public:
    bool operator()(const RegClassScore& first, const RegClassScore& second) {
      return first._score < second._score;
    }
  };

  void _fixup();
  void _setClass(short cl);
  void _setClass(RCArray& rcsarray);

  static Info& updateInfo();
  static ClassTotals& classTotals();

  static UnShrt					_firstLeaf;
  static unsigned				_ttlChanges;
  static unsigned				_ttlUpdates;

  static unsigned				_maxRegClasses;

  const double					_lhoodThresh;
  FastSATList					_fastSATList;
};


// ----- definition for class template `SAT' -----
//
template <class Type>
class SAT : public SATBase {
 public:
  SAT(CodebookSetFastSATPtr& h, UnShrt meanLen, double lhoodThreshhold,
      double massThreshhold,
      unsigned nParts = 1, unsigned part = 1, double mmiMultiplier = 1.0,
      bool meanOnly = false);
  SAT(CodebookSetFastSATPtr& h, UnShrt meanLen, double threshHold,
      unsigned nParts = 1, unsigned part = 1, double mmiMultiplier = 1.0,
      bool meanOnly = false);

 protected:
  void announceAccumulate()     { Type::announceAccumulate(); }
  void announceAccumulateDone() { Type::announceAccumulateDone(); }
  void announceUpdate()         { Type::announceUpdate(); }
  void announceUpdateDone()     { Type::announceUpdateDone(); }
};

template <class Type>
SAT<Type>::SAT(CodebookSetFastSATPtr& cbs, UnShrt meanLen, double lhoodThreshhold,
	       double massThreshhold, unsigned nParts, unsigned part,
	       double mmiMultiplier, bool meanOnly)
  : SATBase(cbs, meanLen, lhoodThreshhold, massThreshhold, mmiMultiplier,
	    nParts, part, meanOnly)
{
  cout << "Attaching SAT accumulators ... ";

  for (CodebookIterator sitr(cbs, _nParts, _part); sitr.more(); sitr++) {
    CodebookFastSATPtr cbk(sitr.cbk());
    for (Iterator itr(cbk); itr.more(); itr++) {
      BaseAccPtr ptr(new Type(itr.mix(), _origFeatLen, lhoodThreshhold));
      _accList.add(ptr);
    }
  }

  cout << "Done" << endl << endl;
}

template <class Type>
SAT<Type>::SAT(CodebookSetFastSATPtr& cbs, UnShrt meanLen, double threshHold,
	       unsigned nParts, unsigned part, double mmiMultiplier,
	       bool meanOnly)
  : SATBase(cbs, meanLen, 0.0, threshHold, mmiMultiplier, nParts, part, meanOnly)
{
  cout << "Attaching SAT accumulators ... ";

  for (CodebookIterator sitr(*this, _nParts, _part); sitr.more(); sitr++) {
    CodebookFastSATPtr cbk(sitr.cbk());
    for (Iterator itr(cbk); itr.more(); itr++) {
      _accList.add(new Type(itr.mix(), _origFeatLen));
    }
  }

  cout << "Done" << endl << endl;
}

typedef SAT<SATBase::MeanAcc>			MeanSAT;
typedef SAT<SATBase::VarianceAcc>		VarianceSAT;
typedef SAT<SATBase::RegClassAcc>		RegClassSAT;
typedef SAT<SATBase::MMIRegClassAcc>		MMIRegClassSAT;

typedef refcount_ptr<MeanSAT>			MeanSATPtr;
typedef refcount_ptr<VarianceSAT>		VarianceSATPtr;
typedef refcount_ptr<RegClassSAT>		RegClassSATPtr;
typedef refcount_ptr<MMIRegClassSAT>		MMIRegClassSATPtr;


// ------------ definition for class template `SATFast' ------------
//
template <class Type>
class SATFast : private SAT<Type> {
public:
  SATFast(CodebookSetFastSATPtr& cbs, UnShrt meanLen, double lhoodThreshhold = 0.1,
	  double massThreshhold = 10.0,
	  unsigned nParts = 1, unsigned part = 1, double mmiMultiplier = 1.0,
	  bool meanOnly = false, int maxNoRegClass = 1)
    : SAT<Type>(cbs, meanLen, lhoodThreshhold, massThreshhold,
		nParts, part, mmiMultiplier, meanOnly)
    {
      SATBase::RegClassAcc::setMaxRegClasses(maxNoRegClass);
    }

  inline double estimate(const SpeakerListPtr& sList, const String& spkrAdaptParms,
			 const String& meanVarsStats);

 private:
  inline void addFastAccs();
};

template <class Type>
double SATFast<Type>::estimate(const SpeakerListPtr& sList, const String& spkrAdaptParms,
			       const String& meanVarsStats)
{
  addFastAccs();
  return SATBase::estimate(sList, spkrAdaptParms, meanVarsStats);
}

template <class Type>
void SATFast<Type>::addFastAccs()
{
  cout << "Adding fast accumulators ... ";

  for (SATBase::AccList::Iterator itr(this->_accList); itr.more(); itr++)
    itr.accu()->addFastAcc();

  cout << "Done" << endl << endl;
}

typedef SATFast<SATBase::FastAcc>		FastMeanVarianceSAT;
typedef SATFast<SATBase::RegClassAcc>		FastRegClassSAT;
typedef SATFast<SATBase::MMIAcc>		FastMaxMutualInfoSAT;
typedef SATFast<SATBase::MMIRegClassAcc>	FastMMIRegClassSAT;

typedef refcount_ptr<FastMeanVarianceSAT>	FastMeanVarianceSATPtr;
typedef refcount_ptr<FastRegClassSAT>		FastRegClassSATPtr;
typedef refcount_ptr<FastMaxMutualInfoSAT>	FastMaxMutualInfoSATPtr;
typedef refcount_ptr<FastMMIRegClassSAT>	FastMMIRegClassSATPtr;

/*@}*/

#endif
