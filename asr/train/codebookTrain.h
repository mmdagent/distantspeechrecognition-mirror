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

#ifndef _codebookTrain_h_
#define _codebookTrain_h_

#include "adapt/codebookAdapt.h"
#include "dictionary/distribTree.h"
#include "fsm/fsm.h"


/**
* \addtogroup Codebook
*/
/*@{*/

/**
* \defgroup CodebookTrainGroup Codebook Train Group
* This group of classes provides the capability for performing ML speaker adaptation
* and estimation of Gaussian mixture components.
*/
/*@{*/

/**
* \defgroup CodebookTrain Codebook Train
*/
/*@{*/

// ----- definition for class `CodebookTrain' -----
// 
class CodebookTrain : public CodebookAdapt {
  static const float gblVarFloor;
  static const float floorRatio;
  static const int   MinFloorCount;
 public:
  CodebookTrain(const String& nm, UnShrt rfN, UnShrt dmN, CovType cvTp = COV_DIAGONAL,
		VectorFloatFeatureStreamPtr feat = NULL);

  CodebookTrain(const String& nm, UnShrt rfN, UnShrt dmN, CovType cvTp = COV_DIAGONAL,
		const String& featureName = "");

  virtual ~CodebookTrain();

  class GaussDensity;	friend class GaussDensity;
  class Iterator;	friend class Iterator;
  class Accu;		friend class Accu;

  typedef refcount_ptr<Accu>	AccuPtr;

        AccuPtr&  accu()        { return _accu; }
  const AccuPtr&  accu()  const { return _accu; }

  // update codebook means and covariances with ML criterion
  void  update(bool verbose = false);

  // update codebook means and covariances with MMI criterion
  void  updateMMI(double E = 1.0);

  // allocate codebook accumulator
  void allocAccu();

  // save codebook accumulator
  void  saveAccu(FILE* fp, bool onlyCountsFlag = false) const;

  // load codebook accumulators
  void  loadAccu(FILE* fp, float factor = 1.0);

  // clear accumulator
  void zeroAccu();

  // accumulate one frame of data
  float accumulate(float factor, const float* dsVal, int frameX, float* mixVal);

  // assign a floor value to all variances
  void floorVariances(unsigned& ttlVarComps, unsigned& ttlFlooredVarComps);

  // split 'addN' Gaussians with most counts
  unsigned split(float minCount = 0.0, float splitFactor = 0.2);

  // update the Gaussian normalization constant
  void fixGConsts();

  // invert covariances
  void invertVariance();

 private:
  void _split(unsigned refX, float splitFactor);

  AccuPtr 		_accu;		// accumulator structure for training data accumulation
};

typedef Inherit<CodebookTrain, CodebookAdaptPtr> CodebookTrainPtr;

/*@}*/

/**
* \defgroup CodebookTrainAccu WFSTAddSelfLoops Add self-loops
*/
/*@{*/

// ----- definition for class `CodebookTrain::Accu' -----
// 
class CodebookTrain::Accu {
  friend class CodebookTrain::GaussDensity;
  friend class CodebookFastSAT;
 public:
  Accu(UnShrt sbN, UnShrt dmN, UnShrt subFeatN, UnShrt rfN, CovType cvTyp, UnShrt odmN = 0);
  virtual ~Accu();

  void zero();
  void save(FILE* fp, const String& name, bool onlyCountsFlag = false);
  void load(FILE* fp, float factor = 1.0);
  void add(const AccuPtr& ac, double factor = 1.0);
  void update(float* count, gsl_matrix_float* rv, gsl_vector_float** cv);
  void updateMMI(float* count, gsl_matrix_float* rv, gsl_vector_float** cv, double E);

  virtual unsigned size(const String& name, bool onlyCountsFlag = false);
  bool zeroOccupancy();

  double postProb(UnShrt refX) const {
    return (_count[_subX][refX] - _denCount[_subX][refX]);
  }
  double numProb(UnShrt refX)             const { return _count[_subX][refX]; }
  double denProb(UnShrt refX)             const { return _denCount[_subX][refX]; }
  double sumO(UnShrt refX, UnShrt idim)   const { return gsl_matrix_get(_rv[_subX], refX, idim); }
  double sumOsq(UnShrt refX, UnShrt idim) const { return gsl_vector_get(_sumOsq[_subX][refX], idim); }

  inline const DNaturalVector sumO(UnShrt refX) const;
  inline const DNaturalVector sumOsq(UnShrt refX) const;
  void accumulate(const float* pattern, float factor, const float* addCount);

  void addCount(unsigned accX, unsigned refX, double prob) { _count[accX][refX] += prob; }
  void addDenCount(unsigned accX, unsigned refX, double prob) { _denCount[accX][refX] += prob; }

  const UnShrt          subN() const { return _subN; }
  const gsl_matrix**      rv() const { return (const gsl_matrix**) _rv; }
  const gsl_vector*** sumOsq() const { return (const gsl_vector***) _sumOsq; }

  static double MinimumCount;

 protected:
  static const int MarkerMagic;

  void			_dumpMarker(FILE* fp);
  void			_checkMarker(FILE* fp);

  void			_saveInfo(const String& name, int countsOnlyFlag, FILE* fp);
  void			_save(bool onlyCountsFlag, FILE* fp);

  bool			_loadInfo(FILE* fp);
  void			_load(FILE* fp, float factor, bool onlyCountsFlag);

  void			_updateCount(float* count);
  void			_updateMean(const float* count, gsl_matrix_float* rv);
  void			_updateCovariance(const float* count, const gsl_matrix_float* rv, gsl_vector_float** cv);
  void			_updateMMI(const float* count, gsl_matrix_float* rv, gsl_vector_float** cv, double E);

  unsigned		_size(const String& name, UnShrt meanLen, UnShrt subN, bool onlyCountsFlag);
  void 			_saveCovariance(unsigned subX, unsigned refX, FILE* fp);
  void			_loadCovariance(unsigned subX, unsigned refX, FILE* fp, float factor);
  void			_addCovariance(unsigned subX, unsigned refX, const float* pattern, float factor);

  UnShrt		_subN;		// how many sub-accumulators are we using
  UnShrt		_dimN;		// dimension of features
  UnShrt		_subFeatN;	// number of sub-features
  UnShrt		_refN;		// number of Gaussian components
  CovType		_covType;	// type of covariance matrix

  double**		_count;		// count[i][j]  = training counts for j-th vector in i-th sub-accum
  double**		_denCount;	// count[i][j]  = training counts for j-th vector in i-th sub-accum
  gsl_matrix**		_rv;		// rv[i]->matPA[j][k] = k-th coeff of j-th vector in i-th sub-accum
  gsl_vector***		_sumOsq;	// sumOsq[i][j] = sum of squares for j-th vector in i-th sub-accum

 private:
  Accu(const Accu& ac);
  Accu& operator=(const Accu& ac);

  UnShrt		_subX;
};

const DNaturalVector CodebookTrain::Accu::sumO(UnShrt refX) const {
  return DNaturalVector(_rv[_subX]->data + refX * _rv[_subX]->size2, _dimN, _subFeatN);
}

const DNaturalVector CodebookTrain::Accu::sumOsq(UnShrt refX) const {
  return DNaturalVector(_sumOsq[_subX][refX]->data, _dimN, _subFeatN);
}

/*@}*/

/**
* \defgroup CodebookTrainGaussDensity CodebookTrain::GaussDensity
*/
/*@{*/

// ----- definition for class `CodebookTrain::GaussDensity' -----
//
class CodebookTrain::GaussDensity : public CodebookAdapt::GaussDensity {
 public:
  GaussDensity(CodebookTrainPtr& cb, int refX)
    : CodebookAdapt::GaussDensity(cb, refX) { }

  void allocMean();
  float invVar(int i) const { return cbk()->_cv[refX()]->data[i]; }
  inline const InverseVariance invVar() const;

  double postProb()          const { return accu()->postProb(refX()); }
  double numProb()           const { return accu()->numProb(refX()); }
  double denProb()           const { return accu()->denProb(refX()); }
  double sumO(UnShrt idim)   const { return accu()->sumO(refX(), idim); }
  double sumOsq(UnShrt idim) const { return accu()->sumOsq(refX(), idim); }

  const DNaturalVector sumO()   const { return accu()->sumO(refX()); }
  const DNaturalVector sumOsq() const { return accu()->sumOsq(refX()); }

  // update gaussian constant based on new covariance matrix
  void fixGConst();

 protected:
        CodebookTrainPtr& cbk()	      { return Cast<CodebookTrainPtr>(_cbk()); }
  const CodebookTrainPtr& cbk() const { return Cast<CodebookTrainPtr>(_cbk()); }

  CodebookBasic::GaussDensity::refX;

  inline const AccuPtr& accu() const;
};

const InverseVariance CodebookTrain::GaussDensity::invVar() const 
{
  return NaturalVector(cbk()->_cv[refX()]->data, featLen(), 1);
}

inline const CodebookTrain::AccuPtr& CodebookTrain::GaussDensity::accu() const
{
  if (cbk()->_accu.isNull())
    throw jconsistency_error("Must allocate codebook accumulators.");
  return cbk()->_accu;
}

/*@}*/

/**
* \defgroup CodebookTrain::Iterator CodebookTrainIterator
*/
/*@{*/

// ----- definition for class `CodebookTrain::Iterator' -----
//
class CodebookTrain::Iterator : public CodebookAdapt::Iterator {
 public:
  Iterator(CodebookTrainPtr& cb) :
    CodebookAdapt::Iterator(cb) { }

  GaussDensity mix() {
    return GaussDensity(cbk(), _refX);
  }

 protected:
  	CodebookTrainPtr& cbk()       { return Cast<CodebookTrainPtr>(_cbk()); }
  const CodebookTrainPtr& cbk() const { return Cast<CodebookTrainPtr>(_cbk()); }
};

/*@}*/

/**
* \defgroup CodebookSetTrain Codebook Set Train
*/
/*@{*/

// ----- definition for class `CodebookSetTrain' -----
//
class CodebookSetTrain : public CodebookSetAdapt {
  friend class AccMap;
 public:
  CodebookSetTrain(const String& descFile = "", FeatureSetPtr fs = NullFeatureSetPtr, const String& cbkFile = "",
		   double massThreshhold = 5.0);
  virtual ~CodebookSetTrain() { }

  class CodebookIterator;  friend class CodebookIterator;
  class GaussianIterator;  friend class GaussianIterator;

  // allocate codebook accumulators
  void allocAccus();

  // save accumulators
  void saveAccus(const String& fileName, float totalPr = 0.0, unsigned totalT = 0,
		 bool onlyCountsFlag = false) const;

  // load accumulators
  void loadAccus(const String& fileName, float& totalPr, unsigned& totalT, float factor = 1.0,
		 unsigned nParts = 1, unsigned part = 1);
  void loadAccus(const String& fileName);

  // zero accumulators
  void zeroAccus(unsigned nParts = 1, unsigned part = 1);

  // update all codebooks with ML criterion
  void update(int nParts = 1, int part = 1, bool verbose = false);

  // update all codebooks with MMI criterion
  void updateMMI(int nParts = 1, int part = 1, double E = 1.0);

  // invert all variances
  void invertVariances(unsigned nParts = 1, unsigned part = 1);

  // update all gaussian constants based on new covariance matrices
  void fixGConsts(unsigned nParts = 1, unsigned part = 1);

  // assign a floor value to all variances
  void floorVariances(unsigned nParts = 1, unsigned part = 1);

        CodebookTrainPtr& find(const String& key)       { return Cast<CodebookTrainPtr>(_find(key)); }
  const CodebookTrainPtr& find(const String& key) const { return Cast<CodebookTrainPtr>(_find(key)); }

        CodebookTrainPtr& find(unsigned cbX)            { return Cast<CodebookTrainPtr>(_find(cbX)); }
  const CodebookTrainPtr& find(unsigned cbX)      const { return Cast<CodebookTrainPtr>(_find(cbX)); }

 protected:
  virtual void _addCbk(const String& name, UnShrt rfN, UnShrt dmN, CovType cvTp,
		       const String& featureName);

  virtual void _addCbk(const String& name, UnShrt rfN, UnShrt dmN, CovType cvTp,
		       VectorFloatFeatureStreamPtr feat);
};

typedef Inherit<CodebookSetTrain, CodebookSetAdaptPtr> CodebookSetTrainPtr;

/*@}*/

/**
* \defgroup CodebookSetTrain::CodebookIterator CodebookSetTrain CodebookIterator
*/
/*@{*/

// ----- definition for container class `CodebookSetTrain::CodebookIterator' -----
//
class CodebookSetTrain::CodebookIterator : private CodebookSetAdapt::CodebookIterator {
  friend class CodebookSetTrain;
 public:
  CodebookIterator(CodebookSetTrainPtr& cbs, int nParts = 1, int part = 1)
    : CodebookSetAdapt::CodebookIterator(cbs, nParts, part) { }

  CodebookSetBasic::CodebookIterator::operator++;
  CodebookSetBasic::CodebookIterator::more;

  CodebookTrainPtr next() {
    if (more()) {
      CodebookTrainPtr cb(cbk());
      operator++(1);
      return cb;
    } else {
      throw jiterator_error("end of codebooks!");
    }
  }

        CodebookTrainPtr& cbk()       { return Cast<CodebookTrainPtr>(_cbk()); }
  const CodebookTrainPtr& cbk() const { return Cast<CodebookTrainPtr>(_cbk()); }

 protected:
  CodebookSetBasic::CodebookIterator::_cbk;
};

/*@}*/

/**
* \defgroup CodebookSetTrain::GaussianIterator CodebookSetTrain GaussianIterator
*/
/*@{*/

// ----- definition for container class `CodebookSetTrain::GaussianIterator' -----
//
class CodebookSetTrain::GaussianIterator : protected CodebookSetAdapt::GaussianIterator {
  typedef CodebookTrain::GaussDensity GaussDensity;
 public:
  GaussianIterator(CodebookSetTrainPtr& cbs, int nParts = 1, int part = 1)
    : CodebookSetAdapt::GaussianIterator(cbs, nParts, part) { }

  CodebookSetAdapt::GaussianIterator::operator++;
  CodebookSetAdapt::GaussianIterator::more;

  GaussDensity mix() { return GaussDensity(cbk(), refX()); }

 protected:
        CodebookTrainPtr& cbk()       { return Cast<CodebookTrainPtr>(_cbk()); }
  const CodebookTrainPtr& cbk() const { return Cast<CodebookTrainPtr>(_cbk()); }
};

/*@}*/

/*@}*/

/*@}*/

#endif
