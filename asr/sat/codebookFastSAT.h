//                              -*- C++ -*-
//
//                              Millennium
//                   Distant Speech Recognition System
//                                 (dsr)
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


#ifndef _codebookFastSAT_h_
#define _codebookFastSAT_h_

#include "train/codebookTrain.h"
#include "adapt/transform.h"

/**
* \addtogroup Codebook
*/
/*@{*/

/**
* \defgroup CodebookSATGroup Codebook structures to perform ML and discriminative training of Gaussian mixture components.
* This group of classes provides the capability for performing ML speaker adaptation
* and estimation of Gaussian mixture components.
*/
/*@{*/

// ----- definition for class `CodebookFastSAT' -----
// 
class CodebookFastSAT : public CodebookTrain {
 public:
  CodebookFastSAT(const String& nm, UnShrt rfN, UnShrt dmN, CovType cvTyp = COV_DIAGONAL,
		  VectorFloatFeatureStreamPtr feat = NULL);

  CodebookFastSAT(const String& nm, UnShrt rfN, UnShrt dmN, CovType cvTyp = COV_DIAGONAL,
		  const String& featureName = "");

  ~CodebookFastSAT();

  class GaussDensity;  friend class GaussDensity;
  class Iterator;      friend class Iterator;
  class FastAccu;      friend class FastAccu;

  typedef Inherit<FastAccu, CodebookTrain::AccuPtr> FastAccuPtr;

  void reallocFastAccu(const TransformerTreePtr& tree, unsigned olen);
  void zeroFastAccu();

  void  saveFastAccu(FILE* fp) const;
  void  loadFastAccu(FILE* fp, float addFactor = 1.0, const String& name = "");

  const FastAccuPtr& fastAccu() const { return _fastAccu; }

 private:
  FastAccuPtr	_fastAccu;  // accumulator structure for fast SAT data accumulation
};

typedef Inherit<CodebookFastSAT, CodebookTrainPtr> CodebookFastSATPtr;


typedef const float* ListPtr;


// ----- definition for class `CodebookFastSAT::GaussDensity' -----
// 
class CodebookFastSAT::GaussDensity : public CodebookTrain::GaussDensity {
  static const double Small_ck;
 public:
  GaussDensity(CodebookFastSATPtr& cbs, int refX) :
    CodebookTrain::GaussDensity(cbs, refX) { }

  void extendMean(int len);
  void replaceIndex(int bestIndex, int index1, int index2);

  class RegClassScore {
  public:
    RegClassScore(UnShrt rc, double sc) : _regClass(rc), _score(sc) { }

    UnShrt      _regClass;
    double      _score;
  };
  typedef vector<RegClassScore>   RCArray;
  typedef RCArray::const_iterator RCAIter;

  void setRegClass(UnShrt reg = 1);
  void setRegClasses(const RCArray& rcsarray);
  inline UnShrt noRegClasses() const;
  CodebookBasic::GaussDensity::regClass;
  inline UnShrt regClass(UnShrt index) const;

  void replaceRegClass(UnShrt best, UnShrt nextBest);
  void fixGConsts();
  UnShrt* descLen();

  int findFastAccX(UnShrt rClass);

  FastAccuPtr& fastAccu() {
    assert(cbk()->_fastAccu.isNull() == false);
    return cbk()->_fastAccu;
  }
  const FastAccuPtr& fastAccu() const {
    assert(cbk()->_fastAccu.isNull() == false);
    return cbk()->_fastAccu;
  }

  ListPtr listPtr() const { return cbk()->_cv[refX()]->data; }

  inline const DNaturalVector fastMean(int accX = 0) const;
  inline const DNaturalVector fastVar(int accX = 0) const;
  inline const double* fastScalar(int accX = 0) const;

  inline const int fastNBlks() const;

  void normFastAccu(const TransformerTreePtr& tree, unsigned olen);

 protected:
        CodebookFastSATPtr&	cbk()       { return Cast<CodebookFastSATPtr>(_cbk()); }
  const CodebookFastSATPtr&	cbk() const { return Cast<CodebookFastSATPtr>(_cbk()); }

  CodebookBasic::GaussDensity::refX;

  static const int				FastAccNotPresent;

 private:
  // static members for fast SAT estimation
  void _checkBlocks(UnShrt nblk, int olen);
  void _allocWorkSpace();
  void _deallocWorkSpace();

  static gsl_matrix*				temp1;

  static gsl_vector*				vec1;
  static gsl_vector*				vec2;

  static UnShrt					_blockLen;
  static UnShrt					_origBlockLen;
  static UnShrt					_nBlocks;

  static bool					_allocated;
};

UnShrt CodebookFastSAT::GaussDensity::noRegClasses() const {
  if ( cbk()->_regClass && cbk()->_regClass[refX()] && cbk()->_regClass[refX()][0] > 0 )
    return cbk()->_regClass[refX()][0];
  else
    return 1;
}

UnShrt CodebookFastSAT::GaussDensity::regClass(UnShrt index) const {
  if ( cbk()->_regClass && cbk()->_regClass[refX()] && cbk()->_regClass[refX()][0] > 0 )
    return cbk()->_regClass[refX()][index+1];
  else
    return 0;
}


// ----- definition for class `CodebookFastSAT::FastAccu' -----
// 
class CodebookFastSAT::FastAccu : public CodebookTrain::Accu {
  friend void CodebookFastSAT::GaussDensity::normFastAccu(const TransformerTreePtr& tree, unsigned olen);
 public:
  FastAccu(UnShrt sbN, UnShrt dmN, UnShrt subFeatN, UnShrt odmN, UnShrt rfN, UnShrt nblk, CovType cvTyp);
  virtual ~FastAccu();
  void zero();

  void save(FILE* fp, const String& name);
  void load(FILE* fp, float addFactor = 1.0, const String& name = "");

  virtual unsigned size(const String& name, bool onlyCountsFlag = false);

  const DNaturalVector fastMean(UnShrt refX, int accX = 0) const {
    assert(accX >= 0 && accX < _subN);
    return DNaturalVector(_rv[accX]->data + refX * _rv[accX]->size2, _rv[accX]->size2);
  }
  inline const DNaturalVector fastVar(UnShrt refX, int accX = 0) const;
  const double* fastScalar(UnShrt refX, int accX = 0) const {
    assert(accX >= 0 && accX < _subN);
    return _scalar[accX][refX];
  }

  UnShrt nblks()   const { return _nblks;   }
  UnShrt orgDimN() const { return _orgDimN; }

 private:
  void		_saveInfo(const String& name, FILE* fp);
  void		_save(FILE* fp);

  bool		_loadInfo(FILE* fp);
  void		_load(FILE* fp, float addFactor, bool onlyCountsFlag);
  
  UnShrt	_orgDimN;	// original dimension of features
  double***	_scalar;	// scalar for fast SAT
  UnShrt	_nblks;		// number of sub-blocks for fast SAT
};

const DNaturalVector CodebookFastSAT::GaussDensity::fastMean(int accX) const 
{
  return fastAccu()->fastMean(refX(), accX);
}

const DNaturalVector CodebookFastSAT::GaussDensity::fastVar(int accX) const 
{
  return fastAccu()->fastVar(refX(), accX);
}

const double* CodebookFastSAT::GaussDensity::fastScalar(int accX) const 
{
  return fastAccu()->fastScalar(refX(), accX);
}

const int CodebookFastSAT::GaussDensity::fastNBlks() const
{
  return fastAccu()->nblks();
}

const DNaturalVector CodebookFastSAT::FastAccu::fastVar(UnShrt refX, int accX) const {
  assert(accX >= 0 && accX < _subN);
  return DNaturalVector(_sumOsq[accX][refX]->data, _dimN, _nblks);
}


// ----- definition for class `CodebookFastSAT::Iterator' -----
//
class CodebookFastSAT::Iterator : public CodebookTrain::Iterator {
 public:
  Iterator(CodebookFastSATPtr& cb) :
    CodebookTrain::Iterator(cb) { }

  GaussDensity mix() {
    return GaussDensity(cbk(), _refX);
  }

 protected:
        CodebookFastSATPtr& cbk()       { return Cast<CodebookFastSATPtr>(_cbk()); }
  const CodebookFastSATPtr& cbk() const { return Cast<CodebookFastSATPtr>(_cbk()); }
};


// ----- definition for class `CodebookSetFastSAT' -----
//
class CodebookSetFastSAT : public CodebookSetTrain {
 public:
  CodebookSetFastSAT(const String& descFile = "", FeatureSetPtr& fs = NullFeatureSetPtr, const String& cbkFile = "");
  virtual ~CodebookSetFastSAT() { }

  class CodebookIterator;  friend class CodebookIterator;
  class GaussianIterator;  friend class GaussianIterator;

  // calculate total description length
  unsigned descLength();

  // clear fast SAT accumulators
  void zeroFastAccus(unsigned nParts = 1, unsigned part = 1);

  // set all regression classes to one
  void setRegClassesToOne();

  // normalize fast SAT accumulators
  void normFastAccus(const TransformerTreePtr& paramTree, unsigned olen = 0);

  // save fast SAT accumulators
  void saveFastAccus(const String& fileName) const;

  // load fast SAT accumulators
  void loadFastAccus(const String& fileName, unsigned nParts, unsigned part);

  // force re-allocation of fast SAT accumulators
  void reallocFastAccus() { _zeroFastAccs = true; }

        CodebookFastSATPtr& find(const String& key)       { return Cast<CodebookFastSATPtr>(_find(key)); }
  const CodebookFastSATPtr& find(const String& key) const { return Cast<CodebookFastSATPtr>(_find(key)); }

        CodebookFastSATPtr& find(unsigned cbX)            { return Cast<CodebookFastSATPtr>(_find(cbX)); }
  const CodebookFastSATPtr& find(unsigned cbX)      const { return Cast<CodebookFastSATPtr>(_find(cbX)); }

 protected:
  virtual void _addCbk(const String& name, UnShrt rfN, UnShrt dmN, CovType cvTp,
		       const String& featureName);

  virtual void _addCbk(const String& name, UnShrt rfN, UnShrt dmN, CovType cvTp,
		       VectorFloatFeatureStreamPtr feat);

 private:
  bool	_zeroFastAccs;
};

typedef Inherit<CodebookSetFastSAT, CodebookSetTrainPtr> CodebookSetFastSATPtr;

void replaceIndex(CodebookSetFastSATPtr& cbs, int bestIndex, int index1, int index2);


// ----- definition for container class `CodebookSetFastSAT::CodebookIterator' -----
//
class CodebookSetFastSAT::CodebookIterator : private CodebookSetTrain::CodebookIterator {
  friend class CodebookSetFastSAT;
 public:
  CodebookIterator(CodebookSetFastSATPtr& cbs, int nParts = 1, int part = 1)
    : CodebookSetTrain::CodebookIterator(cbs, nParts, part) { }

  CodebookSetBasic::CodebookIterator::operator++;
  CodebookSetBasic::CodebookIterator::more;

  CodebookFastSATPtr next() {
    if (more()) {
      CodebookFastSATPtr ret = cbk();
      operator++(1);
      return ret;
    } else {
      throw jiterator_error("end of codebook!");
    }
  }

  	CodebookFastSATPtr& cbk()       { return Cast<CodebookFastSATPtr>(_cbk()); }
  const CodebookFastSATPtr& cbk() const { return Cast<CodebookFastSATPtr>(_cbk()); }

 protected:
  CodebookSetBasic::CodebookIterator::_cbk;
};


// ----- definition for container class `CodebookSetFastSAT::GaussianIterator' -----
//
class CodebookSetFastSAT::GaussianIterator : protected CodebookSetTrain::GaussianIterator {
  typedef CodebookFastSAT::GaussDensity GaussDensity;
 public:
  GaussianIterator(CodebookSetFastSATPtr& cbs, int nParts = 1, int part = 1)
    : CodebookSetTrain::GaussianIterator(cbs, nParts, part) { }

  CodebookSetTrain::GaussianIterator::operator++;
  CodebookSetTrain::GaussianIterator::more;

  GaussDensity mix() { return GaussDensity(cbk(), refX()); }

 protected:
        CodebookFastSATPtr& cbk()       { return Cast<CodebookFastSATPtr>(_cbk()); }
  const CodebookFastSATPtr& cbk() const { return Cast<CodebookFastSATPtr>(_cbk()); }
};

/*@}*/

/*@}*/

#endif
