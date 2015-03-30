//                              -*- C++ -*-
//
//                              Millennium
//                  Automatic Speech Recognition System
//                                 (asr)
//
//  Module:  asr.adapt
//  Purpose: Maximum likelihood model space adaptation.
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


#ifndef _transform_h_
#define _transform_h_

using namespace std;

#include <assert.h>
#include <complex>
#include <map>
#include <vector>
#include <list>
#include <iomanip>
#include <iostream>

#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

#include "adapt/codebookAdapt.h"

typedef complex<double> Complex;

typedef enum { RAPT=0, SLAPT, MLLR, STC, LDA, NoAdaptation, Unspecified } AdaptationType;

AdaptationType getAdaptType(const char* ctype);

bool isAncestor(unsigned ancestor, unsigned child);


// ----- definition for class `CoeffSequence' -----
// 
class CoeffSequence {
  typedef vector<double> _CoeffSequence;
 public:
  CoeffSequence(UnShrt sz = 0);
  CoeffSequence(const NaturalVector& v);
  CoeffSequence(const String& fileName);
  
  CoeffSequence& operator=(double v);
  CoeffSequence& operator=(const NaturalVector& v);

  UnShrt  size()               const { return _coeff.size(); }
  bool    isZero()             const { return _coeff.size() == 0; }
  void    resize(UnShrt sz)          { _coeff.resize(sz); }
  void    zero();
  
  double  operator[](UnShrt i) const { return _coeff[i]; }
  double& operator[](UnShrt i)       { return _coeff[i]; }
  double  operator()(UnShrt i) const { return _coeff[i]; }
  double& operator()(UnShrt i)       { return _coeff[i]; }

private:
  _CoeffSequence _coeff;
};

// class to store the coefficients of a Laurent
// (ie, "double-sided") series
//
static const int DefaultSeriesLength = 150;

class LaurentSeries {
 public:
  LaurentSeries();
  ~LaurentSeries();

  inline double operator[](int index) const;
  inline double& operator[](int index);

  LaurentSeries& operator=(const LaurentSeries& rhs);
  LaurentSeries& operator=(double val);

  int len()  const { return _seqLength;   }
  int last() const { return _seqLength-1; }	// index of last coefficient

  friend void multAdd(double factor, const LaurentSeries& seq, LaurentSeries& out);
  friend void shift(LaurentSeries& seq);
  friend void CauchyProduct(const LaurentSeries& inSeq1,
			    const LaurentSeries& inSeq2, LaurentSeries& outSeq);
  friend ostream& operator<<(ostream& os, const LaurentSeries& seq);

private:
  const int					_seqLength; // length of coeff. sequence with pos. indices
  double*					_seq;
};

double LaurentSeries::operator[](int index) const {
  assert(index >= -last() && index <= last());
  return _seq[index];
}

double& LaurentSeries::operator[](int index) {
  assert(index >= -last() && index <= last());
  return _seq[index];
}


// ----- definition for class `TransformMatrix' -----
//
class TransformMatrix {
  static const UnShrt MaxSubBlocks = 3;
 public:
  ~TransformMatrix();

  bool isZero() const { return _nBlocks == 0; }
  inline const gsl_matrix* matrix(UnShrt index = 0) const;
  inline const gsl_vector* offset(UnShrt index = 0) const;
  UnShrt nBlocks() const { return _nBlocks; }
  
  void _initialize(UnShrt sbFtSz, UnShrt nSubFt, UnShrt orgSbFtSz);

 protected:
  TransformMatrix();

 private:
  UnShrt					_nBlocks;
  gsl_matrix**					_matrix;
  gsl_vector**					_offset;
};

const gsl_matrix* TransformMatrix::matrix(UnShrt index) const
{
  if (index >= _nBlocks)
    throw jindex_error("Block index %d exceeds maximum %d.", index, _nBlocks);

  return _matrix[index];
}

const gsl_vector* TransformMatrix::offset(UnShrt index) const
{
  if (index >= _nBlocks)
    throw jindex_error("Block index %d exceeds maximum %d.", index, _nBlocks);

  return _offset[index];
}


// ----- definition for class `TransformBase' -----
//
static const int gS = 1;      // number of data streams
class ParamBase;

class TransformBase : protected TransformMatrix {
 public:
  TransformBase();
  TransformBase(UnShrt sbFtSz, UnShrt nSubFt, UnShrt orgSbFtSz = 0,
		int trace = 0x0000);
  TransformBase(const ParamBase& par,
		UnShrt sbFtSz, UnShrt nSubFt, UnShrt orgSbFtSz = 0,
		int trace = 0x0000);
  virtual ~TransformBase();

  virtual void transform(const NaturalVector& f, NaturalVector& t,
			 bool useBias = true) const = 0;
  virtual void transform(CodebookAdapt::GaussDensity& mix) const;
  const TransformMatrix& transformMatrix() const {
    return *((TransformMatrix*) this);
  }

  UnShrt subFeatLen()    const { return _subFeatLen; }
  UnShrt orgSubFeatLen() const { return _orgSubFeatLen; }
  UnShrt nSubFeat()      const { return _nSubFeat; }
  UnShrt featLen()       const { return _featLen; }
  UnShrt orgFeatLen()    const { return _orgSubFeatLen * _nSubFeat; }

 protected:
  const UnShrt					_subFeatLen;
  const UnShrt					_nSubFeat;
  const UnShrt					_featLen;
  const UnShrt					_orgSubFeatLen;

  mutable NaturalVector				_transFeat;

  NaturalVector					_bias;
};

typedef refcount_ptr<TransformBase> TransformBasePtr;


// ----- definition for class `APTTransformerBase' -----
//
class APTTransformerBase : virtual public TransformBase {
 public:
  virtual ~APTTransformerBase();

  virtual void transform(const NaturalVector& f, NaturalVector& t,
			 bool useBias = true) const;
  virtual void unitCirclePlot(const String& fileName) = 0;

  UnShrt cepSubFeatLen()    const { return _cepSubFeatLen; }
  UnShrt cepOrgSubFeatLen() const { return _cepOrgSubFeatLen; }
  UnShrt cepNSubFeat()      const { return _cepNSubFeat; }
  UnShrt cepFeatLen()       const { return _cepSubFeatLen * _cepNSubFeat; }

 protected:
  APTTransformerBase(UnShrt nParams = 0);
  APTTransformerBase(UnShrt sbFtSz, UnShrt nSubFt, UnShrt orgSbFtSz = 0,
		     UnShrt noParams = 1, int trace = 0x0000,
		     const String& ldaFile = "", UnShrt ldaFtSize = 32);
  APTTransformerBase(const ParamBase& par,
		     UnShrt sbFtSz, UnShrt nSubFt, UnShrt orgSbFtSz = 0,
		     int trace = 0x0000, const String& ldaFile = "", UnShrt ldaFtSize = 32);

  virtual void	_calcSequence(const CoeffSequence& _params,
			      LaurentSeries& _sequence) = 0;
  void		_calcTransMatrix(const LaurentSeries& sequence);
  void		_printSum(const LaurentSeries& seq);
  Complex	_calcMappedPoint(const Complex& z, const LaurentSeries& _seq);
  void		_unitCirclePlot(FILE* fp, const LaurentSeries& coeffs);
  void		_initTransformMatrix();
  void		_ldaMatrix(const gsl_matrix* cepMatrix, gsl_matrix* ldaMatrix);

  const UnShrt					NoAllPassParams;
  CoeffSequence					params;
  CoeffSequence					p;

  static int					_cnt;
  static LaurentSeries*				_qmn;            // stores sequences q^{(m)}[n]
  static gsl_matrix*				_cepTransMatrix; // cepstral sub-feature transform matrix
  static gsl_matrix_float*			_lda;

  gsl_matrix*					_transMatrix;    // sub-feature transform matrix
  LaurentSeries					_laurentCoeffs;

  static const int				NoPlotPoints;

  const UnShrt					_cepSubFeatLen;
  const UnShrt					_cepOrgSubFeatLen;
  const UnShrt					_cepNSubFeat;
};


// ----- definition for class `ParamBase' -----
//
class ParamBase;
typedef refcount_ptr<ParamBase> ParamBasePtr;
class ParamBase {
  friend class ParamTree;
  friend class BLTEstimatorAdapt;  // why must all these be friends ?
  friend class APTEstimatorAdaptBase;
  friend class MLLREstimatorAdapt;

  static const char* _symMap[];
  static const char* _adaptMap[];
 public:
  virtual ~ParamBase() { }

  unsigned regClass()           const { return _regClass; }
  double   bias(unsigned index) const { return _bias[index]; }
  unsigned biasSize()           const { return _bias.size(); }

  virtual UnShrt size() const = 0;
  virtual bool isZero() const = 0;
  virtual ParamBase* copy() const = 0;
  void write(FILE* fp) const;

  static ParamBasePtr param(FILE* fp);
  static ParamBasePtr param(const String& fileName, const String& spkr);

  virtual const TransformBasePtr
    transformer(UnShrt sbFtSz, UnShrt nSbFt, UnShrt orgSbFtSz = 0,
		int trace = 0, const String& ldaFile = "", UnShrt ldaFeatLen = 0) const = 0;

 protected:
  typedef enum { RegClass=0, RAPTParams, SLAPTParams, MLLRParams, STCParams, LDAParams, BiasOffset,
		 EndClass, AdaptType, EOFSymbol, NullSymbol } Symbol;
  ParamBase();
  ParamBase(unsigned rc, unsigned b_sz);
  ParamBase(unsigned rc, const NaturalVector& bs);

  virtual AdaptationType _type() const = 0;
  void			_readParams(FILE* fp);

  static AdaptationType	_getAdaptType(FILE* fp);
  static Symbol		_getSymbol(FILE* fp);

  void			_putAdaptType(FILE* fp) const;
  void			_putSymbol(FILE* fp, Symbol sym) const;

  virtual void		_getParams(FILE* fp) = 0;
  virtual void		_putParams(FILE* fp) const = 0;

  void			_putEndClass(FILE* fp) const;

  void			_getRegClass(FILE* fp);
  void			_putRegClass(FILE* fp) const;

  void			_getBiasOffset(FILE* fp);
  void			_putBiasOffset(FILE* fp) const;

  void _setClass(unsigned rc) { _regClass = rc; }

  unsigned		_regClass;
  CoeffSequence		_bias;
};


// ----- definition for class `APTParamBase' -----
//
class APTParamBase : public ParamBase {
  friend class BLTEstimatorAdapt;
  friend class APTEstimatorAdaptBase;
 public:
  virtual ~APTParamBase() { }
  double   operator[](unsigned index) const { return _conf[index]; }
  double&  operator[](unsigned index)       { return _conf[index]; }
  virtual  UnShrt size() 	      const { return _conf.size(); }

  void         resize(int sz) { _conf.resize(sz); }
  virtual bool isZero() const { return _conf.size() == 0; }

  virtual void unitCirclePlot(const String& fileName) const = 0;

 protected:
  APTParamBase(unsigned c_sz = 0, unsigned b_sz = 0, unsigned rc = 0);
  APTParamBase(double alpha, const NaturalVector& bias, unsigned rc);
  APTParamBase(const CoeffSequence& seq, const NaturalVector& bias, unsigned rc);

  virtual void _getParams(FILE* fp);
  virtual void _putType(FILE* fp) const = 0;

  CoeffSequence					_conf;
};

typedef Inherit<APTParamBase, ParamBasePtr> APTParamBasePtr;


// ----- definition for class `RAPTParam' -----
//
class RAPTParam : public APTParamBase {
 public:
  RAPTParam(FILE* fp);
  RAPTParam(const String& paramFile);
  RAPTParam(unsigned c_sz = 0, unsigned b_sz = 0, unsigned rc = 0)
    : APTParamBase(c_sz, b_sz, rc) { }
  RAPTParam(double alpha, const NaturalVector& bias, unsigned rc = 0)
    : APTParamBase(alpha, bias, rc) { }
  RAPTParam(const CoeffSequence& seq, const NaturalVector& bias, unsigned rc = 0)
    : APTParamBase(seq, bias, rc) { }

  virtual const TransformBasePtr
  transformer(UnShrt sbFtSz, UnShrt nSbFt, UnShrt orgSbFtSz = 0,
	      int trace = 0, const String& ldaFile = "", UnShrt ldaFeatLen = 0) const;

  virtual void unitCirclePlot(const String& fileName) const;
  virtual RAPTParam* copy() const { return new RAPTParam(*this); }

 protected:
  virtual AdaptationType	_type()              const { return RAPT; }
  virtual void			_putType(FILE* fp)   const { _putSymbol(fp, RAPTParams); }
  virtual void			_putParams(FILE* fp) const;
};

typedef Inherit<RAPTParam, APTParamBasePtr> RAPTParamPtr;


// ----- definition for class `SLAPTParam' -----
//
class SLAPTParam : public APTParamBase {
 public:
  SLAPTParam(FILE* fp);
  SLAPTParam(const String& paramFile);
  SLAPTParam(unsigned c_sz = 0, unsigned b_sz = 0, unsigned rc = 0)
    : APTParamBase(c_sz, b_sz, rc) { }
  SLAPTParam(double alpha, const NaturalVector& bias, unsigned rc = 0)
    : APTParamBase(alpha, bias, rc) { }
  SLAPTParam(const CoeffSequence& seq, const NaturalVector& bias, unsigned rc = 0)
    : APTParamBase(seq, bias, rc) { }

  virtual const TransformBasePtr
  transformer(UnShrt sbFtSz, UnShrt nSbFt, UnShrt orgSbFtSz = 0,
	      int trace = 0, const String& ldaFile = "", UnShrt ldaFeatLen = 0) const;

  virtual void unitCirclePlot(const String& fileName) const;
  virtual SLAPTParam* copy() const { return new SLAPTParam(*this); }

 protected:
  virtual AdaptationType	_type()              const { return SLAPT; }
  virtual void			_putType(FILE* fp)   const { _putSymbol(fp, SLAPTParams); }
  virtual void			_putParams(FILE* fp) const;
};

typedef Inherit<SLAPTParam, APTParamBasePtr> SLAPTParamPtr;


// ----- definition for class `MLLRParam' -----
//
class MLLRParam : public ParamBase {
  friend class MLLREstimatorAdapt;
 public:
  MLLRParam(FILE* fp = NULL);
  MLLRParam(const MLLRParam& m);
  ~MLLRParam();

  MLLRParam& operator=(const MLLRParam& m);

  virtual UnShrt size() const { return _size; }

  virtual const TransformBasePtr
    transformer(UnShrt sbFtSz, UnShrt nSbFt, UnShrt orgSbFtSz = 0,
		int trace = 0, const String& ldaFile = "", UnShrt ldaFeatLen = 0) const;

  const   gsl_matrix*        matrix() const { return _matrix; }
  virtual bool        isZero() const { return (_matrix == NULL); }
  virtual MLLRParam*  copy()   const { return new MLLRParam(*this); }

 protected:
  virtual AdaptationType _type() const { return MLLR; }

 private:
  MLLRParam& operator=(const gsl_matrix* mat);

  virtual void _getParams(FILE* fp);
  virtual void _putParams(FILE* fp) const;

  UnShrt					_size;
  gsl_matrix*					_matrix;
};

typedef Inherit<MLLRParam, ParamBasePtr> MLLRParamPtr;


// ----- definition for class `STCParam' -----
//
class STCParam : public ParamBase {
 public:
  STCParam(FILE* fp = NULL);
  STCParam(const STCParam& m);
  ~STCParam();

  STCParam& operator=(const STCParam& m);

  virtual UnShrt size() const { return _size; }

  virtual const TransformBasePtr
    transformer(UnShrt sbFtSz, UnShrt nSbFt, UnShrt orgSbFtSz = 0,
		int trace = 0, const String& ldaFile = "", UnShrt ldaFeatLen = 0) const;

  const   gsl_matrix*        matrix() const { return _matrix; }
  virtual bool        isZero() const { return (_matrix == NULL); }
  virtual STCParam*   copy()   const { return new STCParam(*this); }

 protected:
  virtual AdaptationType _type() const { return STC; }

 private:
  STCParam& operator=(const gsl_matrix* mat);

  virtual void _getParams(FILE* fp);
  virtual void _putParams(FILE* fp) const;

  UnShrt					_size;
  gsl_matrix*					_matrix;
};

typedef Inherit<STCParam, ParamBasePtr> STCParamPtr;


// ----- definition for class `LDAParam' -----
//
class LDAParam : public ParamBase {
 public:
  LDAParam(FILE* fp = NULL);
  LDAParam(const LDAParam& m);
  ~LDAParam();

  LDAParam& operator=(const LDAParam& m);

  virtual UnShrt size() const { return _size; }

  virtual const TransformBasePtr
    transformer(UnShrt sbFtSz, UnShrt nSbFt, UnShrt orgSbFtSz = 0,
		int trace = 0, const String& ldaFile = "", UnShrt ldaFeatLen = 0) const;

  const   gsl_matrix* matrix() const { return _matrix; }
  virtual bool        isZero() const { return (_matrix == NULL); }
  virtual LDAParam*   copy()   const { return new LDAParam(*this); }

 protected:
  virtual AdaptationType _type() const { return LDA; }

 private:
  LDAParam& operator=(const gsl_matrix* mat);

  virtual void _getParams(FILE* fp);
  virtual void _putParams(FILE* fp) const;

  UnShrt					_size;
  gsl_matrix*					_matrix;
};

typedef Inherit<LDAParam, ParamBasePtr> LDAParamPtr;


// ----- definition for class `ParamTree' -----
//
class ParamTree {
  typedef map<unsigned, ParamBasePtr>		ParamMap;
  typedef ParamMap::iterator			ParamMapIter;
  typedef ParamMap::const_iterator		ParamMapConstIter;
  typedef ParamMap::value_type			ValueType;

 public:
  ParamTree(const String& fileName = "");
  ParamTree(const ParamTree& tree);
  ~ParamTree();

  class Iterator;       friend class Iterator;
  class ConstIterator;  friend class ConstIterator;

  ParamTree& operator=(const ParamTree& tree);
  void write(const String& fileName) const;
  void read(const String& fileName);
  AdaptationType type() const { return _type; }

  ParamBasePtr& find(unsigned initRC, bool useAncestor = true);
  APTParamBasePtr& findAPT(unsigned initRC, AdaptationType typ, bool useAncestor = true);
  MLLRParamPtr&    findMLLR(unsigned initRC, bool useAncestor = true);

  void applySTC(const String& stcFile);

  bool hasIndex(unsigned index) const { return (_map.find(index) != _map.end()); }

  void clear();

 private:
  ParamMap					_map;
  AdaptationType				_type;
};

typedef refcount_ptr<ParamTree> ParamTreePtr;


// ----- definition for class `ParamTree::Iterator' -----
//
class ParamTree::Iterator {
  friend class ParamTree;
 public:
  Iterator(ParamTreePtr& tree) :
    _map(tree->_map), _itr(_map.begin()), _type(tree->_type) { }

  ParamBasePtr& par() { return (*_itr).second; }
  inline APTParamBasePtr& apt();
  inline MLLRParamPtr& mllr();

  void operator++(int) { _itr++; }
  bool more() { return _itr != _map.end(); }
  unsigned regClass() const  { return (*_itr).first; }

 private:
  Iterator(ParamTree& tree) :
    _map(tree._map), _itr(_map.begin()), _type(tree._type) { }

  ParamMap&		_map;
  ParamMapIter		_itr;
  AdaptationType	_type;
};

APTParamBasePtr& ParamTree::Iterator::apt()
{
  if (_type != RAPT && _type != SLAPT)
    throw jtype_error("Adaptation type is neither RAPT nor SLAPT.");

  return Cast<APTParamBasePtr>((*_itr).second);
}

MLLRParamPtr& ParamTree::Iterator::mllr()
{
  if (_type != MLLR)
    throw jtype_error("Adaptation type is not MLLR.");

  return Cast<MLLRParamPtr>((*_itr).second);
}


// ----- definition for class `ParamTree::ConstIterator' -----
//
class ParamTree::ConstIterator {
  friend class ParamTree;
 public:
  ConstIterator(const ParamTreePtr tree) :
    _map(tree->_map), _itr(_map.begin()), _type(tree->_type) { }

  const ParamBasePtr& par() const { return (*_itr).second; }
  inline const APTParamBasePtr& apt() const;
  inline const MLLRParamPtr& mllr() const;

  void operator++(int) { _itr++; }
  bool more() { return _itr != _map.end(); }
  unsigned regClass() const  { return (*_itr).first; }

 private:
  ConstIterator(const ParamTree& tree) :
    _map(tree._map), _itr(_map.begin()), _type(tree._type) { }

  const ParamMap&  _map;
  ParamMapConstIter  _itr;
  AdaptationType  _type;
};

const APTParamBasePtr& ParamTree::ConstIterator::apt() const
{
  if (_type != RAPT && _type != SLAPT)
    throw jtype_error("Adaptation type is neither RAPT nor SLAPT.");

  return Cast<APTParamBasePtr>((*_itr).second);
}

const MLLRParamPtr& ParamTree::ConstIterator::mllr() const
{
  if (_type != MLLR)
    throw jtype_error("Adaptation type is not MLLR.");

  return Cast<MLLRParamPtr>((*_itr).second);
}


// ----- definition of class `BLTTransformer' -----
//
class BLTTransformer : virtual public APTTransformerBase {
  friend void RAPTParam::unitCirclePlot(const String& fileName) const;
 public:
  BLTTransformer(UnShrt sbFtSz, UnShrt nSubFt, UnShrt orgSbFtSz = 0);
  BLTTransformer(double al, UnShrt sbFtSz, UnShrt nSubFt, UnShrt orgSbFtSz);
  BLTTransformer(const RAPTParam& par, UnShrt sbFtSz, UnShrt nSubFt,
		 UnShrt orgSbFtSz = 0, int trace = 0, const String& ldaFile = "",
		 UnShrt ldaFeatLen = 0);

  virtual void unitCirclePlot(const String& fileName);

 protected:
  BLTTransformer() : _alpha(0.0) { }

  void calcTransMatrix(double mu);
  void bilinear(double mu, LaurentSeries& sequence);
  virtual void _calcSequence(const CoeffSequence& _params, LaurentSeries& _sequence);

  double _alpha;

 private:
  BLTTransformer(const RAPTParam& par);
};


static const double DefaultSmallTheta     = 1.0e-03;


// ----- definition of class `RAPTTransformer' -----
//
class RAPTTransformer
: virtual public APTTransformerBase, protected BLTTransformer {
  friend class RAPTEstimatorBase;
  friend void RAPTParam::unitCirclePlot(const String& fileName) const;
 public:
  RAPTTransformer();
  RAPTTransformer(UnShrt sbFtSz, UnShrt nSubFt, UnShrt orgSbFtSz = 0,
		  UnShrt noParams = 1);
  RAPTTransformer(const RAPTParam& par, UnShrt sbFtSz, UnShrt nSubFt,
		  UnShrt orgSbFtSz = 0, int trace = 0, const String& ldaFile = "",
		  UnShrt ldaFeatLen = 0);

  virtual void unitCirclePlot(const String& fileName);

 protected:
  typedef enum _SmallTheta { NotSmall, ThetaZero, ThetaPi } SmallTheta;

  virtual void _calcSequence(const CoeffSequence& _params, LaurentSeries& _sequence);
  void allPass(double _rho, double _theta, LaurentSeries& _sequence,
	       bool _invFlag = false);
  bool allPassSmallTheta(double _rho, double _theta, LaurentSeries& _sequence,
			 bool _invFlag = false);
  SmallTheta isThetaSmall(double& _theta);
  void rect2polar(double _real, double _imag, double& _rho, double& _theta) {
    Complex a(_real, _imag); _rho = sqrt(norm(a)); _theta = arg(a);
  }

  const UnShrt NoComplexPairs;

 private:
  RAPTTransformer(const RAPTParam& par);
  UnShrt _setNoComplexPairs(UnShrt nParams = 0) const;

public:  // !!! HACK; should not be necessary !!!
  // scratch space for series calculation
  LaurentSeries					_temp1;
  LaurentSeries					_temp2;
  LaurentSeries					_temp3;
};


// ----- definition of class `SLAPTTransformer' -----
//
class SLAPTTransformer : virtual public APTTransformerBase {
  friend void SLAPTParam::unitCirclePlot(const String& fileName) const;
 public:
  SLAPTTransformer();
  SLAPTTransformer(UnShrt sbFtSz, UnShrt nSubFt, UnShrt noParams,
		   UnShrt orgSbFtSz = 0);
  SLAPTTransformer(const SLAPTParam& par, UnShrt sbFtSz, UnShrt nSubFt,
		   UnShrt orgSbFtSz = 0, int trace = 0, const String& ldaFile = "",
		   UnShrt ldaFeatLen = 0);
  ~SLAPTTransformer();

  virtual void unitCirclePlot(const String& fileName);

 protected:
  virtual void _calcSequence(const CoeffSequence& _params, LaurentSeries& _sequence);

 private:
  static const UnShrt	NoColumns;
  static CoeffSequence& eCoefficients() {
    static CoeffSequence _eCoefficients;
    return _eCoefficients;
  }

  SLAPTTransformer(const SLAPTParam& par);

  void _initialize() const;
  double _eCoeff(UnShrt i) const { return eCoefficients()[i]; }

  mutable LaurentSeries*			_fmn;
};


// ----- definition of class `MLLRTransformer' -----
//
class MLLRTransformer : virtual public TransformBase {
 public:
  MLLRTransformer(UnShrt sbFtSz = 0, UnShrt nSubFt = 0);
  MLLRTransformer(const MLLRParam& par, UnShrt sbFtSz, UnShrt nSubFt);

  virtual void transform(const NaturalVector& from, NaturalVector& to,
			 bool useBias = true) const;
};


// ----- definition of class `STCTransformer' -----
//
class STCTransformer : public VectorFloatFeatureStream, virtual public TransformBase {
 public:
  STCTransformer(VectorFloatFeatureStreamPtr& src,
		 UnShrt sbFtSz = 0, UnShrt nSubFt = 0, const String& nm = "STC Transformer");

  STCTransformer(VectorFloatFeatureStreamPtr& src, const STCParam& par,
		 UnShrt sbFtSz, UnShrt nSubFt, const String& nm = "STC Transformer");

  virtual void transform(const NaturalVector& from, NaturalVector& to,
			 bool useBias = true) const;

  virtual const gsl_vector_float* next(int frameX = -1);

  void save(const String& fileName);

  void load(const String& fileName);

  virtual void reset() { _src->reset(); VectorFloatFeatureStream::reset(); }

 protected:
  void _transform(const gsl_vector_float* srcVec, gsl_vector_float* transVec);

  VectorFloatFeatureStreamPtr			_src;
};

typedef Inherit<STCTransformer, VectorFloatFeatureStreamPtr> STCTransformerPtr;


// ----- definition of class `LDATransformer' -----
//
class LDATransformer : public VectorFloatFeatureStream, virtual public TransformBase {
 public:
  LDATransformer(VectorFloatFeatureStreamPtr& src,
		 UnShrt sbFtSz = 0, UnShrt nSubFt = 0, const String& nm = "LDA Transformer");
  LDATransformer(VectorFloatFeatureStreamPtr& src, const LDAParam& par,
		 UnShrt sbFtSz, UnShrt nSubFt, const String& nm = "LDA Transformer");

  virtual void transform(const NaturalVector& from, NaturalVector& to, bool useBias = true) const;

  virtual const gsl_vector_float* next(int frameX = -1);

  virtual void reset() { _src->reset(); VectorFloatFeatureStream::reset(); }

  void save(const String& fileName) const;

  void load(const String& fileName);

 protected:
  VectorFloatFeatureStreamPtr				_src;
};

typedef Inherit<LDATransformer, VectorFloatFeatureStreamPtr> LDATransformerPtr;


// ----- definition of class `SpeakerList'  -----
//
class SpeakerList {
  typedef list<String>                 _SpeakerList;
  typedef _SpeakerList::const_iterator _SpeakerListIter;
 public:
  SpeakerList(const String& spkrList);

  class Iterator;  friend class Iterator;

 private:
  _SpeakerList					_slist;
};

typedef refcount_ptr<SpeakerList> SpeakerListPtr;


// ----- definition of class `SpeakerList::Iterator'  -----
//
class SpeakerList::Iterator {
 public:
  Iterator(const SpeakerListPtr& sl)
    : _slist(sl->_slist), _itr(_slist.begin()) { }
  Iterator(const SpeakerList& sl)
    : _slist(sl._slist), _itr(_slist.begin()) { }

  void operator++(int) { _itr++; }
  bool more() { return _itr != _slist.end(); }
  const String& spkr()     { return (*_itr); }
  operator const String&() { return (*_itr); }

 private:
  const _SpeakerList&				_slist;
  _SpeakerListIter				_itr;
};


// ----- definition for class `BaseTree' -----
//
typedef enum { Internal, Leaf } NodeType;

class BaseTree {
 public:
  virtual ~BaseTree();

  UnShrt featLen()          const { return cbs()->featLen(); }
  UnShrt subFeatLen()       const { return cbs()->subFeatLen(); }
  UnShrt nSubFeat()         const { return cbs()->nSubFeat(); }
  UnShrt orgSubFeatLen()    const { return cbs()->orgSubFeatLen(); }

  UnShrt cepSubFeatLen()    const { return cbs()->cepSubFeatLen(); }
  UnShrt cepNSubFeat()      const { return cbs()->cepNSubFeat(); }
  UnShrt cepOrgSubFeatLen() const { return cbs()->cepOrgSubFeatLen(); }

  const char* ldaFile()  const { return cbs()->ldaFile(); }

        CodebookSetAdaptPtr& cbs()       { return _cbs(); }
  const CodebookSetAdaptPtr& cbs() const { return _cbs(); }

  class Node;
  class Iterator;       friend class Iterator;
  class ConstIterator;  friend class ConstIterator;

  typedef refcount_ptr<Node> NodePtr;

 	NodePtr& node(unsigned idx, bool useAncestor = false);
  const NodePtr& node(unsigned idx, bool useAncestor = false) const;
 
 protected:
  BaseTree(CodebookSetAdaptPtr& cbs);

  NodePtr& _leftChild(const NodePtr& p);
  NodePtr& _rightChild(const NodePtr& p);

  void  _setNode(UnShrt idx, NodePtr& n);
  void  _setNodeTypes();
  void  _validateNodeTypes() const;
  bool _nodePresent(UnShrt idx) const;

        CodebookSetAdaptPtr& _cbs()       { return _codebookSet; }
  const CodebookSetAdaptPtr& _cbs() const { return _codebookSet; }

  typedef map<int, NodePtr>			_NodeList;
  typedef _NodeList::iterator			_NodeListIter;
  typedef _NodeList::const_iterator		_NodeListConstIter;
  typedef _NodeList::value_type			_ValueType;

  _NodeList					_nodeList;
  CodebookSetAdaptPtr				_codebookSet;
};

typedef refcount_ptr<BaseTree> BaseTreePtr;


// ----- definition for class `BaseTree::Node' -----
//
class BaseTree::Node {
  friend void BaseTree::_setNodeTypes();
 public:
  virtual ~Node() { }

  NodeType type()  const { return _type;  }
  UnShrt   index() const { return _index; }

 protected:
  Node(BaseTree& tr, UnShrt idx, NodeType typ = Leaf)
    : _tree(tr), _index(idx), _type(typ) { }

  void _setType(NodeType t) { _type = t; }

  BaseTree&					_tree;
  const UnShrt					_index;
  NodeType					_type;
};


// ----- definition for class `BaseTree::ConstIterator' -----
//
class BaseTree::ConstIterator {
  friend class BaseTree;
 public:
  ConstIterator(const BaseTreePtr& tr, bool ol)
    : _tree(tr), _nodeList(tr->_nodeList), _itr(_nodeList.begin()), _onlyLeaves(ol)
  {
    if (_onlyLeaves && more() && (*_itr).second->type() != Leaf)
      (*this)++;
  }

  inline void operator++(int);
  bool more() { return _itr != _nodeList.end(); }

  unsigned regClass() { return (*_itr).first; }
  const NodePtr& node() { return (*_itr).second; }

  const BaseTreePtr& tree() const { return _tree; }

 protected:
  const BaseTreePtr				_tree;

 private:
  const _NodeList&				_nodeList;
  _NodeListConstIter				_itr;
  const bool					_onlyLeaves;
};

void BaseTree::ConstIterator::operator++(int)
{
  do { _itr++; }
  while (_onlyLeaves && more() && (*_itr).second->type() != Leaf);
}


// ----- definition for class `BaseTree::Iterator' -----
//
class BaseTree::Iterator {
  friend class BaseTree;
 public:
  Iterator(BaseTreePtr& tr, bool ol)
    : _tree(tr), _nodeList(tr->_nodeList), _itr(_nodeList.begin()), _onlyLeaves(ol)
  {
    if (_onlyLeaves && more() && (*_itr).second->type() != Leaf)
      (*this)++;
  }

  inline void operator++(int);
  bool more() { return _itr != _nodeList.end(); }

  unsigned regClass() { return (*_itr).first; }
  NodePtr& node() { return (*_itr).second; }

  BaseTreePtr& tree() { return _tree; }

 protected:
  BaseTreePtr					_tree;

 private:
  _NodeList&					_nodeList;
  _NodeListIter					_itr;
  const bool					_onlyLeaves;
};

void BaseTree::Iterator::operator++(int)
{
  do { _itr++; }
  while (_onlyLeaves && more() && (*_itr).second->type() != Leaf);
}


// ----- definition for class `TransformerTree' -----
//
class TransformerTree : public BaseTree {
 public:
  TransformerTree(CodebookSetAdaptPtr& cb, const String& parmFile, UnShrt orgSubFeatLen = 0);
  TransformerTree(CodebookSetAdaptPtr& cb, const ParamTreePtr& paramTree, UnShrt orgSubFeatLen = 0);

  const ParamTreePtr& paramTree() const { return _paramTree; }

  TransformerTree& transform();
  const TransformBasePtr& transformer(UnShrt regClass) const;

  class      Iterator;  friend class      Iterator;
  class ConstIterator;  friend class ConstIterator;

 protected:
  TransformerTree(CodebookSetAdaptPtr& cb, ParamTreePtr& paramTree);

  class Node;  friend class Node;
  typedef Inherit<Node, BaseTree::NodePtr> NodePtr;

  NodePtr& node(UnShrt idx, bool useAncestor) {
    return Cast<NodePtr>(BaseTree::node(idx, useAncestor));
  }
  const NodePtr& node(UnShrt idx, bool useAncestor) const {
    return Cast<const NodePtr>(BaseTree::node(idx, useAncestor));
  }

  ParamTreePtr					_paramTree;
};

typedef Inherit<TransformerTree, BaseTreePtr> TransformerTreePtr;


// ----- definition for class `TransformerTree::Node' -----
//
class TransformerTree::Node : public BaseTree::Node {
  friend class TransformerTree;
 public:
  Node(TransformerTree& tr, unsigned idx, const ParamBasePtr& par,
       UnShrt orgSubFeatLen = 0);

  Node(TransformerTree& tr, unsigned idx, NodeType type, TransformBase* transformer);

  virtual ~Node();

  void transform(CodebookAdapt::GaussDensity& mix);
  const TransformBasePtr& transformer() const { return _trans; }
  const TransformMatrix& transformMatrix() const {
    return _trans->transformMatrix();
  }

 private:
  const TransformBasePtr			_trans;
};


// ----- definition for class `TransformerTree::Iterator' -----
//
class TransformerTree::Iterator : public BaseTree::Iterator {
 public:
  Iterator(TransformerTreePtr& tr, bool ol = true)
    : BaseTree::Iterator(tr, ol) { }

    /*
  Iterator(TransformerTree& tr, bool ol = true)
    : BaseTree::Iterator(tr, ol) { }
    */

        NodePtr& node() { return Cast<NodePtr>(BaseTree::Iterator::node()); }
  const TransformBasePtr& transformer() { return node()->transformer(); }

  TransformerTreePtr& tree() { return Cast<TransformerTreePtr>(_tree); }
};


// ----- definition for class `TransformerTree::ConstIterator' -----
//
class TransformerTree::ConstIterator : public BaseTree::ConstIterator {
 public:
  ConstIterator(const TransformerTreePtr& tr, bool ol = true)
    : BaseTree::ConstIterator(tr, ol) { }

  const NodePtr& node() { return Cast<NodePtr>(BaseTree::ConstIterator::node()); }
  const TransformBasePtr& transformer() { return node()->transformer(); }

  const TransformerTreePtr& tree() const { return Cast<const TransformerTreePtr>(_tree); }
};

void adaptCbk(CodebookSetAdaptPtr& cbs, const ParamTreePtr& pt, UnShrt orgSubFeatLen = 0);

#endif
