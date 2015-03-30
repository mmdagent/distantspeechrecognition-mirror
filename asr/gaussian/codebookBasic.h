//                              -*- C++ -*-
//
//                               Millennium
//                  Distant Speech Recognition System
//                                 (dsr)
//
//  Module:  asr.gaussian
//  Purpose: Basic acoustic likelihood computation.
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


#ifndef _codebookBasic_h_
#define _codebookBasic_h_

#include <vector>

#include "feature/feature.h"
#include <common/refcount.h>
#include "common/mlist.h"
#include "natural/natural.h"
#include <common/jexception.h>

#include <gsl/gsl_matrix_float.h>
#include <gsl/gsl_block.h>
#include <gsl/gsl_vector.h>

typedef unsigned char CBX; // from bbi.h

typedef enum { COV_NO, COV_RADIAL, COV_DIAGONAL, COV_FULL, COV_UNKNOWN } CovType;


// ----- definition for class `LogLhoodIndex' -----
// 
class LogLhoodIndex {
 public:
  LogLhoodIndex(float logLhood = 0.0, int idx = 0)
    : _logLhood(logLhood), _gaussIndex(idx) { }

  float lhood() const { return _logLhood; }
  int   index() const { return _gaussIndex; }

 private:
  float		_logLhood;
  int		_gaussIndex;
};

typedef refcount_ptr<LogLhoodIndex> LogLhoodIndexPtr;

/**
* \defgroup Codebook Manipulation of Gaussian mixture components
* This group of classes provides the capability to manipulate Gaussian mixture components.
*/
/*@{*/

/**
* \defgroup CodebookBasicGroup Base classes for calculating likelihoods of Gaussian components.
*/
/*@{*/

/**
* \defgroup CodebookBaic Basic classes for calculating likelihoods of Gaussian components.
*/
/*@{*/

// ----- definition for class `CodebookBasic' -----
// 
class CodebookBasic;
typedef refcount_ptr<CodebookBasic> CodebookBasicPtr;

class CodebookBasic {
  friend class FeatureSpaceAdaptationFeature;
  friend class STCEstimator;
  friend class LDAEstimator;
  friend class SATBase;
 public:
  CodebookBasic(const String& nm, UnShrt rfN, UnShrt dmN, CovType cvTp = COV_DIAGONAL,
		VectorFloatFeatureStreamPtr feat = NULL);

  CodebookBasic(const String& nm, UnShrt rfN, UnShrt dmN, CovType cvTp = COV_DIAGONAL,
		const String& featureName = "");

  virtual ~CodebookBasic();

  void setFeature(FeatureSetPtr& featureSet);

  class GaussDensity;  friend class GaussDensity;
  class Iterator;      friend class Iterator;

  class Cache;	       friend class Cache;
  typedef refcount_ptr<Cache>	CachePtr;
  const CachePtr& cache() const { return _cache; }

  String name() const { return _name; }
  String puts() const;

  UnShrt refN()       const { return _refN; }
  UnShrt nSubFeat()   const { assert(_nSubFeat != 0);  return _nSubFeat; }
  void setSubFeatN(UnShrt nsub) { _nSubFeat = nsub; }
  UnShrt subFeatLen() const {
    assert( featLen() % nSubFeat() == 0 );  return featLen() / nSubFeat(); }
  UnShrt featLen()    const { return _dimN; }
  UnShrt orgFeatLen() const { return _orgDimN; }

  const float* count() const { return _count; }

  float mean(UnShrt refX, UnShrt compX) const {
    assert(refX < _refN);  assert(compX < _dimN);
    return gsl_matrix_float_get(_rv, refX, compX);
  }
  float invCov(UnShrt refX, UnShrt compX) const {
    assert(refX < _refN);  assert(compX < _dimN);
    return _cv[refX]->data[compX];
  }

  void save(FILE* fp, bool janusFormat = false) const;
  void load(FILE* fp, bool readName = false);
  void loadOld(FILE* fp);

  void write(FILE* fp) const;

  virtual void resetCache();
  void resetFeature();

  inline float score(int frameX, const float* val, float* addCount = NULL);

  void setScoreAll(unsigned cacheN = 100);

  void setRegClasses(UnShrt c = 1);

  void setScale(float scale = 1.0) { _scale = scale; }

  LogLhoodIndexPtr logLhood(const gsl_vector* frame, float* val = NULL) const;

  void copyMeanVariance(const CodebookBasicPtr& cb);

  float ttlCounts() const;

  void applySTC(const gsl_matrix_float* trans);

  void setMean(unsigned refX, unsigned compX, float mean) { gsl_matrix_float_set(_rv, refX, compX, mean); }

  void setVariance(unsigned refX, unsigned compX, float variance) {
    _cv[refX]->data[compX] = variance;
    // throw j_error("Must adjust Gaussian normalization constant.");
  }

 protected:
  void _loadCovariance(unsigned refX, FILE* fp);
  void _saveCovariance(unsigned refX, FILE* fp) const;

  typedef float (CodebookBasic::*_ScoreFunction)(int frameX, const float* val, float* addCount);

  // score all Gaussians
  float _scoreAll(int frameX, const float* val, float* addCount = NULL);

  // score only most likely Gaussian (default)
  float _scoreOpt(int frameX, const float* val, float* addCount = NULL);

  void				_allocCount();
  virtual void			_allocRV();
  void				_allocCovar();
  void				_allocRegClass();
  void				_allocDescLength();
  void				_allocCb();
  void 				_dealloc();

  const String			_name;		// codebook name
  const String			_featureName;	// feature name
  UnShrt	       		_refN;		// number of vectors in the codebook
  UnShrt			_dimN;		// dimensionality of the underlying feature space
  UnShrt			_orgDimN;	// dimensionality of the original feature space
  CovType			_covType;	// the default type of the covariance matrices
  float				_pi;		// the precalculated value of log((2 PI)^dimN)

  gsl_matrix_float*		_rv;		// rv->matPA[j][k] = k-th coeff of j-th vector
  gsl_vector_float**		_cv;		// cv[j] = covariance matrix of j-th vector
  float*			_determinant;	// determinant of Gaussian pdf
  float*			_count;		// the training counts for each reference vector

  UnShrt**			_regClass;	// indices of transformation matrices
  UnShrt**			_descLen;	// description length parameters
  UnShrt			_nSubFeat;	// number of sub-features

  VectorFloatFeatureStreamPtr	_feature;	// feature set for this codebook

 private:
  static const int CodebookMagic;
  static const int MarkerMagic;
  static const double DefaultExptValue;

  static void _dumpMarker(FILE* fp);
  static void _checkMarker(FILE* fp);

  static UnShrt 		RefMax;
  static CBX*   		tmpI;
  static int    		tmpN;

  int				_frameX;
  int				_minDistIdx;
  float				_score;

  float				_scale;
  CachePtr			_cache;		// cache for Gaussian likelihoods
  _ScoreFunction		_scoreFunc;
};

// compute acoustic likelihood score
float CodebookBasic::score(int frameX, const float* val, float* addCount)
{
  return (this->*_scoreFunc)(frameX, val, addCount);
}


// ----- definition for class `CodebookBasic::GaussDensity' -----
//
class CodebookBasic::GaussDensity : protected NaturalIndex {
    friend class Iterator;
 public:
  GaussDensity()
    : _codebook(NULL), _refX(0) { }
  GaussDensity(CodebookBasicPtr& cb, int refX)
    : _codebook(cb), _refX(refX) { }
  ~GaussDensity() { }

  inline const NaturalVector mean() const;
  inline NaturalVector mean();
  inline NaturalVector var();

  inline UnShrt featLen() const;
  inline UnShrt orgFeatLen() const;
  UnShrt nSubFeat() const { return cbk()->_nSubFeat; }

  float var(int i) const { return gsl_vector_float_get(cbk()->_cv[_refX], i); }

  inline UnShrt regClass() const;

  UnShrt* descLen() const;

  float& var(int i) { return cbk()->_cv[_refX]->data[i]; }

 protected:
        CodebookBasicPtr&	cbk()        { return _cbk(); }
  const CodebookBasicPtr&	cbk()  const { return _cbk(); }

        CodebookBasicPtr&	_cbk()       { return _codebook; }
  const CodebookBasicPtr&	_cbk() const { return _codebook; }

  UnShrt    refX() const { return _refX; }

 private:
  CodebookBasicPtr		_codebook;	// pointer to codebook
  UnShrt			_refX;		// index of gaussian
};

typedef refcount_ptr<CodebookBasic::GaussDensity> CodebookBasic_GaussDensityPtr;

UnShrt CodebookBasic::GaussDensity::featLen() const {
  return cbk()->_cv[_refX]->size;
}

UnShrt CodebookBasic::GaussDensity::regClass() const {
  if ( cbk()->_regClass && cbk()->_regClass[_refX] && cbk()->_regClass[_refX][0] > 0 )
    return cbk()->_regClass[_refX][1];
  else
    return 0;
}

const NaturalVector CodebookBasic::GaussDensity::mean() const {
  return NaturalVector(cbk()->_rv->data + _refX * cbk()->featLen(), cbk()->_rv->size2, 0);
}

NaturalVector CodebookBasic::GaussDensity::mean() {
  return NaturalVector(cbk()->_rv->data + _refX * cbk()->featLen(), cbk()->_rv->size2, 0);
}

NaturalVector CodebookBasic::GaussDensity::var() {
  return NaturalVector(cbk()->_cv[_refX]->data, featLen(), 0);
}

typedef CodebookBasic::GaussDensity CodebookBasic_GaussDensity;


// ----- definition for class `CodebookBasic::Iterator' -----
//
class CodebookBasic::Iterator {
 public:
  Iterator(CodebookBasicPtr& cb)
    : _refX(0), _codebook(cb) { }
  ~Iterator() { }

  void operator++(int) {
    if (more()) _refX++;
  }
  bool more() const  { return _refX < cbk()->refN(); }
  GaussDensity mix() {
    return GaussDensity(cbk(), _refX);
  }

 protected:
  	CodebookBasicPtr& cbk()       { return _cbk(); }
  const CodebookBasicPtr& cbk() const { return _cbk(); }

  	CodebookBasicPtr& _cbk()       { return _codebook; }
  const CodebookBasicPtr& _cbk() const { return _codebook; }

  UnShrt		_refX;

 private:
  CodebookBasicPtr	_codebook;
};

/*@}*/

// ----- definition for container class `CodebookSet' -----
//
class CodebookSet {
 public:
  ~CodebookSet() {};

  UnShrt nSubFeat()	  const { return _nSubFeat; }
  UnShrt subFeatLen()     const { return _subFeatLen; }
  UnShrt featLen()        const { return _featLen; }
  UnShrt orgFeatLen()     const { return _orgFeatLen; }
  UnShrt orgSubFeatLen()  const { return _orgSubFeatLen; }

  const String& ldaFile() const { return _ldaFile; }

  UnShrt cepSubFeatLen()  const {
    return (_cepSubFeatLen    == 0) ? _subFeatLen    : _cepSubFeatLen;
  }
  UnShrt cepNSubFeat() const {
    return (_cepNSubFeat      == 0) ? _nSubFeat      : _cepNSubFeat;
  }
  UnShrt cepOrgSubFeatLen() const {
    return (_cepOrgSubFeatLen == 0) ? _orgSubFeatLen : _cepOrgSubFeatLen;
  }

 protected:
  CodebookSet(UnShrt nsub = 0, UnShrt flen = 0, UnShrt olen = 0, UnShrt slen = 0,
	      UnShrt oslen = 0) :
    _nSubFeat(nsub), _featLen(flen), _orgFeatLen(olen), _subFeatLen(slen),
    _orgSubFeatLen(oslen), _cepSubFeatLen(0), _cepNSubFeat(0), _cepOrgSubFeatLen(0) {}

 protected:
  UnShrt					_nSubFeat;
  UnShrt				        _featLen;
  UnShrt				        _orgFeatLen;
  UnShrt				        _subFeatLen;
  UnShrt				        _orgSubFeatLen;

  UnShrt				        _cepSubFeatLen;
  UnShrt					_cepNSubFeat;
  UnShrt        				_cepOrgSubFeatLen;

  String					_ldaFile;
};

/**
* \defgroup CodebookSetBasic Classes for storing all Gaussian components required for a speech recognizer.
*/
/*@{*/

// ----- definition for class `CodebookSetBasic' -----
//
static FeatureSetPtr NullFeatureSetPtr(NULL);

class CodebookSetBasic : public CodebookSet {
 protected:
  typedef List<CodebookBasicPtr>		_CodebookList;
  typedef _CodebookList::Iterator		_Iterator;
  typedef _CodebookList::ConstIterator		_ConstIterator;

 public:
  CodebookSetBasic(const String& descFile = "", FeatureSetPtr& fs = NullFeatureSetPtr, const String& cbkFile = "");
  virtual ~CodebookSetBasic();

  void setFeatures(FeatureSetPtr& featureSet);

  class CodebookIterator;  friend class CodebookIterator;
  class GaussianIterator;  friend class GaussianIterator;

  void setNSubFeat(UnShrt nsub);
  void setSubFeatLen(UnShrt len);
  void setFeatLen(UnShrt len);
  void setOrgFeatLen(UnShrt len);
  void setOrgSubFeatLen(UnShrt len);

  void load(const String& filename);
  void save(const String& filename, bool janusFormat = false, int nParts = 1, int part = 1);
  UnShrt ncbks() const { return _cblist.size(); }

  void write(const String& fileName, const String& time = "") const;

  void resetCache();
  void resetFeature();

  void setScoreAll(unsigned cacheN = 1000);
  void setScale(float scale = 1.0);

  unsigned index(const String& key) const { return _cblist.index(key); }

  void setRegClasses(UnShrt c = 1);

  void applySTC(const gsl_matrix_float* trans);

        CodebookBasicPtr& find(const String& key)        { return _find(key); }
  const CodebookBasicPtr& find(const String& key)  const { return _find(key); }

        CodebookBasicPtr& find(unsigned cbX)             { return _find(cbX); }
  const CodebookBasicPtr& find(unsigned cbX)       const { return _find(cbX); }

  void __add(const String& s);

 protected:
  static const unsigned MaxNameLength;
  static const int      CodebookMagic;

        CodebookBasicPtr& _find(const String& key)        { return _cblist[key]; }
  const CodebookBasicPtr& _find(const String& key)  const { return _cblist[key]; }

        CodebookBasicPtr& _find(unsigned cbX)             { return _cblist[cbX]; }
  const CodebookBasicPtr& _find(unsigned cbX)       const { return _cblist[cbX]; }

  virtual void _addCbk(const String& name, UnShrt rfN, UnShrt dmN, CovType cvTp,
		       const String& featurename);

  virtual void _addCbk(const String& name, UnShrt rfN, UnShrt dmN, CovType cvTp,
		       VectorFloatFeatureStreamPtr feat);

  _CodebookList					_cblist;
  FeatureSetPtr					_featureSet;
};

typedef refcount_ptr<CodebookSetBasic> CodebookSetBasicPtr;


// ----- definition for class `CodebookSetBasic::CodebookIterator' -----
//
class CodebookSetBasic::CodebookIterator {
  friend class CodebookSetBasic;
 public:
  CodebookIterator(CodebookSetBasicPtr& cbs, int nParts = 1, int part = 1)
    : _codebookSet(cbs), _iter(cbs->_cblist, nParts, part) { }

  void operator++(int) { _iter++; }
  bool more() const { return _iter.more(); }

  CodebookBasicPtr next() {
    if (more()) {
      CodebookBasicPtr cb(cbk());
      operator++(1);
      return cb;
    } else {
      throw jiterator_error("end of codebooks!");
    }
  }

        CodebookBasicPtr&	cbk()        { return _cbk(); }
  const CodebookBasicPtr&	cbk()  const { return _cbk(); }

 protected:
        CodebookBasicPtr&	_cbk()       { return *_iter; }
  const CodebookBasicPtr&	_cbk() const { return *_iter; }

 private:
  CodebookSetBasicPtr		_codebookSet;
  _Iterator			_iter;
};


// ----- definition for container class `CodebookSetBasic::GaussianIterator' -----
//
class CodebookSetBasic::GaussianIterator {
  typedef CodebookBasic::GaussDensity GaussDensity;
 public:
  GaussianIterator(CodebookSetBasicPtr& cbs, int nParts = 1, int part = 1)
    : _codebookSet(cbs), _iter(cbs->_cblist, nParts, part), _refX(0) { }

  inline void operator++(int);
  inline bool more() const;

  GaussDensity next() {
    if (more()) {
      operator++(1);
      return GaussDensity(cbk(), _refX);
    } else {
      throw jiterator_error("end of gaussians!");
    }
  }
	
  GaussDensity mix() { return GaussDensity(cbk(), _refX); }

 protected:
  UnShrt nCbk() const { return _codebookSet->ncbks(); }

        CodebookBasicPtr&	cbk()        { return _cbk(); }
  const CodebookBasicPtr&	cbk()  const { return _cbk(); }

        CodebookBasicPtr&	_cbk()       { return *_iter; }
  const CodebookBasicPtr&	_cbk() const { return *_iter; }

  UnShrt			refX() const { return _refX; }

 private:
  CodebookSetBasicPtr		_codebookSet;
  _Iterator			_iter;  
  UnShrt			_refX;		// current gaussian
};

void CodebookSetBasic::GaussianIterator::operator++(int) {
  if (_refX < cbk()->refN() - 1) {
    _refX++;
  } else {
    _iter++; _refX = 0;
  }
}

bool CodebookSetBasic::GaussianIterator::more() const {
  if (_iter.more() && _refX < cbk()->refN())
    return true;
  else
    return false;
}


// ----- definition for class `CodebookBasic::Cache' -----
//
class CodebookBasic::Cache {
  friend class CodebookBasic;
 public:
  Cache(UnShrt refN, UnShrt distCacheN = DefaultDistCacheN);
  ~Cache();

  void reset();

  static const short  EmptyIndicator;
  static const UnShrt DefaultDistCacheN;

 private:
  int*		_distFrameX;	// physical frame indices of the cache entries for
  				// the cb vector distance cache
  float**	_logdist;	// logdist[cacheX][cbX][refX]

  const UnShrt 	_refN;		// how many ref vectors this codebook has
  const UnShrt 	_distCacheN;	// number of frames that are cached
};

/*@}*/

/*@}*/

/*@}*/

#endif
