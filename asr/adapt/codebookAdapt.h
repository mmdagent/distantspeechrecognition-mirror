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

#ifndef _codebookAdapt_h_
#define _codebookAdapt_h_

#include "gaussian/codebookBasic.h"

/**
* \addtogroup Codebook
*/
/*@{*/

/**
* \defgroup CodebookAdaptGroup Classes for calculating acoustic likelihood with adapted Gaussian codebooks.
*/
/*@{*/

/**
* \defgroup CodebookAdapt Basic classes for calculating likelihoods with adapted Gaussian components.
*/
/*@{*/

// ----- definition for class `CodebookAdapt' -----
// 
class CodebookAdapt : public CodebookBasic {
 public:
  CodebookAdapt(const String& nm, UnShrt rfN, UnShrt dmN, CovType cvTp = COV_DIAGONAL,
		VectorFloatFeatureStreamPtr feat = NULL);

  CodebookAdapt(const String& nm, UnShrt rfN, UnShrt dmN, CovType cvTp = COV_DIAGONAL,
		const String& featureName = "");

  virtual ~CodebookAdapt();

  class GaussDensity;  friend class GaussDensity;
  class Iterator;      friend class Iterator;

  void resetMean();

 protected:
  virtual void _allocRV();

  gsl_matrix_float*	_origRV;   // original (i.e., untransformed) mean vectors
};

typedef Inherit<CodebookAdapt, CodebookBasicPtr> CodebookAdaptPtr;


// ----- definition for class `CodebookAdapt::GaussDensity' -----
//
class CodebookAdapt::GaussDensity : public CodebookBasic::GaussDensity {
 public:
  GaussDensity(CodebookAdaptPtr& cb, int refX)
    : CodebookBasic::GaussDensity(cb, refX) { }

  void allocMean();

  inline UnShrt orgFeatLen() const;
  inline const NaturalVector origMean() const;

 protected:
        CodebookAdaptPtr&       cbk()       { return Cast<CodebookAdaptPtr>(_cbk()); }
  const CodebookAdaptPtr&	cbk() const { return Cast<CodebookAdaptPtr>(_cbk()); }

  CodebookBasic::GaussDensity::refX;
};

typedef refcount_ptr<CodebookAdapt::GaussDensity> CodebookAdapt_GaussDensityPtr;

UnShrt CodebookAdapt::GaussDensity::orgFeatLen() const {
  if (cbk()->_origRV)
    return cbk()->_origRV->size2;
  else
    return cbk()->_rv->size2;
}

const NaturalVector CodebookAdapt::GaussDensity::origMean() const {
  if (cbk()->_origRV != NULL)
    return NaturalVector(cbk()->_origRV->data + refX() * cbk()->_origRV->size2, cbk()->_origRV->size2, 0);
  else
    return NaturalVector(cbk()->_rv->data + refX() * cbk()->_rv->size2, cbk()->_rv->size2, 0);
}


// ----- definition for class `CodebookAdapt::Iterator' -----
//
class CodebookAdapt::Iterator : public CodebookBasic::Iterator {
 public:
  Iterator(CodebookAdaptPtr& cb) :
    CodebookBasic::Iterator(cb) { }

  GaussDensity mix() {
    return GaussDensity(cbk(), _refX);
  }

 protected:
  	CodebookAdaptPtr& cbk()       { return Cast<CodebookAdaptPtr>(_cbk()); }
  const CodebookAdaptPtr& cbk() const { return Cast<CodebookAdaptPtr>(_cbk()); }
};

/*@}*/

/**
* \defgroup CodebookSetBasic Classes for storing all Gaussian components required for a speech recognizer.
*/
/*@{*/

// ----- definition for class `CodebookSetAdapt' -----
//
class CodebookSetAdapt : public CodebookSetBasic {
 public:
  CodebookSetAdapt(const String& descFile = "", FeatureSetPtr& fs = NullFeatureSetPtr, const String& cbkFile = "");

  virtual ~CodebookSetAdapt() { }

  class CodebookIterator;  friend class CodebookIterator;
  class GaussianIterator;  friend class GaussianIterator;

        CodebookAdaptPtr& find(const String& key)       { return Cast<CodebookAdaptPtr>(_find(key)); }
  const CodebookAdaptPtr& find(const String& key) const { return Cast<CodebookAdaptPtr>(_find(key)); }

        CodebookAdaptPtr& find(unsigned cbX)            { return Cast<CodebookAdaptPtr>(_find(cbX)); }
  const CodebookAdaptPtr& find(unsigned cbX)      const { return Cast<CodebookAdaptPtr>(_find(cbX)); }

  void resetMean();

 protected:
  virtual void _addCbk(const String& name, UnShrt rfN, UnShrt dmN, CovType cvTp,
		       const String& featureName);

  virtual void _addCbk(const String& name, UnShrt rfN, UnShrt dmN, CovType cvTp,
		       VectorFloatFeatureStreamPtr feat);
};

typedef Inherit<CodebookSetAdapt, CodebookSetBasicPtr> CodebookSetAdaptPtr;


// ----- definition for container class `CodebookSetAdapt::CodebookIterator' -----
//
class CodebookSetAdapt::CodebookIterator : private CodebookSetBasic::CodebookIterator {
  friend class CodebookSetAdapt;
 public:
  CodebookIterator(CodebookSetAdaptPtr& cbs, int nParts = 1, int part = 1)
    : CodebookSetBasic::CodebookIterator(cbs, nParts, part) { }

  CodebookSetBasic::CodebookIterator::operator++;
  CodebookSetBasic::CodebookIterator::more;

  CodebookAdaptPtr next() {
    if (more()) {
      CodebookAdaptPtr ret = cbk();
      operator++(1);
      return ret;
    } else {
      throw jiterator_error("end of codebook!");
    }
  }

        CodebookAdaptPtr& cbk()       { return Cast<CodebookAdaptPtr>(_cbk()); }
  const CodebookAdaptPtr& cbk() const { return Cast<CodebookAdaptPtr>(_cbk()); }

 protected:
  CodebookSetBasic::CodebookIterator::_cbk;
};


// ----- definition for container class `CodebookSetAdapt::GaussianIterator' -----
//
class CodebookSetAdapt::GaussianIterator : protected CodebookSetBasic::GaussianIterator {
  typedef CodebookAdapt::GaussDensity GaussDensity;
 public:
  GaussianIterator(CodebookSetAdaptPtr& cbs, int nParts = 1, int part = 1)
    : CodebookSetBasic::GaussianIterator(cbs, nParts, part) { }

  CodebookSetBasic::GaussianIterator::operator++;
  CodebookSetBasic::GaussianIterator::more;

  GaussDensity mix() { return GaussDensity(cbk(), refX()); }

 protected:
        CodebookAdaptPtr& cbk()       { return Cast<CodebookAdaptPtr>(_cbk()); }
  const CodebookAdaptPtr& cbk() const { return Cast<CodebookAdaptPtr>(_cbk()); }
};

/*@}*/

/*@}*/

/*@}*/

#endif
