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

%module adapt

#ifdef AUTODOC
%section "Adapt"
#endif

/* operator overloading */
%rename(__str__) *::puts();

%include jexception.i
%include typedefs.i

%pythoncode %{
import btk
from btk import feature
from btk import stream
from asr import gaussian
oldimport = """
%}
%import gaussian/gaussian.i
%pythoncode %{
"""
%}

%import gaussian/gaussian.i

%{
#include "feature/feature.h"
#include "feature/lpc.h"
#include "stream/pyStream.h"
#include "adapt/codebookAdapt.h"
#include "adapt/transform.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
%}

#ifdef AUTODOC
%subsection "CodebookAdapt", before
#endif

%include gsl/gsl_types.h
%include complex.i
%include matrix.i
%include vector.i

%rename(CodebookSetAdapt_CodebookIterator) CodebookSetAdapt::CodebookIterator;
%rename(__iter__) CodebookSetAdapt::codebookIter;


// ----- definition of class `SpeakerList'  -----
//
%ignore SpeakerList;
class SpeakerList {
 public:
  SpeakerList(const String& spkrList);
};

class SpeakerListPtr {
 public:
  %extend {
    SpeakerListPtr(const String& spkrList) {
      return new SpeakerListPtr(new SpeakerList(spkrList));
    }
  }
  SpeakerList* operator->();
};


// ----- definition for class `CodebookAdapt' -----
// 
%ignore CodebookAdapt;
class CodebookAdapt : public CodebookBasic {
 public:
  CodebookAdapt(const String nm, UnShrt rfN, UnShrt dmN, CovType cvTp,
		FeatureSet* feat = NULL, UnShrt ftX = 0);
  ~CodebookAdapt();

  void resetMean();
};

class CodebookAdaptPtr : public CodebookBasicPtr {
 public:
  %extend {
    CodebookAdaptPtr(const String nm, UnShrt rfN, UnShrt dmN, CovType cvTp,
		     VectorFloatFeatureStreamPtr feat) {
      return new CodebookAdaptPtr(new CodebookAdapt(nm, rfN, dmN, cvTp, feat));
    }
  }
  CodebookAdapt* operator->();
};


// ----- definition for class `CodebookSetAdapt' -----
//
%ignore CodebookSetAdapt;
class CodebookSetAdapt : public CodebookSetBasic {
public:
  CodebookSetAdapt(const String descFile = "", FeatureSetPtr& fs = NULL, const String cbkFile = "");
  ~CodebookSetAdapt();

  // find a codebook by name
  CodebookAdaptPtr find(const String key);

  void resetMean();
};

class CodebookSetAdaptPtr : public CodebookSetBasicPtr {
 public:
  %extend {
    CodebookSetAdaptPtr(const String descFile = "", FeatureSetPtr fs = NULL, const String cbkFile = "") {
      return new CodebookSetAdaptPtr(new CodebookSetAdapt(descFile, fs, cbkFile));
    }

    // return an iterator
    CodebookSetAdapt::CodebookIterator* __iter__() {
      return new CodebookSetAdapt::CodebookIterator(*self);
    }

    // return a codebook
    CodebookAdaptPtr __getitem__(const String name) {
      return (*self)->find(name);
    }
  }

  CodebookSetAdapt* operator->();
};

class CodebookSetAdapt::CodebookIterator {
  public:
    CodebookSetAdapt::CodebookIterator(CodebookSetAdaptPtr cbs, int nParts = 1,
                                       int part = 1);
    CodebookAdaptPtr next();
};  

%{
#include "transform.h"
%}

#ifdef AUTODOC
%subsection "Transform", before
#endif 

typedef string String;


// ----- definition for class `ParamTree' -----
//
%ignore ParamTree;
class ParamTree {
 public:
  ParamTree(const String fileName = "");
  ~ParamTree();

  // write speaker parameters to a file
  void write(const String fileName);

  // read speaker parameters from a file
  void read(const String fileName);

  // zero out parameters
  void clear();

  // apply STC transformation
  void applySTC(const String& stcFile);
};

class ParamTreePtr {
 public:
  %extend {
    ParamTreePtr(const String fileName = "") {
      return new ParamTreePtr(new ParamTree(fileName));
    }
  }
  ParamTree* operator->();
};


// ----- definition for class `TransformerTree' -----
//
%ignore TransformerTree;
class TransformerTree {
 public:
  TransformerTree(CodebookSetAdaptPtr& cb, const ParamTreePtr& pt,
                  UnShrt orgSubFeatLen = 0);
  ~TransformerTree();

  TransformerTree& transform();
};

class TransformerTreePtr {
 public:
  %extend {
    TransformerTreePtr(CodebookSetAdaptPtr& cb, const ParamTreePtr& pt,
                       UnShrt orgSubFeatLen = 0) {
      return new TransformerTreePtr(new TransformerTree(cb, pt, orgSubFeatLen));
    }
  }
  TransformerTree* operator->();
};

void adaptCbk(CodebookSetAdaptPtr& cbs, const ParamTreePtr& pt, UnShrt orgSubFeatLen = 0);


// ----- definition of class `STCTransformer' -----
//
%ignore STCTransformer;
class STCTransformer : public VectorFloatFeatureStream {
 public:
  STCTransformer(VectorFloatFeatureStreamPtr& src,
		 UnShrt sbFtSz = 0, UnShrt nSubFt = 0, const String& nm = "STC Transformer");

#if 0
  STCTransformer(VectorFloatFeatureStreamPtr& src, const STCParam& par,
		 UnShrt sbFtSz, UnShrt nSubFt, const String& nm = "STC Transformer");
#endif

  virtual void transform(const NaturalVector& from, NaturalVector& to,
			 bool useBias = true) const;

  virtual const gsl_vector_float* next(int frameX = -1);

  void save(const String& fileName);

  virtual void load(const String& fileName);
};

class STCTransformerPtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    STCTransformerPtr(VectorFloatFeatureStreamPtr& src,
		      UnShrt sbFtSz = 0, UnShrt nSubFt = 0, const String& nm = "STC Transformer") {
      return new STCTransformerPtr(new STCTransformer(src, sbFtSz, nSubFt, nm));
    }

#if 0
    STCTransformerPtr(VectorFloatFeatureStreamPtr& src, const STCParam& par,
		      UnShrt sbFtSz, UnShrt nSubFt, const String& nm = "STC Transformer") {
      return new STCTransformerPtr(new STCTransformer(src, par, sbFtSz, nSubFt, nm));
    }
#endif

    STCTransformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  STCTransformer* operator->();
};


// ----- definition of class `LDATransformer' -----
//
%ignore LDATransformer;
class LDATransformer : public VectorFloatFeatureStream {
 public:
  LDATransformer(VectorFloatFeatureStreamPtr& src,
		 UnShrt sbFtSz = 0, UnShrt nSubFt = 0, const String& nm = "LDA Transformer");

  virtual const gsl_vector_float* next(int frameX = -1);

  void save(const String& fileName);

  void load(const String& fileName);
};

class LDATransformerPtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    LDATransformerPtr(VectorFloatFeatureStreamPtr& src,
		      UnShrt sbFtSz = 0, UnShrt nSubFt = 0, const String& nm = "LDA Transformer") {
      return new LDATransformerPtr(new LDATransformer(src, sbFtSz, nSubFt, nm));
    }

    LDATransformerPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  LDATransformer* operator->();
};
