//                              -*- C++ -*-
//
//                              Millennium
//                  Automatic Speech Recognition System
//                                 (asr)
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


%module sat

#ifdef AUTODOC
%section "SAT"
#endif

/* operator overloading */
%rename(__str__) *::puts();

%include gsl/gsl_types.h
%include jexception.i
%include typedefs.i

%pythoncode %{
from btk import feature
from btk import stream
from asr import gaussian
from asr import adapt
from asr import train
oldimport = """
%}
%import train/train.i
%pythoncode %{
"""
%}

%{
#include "feature/feature.h"
#include "feature/lpc.h"
#include "stream/pyStream.h"
#include "train/fsa.h"
#include "sat/codebookFastSAT.h"
#include "sat/sat.h"
#include "sat/cluster.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
%}

#ifdef AUTODOC
%subsection "Codebook Fast SAT", before
#endif

%rename(CodebookSetFastSAT_CodebookIterator) CodebookSetFastSAT::CodebookIterator;


// ----- definition for class `CodebookFastSat' ----- 
//
%ignore CodebookFastSAT;
class CodebookFastSAT : public CodebookTrain {
 public:
  CodebookFastSAT(const String& nm, UnShrt rfN, UnShrt dmN, CovType cvTp,
                  VectorFloatFeatureStreamPtr feat);
  ~CodebookFastSAT();

  void reallocFastAccu(const TransformerTreePtr& tree, unsigned olen);
};

class CodebookFastSATPtr : public CodebookTrainPtr {
 public:
  %extend {
    CodebookFastSATPtr(const String& nm, UnShrt rfN, UnShrt dmN, CovType cvTp,
                       VectorFloatFeatureStreamPtr feat) {
      return new CodebookFastSATPtr(new CodebookFastSAT(nm, rfN, dmN, cvTp, feat));
    }
  }
  CodebookFastSAT* operator->();
};

#if 0

class CodebookSetFastSAT::CodebookIterator {
 public:
  CodebookSetFastSAT::CodebookIterator(CodebookSetFastSATPtr cbs,
                                       int nParts = 1, int part = 1);
  CodebookAdaptPtr next();
};

#endif


// ----- definition for class `CodebookSetFastSat' ----- 
// 
%ignore CodebookSetFastSAT;
class CodebookSetFastSAT : public CodebookSetTrain {
 public:
  CodebookSetFastSAT(const String& descFile = "", FeatureSetPtr fs = NULL, const String& cbkFile = "");
  virtual ~CodebookSetFastSAT();

  class CodebookIterator;  friend class CodebookIterator;
  class GaussianIterator;  friend class GaussianIterator;

  // GaussianIterator* gaussIter();

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
  void reallocFastAccus();

  CodebookFastSATPtr&       find(const String key);
};

class CodebookSetFastSATPtr : public CodebookSetTrainPtr {
 public:
  %extend {
    CodebookSetFastSATPtr(const String descFile = "", FeatureSetPtr fs = NULL,
			  const String cbkFile = "") {
      return new CodebookSetFastSATPtr(new CodebookSetFastSAT(descFile, fs, cbkFile));
    }

    // return an iterator
    CodebookSetFastSAT::CodebookIterator* __iter__() {
      return new CodebookSetFastSAT::CodebookIterator(*self);
    }

    // return a codebook
    CodebookBasicPtr __getitem__(const String name) {
      return (*self)->find(name);
    }
  }
  CodebookSetFastSAT* operator->();
};


// ----- definition for class `FastMeanVarianceSAT' ----- 
// 
%ignore FastMeanVarianceSAT;
class FastMeanVarianceSAT {
 public:
  FastMeanVarianceSAT(CodebookSetFastSATPtr& cbs, UnShrt meanLen, double lhoodThreshhold = 0.1,
		      double massThreshhold = 10.0, unsigned nParts = 1, unsigned part = 1,
		      double mmiMultiplier = 1.0, bool meanOnly = false, int maxNoRegClass = 1);

  double estimate(const SpeakerListPtr& sList, const String& spkrAdaptParms,
		  const String& meanVarsStats);
};

class FastMeanVarianceSATPtr {
 public:
  %extend {
    FastMeanVarianceSATPtr(CodebookSetFastSATPtr& cbs, UnShrt meanLen, double lhoodThreshhold = 0.1,
			   double massThreshhold = 10.0, unsigned nParts = 1, unsigned part = 1,
			   double mmiMultiplier = 1.0, bool meanOnly = false, int maxNoRegClass = 1)
    {
      return new FastMeanVarianceSATPtr(new FastMeanVarianceSAT(cbs, meanLen, lhoodThreshhold,
								massThreshhold, nParts, part,
								mmiMultiplier, meanOnly, maxNoRegClass));
    }
  }
  FastMeanVarianceSAT* operator->();
};


// ----- definition for class `FastRegClassSAT' ----- 
// 
%ignore FastRegClassSAT;
class FastRegClassSAT {
 public:
  FastRegClassSAT(CodebookSetFastSATPtr& cbs, UnShrt meanLen, double lhoodThreshhold = 0.1,
		  double massThreshhold = 10.0, unsigned nParts = 1, unsigned part = 1,
		  double mmiMultiplier = 1.0, bool meanOnly = false, int maxNoRegClass = 1);

  double estimate(const SpeakerListPtr& sList, const String& spkrAdaptParms,
		  const String& meanVarsStats);
};

class FastRegClassSATPtr {
 public:
  %extend {
    FastRegClassSATPtr(CodebookSetFastSATPtr& cbs, UnShrt meanLen, double lhoodThreshhold = 0.1,
		       double massThreshhold = 10.0, unsigned nParts = 1, unsigned part = 1,
		       double mmiMultiplier = 1.0, bool meanOnly = false, int maxNoRegClass = 1)
    {
      return new FastRegClassSATPtr(new FastRegClassSAT(cbs, meanLen, lhoodThreshhold,
							massThreshhold, nParts, part,
							mmiMultiplier, meanOnly, maxNoRegClass));
    }
  }
  FastRegClassSAT* operator->();
};


// ----- definition for class `FastMaxMutualInfoSAT' ----- 
// 
%ignore FastMaxMutualInfoSAT;
class FastMaxMutualInfoSAT {
 public:
  FastMaxMutualInfoSAT(CodebookSetFastSATPtr& cbs, UnShrt meanLen, double lhoodThreshhold,
		       double massThreshhold, unsigned nParts = 1, unsigned part = 1,
		       double mmiMultiplier = 1.0, bool meanOnly = false, int maxNoRegClass = 1);

  double estimate(const SpeakerListPtr& sList, const String& spkrAdaptParms,
		  const String& meanVarsStats);
};

class FastMaxMutualInfoSATPtr {
 public:
  %extend {
    FastMaxMutualInfoSATPtr(CodebookSetFastSATPtr& cbs, UnShrt meanLen, double lhoodThreshhold,
			    double massThreshhold, unsigned nParts = 1, unsigned part = 1,
			    double mmiMultiplier = 1.0, bool meanOnly = false, int maxNoRegClass = 1)
    {
      return new FastMaxMutualInfoSATPtr(new FastMaxMutualInfoSAT(cbs, meanLen, lhoodThreshhold,
								  massThreshhold, nParts, part,
								  mmiMultiplier, meanOnly, maxNoRegClass));
    }
  }
  FastMaxMutualInfoSAT* operator->();
};


// ----- definition for class `FastMMIRegClassSAT' ----- 
// 
%ignore FastMMIRegClassSAT;
class FastMMIRegClassSAT {
 public:
  FastMMIRegClassSAT(CodebookSetFastSATPtr& cbs, UnShrt meanLen, double lhoodThreshhold = 0.1,
		     double massThreshhold = 10.0, unsigned nParts = 1, unsigned part = 1,
		     double mmiMultiplier = 1.0, bool meanOnly = false, int maxNoRegClass = 1);

  double estimate(const SpeakerListPtr& sList, const String& spkrAdaptParms,
		  const String& meanVarsStats);
};

class FastMMIRegClassSATPtr {
 public:
  %extend {
    FastMMIRegClassSATPtr(CodebookSetFastSATPtr& cbs, UnShrt meanLen, double lhoodThreshhold = 0.1,
			  double massThreshhold = 10.0, unsigned nParts = 1, unsigned part = 1,
			  double mmiMultiplier = 1.0, bool meanOnly = false, int maxNoRegClass = 1)
    {
      return new FastMMIRegClassSATPtr(new FastMMIRegClassSAT(cbs, meanLen, lhoodThreshhold,
							      massThreshhold, nParts, part,
							      mmiMultiplier, meanOnly, maxNoRegClass));
    }
  }
  FastMMIRegClassSAT* operator->();
};

void incrementRegClasses(CodebookSetFastSATPtr& cbs, unsigned toAdd = 1, int trace = 0x0041,
			 const String& spkrParamFile = "", const String& spkrMeanVarStats = "",
			 const String& spkrListFile = "", const String& gCovFile = "",
			 bool onlyTopOne = false);
