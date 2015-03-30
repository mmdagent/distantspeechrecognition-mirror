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

%module train

#ifdef AUTODOC
%section "Train"
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
from asr import adapt
from asr import dict
oldimport = """
%}
%import gaussian/gaussian.i
%import adapt/adapt.i
%pythoncode %{
"""
%}

%import dictionary/dict.i
%import gaussian/gaussian.i
%import adapt/adapt.i

%{
#include "feature/feature.h"
#include "feature/lpc.h"
#include "stream/pyStream.h"
#include "train/codebookTrain.h"
#include "train/distribTrain.h"
#include "train/estimateAdapt.h"
#include "train/stateClustering.h"
#include "train/fsa.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
%}

#ifdef AUTODOC
%subsection "CodebookTrain", before
#endif

%rename(CodebookSetTrain_CodebookIterator) CodebookSetTrain::CodebookIterator;
%rename(__iter__) CodebookSetTrain::codebookIter;


// ----- definition for class `CodebookTrain' -----
// 
%ignore CodebookTrain;
class CodebookTrain : public CodebookAdapt {
 public:
  CodebookTrain(const String nm, UnShrt rfN, UnShrt dmN, CovType cvTp = COV_DIAGONAL,
		VectorFloatFeatureStreamPtr feat = NULL);
  ~CodebookTrain();

  typedef refcount_ptr<Accu>	AccuPtr;
  typedef refcount_ptr<Cache>	CachePtr;

  AccuPtr accu();
    
  // update codebook means and covariances with ML criterion
  void  update(bool verbose = false);

  // update codebook means and covariances with MMI criterion
  void  updateMMI(double E = 1.0);

  // allocate codebook accumulator
  void allocAccu();

  // save codebook accumulator
  void  saveAccu(FILE* fp, bool onlyCountsFlag = false) const;

  // load codebook accumulators
  void  loadAccu(FILE* fp);

  // clear accumulator
  void zeroAccu();

  // accumulate one frame of data
  float accumulate(float factor, const float* dsVal, int frameX, float* mixVal);

  // reset the codebook cache
  virtual void resetCache();

  // split 'addN' Gaussians with most counts
  unsigned split(float minCount = 0.0, float splitFactor = 0.2);

  // update the Gaussian normalization constant
  void fixGConsts();

  // invert covariances
  void invertVariance();
};

class CodebookTrainPtr : public CodebookAdaptPtr {
 public:
  %extend {
    CodebookTrainPtr(const String nm, UnShrt rfN, UnShrt dmN, CovType cvTp = COV_DIAGONAL,
		     VectorFloatFeatureStreamPtr feat = NULL) {
      return new CodebookTrainPtr(new CodebookTrain(nm, rfN, dmN, cvTp, feat));
    }
  }
  CodebookTrain* operator->();
};


// ----- definition for class `CodebookSetTrain' -----
//
%ignore CodebookSetTrain;
class CodebookSetTrain : public CodebookSetAdapt {
public:
  CodebookSetTrain(const String descFile = "", FeatureSetPtr fs = NULL, const String cbkFile = "",
		   double massThreshhold = 5.0);
  ~CodebookSetTrain();

  // find a codebook by name
  CodebookTrainPtr& find(const String& key);

  // allocate codebook accumulators
  void allocAccus();

  // save accumulators
  void saveAccus(const String& fileName, float totalPr = 0.0, unsigned totalT = 0,
		 bool onlyCountsFlag = false) const;

  // load accumulators
#if 0
  void loadAccus(const String& fileName, float& totalPr, unsigned& totalT, float factor = 1.0,
		 unsigned nParts = 1, unsigned part = 1);
#endif
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
};

class CodebookSetTrainPtr : public CodebookSetAdaptPtr {
 public:
  %extend {
    CodebookSetTrainPtr(const String descFile = "", FeatureSetPtr fs = NULL,
			const String cbkFile = "", double massThreshhold = 5.0) {
      return new CodebookSetTrainPtr(new CodebookSetTrain(descFile, fs, cbkFile, massThreshhold));
    }

    // return an iterator
    CodebookSetTrain::CodebookIterator* __iter__() {
      return new CodebookSetTrain::CodebookIterator(*self);
    }

    // return a codebook
    CodebookTrainPtr __getitem__(const String name) {
      return (*self)->find(name);
    }
  }
  CodebookSetTrain* operator->();
};

class CodebookSetTrain::CodebookIterator {
  public:
    CodebookSetTrain::CodebookIterator(CodebookSetTrainPtr cbs, int nParts = 1,
                                       int part = 1);
    CodebookTrainPtr next();
};


// ----- definition for class `DistribTrain' -----
//
%ignore DistribTrain;
class DistribTrain : public DistribBasic {
 public:
  DistribTrain(const String& nm, CodebookTrainPtr& cb);
  virtual ~DistribTrain();

  // accumulate one frame of data
  float accumulate(int frameX, float factor = 1.0);

  // save accumulator
  void  saveAccu(FILE* fp) const;

  // load accumulator
  void  loadAccu(FILE* fp);

  // zero accumulator
  void	zeroAccu();

  // update mixture weights with ML criterion
  void	update();

  // update mixture weights with MMI criterion
  void	updateMMI(bool merialdo = false);

  // return codebook for this distribution
  const CodebookTrainPtr cbk() const;

  // split Gaussian in codebook with highest count
  void split(float minCount = 0.0, float splitFactor = 0.2);
};

class DistribTrainPtr {
 public:
  %extend {
    DistribTrainPtr(const String& nm, CodebookTrainPtr& cb) {
      return new DistribTrainPtr(new DistribTrain(nm, cb));
    }
  }

  DistribTrain* operator->();
};

%rename(DistribSetTrain_Iterator) DistribSetTrain::Iterator;
%rename(DistribSetTrain_Iterator) Iterator(DistribSetTrain dss);


// ----- definition for class `DistribSetTrain' -----
//
%ignore DistribSetTrain;
class DistribSetTrain : public DistribSetBasic {
 public:
  DistribSetTrain(CodebookSetTrainPtr& cbs, const String& descFile, const String& distFile = "");

  // find a distribution by name
  DistribTrainPtr& find(const String& key);

  // accumulate forward-backward statistics over path
  double accumulate(DistribPathPtr& path, float factor = 1.0);

  // accumulate forward-backward statistics over all edges in lattice
  void accumulateLattice(LatticePtr& lat, float factor = 1.0);

  // save accumulators
  void saveAccus(const String& fileName);

  // load accumulators
  void loadAccus(const String& fileName, unsigned nParts = 1, unsigned part = 1);

  // clear accumulators
  void zeroAccus(int nParts = 1, int part = 1);

  // update mixture weights with ML criterion
  void update(int nParts = 1, int part = 1);

  // update mixture weights with MMI criterion
  void	updateMMI(int nParts = 1, int part = 1, bool merialdo = false);
};

class DistribSetTrainPtr : public DistribSetBasicPtr {
 public:
  %extend {
    DistribSetTrainPtr(CodebookSetTrainPtr cbs, const String& descFile, const String& distFile = "") {
      return new DistribSetTrainPtr(new DistribSetTrain(cbs, descFile, distFile));
    }

    // return an iterator
    DistribSetTrain::Iterator* __iter__() {
      return new DistribSetTrain::Iterator(*self);
    }

    // return a distrib object
    DistribTrainPtr __getitem__(const String name) {
      return (*self)->find(name);
    }
  }
  DistribSetTrain* operator->();
};

class DistribSetTrain::Iterator {
 public:
  DistribSetTrain::Iterator(DistribSetTrainPtr dss);
  DistribTrainPtr next();
};


// ----- definition for class `EstimatorTree' -----
//
%ignore EstimatorTree;
class EstimatorTree : public TransformerTree {
 protected:
  EstimatorTree(CodebookSetTrainPtr& cb, ParamTreePtr& paramTree,
		int trace, float threshold);

 public:
  void writeParams(const String& fileName);
  const ParamTreePtr& paramTree() const;

  double ttlFrames();
};

class EstimatorTreePtr : public TransformerTreePtr {
 public:
  EstimatorTree* operator->();
};

void estimateAdaptationParameters(EstimatorTreePtr& tree);

void initializeNaturalIndex(UnShrt nsub = 3);


// ----- definition for class `EstimatorTreeMLLR' -----
//
%ignore EstimatorTreeMLLR;
class EstimatorTreeMLLR : public EstimatorTree {
 public:
  EstimatorTreeMLLR(CodebookSetTrainPtr& cb, ParamTreePtr& paramTree,
		    int trace = 1, float threshold = 1500.0);
};

class EstimatorTreeMLLRPtr : public EstimatorTreePtr {
 public:
  %extend {
    EstimatorTreeMLLRPtr(CodebookSetTrainPtr& cb, ParamTreePtr& paramTree,
			 int trace = 1, float threshold = 1500.0) {
      return new EstimatorTreeMLLRPtr(new EstimatorTreeMLLR(cb, paramTree, trace, threshold));
    }
  }

  EstimatorTreeMLLR* operator->();
};


// ----- definition for class `EstimatorTreeSLAPT' -----
//
%ignore EstimatorTreeSLAPT;
class EstimatorTreeSLAPT : public EstimatorTree {
 public:
  EstimatorTreeSLAPT(CodebookSetTrainPtr& cb, ParamTreePtr& paramTree,
		     const String& biasType = "CepstraOnly",
		     UnShrt paramN = 9, unsigned noItns = 10, int trace = 1, float threshold = 150.0);
};

class EstimatorTreeSLAPTPtr : public EstimatorTreePtr {
 public:
  %extend {
    EstimatorTreeSLAPTPtr(CodebookSetTrainPtr& cb, ParamTreePtr& paramTree,
			  const String& biasType = "CepstraOnly",
			  UnShrt paramN = 9, unsigned noItns = 10, int trace = 1, float threshold = 150.0) {
      return new EstimatorTreeSLAPTPtr(new EstimatorTreeSLAPT(cb, paramTree, biasType, paramN, noItns, trace, threshold));
    }
  }

  EstimatorTreeSLAPT* operator->();
};


// ----- definition of class `STCEstimator' -----
//
%ignore STCEstimator;
class STCEstimator : public STCTransformer {
 public:
  STCEstimator(VectorFloatFeatureStreamPtr& src, ParamTreePtr& pt, DistribSetTrainPtr& dss,
	       unsigned idx = 0, BiasType bt = CepstralFeat, bool cascade = false, bool mmiEstimation = false,
	       int trace = 0001, const String& nm = "STC Estimator");

  virtual ~STCEstimator();

  EstimatorAdapt& estimate(unsigned rc = 1, double minE = 1.0, double multE = 1.0);

  void accumulate(DistribPathPtr& path, float factor = 1.0);

  // void save(const String& fileName, const gsl_matrix_float* trans = NULL);
  virtual void load(const String& fileName);

  const gsl_matrix_float* transMatrix();

  void zeroAccu();

  void saveAccu(const String& fileName) const;

  void loadAccu(const String& fileName);

  double totalPr() const;

  unsigned totalT() const;
};

class STCEstimatorPtr : public STCTransformerPtr {
 public:
  %extend {
    STCEstimatorPtr(VectorFloatFeatureStreamPtr& src, ParamTreePtr& pt, DistribSetTrainPtr& dss,
		    unsigned idx = 0, BiasType bt = CepstralFeat, bool cascade = false, bool mmiEstimation = false,
		    int trace = 0001, const String& nm = "STC Estimator") {
      return new STCEstimatorPtr(new STCEstimator(src, pt, dss, idx, bt, cascade, mmiEstimation, trace, nm));
    }

    STCEstimatorPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  STCEstimator* operator->();
};


// ----- definition for class `EstimatorTreeSTC' -----
//
%ignore EstimatorTreeSTC;
class EstimatorTreeSTC : public EstimatorTree {
 public:
  EstimatorTreeSTC(VectorFloatFeatureStreamPtr& src, DistribSetTrainPtr& dss, ParamTreePtr& paramTree,
		   const String& biasType = "CepstraOnly", int trace = 1, float threshold = 150.0);
};

class EstimatorTreeSTCPtr : public EstimatorTreePtr {
 public:
  %extend {
    EstimatorTreeSTCPtr(VectorFloatFeatureStreamPtr& src, DistribSetTrainPtr& dss, ParamTreePtr& paramTree,
			const String& biasType = "CepstraOnly", int trace = 1, float threshold = 150.0) {
      return new EstimatorTreeSTCPtr(new EstimatorTreeSTC(src, dss, paramTree, biasType, trace, threshold));
    }
  }

  EstimatorTreeSTC* operator->();
};


// ----- definition of class `LDAEstimator' -----
//
%ignore LDAEstimator;
class LDAEstimator : public LDATransformer {
 public:
  LDAEstimator(VectorFloatFeatureStreamPtr& src, ParamTreePtr& pt, DistribSetTrainPtr& dss,
	       const CodebookBasicPtr& globalCodebook, unsigned idx = 0, BiasType bt = CepstralFeat, int trace = 0001);

  virtual ~LDAEstimator();

  virtual EstimatorAdapt& estimate(unsigned rc = 1);

  double accumulate(DistribPathPtr& path, float factor = 1.0);

  void zeroAccu();

  void saveAccu(const String& fileName) const;
  void loadAccu(const String& fileName);

  /*
  double totalPr() const;
  unsigned totalT() const;
  */
};

class LDAEstimatorPtr : public LDATransformerPtr {
 public:
  %extend {
    LDAEstimatorPtr(VectorFloatFeatureStreamPtr& src, ParamTreePtr& pt, DistribSetTrainPtr& dss,
		    const CodebookBasicPtr& globalCodebook, unsigned idx = 0, BiasType bt = CepstralFeat, int trace = 0001) {
      return new LDAEstimatorPtr(new LDAEstimator(src, pt, dss, globalCodebook, idx, bt, trace));
    }

    LDAEstimatorPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  LDAEstimator* operator->();
};


#if 0

// ----- definition for class `EstimatorTreeLDA' -----
//
%ignore EstimatorTreeLDA;
class EstimatorTreeLDA : public EstimatorTree {
 public:
  EstimatorTreeLDA(VectorFloatFeatureStreamPtr& src, DistribSetTrainPtr& dss, ParamTreePtr& paramTree,
		   const CodebookBasicPtr& globalCodebook, const String& biasType = "CepstraOnly",
		   int trace = 1, float threshold = 150.0);
};

class EstimatorTreeLDAPtr : public EstimatorTreePtr {
 public:
  %extend {
    EstimatorTreeLDAPtr(VectorFloatFeatureStreamPtr& src, DistribSetTrainPtr& dss, ParamTreePtr& paramTree,
			const CodebookBasicPtr& globalCodebook, const String& biasType = "CepstraOnly",
			int trace = 1, float threshold = 150.0) {
      return new EstimatorTreeLDAPtr(new EstimatorTreeLDA(src, dss, paramTree, globalCodebook, biasType, trace, threshold));
    }
  }

  EstimatorTreeLDA* operator->();
};

#endif


// ----- definition for class `FeatureSpaceAdaptationFeature' -----
//
%ignore FeatureSpaceAdaptationFeature;
class FeatureSpaceAdaptationFeature : public VectorFloatFeatureStream {
  class SortElement;
 public:
  FeatureSpaceAdaptationFeature(VectorFloatFeatureStreamPtr& src,
				unsigned maxAccu = 1, unsigned maxTran = 1,
				const String& nm = "FeatureSpaceAdaptationFeature");
  ~FeatureSpaceAdaptationFeature();

  // accessor methods
  const String& name() const;
  unsigned topN() const;
  void	   setTopN(unsigned topN);
  unsigned count(unsigned accuX = 0) const;

  float    shift() const;
  void     setShift(float shift);
  void     distribSet(DistribSetBasicPtr& dssP);

  // accumulation methods
  void	accumulate(DistribPathPtr& path, unsigned accuX = 0, float factor = 1.0);
  void	accumulateLattice(LatticePtr& lat, unsigned accuX = 0, float factor = 1.0);

  void	zeroAccu(unsigned accuX = 0);
  void	scaleAccu(float scale, unsigned accuX = 0);
  void	addAccu(unsigned accuX, unsigned accuY, float factor = 1.0);

  // estimation methods
  void	add(unsigned dsX);
  void	addAll();
  void	estimate(unsigned iterN = 1, unsigned accuX = 0, unsigned tranX = 0);

  void	clear(unsigned tranX = 0);
  float compareTransform(unsigned tranX, unsigned tranY);

  // read and write methods
  void	saveAccu(const String& name, unsigned accuX = 0);
  void	loadAccu(const String& name, unsigned accuX = 0, float factor = 1.0);

  void	save(const String& name, unsigned tranX = 0);
  void	load(const String& name, unsigned tranX = 0);
};

class FeatureSpaceAdaptationFeaturePtr : public VectorFloatFeatureStreamPtr {
 public:
  %extend {
    FeatureSpaceAdaptationFeaturePtr(VectorFloatFeatureStreamPtr& src,
				     unsigned maxAccu = 1, unsigned maxTran = 1,
				     const String& nm = "FeatureSpaceAdaptationFeature") {
      return new FeatureSpaceAdaptationFeaturePtr(new FeatureSpaceAdaptationFeature(src, maxAccu, maxTran, nm));
    }

    FeatureSpaceAdaptationFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }
  FeatureSpaceAdaptationFeature* operator->();
};


// ----- definition for class `CodebookClusterDistribTree' -----
// 
%ignore CodebookClusterDistribTree;
class CodebookClusterDistribTree : public DistribTree {
 public:
  CodebookClusterDistribTree(const String& nm, PhonesSetPtr& phonesSet, CodebookSetTrainPtr& cbs, const int contextLength = 1, const String& fileName = "",
			     char comment = ';', const String& wb = "WB", const String& eos = "</s>", bool verbose = false, unsigned minCount = 1000);
  ~CodebookClusterDistribTree();

  void grow(unsigned leavesN);
  void write(const String fileName, const String& date = "");
  void writeContexts(const String& fileName = "");
};

class CodebookClusterDistribTreePtr /* : public DistribTreePtr */ {
 public:
  %extend {
    CodebookClusterDistribTreePtr(const String& nm, PhonesSetPtr& phonesSet, CodebookSetTrainPtr& cbs, const int contextLength = 1, const String& fileName = "",
				  char comment = ';', const String& wb = "WB", const String& eos = "</s>", bool verbose = false, unsigned minCount = 1000) {
      return new CodebookClusterDistribTreePtr(new CodebookClusterDistribTree(nm, phonesSet, cbs, contextLength, fileName, comment, wb, eos, verbose, minCount));
    }
  }
  CodebookClusterDistribTree* operator->();
};

CodebookClusterDistribTreePtr
clusterCodebooks(PhonesSetPtr& phonesSet, CodebookSetTrainPtr& cbs, unsigned leavesN, const int contextLength = 1, const String& fileName = "",
		 char comment = ';', const String& wb = "WB", const String& eos = "</s>", bool verbose = false, unsigned minCount = 1000);


// ----- definition for class `MixtureWeightClusterDistribTree' -----
// 
%ignore MixtureWeightClusterDistribTree;
class MixtureWeightClusterDistribTree : public DistribTree {
 public:
  MixtureWeightClusterDistribTree(const String& nm, PhonesSetPtr& phonesSet, DistribSetTrainPtr& dss, const int contextLength = 1, const String& fileName = "",
				  char comment = ';', const String& wb = "WB", const String& eos = "</s>", bool verbose = false, unsigned minCount = 1000);
  ~MixtureWeightClusterDistribTree();

  void grow(unsigned leavesN);
  void write(const String fileName, const String& date = "");
  void writeContexts(const String& fileName = "");
};

class MixtureWeightClusterDistribTreePtr /* : public DistribTreePtr */ {
 public:
  %extend {
    MixtureWeightClusterDistribTreePtr(const String& nm, PhonesSetPtr& phonesSet, DistribSetTrainPtr& dss, const int contextLength = 1, const String& fileName = "",
				       char comment = ';', const String& wb = "WB", const String& eos = "</s>", bool verbose = false, unsigned minCount = 1000) {
      return new MixtureWeightClusterDistribTreePtr(new MixtureWeightClusterDistribTree(nm, phonesSet, dss, contextLength, fileName, comment, wb, eos, verbose, minCount));
    }
  }
  MixtureWeightClusterDistribTree* operator->();
};

MixtureWeightClusterDistribTreePtr
clusterCodebooks(PhonesSetPtr& phonesSet, CodebookSetTrainPtr& cbs, unsigned leavesN, const int contextLength = 1, const String& fileName = "",
		 char comment = ';', const String& wb = "WB", const String& eos = "</s>", bool verbose = false, unsigned minCount = 1000);
