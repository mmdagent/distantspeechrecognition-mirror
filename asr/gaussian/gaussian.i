//                              -*- C++ -*-
//
//                              Millennium
//                  Automatic Speech Recognition System
//                                 (asr)
//
//  Module:  asr.gaussian
//  Purpose: Basic acoustic likelihood computation.
//  Author:  John McDonough

%module gaussian

#ifdef AUTODOC
%section "Gaussian"
#endif

/* operator overloading */
 // %rename(__str__) *::name();

%include typedefs.i
%include jexception.i

typedef unsigned short UnShrt;

%pythoncode %{
import btk
from btk import feature
from btk import stream
oldimport = """
%}
%import feature/feature.i
%pythoncode %{
"""
%}

%{
#include <numpy/arrayobject.h>
#include <stdio.h>

#include "codebookBasic.h"
#include "feature/feature.h"
#include "feature/lpc.h"
#include "stream/pyStream.h"
%}

%init {
  // NumPy needs to set up callback functions
  import_array();
}

#ifdef AUTODOC
%subsection "CodebookBasic", before
#endif

%include gsl/gsl_types.h
%include complex.i
%include matrix.i
%include vector.i

%rename(CodebookSetBasic_CodebookIterator) CodebookSetBasic::CodebookIterator;
%rename(CodebookSetBasic_CodebookIterator) CodebookIterator(CodebookSetBasic& cbs, int nParts = 1, int part = 1);


// ----- definition for class `LogLhoodIndex' -----
// 
%ignore LogLhoodIndex;
class LogLhoodIndex {
 public:
  float lhood() const;
  int   index() const;
};

class LogLhoodIndexPtr {
 public:
   %extend {
     LogLhoodIndexPtr(float logLhood = 0.0, int idx = 0) {
       return new LogLhoodIndexPtr(new LogLhoodIndex(logLhood, idx));
     }
   }
   LogLhoodIndex* operator->();
};


// ----- definition for class `CodebookBasic' -----
// 
%ignore CodebookBasic;
class CodebookBasic {
 public:
  CodebookTrain(const String& nm, UnShrt rfN, UnShrt dmN, CovType cvTp = COV_DIAGONAL,
		VectorFloatFeatureStreamPtr feat = NULL);
  ~CodebookBasic();

  // name of this codebook
  String name();

  // number of Gaussian densities in this codebook
  UnShrt refN();

  // number of sub-features
  UnShrt nSubFeat();

  // length of sub-feature
  UnShrt subFeatLen();

  // final feature length
  UnShrt featLen();

  // original feature length
  UnShrt orgFeatLen();

  // save the codebook to disk
  void save(FILE* fp) const;

  // load the codebook from disk
  void load(FILE* fp, bool readName = false);

  // load the codebook from disk
  void loadOld(FILE* fp);

  // access a mean element
  float mean(UnShrt refX, UnShrt compX) const;

  // access an inverse covariance element
  float invCov(UnShrt refX, UnShrt compX) const;

  // simplified log-likelihood calculation
  LogLhoodIndexPtr logLhood(const gsl_vector* frame, float* val = NULL) const;

  // specify exact acoustic likelihood calculation
  void setScoreAll(unsigned cacheN = 100);

  // set all regression classes
  void setRegClasses(UnShrt c = 1);

  // set acoustic score scale factor
  void setScale(float scale = 1.0);

  // copy mean variance
  void copyMeanVariance(const CodebookBasicPtr& cb);

  // set single mean component
  void setMean(unsigned refX, unsigned compX, float mean);

  // set single variance component
  void setVariance(unsigned refX, unsigned compX, float mean);

  // total training counts
  float ttlCounts() const;
};

class CodebookBasicPtr {
 public:
  %extend {
    CodebookBasicPtr(const String& nm, UnShrt rfN, UnShrt dmN, CovType cvTp = COV_DIAGONAL,
		     VectorFloatFeatureStreamPtr feat = NULL) {
      return new CodebookBasicPtr(new CodebookBasic(nm, rfN, dmN, cvTp, feat));
    }

    void copyMeans(PyObject* means) {
      for (unsigned refX = 0; refX < (*self)->refN(); refX++) {
	for (unsigned compX = 0; compX < (*self)->featLen(); compX++) {
	  (*self)->setMean(refX, compX, Cast<float>(PyArray_GETPTR2(means, Cast<npy_intp>(refX), Cast<npy_intp>(compX))));
	}
      }
    }

    void copyVariances(PyObject* variances) {
      for (unsigned refX = 0; refX < (*self)->refN(); refX++) {
	for (unsigned compX = 0; compX < (*self)->featLen(); compX++) {
	  (*self)->setMean(refX, compX, Cast<float>(PyArray_GETPTR2(variances, Cast<npy_intp>(refX), Cast<npy_intp>(compX))));
	}
      }
    }

  String __str__() {
      return (*self)->name();
    }
  }

  CodebookBasic* operator->();
};


// ----- definition for class `CodebookSetBasic' -----
//
%ignore CodebookSetBasic;
class CodebookSetBasic {
 public:
  CodebookSetBasic(const String& descFile = "", FeatureSetPtr& fs = NullFeatureSetPtr, const String& cbkFile = "");

  ~CodebookSetBasic();

  void setFeatures(FeatureSetPtr& featureSet);

  // number of sub-features
  UnShrt nSubFeat();

  // length of sub-feature
  UnShrt subFeatLen();

  // final feature length
  UnShrt featLen();

  // original feature length
  UnShrt orgFeatLen();

  // original sub-feature length
  UnShrt orgSubFeatLen();

  // set the number of sub-features
  void setNSubFeat(UnShrt nsub);

  // set the sub-feature length
  void setSubFeatLen(UnShrt len);

  // set the feature length
  void setFeatLen(UnShrt len);

  // set the original feature length
  void setOrgFeatLen(UnShrt len);

  // set the sub-feature length
  void setOrgSubFeatLen(UnShrt len);

  // LDA file name
  const String ldaFile();

  // length of cepstral sub-feature
  UnShrt cepSubFeatLen();

  // number of cepstral sub-features
  UnShrt cepNSubFeat();

  // length of original cepstral sub-feature
  UnShrt cepOrgSubFeatLen();

  // reset the cache
  void resetCache();

  // load codebooks from file
  void load(const String filename);

  // save codebooks to file
  void save(const String& filename, bool janusFormat = false, int nParts = 1, int part = 1);

  // write codebook set description to file
  void write(const String& fileName, const String& time = "") const;

  // how many codebooks in this set
  int ncbks() const;

  // find a codebook by name
  CodebookBasicPtr find(const String key);

  // set all regression classes
  void setRegClasses(UnShrt c = 1);

  // set exact acoustic likelihood computation
  void setScoreAll(unsigned cacheN = 1000);

  // apply STC transformation to all Gaussian means
  void applySTC(const gsl_matrix_float* trans);

  // set acoustic score scale factor
  void setScale(float scale = 1.0);
};

class CodebookSetBasic::CodebookIterator {
 public:
  CodebookSetBasic::CodebookIterator(CodebookSetBasicPtr& cbs,
                                     int nParts = 1, int part = 1);

  bool more() const;
  const CodebookBasicPtr cbk() const;
  CodebookBasicPtr next();
};

class CodebookSetBasicPtr {
 public:
  %extend {
    CodebookSetBasicPtr(const String descFile = "", FeatureSetPtr fs = NULL,
			const String cbkFile = "") {
      return new CodebookSetBasicPtr(new CodebookSetBasic(descFile, fs, cbkFile));
    }

    // return an iterator
    CodebookSetBasic::CodebookIterator* __iter__() {
      return new CodebookSetBasic::CodebookIterator(*self);
    }

    // return a codebook
    CodebookBasicPtr __getitem__(const String name) {
      return (*self)->find(name);
    }

    // return a distrib object
    CodebookBasicPtr __getitem__(unsigned dsX) {
      return (*self)->find(dsX);
    }
  }
  CodebookSetBasic* operator->();
};


%{
#include "distribBasic.h"
%}

#ifdef AUTODOC
%subsection "DistribBasic", before
#endif

%rename(DistribSetBasic_Iterator) DistribSetBasic::Iterator;

// ----- definition for class `Distrib' -----
//
%ignore Distrib;
class Distrib {
 public:
  Distrib(const String& nm);
  ~Distrib();

  float score(int frameX);

  const String&	name() const { return _name; }

  CodebookBasicPtr cbk();
};

class DistribPtr {
 public:
  %extend {
    String __str__() {
      return (*self)->name();
    }
  }

  Distrib* operator->();
};


// ----- definition for class `DistribBasic' -----
// 
%ignore DistribBasic;
class DistribBasic : public Distrib {
 public:
  DistribBasic(const String& nm, CodebookBasicPtr& cb);
  virtual ~DistribBasic();

  // save the distribution to a file
  void save(FILE *fp);

  // load the distribution from a file
  void load(FILE *fp, bool readName = false);

  // return codebook for this distribution
  const CodebookBasicPtr cbk() const;

  // return acoustic score for 'frame' and best Gaussian index
  LogLhoodIndexPtr logLhood(const gsl_vector* frame) const;

  // copy mixture weights
  void copyWeights(const DistribBasicPtr& ds);

  // set single weight
  void setWght(unsigned compX, float wgt);
};

class DistribBasicPtr {
 public:
  %extend {
    DistribBasicPtr(const String& nm, CodebookBasicPtr& cb) {
      return new DistribBasicPtr(new DistribBasic(nm, cb));
    }

    void copyMixtureWeights(const PyObject* weights) {
      for (unsigned n = 0; n < (*self)->valN(); n++) {
	(*self)->setWght(n, Cast<float>(PyArray_GETPTR1(weights, Cast<npy_intp>(n))));
      }
    }

    String __str__() {
      return (*self)->name();
    }
  }

  DistribBasic* operator->();
};

%rename(DistribSet_Iterator) DistribSet::Iterator;
%rename(DistribSet_Iterator) Iterator(DistribSet dss);


// ----- definition for class `DistribSet' -----
//
%ignore DistribSet;
class DistribSet {
 public:
  DistribSet();

  // reset the cache
  void resetCache();

  // how many distributions in this set
  int ndists() const;

  // find a distribution by name
  DistribPtr find(const String key);
};

class DistribSetPtr {
 public:
  %extend {
    // return an iterator
    DistribSet::Iterator* __iter__() {
      return new DistribSet::Iterator(*self);
    }

    // return a distrib object
    DistribPtr __getitem__(const String name) {
      return (*self)->find(name);
    }
  }
  DistribSet* operator->();
};

class DistribSet::Iterator {
 public:
  DistribSet::Iterator(DistribSetPtr dss);
  DistribPtr next();
};


// ----- definition for class `DistribSetBasic' -----
//
%ignore DistribSetBasic;
class DistribSetBasic : public DistribSet {
 public:
  DistribSetBasic(CodebookSetBasicPtr& cbs, const String& descFile = "",
		  const String& distFile = "");

  // load all distributions from a file
  void load(const String distFile);

  // save all distributions to a file
  void save(const String distFile) const;

  // write distribution set description to file
  void write(const String& fileName, const String& time = "") const;

  // find a distribution by name
  DistribBasicPtr find(const String key);

  // find a distribution by index
  // DistribBasicPtr find(unsigned dsX);

  void resetCache();

  void resetFeature();
};

class DistribSetBasicPtr  : public DistribSetPtr {
 public:
  %extend {
    DistribSetBasicPtr(CodebookSetBasicPtr cbs, const String descFile, 
		       const String distFile = "") {
      return new DistribSetBasicPtr(new DistribSetBasic(cbs, descFile, distFile));
    }

    // return an iterator
    DistribSetBasic::Iterator* __iter__() {
      return new DistribSetBasic::Iterator(*self);
    }

    // return a distrib object
    DistribBasicPtr __getitem__(const String name) {
      return (*self)->find(name);
    }

    // return a distrib object
    DistribBasicPtr __getitem__(unsigned dsX) {
      return (*self)->find(dsX);
    }
  }
  DistribSetBasic* operator->();
};

class DistribSetBasic::Iterator {
 public:
  DistribSetBasic::Iterator(DistribSetBasicPtr dss);
  DistribBasicPtr next();
};


// ----- definition for class `DistribMap' -----
//
%ignore DistribMap;
class DistribMap : public Countable {
public:
  DistribMap();
  ~DistribMap();

  const String& mapped(const String& name);
  void add(const String& fromName, const String& toName);

  void read(const String& fileName);
  void write(const String& fileName) const;
};

class DistribMapPtr {
 public:
  %extend {
    DistribMapPtr() {
      return new DistribMapPtr(new DistribMap);
    }

    String __getitem__(const String name) {
      return (*self)->mapped(name);
    }
  }

  DistribMap* operator->();
};


// ----- definition for class `DistribSetMultiStream' -----
//
%ignore DistribSetMultiStream;
class DistribSetMultiStream : public DistribSet {
 public:
  DistribSetMultiStream(DistribSetBasicPtr& audioDistribSet, DistribSetBasicPtr& videoDistribSet, double audioWeight = 0.95, double videoWeight = 0.05);
  DistribSetMultiStream(DistribSetBasicPtr& audioDistribSet, DistribSetBasicPtr& videoDistribSet, DistribMapPtr& distribMap, double audioWeight = 0.95, double videoWeight = 0.05);

  // find a distribution by name
  DistribMultiStreamPtr& find(const String key);

  DistribSetBasicPtr& audioDistribSet() { return _audioDistribSet; }
  DistribSetBasicPtr& videoDistribSet() { return _videoDistribSet; }

  void setWeights( double audioWeight, double videoWeight );
};

class DistribSetMultiStreamPtr : public DistribSetPtr {
 public:
  %extend {
    DistribSetMultiStreamPtr(DistribSetBasicPtr& audioDistribSet, DistribSetBasicPtr& videoDistribSet, double audioWeight = 0.95, double videoWeight = 0.05) {
      return new DistribSetMultiStreamPtr(new DistribSetMultiStream(audioDistribSet, videoDistribSet, audioWeight, videoWeight));
    }

    DistribSetMultiStreamPtr(DistribSetBasicPtr& audioDistribSet, DistribSetBasicPtr& videoDistribSet, DistribMapPtr& distribMap, double audioWeight = 0.95, double videoWeight = 0.05) {
      return new DistribSetMultiStreamPtr(new DistribSetMultiStream(audioDistribSet, videoDistribSet, distribMap, audioWeight, videoWeight));
    }

    // return an iterator
    DistribSetMultiStream::Iterator* __iter__() {
      return new DistribSetMultiStream::Iterator(*self);
    }

    // return a distrib object
    DistribMultiStreamPtr __getitem__(const String name) {
      return (*self)->find(name);
    }
  }
  DistribSetMultiStream* operator->();
};

#if 0

class DistribSetMultiStream::Iterator {
 public:
  DistribSetMultiStream::Iterator(DistribSetMultiStreamPtr dss);
  DistribMultiStreamPtr next();
};

#endif
