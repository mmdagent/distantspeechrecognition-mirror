//
//                       Beamforming Toolkit
//                              (btk)
//
//  Module:  btk.portnoff
//  Purpose: Portnoff filter bank.
//  Author:  John McDonough.

%module(package="btk") portnoff

%{
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "portnoff.h"
#include "stream/stream.h"
#include "stream/pyStream.h"
#include "portnoff.h"
%}

%init {
  // NumPy needs to set up callback functions
  import_array();
}

typedef int size_t;

%include jexception.i
%include complex.i
%include matrix.i
%include vector.i

%pythoncode %{
import btk
from btk import stream
oldimport = """
%}
%import stream/stream.i
%pythoncode %{
"""
%}

%rename(__str__) *::print;
%rename(__getitem__) *::operator[](int);
%ignore *::nextSampleBlock(const double* smp);


// ----- definition for class `PortnoffAnalysisBank' -----
// 
%ignore PortnoffAnalysisBank;
class PortnoffAnalysisBank : public VectorComplexFeatureStream {
 public:
  PortnoffAnalysisBank(VectorShortFeatureStreamPtr& src,
		       unsigned fftLen = 256, unsigned nBlocks = 6, unsigned subSampRate = 2,
		       const String& nm = "Portnoff Analysis Bank");
  ~PortnoffAnalysisBank();

  virtual const gsl_vector_complex* next(int frameX = -5);

  virtual void reset();

  unsigned fftLen() const;
  unsigned fftLen2() const;
  unsigned nBlocks() const;
  unsigned nBlocks2() const;
  unsigned subSampRate() const;
};

class PortnoffAnalysisBankPtr : public VectorComplexFeatureStreamPtr {
 public:
  %extend {
    PortnoffAnalysisBankPtr(VectorShortFeatureStreamPtr& src,
                            unsigned fftLen = 256, unsigned nBlocks = 6, unsigned subSampRate = 2,
                            const String& nm = "Portnoff Analysis Bank") {
      return new PortnoffAnalysisBankPtr(new PortnoffAnalysisBank(src, fftLen, nBlocks, subSampRate, nm));
    }

    PortnoffAnalysisBankPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  PortnoffAnalysisBank* operator->();
};


// ----- definition for class `PortnoffSynthesisBank' -----
// 
%ignore PortnoffSynthesisBank;
class PortnoffSynthesisBank : public VectorShortFeatureStream {
 public:
  PortnoffSynthesisBank(VectorComplexFeatureStreamPtr& src,
			unsigned fftLen = 256, unsigned nBlocks = 6, unsigned subSampRate = 2,
			const String& nm = "Portnoff Synthesis Bank");
  ~PortnoffSynthesisBank();

  virtual const gsl_vector_short* next(int frameX = -5);
};

class PortnoffSynthesisBankPtr : public VectorShortFeatureStreamPtr {
 public:
  %extend {
    PortnoffSynthesisBankPtr(VectorComplexFeatureStreamPtr& src,
                             unsigned fftLen = 256, unsigned nBlocks = 6, unsigned subSampRate = 2,
                             const String& nm = "Portnoff Synthesis Bank") {
      return new PortnoffSynthesisBankPtr(new PortnoffSynthesisBank(src, fftLen, nBlocks, subSampRate));
    }

    PortnoffSynthesisBankPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  PortnoffSynthesisBank* operator->();
};


%rename(__str__) print;
%ignore *::print();
