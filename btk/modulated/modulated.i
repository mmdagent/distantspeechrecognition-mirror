//                              -*- C++ -*-
//
//                                Nemesis
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.modulated
//  Purpose: Cosine modulated analysis and synthesis filter banks.
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


%module(package="btk") modulated

%{
#include "modulated/modulated.h"
#include "stream/stream.h"
#include "stream/pyStream.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "modulated/prototypeDesign.h"
%}

%init {
  // NumPy needs to set up callback functions
  import_array();
}

typedef int size_t;

%include gsl/gsl_types.h
%include jexception.i
%include typedefs.i
%include complex.i
%include matrix.i
%include vector.i

%rename(__str__) *::print;
%rename(__getitem__) *::operator[](int);
%ignore *::nextSampleBlock(const double* smp);

%pythoncode %{
import btk
from btk import stream
oldimport = """
%}
%import stream/stream.i
%pythoncode %{
"""
%}

// ----- definition for class `NormalFFTAnalysisBank' -----
//
%ignore NormalFFTAnalysisBank;
class NormalFFTAnalysisBank : public VectorComplexFeatureStream {
public:
  NormalFFTAnalysisBank(VectorFloatFeatureStreamPtr& samp,
			unsigned fftLen,  unsigned r = 1, unsigned windowType = 1, 
			const String& nm = "NormalFFTAnalysisBank");
  ~NormalFFTAnalysisBank();
  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();
  unsigned fftLen() const;
};

class NormalFFTAnalysisBankPtr : public VectorComplexFeatureStreamPtr {
 public:
  %extend {
    NormalFFTAnalysisBankPtr(VectorFloatFeatureStreamPtr& samp,
			     unsigned fftLen,  unsigned r = 1, unsigned windowType = 1, 
			     const String& nm = "NormalFFTAnalysisBank") {
      return new NormalFFTAnalysisBankPtr(new NormalFFTAnalysisBank( samp, fftLen, r, windowType, nm ));
    }

     NormalFFTAnalysisBankPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

   NormalFFTAnalysisBank* operator->();
};

// ----- definition for class `OverSampledDFTAnalysisBank' -----
//
%ignore OverSampledDFTAnalysisBank;
class OverSampledDFTAnalysisBank : public VectorComplexFeatureStream {
 public:
  OverSampledDFTAnalysisBank(VectorShortFeatureStreamPtr& samp,
			     gsl_vector* prototype, unsigned M = 256, unsigned m = 3, unsigned r, unsigned delayCompensationType =0,
			     const String& nm = "OverSampledDFTAnalysisBank");
  ~OverSampledDFTAnalysisBank();

  double polyphase(unsigned m, unsigned n) const;

  virtual const gsl_vector_complex* next(int frameX = -5);

  virtual void reset();

  bool isEnd();
  unsigned fftLen() const;
  unsigned nBlocks() const;
  unsigned subSampRate() const;
};

class OverSampledDFTAnalysisBankPtr : public VectorComplexFeatureStreamPtr {
 public:
  %extend {
    OverSampledDFTAnalysisBankPtr(VectorFloatFeatureStreamPtr& samp,
				  gsl_vector* prototype, unsigned M = 256, unsigned m = 3, unsigned r = 0, unsigned delayCompensationType =0,
				  const String& nm = "OverSampledDFTAnalysisBankFloat") {
      return new OverSampledDFTAnalysisBankPtr(new OverSampledDFTAnalysisBank(samp, prototype, M, m, r, delayCompensationType, nm));
    }

    OverSampledDFTAnalysisBankPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  OverSampledDFTAnalysisBank* operator->();
};


// ----- definition for class `OverSampledDFTSynthesisBank' -----
// 
%ignore OverSampledDFTSynthesisBank;
class OverSampledDFTSynthesisBank : public VectorFloatFeatureStream {
 public:
   OverSampledDFTSynthesisBank(VectorComplexFeatureStreamPtr& subband,
			       gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0, 
			       unsigned delayCompensationType = 0, int gainFactor = 1,
			       const String& nm = "OverSampledDFTSynthesisBank");
  
  /* OverSampledDFTSynthesisBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0, unsigned delayCompensationType = 0, const String& nm = "FFTSynthesisBF");*/

  ~OverSampledDFTSynthesisBank();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();
  void inputSourceVector(const gsl_vector_complex* block);
  void doNotUseStreamFeature( bool flag=true );
  double polyphase(unsigned m, unsigned n) const;
};

class OverSampledDFTSynthesisBankPtr : public VectorFloatFeatureStreamPtr {
public:
  %extend {
    OverSampledDFTSynthesisBankPtr(VectorComplexFeatureStreamPtr& samp,
				   gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0, 
				   unsigned delayCompensationType = 0, int gainFactor = 1,
				   const String& nm = "OverSampledDFTSynthesisBank") {
      return new OverSampledDFTSynthesisBankPtr(new OverSampledDFTSynthesisBank(samp, prototype, M, m, r, delayCompensationType, gainFactor, nm));
    }

    OverSampledDFTSynthesisBankPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  OverSampledDFTSynthesisBank* operator->();
};


// ----- definition for class `PerfectReconstructionFFTAnalysisBank' -----
//
%ignore PerfectReconstructionFFTAnalysisBank;
class PerfectReconstructionFFTAnalysisBank : public VectorComplexFeatureStream {
 public:
  PerfectReconstructionFFTAnalysisBank(VectorFloatFeatureStreamPtr& samp,
				       gsl_vector* prototype, unsigned M = 256, unsigned m = 3,
				       const String& nm = "PerfectReconstructionFFTAnalysisBank");
  ~PerfectReconstructionFFTAnalysisBank();

  double polyphase(unsigned m, unsigned n) const;

  virtual const gsl_vector_complex* next(int frameX = -5);

  virtual void reset();

  unsigned fftLen() const;
  unsigned nBlocks() const;
  unsigned subSampRate() const;
};

class PerfectReconstructionFFTAnalysisBankPtr : public VectorComplexFeatureStreamPtr {
 public:
  %extend {
    PerfectReconstructionFFTAnalysisBankPtr(VectorFloatFeatureStreamPtr& samp,
					    gsl_vector* prototype, unsigned M = 256, unsigned m = 3, unsigned r = 0,
					    const String& nm = "PerfectReconstructionFFTAnalysisBankFloat") {
      return new PerfectReconstructionFFTAnalysisBankPtr(new PerfectReconstructionFFTAnalysisBank(samp, prototype, M, m, r, nm));
    }

    PerfectReconstructionFFTAnalysisBankPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  PerfectReconstructionFFTAnalysisBank* operator->();
};


// ----- definition for class `PerfectReconstructionFFTSynthesisBank' -----
// 
%ignore PerfectReconstructionFFTSynthesisBank;
class PerfectReconstructionFFTSynthesisBank : public VectorFloatFeatureStream {
 public:
  PerfectReconstructionFFTSynthesisBank(gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0,
					const String& nm = "PerfectReconstructionFFTSynthesisBF");

  ~PerfectReconstructionFFTSynthesisBank();

  virtual const gsl_vector_float* next(int frameX = -5);

  virtual void reset();

  double polyphase(unsigned m, unsigned n) const;
};

class PerfectReconstructionFFTSynthesisBankPtr : public VectorFloatFeatureStreamPtr {
public:
  %extend {
    PerfectReconstructionFFTSynthesisBankPtr(VectorComplexFeatureStreamPtr& samp,
					     gsl_vector* prototype, unsigned M, unsigned m, unsigned r = 0,
					     const String& nm = "PerfectReconstructionFFTSynthesisBank") {
      return new PerfectReconstructionFFTSynthesisBankPtr(new PerfectReconstructionFFTSynthesisBank(samp, prototype, M, m, r, nm));
    }

    PerfectReconstructionFFTSynthesisBankPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  PerfectReconstructionFFTSynthesisBank* operator->();
};

// ----- definition for class `DelayFeature' -----
//
%ignore DelayFeature;
class DelayFeature : public VectorComplexFeatureStream {
 public:
  DelayFeature( const VectorComplexFeatureStreamPtr& samp, double delayT=0.0, const String& nm = "DelayFeature");
  ~DelayFeature();

  void setDelayTime( double delayT );
  virtual const gsl_vector_complex* next(int frameX = -5);

  virtual void reset();

};

class DelayFeaturePtr : public VectorComplexFeatureStreamPtr {
public:
  %extend {
    DelayFeaturePtr( const VectorComplexFeatureStreamPtr& samp, double delayT=0.0, const String& nm = "DelayFeature" ) {
      return new DelayFeaturePtr(new DelayFeature(samp, delayT, nm));
    }

    DelayFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  DelayFeature* operator->();
};


// ----- definition for class `CosineModulatedPrototypeDesign' -----
// 
class CosineModulatedPrototypeDesign {
 public:
  CosineModulatedPrototypeDesign(int M = 256, int N = 3072, double fs = 1.0);
  ~CosineModulatedPrototypeDesign();

  void fcn(const double* x, double* f);
  void grad(const double* x, double* g);

  int M() const;
  int N() const;
  int m() const;
  int J() const;

  // return (one-half) of the prototype filter
  const gsl_vector* proto();
};

double design_f(const gsl_vector* v, void* params);
void   design_df(const gsl_vector* v, void* params, gsl_vector* df);
void   design_fdf(const gsl_vector* v, void* params, double* f, gsl_vector* df);

void writeGSLFormat(const String& fileName, const gsl_vector* prototype);


// ----- definition for class `AnalysisOversampledDFTDesign' -----
//
%ignore AnalysisOversampledDFTDesign;
class AnalysisOversampledDFTDesign {
public:
  AnalysisOversampledDFTDesign(unsigned M = 512, unsigned m = 2, unsigned r = 1, unsigned wp = 512, int tau_h = -1 );
  ~AnalysisOversampledDFTDesign();

  // design prototype
  const gsl_vector* design(double tolerance = 1.0E-07);

  // calculate distortion measures
  const gsl_vector* calcError(bool doPrint = true);

  void save(const String& fileName);
};

class AnalysisOversampledDFTDesignPtr {
public:
  %extend {
    AnalysisOversampledDFTDesignPtr(unsigned M = 512, unsigned m = 2, unsigned r = 1, unsigned wp = 512, int tau_h = -1 ) {
      return new AnalysisOversampledDFTDesignPtr(new AnalysisOversampledDFTDesign(M, m, r, wp, tau_h ));
    }
  }

  AnalysisOversampledDFTDesign* operator->();
};


// ----- definition for class `SynthesisOversampledDFTDesign' -----
//
%ignore SynthesisOversampledDFTDesign;
class SynthesisOversampledDFTDesign {
public:
  SynthesisOversampledDFTDesign(const gsl_vector* h, unsigned M = 512, unsigned m = 2, unsigned r = 1,
				double v = 0.01, unsigned wp = 512, int tau_T = -1 );
  ~SynthesisOversampledDFTDesign();

  // design prototype
  const gsl_vector* design(double tolerance = 1.0E-07);

  // calculate distortion measures
  const gsl_vector* calcError(bool doPrint = true);

  void save(const String& fileName);
};

class SynthesisOversampledDFTDesignPtr {
public:
  %extend {
    SynthesisOversampledDFTDesignPtr(const gsl_vector* h, unsigned M = 512, unsigned m = 2, unsigned r = 1,
				     double v = 0.01, unsigned wp = 512, int tau_T = -1 ) {
      return new SynthesisOversampledDFTDesignPtr(new SynthesisOversampledDFTDesign(h, M, m, r, v, wp, tau_T));
    }
  }

  SynthesisOversampledDFTDesign* operator->();
};


// ----- definition for class `AnalysisNyquistMDesign' -----
//
%ignore AnalysisNyquistMDesign;
class AnalysisNyquistMDesign : public AnalysisOversampledDFTDesign {
public:
  AnalysisNyquistMDesign(unsigned M = 512, unsigned m = 2, unsigned r = 1, double wp = 1.0);
  ~AnalysisNyquistMDesign();
};

class AnalysisNyquistMDesignPtr : public AnalysisOversampledDFTDesignPtr{
public:
  %extend {
    AnalysisNyquistMDesignPtr(unsigned M = 512, unsigned m = 2, unsigned r = 1, double wp = 1.0) {
      return new AnalysisNyquistMDesignPtr(new AnalysisNyquistMDesign(M, m, r, wp));
    }
  }

  AnalysisNyquistMDesign* operator->();
};


// ----- definition for class `SynthesisNyquistMDesign' -----
//
%ignore SynthesisNyquistMDesign;
class SynthesisNyquistMDesign : public SynthesisOversampledDFTDesign {
public:
  SynthesisNyquistMDesign(const gsl_vector* h, unsigned M = 512, unsigned m = 2, unsigned r = 1,
			  double wp = 1.0);
  ~SynthesisNyquistMDesign();
};

class SynthesisNyquistMDesignPtr : public SynthesisOversampledDFTDesignPtr {
public:
  %extend {
    SynthesisNyquistMDesignPtr(const gsl_vector* h, unsigned M = 512, unsigned m = 2, unsigned r = 1,
			       double wp = 1.0) {
      return new SynthesisNyquistMDesignPtr(new SynthesisNyquistMDesign(h, M, m, r, wp));
    }
  }

  SynthesisNyquistMDesign* operator->();
};


// ----- definition for class `SynthesisNyquistMDesignCompositeResponse' -----
//
%ignore SynthesisNyquistMDesignCompositeResponse;
class SynthesisNyquistMDesignCompositeResponse : public SynthesisNyquistMDesign {
public:
  SynthesisNyquistMDesignCompositeResponse(const gsl_vector* h, unsigned M = 512, unsigned m = 2, unsigned r = 1,
					   double wp = 1.0);
  ~SynthesisNyquistMDesignCompositeResponse();
};

class SynthesisNyquistMDesignCompositeResponsePtr : public SynthesisNyquistMDesignPtr {
public:
  %extend {
    SynthesisNyquistMDesignCompositeResponsePtr(const gsl_vector* h, unsigned M = 512, unsigned m = 2, unsigned r = 1,
						double wp = 1.0) {
      return new SynthesisNyquistMDesignCompositeResponsePtr(new SynthesisNyquistMDesignCompositeResponse(h, M, m, r, wp));
    }
  }

  SynthesisNyquistMDesignCompositeResponse* operator->();
};

gsl_vector* getWindow( unsigned winType, unsigned winLen );
void writeGSLFormat(const String& fileName, const gsl_vector* prototype);

%rename(__str__) print;
%ignore *::print();
