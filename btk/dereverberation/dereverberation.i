//                              -*- C++ -*-
//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.dereverberation
//  Purpose: Single- and multi-channel dereverberation base on linear
//	     prediction in the subband domain.
//  Author:  John McDonough.

%module(package="btk") dereverberation

%{
#include "stream/stream.h"
#include "stream/pyStream.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "dereverberation/dereverberation.h"
%}

%init {
  // NumPy needs to set up callback functions
  import_array();
}

typedef int size_t;

%include gsl/gsl_types.h
%include jexception.i
%include complex.i
%include matrix.i
%include vector.i

%rename(__str__) *::print;
%rename(__getitem__) *::operator[](int);
%ignore *::nextSampleBlock(const double* smp);

%pythoncode %{
import btk
from btk import stream
from btk import feature
oldimport = """
%}
%import stream/stream.i
%pythoncode %{
"""
%}


// ----- definition for class `SingleChannelWPEDereverberationFeature' -----
// 
%ignore SingleChannelWPEDereverberationFeature;
class SingleChannelWPEDereverberationFeature : public VectorComplexFeatureStream {
 public:
  SingleChannelWPEDereverberationFeature(VectorComplexFeatureStreamPtr& samples, unsigned lowerN, unsigned upperN, unsigned iterationsN = 2, double loadDb = -20.0, double bandWidth = 0.0, double sampleRate = 16000.0,
					 const String& nm = "SingleChannelWPEDereverberationFeature");

  ~SingleChannelWPEDereverberationFeature();

  virtual const gsl_vector_complex* next(int frameX = -5);

  virtual void reset();

  void nextSpeaker();
};

class SingleChannelWPEDereverberationFeaturePtr : public VectorComplexFeatureStreamPtr {
public:
  %extend {
    SingleChannelWPEDereverberationFeaturePtr(VectorComplexFeatureStreamPtr& samples, unsigned lowerN, unsigned upperN, unsigned iterationsN = 2, double loadDb = -20.0, double bandWidth = 0.0, double sampleRate = 16000.0,
					      const String& nm = "SingleChannelWPEDereverberationFeature") {
      return new SingleChannelWPEDereverberationFeaturePtr(new SingleChannelWPEDereverberationFeature(samples, lowerN, upperN, iterationsN, loadDb, bandWidth, sampleRate, nm));
    }

    SingleChannelWPEDereverberationFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SingleChannelWPEDereverberationFeature* operator->();
};


// ----- definition for class `MultiChannelWPEDereverberation' -----
// 
%ignore MultiChannelWPEDereverberation;
class MultiChannelWPEDereverberation {
public:
  MultiChannelWPEDereverberation(unsigned subbandsN, unsigned channelsN, unsigned lowerN, unsigned upperN, unsigned iterationsN = 2, double loadDb = -20.0, double bandWidth = 0.0,  double sampleRate = 16000.0);

  ~MultiChannelWPEDereverberation();

  void reset();

  unsigned size() const { return _subbandsN; }

  void setInput(VectorComplexFeatureStreamPtr& samples);

  void nextSpeaker();

  const gsl_vector_complex* getOutput(unsigned channelX);
};

class MultiChannelWPEDereverberationPtr {
public:
  %extend {
    MultiChannelWPEDereverberationPtr(unsigned subbandsN, unsigned channelsN, unsigned lowerN, unsigned upperN, unsigned iterationsN = 2, double loadDb = -20.0, double bandWidth = 0.0,  double sampleRate = 16000.0) {
      return new MultiChannelWPEDereverberationPtr(new MultiChannelWPEDereverberation(subbandsN, channelsN, lowerN, upperN, iterationsN, loadDb, bandWidth, sampleRate));
    }

    MultiChannelWPEDereverberationPtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  MultiChannelWPEDereverberation* operator->();
};


// ----- definition for class `MultiChannelWPEDereverberationFeature' -----
// 
%ignore MultiChannelWPEDereverberationFeature;
class MultiChannelWPEDereverberationFeature : public VectorComplexFeatureStream {
 public:
  MultiChannelWPEDereverberationFeature(MultiChannelWPEDereverberationPtr& source, unsigned channelX, const String& nm = "MultiChannelWPEDereverberationFeature");

  ~MultiChannelWPEDereverberationFeature();

  virtual const gsl_vector_complex* next(int frameX = -5);

  virtual void reset();
};

class MultiChannelWPEDereverberationFeaturePtr : public VectorComplexFeatureStreamPtr {
public:
  %extend {
    MultiChannelWPEDereverberationFeaturePtr(MultiChannelWPEDereverberationPtr& source, unsigned channelX, const String& nm = "MultiChannelWPEDereverberationFeature") {
      return new MultiChannelWPEDereverberationFeaturePtr(new MultiChannelWPEDereverberationFeature(source, channelX, nm));
    }

    MultiChannelWPEDereverberationFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  MultiChannelWPEDereverberationFeature* operator->();
};

%rename(__str__) print;
%ignore *::print();
