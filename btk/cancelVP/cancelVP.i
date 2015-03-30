//                              -*- C++ -*-
//
//                          Beamforming Toolkit
//                                 (btk)
//
//  Module:  btk.cancelVP
//  Purpose: Cancelation of a voice prompt based on either NLMS or Kalman
//	     filter algorithms.
//  Author:  John McDonough ,Wei Chu and Kenichi Kumatani
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


%module(package="btk") cancelVP

%{
#include "stream/stream.h"
#include "stream/pyStream.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include "squareRoot/squareRoot.h"
#include "cancelVP/cancelVP.h"
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
oldimport = """
%}
%import stream/stream.i
%pythoncode %{
"""
%}


// ----- definition for class `NLMSAcousticEchoCancellationFeature' -----
// 
%ignore NLMSAcousticEchoCancellationFeature;
class NLMSAcousticEchoCancellationFeature : public VectorComplexFeatureStream {
public:
  NLMSAcousticEchoCancellationFeature(const VectorComplexFeatureStreamPtr& original, const VectorComplexFeatureStreamPtr& distorted,
				  double delta = 100.0, double epsilon = 1.0E-04, double threshold = 100.0, const String& nm = "AEC");

  const gsl_vector_complex* next(int frameX = -5) const;

};

class NLMSAcousticEchoCancellationFeaturePtr : public VectorComplexFeatureStreamPtr {
 public:
  %extend {
    NLMSAcousticEchoCancellationFeaturePtr(const VectorComplexFeatureStreamPtr& original, const VectorComplexFeatureStreamPtr& distorted,
					   double delta = 100.0, double epsilon = 1.0E-04, double threshold = 100.0, const String& nm = "AEC") {
      return new NLMSAcousticEchoCancellationFeaturePtr(new NLMSAcousticEchoCancellationFeature(original, distorted, delta, epsilon, threshold, nm));
    }

    NLMSAcousticEchoCancellationFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  NLMSAcousticEchoCancellationFeature* operator->();
};


// ----- definition for class `KalmanFilterEchoCancellationFeature' -----
// 
%ignore KalmanFilterEchoCancellationFeature;
class KalmanFilterEchoCancellationFeature : public VectorComplexFeatureStream {
public:
  KalmanFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded, 
				      double beta = 0.95, double sigmau2 = 10e-4, double sigma2 = 5.0, double threshold = 100.0, const String& nm = "KFEchoCanceller");

  const gsl_vector_complex* next(int frameX = -5) const;
};

class KalmanFilterEchoCancellationFeaturePtr : public VectorComplexFeatureStreamPtr {
 public:
  %extend {
    KalmanFilterEchoCancellationFeaturePtr(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
					   double beta = 0.95, double sigmau2 = 10e-4, double sigma2 = 5.0, double threshold = 100.0, double crossCorrTh = 0.5,
					   const String& nm = "KFEchoCanceller")
    {
      return new KalmanFilterEchoCancellationFeaturePtr(new KalmanFilterEchoCancellationFeature(played, recorded, beta, sigma2, threshold, nm));
    }

    KalmanFilterEchoCancellationFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  KalmanFilterEchoCancellationFeature* operator->();
};


// ----- definition for class `BlockKalmanFilterEchoCancellationFeature' -----
//
%ignore BlockKalmanFilterEchoCancellationFeature;
class BlockKalmanFilterEchoCancellationFeature : public VectorComplexFeatureStream {
public:
  BlockKalmanFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
					   unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmak2 = 5.0, double threshold = 100.0,
					   double amp4play = 1.0, 
					   const String& nm = "BlockKFEchoCanceller");

  const gsl_vector_complex* next(int frameX = -5) const;
};

class BlockKalmanFilterEchoCancellationFeaturePtr : public VectorComplexFeatureStreamPtr {
 public:
  %extend {
    BlockKalmanFilterEchoCancellationFeaturePtr(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
						unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmak2 = 5.0, double threshold = 100.0,
						double amp4play = 1.0, 
						const String& nm = "BlockKFEchoCanceller") {
      return new BlockKalmanFilterEchoCancellationFeaturePtr(new BlockKalmanFilterEchoCancellationFeature(played, recorded, sampleN, beta, sigmau2, sigmak2, threshold, amp4play, nm));
    }

    BlockKalmanFilterEchoCancellationFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  BlockKalmanFilterEchoCancellationFeature* operator->();
};


// ----- definition for class `InformationFilterEchoCancellationFeature' -----
//
%ignore InformationFilterEchoCancellationFeature;
class InformationFilterEchoCancellationFeature : public BlockKalmanFilterEchoCancellationFeature {
public:
  InformationFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
					   unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmak2 = 5.0, double snrTh = 2.0,
					   double engTh = 100.0, double smooth = 0.9, double loading = 1.0e-02,
					   double amp4play = 1.0, 
					   const String& nm = "DTDBlockKFEchoCanceller");

  const gsl_vector_complex* next(int frameX = -5) const;
};

class InformationFilterEchoCancellationFeaturePtr : public BlockKalmanFilterEchoCancellationFeaturePtr {
 public:
  %extend {
    InformationFilterEchoCancellationFeaturePtr(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
						unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmak2 = 5.0, double snrTh = 2.0,
						double engTh = 100.0, double smooth = 0.9, double loading = 1.0e-02,
						double amp4play = 1.0, 
						const String& nm = "DTDBlockKFEchoCanceller") {
      return new InformationFilterEchoCancellationFeaturePtr(new InformationFilterEchoCancellationFeature(played, recorded, sampleN, beta, sigmau2, sigmak2, snrTh, engTh, smooth, loading, amp4play, nm));
    }

    InformationFilterEchoCancellationFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  InformationFilterEchoCancellationFeature* operator->();
};


// ----- definition for class `SquareRootInformationFilterEchoCancellationFeature' -----
//
%ignore SquareRootInformationFilterEchoCancellationFeature;
class SquareRootInformationFilterEchoCancellationFeature : public InformationFilterEchoCancellationFeature {
public:
  SquareRootInformationFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
						     unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmak2 = 5.0, double snrTh = 2.0,
						     double engTh = 100.0, double smooth = 0.9, double loading = 1.0e-02,
                                                     double amp4play = 1.0, 
						     const String& nm = "Square Root Information Filter Echo Cancellation Feature");

  const gsl_vector_complex* next(int frameX = -5) const;
};

class SquareRootInformationFilterEchoCancellationFeaturePtr : public InformationFilterEchoCancellationFeaturePtr {
 public:
  %extend {
    SquareRootInformationFilterEchoCancellationFeaturePtr(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
							  unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmak2 = 5.0, double snrTh = 2.0,
							  double engTh = 100.0, double smooth = 0.9, double loading = 1.0e-02,
							  double amp4play = 1.0, 
							  const String& nm = "Square Root Information Filter Echo Cancellation Feature") {
      return new SquareRootInformationFilterEchoCancellationFeaturePtr(new SquareRootInformationFilterEchoCancellationFeature(played, recorded, sampleN, beta, sigmau2, sigmak2, snrTh, engTh, smooth, loading, amp4play, nm));
    }

    SquareRootInformationFilterEchoCancellationFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  SquareRootInformationFilterEchoCancellationFeature* operator->();
};


// ----- definition for class `DTDBlockKalmanFilterEchoCancellationFeature' -----
//
%ignore DTDBlockKalmanFilterEchoCancellationFeature;
class DTDBlockKalmanFilterEchoCancellationFeature : public BlockKalmanFilterEchoCancellationFeature {
public:
  DTDBlockKalmanFilterEchoCancellationFeature(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
					      unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmak2 = 5.0, double snrTh = 2.0,
					      double engTh = 100.0, double smooth = 0.9,
					      double amp4play = 1.0, 
					      const String& nm = "DTDBlockKFEchoCanceller");

  const gsl_vector_complex* next(int frameX = -5) const;
};

class DTDBlockKalmanFilterEchoCancellationFeaturePtr : public BlockKalmanFilterEchoCancellationFeaturePtr {
 public:
  %extend {
    DTDBlockKalmanFilterEchoCancellationFeaturePtr(const VectorComplexFeatureStreamPtr& played, const VectorComplexFeatureStreamPtr& recorded,
						   unsigned sampleN = 1, double beta = 0.95, double sigmau2 = 10e-4, double sigmak2 = 5.0, double snrTh = 2.0,
						   double engTh = 100.0, double smooth = 0.9,
						   double amp4play = 1.0, 
						   const String& nm = "DTDBlockKFEchoCanceller") {
      return new DTDBlockKalmanFilterEchoCancellationFeaturePtr(new DTDBlockKalmanFilterEchoCancellationFeature(played, recorded, sampleN, beta, sigmau2, sigmak2, snrTh, engTh, smooth, amp4play, nm));
    }

    DTDBlockKalmanFilterEchoCancellationFeaturePtr __iter__() {
      (*self)->reset();  return *self;
    }
  }

  DTDBlockKalmanFilterEchoCancellationFeature* operator->();
};

