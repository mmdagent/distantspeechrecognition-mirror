//
//                       Beamforming Toolkit
//                              (btk)
//
//  Module:  btk.modulated
//  Purpose: simple PR analysis and synthesis filterbank
//  Author:  Uwe Mayer and Andrej Schkarbanenko
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


#ifndef _SIMPLEFILTERBANK_H_
#define _SIMPLEFILTERBANK_H_


#include "stream/stream.h"

#include <assert.h>
//#include <math.h>

#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>

#include <fftw3.h>

#include "utils.h"


// ----- definition for class `FilterbankBlackboard` ---------------------------

class FilterbankBlackboard
{
public:
  FilterbankBlackboard(gsl_vector* prototype, int M, int m, int windowLength);
  ~FilterbankBlackboard();
  
  const unsigned M;		// number of frequency bands
  const unsigned m;		// length of polyphase component
  const unsigned M2;		// 2*M
  const unsigned m2;		// 2*m
  const unsigned N;		// constant: 2Mm-1, the length of the prototype
  const unsigned K;		// constant: 2M(m-1), total delay of the system
  const unsigned windowLength;	// length of each input vector
  const unsigned xwidth;	// width of input signal matrix
  const unsigned filtxwidth;	// width of input matrix filtered with polyphase
  const unsigned filtywidth;	// width of output matrix filtered with polyphase
  const gsl_vector* prototype;	// prototype
  const gsl_complex W;		// constant: e^(-i*2*pi/M2), 2M-th root of 1
  const gsl_complex FFTnormalise; // normalisation factor for (I)FFT

  gsl_matrix *G_a;		// polyphase matrix of analysis bank
  gsl_matrix *G_s;		// polyphase matrix of synthesis bank
};

typedef  refcount_ptr<FilterbankBlackboard> FilterbankBlackboardPtr;



// ----- definition for class `SimpleAnalysisbank' -----------------------------

class SimpleAnalysisbank : 
public VectorComplexFeatureStream {

public:
  SimpleAnalysisbank(VectorFeatureStreamPtr& samples, 
		     FilterbankBlackboardPtr& bb);
  ~SimpleAnalysisbank();

  virtual const gsl_vector_complex* next(int frameX = -5);
  virtual void reset();


private:
  const VectorFeatureStreamPtr samples; // provides input of samples
  const FilterbankBlackboardPtr bb; // defines constants

  int bufferIndex;		// index in result matrix S
  gsl_matrix* X;		// input samples are embedded in here
  gsl_matrix_complex* S;	// data after filtering with polyphase filters

  fftw_plan plan;		// plan for 2d IFFT
};

typedef Inherit<SimpleAnalysisbank, VectorComplexFeatureStreamPtr> SimpleAnalysisbankPtr;



// ----- definition for class `SimpleSynthesisbank' ----------------------------

class SimpleSynthesisbank : 
public VectorFeatureStream {

public:
  SimpleSynthesisbank(VectorComplexFeatureStreamPtr& subband,
		      FilterbankBlackboardPtr& bb);
  ~SimpleSynthesisbank();

  virtual const gsl_vector* next(int frameX = -5);
  virtual void reset();

  
private:
  const VectorComplexFeatureStreamPtr subband; // provides input of subband bins
  const FilterbankBlackboardPtr bb; // defines constants

  gsl_matrix_complex* S;	// data after applying DFT matrix
  gsl_matrix_complex* Y;	// resulting data after filtering and modulation

  fftw_plan plan;		// plan for 2d FFT
};

typedef Inherit<SimpleSynthesisbank, VectorFeatureStreamPtr> SimpleSynthesisbankPtr;



#endif

