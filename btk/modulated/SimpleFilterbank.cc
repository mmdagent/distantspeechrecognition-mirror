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


#include "SimpleFilterbank.h"


/***************************************************/
/* methods for class `FilterbankBlackboard'	   */
/***************************************************/

//-- constructor ---------------------------------------------------------------
// constructs a data structure which contains initialised parameters needed
// by each pair of analysisbank /synthesisbank
//
// Param: prototype     gsl vector of the prototype filter
//        M             number of frequency bands
//        m             multiple of the length of the prototype
//        windowLength  specifies the number of samples that are processed
//                      in each step
FilterbankBlackboard::
FilterbankBlackboard(gsl_vector* prototype, int M, int m, int windowLength) :
  M(M), m(m), M2(2*M), m2(m*2), N(M2*m), K(M2*(m-1)), 
  windowLength(windowLength), 
  xwidth(1+(unsigned)ceil((double)windowLength/M)),
  filtxwidth(xwidth +m2-1 -1),	// polyphase matrix: 2Mx(2m-1), delay=a+b-1
  filtywidth(filtxwidth +m2-1 -1), // same as above for synthesis fb
  prototype(gsl_vector_calloc(prototype->size)), 
  W(gsl_complex_polar(1, -2*M_PI/M2)), // e^(i* -2*pi/M2)
  FFTnormalise(gsl_complex_rect(sqrt((double)1/(M2*filtxwidth)), 0))
{
  //-- create polyphase matricies G_a, G_s
  G_a = gsl_matrix_calloc(M2, m2-1); // M2 x m2-1; init to zero
  G_s = gsl_matrix_calloc(M2, m2-1); // M2 x m2-1; init to zero
  
  int sign = -1;	// init sign, so that first column is positive
  for (int i=0; i < N; i++){
    int row = i %M2;
    int col = 2*(i /M2);	// 0, 2, 4, 6, 8, ...
    if (i %M2 == 0) sign = -sign; // swap sign every 2M 

    // fill from first column onwards
    gsl_matrix_set(G_a, row, col, sign*prototype->data[i]);
    // fill from last column backwards
    gsl_matrix_set(G_s, row, (m2-1)-1-col, sign*prototype->data[i]);
  }
}


FilterbankBlackboard::~FilterbankBlackboard()
{
  gsl_vector_free(const_cast<gsl_vector*>(prototype));

  gsl_matrix_free(G_a);
  gsl_matrix_free(G_s);
}




/*************************************************************/
/* methods for class `SimpleAnalysisbank'              	     */
/*************************************************************/


//-- Constructor ---------------------------------------------------------------
//SimpleAnalysisbank(VectorShortFeatureStreamPtr& samples,
SimpleAnalysisbank::
SimpleAnalysisbank(VectorFeatureStreamPtr& samples,
		   FilterbankBlackboardPtr& bb):
  VectorComplexFeatureStream(bb->M2, "SimpleAnalysisbank"), 
  samples(samples), bb(bb)
{ 
  // signal is initially embedded here
  X = gsl_matrix_alloc(bb->M2, bb->xwidth);
  // takes up signal after filtering with polyphase matrix
  S = gsl_matrix_complex_alloc(bb->M2, bb->filtxwidth);

  // create IFFT plan for later execution
  plan = fftw_plan_dft_2d(/* rows= */bb->M2, /* cols= */bb->filtxwidth,
			  /* in= */reinterpret_cast<fftw_complex*>(S->data), 
			  /* out= */reinterpret_cast<fftw_complex*>(S->data),
			  /* sign= */FFTW_BACKWARD, /* flags= */FFTW_PATIENT);

  this->reset();
}



//-- Destructor ----------------------------------------------------------------
SimpleAnalysisbank::~SimpleAnalysisbank() { 
  gsl_matrix_free(X);
  gsl_matrix_complex_free(S);

  fftw_destroy_plan(plan);
}



//-- main routine --------------------------------------------------------------
const gsl_vector_complex* SimpleAnalysisbank::next(int frameX) {
  // adhere to (strange) iterator protocol
  if (frameX == VectorComplexFeatureStream::_frameX) 
    return VectorComplexFeatureStream::_vector;

  // initially:
  // if index into S matrix == 0 then process new input vector
  if (bufferIndex == 0){
    //-- get input signal as vector
    const gsl_vector *x_vector = samples->next();

    //-- initialise input matrix
    // need to set zero, because not all elements are touched
    gsl_matrix_set_zero(X);
    // S needs no initialisation, because each element is set explicitly
  
    //-- read samples into matrix
    for (int i=0; i < bb->windowLength; i++){
      unsigned row = (bb->M -1) -(i %bb->M); // row index in upper half of matrix
      unsigned col = i /bb->M;	// col index in upper half of matrix

      // set element in upper half of matrix
      gsl_matrix_set(X, row, col, x_vector->data[i]);
      // set element in lower half of matrix
      gsl_matrix_set(X, row +bb->M, col +1, x_vector->data[i]);
    }

    //-- filter input matrix with polyphase filter and modulate it
    for (int k=0; k < bb->M2; k++){
      gsl_vector_view G_a_k = gsl_matrix_row(bb->G_a, k); // get row k of G_a matrix
      gsl_vector_view X_k   = gsl_matrix_row(X, k); // row k of X input matrix

      // filter row k with polyphase filter
      gsl_vector *G_aOut_k = filter_FIR((gsl_vector*)&G_a_k, (gsl_vector*)&X_k);
      assert(G_aOut_k->size == bb->filtxwidth);
    
      // modulate row with W^(-0.5*k)
      gsl_complex W_k = gsl_complex_pow_real(bb->W, -0.5*k);
      for (int i=0; i < bb->filtxwidth; i++)
	gsl_matrix_complex_set(S, k, i, gsl_complex_mul_real(W_k, G_aOut_k->data[i]));
    
      gsl_vector_free(G_aOut_k);
    }

    // do IFFT
    fftw_execute(plan);
    // normalize result matrix 
    gsl_matrix_complex_scale(S, bb->FFTnormalise);
  } 

  // copy next vector to output structure
  gsl_matrix_complex_get_col(VectorComplexFeatureStream::_vector, S, bufferIndex);
  bufferIndex = (bufferIndex+1) %bb->filtxwidth; // increase column index with respect to width

  // return result vector 
  VectorComplexFeatureStream::_increment();
  return VectorComplexFeatureStream::_vector;			    
}


void SimpleAnalysisbank::reset()
{
  VectorComplexFeatureStream::reset(); 
  samples->reset();
  bufferIndex = 0;
}



/*********************************************************************/
/* methods for class `SimpleSynthesisbank'	                     */
/*********************************************************************/

//-- constructor ---------------------------------------------------------------
SimpleSynthesisbank::
SimpleSynthesisbank(VectorComplexFeatureStreamPtr& subband, 
		    FilterbankBlackboardPtr& bb): 
  VectorFeatureStream(bb->windowLength, "SimpleSynthesisbank"),
  subband(subband), bb(bb)
{
  // takes result of DFT computation
  S = gsl_matrix_complex_alloc(bb->M2, bb->filtxwidth);
  // takes result after filtering with polyphase and modulation
  Y = gsl_matrix_complex_alloc(bb->M2, bb->filtywidth);

  // create plan for inverse FFT
  plan = fftw_plan_dft_2d(/* rows= */bb->M2, /* cols= */bb->filtxwidth,
			  /* in= */reinterpret_cast<fftw_complex*>(S->data), 
			  /* out= */reinterpret_cast<fftw_complex*>(S->data),
			  /* sign= */FFTW_FORWARD, /* flags= */FFTW_PATIENT);

  this->reset();
}


//-- destructor ----------------------------------------------------------------
SimpleSynthesisbank::~SimpleSynthesisbank() {
  gsl_matrix_complex_free(S);
  gsl_matrix_complex_free(Y);

  fftw_destroy_plan(plan);
}


//-- main routine --------------------------------------------------------------
const gsl_vector* SimpleSynthesisbank::next(int frameX){
  // adhere to (strange) iterator protocol
  if (frameX == VectorFeatureStream::_frameX) 
    return VectorFeatureStream::_vector;

  // fill whole buffer with subband snapshots
  for (int i=0; i < bb->filtxwidth; i++){
    const gsl_vector_complex *band = subband->next();
    gsl_matrix_complex_set_col(S, i, band);
  }

  // normalize result matrix
  gsl_matrix_complex_scale(S, bb->FFTnormalise);
  // do FFT 
  fftw_execute(plan);
  
  //-- filter signal with polyphase filter and modulate it
  for (int k=0; k < bb->M2; k++){
    gsl_vector_view G_s_k       = gsl_matrix_row(bb->G_s, k); // get row k of G_s matrix
    gsl_vector_complex_view S_k = gsl_matrix_complex_row(S, k);	// row k of S matrix

    // filter row k with polyphase filter
    gsl_vector_complex* G_sOut_k = filter_FIR((gsl_vector*)&G_s_k, (gsl_vector_complex*)&S_k);
    assert(G_sOut_k->size == bb->filtywidth);

    // modulate row with W^(0.5*k)
    gsl_complex W_k = gsl_complex_pow_real(bb->W, 0.5*k);
    for (int i=0; i < bb->filtywidth; i++) {
      gsl_complex G_sOut_k_i = gsl_vector_complex_get(G_sOut_k, i);
      gsl_matrix_complex_set(Y, k, i, gsl_complex_mul(G_sOut_k_i, W_k));
    }

    gsl_vector_complex_free(G_sOut_k);
  }

  //-- add up result
  gsl_vector_set_zero(VectorFeatureStream::_vector);
  for (int i=bb->K; i < bb->windowLength +bb->K; i++){
    int row = (bb->M -1) -(i %bb->M); // row index in upper portion of result matrix
    int col = i /bb->M;		// col index in upper portion of result matrix

    gsl_complex sum = gsl_complex_add(gsl_matrix_complex_get(Y, row, col),
				      gsl_matrix_complex_get(Y, row +bb->M, col +1));
    VectorFeatureStream::_vector->data[i -bb->K] = GSL_REAL(sum);
  }

  // return result
  VectorFeatureStream::_increment();
  return VectorFeatureStream::_vector;
}



void SimpleSynthesisbank::reset()
{
  VectorFeatureStream::reset(); 
  subband->reset();  
}


