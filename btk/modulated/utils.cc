#include "utils.h"


//--
// convert gsl_vector to double list
//--
double* dvector2darray(gsl_vector const * const vector){
  double* dest = new double[vector->size]; // create new double on heap
  memcpy(dest, vector->data, sizeof(double)*vector->size);
  return dest;
}




gsl_vector* filter_FIR(gsl_vector* b, gsl_vector* x){
  gsl_vector* y = gsl_vector_calloc(b->size +x->size -1); // resulting vector

  double y_n, b_k, x_nk;
  for (unsigned int n=0; n < y->size; n++){
    y_n = 0;

    for (unsigned int k=0; k < b->size; k++){
      b_k = b->data[k];
      x_nk = (n-k < 0) || (n-k >= x->size)? 0: x->data[n-k];

      y_n += b_k*x_nk;		// y[n] += b[k]*x[n-k]
    }

    y->data[n] = y_n;
  }

  return y;
}


gsl_vector_complex* filter_FIR(gsl_vector* b, gsl_vector_complex* x){
  gsl_vector_complex* y = gsl_vector_complex_calloc(b->size +x->size -1); // resulting vector

  double b_k;
  gsl_complex y_n, x_nk;
  for (unsigned int n=0; n < y->size; n++){
    y_n = gsl_complex_rect(0,0); // initialise to 0

    for (unsigned int k=0; k < b->size; k++){
      b_k = b->data[k];
      x_nk = (n-k < 0) || (n-k >= x->size)? gsl_complex_rect(0,0): gsl_vector_complex_get(x, n-k);

      y_n = gsl_complex_add(y_n, gsl_complex_mul_real(x_nk, b_k)); // y[n] += b[k]*x[n-k]
    }

    gsl_vector_complex_set(y, n, y_n);
  }

  return y;
}


gsl_vector_complex* filter_FIR(gsl_vector_complex* b, gsl_vector_complex* x){
  gsl_vector_complex* y = gsl_vector_complex_calloc(b->size +x->size -1); // resulting vector

  gsl_complex y_n, b_k, x_nk;
  for (unsigned int n=0; n < y->size; n++){
    y_n = gsl_complex_rect(0,0); // initialise to 0

    for (unsigned int k=0; k < b->size; k++){
      b_k = gsl_vector_complex_get(b,k);
      x_nk = (n-k < 0) || (n-k >= x->size)? gsl_complex_rect(0,0): gsl_vector_complex_get(x, n-k);

      y_n = gsl_complex_add(y_n, gsl_complex_mul(b_k, x_nk)); // y[n] += b[k]*x[n-k]
    }

    gsl_vector_complex_set(y, n, y_n);
  }

  return y;
}



gsl_vector_complex* gsl_vector_mul_complex(gsl_vector const *const x, gsl_complex const W_k){
  gsl_vector_complex* result = gsl_vector_complex_alloc(x->size);
  gsl_complex product;

  for (unsigned int i=0; i < result->size; i++){
    product = gsl_complex_mul_real(W_k, x->data[i]);
    gsl_vector_complex_set(result, i, product);
  }    

  return result;
}



void gsl_vector_add(gsl_vector_complex* a, gsl_vector_complex const* const b){
  assert(a->size == b->size);
  gsl_complex *a_ptr = (gsl_complex*)a->data;
  gsl_complex *b_ptr = (gsl_complex*)b->data;
  for (unsigned int i=0; i < a->size; i++){
    gsl_vector_complex_set(a, i, gsl_complex_add(a_ptr[i], b_ptr[i]) );
  }
}



gsl_vector_complex* gsl_vector_short2complex(gsl_vector_short const * const a){
  gsl_vector_complex* copy = gsl_vector_complex_calloc(a->size);	// init to 0
  for (unsigned int i = 0; i < a->size; i++)
    ((double*)copy->data)[i*2] = a->data[i];
  return copy;
}



void gsl_vector_complex_REAL(gsl_vector* dest, gsl_vector_complex const * const src){
  assert(dest->size == src->size); // same length
  for (unsigned i=0; i < dest->size; i++)
    dest->data[i] = GSL_REAL( ((gsl_complex*)(src->data))[i] );
}
