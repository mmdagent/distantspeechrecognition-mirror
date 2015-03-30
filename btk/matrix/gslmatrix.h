#ifndef _gslmatrix_h_
#define _gslmatrix_h_

#include <gsl/gsl_matrix.h>

gsl_matrix_float* gsl_matrix_float_resize(gsl_matrix_float* m, size_t size1, size_t size2);

void gsl_matrix_float_set_cosine(gsl_matrix_float* m, size_t i, size_t j, int type);

gsl_matrix_float* gsl_matrix_float_load(gsl_matrix_float* m, const char* filename, bool old = false);

gsl_vector_float* gsl_vector_float_load(gsl_vector_float* m, const char* filename, bool old = false);

#endif
