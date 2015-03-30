//
//                           Speech Front End
//                                 (sfe)
//
//  Module:  sfe.feature
//  Purpose: Representation of Python feature streams.
//  Author:  John McDonough.
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


#include "stream/pyStream.h"
#include <numpy/arrayobject.h>
#include "common/jexception.h"


// template<>
// void PyFeatureStream<gsl_vector_char, char, PyArray_SBYTE>::_gsl_vector_set(gsl_vector_char *vector, int index, char value) {
//   gsl_vector_char_set(vector, index, value);
// };

// template<>
// void PyFeatureStream<gsl_vector_short, short, PyArray_SHORT>::_gsl_vector_set(gsl_vector_short *vector, int index, short value) {
//   gsl_vector_short_set(vector, index, value);
// };

// template<> void PyFeatureStream<gsl_vector_float, float, PyArray_FLOAT>::_gsl_vector_set(gsl_vector_float *vector, int index, float value) {
//   gsl_vector_float_set(vector, index, value);
// };

// template<> void PyFeatureStream<gsl_vector, double, PyArray_DOUBLE>::_gsl_vector_set(gsl_vector *vector, int index, double value) {
//   gsl_vector_set(vector, index, value);
// };

// template<> void PyFeatureStream<gsl_vector_complex, gsl_complex, PyArray_CDOUBLE>::_gsl_vector_set(gsl_vector_complex *vector, int index, gsl_complex value) {
//   gsl_vector_complex_set(vector, index, value);
// };

//template <class c_type, int numpy_type>
//gsl_vector_char* PyFeatureStream<gsl_vector_char, c_type, numpy_type>::gsl_vector_set() {
//}

// template<>
// const gsl_vector_char* PyFeatureStream<gsl_vector_char>::next(int frameX)
// { 
//   if (frameX == _frameX) return _vector;

//   PYTHON_START;
//   PyObject* pyObj = PyObject_CallMethod(_iter, "next", NULL);
//   if (PyObject *e = PyErr_Occurred()) {
//     if (PyErr_GivenExceptionMatches(e, PyExc_StopIteration)) {
//       PyErr_Clear();
//       PYTHON_END;
//       throw jiterator_error("No more samples!");
//     } else {
//       PyObject *etype;
//       PyObject *evalue;
//       PyObject *etrace;
//       PyErr_Fetch(&etype, &evalue, &etrace);      
//       PyErr_Clear();
//       PYTHON_END;
//       Py_XDECREF(pyObj);
//       throw jpython_error(etype, evalue, etrace);
//     }
//   }
//   if (pyObj == NULL) {
//     PYTHON_END;
//     throw jiterator_error("No more samples!");
//   }
// #if 0
//   PyArrayObject* pyVec = (PyArrayObject*)
//     PyArray_ContiguousFromObject(pyObj, PyArray_CHAR, 1, 1);
// #else
//   PyArrayObject* pyVec = (PyArrayObject*) pyObj;
// #endif

//   char* comp = (char*) pyVec->data;

//   // check for zero samples
//   bool allZero = true;
//   for (unsigned i = 0; i < _size; i++)  
//     if (comp[i] != 0.0) { allZero = false;  break; }

// //   if (allZero) {
// //     PYTHON_END;
// //     throw jconsistency_error("All samples are zero!");
// //   }

//   for (unsigned i = 0; i < _size; i++)
//     gsl_vector_char_set(_vector, i, comp[i]);

//   Py_XDECREF(pyObj);
//   Py_XDECREF(pyVec);
//   PYTHON_END;

//   _increment();

//   return _vector;
// }

// ----- partial specializations for class template 'PyFeatureStream' -----
//
// template<>
// const gsl_vector_short* PyFeatureStream<gsl_vector_short>::next(int frameX)
// { 
//   if (frameX == _frameX) return _vector;

//   PYTHON_START;
//   PyObject* pyObj = PyObject_CallMethod(_iter, "next", NULL);
//   if (PyObject *e = PyErr_Occurred()) {
//     if (PyErr_GivenExceptionMatches(e, PyExc_StopIteration)) {
//       PYTHON_END;
//       throw jiterator_error("No more samples!");
//     } else {
//       PyObject *etype;
//       PyObject *evalue;
//       PyObject *etrace;
//       PyErr_Fetch(&etype, &evalue, &etrace);      
//       PYTHON_END;
//       throw jpython_error(etype, evalue, etrace);
//     }
//   }
//   if (pyObj == NULL) {
//     PYTHON_END;
//     throw jiterator_error("No more samples!");
//   }
// #if 0
//   PyArrayObject* pyVec = (PyArrayObject*);
//   PyArray_ContiguousFromObject(pyObj, PyArray_SHORT, 1, 1);
// #else
//   PyArrayObject* pyVec = (PyArrayObject*) pyObj;
// #endif

//   short* comp = (short*) pyVec->data;

//   // check for zero samples
//   bool allZero = true;
//   for (unsigned i = 0; i < _size; i++)  
//     if (comp[i] != 0.0) { allZero = false;  break; }

// //   if (allZero) {
// //     PYTHON_END;
// //     throw jconsistency_error("All samples are zero!");
// //   }

//   for (unsigned i = 0; i < _size; i++)
//     gsl_vector_short_set(_vector, i, comp[i]);

//   Py_DECREF(pyVec);
//   PYTHON_END;

//   _increment();

//   return _vector;
// }

// template<>
// const gsl_vector_float* PyFeatureStream<gsl_vector_float>::next(int frameX)
// {
//   if (frameX == _frameX) return _vector;
  
//   PYTHON_START;
//   PyObject* pyObj = PyObject_CallMethod(_iter, "next", NULL);
//   if (PyObject *e = PyErr_Occurred()) {
//     if (PyErr_GivenExceptionMatches(e, PyExc_StopIteration)) {
//       PYTHON_END;
//       throw jiterator_error("No more samples!");
//     } else {
//       PyObject *etype;
//       PyObject *evalue;
//       PyObject *etrace;
//       PyErr_Fetch(&etype, &evalue, &etrace);      
//       PYTHON_END;
//       throw jpython_error(etype, evalue, etrace);
//     }
//   }
//   if (pyObj == NULL) {
//     PYTHON_END;
//     throw jiterator_error("No more samples!");
//   }


// #if 0
//   PyArrayObject* pyVec = (PyArrayObject*);
//   PyArray_ContiguousFromObject(pyObj, PyArray_FLOAT, 1, 1);
// #else
//   PyArrayObject* pyVec = (PyArrayObject*) pyObj;
// #endif

//   float* comp = (float*) pyVec->data;
//   for (unsigned i = 0; i < size(); i++)
//     gsl_vector_float_set(_vector, i, comp[i]);

//   Py_DECREF(pyVec);
//   PYTHON_END;

//   _increment();

//   return _vector;
// }

// template<>
// const gsl_vector* PyFeatureStream<gsl_vector>::next(int frameX)
// {
//   if (frameX == _frameX) return _vector;

//   PYTHON_START;
//   PyObject* pyObj = PyObject_CallMethod(_iter, "next", NULL);
//   if (PyObject *e = PyErr_Occurred()) {
//     if (PyErr_GivenExceptionMatches(e, PyExc_StopIteration)) {
//       PYTHON_END;
//       throw jiterator_error("No more samples!");
//     } else {
//       PyObject *etype;
//       PyObject *evalue;
//       PyObject *etrace;
//       PyErr_Fetch(&etype, &evalue, &etrace);      
//       PYTHON_END;
//       throw jpython_error(etype, evalue, etrace);
//     }
//   }
//   if (pyObj == NULL) {
//     PYTHON_END;
//     throw jiterator_error("No more samples!");
//   }

// #if 0
//   PyArrayObject* pyVec = (PyArrayObject*);
//   PyArray_ContiguousFromObject(pyObj, PyArray_DOUBLE, 1, 1);
// #else
//   PyArrayObject* pyVec = (PyArrayObject*) pyObj;
// #endif

//   double* comp = (double*) pyVec->data;
//   for (unsigned i = 0; i < size(); i++)
//     gsl_vector_set(_vector, i, comp[i]);

//   Py_DECREF(pyVec);
//   PYTHON_END;

//   _increment();

//   return _vector;
// }

// template<>
// const gsl_vector_complex* PyFeatureStream<gsl_vector_complex>::next(int frameX)
// {
//   if (frameX == _frameX) return _vector;

//   PYTHON_START;
//   PyObject* pyObj = PyObject_CallMethod(_iter, "next", NULL);
//   if (PyObject *e = PyErr_Occurred()) {
//     if (PyErr_GivenExceptionMatches(e, PyExc_StopIteration)) {
//       PYTHON_END;
//       throw jiterator_error("No more samples!");
//     } else {
//       PyObject *etype;
//       PyObject *evalue;
//       PyObject *etrace;
//       PyErr_Fetch(&etype, &evalue, &etrace);      
//       PYTHON_END;
//       throw jpython_error(etype, evalue, etrace);
//     }
//   }
//   if (pyObj == NULL) {
//     PYTHON_END;
//     throw jiterator_error("No more samples!");
//   }

// #if 0
//   PyArrayObject* pyVec = (PyArrayObject*);
//   PyArray_ContiguousFromObject(pyObj, PyArray_CDOUBLE, 1, 1);
// #else
//   PyArrayObject* pyVec = (PyArrayObject*) pyObj;
// #endif

//   gsl_complex* comp = (gsl_complex*) pyVec->data;
//   for (unsigned i = 0; i < size(); i++)
//     gsl_vector_complex_set(_vector, i, comp[i]);

//   Py_DECREF(pyVec);
//   PYTHON_END;

//   _increment();

//   return _vector;
// }
