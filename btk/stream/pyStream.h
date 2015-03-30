//                              -*- C++ -*-
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


#ifndef _pyStream_h_
#define _pyStream_h_

#include <Python.h>
#include "stream/stream.h"
#include <numpy/arrayobject.h>
#include "common/jexception.h"
#include "common/jpython_error.h"

#ifdef THREADING
#define PYTHON_START PyGILState_STATE _state = PyGILState_Ensure()
#define PYTHON_END PyGILState_Release(_state)
#else
#define PYTHON_START
#define PYTHON_END
#endif

// ----- interface class for 'PyFeatureStream' -----
//
template <typename gsl_type, typename c_type, int numpy_type>
class PyFeatureStream : public FeatureStream<gsl_type, c_type> {
  typedef FeatureStream<gsl_type, c_type> _FeatureStream;
 public:
  PyFeatureStream(PyObject* c, const String& nm);
  ~PyFeatureStream();

  virtual const gsl_type* next(int frameX = -5);
  virtual void reset();

 private:
  unsigned _getSize(PyObject* c) const;

  PyObject*		_cont;
  PyObject*		_iter;
};


template <typename gsl_type, typename c_type, int numpy_type>
PyFeatureStream<gsl_type, c_type, numpy_type>::PyFeatureStream(PyObject* c, const String& nm) :
  FeatureStream<gsl_type, c_type>(_getSize(c), nm), _cont(c)
{
  char iter[] = "__iter__";

  PYTHON_START;
  _iter = PyObject_CallMethod(_cont, iter, NULL);
  if (_iter == NULL) {
    PYTHON_END;
    throw jpython_error();
  }
  Py_INCREF(_cont); Py_INCREF(_iter);
  PYTHON_END;
}


template <typename gsl_type, typename c_type, int numpy_type>
PyFeatureStream<gsl_type, c_type, numpy_type>::~PyFeatureStream() { 
  PYTHON_START;
  Py_DECREF(_cont); 
  Py_DECREF(_iter);
  PYTHON_END;
}


template <typename gsl_type, typename c_type, int numpy_type>
const gsl_type* PyFeatureStream<gsl_type, c_type, numpy_type>::next(int frameX)
{ 
  if (frameX == _FeatureStream::_frameX) return _FeatureStream::_vector;

  PYTHON_START;
  // import_array();

  bool err = false;
  c_type *data;
  PyObject* pyObj = NULL;
  PyArrayObject* pyVec = NULL;

  char next[] = "next";
  pyObj = PyObject_CallMethod(_iter, next, NULL);
  if (pyObj == NULL) {
    if (!PyErr_ExceptionMatches(PyExc_StopIteration)) {
      err = true;
      goto error;
    }
    PyErr_Clear();
    throw jiterator_error("No more samples!");
  }

  pyVec = (PyArrayObject*) PyArray_ContiguousFromObject(pyObj, numpy_type, 1, 1); 
  if (pyVec == NULL) {
    err = true;
    goto error;
  }

  data = (c_type*) pyVec->data;
  for (unsigned i = 0; i < _FeatureStream::_size; i++)
    _gsl_vector_set(_FeatureStream::_vector, i, data[i]);

  // cleanup code
 error:
  Py_XDECREF(pyObj);
  Py_XDECREF(pyVec);
  PYTHON_END;
  if (err) throw jpython_error();
  _FeatureStream::_increment();
  return _FeatureStream::_vector;
}


template <typename gsl_type, typename c_type, int numpy_type>
void PyFeatureStream<gsl_type, c_type, numpy_type>::reset() {
  char reset[] = "reset";
  char iter[]  = "__iter__";

  PYTHON_START;
  if (PyObject_CallMethod(_cont, reset, NULL) == NULL) {      
    PYTHON_END;
    throw jpython_error();
  }
  Py_DECREF(_iter);
  _iter = PyObject_CallMethod(_cont, iter, NULL);
  if (_iter == NULL) {
    PYTHON_END;
    throw jpython_error();
  }
  Py_INCREF(_iter);
  PYTHON_END;
  FeatureStream<gsl_type, c_type>::reset();
}


template <typename gsl_type, typename c_type, int numpy_type>
unsigned PyFeatureStream<gsl_type, c_type, numpy_type>::_getSize(PyObject* c) const {
  PYTHON_START;
  bool err = false;
  long sz;
  Py_INCREF(c);
  char size[] = "size";
  
  PyObject* pyObj = NULL;
  pyObj = PyObject_CallMethod(c, size, NULL);
  if (pyObj == NULL) {
    err = true;
    goto error;
  }

  sz = PyInt_AsLong(pyObj);

  // clean up
 error:
  Py_XDECREF(c);  Py_XDECREF(pyObj);
  PYTHON_END;
  if (err) throw jpython_error();

  return unsigned(sz);
}

// typedef PyFeatureStream<gsl_vector_char, char, PyArray_SBYTE>		  PyVectorCharFeatureStream;
typedef PyFeatureStream<gsl_vector_short, short, PyArray_SHORT>		  PyVectorShortFeatureStream;
typedef PyFeatureStream<gsl_vector_float, float, PyArray_FLOAT>		  PyVectorFloatFeatureStream;
typedef PyFeatureStream<gsl_vector, double, PyArray_DOUBLE>		  PyVectorFeatureStream;
typedef PyFeatureStream<gsl_vector_complex, gsl_complex, PyArray_CDOUBLE> PyVectorComplexFeatureStream;


// ----- smart pointer declarations 'PyFeatureStream' -----
//
// typedef Inherit<PyVectorCharFeatureStream,    VectorCharFeatureStreamPtr>    PyVectorCharFeatureStreamPtr;
typedef Inherit<PyVectorShortFeatureStream,   VectorShortFeatureStreamPtr>   PyVectorShortFeatureStreamPtr;
typedef Inherit<PyVectorFloatFeatureStream,   VectorFloatFeatureStreamPtr>   PyVectorFloatFeatureStreamPtr;
typedef Inherit<PyVectorFeatureStream,        VectorFeatureStreamPtr>        PyVectorFeatureStreamPtr;
typedef Inherit<PyVectorComplexFeatureStream, VectorComplexFeatureStreamPtr> PyVectorComplexFeatureStreamPtr;


#endif
