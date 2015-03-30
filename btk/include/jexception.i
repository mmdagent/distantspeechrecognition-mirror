// -*- C++ -*- forwarding header.
//
//                                Millenium
//                   Automatic Speech Recognition System
//                                  (asr)
//
//  Module:
//  Purpose: SWIG type maps.
//  Author:  Fabian Jakobs

%{
#include "common/jexception.h"
#include "common/jpython_error.h"

using namespace std;
%}

%init %{
#ifdef THREADING
  PyEval_InitThreads();
#endif
%}

// // exception handling
// #ifdef THREADING
// %exception {
//   PyThreadState *_save;
//   Py_UNBLOCK_THREADS
//   try {
//     $action
//   } catch(jio_error& e) {
//     Py_BLOCK_THREADS
//     PyErr_SetString(PyExc_IOError, e.what());
//     return NULL;
//   } catch(jiterator_error& e) {
//     Py_BLOCK_THREADS
//     PyErr_SetString(PyExc_StopIteration, "");
//     return NULL;
//   } catch(jpython_error& e) {
//     Py_BLOCK_THREADS
//     PyErr_Restore(e.getType(), e.getValue(), e.getTrace());
//     return NULL;
//   } catch(j_error& e) {
//     if (e.getCode() == JPYTHON) {
//       jpython_error *pe = static_cast<jpython_error*>(&e);
//       Py_BLOCK_THREADS;
//       PyErr_Restore(pe->getType(), pe->getValue(), pe->getTrace());
//       return NULL;
//     }
//     Py_BLOCK_THREADS;
//     PyErr_SetString(PyExc_Exception, e.what());
//     return NULL;
//   }
//   Py_BLOCK_THREADS
// }
// #else
%exception {
  jpython_error *pe;
  try {
    $action
  } catch(j_error& e) {
    switch (e.getCode()) {
    case JITERATOR:
      PyErr_SetString(PyExc_StopIteration, "");
      return NULL;
    case JIO:
      PyErr_SetString(PyExc_IOError, e.what());
      return NULL;
    case JPYTHON: 
      //pe = static_cast<jpython_error*>(&e);
      //PyErr_Restore(pe->getType(), pe->getValue(), pe->getTrace());
      return NULL;
    default:
      break;
    }
    PyErr_SetString(PyExc_Exception, e.what());
    return NULL;
  } catch (...) {
    PyErr_SetString(PyExc_Exception, "unknown error");
    return NULL;
  };
}
//#endif
