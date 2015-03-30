//                              -*- C++ -*-
//
//                           Speech Front End
//                                 (sfe)
//
//  Module:  sfe.stream
//  Purpose: Representation of feature streams.
//  Author:  John McDonough
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


#ifndef _stream_h_
#define _stream_h_

#include <gsl/gsl_vector.h>
#include <gsl/gsl_complex.h>
#include "common/refcount.h"


// ----- interface class for 'FeatureStream' -----
//
template <typename Type, typename item_type>
class FeatureStream : public Countable {
 public:
  virtual ~FeatureStream();

  const String& name() const { return _name; }
  unsigned      size() const { return _size; }

  virtual const Type* next(int frameX = -5) = 0;
  const Type* current() {
    if (_frameX < 0)
      throw jconsistency_error("Frame index (%d) < 0.", _frameX);
    return next(_frameX);
  }

  bool isEnd(){ return _endOfSamples; };

  virtual void reset() {
    // printf("Resetting Feature %s.\n", _name.c_str());
    _frameX = FrameResetX;
    _endOfSamples = false;
  }

  virtual int frameX() const { return _frameX; }

 protected:
  FeatureStream(unsigned sz, const String& nm);
  void _gsl_vector_set(Type *vector, int index, item_type value);
  size_t itemsize() {return sizeof(item_type); };

  void _increment() { _frameX++; }

  const int					FrameResetX;
  const unsigned				_size;
  int						_frameX;
  Type*						_vector;
  bool                                          _endOfSamples;

 private:
  const String					_name;
};


// ----- declare partial specializations of 'FeatureStream' -----
//
template<> FeatureStream<gsl_vector_char, char>::FeatureStream(unsigned sz, const String& nm);
template<> FeatureStream<gsl_vector_short, short>::FeatureStream(unsigned sz, const String& nm);
template<> FeatureStream<gsl_vector_float, float>::FeatureStream(unsigned sz, const String& nm);
template<> FeatureStream<gsl_vector, double>::FeatureStream(unsigned sz, const String& nm);
template<> FeatureStream<gsl_vector_complex, gsl_complex>::FeatureStream(unsigned sz, const String& nm);
template<> FeatureStream<gsl_vector_char, char>::~FeatureStream();
template<> FeatureStream<gsl_vector_short, short>::~FeatureStream();
template<> FeatureStream<gsl_vector_float, float>::~FeatureStream();
template<> FeatureStream<gsl_vector, double>::~FeatureStream();
template<> FeatureStream<gsl_vector_complex, gsl_complex>::~FeatureStream();
template<> void FeatureStream<gsl_vector_char, char>::_gsl_vector_set(gsl_vector_char *vector, int index, char value);
template<> void FeatureStream<gsl_vector_short, short>::_gsl_vector_set(gsl_vector_short *vector, int index, short value);
template<> void FeatureStream<gsl_vector_float, float>::_gsl_vector_set(gsl_vector_float *vector, int index, float value);
template<> void FeatureStream<gsl_vector, double>::_gsl_vector_set(gsl_vector *vector, int index, double value);
template<> void FeatureStream<gsl_vector_complex, gsl_complex>::_gsl_vector_set(gsl_vector_complex *vector, int index, gsl_complex value);


typedef FeatureStream<gsl_vector_char, char>		VectorCharFeatureStream;
typedef FeatureStream<gsl_vector_short, short>		VectorShortFeatureStream;
typedef FeatureStream<gsl_vector_float, float>		VectorFloatFeatureStream;
typedef FeatureStream<gsl_vector, double>		VectorFeatureStream;
typedef FeatureStream<gsl_vector_complex, gsl_complex>	VectorComplexFeatureStream;

typedef refcountable_ptr<VectorCharFeatureStream>	VectorCharFeatureStreamPtr;
typedef refcountable_ptr<VectorShortFeatureStream>	VectorShortFeatureStreamPtr;
typedef refcountable_ptr<VectorFloatFeatureStream>	VectorFloatFeatureStreamPtr;
typedef refcountable_ptr<VectorFeatureStream>		VectorFeatureStreamPtr;
typedef refcountable_ptr<VectorComplexFeatureStream>	VectorComplexFeatureStreamPtr;

#endif
