//
//                           Speech Front End
//                                 (sfe)
//
//  Module:  sfe.stream
//  Purpose: Representation of feature streams.
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


#include "stream/stream.h"


// ----- partial specializations for class template 'FeatureStream' -----
//
template<> FeatureStream<gsl_vector_char, char>::
FeatureStream(unsigned sz, const String& nm) :
  FrameResetX(-1), _size(sz), _frameX(-1), _vector(gsl_vector_char_calloc(_size)),_endOfSamples(false),
  _name(nm)
{
  gsl_vector_char_set_zero(_vector);
}

template<> FeatureStream<gsl_vector_short, short>::
FeatureStream(unsigned sz, const String& nm) :
  FrameResetX(-1), _size(sz), _frameX(-1), _vector(gsl_vector_short_calloc(_size)),_endOfSamples(false),
  _name(nm)
{
  gsl_vector_short_set_zero(_vector);
}

template<> FeatureStream<gsl_vector_float, float>::
FeatureStream(unsigned sz, const String& nm) :
  FrameResetX(-1), _size(sz), _frameX(-1), _vector(gsl_vector_float_calloc(_size)),_endOfSamples(false),
  _name(nm)
{
  gsl_vector_float_set_zero(_vector);
}

template<> FeatureStream<gsl_vector, double>::
FeatureStream(unsigned sz, const String& nm) :
  FrameResetX(-1), _size(sz), _frameX(-1), _vector(gsl_vector_calloc(_size)),_endOfSamples(false),
  _name(nm)
{
  gsl_vector_set_zero(_vector);
}

template<> FeatureStream<gsl_vector_complex, gsl_complex>::
FeatureStream(unsigned sz, const String& nm) :
  FrameResetX(-1), _size(sz), _frameX(-1), _vector(gsl_vector_complex_calloc(_size)),_endOfSamples(false),
  _name(nm)
{
  gsl_vector_complex_set_zero(_vector);
}

template<> FeatureStream<gsl_vector_char, char>::~FeatureStream()    { gsl_vector_char_free(_vector); }
template<> FeatureStream<gsl_vector_short, short>::~FeatureStream()   { gsl_vector_short_free(_vector); }
template<> FeatureStream<gsl_vector_float, float>::~FeatureStream()   { gsl_vector_float_free(_vector); }
template<> FeatureStream<gsl_vector, double>::~FeatureStream()         { gsl_vector_free(_vector); }
template<> FeatureStream<gsl_vector_complex, gsl_complex>::~FeatureStream() { gsl_vector_complex_free(_vector); }

template<>
void FeatureStream<gsl_vector_char, char>::_gsl_vector_set(gsl_vector_char *vector, int index, char value) {
  gsl_vector_char_set(vector, index, value);
};

template<>
void FeatureStream<gsl_vector_short, short>::_gsl_vector_set(gsl_vector_short *vector, int index, short value) {
  gsl_vector_short_set(vector, index, value);
};

template<> void FeatureStream<gsl_vector_float, float>::_gsl_vector_set(gsl_vector_float *vector, int index, float value) {
  gsl_vector_float_set(vector, index, value);
};

template<> void FeatureStream<gsl_vector, double>::_gsl_vector_set(gsl_vector *vector, int index, double value) {
  gsl_vector_set(vector, index, value);
};

template<> void FeatureStream<gsl_vector_complex, gsl_complex>::_gsl_vector_set(gsl_vector_complex *vector, int index, gsl_complex value) {
  gsl_vector_complex_set(vector, index, value);
};
