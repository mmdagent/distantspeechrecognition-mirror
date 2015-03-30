//                              -*- C++ -*-
//
//                               Millenium
//                  Automatic Speech Recognition System
//                                 (asr)
//
//  Module:  asr.natural
//  Purpose: Common operations.
//  Author:  John McDonough

#ifndef _natural_h_
#define _natural_h_

#include <assert.h>
#include <map>
#include <vector>
#include <list>

#include "common/jexception.h"
#include "common/mlist.h"

// ----- definition for class `NaturalIndex' -----
//
class NaturalIndex {
  friend class HiddenMarkovModel;
 public:
  static UnShrt initialize(UnShrt nsub);
  static void   deinitialize();

 protected:
  inline NaturalIndex(UnShrt nsub = 0);
  /* inline */ NaturalIndex(UnShrt _size, UnShrt nsub);

  static       UnShrt		_nSubFeat;  // no. of sub-features
  static const UnShrt*  	_Single[];  // single index over entire feature
  static const UnShrt**		_Double[];  // index by sub-feature and cep. no.

 private:
  static const UnShrt _MaxNoVectorSizes;
};

NaturalIndex::NaturalIndex(UnShrt nsub)
{
  // ---- !!!Hack!!! ----
  NaturalIndex::initialize(nsub);
  // ---- !!!Hack!!! ----

  if (_nSubFeat == 0)
    throw jinitialization_error("NaturalIndex has not been initialized properly.");
}

#ifdef JMcD

NaturalIndex::NaturalIndex(UnShrt _size, UnShrt nsub)
{
  if (_size < 1)
    throw jdimension_error("Must specify non-zero size for NaturalIndex.");

  if (nsub != 0 && (_size / nsub) >= _MaxNoVectorSizes)
    throw jdimension_error("Sub-feature size %d is not supported.", _size);

  initialize(nsub);
}

#endif

template <class Type> class _NaturalVector;

template <class Type>
ostream& operator<<(ostream&, const _NaturalVector<Type>&);

template <class Type>
class _NaturalVector : private NaturalIndex {
  friend class CodebookBasic;
  template <class OtherType> friend class _NaturalVector;
 public:
  inline _NaturalVector();
  inline _NaturalVector(Type* v, UnShrt sz, UnShrt nsub = 0);
  inline _NaturalVector(UnShrt sz, UnShrt nsub = 0);
  inline _NaturalVector(const _NaturalVector& v, bool deepCopy = false);

  template <class OtherType>
    _NaturalVector(const _NaturalVector<OtherType>& ot)
    : _vec(NULL), _subsize(ot._subsize), _size(ot._size),
    _single(ot._single), _double(ot._double),
    _allocFlag(true)
    {
      if (_initFlag == false) _initialize();

      _vec = new Type[_size];
      for (int i = 0; i < _size; i++) _vec[i] = (Type) ot._vec[i];
      _nVec++;
    }

  inline ~_NaturalVector();

  inline void resize(UnShrt sz, UnShrt nsub = 0);

  inline _NaturalVector& operator=(double val);
  inline _NaturalVector& operator*=(double val);
  inline _NaturalVector& operator=(const _NaturalVector& rhs);
  inline _NaturalVector& operator/=(double val);

  inline Type   operator[](int icep) const;
  inline Type&  operator[](int icep);
  inline Type   operator()(int icep) const;
  inline Type&  operator()(int icep);
  inline Type   operator()(int isub, int icep) const;
  inline Type&  operator()(int isub, int icep);

  friend ostream& operator<< <> (ostream& os, const _NaturalVector& v);

  bool isZero()       const { return (_vec == NULL); }
  UnShrt featLen()    const { return (_vec == NULL) ? 0 : _size; }
  UnShrt subFeatLen() const { return (_vec == NULL) ? 0 : _subsize; }
  UnShrt nSubFeat()   const { return (_vec == NULL) ? 0 : _nSubFeat; }
  _NaturalVector<Type>& square();

 private:
  static void		_initialize();
  static int		_nVec;
  static bool		_initFlag;

  // const Type* vec() const { return (_vec); }
  operator const Type*() const { return (_vec); }

  UnShrt		_nSubFeat;
  Type*			_vec;
  UnShrt		_subsize;
  UnShrt		_size;
  const UnShrt*		_single;
  const UnShrt**	_double;
  bool			_allocFlag;
};

// ----- static variables and inline methods for `_NaturalVector' -----
//
template <class Type>
int  _NaturalVector<Type>::_nVec     = 0;
template <class Type>
bool _NaturalVector<Type>::_initFlag = false;

template <class Type>
_NaturalVector<Type>::_NaturalVector()
  : _nSubFeat(0), _vec(NULL), _subsize(0), _size(0), _single(NULL), _double(NULL),
    _allocFlag(false)
{
  _nVec++;
}

template <class Type>
_NaturalVector<Type>::_NaturalVector(Type* v, UnShrt sz, UnShrt nsub)
  : _nSubFeat((nsub == 0) ? NaturalIndex::_nSubFeat : nsub), _vec(v),
    _subsize(sz / _nSubFeat), _size(sz),
    _single(_Single[_subsize]), _double(_Double[_subsize]),
    _allocFlag(false)
{
  assert(_Single[_subsize]);

  if (_initFlag == false) _initialize();  // not strictly necessary

  _nVec++;
}

template <class Type>
_NaturalVector<Type>::_NaturalVector(UnShrt sz, UnShrt nsub)
  : NaturalIndex(sz, nsub),
    _nSubFeat((nsub == 0) ? NaturalIndex::_nSubFeat : nsub),
    _subsize(sz / nsub), _size(sz),
    _single(_Single[_subsize]), _double(_Double[_subsize]),
    _allocFlag(true)
{
  if (_initFlag == false) _initialize();

  _vec = new Type[sz];
  for (int i = 0; i < sz; i++) _vec[i] = 0.0;
  _nVec++;
}

template <class Type>
_NaturalVector<Type>::_NaturalVector(const _NaturalVector& v, bool deepCopy)
  : _nSubFeat(v._nSubFeat), _vec(v._vec),
    _subsize(v._subsize), _size(v._size),
    _single(v._single), _double(v._double),
    _allocFlag(deepCopy)
{
  _nVec++;

  if (deepCopy == false) return;

  _vec = new Type[_size];
  for (UnShrt i = 0; i < _size; i++)
    _vec[i] = v._vec[i];
}

template <class Type>
_NaturalVector<Type>::~_NaturalVector()
{
  if (_allocFlag) delete[] _vec;
}
  
template <class Type>
Type _NaturalVector<Type>::operator[](int icep) const {
  assert(icep >= 0 && icep < featLen());
  return _vec[_single[icep]];
}

template <class Type>
Type& _NaturalVector<Type>::operator[](int icep) {
  assert(icep >= 0 && icep < featLen());
  return _vec[_single[icep]];
}

template <class Type>
Type _NaturalVector<Type>::operator()(int icep) const {
  assert((icep >= 0) && (icep < featLen()));
  return _vec[_single[icep]];
}

template <class Type>
Type& _NaturalVector<Type>::operator()(int icep) {
  assert(icep >= 0 && icep < featLen());
  return _vec[_single[icep]];
}

template <class Type>
Type _NaturalVector<Type>::operator()(int isub, int icep) const {
  assert(isub >= 0 && isub < nSubFeat());
  assert(icep >= 0 && icep < subFeatLen());
  return _vec[_double[isub][icep]];
}

template <class Type>
Type& _NaturalVector<Type>::operator()(int isub, int icep) {
  assert(isub >= 0 && isub < nSubFeat());
  assert(icep >= 0 && icep < subFeatLen());
  return _vec[_double[isub][icep]];
}

template <class Type>
void _NaturalVector<Type>::_initialize()
{
  _initFlag = true;
}

template <class Type>
void _NaturalVector<Type>::resize(UnShrt sz, UnShrt nsub)
{
  nsub = NaturalIndex::initialize(nsub);
  if (_initFlag == false) _initialize();

  if (sz == _size  && _nSubFeat  == nsub) return;

  if (_vec != NULL && _allocFlag == true) delete[] _vec;

  _nSubFeat  = nsub;
  _vec       = new Type[sz];
  _size      = sz;
  _subsize   = sz / nsub;
  _single    = _Single[_subsize];
  _double    = _Double[_subsize];
  _allocFlag = true;
}

template <class Type>
_NaturalVector<Type>& _NaturalVector<Type>::operator=(double val)
{
  for (UnShrt j = 0; j < _size; j++) _vec[j] = val;
  return *this;
}

template <class Type>
_NaturalVector<Type>& _NaturalVector<Type>::operator*=(double val)
{
  for (UnShrt j = 0; j < _size; j++) _vec[j] *= val;
  return *this;
}

template <class Type>
_NaturalVector<Type>& _NaturalVector<Type>::operator/=(double val)
{
  for (UnShrt j = 0; j < _size; j++) _vec[j] /= val;
  return *this;
}

// assignment operator: deftly handles vectors of unequal length
//
template <class Type>
_NaturalVector<Type>& _NaturalVector<Type>::operator=(const _NaturalVector& rhs)
{
  const UnShrt subSize = min(_subsize, rhs._subsize);
  (*this) = 0.0;

  for (UnShrt isub = 0; isub < _nSubFeat; isub++)
    for (UnShrt icep = 0; icep < subSize; icep++)
      (*this)(isub,icep) = rhs(isub,icep);

  return *this;
}

template <class Type>
_NaturalVector<Type>& _NaturalVector<Type>::square()
{
  for (UnShrt n = 0; n < _size; n++) _vec[n] *= _vec[n];

  return *this;
}

template <class Type>
ostream& operator<<(ostream& os, const _NaturalVector<Type>& v)
{
  os << "Feat Len     = " << v.featLen()    << endl
     << "Sub Feat Len = " << v.subFeatLen() << endl
     << "No. Sub Len  = " << v.nSubFeat()   << endl;
    
#if 0
  for (unsigned j = 0; j < /* v._subsize */ v._size; j++)
    os << " " << setw(4) << j << " : "
       << setprecision(16) << setw(16) << v[j] << endl;
#endif

  return os;
}

typedef _NaturalVector<float>		NaturalVector;
typedef _NaturalVector<double>		DNaturalVector;
typedef _NaturalVector<float>		InverseVariance;

#endif
