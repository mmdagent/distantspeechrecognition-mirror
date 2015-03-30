//                              -*- C++ -*-
//
//                              Millennium
//                   Distant Speech Recognition System
//                                (dsr)
//
//  Module:  asr.fsm
//  Purpose: Representation and manipulation of finite state machines.
//  Author:  Stephan Kanthak, ported to Millennium by John McDonough
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


#ifndef _skhash_h_
#define _skhash_h_

#include <stdint.h>
#include <vector>
#include <limits.h>
#include <string.h>

typedef int8_t      s8;
typedef uint8_t     u8;
typedef int16_t     s16;
typedef uint16_t    u16;
typedef int32_t     s32;
typedef uint32_t    u32;
typedef int64_t     s64;
typedef uint64_t    u64;

typedef float       f32;
typedef double      f64;

typedef u32 StateId;
typedef StateId StateTag;
static const StateTag StateIdMask = 0x3fffffff;
static const int StateIdBits = 30;

template<class T> class Vector : public std::vector<T> {
 public:
  typedef std::vector<T> Precursor;
  typedef typename Precursor::iterator iterator;
  typedef typename Precursor::const_iterator const_iterator;
 public:
  Vector() : Precursor() {}
  Vector(size_t size) : std::vector<T>(size) {}
  Vector(size_t size, const T &t) : std::vector<T>(size, t) {}

  void grow(size_t s, const T &init = T()) {
    if (s >= capacity()) this->reserve(2 * capacity());
    if (s >= size()) this->insert(end(), s - size() + 1, init);
  }

  /** Free over-allocated storage. */
  void yield() {
    std::vector<T> tmp(*this);
    swap(tmp);
    ensure(capacity() == size());
  }

  void minimize() {
    // not implemented
  }

  size_t capacity() const { return std::vector<T>::capacity(); }

  size_t size() const { return std::vector<T>::size(); }

        iterator end()       { return std::vector<T>::end(); }
  const_iterator end() const { return std::vector<T>::end(); }

  size_t getMemoryUsed() const {
    return sizeof(T) * capacity() + sizeof(*this);
  }
};

u32  estimateBytes(u32 x);
void setBytes(Vector<u8>::iterator i, u32 x, int nBytes);
void appendBytes(Vector<u8>& v, u32 x, int nBytes);
u32  getBytesAndIncrement(Vector<u8>::const_iterator &a, int nBytes);
u32  getBytes(Vector<u8>::const_iterator a, int nBytes);

class Weight {
 public:
  Weight() {}
  explicit Weight(s32 value) { value_.s = value; }
  explicit Weight(u32 value) { value_.u = value; }
  explicit Weight(f32 value) { value_.f = value; }
  explicit Weight(double value) { value_.f = value; }

  operator int() { return value_.s; }
  operator int() const { return value_.s; }
  operator u32() { return value_.u; }
  operator u32() const { return value_.u; }
  operator float() { return value_.f; }
  operator float() const { return value_.f; }
  const Weight& operator= (const Weight &w) { value_ = w.value_; return *this; }

  bool operator== (const Weight &w) const {
    return memcmp(&value_, &w.value_, sizeof(Value)) == 0;
  }

  bool operator!= (const Weight &w) const {
    return memcmp(&value_, &w.value_, sizeof(Value)) != 0;
  }

  bool operator< (const Weight &w) const {
    return memcmp(&value_, &w.value_, sizeof(Value)) < 0;
  }

  float hash() const {
    int tmp = (int) (value_.f * 1000);
    return tmp / 1000.0;
  }

 private:
  union Value {
    s32 s;
    u32 u;
    f32 f;
  };
  Value value_;
};

// Static information about elementary types.
template<class T> struct Type {

  // Name to be used to represent data type.
  static const char *name;

  // Largest representable value of data type.
  static const T max;

  // Smallest representable value of data type.
  // Note that unlike std::numeric_limits<>::min this is the most negative
  // value also for floating point types.
  static const T min;

  // The difference between the smallest value greater than one and one.
  static const T epsilon;

  // Smallest representable value greater than zero.
  // For all integer types this is one.  For floating point
  // types this is the same as std::numeric_limits<>::min or
  // FLT_MIN / DBL_MIN.
  static const T delta;
};


template<class T, class HashKey, class HashEqual /* = std::equal_to<T> */ >
class Hash {
  public:
  typedef u32 Cursor;
  private:
  Vector<Cursor> bins_;
  class Element {
  public:
    Cursor next_;
    T data_;
  public:
    Element(Cursor next, const T &data) : next_(next), data_(data) {}
  };
  HashKey hashKey_;
  HashEqual hashEqual_;
  std::vector<Element> elements_;

  public:
  class const_iterator : public std::vector<Element>::const_iterator {
  public:
    const_iterator(const typename std::vector<Element>::const_iterator &i) :
      std::vector<Element>::const_iterator(i) {}
      const T& operator*() const { return std::vector<Element>::const_iterator::operator*().data_; }
      const T* operator->() const { return &std::vector<Element>::const_iterator::operator*().data_; }
  };

  public:
  Hash(u32 defaultSize = 10) {
    bins_.resize(defaultSize, UINT_MAX);
  }
  Hash(const HashKey &key, const HashEqual &equal, u32 defaultSize = 10) :
    hashKey_(key), hashEqual_(equal) {
    bins_.resize(defaultSize, UINT_MAX);
  }
  void clear() {
    std::fill(bins_.begin(), bins_.end(), UINT_MAX);
    elements_.erase(elements_.begin(), elements_.end());
  }
  void resize(u32 size) {
    std::fill(bins_.begin(), bins_.end(), UINT_MAX);
    bins_.grow(size, UINT_MAX);
    size = bins_.size();
    for (typename std::vector<Element>::iterator i = elements_.begin(); i != elements_.end(); ++i) {
      u32 key = hashKey_((*i).data_) % size;
      i->next_ = bins_[key];
      bins_[key] = i - elements_.begin();
    }
  }
  Cursor insertWithoutResize(const T &d) {
    u32 key = hashKey_(d) % bins_.size(), i = bins_[key];
    for (; (i != UINT_MAX) && (!hashEqual_(elements_[i].data_, d)); i = elements_[i].next_);
    if (i == UINT_MAX) {
      i = elements_.size();
      elements_.push_back(Element(bins_[key], d));
      bins_[key] = i;
    }
    return i;
  }
  std::pair<Cursor, bool> insertExisting(const T &d) {
    u32 key = hashKey_(d), i = bins_[key % bins_.size()];
    for (; (i != UINT_MAX) && (!hashEqual_(elements_[i].data_, d)); i = elements_[i].next_);
    if (i != UINT_MAX) return std::make_pair(i, true);
    if (elements_.size() > 2 * bins_.size()) resize(2 * bins_.size() - 1);
    i = elements_.size();
    key = key % bins_.size();
    elements_.push_back(Element(bins_[key], d));
    bins_[key] = i;
    return std::make_pair(i, false);
  }
  Cursor insert(const T &d) {
    std::pair<Cursor, bool> tmp = insertExisting(d);
    return tmp.first;
  }
  size_t size() const { return elements_.size(); }
        T& operator[] (const Cursor p)       { return elements_[p].data_; }
  const T& operator[] (const Cursor p) const { return elements_[p].data_; }
  const_iterator begin() const { return elements_.begin(); }
  const_iterator end() const { return elements_.end(); }
  size_t getMemoryUsed() const { return bins_.getMemoryUsed() + sizeof(Element) * elements_.size(); }
};

#endif
