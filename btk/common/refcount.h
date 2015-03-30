//                              -*- C++ -*-
//
//                           Speech Front End
//                                 (sfe)
//
//  Module:  sfe.common
//  Purpose: Common operations.
//  Author:  Fabian Jakobs
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


#ifndef _refcount_h_
#define _refcount_h_

#include "common/memoryManager.h"

// class template to assist with upcasting
template <class To, class From>
To& Cast(const From& from) { return *((To *) &from); }

// class template to give smart pointers the same inheritance as the
// object pointed to
template <class DerivedType, class BaseTypePtr>
class Inherit : public BaseTypePtr {
 public:
  Inherit(DerivedType* s = NULL) : BaseTypePtr(s) { }

  DerivedType* operator->() const { return (DerivedType*) BaseTypePtr::the_p; }

  /*
        DerivedType& operator*() { return *((DerivedType*) the_p); }
  const DerivedType& operator*() { return *((DerivedType*) the_p); }
  */
};

class ReferenceCount {
  template<class T>
    friend class refcount_ptr;
  class _ReferenceCount {
  public:
    _ReferenceCount(unsigned c)
      : _count(c) { }

    void* operator new(size_t sz) { return memoryManager().newElem(); }
    void operator delete(void* e) { memoryManager().deleteElem(e); }

    static MemoryManager<_ReferenceCount>& memoryManager();

    unsigned		_count;
  };

 public:
  // create with count of 1
  ReferenceCount(): p_refcnt(new _ReferenceCount(1)) { };
  // copy and increment count
  ReferenceCount(const ReferenceCount& anRC): p_refcnt(anRC.p_refcnt) {
    p_refcnt->_count++;
  };

  // decrement count, delete if 0
  ~ReferenceCount() { decrement(); }

  // Assign, decrement lhs count, increment rhs
  ReferenceCount& operator=(const ReferenceCount& rhs) {
    rhs.p_refcnt->_count++;
    decrement();
    p_refcnt = rhs.p_refcnt;
    return *this;
  }

  // True if count is 1
  bool unique() const { return p_refcnt->_count == 1;};

 private:
  _ReferenceCount* p_refcnt;

  // Decrement count; delete if 0
  void decrement() {
    if (unique()) delete p_refcnt;
    else p_refcnt->_count--;
  }
};


// Implementation of a reference-counted object pointer
// class as described in Barton and Nackman, 1988
template<class T>
class refcount_ptr {
 public:
  // construct pointing to a heap object
  refcount_ptr(T* newobj = NULL)
    : the_p(newobj), smartBehavior(true) { }

  refcount_ptr(const refcount_ptr& rhs)
    : the_p(rhs.the_p), smartBehavior(true), refCount(rhs.refCount) { }

  virtual ~refcount_ptr() {
    if (smartBehavior) {
      if (unique()) delete the_p;
    } else {
      refCount.p_refcnt->_count++;
    }
  }

  void disable()
  {
    if (isNull())
      throw jconsistency_error("Attempted to disable a NULL pointer.");

    if (unique())
      throw jconsistency_error("Attempted to disable a unique pointer.");
    smartBehavior = false;
    refCount.p_refcnt->_count--;
  }

  refcount_ptr<T>& operator=(const refcount_ptr<T>& rhs) {
    if (the_p != rhs.the_p) {
      if (smartBehavior) {
	if (unique()) delete the_p;
      } else {
	refCount.p_refcnt->_count++;
	smartBehavior = true;
      }
      the_p = rhs.the_p;
      refCount = rhs.refCount;
    }
    return *this;
  }

  refcount_ptr<T>& operator=(T* rhs) {
    if (smartBehavior) {
      if (unique()) delete the_p;
    } else {
      refCount.p_refcnt->_count++;
      smartBehavior = true;
    }
    the_p = rhs;
    refCount = ReferenceCount();
    return *this;
  }

        T& operator*()        { return *the_p; }
  const T& operator*()  const { return *the_p; }

  T* operator->() const { return  the_p; }

  // Is count one?
  bool unique() const { return refCount.unique(); }

  // Is the_p pointing to NULL?
  bool isNull() const { return the_p == 0; }

  friend
    bool operator==(const refcount_ptr<T>& lhs, const refcount_ptr<T>& rhs) {
    return lhs.the_p == rhs.the_p;
  }
  friend
    bool operator!=(const refcount_ptr<T>& lhs, const refcount_ptr<T>& rhs) {
    return lhs.the_p != rhs.the_p;
  }

 protected:
  T*				the_p;

 private:
  bool 				smartBehavior;
  ReferenceCount		refCount;	// number of pointers to the heap object
};


// ----- definition of class `Countable' -----
//
class Countable {
 public:
  virtual ~Countable() { }

  bool unique() const { return _count == 1; }
  void increment() { _count++;  assert(_count < UINT_MAX); }
  void decrement() { assert(_count > 0); _count--;  assert(_count > 0); }

 protected:
  Countable() : _count(0) { }

 private:
  unsigned			_count;
};


// ----- definition of class template `refcountable_ptr' -----
//
template<class T>
class refcountable_ptr {
 public:
  // construct pointing to a heap object
  refcountable_ptr(T* newobj = NULL)
    : the_p(newobj), smartBehavior(true) { increment(); }

  refcountable_ptr(const refcountable_ptr& rhs)
    : the_p(rhs.the_p), smartBehavior(true) { increment(); }

  virtual ~refcountable_ptr() {
    if (smartBehavior)
      decrement();
  }

  void disable()
  {
    if (isNull())
      throw jconsistency_error("Attempted to disable a NULL pointer.");

    if (unique())
      throw jconsistency_error("Attempted to disable a unique pointer.");
    smartBehavior = false;
    decrement();
  }

  refcountable_ptr<T>& operator=(const refcountable_ptr<T>& rhs) {
    if (the_p != rhs.the_p) {
      if (smartBehavior)
	decrement();

      the_p = rhs.the_p;
      increment();
    }
    return *this;
  }

  refcountable_ptr<T>& operator=(T* rhs) {
    if (smartBehavior)
      decrement();
    else
      smartBehavior = true;

    the_p = rhs;
    increment();

    return *this;
  }

        T& operator*()        { return Cast<T>(*the_p); }
  const T& operator*()  const { return Cast<T>(*the_p); }

        T* operator->() const { return  Cast<T*>(the_p); }

  // Is count one?
  bool unique() const { return ((isNull() == false) && the_p->unique()); }

  void increment() { if (isNull()) return; the_p->increment(); }
  void decrement() {
    if (isNull()) return;
    if (unique()) delete the_p;
    else the_p->decrement();
  }

  // Is the_p pointing to NULL?
  bool isNull() const { return the_p == 0; }

  friend
    bool operator==(const refcountable_ptr<T>& lhs, const refcountable_ptr<T>& rhs) {
    return lhs.the_p == rhs.the_p;
  }
  friend
    bool operator!=(const refcountable_ptr<T>& lhs, const refcountable_ptr<T>& rhs) {
    return lhs.the_p != rhs.the_p;
  }

 protected:
  Countable*			the_p;

 private:
  bool 				smartBehavior;
};

#endif
