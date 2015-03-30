//
//                                Millenium
//                   Automatic Speech Recognition System
//                                  (asr)
//
//  Module:  asr.common
//  Purpose: Common operations.
//  Author:  John McDonough

#ifndef _memoryManager_h_
#define _memoryManager_h_

#include <list>
#include <typeinfo>

#include "common/mlist.h"


// ----- definition of class `_MemoryAllocator' -----
// 
class _MemoryAllocator {
  union _Element {
    _Element*					_next;
  };

  typedef list<char*>				_AllocList;
  typedef _AllocList::iterator			_AllocListIterator;

 public:
  _MemoryAllocator(unsigned elemSize, unsigned blkSize = 1000, unsigned limit = 0);
  ~_MemoryAllocator();

  void*	newElem();
  void	deleteElem(void* e);

  void report(FILE* fp = stdout) const;

  const unsigned cnt()       const { return _cnt;     };
  const unsigned blockSize() const { return _blkSize; };
  const size_t   size()      const { return _size;    };
  void  setLimit(unsigned limit) { _limit = limit; }

 private:
  void _newBlock();

  const size_t					_size;
  const unsigned				_blkSize;

  _AllocList					_allocList;
  _Element*					_list;

  unsigned					_cnt;
  unsigned					_limit;
};


// ----- definition of class `__MemoryManager' -----
// 
class __MemoryManager {

  typedef map<unsigned, _MemoryAllocator*>	_AllocatorList;
  typedef _AllocatorList::iterator		_AllocatorListIterator;
  typedef _AllocatorList::value_type		_AllocatorListValueType;

 public:
  __MemoryManager(unsigned elemSize, unsigned blkSize = 1000, unsigned limit = 0);
  ~__MemoryManager();

  inline void*	_newElem() { return _allocator->newElem(); }
  inline void	_deleteElem(void* e) { _allocator->deleteElem(e); }

  void report(FILE* fp = stdout) { _allocator->report(fp); }

  const unsigned cnt()       const { return _allocator->cnt();       };
  const unsigned blockSize() const { return _allocator->blockSize(); };
  const size_t   size()      const { return _allocator->size();      };
  void setLimit(unsigned limit)    { _allocator->setLimit(limit); }

 private:
  _MemoryAllocator* _initialize(unsigned elemSize, unsigned blkSize, unsigned limit);

  static _AllocatorList				_allocatorList;
  _MemoryAllocator*				_allocator;
};


// ----- definition of class template `MemoryManager' -----
// 
template <class Type>
class MemoryManager : public __MemoryManager {
 public:
  MemoryManager(const String& type, unsigned blkSize = 1000, unsigned limit = 0);
  ~MemoryManager();

  Type*	newElem() { return (Type*) _newElem(); }
  void	deleteElem(void* e) { _deleteElem(e); }

  const String&	type() const { return _type; }

 private:
  const String					_type;
};


// ----- methods for class template `MemoryManager' -----
//
template <class Type>
MemoryManager<Type>::
MemoryManager(const String& type, unsigned blkSize, unsigned limit)
  : __MemoryManager(sizeof(Type), blkSize, limit), _type(type)
{
  cout << "Creating 'MemoryManager' for type '" << type << "'." << endl;
}

template <class Type>
MemoryManager<Type>::~MemoryManager() { }


#endif
