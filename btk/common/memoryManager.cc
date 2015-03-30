//
//                                Millenium
//                   Automatic Speech Recognition System
//                                  (asr)
//
//  Module:  asr.common
//  Purpose: Common operations.
//  Author:  John McDonough

#include "common/memoryManager.h"
#include <algorithm>

// ----- methods for class `_MemoryAllocator' -----
//
_MemoryAllocator::_MemoryAllocator(unsigned elemSize, unsigned blkSize, unsigned limit)
  : _size(std::max<unsigned>(elemSize, sizeof(void*))), _blkSize(blkSize), _list(NULL),
    _cnt(0), _limit(limit)
{
  cout << "Creating allocator for objects of size " << elemSize << " ... " << endl;

  _newBlock();
}

_MemoryAllocator::~_MemoryAllocator()
{
  /* this does not belong here! (merkosh) */
  //  printf("Freeing all memory for type '%s'.\n", _type.c_str());
  for (_AllocListIterator itr = _allocList.begin(); itr != _allocList.end(); itr++)
    free(*itr);
}

void _MemoryAllocator::_newBlock()
{
  /*
  if (_limit > 0) {
    printf("Allocating block %d with size %d : Total Allocated %d : Limit %d\n",
	   _allocList.size(), _size * _blkSize, _allocList.size() * _size * _blkSize, _limit);
    fflush(stdout);
  }
  */

  if (_limit > 0 && (_allocList.size() * _size * _blkSize) > _limit)
    throw jallocation_error("Tried to allocate more than %d bytes", _limit);

#ifndef MEMORYDEBUG
  char* list = (char*) malloc(_size * _blkSize);
  _allocList.push_back(list);

  for (unsigned iblk = 0; iblk < _blkSize; iblk++) {
    _Element* elem = (_Element*) (list + (iblk * _size));
    elem->_next    = _list;
    _list          = elem;
  }
#endif
}

void _MemoryAllocator::report(FILE* fp) const
{
  fprintf(fp, "\nMemory Manager\n");
  fprintf(fp, "C++ Type:  %s\n", typeid(this).name());
  fprintf(fp, "Type Size: %d\n", _size);
  fprintf(fp, "Space allocated for %d objects\n",
	  _allocList.size() * _blkSize);
  fprintf(fp, "Total allocated space is %d\n",
	  _allocList.size() * _blkSize * _size);
  fprintf(fp, "There are %d allocated objects\n", _cnt);
  fprintf(fp, "\n");

  fflush(fp);
}

void* _MemoryAllocator::newElem()
{
  _cnt++;

  /*
  printf("Allocating element %d of type %s\n",
	 _cnt, _type.c_str());
  */

#ifdef MEMORYDEBUG
  return (Type*) malloc(_size);
#else
  if (_list == NULL)
    _newBlock();

  _Element* e = _list;
  _list = e->_next;
  return e;
#endif
}

void _MemoryAllocator::deleteElem(void* e)
{
  /*
  printf("Deleting element %d of type %s\n",
	 _cnt, _type.c_str());
  */

  if (e == NULL) return;

  _cnt--;

#ifdef MEMORYDEBUG
  free(e);
#else
  _Element* elem = (_Element*) e;
  elem->_next = _list;
  _list = elem;
#endif
}


// ----- methods for class `__MemoryManager' -----
//
__MemoryManager::_AllocatorList __MemoryManager::_allocatorList;

__MemoryManager::__MemoryManager(unsigned elemSize, unsigned blkSize, unsigned limit)
  : _allocator(_initialize(elemSize, blkSize, limit)) { }

__MemoryManager::~__MemoryManager() { }

_MemoryAllocator* __MemoryManager::_initialize(unsigned elemSize, unsigned blkSize, unsigned limit)
{
  _AllocatorListIterator itr = _allocatorList.find(elemSize);
  if (itr == _allocatorList.end()) {
    _allocatorList.insert(_AllocatorListValueType(elemSize, new _MemoryAllocator(elemSize, blkSize, limit)));
    itr = _allocatorList.find(elemSize);
  }

  return (*itr).second;
}
