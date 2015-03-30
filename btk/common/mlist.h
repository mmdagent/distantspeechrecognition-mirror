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


#ifndef _mlist_h_
#define _mlist_h_

#include <string>
#include <list>
#include <vector>
#include <map>
#include <iostream>

#include "common/common.h"
#include "common/jexception.h"

/*
#include "common/refcount.h"
*/

using namespace std;

#define Warning if(_setErrLine(__LINE__, __FILE__)) _warnMsg
extern int _setErrLine(int line, const char* file);
extern void _warnMsg(const char* message, ...);

// ----- definition of class `String' -----
// 
class String : public string {
 public:
  String () : string("") {}
  String (const char* s) : string(((s == NULL) ? "" : s)) {}
  String (const string& s) : string(s) {}

  operator const char*() const { return c_str(); }
  const char* chars() const { return c_str(); }
};


char* dateString(void);
void splitList(const String& line, std::list<String>& out);

typedef unsigned short UnShrt;
typedef float  LogFloat;
typedef double LogDouble;


// ----- definition of class template `List' -----
// 
template <class Type, class Key = String>
class List {
  typedef vector<Type>   	 		_ListVector;
  typedef typename _ListVector::iterator	_ListVectorIterator;
  typedef typename _ListVector::const_iterator	_ListVectorConstIterator;
  typedef map< Key, Type> 	 		_TypeMap;
  typedef typename _TypeMap::iterator	 	_TypeMapIterator;
  typedef typename _TypeMap::const_iterator	_TypeMapConstIterator;
  typedef map< Key, unsigned>	 		_IndexMap;
  typedef typename _IndexMap::iterator	 	_IndexMapIterator;
  typedef typename _IndexMap::const_iterator	_IndexMapConstIterator;

 public:
  List(const String& nm) : _name(nm) { }

  const String& name() const { return _name; }

  inline unsigned add(const Key& key, Type item);

  Type& operator[](unsigned index) {
    assert(index < _listVector.size());
    return _listVector[index];
  }
  const Type& operator[](unsigned index) const {
    assert(index < _listVector.size());
    return _listVector[index];
  }
  Type& operator[](const Key& key) {
    _TypeMapIterator itr = _typeMap.find(key);
    if (itr == _typeMap.end())
      throw j_error("Could not find key %s in map '%s'.", key.chars(), _name.c_str());
    return (*itr).second;
  }
  const Type& operator[](const Key& key) const {
    _TypeMapConstIterator itr = _typeMap.find(key);
    if (itr == _typeMap.end())
      throw j_error("Could not find key %s in map '%s'.", key.chars(), _name.c_str());
    return (*itr).second;
  }
  unsigned index(const Key& key) {
    _IndexMapIterator itr = _indexMap.find(key);
    if (itr == _indexMap.end())
      throw j_error("Could not find key %s in map '%s'.", key.chars(), _name.c_str());
    return (*itr).second;
  }
  unsigned index(const Key& key) const {
    _IndexMapConstIterator itr = _indexMap.find(key);
    if (itr == _indexMap.end())
      throw j_error("Could not find key %s in map '%s'.", key.chars(), _name.c_str());
    return (*itr).second;
  }
  bool isPresent(const Key& key) const {
    _IndexMapConstIterator itr = _indexMap.find(key);
    return itr != _indexMap.end();
  }

  unsigned size() const { return _listVector.size(); }
  void clear() { _listVector.clear();  _typeMap.clear();  _indexMap.clear(); }

  class Iterator;  	friend class Iterator;
  class ConstIterator;  friend class ConstIterator;

 private:
  unsigned _firstComp(int nParts, int part) const;
  unsigned _lastComp(int nParts, int part) const;

  const String					_name;

  _ListVector					_listVector;
  _TypeMap					_typeMap;
  _IndexMap					_indexMap;
};

template <class Type, class Key>
unsigned List<Type,Key>::add(const Key& key, Type item)
{
  unsigned idx = size();

  _indexMap[key] = idx;
  _typeMap[key]  = item;
  _listVector.push_back(item);

  return idx;
}

template <class Type, class Key>
unsigned List<Type,Key>::_firstComp(int nParts, int part) const
{
  unsigned segment = _listVector.size() / nParts;

  return (part - 1) * segment;
}

template <class Type, class Key>
unsigned List<Type,Key>::_lastComp(int nParts, int part) const
{
  unsigned ttlComps = _listVector.size();
  unsigned segment  = ttlComps / nParts;

  return (part == nParts) ? (ttlComps - 1) : (part * segment - 1);
}

template <class Type, class Key>
class List<Type,Key>::Iterator {
 public:
  Iterator(List& lst, int nParts = 1, int part = 1) :
    _beg(lst._listVector.begin() + lst._firstComp(nParts, part)),
    _cur(lst._listVector.begin() + lst._firstComp(nParts, part)),
    _end(lst._listVector.begin() + lst._lastComp(nParts, part) + 1) { }

  bool more() const { return _cur != _end; }
  void operator++(int) {
    if (more()) _cur++;
  }
  // Type& operator->() { return *_cur; }
        Type& operator*()       { return *_cur; }
  const Type& operator*() const { return *_cur; }

 private:
  _ListVectorIterator				_beg;
  _ListVectorIterator				_cur;
  _ListVectorIterator				_end;
};

template <class Type, class Key>
class List<Type,Key>::ConstIterator {
 public:
  ConstIterator(const List& lst, int nParts = 1, int part = 1) :
    _beg(lst._listVector.begin() + lst._firstComp(nParts, part)),
    _cur(lst._listVector.begin() + lst._firstComp(nParts, part)),
    _end(lst._listVector.begin() + lst._lastComp(nParts, part) + 1) { }

  bool more() const { return _cur != _end; }
  void operator++(int) {
    if (more()) _cur++;
  }
  const Type& operator*() const { return *_cur; }

 private:
  _ListVectorConstIterator			_beg;
  _ListVectorConstIterator			_cur;
  _ListVectorConstIterator			_end;
};

/* adds alls items of the file and passes them as string to T's addmethod->__add */
template<class T>
void freadAdd(const String& fileName, char commentChar, T* addmethod)
{
  FILE* fp = fileOpen(fileName,"r");

  if (fp == NULL)
    throw jio_error("Can't open file '%s' for reading.\n", fileName.c_str());

  cout << "Reading: " << fileName << endl;

  static char line[100000];
  while (1) {
    list<string> items;
    items.clear();
    char* p;
    int   f = fscanf(fp,"%[^\n]",&(line[0]));

    assert( f < 100000);

    if      ( f <  0)   break;
    else if ( f == 0) { fscanf(fp,"%*c"); continue; }

    if ( line[0] == commentChar) continue;

    for (p=&(line[0]); *p!='\0'; p++)
      if (*p>' ') break; if (*p=='\0') continue;

    try {
      // cout << "Adding: " << line << endl;
      addmethod->__add(line);
    } catch (j_error) {
      // cout << "Closing file ...";
      fileClose( fileName, fp);
      // cout << "Done" << endl;
      throw;
    }
  }
  fileClose( fileName, fp);
}

#endif
