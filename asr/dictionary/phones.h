//
//                               Millenium
//                   Automatic Speech Recognition System
//                                 (asr)
//
//  Module:  asr.dict
//  Purpose: Dictionary module.
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


#ifndef _phones_h_
#define _phones_h_

#include <map>
#include <list>
#include <string>
#include <stdio.h>
#include "common/refcount.h"
#include "common/jexception.h"
#include "common/mlist.h"

// ----- definition for class `Phones' -----
//
class Phones : public Countable {
  typedef List<String>				_Phones;
  typedef _Phones::Iterator			_Iterator;
  typedef _Phones::ConstIterator		_ConstIterator;

 public:
  // A 'Phones' object is an array of strings, each of which is a phoneme.
  Phones() : _commentChar(';'), _phones("Phones") { }
  Phones(const String& n) : _commentChar(';'), _phones("Phones"), _name(n) { }
  ~Phones() { }

  class Iterator;  friend class Iterator;

  String operator[](const String& key) { return _phones[key]; }
  String operator[](unsigned index)    { return _phones[index]; }

  // prints the contents of a set of phone-sets
  String puts();

  // add new phone(s) to a phone-set
  void add(const String& phone);

  // returns true if phone is in this class
  bool hasPhone(const String& phone);

  // read a phone-set from a file
  int read(const String& fileName);

  // write a phone-set from a file
  void write(const String& filename);

  // write a set of phone-sets into a open file
  void writeFile(FILE* fp);

  // return the name of the phones
  const String& name() const { return _name; };

  // index of a phone set
  unsigned index(const String& phone) const { return _phones.index(phone); }

  // return the number of phones
  unsigned phonesN() const { return _phones.size(); }

 protected:
  char						_commentChar;
  _Phones					_phones;
  String					_name;
};

typedef refcountable_ptr<Phones> PhonesPtr;


// ----- definition for class `Phones::Iterator' -----
//
class Phones::Iterator {
 public:
  Iterator(PhonesPtr& phones)
    : _phones(phones), _itr(Phones::_Iterator(_phones->_phones)) { }

  void operator++(int) { _itr++; }
  bool more() const { return _itr.more(); }
  const String& phone() const { return *_itr; }
  const String& operator*() const { return *_itr; }
  const String& next() {
    if (more()) {
      const String& str(phone());
      operator++(1);
      return str;
    } else {
      throw jiterator_error("end of phones!");
    }
  }

 private:
  PhonesPtr					_phones;
  Phones::_Iterator				_itr;
};

int phonesCharFunc(PhonesPtr superset, PhonesPtr subset, char* charFunc);


// ----- definition for class `PhonesSet' -----
//
class PhonesSet {
  typedef List<PhonesPtr>		_PhonesSet;
  typedef _PhonesSet::Iterator		_Iterator;
  typedef _PhonesSet::ConstIterator	_ConstIterator;

 public:
// A 'PhonesSet' object is a set of 'Phones' objects.
  PhonesSet(const String& nm = "") :
    _commentChar(';'), _phonesSet((nm == "") ? String("(null)") : nm),
    _name((nm == "") ? String("(null)") : nm) { }
  ~PhonesSet() { }

  class Iterator;	friend class Iterator;

  PhonesPtr operator[](int i)             const { return _phonesSet[i]; }
  PhonesPtr operator[](const String& key) const { return _phonesSet[key]; }
  PhonesPtr find(const String& key)       const { return _phonesSet[key]; }

  // displays the contents of a set of phone-sets
  String puts();

  // add new phone-set to a set of phones-set
  void add(const String& phoneName, const PhonesPtr& phones);
  void add(const String& phoneName, const list<String>& phones);

  // delete phone-set(s) from a set of phone-sets
  void remove(const String& phone);

  // read a set of phone-sets from a file
  int read(const String& filename);

  // write a set of phone-sets into a file
  void write(const String& filename);

  // name of phone set
  const String& name() { return _name; };

  // index of a phone set
  unsigned index(const String& phone) const { return _phonesSet.index(phone); }

  // returns true if phone is in this class
  bool hasPhones(const String& phones);

 protected:
  char						_commentChar;
  _PhonesSet					_phonesSet;
  String					_name;
};

typedef refcount_ptr<PhonesSet> PhonesSetPtr;


// ----- definition for class `PhonesSet::Iterator' -----
//
class PhonesSet::Iterator {
 public:
  Iterator(PhonesSetPtr& p):
    _phonesSet(p), _iter(_phonesSet->_phonesSet) {}

  PhonesPtr operator->() { return *_iter; }
  PhonesPtr operator*()  { return *_iter; }
  void operator++(int)   { _iter++; }
  bool more() { return _iter.more(); }

 private:
  PhonesSetPtr					_phonesSet;
  _Iterator					_iter;
};

#endif
