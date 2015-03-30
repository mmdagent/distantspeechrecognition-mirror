//                              -*- C++ -*-
//
//                               Millenium
//                  Automatic Speech Recognition System
//                                 (asr)
//
//  Module:  asr.path
//  Purpose: Viterbi path storage and feature space adaptation.
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


#ifndef _distribPath_h_
#define _distribPath_h_

#include "common/refcount.h"
#include "gaussian/distribBasic.h"
#include "common/mach_ind_io.h"

template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr> class _Decoder;

class DistribPath;

typedef refcount_ptr<DistribPath> DistribPathPtr;

class DistribPath {
  template <class WFSType, class NodePtr, class EdgePtr, class WFSTypePtr> friend class _Decoder;
  typedef vector<String>			_List;
  typedef _List::const_iterator			_Iterator;
 public:
  inline DistribPath(DistribSetPtr& dss, const String& fileName = "");
  inline DistribPath(DistribSetPtr& dss, const list<String>& distNames);
  inline DistribPath(DistribSetPtr& dss, DistribPathPtr& path);
  inline ~DistribPath();

  class Iterator;  friend class Iterator;

  // length of the path
  unsigned len() const { return _distNames->size(); }

  // read a list of distributions from a file
  inline void read(const String& fileName);

  // read a list of distributions
  inline void read(const list<String>& distNames);

  // write a list of distributions
  void write(const String& fileName) const;

  // find a distribution based on a state index
  DistribPtr& get(unsigned i);

  // string of all names on path
  String names() const;

 private:
  DistribSetPtr					_dss;
  _List*					_distNames;
};

class DistribPath::Iterator {
 public:
  Iterator(DistribPathPtr& path);

        bool              more()  const { return _iterator != _distNames.end(); }
  const String&           name()  const { return (*_iterator); }
  const DistribPtr&       ds()    const { return _dss->find(name()); }
  unsigned                index() const { String nm(name()); /* if (nm == "SIL(|)-m") nm = "SIL-m"; */ return _dss->index(nm); }
  void operator++(int) {
    if (more()) _iterator++;
  }
  inline DistribPtr next();

 private:
  DistribSetPtr					_dss;
  const _List&					_distNames;
  _Iterator					_iterator;
};

// needed for Python iterator
//
DistribPtr DistribPath::Iterator::next() {
  if (!more())
    throw jiterator_error("end of iterators!");

  DistribPtr dss(ds());
  operator++(1);
  return dss;
}


DistribPath::DistribPath(DistribSetPtr& dss, const String& fileName)
  : _dss(dss), _distNames(NULL)
{
  if (fileName == "") return;

  read(fileName);
}

DistribPath::DistribPath(DistribSetPtr& dss, const list<String>& distNames)
  : _dss(dss), _distNames(new _List(distNames.size()))
{
  read(distNames);
}

DistribPath::DistribPath(DistribSetPtr& dss, DistribPathPtr& path)
  : _dss(dss), _distNames(new _List)
{
  for (Iterator itr(path); itr.more(); itr++) {
    String phone(itr.name());

    String::size_type pos     = 0;
    String::size_type prevPos = 0;
    pos = phone.find_first_of('(', pos);
    String subString(phone.substr(prevPos, pos - prevPos));
    // cout << "Adding: " << subString << endl;
    (*_distNames).push_back(subString);
  }
}

DistribPath::~DistribPath()
{
  delete _distNames;
}

void DistribPath::read(const String& fileName)
{
  static char dsName[256];

  FILE* fp = fileOpen(fileName, "r");
  int listN = read_int(fp);
  delete _distNames;
  _distNames = new _List(listN);
  for (int i = 0; i < listN; i++) {
    read_string(fp, dsName);
    (*_distNames)[i] = String(dsName);
  }
  fileClose( fileName, fp);
}

void DistribPath::read(const list<String>& distNames)
{
  delete _distNames;
  _distNames = new _List(distNames.size());
  unsigned i = 0;
  for (list<String>::const_iterator itr = distNames.begin(); itr != distNames.end(); itr++)
    (*_distNames)[i++] = (*itr);
}

#endif
