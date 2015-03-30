//
//                                Millenium
//                   Automatic Speech Recognition System
//                                  (asr)
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


#include "path/distribPath.h"
#include "common/mlist.h"

#if 0

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

#endif

String DistribPath::names() const
{
  String nm("");
  for (_Iterator itr = _distNames->begin(); itr != _distNames->end(); itr++)
      nm += (*itr) + " ";

  return nm;
}

#if 0

void DistribPath::read(const list<String>& distNames)
{
  delete _distNames;
  _distNames = new _List(distNames.size());
  unsigned i = 0;
  for (list<String>::const_iterator itr = distNames.begin(); itr != distNames.end(); itr++)
    (*_distNames)[i++] = (*itr);
}

#endif

void DistribPath::write(const String& fileName) const
{
  FILE* fp = fileOpen(fileName, "w");

  write_int(fp, len());
  for (_Iterator itr = _distNames->begin(); itr != _distNames->end(); itr++)
    write_string(fp, (*itr).chars());

  fileClose( fileName, fp);
}

DistribPtr& DistribPath::get(unsigned i)
{
  if (i >= len())
    throw jio_error("Index too large.");

  const String& nm((*_distNames)[i]);
  return _dss->find(nm);
}

DistribPath::Iterator::Iterator(DistribPathPtr& path)
  : _dss(path->_dss), _distNames(*(path->_distNames)), _iterator(_distNames.begin()) { }
