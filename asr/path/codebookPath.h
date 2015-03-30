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


#ifndef _codebookPath_h_
#define _codebookPath_h_

#include "common/refcount.h"
#include "gaussian/codebookBasic.h"

class CodebookPath {
  typedef vector<String>			_List;
  typedef _List::const_iterator			_Iterator;

 public:
  CodebookPath(CodebookSetBasicPtr cbs, const String fileName = "");
  ~CodebookPath();

  class Iterator;  friend class Iterator;

  unsigned len() const { return _cbkNames->size(); }
  void read(const String& fileName);
  void write(const String& fileName) const;
  CodebookBasicPtr get(unsigned i);

 private:
  CodebookSetBasicPtr				_cbs;
  _List*					_cbkNames;
};

typedef refcount_ptr<CodebookPath> CodebookPathPtr;

class CodebookPath::Iterator {
 public:
  Iterator(CodebookPathPtr& path);

        bool              more()  const { return _iterator != _cbkNames.end(); }
  const String&           name()  const { return (*_iterator); }
  const CodebookBasicPtr& cbk()   const { return _cbs->find(name()); }
  unsigned                index() const { return _cbs->index(name()); }
  void operator++(int) {
    if (more()) _iterator++;
  }
  inline CodebookBasicPtr next();

 private:
  CodebookSetBasicPtr				_cbs;
  const _List&					_cbkNames;
  _Iterator					_iterator;
};

// needed for Python iterator
//
CodebookBasicPtr CodebookPath::Iterator::next() {
  if (!more())
    throw jiterator_error("end of codebook!");

  CodebookBasicPtr cb(cbk());
  operator++(1);
  return cb;
}

#endif
