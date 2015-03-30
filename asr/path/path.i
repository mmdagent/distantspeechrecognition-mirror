//                              -*- C++ -*-
//
//                               Millenium
//                  Automatic Speech Recognition System
//                                 (asr)
//
//  Module:  asr.path
//  Purpose: Viterbi path storage and feature space adaptation.
//  Author:  John McDonough.
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


%module path

#ifdef AUTODOC
%section "Path"
#endif

/* operator overloading */
%rename(__str__) *::puts();

%include typedefs.i
%include jexception.i

typedef unsigned short UnShrt;

%pythoncode %{
from btk import stream
from asr import gaussian
oldimport = """
%}
%import gaussian/gaussian.i
%pythoncode %{
"""
%}

%{
#include "stream/pyStream.h"
#include "feature/feature.h"
#include <numpy/arrayobject.h>
#include "codebookPath.h"
%}

#ifdef AUTODOC
%subsection "Path", before
#endif

%rename(CodebookPath_Iterator) CodebookPath::Iterator;
%rename(CodebookPath_Iterator) CodebookPath::Iterator(CodebookPath& path);


// ----- definition for class `CodebookPath' -----
// 
%ignore CodebookPath;
class CodebookPath {
 public:
  CodebookPath(CodebookSetBasicPtr cbs, const String fileName = "");
  ~CodebookPath();

  // length of the path
  unsigned len();

  // read a list of codebooks
  void read(const String fileName);

  // write a list of codebooks
  void write(const String fileName);

  // find a codebook based on a state index
  CodebookBasicPtr get(unsigned i);
};

class CodebookPath::Iterator {
 public:
  CodebookPath::Iterator(CodebookPathPtr path);

  bool more() const;
  const String& name() const;
  const CodebookBasicPtr cbk() const;
  CodebookBasicPtr next();
};

class CodebookPathPtr {
 public:
  %extend {
    CodebookPathPtr(CodebookSetBasicPtr cbs, const String fileName = "") {
      return new CodebookPathPtr(new CodebookPath(cbs, fileName));
    }

    CodebookPath::Iterator* __iter__() {
      return new CodebookPath::Iterator(*self);
    }
  }
  CodebookPath* operator->();
};

%{
#include "distribPath.h"
%}

%rename(DistribPath_Iterator) DistribPath::Iterator;
%rename(DistribPath_Iterator) DistribPath::Iterator(DistribPath& path);


// ----- definition for class `DistribPath' -----
// 
%ignore DistribPath;
class DistribPath {
 public:
  DistribPath(DistribSetPtr& dss, const String& fileName = "");
  DistribPath(DistribSetPtr& dss, const list<String>& distNames);
  DistribPath(DistribSetPtr& dss, DistribPathPtr& path);
  ~DistribPath();

  // length of the path
  unsigned len();

  // read a list of distributions
  void read(const String fileName);

  // write a list of distributions
  void write(const String fileName);

  // find a distribution based on a state index
  DistribPtr get(unsigned i);

  // string of all names on path
  String names() const;
};

class DistribPath::Iterator {
 public:
  DistribPath::Iterator(DistribPathPtr& path);

  bool more() const;
  const String& name() const;
  const DistribPtr ds() const;
        unsigned        index() const;
  DistribPtr next();
};

class DistribPathPtr {
 public:
  %extend {
    DistribPathPtr(DistribSetPtr& dss, const String& fileName = "") {
      return new DistribPathPtr(new DistribPath(dss, fileName));
    }

    DistribPathPtr(DistribSetPtr& dss, DistribPathPtr& path) {
      return new DistribPathPtr(new DistribPath(dss, path));
    }

    DistribPath::Iterator* __iter__() {
      return new DistribPath::Iterator(*self);
    }
  }
  DistribPath* operator->();
};
