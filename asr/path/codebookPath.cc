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


#include "codebookPath.h"
#include "common/mlist.h"
#include "common/mach_ind_io.h"

CodebookPath::CodebookPath(CodebookSetBasicPtr cbs, const String fileName)
  : _cbs(cbs), _cbkNames(NULL)
{
  if (fileName == "") return;

  read(fileName);
}

CodebookPath::~CodebookPath()
{
  delete _cbkNames;
}

void CodebookPath::read(const String& fileName)
{
  static char cbkName[256];

  FILE* fp = fileOpen(fileName, "r");
  int listN = read_int(fp);
  delete _cbkNames;
  _cbkNames = new _List(listN);
  for (int i = 0; i < listN; i++) {
    read_string(fp, cbkName);
    (*_cbkNames)[i] = String(cbkName);
  }
  fileClose( fileName, fp);
}

void CodebookPath::write(const String& fileName) const
{
  FILE* fp = fileOpen(fileName, "w");

  write_int(fp, len());
  for (_Iterator itr = _cbkNames->begin(); itr != _cbkNames->end(); itr++)
    write_string(fp, (*itr).chars());

  fileClose( fileName, fp);
}

CodebookBasicPtr CodebookPath::get(unsigned i)
{
  if (i >= len())
    throw jio_error("Index too large.");

  const String& nm((*_cbkNames)[i]);
  return _cbs->find(nm);
}

CodebookPath::Iterator::Iterator(CodebookPathPtr& path)
  : _cbs(path->_cbs), _cbkNames(*(path->_cbkNames)), _iterator(_cbkNames.begin())
{
  
}
